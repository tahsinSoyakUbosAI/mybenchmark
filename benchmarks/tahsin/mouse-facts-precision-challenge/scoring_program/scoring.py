#!/usr/bin/env python3
"""Decide whether the model's answers were right, and write the scores.

BenchGen runs this after the model has answered, with:

    /app/input/res/   the ingestion program's predictions.json
    /app/input/ref/   this bundle's phase/reference_data
    /app/output/      where scores.json goes
    /app/program/     this folder

KIND picks how an answer is judged. `scaffold --kind` sets it, and writes the
leaderboard columns in competition.yaml to match METRICS[KIND]. Every key this
program writes into scores.json must be a leaderboard column key, or that
column stays empty. Change the two together or neither.

    tool_call  the answer is a tool call; name and arguments scored apart
    exact      one short answer, matched ignoring case and spacing
    number     a number, right when within the item's tolerance
    keywords   an open answer, scored by how many required points it mentions

To measure something else, add a scorer below, name its metrics in METRICS,
and set KIND to it.
"""
# --- LiteLLM attribution shim (auto-injected by the platform at bundle upload) ---
# Adds the `x-litellm-tags` header to outgoing HTTP requests so gateway spend
# is attributed to this run (benchmark:/run:/owner:, rl_loop:/round: tags).
# No-op when LITELLM_TAGS is absent from the environment.
def _install_litellm_attribution_shim():
    import os as _os
    _tags = (_os.environ.get("LITELLM_TAGS") or "").strip()
    if not _tags:
        return
    try:
        import urllib.request as _ur
        if not getattr(_ur, "_litellm_attribution_shim", False):
            _ur._litellm_attribution_shim = True
            _orig_urlopen = _ur.urlopen
            def _tagged_urlopen(url, *args, **kwargs):
                try:
                    if isinstance(url, str):
                        url = _ur.Request(url)
                    if isinstance(url, _ur.Request):
                        url.add_header("x-litellm-tags", _tags)
                except Exception:
                    pass
                return _orig_urlopen(url, *args, **kwargs)
            _ur.urlopen = _tagged_urlopen
    except Exception:
        pass
    try:
        import requests as _rq
        if not getattr(_rq.Session, "_litellm_attribution_shim", False):
            _rq.Session._litellm_attribution_shim = True
            _orig_send = _rq.Session.send
            def _tagged_send(self, request, **kwargs):
                try:
                    request.headers.setdefault("x-litellm-tags", _tags)
                except Exception:
                    pass
                return _orig_send(self, request, **kwargs)
            _rq.Session.send = _tagged_send
    except Exception:
        pass
    try:
        import httpx as _hx
        for _client_cls in (_hx.Client, _hx.AsyncClient):
            if getattr(_client_cls, "_litellm_attribution_shim", False):
                continue
            _client_cls._litellm_attribution_shim = True
            def _make_send(_orig):
                def _tagged(self, request, *args, **kwargs):
                    try:
                        request.headers.setdefault("x-litellm-tags", _tags)
                    except Exception:
                        pass
                    return _orig(self, request, *args, **kwargs)
                return _tagged
            _client_cls.send = _make_send(_client_cls.send)
    except Exception:
        pass
_install_litellm_attribution_shim()
del _install_litellm_attribution_shim
# --- end LiteLLM attribution shim ---
import ast
import json
import os
import re
import sys
import unicodedata

from item_results import item_row, write_item_results, write_results_page

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

KIND = 'precision'

#: The per-item metrics each kind produces. overall_accuracy, the column the
#: leaderboard sorts on, is always their average and always written.
METRICS = {
    'tool_call': ('tool_name_accuracy', 'argument_accuracy', 'exact_match'),
    'exact': ('exact_match',),
    'number': ('numeric_accuracy',),
    'keywords': ('keyword_coverage', 'all_keywords'),
    'precision': ('correctness', 'precision_bonus', 'overreach_clean'),
}
OVERALL = 'overall_accuracy'


def normalise(value):
    """Lower case, single spaces, accents folded, so trivial differences never count."""
    text = unicodedata.normalize('NFKD', str(value if value is not None else ''))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return ' '.join(text.casefold().split())


# --- tool_call ---------------------------------------------------------------

def parse_call(text):
    """A tool call out of the model's answer, as {name, arguments}, or None.

    Accepts bare JSON, JSON in a ``` fence, and name(arg=value) shorthand.
    """
    blob = str(text or '').strip()
    fence = re.search(r'```(?:json)?\s*(.+?)```', blob, re.S)
    if fence:
        blob = fence.group(1).strip()
    try:
        loaded = json.loads(blob)
        if isinstance(loaded, dict) and loaded.get('name'):
            return {'name': str(loaded['name']),
                    'arguments': loaded.get('arguments') or loaded.get('args') or {}}
    except (ValueError, TypeError):
        pass
    shorthand = re.match(r'^\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$', blob, re.S)
    if shorthand:
        args = {}
        for key, raw in re.findall(r'(\w+)\s*=\s*([^,]+)', shorthand.group(2)):
            try:
                args[key] = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                args[key] = raw.strip().strip('"\'')
        return {'name': shorthand.group(1), 'arguments': args}
    return None


def score_tool_call(prediction, expected, item):
    got = parse_call(prediction)
    if got is None or not isinstance(expected, dict):
        return {'tool_name_accuracy': 0.0, 'argument_accuracy': 0.0, 'exact_match': 0.0}
    right_name = float(normalise(got['name']) == normalise(expected.get('name')))
    wanted = expected.get('arguments') or {}
    given = got.get('arguments') or {}
    if wanted:
        right_args = sum(1 for key, value in wanted.items()
                         if normalise(given.get(key)) == normalise(value)) / len(wanted)
    else:
        right_args = float(not given)
    return {'tool_name_accuracy': right_name,
            'argument_accuracy': right_args,
            'exact_match': float(right_name == 1.0 and right_args == 1.0)}


# --- exact -------------------------------------------------------------------

def score_exact(prediction, expected, item):
    """Right when the answer equals the expected one, or any listed alternative."""
    accepted = list(expected) if isinstance(expected, list) else [expected]
    accepted += item.get('also_accept') or []
    answer = normalise(prediction).rstrip('.')
    return {'exact_match': float(any(answer == normalise(a).rstrip('.') for a in accepted))}


# --- number ------------------------------------------------------------------

NUMBER = re.compile(r'-?\d+(?:[.,]\d+)?')


def first_number(text):
    """The first number in the answer. A comma counts as a decimal point."""
    match = NUMBER.search(str(text or '').replace('−', '-'))
    return float(match.group(0).replace(',', '.')) if match else None


def score_number(prediction, expected, item):
    got = first_number(prediction)
    want = first_number(expected)
    if got is None or want is None:
        return {'numeric_accuracy': 0.0}
    tolerance = float(item.get('tolerance') or 0)
    return {'numeric_accuracy': float(abs(got - want) <= tolerance + 1e-9)}


# --- keywords ----------------------------------------------------------------

def mentions(answer, alternative):
    """Whether a normalised answer covers one alternative of a key point.

    A single word matches as a substring, so a stem like "irritat" covers
    "irritated". Several words match when every one of them appears, in any
    order: "age estimate" covers "estimate its age". Matching the phrase as
    written marked a correct answer wrong just for its word order.
    """
    words = normalise(alternative).split()
    return bool(words) and all(word in answer for word in words)


def score_keywords(prediction, expected, item):
    """Coverage of the required points; each point may list synonyms with '|'."""
    points = expected.get('keywords') if isinstance(expected, dict) else expected
    points = [p for p in (points or []) if str(p).strip()]
    if not points:
        return {'keyword_coverage': 0.0, 'all_keywords': 0.0}
    answer = normalise(prediction)
    missed = [str(point) for point in points
              if not any(mentions(answer, alt) for alt in str(point).split('|'))]
    found = len(points) - len(missed)
    coverage = found / len(points)
    # 'missed' is not a metric: it goes into the per-question results, so a
    # miss can be explained from what the scoring saw instead of guessed.
    return {'keyword_coverage': coverage, 'all_keywords': float(found == len(points)), 'missed': missed}


# --- precision ---------------------------------------------------------------

SPECIFIC = re.compile(r'\d')


def score_mine(prediction, expected, item):
    """Reward a correct, specific answer; reward extra correct detail; punish
    confident wrong specifics.

    The reference item carries the core fact in `answer` (a plain string,
    `"a|b"` accepts either alternative), plus item-level `bonus` (a list of
    extra correct details) and `traps` (plausible but wrong specifics), the
    same way the `number` kind keeps `tolerance` on the item rather than
    inside the reference value. Bonus points are only credited once the core
    is right, so a wrong core answer cannot farm bonus credit by padding with
    true but off-topic detail. Overreach is clean when the core is right, OR
    the answer gave no specific number/term at all (safely vague). It is
    dirty either when a listed trap phrase appears, or when the core is wrong
    yet the answer still states some other specific number as if certain:
    that is confidently wrong, not merely incomplete.
    """
    answer = normalise(prediction)
    core = expected
    bonus = item.get('bonus') or []
    traps = item.get('traps') or []

    correctness = float(bool(core) and any(mentions(answer, alt) for alt in str(core).split('|')))
    if bonus:
        hit = sum(1 for point in bonus if any(mentions(answer, alt) for alt in str(point).split('|')))
        bonus_coverage = hit / len(bonus)
    else:
        bonus_coverage = 0.0
    precision_bonus = bonus_coverage if correctness else 0.0
    hit_trap = any(mentions(answer, alt) for trap in traps for alt in str(trap).split('|'))
    stated_other_specific = (not correctness) and bool(SPECIFIC.search(answer))
    overreach_clean = float(not hit_trap and not stated_other_specific)
    return {'correctness': correctness, 'precision_bonus': precision_bonus, 'overreach_clean': overreach_clean}


SCORERS = {
    'tool_call': score_tool_call,
    'exact': score_exact,
    'number': score_number,
    'keywords': score_keywords,
    'precision': score_mine,
}


def read_json_dir(path, key='data'):
    items = []
    if not os.path.isdir(path):
        return items
    for name in sorted(os.listdir(path)):
        if not name.endswith('.json'):
            continue
        with open(os.path.join(path, name), encoding='utf-8') as handle:
            payload = json.load(handle)
        items.extend(payload if isinstance(payload, list) else payload.get(key, []))
    return items


def main():
    if KIND not in SCORERS:
        raise SystemExit(f'KIND is {KIND!r}; expected one of {", ".join(SCORERS)}')
    score_one = SCORERS[KIND]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions = read_json_dir(os.path.join(INPUT_DIR, 'res'))
    reference = read_json_dir(os.path.join(INPUT_DIR, 'ref'))
    answers = {str(p.get('id', i)): p for i, p in enumerate(predictions)}
    print(f'[scoring] {KIND}: {len(predictions)} predictions, {len(reference)} reference items', flush=True)

    totals = {metric: 0.0 for metric in METRICS[KIND]}
    scored, rows = 0, []
    # Walk the reference, not the answers: a question the model never answered
    # still counts against it, and still gets a per-question row.
    for index, item in enumerate(reference):
        prediction = answers.get(str(item.get('id', index)))
        answer = prediction.get('prediction') if prediction else ''
        result = score_one(answer, item.get('answer'), item) if prediction else {}
        values = {metric: float(result.get(metric, 0.0)) for metric in METRICS[KIND]}
        for metric, value in values.items():
            totals[metric] += value
        scored += int(prediction is not None)
        rows.append(item_row(item.get('id', index),
                             (prediction or {}).get('group') or item.get('group'),
                             (prediction or {}).get('prompt'), item.get('answer'), answer,
                             bool(values) and min(values.values()) >= 1.0, values,
                             missed=result.get('missed')))
    write_item_results(rows, OUTPUT_DIR)

    # An item the model never answered still counts against it: divide by the
    # reference set, not by what came back.
    denominator = max(len(reference), 1)
    # Leaderboard columns are percentages on the 0-100 scale, like the platform's
    # own scorers: the web app reads a value below 1 as a ratio and a value of
    # 1 or more as a percentage, so a perfect run written as 1.0 shows as 1%.
    scores = {metric: round(100.0 * total / denominator, 2) for metric, total in totals.items()}
    scores[OVERALL] = round(sum(scores.values()) / len(scores), 2) if scores else 0.0

    with open(os.path.join(OUTPUT_DIR, 'scores.json'), 'w', encoding='utf-8') as handle:
        json.dump(scores, handle, ensure_ascii=False, indent=2)
    write_results_page(rows, OUTPUT_DIR, scores)
    print(f'[scoring] scored {scored} of {len(reference)}: {scores}', flush=True)


if __name__ == '__main__':
    main()
