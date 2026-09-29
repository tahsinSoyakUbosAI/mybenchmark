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
import zipfile

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/app/output'

KIND = 'keywords'

#: The per-item metrics each kind produces. overall_accuracy, the column the
#: leaderboard sorts on, is always their average and always written.
METRICS = {
    'tool_call': ('tool_name_accuracy', 'argument_accuracy', 'exact_match'),
    'exact': ('exact_match',),
    'number': ('numeric_accuracy',),
    'keywords': ('keyword_coverage', 'all_keywords', 'conciseness_score'),
}
OVERALL = 'overall_accuracy'

#: Metrics folded into overall_accuracy. Conciseness is measured and shown on
#: its own column, but never folded into the overall score: it would punish a
#: missed key point twice (once in coverage, again in conciseness), and the
#: two columns should be free to move independently.
OVERALL_METRICS = dict(METRICS)
OVERALL_METRICS['keywords'] = ('keyword_coverage', 'all_keywords')

#: A two-to-three-sentence answer this benchmark expects lands under this.
MAX_REASONABLE_TOKENS = 80


def count_tokens(text):
    """Whitespace/punctuation word count, used as a simple token count."""
    return len(re.findall(r"\w+", str(text or ''), re.UNICODE))


def score_conciseness(coverage, token_count):
    """Coverage of the required points, discounted for a rambling answer.

    An answer that hits no key points is never "concise": there is nothing in
    it worth being concise about, so this is 0 whenever coverage is 0. Above
    that, it only falls once the answer runs past MAX_REASONABLE_TOKENS.
    """
    if token_count <= MAX_REASONABLE_TOKENS:
        length_factor = 1.0
    else:
        length_factor = max(0.0, 1.0 - (token_count - MAX_REASONABLE_TOKENS) / MAX_REASONABLE_TOKENS)
    return round(coverage * length_factor, 4)


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

def phrase_hit(phrase, answer_words):
    """True when every word of `phrase` appears somewhere in the answer,
    regardless of the order the model wrote them in. A phrase alternative
    like 'flight response' matches 'the response to flight' just as well as
    'flight response'."""
    words = [normalise(w) for w in re.findall(r"\w+", str(phrase), re.UNICODE)]
    words = [w for w in words if w]
    if not words:
        return False
    return all(w in answer_words for w in words)


def score_keywords(prediction, expected, item):
    """Coverage of the required points; each point may list synonyms with '|'.
    A point counts as hit when any one of its '|' alternatives has all of its
    words present in the answer, in any order."""
    points = expected.get('keywords') if isinstance(expected, dict) else expected
    points = [p for p in (points or []) if str(p).strip()]
    tokens = count_tokens(prediction)
    if not points:
        return {'keyword_coverage': 0.0, 'all_keywords': 0.0, 'conciseness_score': 0.0}
    answer_words = set(re.findall(r"\w+", normalise(prediction), re.UNICODE))
    found = sum(1 for point in points
                if any(phrase_hit(alt, answer_words) for alt in str(point).split('|') if alt.strip()))
    coverage = found / len(points)
    return {'keyword_coverage': coverage, 'all_keywords': float(found == len(points)),
            'conciseness_score': score_conciseness(coverage, tokens)}


SCORERS = {
    'tool_call': score_tool_call,
    'exact': score_exact,
    'number': score_number,
    'keywords': score_keywords,
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


def answer_text(expected):
    if isinstance(expected, (dict, list)):
        return json.dumps(expected, ensure_ascii=False)
    return str(expected if expected is not None else '')


def write_item_results(rows):
    """Per-question results, in the file the platform's own scorers write.

    finetune_artifact.zip holds all.jsonl (every question: prompt, the right
    answer, the model's answer, is_correct) and failures.jsonl (the ones it got
    wrong). The platform lifts it out of the scoring output after the run. It
    is what the agent analyses question by question, and what training from a
    run's failures reads. Best effort: it can never cost the run its scores.
    """
    try:
        failures = [{'prompt': r['prompt'], 'response': r['response']} for r in rows if not r['is_correct']]
        manifest = {'num_all': len(rows), 'num_correct': len(rows) - len(failures),
                    'num_examples': len(failures), 'format': 'prompt_response_failures',
                    'files': {'all': 'all.jsonl', 'failures': 'failures.jsonl'},
                    'source': 'benchgen_custom_format_scorer', 'kind': KIND}
        path = os.path.join(OUTPUT_DIR, 'finetune_artifact.zip')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('all.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
            archive.writestr('failures.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in failures))
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 scores matter more than this file
        print(f'[scoring] per-question results not written: {exc}', flush=True)


def main():
    if KIND not in SCORERS:
        raise SystemExit(f'KIND is {KIND!r}; expected one of {", ".join(SCORERS)}')
    score_one = SCORERS[KIND]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions = read_json_dir(os.path.join(INPUT_DIR, 'res'))
    reference = read_json_dir(os.path.join(INPUT_DIR, 'ref'))
    by_id = {str(item.get('id', i)): item for i, item in enumerate(reference)}
    print(f'[scoring] {KIND}: {len(predictions)} predictions, {len(reference)} reference items', flush=True)

    preds_by_id = {str(p.get('id', i)): p for i, p in enumerate(predictions)}
    totals = {metric: 0.0 for metric in METRICS[KIND]}
    scored, rows = 0, []
    # Walk the reference, not the answers: a question the model never answered
    # still counts against it, and still gets a per-question row.
    for index, item in enumerate(reference):
        prediction = preds_by_id.get(str(item.get('id', index)))
        answer = prediction.get('prediction') if prediction else ''
        result = score_one(answer, item.get('answer'), item) if prediction else {}
        values = {metric: float(result.get(metric, 0.0)) for metric in METRICS[KIND]}
        for metric, value in values.items():
            totals[metric] += value
        scored += int(prediction is not None)
        overall_keys = OVERALL_METRICS[KIND]
        is_correct = bool(values) and min(values.get(k, 0.0) for k in overall_keys) >= 1.0
        rows.append({
            'id': item.get('id', index),
            'group': (prediction or {}).get('group') or str(item.get('group') or ''),
            'prompt': (prediction or {}).get('prompt') or item.get('question') or '',
            'response': answer_text(item.get('answer')),
            'model_response': answer or '',
            'is_correct': is_correct,
            'scores': values,
        })
    write_item_results(rows)

    # An item the model never answered still counts against it: divide by the
    # reference set, not by what came back.
    denominator = max(len(reference), 1)
    scores = {metric: round(total / denominator, 4) for metric, total in totals.items()}
    overall_keys = OVERALL_METRICS[KIND]
    overall_values = [scores[k] for k in overall_keys if k in scores]
    scores[OVERALL] = round(sum(overall_values) / len(overall_values), 4) if overall_values else 0.0

    with open(os.path.join(OUTPUT_DIR, 'scores.json'), 'w', encoding='utf-8') as handle:
        json.dump(scores, handle, ensure_ascii=False, indent=2)

    write_detailed_results(OUTPUT_DIR, predictions, reference, by_id, score_one)
    print(f'[scoring] scored {scored} of {len(reference)}: {scores}', flush=True)


def write_detailed_results(output_dir, predictions, reference, by_id, score_one):
    """Per-item HTML breakdown, with a summary header and matched/missed points."""
    preds_by_id = {str(p.get('id', i)): p for i, p in enumerate(predictions)}
    cards = []
    total_coverage = 0.0
    total_all = 0
    total_conciseness = 0.0
    total_tokens = 0
    for index, item in enumerate(reference):
        key = str(item.get('id', index))
        prediction = preds_by_id.get(key)
        answer_text = prediction.get('prediction') if prediction else None
        result = score_one(answer_text, item.get('answer'), item)
        coverage = result.get('keyword_coverage', 0.0)
        all_ok = bool(result.get('all_keywords'))
        conciseness = result.get('conciseness_score', 0.0)
        tokens = count_tokens(answer_text)
        total_coverage += coverage
        total_all += int(all_ok)
        total_conciseness += conciseness
        total_tokens += tokens

        points = item.get('answer')
        points = points.get('keywords') if isinstance(points, dict) else points
        points = [p for p in (points or []) if str(p).strip()]
        answer_words_set = set(re.findall(r"\w+", normalise(answer_text), re.UNICODE))
        points_html = ''
        for point in points:
            hit = any(phrase_hit(alt, answer_words_set)
                       for alt in str(point).split('|') if alt.strip())
            cls = 'point-hit' if hit else 'point-miss'
            mark = '✓' if hit else '✗'
            points_html += f'<li class="{cls}">{mark} {esc(point)}</li>'

        status_cls = 'card-pass' if all_ok else ('card-partial' if coverage > 0 else 'card-fail')
        badge = '✅ All points' if all_ok else ('️⚠ Partial' if coverage > 0 else '❌ Missed')
        length_note = 'ideal length' if tokens <= MAX_REASONABLE_TOKENS else 'too long'
        cards.append(f'''
    <div class="card {status_cls}">
      <div class="card-head">
        <span class="qid">Q{esc(key)}</span>
        <span class="badge">{badge}</span>
        <span class="coverage">{coverage:.0%} coverage</span>
        <span class="metric-chip">conciseness {conciseness:.0%}</span>
        <span class="metric-chip">{tokens} tokens ({length_note})</span>
      </div>
      <div class="question">{esc(item.get('question', ''))}</div>
      <div class="answer-label">Model answer</div>
      <div class="answer">{esc(answer_text or '(no answer)')}</div>
      <div class="answer-label">Required key points</div>
      <ul class="points">{points_html}</ul>
    </div>''')

    n = max(len(reference), 1)
    avg_coverage = total_coverage / n
    all_rate = total_all / n
    avg_conciseness = total_conciseness / n
    avg_tokens = total_tokens / n
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Detailed Results</title>
<style>
  :root {{ color-scheme: light; }}
  body {{
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 24px; background: #f6f7fb; color: #1f2430;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: #616a80; font-size: 13px; margin-bottom: 20px; }}
  .summary {{
    display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px;
  }}
  .stat {{
    background: #fff; border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); min-width: 150px;
  }}
  .stat .label {{ font-size: 12px; color: #757e94; text-transform: uppercase; letter-spacing: .04em; }}
  .stat .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
  .cards {{ display: flex; flex-direction: column; gap: 14px; }}
  .card {{
    background: #fff; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 5px solid #ccc;
  }}
  .card-pass {{ border-left-color: #22a35e; }}
  .card-partial {{ border-left-color: #e0a300; }}
  .card-fail {{ border-left-color: #d64545; }}
  .card-head {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap;
  }}
  .metric-chip {{
    font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
    background: #eef2ff; color: #3a4b8f;
  }}
  .qid {{ font-weight: 700; font-size: 13px; color: #43485a; }}
  .badge {{
    font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
    background: #eef0f5; color: #43485a;
  }}
  .coverage {{ margin-left: auto; font-size: 12px; color: #757e94; }}
  .question {{ font-weight: 600; font-size: 15px; margin-bottom: 10px; }}
  .answer-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #9099ad; margin-top: 8px; }}
  .answer {{
    background: #f9fafc; border: 1px solid #eceef3; border-radius: 8px;
    padding: 10px 12px; font-size: 14px; margin-top: 4px; white-space: pre-wrap;
  }}
  .points {{ list-style: none; padding: 0; margin: 6px 0 0; font-size: 13px; }}
  .points li {{ padding: 3px 0; }}
  .point-hit {{ color: #1c7a45; }}
  .point-miss {{ color: #b23b3b; }}
</style></head><body>
<h1>Detailed Results — Atlar Hakkında Temel Bilgiler (Açık Uçlu)</h1>
<div class="subtitle">Per-question breakdown of key-point coverage, conciseness, and answer length</div>
<div class="summary">
  <div class="stat"><div class="label">Questions</div><div class="value">{n}</div></div>
  <div class="stat"><div class="label">Avg. coverage</div><div class="value">{avg_coverage:.0%}</div></div>
  <div class="stat"><div class="label">All points hit</div><div class="value">{all_rate:.0%}</div></div>
  <div class="stat"><div class="label">Avg. conciseness</div><div class="value">{avg_conciseness:.0%}</div></div>
  <div class="stat"><div class="label">Avg. tokens</div><div class="value">{avg_tokens:.0f}</div></div>
</div>
<div class="cards">
{''.join(cards)}
</div>
</body></html>'''
    with open(os.path.join(output_dir, 'detailed_results.html'), 'w', encoding='utf-8') as handle:
        handle.write(html)


def esc(text):
    return (str(text)
            .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


if __name__ == '__main__':
    main()
