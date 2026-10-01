"""Per-question results, in the file the platform's own scorers write.

Import this from any scoring program:

    from item_results import write_item_results

    rows.append(item_row(item_id, group, prompt, expected, answer, correct, scores,
                         missed=points_not_found))
    ...
    write_item_results(rows, OUTPUT_DIR)

finetune_artifact.zip holds all.jsonl (every question: its prompt, the right
answer, the model's answer, whether it was right, and its per-metric scores)
and failures.jsonl (the ones it got wrong). The platform lifts it out of the
scoring output after the run. It is what the agent analyses question by
question, and what training from a run's failures reads.

The results page (detailed_results.html) is a different thing: it is for
people to read and nothing can analyse it. A benchmark needs this file too.
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
import json
import os
import zipfile

FILENAME = 'finetune_artifact.zip'


def answer_text(value):
    """A reference answer as text: JSON for structures, plain text otherwise."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value if value is not None else '')


def item_row(item_id, group, prompt, expected, answer, correct, scores=None, missed=None):
    """One question's result, in the shape the platform reads.

    missed: what the answer lacked, in the answer key's own words (for key
    points, each point it did not find). Analysis quotes it instead of
    guessing why an answer was marked wrong.
    """
    return {
        'id': item_id,
        'group': str(group or ''),
        'prompt': str(prompt or ''),
        'response': answer_text(expected),
        'model_response': str(answer or ''),
        'is_correct': bool(correct),
        'scores': dict(scores or {}),
        'missed': [str(m) for m in (missed or [])],
    }


def write_item_results(rows, output_dir, source='benchgen_custom_format_scorer'):
    """Write the file. Best effort: it can never cost the run its scores."""
    try:
        failures = [{'prompt': r['prompt'], 'response': r['response']} for r in rows if not r['is_correct']]
        manifest = {'num_all': len(rows), 'num_correct': len(rows) - len(failures),
                    'num_examples': len(failures), 'format': 'prompt_response_failures',
                    'files': {'all': 'all.jsonl', 'failures': 'failures.jsonl'}, 'source': source}
        with zipfile.ZipFile(os.path.join(output_dir, FILENAME), 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('all.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
            archive.writestr('failures.jsonl', '\n'.join(json.dumps(r, ensure_ascii=False) for r in failures))
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f'[scoring] per-question results written for {len(rows)} questions', flush=True)
    except Exception as exc:  # noqa: BLE001 scores matter more than this file
        print(f'[scoring] per-question results not written: {exc}', flush=True)


def _esc(value):
    text = str(value if value is not None else '')
    return (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))


def write_results_page(rows, output_dir, scores=None):
    """detailed_results.html: the run page's "Detailed Results" panel shows
    this file in a frame. Without it the panel is blank. Best effort, like
    the artifact: it can never cost the run its scores."""
    try:
        right = sum(1 for r in rows if r['is_correct'])
        groups = {}
        for r in rows:
            g = groups.setdefault(r['group'] or '(no group)', [0, 0])
            g[0] += int(r['is_correct'])
            g[1] += 1
        parts = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Detailed results</title>',
                 '<style>body{font:14px/1.45 system-ui,sans-serif;margin:16px;color:#222}'
                 'table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #ddd;'
                 'padding:6px 8px;vertical-align:top;text-align:left}th{background:#f5f5f5}'
                 'td.ok{color:#137333;font-weight:600}td.bad{color:#b3261e;font-weight:600}'
                 'pre{white-space:pre-wrap;margin:0;font:12px/1.4 ui-monospace,monospace}</style></head><body>']
        parts.append(f'<h2>{right} of {len(rows)} right</h2>')
        if scores:
            parts.append('<p>' + ', '.join(f'{_esc(k)}: {float(v):.2f}%' for k, v in scores.items()) + '</p>')
        parts.append('<table><tr><th>Group</th><th>Right</th><th>Of</th></tr>')
        for name, (ok, total) in sorted(groups.items()):
            parts.append(f'<tr><td>{_esc(name)}</td><td>{ok}</td><td>{total}</td></tr>')
        parts.append('</table><table><tr><th>#</th><th>Group</th><th>Question</th><th>Expected</th>'
                     '<th>Answer</th><th>Result</th><th>Missed</th></tr>')
        for r in rows:
            cls = 'ok' if r['is_correct'] else 'bad'
            verdict = 'right' if r['is_correct'] else 'wrong'
            parts.append(f'<tr><td>{_esc(r["id"])}</td><td>{_esc(r["group"])}</td>'
                         f'<td><pre>{_esc(r["prompt"])}</pre></td><td><pre>{_esc(r["response"])}</pre></td>'
                         f'<td><pre>{_esc(r["model_response"])}</pre></td><td class="{cls}">{verdict}</td>'
                         f'<td>{_esc("; ".join(r.get("missed") or []))}</td></tr>')
        parts.append('</table></body></html>')
        with open(os.path.join(output_dir, 'detailed_results.html'), 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(parts))
        print(f'[scoring] results page written ({len(rows)} rows)', flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f'[scoring] results page not written: {exc}', flush=True)
