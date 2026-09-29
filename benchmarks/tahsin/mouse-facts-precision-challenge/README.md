# Mouse Facts: Precision Challenge

Benchmark 106 on the BenchGen platform.

This folder is the benchmark as the platform holds it: the scoring program, the
questions and the answers behind it. The history of this folder is the history of
the benchmark, so a change to the scoring or the answer key is a diff you can read.



## Using it again

The platform can build a benchmark straight from this folder:

```json
{"repo": "https://github.com/tahsinSoyakUbosAI/mybenchmark", "path": "benchmarks/tahsin/mouse-facts-precision-challenge"}
```

POST that to /api/competitions/create_from_repo/ (add ?dry_run=1 to check first).
