# Atlar Hakkında Temel Bilgiler (Açık Uçlu)

Benchmark 97 on the BenchGen platform.

This folder is the benchmark as the platform holds it: competition.yaml, the
scoring program, the questions and the answers behind it. The history of this
folder is the history of the benchmark, so a change to the scoring or the answer
key is a diff you can read.



## Using it again

The platform can build a benchmark straight from this folder:

```json
{"repo": "https://github.com/tahsinSoyakUbosAI/mybenchmark", "path": "benchmarks/atlar-hakkinda-temel-bilgiler-acik-uclu-97"}
```

POST that to /api/competitions/create_from_repo/ (add ?dry_run=1 to check first).
A private repository also needs "connection", the id of your git connection.
