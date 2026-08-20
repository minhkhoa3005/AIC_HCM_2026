# BTC Answer Files

Store one ground-truth text file per query. The filename must match the query
stem, for example:

```text
answer/
├── query-1-kis.txt
├── query-2-qa.txt
└── query-4-trake.txt
```

Supported fields are `Video ID`, `Frame Range: [start, end]`, optional
`Answer`, and for TRAKE `Event N: [start, end]`. Ranges are inclusive.

Evaluate result files with:

```powershell
python answer/evaluate.py --answer-dir answer --result-dir outputs/results
```
