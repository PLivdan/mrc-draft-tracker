# MRC Draft Tracker

Live Marvel Rivals Championship ban/protect sheet with next-action forecasts,
player hero pools, and conditional playtime forecasts. The bundled data contains
868 QA-passed competitive maps from 314 series, March 27–August 1, 2026.

Open `index.html` in a browser with `lineup-engine.js` beside it, or use the
GitHub Pages site. Everything runs client-side; the actual draft and theme
persist in browser storage.

- Enter bans and protects as revealed. Stages 1 and 5 remain simultaneous:
  neither pick is applied until both are entered and valid.
- Next-action forecasts describe likely choices. Clicking one explores a
  hypothetical line while preserving the actual draft.
- Projected player pools respond to the current bans, protects and map.
  Solid boxes mark distinct primary heroes; dashed boxes mark individual
  alternatives. Percentages are expected playtime shares, not win probabilities
  or probabilities of a complete six-hero composition.
- The page shows the dataset date, last observed roster dates, and replay
  accuracy at the corresponding draft checkpoints.

## Reproduce the lineup model

Python with NumPy/SciPy and Node.js are needed for evaluation and runtime checks.
The website itself needs neither Python nor a server.

```bash
python -m pip install -r requirements-lineup.txt
python evaluate_lineup.py --embed index.html
python -m unittest -v test_lineup.py
```

The evaluator reads `league_bundle.tar.gz` directly by default, filters rejected
maps, selects parameters on an earlier development window of whole series,
then reports the existing 45-series historical replay. It compares the
corrected original pooling approach with role-aware pooling, checks Python/JS
agreement, and exports full distributions plus their matching map offsets.

Outputs are `evaluation/lineup_report.json` and
`evaluation/lineup_params.json`. The latter is a reproducible intermediate and
is ignored by Git; the deployment payload lives in `index.html`.
`--embed` updates only lineup parameters, history, validation and provenance;
the embedded ban/protect coefficients remain the existing v4 fit.

For newer processed data:

```bash
python evaluate_lineup.py --data-root /path/to/marvel-draft-model --embed index.html
```

The data root must contain `data/roles/heroes.json` and
`data/processed/maps/*.json`. `DRAFT_ROOT` is also supported. The other
harnesses are historical research experiments, not deployment entrypoints.

Do not treat the existing last-45-series window as an untouched test set:
earlier experiments already used it. Before making claims about current-event
performance, freeze a model and test on newly collected series. Replay assumes
recorded rosters and a state refresh after every completed map; the live page
uses its last exported snapshot.

See [ALGORITHM_REVIEW.md](ALGORITHM_REVIEW.md) for the model, measured results,
implementation fixes, limitations, and the next improvements worth making.
