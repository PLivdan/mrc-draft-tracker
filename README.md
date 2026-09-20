# MRC Draft Tracker

Live Marvel Rivals Championship ban/protect sheet with next-action forecasts,
player hero pools, and conditional playtime forecasts. The embedded model is
fitted on 947 QA-passed competitive maps from 345 series, March 27–September 20,
2026, including the first 79 maps of Ignite 2026 Stage 2, which introduced a
simultaneous fifth ban (stage 9, slot B5). The page drafts in that five-ban
format; the bundled `league_bundle.tar.gz` still holds the 868-map four-ban
history used by the lineup unit tests.

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

## Reproduce the ban/protect model

`fit_draft.py` is the deployment harness for the ban/protect forecasts (hybrid
v5): one chronological as-of feature pass over the processed maps of the
research repo, shared feature weights with per-slot temperatures, a fast/slow
habit mixture for bans and hazard lookahead for protects.

```bash
python -m unittest -v test_fit_draft.py
python fit_draft.py fit --phi 0.25 --out evaluation/draft_params.json --embed index.html
python fit_draft.py holdout --phi 0.25 --start 2026-09-17 --variants v4,own,share,pooled
python fit_draft.py fit --as-of 2026-08-01 --phi 0.25 --out /tmp/v4.json   # reproduces the v4 fit
```

`--data-root` (or `DRAFT_ROOT`) points at the research repo; `--as-of` keeps
maps played on or before a date and heroes released by then. `holdout` refits
on everything before each block of five series and scores the block; `v4`
freezes the coefficients embedded in `index.html`, and the `own`/`share`/
`pooled` variants differ only in how the fifth ban is keyed (its own habit and
temperature; B4's for both; or B4's habit with its own temperature).
Embedding replaces the draft keys of `modelData` and leaves the lineup keys
untouched, so run it before the lineup embed below whenever the roster grew.

### v5 scorecard (2026-09-20)

Restricted to the 868 maps through 2026-08-01 and the 54 heroes released by
then, `fit_draft.py` reproduces the embedded v4 coefficients to within 0.0014
and the habit alphas exactly. Chronological holdout over the 31 Stage 2 series
played by 2026-09-20 (blocks of five series, refit before each block), mean
loss per decision in nats; "v4" is the frozen v4 model scored prospectively
on the slots it knows:

| variant | bans B1–B4 (n=632) | B4 | B5 (n=158) | protects (n=316) |
|---|---|---|---|---|
| v4 frozen | 1.748 | 2.333 | — | 1.800 |
| own-slot B5 | 1.742 | 2.324 | 2.369 | 1.797 |
| share-B4 (shipped) | 1.714 | 2.233 | 2.404 | 1.794 |
| pooled habit, own T | 1.713 | 2.230 | 2.401 | 1.793 |

Paired series bootstrap (4000 draws): share − own is −0.028 [−0.057, −0.003]
on B1–B4 and −0.003 [−0.006, −0.000] on protects, while B5 itself is a wash
(+0.035 [−0.038, +0.10]); pooled − share is −0.001 [−0.002, +0.001]. A
separate fifth-ban habit therefore does not earn its place: the fifth ban is
keyed to B4's late-ban habit and temperature, which is the fewest parameters
among the tied variants. Its main effect is on B4 itself, which is no longer
the final ban. On the four-ban era (last 45 series before Stage 2) the same
code path scores 1.712 / 1.699 (v4 ledger: 1.707 / 1.710). The Stage 2
matches still to be played are the prospective test for this fit.

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
