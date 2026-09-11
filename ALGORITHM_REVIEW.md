# Draft tracker: assessment and implemented improvements

## The useful product

The tracker can reduce the amount a player or coach has to remember during a
draft. Given two teams, a map, the previous map and the revealed choices, it can
answer three practical questions: what is the opponent likely to do next;
which heroes remain plausible for each player; and how does a proposed line
change those forecasts? Those are useful scouting tasks even when the most
likely exact pick is wrong more often than it is right.

The central statistical advantage is shared information. A team with twenty
maps does not need a completely separate model: league habits estimate the
common component, with team and player histories supplying smaller deviations.
The existing conditional-logit draft model already follows this logic. A large
new model is not the obvious next step.

The distinction between a forecast and a recommendation matters. A hero being
frequently banned can identify a common opponent habit. It does not establish
that banning that hero will improve your chance of winning. Selecting a
hypothetical ban and recomputing the remaining pools is useful scenario
analysis, but the learned protect and map coefficients are observational.

## What the supplied data supports

The GitHub Pages HTML matched the repository exactly when inspected. The
supplied archive contains 884 map records: 868 passed QA and 16 were rejected.
The passed maps span 314 series and 54 teams, March 27–August 1, 2026. Team
sample sizes range from 2 to 71 maps, with a median of 27. On September 11 the
dataset is about six weeks behind the live game. Several team snapshots are
older still. All event names identify Ignite competitions; no separately
labeled scrim sample is supplied.

Each map has draft actions with phase, actor and hero; team and player IDs;
player hero playtime; map; timestamp; and winner. These records support
conditional draft-choice forecasts and map-level hero-use forecasts. They do
not identify an opening lineup, simultaneous hero occupancy, ability use,
positions, fight outcomes, or why a player swapped. A player's most-used hero
over a map is not necessarily their starting hero. Two teammates can have the
same most-used hero at different times.

The number of player rows is not the number of independent observations.
Teammates share a match, consecutive maps share a series, and series share a
meta. Uncertainty comparisons should respect those clusters. More useful
observations would be fresh, accurately labeled matches, especially roster and
patch transitions, rather than duplicated player rows from existing matches.

## Problems fixed in this branch

| Finding | Change |
|---|---|
| The lineup harness loaded rejected records. | One loader filters strictly to `qa.status == pass`; it also rejects duplicate map IDs and invalid playtime. |
| Development and replay series could overlap in clock time. | Series ending at or after the replay start are excluded from parameter selection, even if their first map was earlier. |
| The evaluator used all heroes; the page received at most eight per player. | Export every modeled hero with full precision. Candidate truncation now affects only the abbreviated history display. |
| The page conditioned lineups using the ban model's map offsets, while the lineup harness used another estimator. | Export the lineup estimator's own offsets and use those in both runtimes. Map totals now count team-maps consistently with six player shares. |
| Alternatives were selected in the old, unconditioned order. | Sort again after all current bans, protects and map adjustments. |
| Python used exact assignment, while the page searched a truncated candidate pool. | Both use the same deterministic Hungarian assignment over the full pool, including a null assignment when no legal candidate exists. |
| Some projected heroes were absent from the displayed history rows. | Add every displayed primary and alternative to the visible pool, including low-history and prior-driven candidates. |
| The old box-coverage metric measured marginal top-two heroes rather than the displayed assignment and thresholded alternative. | Score the actual boxes, with the same 10% alternative threshold used on the page. |
| One completed-draft accuracy figure appeared throughout the draft. | Report the corresponding opening, first-protect, second-blind-ban or complete-draft checkpoint. |
| Historical and projected percentages could be confused. | Label them separately and show dataset and last observed roster dates. |

The draft phase logic, legality rules, actual-versus-hypothetical draft state,
and existing v4 ban/protect coefficients are preserved. The change is concentrated
on the lineup model and on making its evidence and display agree.

## The model change

Let \(C_{ih}\) be player \(i\)'s decayed sum of playtime shares on hero \(h\),
and \(A_{ih}\) their decayed number of maps on which that hero was not banned.
The existing estimator shrinks the player toward team and league usage:

\[
s_{th}=\frac{T_{th}+\kappa_t g_h}{\sum_jT_{tj}+\kappa_t}.
\]

The weakness is that this prior distributes a specialist's scarce observations
over the entire team's hero pool. A support player inherits some tank and
duelist probability simply because their teammates play those roles.

The new prior learns the player's role mixture from their own past playtime.
For role \(r\), let \(s_{tr}=\sum_{h\in r}s_{th}\), and set

\[
\pi_{ir}=\frac{\sum_{h\in r}C_{ih}+s_{tr}}{\sum_hC_{ih}+1},
\qquad
\widetilde s_{ih}=\pi_{ir(h)}\frac{s_{th}}{s_{t,r(h)}}.
\]

This is a soft prior. It retains flex play and gives a player with no history
the original team prior. It does not use their current map's observed role or
impose a hard role lock. Candidate probabilities then satisfy

\[
q_{ih}\ \propto\
\frac{C_{ih}+\kappa_p\widetilde s_{ih}}{A_{ih}+\kappa_p}
\exp\!\left(b\,1\{h\text{ protected}\}+\ell\,m_h\right)
1\{h\text{ not banned against the team}\}.
\]

The selected configuration uses player decay 0.88 per observed appearance,
player prior strength 2, team prior strength 50, protect loading 1 and map
loading 0.5. Team and league decay remain 0.88 and 0.995. The new ingredient
selected by the development data is the role-conditioned prior. This is an
exposure-adjusted predictive score, not an exact conjugate Bayesian posterior:
fractional shares, discounting, unequal availability and later conditioning do
not support an automatic Dirichlet credible interval.

For the primary boxes, the display maximizes the sum of log marginal shares
subject to distinct heroes. Costs are rounded at twelve decimal places so
equivalent assignments have deterministic behavior across runtimes. This
presentation constraint does not turn the product of marginal shares into a
calibrated joint probability of a six-player composition. Alternatives are
per-player possibilities and can overlap with another player's primary.

## Measured result

The earlier development window contains 121 maps in 44 complete series. The
existing final 45-series replay contains 157 maps and 1,884 player-map
observations. Selection uses only the earlier development window. Historical
state updates occur after each map; the current map's actions are included only
up to the checkpoint being scored, and its playtime enters history only after
predictions.

The comparison below isolates the new pooling from a corrected version of the
old pooling with the same QA filtering, candidate set, map estimator, assignment
and display metric. It is **not** a replay of the buggy former browser output.

| Completed-draft metric | Corrected original pooling | Role-aware pooling |
|---|---:|---:|
| Playtime-share cross-entropy, lower is better | 2.0163 | 1.8907 |
| Primary matches the player's most-used hero | 44.64% | 44.64% |
| Either displayed box contains that hero | 62.05% | 62.90% |
| Actual playtime covered by displayed boxes | 58.90% | 59.80% |

Cross-entropy falls 6.23%. The improvement is principally in the probability
distribution; it does not produce a higher exact-primary hit rate. The paired
series bootstrap interval for the cross-entropy change is approximately
[-0.140, -0.112] nats per player-map. This describes uncertainty within this
historical sample; it does not account for the many earlier model searches or
guarantee transfer to a new patch.

| Draft information available | Primary right | Either box right | Playtime covered |
|---|---:|---:|---:|
| Before any picks | 33.81% | 52.65% | 49.62% |
| After both first protects | 36.68% | 55.68% | 52.83% |
| After the second blind bans | 39.28% | 58.81% | 55.89% |
| Complete draft | 44.64% | 62.90% | 59.80% |

This replay window was used repeatedly by earlier repository experiments. It
cannot honestly be called an untouched holdout. All metrics also assume the
recorded roster is known; 99.31% of replay player rows match their team's
previous observed roster. The static page cannot automatically update player
histories after a map, whereas the historical replay does. These limits are
included in the machine-readable report and the page's explanatory text.

## What to do next

1. **Refresh and date the evidence.** Bring in matches after August 1 and
   verified event eligibility for every hero. Store actual result-availability
   timestamps as well as played-at timestamps. Associate observations with
   patches, teams and roster IDs. A predictive model cannot recover a new meta
   from matches it has never seen.

2. **Freeze a genuine forward test.** Save forecasts before the next event,
   with snapshot time, team roster, map, revealed actions and legal set. Compare
   league-only habits, team personalization and the full model on the same
   decisions. Score log loss, top-three coverage and probability calibration
   by draft slot; score lineup share error and actual displayed-box coverage
   separately. Once that event is used for tuning, stop calling it a holdout.

3. **Distinguish roster change from lack of evidence.** Let an analyst confirm
   the announced six players and show which forecast is mostly prior-driven.
   Pool specialist behavior across team moves using player IDs, while shrinking
   team habits when the roster changes. Current player decay is per appearance;
   inactive players do not automatically forget old hero preferences with
   calendar time. Test a calendar/patch clock after adding recent observations.

4. **Evaluate at the update cadence the product can sustain.** The current
   replay refreshes state after every map. Report an event-frozen or
   series-frozen variant before using its numbers to describe a page that is
   only refreshed before an event. This may matter more than another feature.

5. **Make scenario analysis useful without claiming a win effect.** A coach
   can inspect which players are forced off their likely heroes, the plausible
   replacements and where the forecast depends on little history. Those
   changes are directly connected to the data. Estimating the effect of an
   imposed ban on winning requires stronger data and identification than a
   high pick-prediction score.

6. **Add scrims only with provenance.** Source, date, roster, opponent and
   experimental context should travel with each record. Tune any scrim discount
   on future competitive forecasts; do not assign a universal weight simply
   because a prior sounds Bayesian. The supplied archive cannot identify an
   appropriate scrim weight.

Full hero-pair, team-pair, map, roster and patch interactions would multiply
parameters faster than this sample supplies independent information. A neural
fight model or a reinforcement-learning optimal draft is a separate data
collection project. The feasible path here is better evidence, reliable
chronology and partial pooling around the scouting task the tracker already
serves.

## Reproduction and checks

Run `python evaluate_lineup.py --embed index.html` from this repository. It
reads the supplied archive, writes `evaluation/lineup_report.json`, checks
Python/JavaScript agreement on 96 historical draft states, and updates the
embedded lineup payload. `python -m unittest -v test_lineup.py` also checks
QA exclusion, chronological boundaries, prefix invariance, blind-phase masks,
complete exports, legal empty pools and matching against SciPy's independent
assignment solver. JavaScript syntax and local asset references were checked.
No browser visual QA was run.
