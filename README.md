# MRC Draft Tracker

Live ban/protect sheet for the Marvel Rivals Championship draft, with next-pick
forecasts from a soft-denial conditional-logit model fitted on the full
Ignite 2026 record (868 competitive maps).

Open `index.html` in a browser, or use the hosted page. Everything runs
client-side: no server, no accounts. Draft state persists in the browser.

- Enter picks as they are revealed; legality is enforced exactly
  (bans remove heroes from the opponent's pool, protects block opponent bans,
  stages 1 and 5 are simultaneous blind picks).
- The model forecast for the next pick appears under the active stage, with
  the utility decomposition (slot habit, team capability, denial).
- Click a forecast hero to explore that line as a counterfactual; the actual
  draft is preserved.
- Roll forward: Monte Carlo over the remaining draft.
- Player pools: each player's revealed heroes, struck out live as bans land.
- Light/dark toggle.

Model and data pipeline: separate research repo (draft records scraped from
mrvl.net, chronological leave-series-out evaluation). Parameters are embedded
in the page and refreshed as the pipeline improves.
