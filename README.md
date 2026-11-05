# leakcheck

A small Python package that checks athlete injury-prediction datasets for five kinds of leakage and reports model accuracy against simple baselines.

Created for a Regeneron Science Talent Search 2027 research project by John Tewolde (hoping for good results!)

**AI disclosure:** the code in this repository was written with AI coding assistance (Antigravity, by Google) and reviewed by the author (me, John).

## What it checks

| Check | What it flags |
| --- | --- |
| `check_lag` | features whose source date is on or after the observation date |
| `check_relog` | positive events that fall within a set number of days of the same athlete's previous positive event (repeat entries) |
| `check_carryover` | positive events where the athlete was already positive on the previous row (a persistent state) |
| `check_clock` | days in the study alone separating the label |
| `check_sameday` | a feature that separates the same day's label and scores higher than its one-step-lagged version |

`fix.py` has functions that shift features, collapse repeat entries into episodes and exclude already-injured rows. `report.py` computes AUC, a persistence baseline, a study-clock baseline, excess AUC, a noise floor and a player-clustered bootstrap confidence interval.

## Install and test

```bash
pip install -e .
pip install pytest
pytest -q
```

## Limits

- Passing the checks is necessary for a valid model, not sufficient.
- A check that fires on a file does not show that any published paper is wrong.
- It was validated on synthetic data with planted leaks and on two real datasets (SoccerMon and NFL availability); it has not been tested on published papers' data.
- It does not select features or train models.

## Licence

MIT. See `LICENSE`.
