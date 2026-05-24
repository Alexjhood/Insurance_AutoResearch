# Experiment Protocol

## Evaluation layers
- Search evaluation: fast and cheap
- Promotion evaluation: repeated resampling + paired comparison
- Milestone evaluation: frozen holdout only

## Promotion principle
A candidate is not promoted on point estimate alone.
Promotion should consider:
- mean score
- variability across resamples
- paired lift vs current champion
- bootstrap interval
- win rate vs current champion

## Required artifacts per experiment
- config snapshot
- code version / git hash if available
- metrics json
- uncertainty summary
- plots
- rationale text
- change summary