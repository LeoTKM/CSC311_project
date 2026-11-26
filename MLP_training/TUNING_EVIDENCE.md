# Original MLP tuning evidence

The repository contains the search implementation and configuration, the selected
hyperparameters, trained weights, and validation/test prediction archives.
It does not contain the original `out/optuna/trials.csv`, `best_params.json`, or
study database. The README previously described those files as available.

The configuration records trial 19 and CV accuracy 0.9092 +/- 0.0186, macro-F1
0.9089; the README described a 60-trial search. This is insufficient to verify
the original trial count, completed/pruned split, competing scores, or selection.
No missing trial history has been fabricated and no replacement search was run.

The original run's owner should export and add `MLP_training/out/optuna/trials.csv`
and `best_params.json`. Ignore rules now permit these compact files while keeping
the database ignored. Check their selected hyperparameters against `config.yaml`
and `out/final_trainval/metrics.json` before treating the gap as resolved.

This gap affects the tuning audit and final-report evidence, not the standalone
predictor's ability to load the existing weights and reproduce archived labels.
