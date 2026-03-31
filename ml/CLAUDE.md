# Bot-B-Gone ML — Claude Code Agent Instructions

You are an ML research agent running Karpathy's autoresearch loop on a bot detection model.

## Your Mission
Maximize the composite score in `prepare.py` by modifying ONLY `train.py`. Follow the strict "one change at a time" discipline.

## Getting Started
1. Read `program.md` — it has the full experiment history, what worked, what failed, and the next experiments to try
2. Read `results.tsv` — it has every experiment's score
3. Read `train.py` — this is the file you modify
4. Read `prepare.py` — understand the evaluation metrics (but NEVER modify it)

## Rules
1. **ONE change at a time.** Never modify more than one concept per experiment.
2. **NEVER modify `prepare.py`** — it's the immutable evaluation harness.
3. **NEVER modify `data/soft_labeled.csv`** — it's the immutable training data.
4. **Git commit every experiment** — `git add train.py && git commit -m "expN: description"`
5. **Revert failures** — `git checkout HEAD~1 -- train.py` if composite didn't improve.
6. **Log analysis** — After each run, explain WHY the score moved before trying the next thing.
7. **Never stop** — keep running experiments until interrupted.

## Current State
- Best composite: **94.00/100** (Experiment 40)
- Bottleneck: Spread (84.18/100) — model hedges on 54% ambiguous events
- Hyperparameters are SATURATED — don't tune learning_rate, min_child_samples, etc.
- Next experiments: Start at **Experiment 46** (feature ablation) in program.md

## How to Run Each Experiment
```bash
python3 train.py 2>&1 | tee run.log
grep "COMPOSITE SCORE\|Validation Composite\|Test Composite" run.log
```

## Key Insight
The only path past 94.00 is better FEATURES or better LABELS. Not hyperparameters.
Spread and MSE are in tension — you need changes that make the model **confidently right**, not just more extreme.
