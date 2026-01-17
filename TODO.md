# AI Stocks - Roadmap / TODO

This file tracks planned features, architecture improvements, optimizations, and quality work.


- [x] Model Test & ML Classification Visuals
  - [x] Add machine-learning classification charts in Model Test for both single-model and multi-model modes.
  - [x] Visualize classification performance (precision/recall, confusion-style summaries) alongside price charts.

- [x] AI Model Training & Optimization Strategy
  - [x] Implement small-dataset training mode (single-symbol ~1,700 rows): enable full LightGBM GridSearchCV (≈108 combinations) optimized for precision; expected runtime ~1 minute.
  - [x] Implement large-dataset training mode (Global Model ~400k+ rows): avoid full grid search.
  - [x] Add a "Golden Mix" default LightGBM configuration for global training:
    - `n_estimators = 500`
    - `learning_rate = 0.05`
    - `num_leaves = 31`
    - `max_depth = -1`
    - `class_weight = "balanced"`
    - `n_jobs = -1`
    - with early stopping enabled on a validation split.
  - [x] Add optional lightweight RandomizedSearch (RandomizedSearchCV) for global models with `n_iter ≈ 5` focused on precision.
  - [x] Expose training strategy choices in the admin UI: "Full Grid Search (small data)", "Golden Mix (default)", and "Random Search (fast tuning)".


- [ ] Admin UI: AI & Automation Training Sub-Tabs
  - [x] Add internal sub-tabs: "Classic" (default) + "Genetic Algorithms".
  - [ ] Keep Classic tab as the current LightGBM workflow.
  - [ ] Genetic Algorithms tab: dedicated workspace for evolutionary algorithms.
  - [ ] Add placeholders for controls (population, mutation rate, generations), live charts, and state timeline.

- [ ] Live Training Visualization (Classic)
  - [ ] Stream training iteration metrics via SSE (target: RMSE over iterations).
  - [ ] Add live chart area in Classic tab (learning curve).
  - [ ] Add iteration bar (trees built / total) and phase timeline.
  - [ ] Add an optional "Stop Training" mechanism (cancel flag + safe checkpoints).
