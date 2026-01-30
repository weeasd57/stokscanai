
## 🚀 Model & UI Improvements Roadmap

### 📋 General
- [ ] Add **Learning Rate** in model cards (Metadata display)

### 🧠 Backend Strategy (`api/train_exchange_model.py`)
- [ ] **Fix Look-Ahead Bias**: Ensure targeting doesn't leak future data (especially `next_open`).
- [ ] **Feature Computation Optimization**: Implement preset-based feature sets (`core`, `extended`, `max`) to avoid computing unused indicators.
- [ ] **Symbol Retention**: Lower history requirements for simpler feature presets to include newer or less liquid symbols.
- [ ] **Data Loading Optimization**: Implement parallel fetching with connection pooling.
- [ ] **Feature Caching**: Add disk-based caching for computed features to speed up retraining.
- [ ] **Memory Management**: Use downcasting and categorical types for large DataFrames.
- [ ] **Architecture Refactoring**: Transition to a class-based `ModelTrainer` to separate data, features, training, and evaluation logic.
- [ ] **Data Validation**: Add robust pre-training validation (null checks, column verification, data size).
- [ ] **Handle Class Imbalance**: Implement SMOTE or weighted loss to address the low frequency of positive signals.
- [ ] **Enhanced Evaluation**: Use PR-Curves and F1-score optimization instead of just accuracy.
- [ ] **Model Versioning**: Implement formal version tracking and metadata persistence.
- [ ] **Feature Importance Analysis**: Automatically identify and prune low-importance features to simplify the model.

### 🔄 Adaptive & Active Learning (`api/adaptive_learning.py`)
- [ ] **Fix Incremental Learning Logic**: Correct usage of `init_model` and ensure the native LightGBM API is used for proper weighted updates.
- [ ] **Refine Mistake Detection**: Differentiate between "False Positives" (wrong signals) and "Missed Opportunities" (False Negatives) for targeted refinement.
- [ ] **Implement Stop-Loss Verification**: Add checks to see if a prediction would have hit a stop-loss before reaching the target.
- [ ] **Address Race Conditions**: Ensure thread-safety when updating shared resources like model logs or the model itself in background workers.
- [ ] **Performance Tracking**: Implement a "Safety Check" that compares the updated model's performance against the previous version before swapping.

### 🎨 Frontend Excellence (`TestModelTab`)

- [ ] **Component Decomposition**: Break down the massive `TestModelTab` into modular sub-components.
- [ ] **Performance Virtualization**: Use virtual scrolling for large prediction tables (e.g., `@tanstack/react-virtual`).
- [ ] **Strict Type Safety**: Replace `any` types with comprehensive interfaces for models and predictions.
- [ ] **Optimize Rendering**: Apply `React.memo` and `lazy` loading for heavy chart components.
- [ ] **Robust Error Handling**: Enhance multi-model test tracking to show specific failures without stopping the entire batch.
- [ ] **UX Polish**: Add skeleton loaders, toast notifications, and debounced search filters.
- [ ] **State Management**: Migrate complex component state to `useReducer`.
- [ ] **Automated Testing**: Add unit tests for KPI calculations and integration tests for the test runners.

