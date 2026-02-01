# AI Stocks Project TODO

## Phase 1: Ensemble Learning (The Council) 🗳️
Implementing a multi-model voting system to improve prediction accuracy and stability.

- [ ] Define the `TheCouncil` class to aggregate predictions from multiple models.
- [ ] Integrate the top 4 performing models into the council:
    - [ ] King Model
    - [ ] Miner Model
    - [ ] collector Model
    - [ ] (Optional) Core Model or Price Action specialist
- [ ] Implement Voting Logic:
    - [ ] **Hard Voting:** Decision based on majority vote (e.g., 3/4 or 4/4).
    - [ ] **Confidence Weighting:** Use model probabilities to weight votes.
- [ ] Update API predict endpoints to use `TheCouncil` instead of a single model.
- [ ] Add "Consensus Score" to the prediction response for better transparency.
- [ ] Verify performance improvement through backtesting with `TheCouncil`.

## Phase 2: Stacking (Meta-Model) 🧠
*Coming soon after Voting is stable.*
- [ ] Train a "Boss" model that takes other models' outputs as features.
- [ ] Implement adaptive weighting based on historical performance in different market conditions.
