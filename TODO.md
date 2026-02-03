# AI Stocks Project TODO

## Phase 1: Radar Layer (Locked) 📡

> **Single source of truth for opportunity detection**

* [x] Lock `collector 🎁.pkl` as the **only Radar model**
* [x] Target: 3% | High Recall
* [ ] Expose Radar output as `DETECTED` only (no BUY/SELL)
* [ ] Log all Radar detections for downstream learning

---

## Phase 2: Ensemble Filtering (TheCouncil) 🗳️

> **Consensus-based noise reduction (NOT final decision)**

### Council Composition

* [ ] Implement `TheCouncil` class
* [ ] Council Members (diverse roles only):

  * [ ] collector 🎁 (Recall / Momentum)
  * [ ] KING 👑 (Context / AUC)
  * [ ] (Optional) Price Action Specialist

### Voting Logic

* [ ] Implement **Weighted Soft Voting** (no hard voting)
* [ ] Define static initial weights:

  * collector: 0.25
  * miner: 0.25
  * KING: 0.40
  * PA: 0.10
* [ ] Compute `ConsensusStrength = Σ(probability × weight)`
* [ ] Council Pass Threshold:

  * Radar Filter: `Consensus ≥ 0.55`

### API Changes

* [ ] Update predict endpoint:

  * Input: Radar-detected symbols only
  * Output:

    ```json
    {
      "status": "FILTERED",
      "consensus_strength": 0.68,
      "layers_passed": ["Radar", "Council"]
    }
    ```

---

## Phase 3: Big Move Validation (Striker) 👑

> **Final gate for 10% moves**

* [ ] Lock `KING 👑.pkl` as **final validator only**
* [ ] Apply only to Council-approved candidates
* [ ] Validation Threshold:

  * Meta probability ≥ 0.60
* [ ] Final Output:

  ```json
  {
    "status": "CONFIRMED",
    "target": "10%",
    "confidence": 0.72,
    "layers_passed": ["Radar", "Council", "KING"]
  }
  ```

---

## Phase 4: Backtesting & Logs 📊

* [ ] Backtest full pipeline (Radar → Council → KING)
* [ ] Report:

  * Trades count
  * Avg return
  * Max DD
  * Time-to-target
* [ ] Store prediction outcomes for learning

---

## Phase 5: Stacking Meta-Model (Boss) 🧠 *(Future)*

> **Adaptive intelligence (DO NOT IMPLEMENT YET)**

* [ ] Train Boss model on:

  * Council scores
  * KING confidence
  * Market regime features
* [ ] Adaptive weighting by regime (bull / bear / chop)
* [ ] Replace static Council weights only after sufficient logs

---

## Naming & Compliance Rules ⚠️

* ❌ BUY / SELL
* ❌ Signal / Recommendation
* ✅ DETECTED / FILTERED / CONFIRMED
* ✅ Consensus Strength
* ✅ Market Opportunity

---

## Golden Rule

> **Radar finds → Council filters → KING confirms**
> No model decides alone.
