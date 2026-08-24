# Final empirical decision record

## RQ1 — Incremental technical-indicator information

Across Log-HAR-X, LightGBM, and LSTM at 5-, 10-, and 20-day horizons, no technical-indicator augmentation produces a positive B→C QLIKE gain that is significant under both Holm-adjusted HAC and moving-block-bootstrap inference.

One robust deterioration is present: Log-HAR-X at the 20-day horizon (negative incremental QLIKE gain; adjusted HAC and bootstrap p-values below 0.05).

The primitive information set B improves materially over volatility-memory-only A across all three architectures, whereas the additional technical transformations C do not provide stable incremental value.

## RQ2 — Liquidity-conditioned heterogeneity

Descriptively, technical-indicator gains are often less favorable in the high-liquidity state, especially for LightGBM and LSTM. Formal daily low-minus-high QLIKE-differential contrasts, however, do not survive Holm adjustment across the three horizons.

- LOG_HAR_X, 5 days: low-minus-high mean loss-differential contrast=0.002986; HAC p=0.5739, Holm=1.0000; bootstrap p=0.5822, Holm=1.0000.
- LOG_HAR_X, 10 days: low-minus-high mean loss-differential contrast=-0.001065; HAC p=0.8089, Holm=1.0000; bootstrap p=0.8032, Holm=1.0000.
- LOG_HAR_X, 20 days: low-minus-high mean loss-differential contrast=0.000273; HAC p=0.9261, Holm=1.0000; bootstrap p=0.9230, Holm=1.0000.
- LSTM, 5 days: low-minus-high mean loss-differential contrast=0.014097; HAC p=0.0229, Holm=0.0688; bootstrap p=0.0374, Holm=0.1122.
- LSTM, 10 days: low-minus-high mean loss-differential contrast=0.013759; HAC p=0.0518, Holm=0.1035; bootstrap p=0.0526, Holm=0.1122.
- LSTM, 20 days: low-minus-high mean loss-differential contrast=0.015658; HAC p=0.2805, Holm=0.2805; bootstrap p=0.2712, Holm=0.2712.
- LightGBM, 5 days: low-minus-high mean loss-differential contrast=0.015498; HAC p=0.0241, Holm=0.0722; bootstrap p=0.0338, Holm=0.1014.
- LightGBM, 10 days: low-minus-high mean loss-differential contrast=0.008180; HAC p=0.1827, Holm=0.2300; bootstrap p=0.1826, Holm=0.2276.
- LightGBM, 20 days: low-minus-high mean loss-differential contrast=0.008905; HAC p=0.1150, Holm=0.2300; bootstrap p=0.1138, Holm=0.2276.

## Decision

Stop model expansion. Treat RQ1 as confirmatory and RQ2 as secondary/suggestive. Do not add XGBoost, Random Forest, Transformer, TCN, or stacking solely to search for a positive technical-indicator result.

Primary reporting order: (1) sample and information-set design; (2) A→B progression; (3) B→C incremental test across architectures; (4) stock-level breadth; (5) indicator-family ablations; (6) liquidity heterogeneity; (7) GJR-GARCH benchmark and robustness.