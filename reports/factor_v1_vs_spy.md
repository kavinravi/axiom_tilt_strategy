# Factor screen v1 vs SPY (deterministic backtests)

```

=== 2009-2025 (full) ===
strategy                    weeks     totret     annret        vol    sharpe       mdd    calmar
factor_v1+mcap              855    803.36%     14.32%     16.74%     0.856   -37.91%     0.378
factor_v1+equal             855    834.58%     14.56%     17.46%     0.834   -41.89%     0.348
factor_v1+invvol            855    684.69%     13.35%     16.03%     0.832   -40.47%     0.330
factor_v1+minvar            855    432.93%     10.71%     14.23%     0.753   -36.50%     0.294
factor_v1+maxsharpe         855    582.01%     12.39%     18.31%     0.676   -39.28%     0.315
PRIOR mcap (LightGBM)         855   3872.41%     25.10%     28.27%     0.888   -43.36%     0.579
SPY (benchmark)               855    898.42%     15.02%     17.35%     0.866   -31.83%     0.472


=== 2010-2024 ===
strategy                    weeks     totret     annret        vol    sharpe       mdd    calmar
factor_v1+mcap              757    678.08%     15.13%     16.70%     0.906   -37.91%     0.399
factor_v1+equal             757    715.54%     15.51%     17.46%     0.888   -41.89%     0.370
factor_v1+invvol            757    596.51%     14.26%     16.13%     0.884   -40.47%     0.352
factor_v1+minvar            757    422.11%     12.02%     14.33%     0.839   -36.50%     0.329
factor_v1+maxsharpe         757    535.39%     13.54%     18.33%     0.739   -39.28%     0.345
PRIOR mcap (LightGBM)         757   1938.00%     23.01%     25.08%     0.917   -43.36%     0.531
SPY (benchmark)               757    606.63%     14.38%     16.67%     0.862   -31.83%     0.452


=== 2010-2025 (BAR vs SPY) ===
strategy                    weeks     totret     annret        vol    sharpe       mdd    calmar
factor_v1+mcap              806    731.72%     14.64%     16.67%     0.879   -37.91%     0.386
factor_v1+equal             806    798.44%     15.22%     17.31%     0.879   -41.89%     0.363
factor_v1+invvol            806    654.90%     13.93%     15.97%     0.872   -40.47%     0.344
factor_v1+minvar            806    414.04%     11.14%     14.24%     0.782   -36.50%     0.305
factor_v1+maxsharpe         806    624.97%     13.63%     18.23%     0.748   -39.28%     0.347
PRIOR mcap (LightGBM)         806   1864.50%     21.18%     25.33%     0.836   -43.36%     0.488
SPY (benchmark)               806    723.09%     14.57%     16.70%     0.872   -31.83%     0.458

```
