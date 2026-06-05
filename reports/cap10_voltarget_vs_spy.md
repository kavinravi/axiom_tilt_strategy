# cap10 + vol-targeting overlay vs SPY

target=16.5%, lookback=16w, max_leverage=1.0, cash=zero

```
=== 2009-2025 (full) ===
strat                   weeks     totret     annret        vol    sharpe   sortino       mdd    calmar       hit
cap10 (no overlay)        855   4619.47%     26.42%     32.16%     0.821     1.280   -54.56%     0.484    54.39%
cap10 + voltarget         855    779.04%     14.13%     21.73%     0.650     0.976   -40.86%     0.346    54.39%
SPY                       855    898.42%     15.02%     17.35%     0.866     1.138   -31.83%     0.472    58.83%

=== 2010-2024 ===
strat                   weeks     totret     annret        vol    sharpe   sortino       mdd    calmar       hit
cap10 (no overlay)        757   1966.64%     23.13%     28.76%     0.804     1.191   -54.56%     0.424    54.43%
cap10 + voltarget         757    505.06%     13.16%     17.79%     0.740     1.080   -37.16%     0.354    54.43%
SPY                       757    580.66%     14.08%     16.65%     0.846     1.094   -31.83%     0.442    59.05%

=== 2010-2025 ===
strat                   weeks     totret     annret        vol    sharpe   sortino       mdd    calmar       hit
cap10 (no overlay)        806   1648.54%     20.27%     28.58%     0.709     1.055   -54.56%     0.372    53.97%
cap10 + voltarget         806    441.13%     11.51%     17.82%     0.646     0.940   -37.16%     0.310    53.97%
SPY                       806    691.77%     14.28%     16.67%     0.856     1.105   -31.83%     0.449    58.93%
```
