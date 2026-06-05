# Walk-1 feature pruning v1

**Date:** 2026-06-01

**Rule:** Drop features where `permutation_importance <= 0.0001` OR feature is a USD-duplicate of a non-USD counterpart.


## Summary

- Total features (walk-1 ranker): 190
- Dropped by low PI:    29
- Dropped by USD-dup:   8
- Total dropped (union): 36
- **Total kept: 154**

## By type

```
            dropped  kept
type                     
crsp_price        0     8
macro             1     3
pca_text          5    74
sharadar         30    66
text_aux          0     3
```

## Dropped features
```
  assetsc                   PI=+0.000096  type=sharadar  reason=low_PI
  bvps                      PI=-0.002283  type=sharadar  reason=low_PI
  cashnequsd                PI=+0.000565  type=sharadar  reason=usd_dup
  consolinc                 PI=-0.000189  type=sharadar  reason=low_PI
  debt                      PI=+0.000094  type=sharadar  reason=low_PI
  debtusd                   PI=+0.000324  type=sharadar  reason=usd_dup
  deposits                  PI=-0.000840  type=sharadar  reason=low_PI
  divyield                  PI=-0.005817  type=sharadar  reason=low_PI
  dps                       PI=+0.000000  type=sharadar  reason=low_PI
  ebitdausd                 PI=+0.000123  type=sharadar  reason=usd_dup
  ebitusd                   PI=+0.000919  type=sharadar  reason=usd_dup
  eps                       PI=-0.000484  type=sharadar  reason=low_PI
  epsdil                    PI=+0.000072  type=sharadar  reason=low_PI
  epsusd                    PI=+0.002792  type=sharadar  reason=usd_dup
  equity                    PI=-0.002109  type=sharadar  reason=low_PI
  equityusd                 PI=+0.001320  type=sharadar  reason=usd_dup
  evebit                    PI=-0.001244  type=sharadar  reason=low_PI
  gp                        PI=-0.000830  type=sharadar  reason=low_PI
  investments               PI=-0.004590  type=sharadar  reason=low_PI
  liabilitiesnc             PI=-0.000159  type=sharadar  reason=low_PI
  macro_dgs10               PI=-0.000256  type=macro  reason=low_PI
  ncfdiv                    PI=-0.002021  type=sharadar  reason=low_PI
  netinc                    PI=+0.000000  type=sharadar  reason=low_PI
  netinccmn                 PI=-0.000334  type=sharadar  reason=low_PI
  netinccmnusd              PI=+0.000292  type=sharadar  reason=usd_dup
  netincdis                 PI=-0.000012  type=sharadar  reason=low_PI
  netincnci                 PI=+0.000000  type=sharadar  reason=low_PI
  pca_17                    PI=-0.000127  type=pca_text  reason=low_PI
  pca_20                    PI=-0.000217  type=pca_text  reason=low_PI
  pca_33                    PI=-0.000627  type=pca_text  reason=low_PI
  pca_37                    PI=-0.000486  type=pca_text  reason=low_PI
  pca_52                    PI=-0.000019  type=pca_text  reason=low_PI
  revenueusd                PI=-0.000001  type=sharadar  reason=usd_dup, low_PI
  rnd                       PI=-0.000747  type=sharadar  reason=low_PI
  tangibles                 PI=+0.000003  type=sharadar  reason=low_PI
  taxassets                 PI=+0.000012  type=sharadar  reason=low_PI
```
