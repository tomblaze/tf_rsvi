# tf_rsvi
Pure Tensorflow implementation of Rejection Sampling Variational Inference.

Code implements a sparse Gamma Deep Exponential Family model with Rejection Sampling VI.

Citations:
```
Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms.
Christian A. Naesseth, Francisco J. R. Ruiz, Scott W. Linderman, and David M. Blei
Proceedings of the 20th International Conference on Artificial Intelligence and Statistics 2017,
Fort Lauderdale, Florida, USA.
```

```
Deep Exponential Families.
Rajesh Ranganath, Linpeng Tang, Laurent Charlin, David M. Blei.
Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics,
PMLR 38:762-771, 2015.
```

Code is deeply indebted to the work at [blei-lab/ars-reparameterization](https://github.com/blei-lab/ars-reparameterization).

That code provides a few more examples (work in progress) but is designed to run on CPU. This code runs on Tensorflow v1.4 and runs ~10x faster on a GTX 980.

Unconditional Poisson samples from model after 500 steps...

![unconditional samples](https://github.com/tomblaze/tf_rsvi/blob/master/unconditional_sample.png)

ELBO over course of training (500 steps, comparable to RSVI paper but ~40s).

![ELBO over time](https://github.com/tomblaze/tf_rsvi/blob/master/example_run.png)

Log of run:
```
Epoch    0: total loss is -8.1491e+07 (est elbo: -8.1209e+07) || time elapsed: 2.76 s
Epoch   10: total loss is -1.2515e+07 (est elbo: -1.2606e+07) || time elapsed: 5.73 s
Epoch   20: total loss is -1.2175e+07 (est elbo: -1.2222e+07) || time elapsed: 6.58 s
Epoch   30: total loss is -1.2156e+07 (est elbo: -1.2157e+07) || time elapsed: 7.29 s
Epoch   40: total loss is -1.2082e+07 (est elbo: -1.2027e+07) || time elapsed: 8.00 s
Epoch   50: total loss is -1.1845e+07 (est elbo: -1.1941e+07) || time elapsed: 8.84 s
Epoch   60: total loss is -1.1863e+07 (est elbo: -1.1826e+07) || time elapsed: 9.60 s
Epoch   70: total loss is -1.1986e+07 (est elbo: -1.1686e+07) || time elapsed: 10.41 s
Epoch   80: total loss is -1.1544e+07 (est elbo: -1.1542e+07) || time elapsed: 11.18 s
Epoch   90: total loss is -1.146e+07 (est elbo: -1.1315e+07) || time elapsed: 12.05 s
Epoch  100: total loss is -1.1277e+07 (est elbo: -1.1153e+07) || time elapsed: 12.79 s
Epoch  110: total loss is -1.0845e+07 (est elbo: -1.149e+07) || time elapsed: 13.59 s
Epoch  120: total loss is -1.0608e+07 (est elbo: -1.0645e+07) || time elapsed: 14.31 s
Epoch  130: total loss is -1.0354e+07 (est elbo: -1.0461e+07) || time elapsed: 15.14 s
Epoch  140: total loss is -1.0165e+07 (est elbo: -1.014e+07) || time elapsed: 15.84 s
Epoch  150: total loss is -9.96e+06 (est elbo: -9.9365e+06) || time elapsed: 16.60 s
Epoch  160: total loss is -9.7812e+06 (est elbo: -9.7667e+06) || time elapsed: 17.35 s
Epoch  170: total loss is -9.7513e+06 (est elbo: -9.59e+06) || time elapsed: 18.08 s
Epoch  180: total loss is -9.6784e+06 (est elbo: -9.4965e+06) || time elapsed: 18.90 s
Epoch  190: total loss is -9.3203e+06 (est elbo: -9.3774e+06) || time elapsed: 19.62 s
Epoch  200: total loss is -1.0801e+07 (est elbo: -9.2103e+06) || time elapsed: 20.41 s
Epoch  210: total loss is -9.1732e+06 (est elbo: -9.228e+06) || time elapsed: 21.13 s
Epoch  220: total loss is -9.3691e+06 (est elbo: -8.9939e+06) || time elapsed: 21.86 s
Epoch  230: total loss is -8.9279e+06 (est elbo: -8.9798e+06) || time elapsed: 22.73 s
Epoch  240: total loss is -8.9785e+06 (est elbo: -8.8155e+06) || time elapsed: 23.48 s
Epoch  250: total loss is -8.7993e+06 (est elbo: -8.8227e+06) || time elapsed: 24.28 s
Epoch  260: total loss is -8.8544e+06 (est elbo: -8.782e+06) || time elapsed: 24.99 s
Epoch  270: total loss is -8.721e+06 (est elbo: -8.6788e+06) || time elapsed: 25.86 s
Epoch  280: total loss is -8.5955e+06 (est elbo: -8.855e+06) || time elapsed: 26.59 s
Epoch  290: total loss is -8.4728e+06 (est elbo: -8.7095e+06) || time elapsed: 27.31 s
Epoch  300: total loss is -8.5743e+06 (est elbo: -8.5585e+06) || time elapsed: 28.11 s
Epoch  310: total loss is -1.011e+07 (est elbo: -8.4337e+06) || time elapsed: 28.83 s
Epoch  320: total loss is -8.3775e+06 (est elbo: -8.382e+06) || time elapsed: 29.67 s
Epoch  330: total loss is -8.4295e+06 (est elbo: -8.3574e+06) || time elapsed: 30.39 s
Epoch  340: total loss is -8.3622e+06 (est elbo: -8.4462e+06) || time elapsed: 31.15 s
Epoch  350: total loss is -8.4458e+06 (est elbo: -8.2217e+06) || time elapsed: 31.90 s
Epoch  360: total loss is -8.3984e+06 (est elbo: -8.2326e+06) || time elapsed: 32.62 s
Epoch  370: total loss is -8.1848e+06 (est elbo: -8.2528e+06) || time elapsed: 33.46 s
Epoch  380: total loss is -8.2158e+06 (est elbo: -9.9144e+06) || time elapsed: 34.19 s
Epoch  390: total loss is -8.1491e+06 (est elbo: -8.1303e+06) || time elapsed: 35.00 s
Epoch  400: total loss is -8.0659e+06 (est elbo: -1.1607e+07) || time elapsed: 35.70 s
Epoch  410: total loss is -8.1119e+06 (est elbo: -8.0722e+06) || time elapsed: 36.49 s
Epoch  420: total loss is -8.3182e+06 (est elbo: -7.9988e+06) || time elapsed: 37.29 s
Epoch  430: total loss is -8.3239e+06 (est elbo: -8.0499e+06) || time elapsed: 38.00 s
Epoch  440: total loss is -7.9886e+06 (est elbo: -8.0195e+06) || time elapsed: 38.79 s
Epoch  450: total loss is -7.9826e+06 (est elbo: -8.3389e+06) || time elapsed: 39.54 s
Epoch  460: total loss is -8.0219e+06 (est elbo: -7.995e+06) || time elapsed: 40.41 s
Epoch  470: total loss is -7.9532e+06 (est elbo: -8.1381e+06) || time elapsed: 41.14 s
Epoch  480: total loss is -7.974e+06 (est elbo: -7.9532e+06) || time elapsed: 41.95 s
Epoch  490: total loss is -8.9106e+06 (est elbo: -8.0006e+06) || time elapsed: 42.68 s

```
