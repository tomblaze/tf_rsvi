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

Poisson samples from model after 500 steps...

![samples](https://github.com/tomblaze/tf_rsvi/blob/master/unconditional_sample.png)

ELBO over course of training (500 steps, comparable to RSVI paper but ~40s).

![ELBO over time](https://github.com/tomblaze/tf_rsvi/blob/master/example_run.png)

Log of run:
```
Epoch    0: total loss is -8.1686e+07 (est elbo: -8.1766e+07) || time elapsed: 2.75 s
Epoch   10: total loss is -1.2606e+07 (est elbo: -1.2428e+07) || time elapsed: 5.36 s
Epoch   20: total loss is -1.2216e+07 (est elbo: -1.22e+07) || time elapsed: 6.06 s
Epoch   30: total loss is -1.1942e+07 (est elbo: -1.2095e+07) || time elapsed: 6.73 s
Epoch   40: total loss is -1.1964e+07 (est elbo: -1.201e+07) || time elapsed: 7.39 s
Epoch   50: total loss is -1.1856e+07 (est elbo: -1.187e+07) || time elapsed: 8.06 s
Epoch   60: total loss is -1.1787e+07 (est elbo: -1.1723e+07) || time elapsed: 8.73 s
Epoch   70: total loss is -1.1678e+07 (est elbo: -1.1724e+07) || time elapsed: 9.38 s
Epoch   80: total loss is -1.1408e+07 (est elbo: -1.1446e+07) || time elapsed: 10.05 s
Epoch   90: total loss is -1.1305e+07 (est elbo: -1.1356e+07) || time elapsed: 10.71 s
Epoch  100: total loss is -1.1055e+07 (est elbo: -1.1067e+07) || time elapsed: 11.39 s
Epoch  110: total loss is -1.0793e+07 (est elbo: -1.0763e+07) || time elapsed: 12.06 s
Epoch  120: total loss is -1.0493e+07 (est elbo: -1.052e+07) || time elapsed: 12.71 s
Epoch  130: total loss is -1.0261e+07 (est elbo: -1.0274e+07) || time elapsed: 13.39 s
Epoch  140: total loss is -1.0007e+07 (est elbo: -1.0036e+07) || time elapsed: 14.07 s
Epoch  150: total loss is -9.8497e+06 (est elbo: -9.8733e+06) || time elapsed: 14.73 s
Epoch  160: total loss is -9.6353e+06 (est elbo: -9.6608e+06) || time elapsed: 15.42 s
Epoch  170: total loss is -9.4338e+06 (est elbo: -9.5857e+06) || time elapsed: 16.10 s
Epoch  180: total loss is -9.3352e+06 (est elbo: -9.3854e+06) || time elapsed: 16.76 s
Epoch  190: total loss is -9.2268e+06 (est elbo: -9.2198e+06) || time elapsed: 17.45 s
Epoch  200: total loss is -9.0963e+06 (est elbo: -9.1414e+06) || time elapsed: 18.13 s
Epoch  210: total loss is -9.0205e+06 (est elbo: -8.9437e+06) || time elapsed: 18.78 s
Epoch  220: total loss is -8.8646e+06 (est elbo: -8.8987e+06) || time elapsed: 19.44 s
Epoch  230: total loss is -8.7613e+06 (est elbo: -8.7465e+06) || time elapsed: 20.12 s
Epoch  240: total loss is -8.6495e+06 (est elbo: -8.6615e+06) || time elapsed: 20.78 s
Epoch  250: total loss is -8.6095e+06 (est elbo: -8.6393e+06) || time elapsed: 21.45 s
Epoch  260: total loss is -8.5225e+06 (est elbo: -8.5681e+06) || time elapsed: 22.12 s
Epoch  270: total loss is -8.537e+06 (est elbo: -8.4787e+06) || time elapsed: 22.78 s
Epoch  280: total loss is -8.4092e+06 (est elbo: -8.42e+06) || time elapsed: 23.45 s
Epoch  290: total loss is -8.3269e+06 (est elbo: -8.3354e+06) || time elapsed: 24.11 s
Epoch  300: total loss is -8.2921e+06 (est elbo: -8.2745e+06) || time elapsed: 24.78 s
Epoch  310: total loss is -8.2316e+06 (est elbo: -8.183e+06) || time elapsed: 25.45 s
Epoch  320: total loss is -8.2003e+06 (est elbo: -8.1439e+06) || time elapsed: 26.10 s
Epoch  330: total loss is -8.1047e+06 (est elbo: -8.1114e+06) || time elapsed: 26.78 s
Epoch  340: total loss is -8.1485e+06 (est elbo: -8.0644e+06) || time elapsed: 27.45 s
Epoch  350: total loss is -7.9622e+06 (est elbo: -8.0318e+06) || time elapsed: 28.12 s
Epoch  360: total loss is -7.9918e+06 (est elbo: -7.9756e+06) || time elapsed: 28.80 s
Epoch  370: total loss is -7.9623e+06 (est elbo: -7.9398e+06) || time elapsed: 29.46 s
Epoch  380: total loss is -7.9109e+06 (est elbo: -7.9497e+06) || time elapsed: 30.12 s
Epoch  390: total loss is -7.8882e+06 (est elbo: -7.8894e+06) || time elapsed: 30.82 s
Epoch  400: total loss is -7.8905e+06 (est elbo: -7.8856e+06) || time elapsed: 31.50 s
Epoch  410: total loss is -7.7921e+06 (est elbo: -7.8335e+06) || time elapsed: 32.16 s
Epoch  420: total loss is -7.8271e+06 (est elbo: -7.8505e+06) || time elapsed: 32.82 s
Epoch  430: total loss is -7.8309e+06 (est elbo: -7.7842e+06) || time elapsed: 33.49 s
Epoch  440: total loss is -7.7404e+06 (est elbo: -7.7589e+06) || time elapsed: 34.14 s
Epoch  450: total loss is -7.7268e+06 (est elbo: -7.7499e+06) || time elapsed: 34.81 s
Epoch  460: total loss is -7.662e+06 (est elbo: -7.6812e+06) || time elapsed: 35.49 s
Epoch  470: total loss is -7.6888e+06 (est elbo: -7.6831e+06) || time elapsed: 36.14 s
Epoch  480: total loss is -7.6736e+06 (est elbo: -7.6294e+06) || time elapsed: 36.84 s
Epoch  490: total loss is -7.6456e+06 (est elbo: -7.6613e+06) || time elapsed: 37.53 s

```
