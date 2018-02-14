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
