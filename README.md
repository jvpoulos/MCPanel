# MCPanel
Matrix Completion Methods for Causal Panel Data Models

The __MCPanel__ package provides functions to fit a low-rank model to a partially observed matrix. 

Prerequsites
------

* **R** 3.6 (tested on 3.6.1)
 * Rcpp, evalCpp, glmnet, latex2exp, ggplot2

To install this package in R, run the following commands:

```R
install.packages("devtools")
install.packages("latex2exp")
library(devtools) 
install_github("jvpoulos/MCPanel")
```

Example usage:

```R
library(MCPanel)
estimated_obj <- mcnnm_cv(M, mask, W, to_estimate_u = 0, to_estimate_v = 0, num_lam_L = 40)
                  
best_lam_L <- estimated_obj$best_lambda
estimated_mat <- estimated_obj$L

```
Note: it may be necessary for Windows R 3.4.2 users to use the patched version of R: https://cran.r-project.org/bin/windows/base/rpatched.html

More details will be added soon.

#### References
Susan Athey, Mohsen Bayati, Nikolay Doudchenko, Guido Imbens, and Khashayar Khosravi. <b>Matrix Completion Methods for Causal Panel Data Models</b> [<a href="http://arxiv.org/abs/1710.10251">link</a>]
