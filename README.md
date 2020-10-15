# MCPanel+

This repo is forked from [MCPanel](https://github.com/susanathey/MCPanel/). The model is described in the paper [Matrix Completion Methods for Causal Panel Data Models](http://arxiv.org/abs/1710.10251) by Athey et al. 

The code in this fork allows for unit-time specific covariates in the form of a *N x T* matrix **C** in lieu of the vectors of unit-specific and time-specific covariates in the parent repo. 

The code also allows for a *N x T* matrix **W** used to weight the loss function, as described in Sec. 8.3 of the Athey et al. paper. 


Prerequsites
------

* **R** >= 3.5.0 (tested on 3.6.1)
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

T <- 50 # No. time periods
N <- 50 # No. units

Y <- replicate(T,rnorm(N)) # simulated observed outcomes

X <- replicate(T,rnorm(N)) # simulated covariates

treat_mat <- stag_adapt(M = Y, N_t = (N/2+1), T0= T/2, treat_indices=seq(N/2, N, 1)) # staggered adoption

Y_obs <- Y * treat_mat

# Estimate weights by matrix completion

est_weights <- mcnnm_cv(M = treat_mat, mask = matrix(1, nrow(treat_mat), ncol(treat_mat)), W = matrix(0.5, nrow(treat_mat), ncol(treat_mat)), 
	to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

W <- est_weights$L + replicate(T,est_weights$u) + t(replicate(N,est_weights$v))

weights <- (1-treat_mat) + (treat_mat)*W/(1-W) # weighting by the odds

# Model without covariates

est_model_MCPanel <- mcnnm_cv(M = Y_obs, mask = treat_mat, W = weights, 
	to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

est_model_MCPanel$Mhat <- est_model_MCPanel$L + replicate(T,est_model_MCPanel$u) + t(replicate(N,est_model_MCPanel$v))

est_model_MCPanel$msk_err <- (est_model_MCPanel$Mhat - Y)*(1-treat_mat)
est_model_MCPanel$test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_MCPanel$msk_err^2, na.rm = TRUE))
est_model_MCPanel$test_RMSE

# Model with covariates
est_model_MCPanel_w <- mcnnm_wc_cv(M = Y_obs, C = X, mask = treat_mat, W = weights, 
	to_normalize = 1, to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, num_lam_B = 5, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

est_model_MCPanel_w$Mhat <- est_model_MCPanel_w$L + X%*%replicate(T,as.vector(est_model_MCPanel_w$B)) + replicate(T,est_model_MCPanel_w$u) + t(replicate(N,est_model_MCPanel_w$v))
est_model_MCPanel_w$msk_err <- (est_model_MCPanel_w$Mhat - Y)*(1-treat_mat)
est_model_MCPanel_w$test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_MCPanel_w$msk_err^2, na.rm = TRUE))
est_model_MCPanel_w$test_RMSE
```