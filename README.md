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

treat_mat <- stag_adapt(M = Y, N_t = (N/2+1), T0= T/2, treat_indices=seq(N/2, N, 1)) # 0s are treated

Y_obs <- Y * treat_mat

# Estimate weights by matrix completion

est_weights <- mcnnm_wc_cv(M = treat_mat, C=X, mask = matrix(1, nrow(treat_mat), ncol(treat_mat)), W = matrix(1, nrow(treat_mat), ncol(treat_mat)), 
	to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 5, num_lam_B = 5, niter = 100, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0) # no missing values

W <- est_weights$L + X%*%replicate(T,as.vector(est_weights$B)) + replicate(T,est_weights$u) + t(replicate(N,est_weights$v))

W[W<=0] <- min(W[W>0]) # set floor
W[W>=1] <- max(W[W<1]) # set ceiling

weights <- (1-treat_mat) + (treat_mat)*((1-W)/(W))  # weight adjustment (treated are 0)

# Model with covariates
est_model_MCPanel_w <- mcnnm_wc_cv(M = Y_obs, C = X, mask = treat_mat, W = weights, to_normalize = 1, to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 5, num_lam_B = 5, niter = 100, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

est_model_MCPanel_w$Mhat <- est_model_MCPanel_w$L + X%*%replicate(T,as.vector(est_model_MCPanel_w$B)) + replicate(T,est_model_MCPanel_w$u) + t(replicate(N,est_model_MCPanel_w$v))
est_model_MCPanel_w$msk_err <- (Y-est_model_MCPanel_w$Mhat)*(1-treat_mat)
est_model_MCPanel_w$att<- (1/sum(1-treat_mat)) * sum(est_model_MCPanel_w$msk_err)
est_model_MCPanel_w$att
```