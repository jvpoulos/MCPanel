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

T <- 5 # No. time periods
N <- 5 # No. units

Y <- replicate(T,rnorm(N)) # simulated observed outcomes

X <- replicate(T,rnorm(N)) # simulated covariates

treat_mat <- simul_adapt(M = Y, N_t = 2, T0= 3, treat_indices=c(4,5))

Y_obs <- Y * treat_mat

W <- rbind(matrix(runif(3*T,0,0.5),3,T),
		   matrix(runif(2*T,0.5,1),2,T)) # simulated unit-specific propensity score
 
weights <- matrix(NA, N, T) # transform weights for regression
weights[c(4,5),] <- 1/(W[c(4,5),]) # treated group
weights[-c(4,5),] <- 1/(1-W[-c(4,5),]) # control group

# Model without covariates

est_model_MCPanel <- mcnnm_cv(M = Y_obs, mask = treat_mat, W = weights, 
	to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

est_model_MCPanel$Mhat <- est_model_MCPanel$L + replicate(T,est_model_MCPanel$u) + t(replicate(N,est_model_MCPanel$v))

# Model with covariates
est_model_MCPanel_w <- mcnnm_wc_cv(M = Y_obs, C = X, mask = treat_mat, W = weights, 
	to_normalize = 1, to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, num_lam_B = 10, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

est_model_MCPanel_w$Mhat <- est_model_MCPanel_w$L + X%*%replicate(T,as.vector(est_model_MCPanel_w$B)) + replicate(T,est_model_MCPanel_w$u) + t(replicate(N,est_model_MCPanel_w$v))
```