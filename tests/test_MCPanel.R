library(MCPanel)
library(matrixStats)

library(ggplot2)
library(latex2exp)
library(dplyr)

# Setup parallel processing
library(parallel)
library(doParallel)

cores <- detectCores()

cl <- parallel::makeForkCluster(cores)

doParallel::registerDoParallel(cores) # register cores (<p)

RNGkind("L'Ecuyer-CMRG") # ensure random number generation

RunMCtest <- function(N,T,R,noise_sc,delta_sc,gamma_sc,fr_obs){
  ## Create Matrices
  A <- replicate(R,rnorm(N))
  B <- replicate(T,rnorm(R))
  X <- replicate(T,rnorm(N))
  delta <- delta_sc*rnorm(N)
  gamma <- gamma_sc*rnorm(T)
  beta <- beta_sc*rnorm(T)
  noise <- noise_sc*replicate(T,rnorm(N))
  true_mat <- A %*% B + X%*%replicate(T,as.vector(beta)) + replicate(T,delta) + t(replicate(N,gamma))
  noisy_mat <- true_mat + noise
  mask <- matrix(rbinom(N*T,1,fr_obs),N,T)
  obs_mat <- noisy_mat * mask

  # Estimate weights by matrix completion

  est_weights <- mcnnm_cv(M = mask, mask = matrix(1, nrow(mask), ncol(mask)), W = matrix(0.5, nrow(mask), ncol(mask)),
                          to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 0)

  W <- est_weights$L + replicate(T,est_weights$u) + t(replicate(N,est_weights$v))

  weights <- (1-mask) + (mask)*W/(1-W) # weighting by the odds

  ## Estimate using mcnnm_cv (cross-validation) on lambda values
  model_without_effects <- mcnnm_wc_cv(obs_mat, X, mask, weights, to_normalize = 1, to_estimate_u = 0, to_estimate_v = 0, num_lam_L = 10, num_lam_B = 5, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 1)
  model_with_both <- mcnnm_wc_cv(obs_mat, X, mask, weights, to_normalize = 1, to_estimate_u = 1, to_estimate_v = 1, num_lam_L = 10, num_lam_B = 5, niter = 1000, rel_tol = 1e-05, cv_ratio = 0.8, num_folds = 2, is_quiet = 1)

  ## Check criteria
  sum(model_without_effects$u == 0) == N ## Checking if row-wise effects are zero
  sum(model_without_effects$v == 0) == T ## Checking if column-wise effects are zero
  #
  sum(model_with_both$u == 0) == N
  sum(model_with_both$v == 0) == T
  #

  ## Comparing minimum RMSEs

  model_without_effects$min_RMSE
  model_with_both$min_RMSE

  ## Construct estimations based on models

  model_without_effects$est <- model_without_effects$L + X%*%replicate(T,as.vector(model_without_effects$B))
  model_with_both$est <- model_with_both$L + X%*%replicate(T,as.vector(model_with_both$B)) + replicate(T,model_with_both$u) + t(replicate(N,model_with_both$v))

  ## Compute error matrices

  model_without_effects$err <- model_without_effects$est - true_mat
  model_with_both$err <- model_with_both$est - true_mat

  ## Compute masked error matrices

  model_without_effects$msk_err <- model_without_effects$err*(1-mask)
  model_with_both$msk_err <- model_with_both$err*(1-mask)

  ## Compute RMSE on test set

  model_without_effects$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_without_effects$msk_err^2))
  model_with_both$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_with_both$msk_err^2))

  return(list("no_effects_error"=model_without_effects$test_RMSE, "both_error"=model_with_both$test_RMSE))
}

N <- 50 # Number of units
T <- 50 # Number of time-periods
#R <- 2 # Rank of matrix
R.set <- c(1,5,10,15,20)
noise_sc <- 0.1 # Noise scale
delta_sc <- 0.1 # delta scale
gamma_sc <- 0.1 # gamma scale
beta_sc <- 0.1 # beta scale
fr_obs <- 0.8 # fraction of observed entries
n <- 100 # Num. simulation runs

results <- foreach(R = R.set, .combine='rbind') %dopar% {
  replicate(n,RunMCtest(N,T,R,noise_sc,delta_sc,gamma_sc,fr_obs))
}
results <- matrix(unlist(results), ncol = n, byrow = FALSE) # coerce into matrix
saveRDS(results, "rank-test.rds")

model.names <- c("Without fixed effects ", "With fixed effects")
sim.data <- data.frame("mean"=rowMeans(results),
                       "sd"=rowSds(results),
                       "Model"=rep(model.names, length(R.set)),
                       "R"=rep(R.set,each=length(model.names)))
# Plot

MCtest <- ggplot(data = sim.data, aes(R, mean, color = Model, shape =Model)) +
  geom_point(size = 5, position=position_dodge(width=0.3)) +
  geom_line(position=position_dodge(width=0.3)) +
  geom_errorbar(
    aes(ymin = mean-1.96*sd, ymax = mean+1.96*sd),
    width = 0.1,
    linetype = "solid",
    position=position_dodge(width=0.3)) +
  scale_shape_manual(values=c(1:length(model.names))) +
  theme_bw() +
  xlab(TeX('$Rank(L^{*})$')) +
  ylab("Average RMSE") +
  theme(axis.title=element_text(family="serif", size=16)) +
  theme(axis.text=element_text(family="serif", size=14)) +
  theme(legend.text=element_text(family="serif", size = 12)) +
  theme(legend.title=element_text(family="serif", size = 12)) +
  theme(axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l =0))) +
  theme(axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l =0))) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))  # rm background

ggsave("plots/MCtest.png", MCtest, width=8.5, height=11)
