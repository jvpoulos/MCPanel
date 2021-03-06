% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{mcnnm_lam_range}
\alias{mcnnm_lam_range}
\title{This function computes the minimum value of lambda_L which causes L to be zero. 
User can get the output of this function and fit the low-rank model with her desired value of lambda_L.}
\usage{
mcnnm_lam_range(M, mask, W, to_estimate_u = 1L, to_estimate_v = 1L,
  niter = 1000L, rel_tol = 1e-05)
}
\arguments{
\item{M}{Matrix of observed entries. The input should be N (number of units) by T (number of time periods).}

\item{mask}{Binary mask with the same shape as M containing observed entries.}

\item{W}{Matrix of weights. The input should be N (number of units) by T (number of time periods).}

\item{to_estimate_u}{Optional boolean input for wheter estimating fixed unit effects (row means of M) or not. Default is 1.}

\item{niter}{Optional parameter on the number of iterations taken in the algorithm for each fixed value of lambda_L. The default value is 1000 and it is sufficiently large as the algorithm is using warm-start strategy.}

\item{rel_tol}{Optional parameter on the stopping rule. Once the relative improve in objective value drops below rel_tol, execution is halted. Default value is 1e-5.}

\item{to_estimate_u}{Optional boolean input for wheter estimating fixed time effects (column means of M) or not. Default is 1.}
}
\value{
The minimum value of lambda_L causing L to be zero.
}
\description{
This function computes the minimum value of lambda_L which causes L to be zero. 
User can get the output of this function and fit the low-rank model with her desired value of lambda_L.
}
\examples{
mcnnm_lam_range(M = replicate(5,rnorm(5)), mask = matrix(rbinom(5*5,1,0.8),5,5), W = matrix(runif(5*5,0,1),5,5))
}
