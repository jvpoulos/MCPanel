% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{mcnnm_wc_fit}
\alias{mcnnm_wc_fit}
\title{This function trains a model for specified values of lambda_L and lambda_B. The result contains L, B, u, v, lambda_L, and lambda_B. 
The only difference with mcnnm_wc is that this function trains only for specified pair of values (lambda_L, lambda_B).}
\usage{
mcnnm_wc_fit(M, C, mask, W, lambda_L, lambda_B, to_normalize = 1L,
  to_estimate_u = 1L, to_estimate_v = 1L, niter = 100L,
  rel_tol = 1e-05, is_quiet = 1L)
}
\arguments{
\item{M}{Matrix of observed entries. The input should be N (number of units) by T (number of time periods).}

\item{C}{Matrix containing time -and unit-related covariates. The dimensions of C should match those of M. If time- and unit-related covariates do not exist C = matrix(0L,0,0,0,0) should be used as input.}

\item{mask}{Binary mask with the same shape as M containing observed entries.}

\item{W}{Matrix of weights. The input should be N (number of units) by T (number of time periods).}

\item{lambda_L}{Required parameter for fitting the model as this function computes the result for a specified value of lambda_L.}

\item{lambda_B}{Required parameter for fitting the model as this function computes the result for a specified value of lambda_B.}

\item{to_normalize}{Optional boolean parameter indicating whether to normalize covariates or not (columns of C). The default value is 1. If this value is set to 0, the result would be sensitive to scales in covariates.}

\item{to_estimate_u}{Optional boolean input for wheter estimating fixed unit effects (row means of M) or not. Default is 1.}

\item{niter}{Optional parameter on the number of iterations taken in the algorithm for each fixed value of lambda_L. The default value is 1000 and it is sufficiently large as the algorithm is using warm-start strategy.}

\item{rel_tol}{Optional parameter on the stopping rule. Once the relative improve in objective value drops below rel_tol, execution is halted. Default value is 1e-5.}

\item{is_quiet}{Optional boolean input which indicates whether to print the status of learning and convergence results for Cyclic Coordinate Descent algorithm or not. The default value is 1 (no output is printed).}
}
\value{
The fitted model for the given value of lambda_L and lambda_B. The result contains L, B, u, v, lambda_L, and lambda_B.
}
\description{
This function trains a model for specified values of lambda_L and lambda_B. The result contains L, B, u, v, lambda_L, and lambda_B. 
The only difference with mcnnm_wc is that this function trains only for specified pair of values (lambda_L, lambda_B).
}
\examples{
mcnnm_wc_fit(M = replicate(5,rnorm(5)), C = replicate(5,rnorm(5)), mask = matrix(rbinom(5*5,1,0.8),5,5), W = matrix(runif(5*5,0,1),5,5), lambda_L=0.1, lambda_B=0.1)
}
\seealso{
\code{\link{mcnnm_fit}}, \code{\link{mcnnm_wc}}
}
