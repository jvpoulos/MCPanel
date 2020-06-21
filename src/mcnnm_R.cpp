#include <iostream>
#include <cmath>
#include <Rcpp.h>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include "Eigen/Core"
#include "Eigen/Sparse"
#include <RcppEigen.h>
#include <random>
#include <stdlib.h>

using namespace Eigen;
using namespace Rcpp;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////// Core Functions : All functions that have _B in the very end of their name,
////////                  consider the case where covariates exist.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
List MySVD(NumericMatrix M){

  // This function computes the Singular Value Decomposition and it passes U,V,Sigma.
  // As SVD is one of the most time consuming part of our algorithm, this function is created to effectivelly
  // compute and compare different algorithms' speeds.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));

  JacobiSVD<MatrixXd> svd( M_.rows(), M_.cols(), ComputeThinV | ComputeThinU );
  svd.compute(M_);
  VectorXd sing = svd.singularValues();
  MatrixXd U = svd.matrixU();
  MatrixXd V = svd.matrixV();
  return List::create(Named("U") = U,
                      Named("V") = V,
                      Named("Sigma") = sing);

}

NumericVector logsp(double start_log, double end_log, int num_points){

  // This function creates logarithmically spaced numbers.

  NumericVector res(num_points);
  if(num_points == 1){
    res[0]=end_log;
    }
  else{
    double step_size = (end_log-start_log)/(num_points-1);
    for (int i = 0; i<num_points; i++){
      res[i]=pow(10.0,start_log+i*step_size);
     }
  }
  return res;
}

NumericMatrix ComputeMatrix(NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes L + u1^T + 1v^T, which is our decomposition.

  using Eigen::Map;
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  int num_rows = L_.rows();
  int num_cols = L_.cols();
  MatrixXd res_ = u_ * VectorXd::Constant(num_cols,1).transpose() + VectorXd::Constant(num_rows,1) * v_.transpose() + L_;
  return wrap(res_);
}

NumericMatrix ComputeMatrix_B(NumericMatrix L, NumericMatrix C, NumericMatrix B, NumericVector u, NumericVector v){

  // This function computes L + B*C + u1^T + 1v^T, which is our decomposition.

  using Eigen::Map;
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  const Map<MatrixXd> B_(as<Map<MatrixXd> >(B));

  int num_rows = L_.rows();
  int num_cols = L_.cols();

  MatrixXd res_ = L_ + C_ * B_ + u_ * VectorXd::Constant(num_cols,1).transpose() + VectorXd::Constant(num_rows,1) * v_.transpose();

  return wrap(res_);
}

double Compute_objval(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericMatrix W, NumericVector u, NumericVector v, double sum_sing_vals, double lambda_L){

  // This function computes our objective value which is decomposed as the weighted combination of error plus nuclear norm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));

  int train_size = mask_.sum();

  NumericMatrix est_mat = ComputeMatrix(L,u,v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));

  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = (err_mat_.array()) * mask_.array();

  MatrixXd w_mask_ = (W_.array()) * mask_.array();

  double obj_val = (double(1)/train_size) * w_mask_.sum() * (err_mask_.cwiseProduct(err_mask_)).sum() + lambda_L * sum_sing_vals;
  return obj_val;
}

double Compute_objval_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericMatrix W, NumericVector u, NumericVector v, double sum_sing_vals, double lambda_L, double lambda_B){

  // This function computes our objective value which is decomposed as the weighted combination of error plus nuclear norm of L
  // and also element-wise norm 1 of B.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  const Map<MatrixXd> B_(as<Map<MatrixXd> >(B));

  int train_size = mask_.sum();

  double norm_B = B_.array().abs().sum();

  NumericMatrix est_mat = ComputeMatrix_B(L, C, B, u, v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));

  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = (err_mat_.array()) * mask_.array();

  MatrixXd w_mask_ = (W_.array()) * mask_.array();

  double obj_val = (double(1)/train_size) * w_mask_.sum() * (err_mask_.cwiseProduct(err_mask_)).sum() + lambda_L * sum_sing_vals + lambda_B*norm_B;
  return obj_val;
}

double Compute_RMSE(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes Root Mean Squared Error of computed decomposition L,u,v.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  double res = 0;
  int valid_size = mask_.sum();
  NumericMatrix est_mat = ComputeMatrix(L,u,v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));
  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = err_mat_.array() * mask_.array();
  res = std::sqrt((double(1.0)/valid_size) * (err_mask_.cwiseProduct(err_mask_)).sum());
  return res;
}

double Compute_RMSE_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes Root Mean Squared Error of computed decomposition L, B, u, v.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  double res = 0;
  int valid_size = mask_.sum();
  NumericMatrix est_mat = ComputeMatrix_B(L, C, B, u, v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));
  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = err_mat_.array() * mask_.array();
  res = std::sqrt((double(1.0)/valid_size) * (err_mask_.cwiseProduct(err_mask_)).sum());
  return res;
}

NumericMatrix SVT(NumericMatrix U, NumericMatrix V, NumericVector &sing_values, double sigma){

  // Given a singular value decomposition and a threshold sigma, this function computes Singular Value Thresholding operator.
  // Furthermore, it updates the singular values with the truncated version (new singular values of L) which would
  // then be used to compute objective value.

  using Eigen::Map;
  const Map<MatrixXd> U_(as<Map<MatrixXd> >(U));
  const Map<MatrixXd> V_(as<Map<MatrixXd> >(V));
  const Map<VectorXd> sing_values_(as<Map<VectorXd> >(sing_values));

  VectorXd trunc_sing = sing_values_ - VectorXd::Constant(sing_values_.size(),sigma);
  trunc_sing = trunc_sing.cwiseMax(0);
  MatrixXd Cp_ = U_ * trunc_sing.asDiagonal() * V_.transpose();
  sing_values = trunc_sing;
  return wrap(Cp_);
}

List update_L(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_L){

  // This function updates L in coordinate descent algorithm. The core step of this part is
  // performing a SVT update. Furthermore, it saves the singular values (needed to compute objective value) later.
  // This would help us to only perform one SVD per iteration.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));

  int train_size = mask_.sum();
  NumericMatrix H = ComputeMatrix(L,u,v);
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  MatrixXd P_omega_ = M_ - H_;
  MatrixXd masked_P_omega_ = P_omega_.cwiseProduct(mask_);
  MatrixXd proj_ = masked_P_omega_ + L_;
  NumericMatrix proj = wrap(proj_);
  List svd_dec = MySVD(proj);
  MatrixXd U_ = svd_dec["U"];
  MatrixXd V_ = svd_dec["V"];
  VectorXd sing_ = svd_dec["Sigma"];
  NumericMatrix U = wrap(U_);
  NumericMatrix V = wrap(V_);
  NumericVector sing = wrap(sing_);
  NumericMatrix L_upd = SVT(U, V, sing, lambda_L*train_size/2 );
  //L = SVT(U,V,sing,lambda_L/2);
  return List::create(Named("L") = L_upd,
                      Named("Sigma") = sing);
}

List update_L_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_L){

  // This function updates L in coordinate descent algorithm. The core step of this part is
  // performing a SVT update. Furthermore, it saves the singular values (needed to compute objective value) later.
  // This would help us to only perform one SVD per iteration. Note that this function includes covariates as well.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));

  int train_size = mask_.sum();
  NumericMatrix P = ComputeMatrix_B(L, C, B, u, v);
  const Map<MatrixXd> P_(as<Map<MatrixXd> >(P));
  MatrixXd P_omega_ = M_ - P_;
  MatrixXd masked_P_omega_ = P_omega_.cwiseProduct(mask_);
  MatrixXd proj_ = masked_P_omega_ + L_;
  NumericMatrix proj = wrap(proj_);
  List svd_dec = MySVD(proj);
  MatrixXd U_ = svd_dec["U"];
  MatrixXd V_ = svd_dec["V"];
  VectorXd sing_ = svd_dec["Sigma"];
  NumericMatrix U = wrap(U_);
  NumericMatrix V = wrap(V_);
  NumericVector sing = wrap(sing_);
  NumericMatrix L_upd = SVT(U, V, sing, lambda_L*train_size/2 );
  //L = SVT(U,V,sing,lambda_L/2);
  return List::create(Named("L") = L_upd,
                      Named("Sigma") = sing);
}

NumericVector update_u(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector v){

  // This function updates u in coordinate descent algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  VectorXd res(M_.rows(),1);
  for (int i = 0; i<M_.rows(); i++){
    VectorXd b_ = L_.row(i)+v_.transpose()-M_.row(i);
    VectorXd h_ = mask_.row(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_u_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericVector v){

  // This function updates u in coordinate descent algorithm, when covariates are available.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  const Map<MatrixXd> B_(as<Map<MatrixXd> >(B));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  MatrixXd T_ = C_ * B_;

  VectorXd res(M_.rows(),1);
  for (int i = 0; i<M_.rows(); i++){
    VectorXd b_ = T_.row(i)+L_.row(i)+v_.transpose()-M_.row(i);
    VectorXd h_ = mask_.row(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_v(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u){

  // This function updates the matrix v in the coordinate descent algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  VectorXd res(M_.cols(),1);
  for (int i = 0; i<M_.cols(); i++)
  {
    VectorXd b_ = L_.col(i)+u_-M_.col(i);
    VectorXd h_ = mask_.col(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_v_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericVector u){

  // This function updates the matrix v in the coordinate descent algorithm, when covariates exist.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  const Map<MatrixXd> B_(as<Map<MatrixXd> >(B));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  MatrixXd T_ = C_ * B_;

  VectorXd res(M_.cols(),1);
  for (int i = 0; i<M_.cols(); i++)
  {
    VectorXd b_ = T_.col(i)+L_.col(i)+u_-M_.col(i);
    VectorXd h_ = mask_.col(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector Reshape_Mat(NumericMatrix M){
  // This function reshapes a matrix into a vector.
  NumericVector res(M.rows()*M.cols());
  for(int j=0; j<M.cols(); j++){
    for(int i=0; i<M.rows(); i++){
      res(j*M.rows()+i) = M(i,j);
    }
  }
  return wrap(res);
}

NumericMatrix Reshape(NumericVector M, int row, int col){
  // This function reshapes a given vector, into a matrix with given rows and columns.
  NumericMatrix res(row,col);
  for(int j=0; j<col; j++){
    for(int i=0; i<row; i++){
      res(i,j)=M(j*row+i);
    }
  }
  return wrap(res);
}

NumericMatrix update_B_B(NumericMatrix M, NumericMatrix C, NumericMatrix B, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_B){

  // This function updates the matrix B in the coordinate descent algorithm. The core step of this part is
  // performing a SVT update.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> B_(as<Map<MatrixXd> >(B));

  int train_size = mask_.sum();
  NumericMatrix P = ComputeMatrix_B(L, C, B, u, v);
  const Map<MatrixXd> P_(as<Map<MatrixXd> >(P));
  MatrixXd P_omega_ = M_ - P_;
  MatrixXd masked_P_omega_ = P_omega_.cwiseProduct(mask_);
  MatrixXd proj_ = masked_P_omega_ + B_;
  NumericMatrix proj = wrap(proj_);
  List svd_dec = MySVD(proj);
  MatrixXd U_ = svd_dec["U"];
  MatrixXd V_ = svd_dec["V"];
  VectorXd sing_ = svd_dec["Sigma"];
  NumericMatrix U = wrap(U_);
  NumericMatrix V = wrap(V_);
  NumericVector sing = wrap(sing_);
  NumericMatrix B_upd = SVT(U, V, sing, lambda_B*train_size/2 );
  //L = SVT(U,V,sing,lambda_L/2);
  return wrap(B_upd);
}

List initialize_uv(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int niter = 1000, double rel_tol = 1e-5){

  // This function solves finds the optimal u and v assuming that L is zero. This would be later
  // helpful when we want to perform warm start on values of lambda_L. This function also outputs
  // the smallest value of lambda_L which causes L to be zero (all singular values vanish after a SVT update)

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));

  JacobiSVD<MatrixXd> svd(M_.rows(), M_.cols());
  double obj_val=0;
  double new_obj_val=0;
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  VectorXd u_ = VectorXd::Zero(num_rows);
  VectorXd v_ = VectorXd::Zero(num_cols);
  MatrixXd L_ = MatrixXd::Zero(num_rows,num_cols);
  NumericVector u = wrap(u_);
  NumericVector v = wrap(v_);
  NumericMatrix L = wrap(L_);
  obj_val = Compute_objval(M, mask, L, W, u , v, 0, 0);
  for(int iter = 0; iter < niter; iter++){
    if(to_estimate_u == 1){
      u = update_u(M, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(num_rows));
    }
    if(to_estimate_v == 1){
      v = update_v(M, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(num_cols));
    }
    new_obj_val = Compute_objval(M, mask, L, W, u, v, 0, 0);
    double rel_error = (new_obj_val-obj_val)/obj_val;
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    obj_val = new_obj_val;
  }
  NumericMatrix E = ComputeMatrix(L, u, v);
  const Map<MatrixXd> E_(as<Map<MatrixXd> >(E));
  MatrixXd P_omega_ = (M_ - E_).array()*mask_.array();

  svd.compute(P_omega_);
  double lambda_L_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

  return List::create(Named("u") = u,
                      Named("v") = v,
                      Named("lambda_L_max") = lambda_L_max);
  }

List initialize_uv_B(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int niter = 1000, double rel_tol = 1e-5){

    // This function solves finds the optimal u and v assuming that L and B are zero. This would be later
    // helpful when we want to perform warm start on values of lambda_L and  lambda_B. This function also outputs
    // the smallest value of lambda_L and labmda_B which causes L and B to be zero.

    using Eigen::Map;
    const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
    const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
    const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
    int num_rows = M_.rows();
    int num_cols = M_.cols();

    JacobiSVD<MatrixXd> svd(M_.rows(), M_.cols());
    double obj_val=0;
    double new_obj_val=0;
    VectorXd u_ = VectorXd::Zero(num_rows);
    VectorXd v_ = VectorXd::Zero(num_cols);
    MatrixXd L_ = MatrixXd::Zero(num_rows,num_cols);
    MatrixXd B_ = MatrixXd::Zero(num_rows,num_cols);
    NumericVector u = wrap(u_);
    NumericVector v = wrap(v_);
    NumericMatrix L = wrap(L_);
    NumericMatrix B = wrap(B_);
    obj_val = Compute_objval_B(M, C, B, mask, L, W, u , v, 0, 0, 0);
    for(int iter = 0; iter < niter; iter++){
      if(to_estimate_u == 1){
        u = update_u_B(M, C, B, mask, L, v);
      }
      else{
        u = wrap(VectorXd::Zero(num_rows));
      }
      if(to_estimate_v == 1){
        v = update_v_B(M, C, B, mask, L, u);
      }
      else{
        v = wrap(VectorXd::Zero(num_rows));
      }
      new_obj_val = Compute_objval_B(M, C, B, mask, L, W, u, v, 0, 0, 0);
      double rel_error = (new_obj_val-obj_val)/obj_val;
      if(rel_error < rel_tol && rel_error >= 0){
        break;
      }
      obj_val = new_obj_val;
    }
    NumericMatrix E = ComputeMatrix_B(L, C, B, u, v);
    const Map<MatrixXd> E_(as<Map<MatrixXd> >(E));
    MatrixXd P_omega_ = (M_ - E_).array()*mask_.array();
    svd.compute(P_omega_);
    double lambda_L_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

    double lambda_B_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

    return List::create(Named("u") = u,
                        Named("v") = v,
                        Named("lambda_L_max") = lambda_L_max,
                        Named("lambda_B_max") = lambda_B_max);
}

List create_folds(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds=5){

  // This function creates folds for cross-validation. Each fold contains a training and validation sets.
  // For each of these folds the initial solutions for fixed effects are then computed, as for large lambda_L
  // L would be equal to zero. This initialization is very helpful as it will be used later for the warm start.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List out(num_folds);
  std::default_random_engine generator;
  std::bernoulli_distribution distribution(cv_ratio);
  for (int k = 0; k<num_folds; k++){
    MatrixXd ma_new(num_rows,num_cols);
    for (int i = 0; i < num_rows; i++){
      for (int j = 0; j < num_cols; j++){
        ma_new(i,j)=distribution(generator);
      }
    }
    MatrixXd fold_mask_ = mask_.array() * ma_new.array();
    MatrixXd M_tr_ = M_.array() * fold_mask_.array();
    NumericMatrix fold_mask = wrap(fold_mask_);
    MatrixXd W_tr_ = W_.array() * fold_mask_.array();
    NumericMatrix M_tr = wrap(M_tr_);
    NumericMatrix W_tr = wrap(W_tr_);
    List tmp_uv = initialize_uv(M_tr, fold_mask, W_tr, to_estimate_u, to_estimate_v, niter = 1000, rel_tol = 1e-5);
    List fold_k = List::create(Named("u") = tmp_uv["u"],
                               Named("v") = tmp_uv["v"],
                               Named("lambda_L_max") = tmp_uv["lambda_L_max"],
                               Named("fold_mask") = fold_mask);
    out[k] = fold_k;
  }
  return out;
}

List create_folds_B(NumericMatrix M, NumericMatrix C, bool to_estimate_u, bool to_estimate_v, NumericMatrix mask, NumericMatrix W, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds=5){

  // This function creates folds for cross-validation. Each fold contains a training and validation sets.
  // For each of these folds the initial solutions for fixed effects are then computed, as for large lambda_L and and lambda_B,
  // L and B would be equal to zero. This initialization is very helpful as it will be used later for the warm start.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List out(num_folds);
  std::default_random_engine generator;
  std::bernoulli_distribution distribution(cv_ratio);
  for (int k = 0; k<num_folds; k++){
    MatrixXd ma_new(num_rows,num_cols);
    for (int i = 0; i < num_rows; i++){
      for (int j = 0; j < num_cols; j++){
        ma_new(i,j)=distribution(generator);
      }
    }
    MatrixXd fold_mask_ = mask_.array() * ma_new.array();
    MatrixXd M_tr_ = M_.array() * fold_mask_.array();
    MatrixXd W_tr_ = W_.array() * fold_mask_.array();
    NumericMatrix fold_mask = wrap(fold_mask_);
    NumericMatrix M_tr = wrap(M_tr_);
    NumericMatrix W_tr = wrap(W_tr_);
    List tmp_uv = initialize_uv_B(M_tr, C, fold_mask, W_tr, to_estimate_u, to_estimate_v, niter = 1000, rel_tol = 1e-5);
    List fold_k = List::create(Named("u") = tmp_uv["u"],
                               Named("v") = tmp_uv["v"],
                               Named("lambda_L_max") = tmp_uv["lambda_L_max"],
                               Named("lambda_B_max") = tmp_uv["lambda_B_max"],
                               Named("fold_mask") = fold_mask);
    out[k] = fold_k;
  }
  return out;
}

List NNM_fit(NumericMatrix M, NumericMatrix mask, NumericMatrix L_init, NumericMatrix W, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, double lambda_L, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function performs coordinate descent updates.
  // For given matrices M, mask, and initial starting decomposition given by L_init, u_init, and v_init,
  // matrices L, u, and v are updated till convergence via coordinate descent.

  double obj_val;
  double new_obj_val=0;
  List svd_dec;
  svd_dec = MySVD(L_init);
  VectorXd sing = svd_dec["Sigma"];
  double sum_sigma = sing.sum();
  obj_val = Compute_objval(M, mask, L_init, W, u_init, v_init, sum_sigma, lambda_L);
  NumericMatrix L = L_init;
  NumericVector u = u_init;
  NumericVector v = v_init;
  int term_iter = 0;
  for(int iter = 0; iter < niter; iter++){
    // Update u
    if(to_estimate_u == 1){
      u = update_u(M, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(M.rows()));
    }
    // Update v
    if(to_estimate_v == 1){
      v = update_v(M, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(M.cols()));
    }
    // Update L
    List upd_L = update_L(M, mask, L, u, v, lambda_L);
    NumericMatrix L_upd = upd_L["L"];
    L = L_upd;
    sing = upd_L["Sigma"];
    double sum_sigma = sing.sum();
    // Check if accuracy is achieved
    new_obj_val = Compute_objval(M, mask, L, W, u, v, sum_sigma, lambda_L);
    double rel_error = (obj_val-new_obj_val)/obj_val;
    if(new_obj_val < 1e-8){
      break;
    }
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    term_iter = iter;
    obj_val = new_obj_val;
  }
  if(is_quiet == 0){
    std::cout << "Terminated at iteration : " << term_iter << ", for lambda_L :" << lambda_L << ", with obj_val :" << new_obj_val << std::endl;
  }
  return List::create(Named("L") = L,
                      Named("u") = u,
                      Named("v") = v);
}

List NNM_fit_B(NumericMatrix M, NumericMatrix C, NumericMatrix B_init, NumericMatrix mask, NumericMatrix L_init, NumericMatrix W, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, double lambda_L, double lambda_B, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function performs cyclic coordinate descent updates.
  // For given matrices M, mask, and initial starting decomposition given by L_init, u_init, and v_init,
  // matrices L, B, u, and v are updated till convergence via coordinate descent.
  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  int num_rows = M_.rows();
  int num_cols = M_.cols();

  double obj_val;
  double new_obj_val=0;
  List svd_dec;
  svd_dec = MySVD(L_init);
  VectorXd sing = svd_dec["Sigma"];
  double sum_sigma = sing.sum();
  obj_val = Compute_objval_B(M, C, B_init, mask, L_init, W, u_init, v_init, sum_sigma, lambda_L, lambda_B);
  NumericMatrix B = B_init;
  NumericMatrix L = L_init;
  NumericVector u = u_init;
  NumericVector v = v_init;
  int term_iter = 0;
  for(int iter = 0; iter < niter; iter++){
    // Update u
    if(to_estimate_u == 1){
      u = update_u_B(M, C, B, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(M.rows()));
    }
    // Update v
    if(to_estimate_v == 1){
      v = update_v_B(M, C, B, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(M.cols()));
    }
    // Update B
    NumericMatrix upd_B = update_B_B(M, C, B, mask, L, u, v, lambda_B);
    B = upd_B;
    // Update L
    List upd_L = update_L_B(M, C, B, mask, L, u, v, lambda_L);
    NumericMatrix L_upd = upd_L["L"];
    L = L_upd;
    sing = upd_L["Sigma"];
    double sum_sigma = sing.sum();
    // Check if accuracy is achieved
    new_obj_val = Compute_objval_B(M, C, B, mask, L, W, u, v, sum_sigma, lambda_L, lambda_B);
    double rel_error = (obj_val-new_obj_val)/obj_val;
    if(new_obj_val < 1e-8){
      break;
    }
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    term_iter = iter;
    obj_val = new_obj_val;
  }
  if(is_quiet == 0){
    std::cout << "Terminated at iteration : " << term_iter << ", for lambda_L :" << lambda_L << ", with obj_val :" << new_obj_val << std::endl;
  }
  return List::create(Named("B") = B,
                      Named("L") = L,
                      Named("u") = u,
                      Named("v") = v);
}

List NNM_with_uv_init(NumericMatrix M, NumericMatrix mask, NumericMatrix W, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function actually does the warm start.
  // Idea here is that we start from L_init=0, and converged u_init and v_init and then find the
  // fitted model, i.e, new L,u, and v. Then, we pass these parameters for the next value of lambda_L.
  // It is worth noting that lambda_L's are sorted in decreasing order.

  int num_lam = lambda_L.size();
  int num_rows = M.rows();
  int num_cols = M.cols();
  List res(num_lam);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  for (int i = 0; i<num_lam; i++){
    List fact = NNM_fit(M, mask, L_init, W, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L[i], niter, rel_tol, is_quiet);
    res[i] = fact;
    NumericMatrix L_upd = fact["L"];
    L_init = L_upd;
    u_init = fact["u"];
    v_init = fact["v"];
  }
  return res;
}

List NNM_with_uv_init_B(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, NumericVector lambda_B, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function actually does the warm start.
  // Idea here is that we start from L_init=0 and and B_init, and converged u_init and v_init and then find the
  // fitted model, i.e, new L, B, u, and v. Then, we pass these parameters for the next value of lambda_L and lambda_B.
  // Both lambda_L  and lambda_B vectors are sorted in decreasing order. As we are cross-validating over a grid, there
  // are two options for taking as the previous model (initialization of new point on grid). Here we always use the model
  // with lambda_L just before this model, keeping lambda_B fixed. The only exception is when we are at the largest lambda_L,
  // for which we take previous lambda_B, while keep lambda_L fixed.

  int num_lam_L = lambda_L.size();
  int num_lam_B = lambda_B.size();
  int num_rows = M.rows();
  int num_cols = M.cols();
  List res(num_lam_L*num_lam_B);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  NumericMatrix B_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  for (int j = 0; j<num_lam_B; j++){
  	if(j > 0){
      List previous_B = res[(j-1)*num_lam_B];
      NumericMatrix L_upd = previous_B["L"];
      NumericMatrix B_upd = previous_B["B"];
      L_init = L_upd;
      u_init = previous_B["u"];
      v_init = previous_B["v"];
      B_init = B_upd;
   }
   for (int i = 0; i<num_lam_L; i++){
     List previous_L = NNM_fit_B(M, C, B_init, mask, L_init, W, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L[i], lambda_B[j], niter, rel_tol, is_quiet);
     res[j*num_lam_L+i] = previous_L;
     NumericMatrix L_upd = previous_L["L"];
     NumericMatrix B_upd = previous_L["B"];
     L_init = L_upd;
     u_init = previous_L["u"];
     v_init = previous_L["v"];
     B_init = B_upd;
   }
 }
  return res;
}

List NNM_B(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, int num_lam_L = 30, int num_lam_B = 30, NumericVector lambda_L = NumericVector::create(), NumericVector lambda_B = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function in its default format is just a wraper, which only passes vectors of all zero for u_init and v_init to NNM_with_uv_init_B.
  // The function computes the good range for lambda_L and lambda_B and fits using warm-start described in NNM_with_uv_init_B to all those values.
  // The user has the ability to set vectors of lambda_L and lambda_B manually, although it is not advisable.

  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv_B(M, C, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(lambda_L.size() == 0){
    NumericVector lambda_Ls(num_lam_L);
    double max_lam_L=tmp_uv["lambda_L_max"];
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    lambda_L = lambda_Ls;
  }
  else{
    num_lam_L = lambda_L.size();
  }
  if(lambda_B.size() == 0){
    NumericVector lambda_Bs(num_lam_B);
    double max_lam_B=tmp_uv["lambda_B_max"];
    NumericVector lambda_Bs_without_zero = logsp(log10(max_lam_B), log10(max_lam_B)-3, num_lam_B-1);
    for(int i=0; i<num_lam_B-1; i++){
      lambda_Bs(i)= lambda_Bs_without_zero(i);
    }
    lambda_Bs(num_lam_B-1) = 0;
    lambda_B = lambda_Bs;
  }
  else{
    num_lam_B = lambda_B.size();
  }
  List tmp_res;
  if(to_estimate_u == 1 || to_estimate_v ==1){
    tmp_res =  NNM_with_uv_init_B(M, C, mask, W, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, lambda_L, lambda_B, niter, rel_tol, is_quiet);
  }
  else{
    tmp_res = NNM_with_uv_init_B(M, C, mask, W, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, lambda_L, lambda_B, niter, rel_tol, is_quiet);
  }

  List out(num_lam_L*num_lam_B);
  for (int j = 0; j< num_lam_B; j++){
    for (int i = 0; i < num_lam_L; i++){
      int current_ind = j*i;
      List current_config = tmp_res(j*i);
      List this_config = List::create(Named("B") = current_config["B"],
        Named("L") = current_config["L"],
        Named("u") = current_config["u"],
        Named("v") = current_config["v"],
        Named("lambda_L") = lambda_L[i],
        Named("lambda_B") = lambda_B[j]);
      out[current_ind] = this_config;
    }
  }
  return out;
}

List NNM_with_uv_init_B_opt_path(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, NumericVector lambda_B, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  // This function is similar to NNM_B, with one key difference. This function instead of fitting to all models on the grid described by lambda_Ls and lambda_Bs
  // only considers the shortest path from the point on the grid with highest lambda_L and lambda_B to the point on the grid with smallest values of lambda_L
  // and and lambda_B. The key benefit of using this function is that, for chosen values of lambda_L and lambda_B, training can be much faster as the number of
  // trained models is M+O-1 compared to M*O, where M is the length of lambda_L and O is the length of lambda_B.

  int num_lam_L = lambda_L.size();
  int num_lam_B = lambda_B.size();
  int num_rows = M.rows();
  int num_cols = M.cols();

  List res(num_lam_L+num_lam_B-1);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  NumericMatrix B_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  for (int j = 0; j<num_lam_B; j++){
    List previous_pt = NNM_fit_B(M, C, B_init, mask, L_init, W, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L(0), lambda_B(j), niter, rel_tol, is_quiet);
    NumericMatrix L_upd = previous_pt["L"];
    NumericMatrix B_upd = previous_pt["B"];
    res[j] = previous_pt;
    L_init = L_upd;
    B_init = B_upd;
    u_init = previous_pt["u"];
    v_init = previous_pt["v"];
  }
  for (int i = 1; i<num_lam_L; i++){
    List previous_pt = NNM_fit_B(M, C, B_init, mask, L_init, W, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L(i), lambda_B(num_lam_B-1), niter, rel_tol, is_quiet);
    res[(i-1)+num_lam_B] = previous_pt;
    NumericMatrix L_upd = previous_pt["L"];
    NumericMatrix B_upd = previous_pt["B"];
    L_init = L_upd;
    B_init = B_upd;
    u_init = previous_pt["u"];
    v_init = previous_pt["v"];
  }
  return res;
}

List NNM_B_opt_path(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, NumericVector lambda_B, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  // This is just a wrapper for NNM_with_uv_init_B_opt_path, which just passes the initialization to this function.
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv_B(M, C, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(to_estimate_u == 1 || to_estimate_v ==1){
    return NNM_with_uv_init_B_opt_path(M, C, mask, W, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, lambda_L, lambda_B, niter, rel_tol, is_quiet);
  }
  else{
    return NNM_with_uv_init_B_opt_path(M, C, mask, W, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, lambda_L, lambda_B, niter, rel_tol, is_quiet);
  }
}

List NNM(NumericMatrix M, NumericMatrix mask, NumericMatrix W, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function is just a wraper, which only passes vectors of all zero for u_init and v_init to NNM_with_uv_init.
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv(M, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(lambda_L.size() == 0){
    NumericVector lambda_Ls(num_lam_L);
    double max_lam_L=tmp_uv["lambda_L_max"];
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    lambda_L = lambda_Ls;
  }
  else{
    num_lam_L = lambda_L.size();
  }
  List tmp_res;
  if(to_estimate_u == 1 || to_estimate_v ==1){
    tmp_res = NNM_with_uv_init(M, mask, W, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
  else{
    tmp_res = NNM_with_uv_init(M, mask, W, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
  List out(num_lam_L);
  for (int i = 0; i < num_lam_L; i++){
    int current_ind = i;
    List current_config = tmp_res(current_ind);
    List this_config = List::create(Named("L") = current_config["L"],
                                    Named("u") = current_config["u"],
                                    Named("v") = current_config["v"],
                                    Named("lambda_L") = lambda_L[i]);
    out[current_ind] = this_config;
  }
  return out;
}

List NNM_CV(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int num_lam, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.6, int num_folds = 5, bool is_quiet = 1){

  // This function is the core function of NNM. Basically, it creates num_folds number of folds and does cross-validation
  // for choosing the best value of lambda_L, using data. Then, using the best model it fits to the
  // entire training set and computes resulting L, u, and v. The output of this function basically
  // contains all the information that we want from the algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List confgs = create_folds(M, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol , cv_ratio, num_folds);
  MatrixXd MSE(num_lam, num_folds);
  double max_lam_L=-1;
  for(int k=0; k<num_folds; k++){
    List h = confgs[k];
    double lam_max = h["lambda_L_max"];
    if(lam_max > max_lam_L){
      max_lam_L = lam_max;
    }
  }
  NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam-1);
  NumericVector lambda_Ls(num_lam);
  for(int i=0; i<num_lam-1; i++){
    lambda_Ls(i)= lambda_Ls_without_zero(i);
  }
  lambda_Ls(num_lam-1) = 0;
  for(int k=0; k<num_folds; k++){
    if(is_quiet == 0){
      std::cout << "Fold number " << k << " started" << std::endl;
    }
    List h = confgs[k];
    NumericMatrix mask_training = h["fold_mask"];
    const Map<MatrixXd> mask_training_(as<Map<MatrixXd> >(mask_training));
    MatrixXd M_tr_ = mask_training_.array() * M_.array();
    MatrixXd W_tr_ = mask_training_.array() * W_.array();
    NumericMatrix M_tr = wrap(M_tr_);
    NumericMatrix W_tr = wrap(W_tr_);
    MatrixXd mask_validation_ = mask_.array() * (MatrixXd::Constant(num_rows,num_cols,1.0)-mask_training_).array();
    NumericMatrix mask_validation = wrap(mask_validation_);
    List train_configs = NNM_with_uv_init(M_tr, mask_training, W_tr, h["u"], h["v"], to_estimate_u, to_estimate_v, lambda_Ls, niter, rel_tol, is_quiet);
    for (int i = 0; i < num_lam; i++){
      List this_config = train_configs[i];
      NumericMatrix L_use = this_config["L"];
      NumericVector u_use = this_config["u"];
      NumericVector v_use = this_config["v"];
      MSE(i,k) = std::pow(Compute_RMSE(M, mask_validation, L_use, u_use, v_use) ,2);
    }
  }
  VectorXd Avg_MSE = MSE.rowwise().mean();
  VectorXd Avg_RMSE = Avg_MSE.array().sqrt();
  Index minindex;
  double minRMSE = Avg_RMSE.minCoeff(&minindex);
  if(is_quiet == 0){
    std::cout << "Minimum RMSE achieved on validation set :" << minRMSE << std::endl;
    std::cout << "Optimum value of lambda_L : " << lambda_Ls[minindex] << std::endl;
    std::cout << "Fitting to the test set using optimum lambda_L..." << std::endl;
  }
  NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_Ls(minindex)];
  List final_config = NNM(M, mask, W, lambda_Ls_n.size(), lambda_Ls, to_estimate_u, to_estimate_v, niter, rel_tol, 1);
  List z = final_config[minindex];
  MatrixXd L_fin = z["L"];
  VectorXd u_fin = z["u"];
  VectorXd v_fin = z["v"];
  return List::create(Named("L") = L_fin,
                      Named("u") = u_fin,
                      Named("v") = v_fin,
                      Named("Avg_RMSE") = Avg_RMSE,
                      Named("best_lambda") = lambda_Ls[minindex],
                      Named("min_RMSE") = minRMSE,
                      Named("lambda_L") = lambda_Ls);
}

List NNM_CV_B(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int num_lam_L, int num_lam_B, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){

  // This function is the core function of NNM. Basically, it creates num_folds number of folds and does cross-validation
  // for choosing the best value of lambda_L, using data. Then, using the best model it fits to the
  // entire training set and computes resulting L, u, and v. The output of this function basically
  // contains all the information that we want from the algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List confgs = create_folds_B(M, C, to_estimate_u, to_estimate_v, mask, W, niter, rel_tol, cv_ratio, num_folds);

  double max_lam_L=-1;
  double max_lam_B=-1;
  for(int k=0; k<num_folds; k++){
    List h = confgs[k];
    double lam_L_max = h["lambda_L_max"];
    double lam_B_max = h["lambda_B_max"];
    if(lam_L_max > max_lam_L){
      max_lam_L = lam_L_max;
    }
    if(lam_B_max > max_lam_B){
      max_lam_B = lam_B_max;
    }
  }
  NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
  NumericVector lambda_Bs_without_zero = logsp(log10(max_lam_B), log10(max_lam_B)-3, num_lam_B-1);
  NumericVector lambda_Ls(num_lam_L);
  for(int i=0; i<num_lam_L-1; i++){
    lambda_Ls(i)= lambda_Ls_without_zero(i);
  }
  lambda_Ls(num_lam_L-1) = 0;
  NumericVector lambda_Bs(num_lam_B);
  for(int i=0; i<num_lam_B-1; i++){
    lambda_Bs(i)= lambda_Bs_without_zero(i);
  }
  lambda_Bs(num_lam_B-1) = 0;
  MatrixXd MSE = MatrixXd::Zero(num_lam_L,num_lam_B);
  for(int k=0; k<num_folds; k++){
    if(is_quiet == 0){
      std::cout << "Fold number " << k << " started" << std::endl;
    }
    List h = confgs[k];
    NumericMatrix mask_training = h["fold_mask"];
    const Map<MatrixXd> mask_training_(as<Map<MatrixXd> >(mask_training));
    MatrixXd M_tr_ = mask_training_.array() * M_.array();
    MatrixXd W_tr_ = mask_training_.array() * W_.array();
    NumericMatrix M_tr = wrap(M_tr_);
    NumericMatrix W_tr = wrap(W_tr_);
    MatrixXd mask_validation_ = mask_.array() * (MatrixXd::Constant(num_rows,num_cols,1.0)-mask_training_).array();
    NumericMatrix mask_validation = wrap(mask_validation_);
    List train_configs = NNM_with_uv_init_B(M_tr, C, mask_training, W_tr, h["u"], h["v"], to_estimate_u, to_estimate_v, lambda_Ls, lambda_Bs, niter, rel_tol, is_quiet);
    for (int i = 0; i < num_lam_L; i++){
      for (int j = 0; j < num_lam_B; j++){
        List this_config = train_configs[j*num_lam_L+i];
        NumericMatrix L_use = this_config["L"];
        NumericVector u_use = this_config["u"];
        NumericVector v_use = this_config["v"];
        NumericMatrix B_use = this_config["B"];
        MSE(i,j) += std::pow(Compute_RMSE_B(M, C, B_use, mask_validation, L_use, u_use, v_use) ,2);
      }  
    }
  }
  MatrixXd Avg_MSE = MSE/num_folds;
  MatrixXd Avg_RMSE = Avg_MSE.array().sqrt();
  Index min_L_index;
  Index min_B_index;
  double minRMSE = Avg_RMSE.minCoeff(&min_L_index, &min_B_index); 
  if(is_quiet == 0){
    std::cout << "Minimum RMSE achieved on validation set :" << minRMSE << std::endl;
    std::cout << "Optimum value of lambda_L : " << lambda_Ls(min_L_index) << std::endl;
    std::cout << "Optimum value of lambda_B : " << lambda_Bs(min_B_index) << std::endl;
    std::cout << "Fitting to the test set using optimum lambda_L and lambda_B ..." << std::endl;
  }
  NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_Ls(min_L_index)];
  NumericVector lambda_Bs_n = lambda_Bs[lambda_Bs >= lambda_Bs(min_B_index)];
  List final_config = NNM_B_opt_path(M, C, mask, W, to_estimate_u, to_estimate_v, lambda_Ls_n, lambda_Bs_n, niter, rel_tol, 1);
  List z = final_config[min_L_index+min_B_index-1];
  MatrixXd B_fin = z["B"];
  MatrixXd L_fin = z["L"];
  VectorXd u_fin = z["u"];
  VectorXd v_fin = z["v"];
  return List::create(Named("B") = B_fin,
                      Named("L") = L_fin,
                      Named("u") = u_fin,
                      Named("v") = v_fin,
                      Named("Avg_RMSE") = Avg_RMSE,
                      Named("best_lambda_L") = lambda_Ls[min_L_index],
                      Named("best_lambda_B") = lambda_Bs[min_B_index],
                      Named("min_RMSE") = minRMSE, 
                      Named("lambda_L") = lambda_Ls,
                      Named("lambda_B") = lambda_Bs);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
/////// Input Checks
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////


bool mask_check(NumericMatrix mask){
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  MatrixXd H = (MatrixXd::Constant(mask_.rows(),mask_.cols(),1.0) - mask_).cwiseProduct(mask_);
  return(H.isZero(1e-5));
}

bool C_size_check(NumericMatrix M, NumericMatrix C){
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  return (M_.rows() == C_.rows());
}

bool mask_size_check(NumericMatrix M, NumericMatrix mask){
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  return (M_.rows() == mask_.rows() && M_.cols() == mask_.cols());
}

bool mask_size_check_W(NumericMatrix W, NumericMatrix mask){
  const Map<MatrixXd> W_(as<Map<MatrixXd> >(W));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  return (W_.rows() == mask_.rows() && W_.cols() == mask_.cols());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
////// Export functions to use in R
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////
// EXPORT mcnnm_lam_range
//////////////////////////////

int mcnnm_lam_range_check(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check_W(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  return 1;
}
// [[Rcpp::export]]
double mcnnm_lam_range(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){
  int input_checks = mcnnm_lam_range_check(M, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  List res= initialize_uv(M, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  return res["lambda_L_max"];
}

///////////////////////////////
// EXPORT mcnnm_lam_range
//////////////////////////////

int mcnnm_wc_lam_range_check(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check_W(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(C_.rows() > 0 && C_size_check(M,C) == 0){
    std::cerr << "Error: Number of rows of C should match with the number of rows of M" << std::endl;
    return 0;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc_lam_range(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){

  int input_checks = mcnnm_wc_lam_range_check(M, C, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  List res= initialize_uv_B(M, C, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  return List::create(Named("lambda_L_max") = res["lambda_L_max"],
                      Named("lambda_B_max") = res["lambda_B_max"]);
}

/////////////////////////////////
// EXPORT mcnnm
/////////////////////////////////

int mcnnm_check(NumericMatrix M, NumericMatrix mask, NumericMatrix W, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  if(lambda_L.size() > 0){
    num_lam_L = lambda_L.size();
  }
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }

  if(num_lam_L > 2500){
    std::cerr << "Warning: The training might take very long. Please decrease number of lambda_Ls" << std::endl;
  }
  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm(NumericMatrix M, NumericMatrix mask, NumericMatrix W, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_check(M, mask, W, num_lam_L, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  return NNM(M, mask, W, num_lam_L, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
}


/////////////////////////////////
// EXPORT mcnnm_fit
/////////////////////////////////

int mcnnm_fit_check(NumericMatrix M, NumericMatrix mask, NumericMatrix W, double lambda_L, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_fit(NumericMatrix M, NumericMatrix mask, NumericMatrix W, double lambda_L, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_fit_check(M, mask, W, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  double max_lam_L = mcnnm_lam_range(M, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(lambda_L >= max_lam_L){
    NumericVector lambda_Ls(1);
    lambda_Ls(0) = lambda_L;
    List Q = NNM(M, mask, W, 1, lambda_Ls, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
    List final_config = Q[0];
    return List::create(Named("L") = final_config["L"],
                        Named("u") = final_config["u"],
                        Named("v") = final_config["v"],
                        Named("lambda_L") = lambda_L);
  }
  else{
    int num_lam_L = 100;
    NumericVector lambda_Ls(num_lam_L);
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_L];
    int num_lam_L_n = lambda_Ls_n.size();
    NumericVector lambda_Ls_fin(num_lam_L_n+1);
    for(int i=0; i<num_lam_L_n; i++){
      lambda_Ls_fin(i)= lambda_Ls_n(i);
    }
    lambda_Ls_fin(num_lam_L_n) = lambda_L;
    List Q = NNM(M, mask, W, num_lam_L_n+1, lambda_Ls_fin, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
    List final_config = Q[num_lam_L_n];
    return List::create(Named("L") = final_config["L"],
                        Named("u") = final_config["u"],
                        Named("v") = final_config["v"],
                        Named("lambda_L") = lambda_L);
  }
}

/////////////////////////////////
// EXPORT mcnnm_wc
////////////////////////////////

int mcnnm_wc_check(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, int num_lam_L = 30, int num_lam_B = 30, NumericVector lambda_L = NumericVector::create(),NumericVector lambda_B = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  if(lambda_L.size() > 0){
    num_lam_L = lambda_L.size();
  }
  if(lambda_B.size() > 0){
    num_lam_B = lambda_B.size();
  }
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));

  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(C_.rows() > 0 && C_size_check(M,C) == 0){
    std::cerr << "Error: Number of rows of C should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(num_lam_L * num_lam_B > 2500){
    std::cerr << "Warning: The training might take very long. Please decrease number of lambda_Ls or lambda_Bs" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, int num_lam_L = 30, int num_lam_B = 30, NumericVector lambda_L = NumericVector::create(), NumericVector lambda_B = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){

  int input_checks = mcnnm_wc_check(M, C, mask, W, num_lam_L, num_lam_B, lambda_L, lambda_B, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  List res = NNM_B(M, C, mask, W, num_lam_L, num_lam_B, lambda_L, lambda_B, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);

  return res;
}

/////////////////////////////////
// EXPORT mcnnm_wc
////////////////////////////////

int mcnnm_wc_fit_check(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, double lambda_L, double lambda_B, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(C,mask) == 0){
    std::cerr << "Error: C matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(C_.rows() > 0 && C_size_check(M,C) == 0){
    std::cerr << "Error: Number of rows of C should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc_fit(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, double lambda_L, double lambda_B, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  int input_checks = mcnnm_wc_fit_check(M, C, mask, W, lambda_L, lambda_B, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  List ranges = mcnnm_wc_lam_range(M, C, mask, W, to_estimate_u, to_estimate_v, niter, rel_tol);
  double max_lam_L = ranges["lambda_L_max"];
  double max_lam_B = ranges["lambda_B_max"];
  NumericVector lambda_Ls_fin;
  NumericVector lambda_Bs_fin;
  if(lambda_L >= max_lam_L){
    NumericVector lambda_Ls(1);
    lambda_Ls(0) = lambda_L;
    lambda_Ls_fin = lambda_Ls;
  }
  else{
    int num_lam_L = 30;
    NumericVector lambda_Ls(num_lam_L);
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_L];
    int num_lam_L_n = lambda_Ls_n.size();
    NumericVector lambda_Ls_(num_lam_L_n+1);
    for(int i=0; i<num_lam_L_n; i++){
      lambda_Ls_(i)= lambda_Ls_n(i);
    }
    lambda_Ls_(num_lam_L_n) = lambda_L;
    lambda_Ls_fin = lambda_Ls_;
  }
  if(lambda_B >= max_lam_B){
    NumericVector lambda_Bs(1);
    lambda_Bs(0) = lambda_B;
    lambda_Bs_fin = lambda_Bs;
  } else{
    int num_lam_B = 30;
    NumericVector lambda_Bs(num_lam_B);
    NumericVector lambda_Bs_without_zero = logsp(log10(max_lam_B), log10(max_lam_B)-3, num_lam_B-1);
    for(int i=0; i<num_lam_B-1; i++){
      lambda_Bs(i)= lambda_Bs_without_zero(i);
    }
    lambda_Bs(num_lam_B-1) = 0;
    NumericVector lambda_Bs_n = lambda_Bs[lambda_Bs >= lambda_B];
    int num_lam_B_n = lambda_Bs_n.size();
    NumericVector lambda_Bs_(num_lam_B_n+1);
    for(int i=0; i<num_lam_B_n; i++){
      lambda_Bs_(i)= lambda_Bs_n(i);
    }
    lambda_Bs_(num_lam_B_n) = lambda_B;
    lambda_Bs_fin = lambda_Bs_;
  }
  List Q = NNM_B_opt_path(M, C, mask, W, to_estimate_u, to_estimate_v, lambda_Ls_fin, lambda_Bs_fin, niter, rel_tol, is_quiet);
  List final_config = Q[Q.size()-1];

  return List::create(Named("B") = final_config["B"],
                      Named("L") = final_config["L"],
                      Named("u") = final_config["u"],
                      Named("v") = final_config["v"],
                      Named("lambda_L") = lambda_L,
                      Named("lambda_B") = lambda_B);
}

/////////////////////////////////
// EXPORT mcnnm_cv
/////////////////////////////////

int mcnnm_cv_check(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int num_lam_L , int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }

  if(num_lam_L> 2500){
    std::cerr << "Warning: The cross-validation might take very long. Please decrease number of lambda_Ls" << std::endl;
  }
  if(cv_ratio < 0.1 || cv_ratio > 0.9){
    std::cerr << "Error: The cross-validation ratio should be between 10 to 90 percent for getting accurate results. Please modify it" << std::endl;
    return 0;
  }

  if(num_folds > 20){
    std::cerr << "Warning: Number of random folds are chosen to be greater than 20. This process might take long" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}

// [[Rcpp::export]]
List mcnnm_cv(NumericMatrix M, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1, int num_lam_L = 100, int niter = 400, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds = 5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_cv_check(M, mask, W, to_estimate_u, to_estimate_v, num_lam_L, niter, rel_tol, cv_ratio, num_folds, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  return NNM_CV(M, mask, W, to_estimate_u, to_estimate_v, num_lam_L, niter, rel_tol, cv_ratio, num_folds, is_quiet);
}

//////////////////////////////////
// EXPORT mcnnm_wc_cv
/////////////////////////////////

int mcnnm_wc_cv_check(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u, bool to_estimate_v, int num_lam_L, int num_lam_B, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){
  const Map<MatrixXd> C_(as<Map<MatrixXd> >(C));  
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check_W(W,mask) == 0){
    std::cerr << "Error: W matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(C_.rows() > 0 && C_size_check(M,C) == 0){
    std::cerr << "Error: Number of rows of C should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(num_lam_L * num_lam_B > 2500){
    std::cerr << "Warning: The cross-validation might take very long. Please decrease number of lambda_Ls or lambda_Bs" << std::endl;
  }
  if(cv_ratio < 0.1 || cv_ratio > 0.9){
    std::cerr << "Error: The cross-validation ratio should be between 10 to 90 percent for getting accurate results. Please modify it" << std::endl;
    return 0;
  }

  if(num_folds > 3){
    std::cerr << "Warning: Number of random folds are chosen to be greater than 3. This process might take long" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}

// [[Rcpp::export]]
List mcnnm_wc_cv(NumericMatrix M, NumericMatrix C, NumericMatrix mask, NumericMatrix W, bool to_estimate_u = 1, bool to_estimate_v = 1,  int num_lam_L = 30, int num_lam_B = 30, int niter = 100, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds = 1, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_wc_cv_check(M, C, mask, W, to_estimate_u, to_estimate_v, num_lam_L, num_lam_B, niter, rel_tol, cv_ratio, num_folds, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  res = NNM_CV_B(M, C, mask, W, to_estimate_u, to_estimate_v, num_lam_L, num_lam_B, niter, rel_tol, cv_ratio, num_folds, is_quiet);

  return res;
}
