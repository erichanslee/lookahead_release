# -*- coding: utf-8 -*-
import numpy
import scipy.linalg

def compute_cholesky_for_gp_sampling(covariance_matrix):
  try:
    chol_cov = -scipy.linalg.cholesky(covariance_matrix, lower=True, overwrite_a=True, check_finite=False)
  except scipy.linalg.LinAlgError:
    U, E, _ = scipy.linalg.svd(covariance_matrix, overwrite_a=True, check_finite=False)
    chol_cov = U * numpy.sqrt(E)[numpy.newaxis, :]
    chol_cov = -scipy.linalg.qr(chol_cov.T, mode='r', overwrite_a=True, check_finite=False)[0].T
  return chol_cov
