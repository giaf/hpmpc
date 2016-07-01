/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
*                                                                                                 *
* HPMPC is free software; you can redistribute it and/or                                          *
* modify it under the terms of the GNU Lesser General Public                                      *
* License as published by the Free Software Foundation; either                                    *
* version 2.1 of the License, or (at your option) any later version.                              *
*                                                                                                 *
* HPMPC is distributed in the hope that it will be useful,                                        *
* but WITHOUT ANY WARRANTY; without even the implied warranty of                                  *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                                            *
* See the GNU Lesser General Public License for more details.                                     *
*                                                                                                 *
* You should have received a copy of the GNU Lesser General Public                                *
* License along with HPMPC; if not, write to the Free Software                                    *
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA                  *
*                                                                                                 *
* Author: Gianluca Frison, giaf (at) dtu.dk                                                       *
*                                                                                                 *
**************************************************************************************************/

#include "target.h"
#include "block_size.h"

#ifdef __cplusplus
extern "C" {
#endif

// work space: dynamic definition as function return value

// Riccati-based IP method for hard-constrained MPC, double precision
int hpmpc_d_ip_mpc_hard_tv_work_space_size_doubles(int N, int nx, int nu, int nb, int ng, int ngN);



// matrices are assumed to be passed in c (or row-major) order
int c_order_d_ip_mpc_hard_tv( int *kk, int k_max, double mu0, double mu_tol, int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, int free_x0, int warm_start, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *pi, double *lam, double *t, double *inf_norm_res, double *work0, double *stat);
void c_order_d_solve_kkt_new_rhs_mpc_hard_tv(int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, int free_x0, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *pi, double *lam, double *t,	double *inf_norm_res, double *work0);




// matrices are assumed to be passed in fortran (or column-major) order
int fortran_order_d_ip_mpc_hard_tv( int *kk, int k_max, double mu0, double mu_tol, int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, int free_x0, int warm_start, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *pi, double *lam, double *t, double *inf_norm_res, double *work0, double *stat);
void fortran_order_d_solve_kkt_new_rhs_mpc_hard_tv(int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, int free_x0, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *pi, double *lam, double *t,	double *inf_norm_res, double *work0);



// new interfaces
int hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int *ng);

int c_order_d_ip_ocp_hard_tv(
							int *kk, int k_max, double mu0, double mu_tol,
							int N, int const *nx, int const *nu, int const *nb, int const *ng,
							int warm_start,
							double const * const *A, double const * const *B, double const * const *b,
							double const * const *Q, double const * const *S, double const * const *R, double const * const *q, double const * const *r,
							double const * const *lb, double const * const *ub,
							double const * const *C, double const * const *D, double const * const *lg, double const * const *ug,
							double * const *x, double * const *u, double * const *pi, double * const *lam, double * const *t,
							double *inf_norm_res,
							void *work0,
							double *stat);

void c_order_d_solve_kkt_new_rhs_ocp_hard_tv(int N, int *nx, int *nu, int *nb, int *ng,	double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, double **lb, double **ub, double **C, double **D, double **lg, double **ug,	double **x, double **u, double **pi, double **lam, double **t, double *inf_norm_res, double *work0);

int fortran_order_d_ip_ocp_hard_tv(int *kk, int k_max, double mu0, double mu_tol, int N, int *nx, int *nu, int *nb, int *ng, int warm_start, double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, double **lb, double **ub, double **C, double **D, double **lg, double **ug, double **x, double **u, double **pi, double **lam, double **t, double *inf_norm_res, void *work0, double *stat);
void fortran_order_d_solve_kkt_new_rhs_ocp_hard_tv(int N, int *nx, int *nu, int *nb, int *ng,	double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, double **lb, double **ub, double **C, double **D, double **lg, double **ug,	double **x, double **u, double **pi, double **lam, double **t, double *inf_norm_res, double *work0);



#ifdef __cplusplus
}	// extern "C"
#endif
