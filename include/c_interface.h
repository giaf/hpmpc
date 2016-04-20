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



// work space: dynamic definition as function return value

// Riccati-based IP method for hard-constrained MPC, double precision
int hpmpc_d_ip_mpc_hard_tv_work_space_size_doubles(int N, int nx, int nu, int nb, int ng, int ngN);



// matrices are assumed to be passed in c (or row-major) order
int c_order_d_ip_mpc_hard_tv( int *kk, int k_max, double mu0, double mu_tol, char prec, int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *work0, double *stat, int compute_res, double *inf_norm_res, int compute_mult, double *pi, double *lam, double *t);



// matrices are assumed to be passed in fortran (or column-major) order
int fortran_order_d_ip_mpc_hard_tv( int *kk, int k_max, double mu0, double mu_tol, char prec, int N, int nx, int nu, int nb, int ng, int ngN, int time_invariant, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double *lb, double *ub, double *C, double *D, double *lg, double *ug, double *Cf, double *lgf, double *ugf, double* x, double* u, double *work0, double *stat, int compute_res, double *inf_norm_res, int compute_mult, double *pi, double *lam, double *t);


