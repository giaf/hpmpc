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

// hard-constrained Riccati-based IPM for MPC
int d_ip2_mpc_hard_tv_work_space_size_doubles(int N, int *nx, int *nu, int *nb, int *ng);
int d_ip2_mpc_hard_tv(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **pBAbt, double **pQ, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory);
void d_res_ip_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **hpBAbt, double **hb, double **hpQ, double **hq, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu);

// soft-constrained Riccati-based IPM for MPC
int d_ip2_mpc_soft_tv_work_space_size_doubles(int N, int *nx, int *nu, int *nb, int *ng, int *ns);
int d_ip2_mpc_soft_tv(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **pBAbt, double **pQ, double **Z, double **z, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory);
void d_res_ip_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **hpBAbt, double **hpQ, double **hq, double **hZ, double **hz, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double **hrz, double *mu);


