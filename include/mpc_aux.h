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

// box-constrained MPC
void d_init_var_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start);
void d_update_hessian_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, double **t, double **tinv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **qx2, double **bd, double **bl, double **pd, double **pl, double **db);
void d_update_gradient_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2, double **qx);
void d_compute_alpha_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db);
void d_update_var_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_update_var_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_compute_mu_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt);

// size-variant soft-constrained MPC
void d_init_var_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **ux, double **pi, double **pDCt, double **db, double **t, double **lam, double mu0, int warm_start);
void d_update_hessian_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double sigma_mu, double **t, double **tinv, double **lam, double **lamt, double **dlam, double **Qx, double **qx, double **qx2, double **bd, double **bl, double **pd, double **pl, double **db, double **Z, double **z, double **Zl, double **zl);
void d_update_gradient_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double sigma_mu, double **dt, double **dlam, double **t_inv, double **lamt, double **pl2, double **qxr, double **Zl, double **zl);
void d_compute_alpha_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **pDCt, double **db, double **Zl, double **zl);
void d_update_var_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_compute_mu_mpc_soft_tv(int N, int *nx, int *nu, int *nb, int *ng, int *ns, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt);
