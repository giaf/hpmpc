/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *
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
void d_init_var_box_mpc(int N, int nx, int nu, int nb, double **ux, double **pi, double **db, double **t, double **lam, int warm_start);
void d_update_hessian_box_mpc(int N, int k0, int k1, int kmax, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db);
void d_update_jacobian_box_mpc(int N, int nx, int nu, int nb, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2);
void d_compute_alpha_box_mpc(int N, int nx, int nu, int nb, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db);
void d_update_var_box_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_compute_mu_box_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt);
void s_init_ux_pi_t_box_mpc(int N, int nx, int nu, int nbu, int nb, float **ux, float **pi, float **db, float **t, int warm_start);
void s_init_lam_mpc(int N, int nu, int nbu, int nb, float **t, float **lam);
void s_update_hessian_box_mpc(int N, int k0, int k1, int kmax, int cnz, float sigma_mu, float **t, float **t_inv, float **lam, float **lamt, float **dlam, float **bd, float **bl, float **pd, float **pl, float **pl2, float **db);
void s_compute_alpha_box_mpc(int N, int k0, int k1, int kmax, float *ptr_alpha, float **t, float **dt, float **lam, float **dlam, float **lamt, float **dux, float **db);
void s_update_var_mpc(int nx, int nu, int N, int nb, int nbu, float *ptr_mu, float mu_scal, float alpha, float **ux, float **dux, float **t, float **dt, float **lam, float **dlam, float **pi, float **dpi);
void s_compute_mu_mpc(int N, int nbu, int nu, int nb, float *ptr_mu, float mu_scal, float alpha, float **lam, float **dlam, float **t, float **dt);

// soft-constrained MPC
void d_init_var_soft_mpc(int N, int nx, int nu, int nb, double **ux, double **pi, double **db, double **t, double **lam, int warm_start);
void d_update_hessian_soft_mpc(int N, int nx, int nu, int nb, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db, double **Z, double **z, double **Zl, double **zl);
void d_update_jacobian_soft_mpc(int N, int nx, int nu, int nb, double sigma_mu, double **dt, double **dlam, double **t_inv, double **lamt, double **pl2, double **Zl, double **zl);
void d_compute_alpha_soft_mpc(int N, int nx, int nu, int nb, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db, double **Zl, double **zl);
void d_update_var_soft_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_compute_mu_soft_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt);

// MHE wrong
void d_init_ux_pi_t_box_mhe_old(int N, int nx, int nu, int nbu, int nb, double **ux, double **pi, double **db, double **t, int warm_start);
void d_init_lam_mhe_old(int N, int nu, int nbu, int nb, double **t, double **lam);
void d_update_hessian_box_mhe_old(int N, int k0, int k1, int kmax, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db);
void d_compute_alpha_box_mhe_old(int N, int k0, int k1, int kmax, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db);
void d_update_var_mhe_old(int nx, int nu, int N, int nb, int nbu, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi);
void d_compute_mu_mhe_old(int N, int nbu, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt);
void s_init_ux_pi_t_box_mhe_old(int N, int nx, int nu, int nbu, int nb, float **ux, float **pi, float **db, float **t, int warm_start);
void s_init_lam_mhe_old(int N, int nu, int nbu, int nb, float **t, float **lam);
void s_update_hessian_box_mhe_old(int N, int k0, int k1, int kmax, int cnz, float sigma_mu, float **t, float **t_inv, float **lam, float **lamt, float **dlam, float **bd, float **bl, float **pd, float **pl, float **pl2, float **db);
void s_compute_alpha_box_mhe_old(int N, int k0, int k1, int kmax, float *ptr_alpha, float **t, float **dt, float **lam, float **dlam, float **lamt, float **dux, float **db);
void s_update_var_mhe_old(int nx, int nu, int N, int nb, int nbu, float *ptr_mu, float mu_scal, float alpha, float **ux, float **dux, float **t, float **dt, float **lam, float **dlam, float **pi, float **dpi);
void s_compute_mu_mhe_old(int N, int nbu, int nu, int nb, float *ptr_mu, float mu_scal, float alpha, float **lam, float **dlam, float **t, float **dt);
