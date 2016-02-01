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

void dgemm_kernel_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, int tc, int td);
void dgemm_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, int tc, int td);
void dgemm_nn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, int tc, int td);
void dtrmm_nt_u_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dtrmm_nt_l_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dsyrk_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dsyrk_nn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dpotrf_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
//void dsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *diag, int alg);
void dsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
void dgetrf_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
void dgetrf_pivot_lib(int m, int n, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D, int *ipiv);
void dgemv_n_lib(int n, int m, double *pA, int sda, double *x, int alg, double *y, double *z);
void dgemv_t_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z);
void dtrmv_u_n_lib(int m, double *pA, int sda, double *x, int alg, double *y);
void dtrmv_u_t_lib(int m, double *pA, int sda, double *x, int alg, double *y);
void dsymv_lib(int m, int n, double *pA, int sda, double *x, int alg, double *y, double *z);
void dgemv_nt_lib(int m, int n, double *pA, int sda, double *x_n, double *x_t, int alg, double *y_n, double *y_t, double *z_n, double *z_t);
void dtrsv_n_lib(int m, int n, double *pA, int sda, int use_inv_diag_A, double *inv_diag_A, double *x, double *y);
void dtrsv_t_lib(int m, int n, double *pA, int sda, int use_inv_diag_A, double *inv_diag_A, double *x, double *y);
void dgecp_lib(int m, int n, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrcp_l_lib(int m, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dgead_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dgetr_lib(int m, int n, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_l_lib(int m, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_u_lib(int m, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dsyttmm_lu_lib(int m, double *pA, int sda, double *pC, int sdc);
void dsyttmm_ul_lib(int m, double *pA, int sda, int alg, double *pC, int sdc, double *pD, int sdd);
void dttmm_uu_lib(int m, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dttmm_ll_lib(int m, double *pA, int sda, double *pB, int sdb, double *pC, int sdc);
void dtrinv_lib(int m, double *pA, int sda, double *pC, int sdc);
//void dsyrk_dpotrf_dtrinv_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pE, int sde, double *diag, int alg);
void dsyrk_dpotrf_dtrinv_lib(int m, int n, int k, double *pA, int sda, int alg, double *pC, int sdd, double *pD, int sdc, double *pE, int sde, double *diag);
//void dtsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *diag, int alg);
void dtsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, int alg, double *pC, int sdc, double *pD, int sdd, double *diag);
void dgemm_diag_right_lib(int m, int n, double *pA, int sda, double *dB, int alg, double *pC, int sdc, double *pD, int sdd);
void dgemm_diag_left_lib(int m, int n, double *dA, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dsyrk_diag_left_right_lib(int m, double *Al, double *Ar, double *B, int sdb, int alg, double *C, int sdc, double *D, int sdd);
void dgemv_diag_lib(int m, double *dA, double *x, int alg, double *y, double *z);

// auxiliary routines
void ddiareg_lib(int kmax, double reg, int offset, double *pD, int sdd);
void ddiain_lib(int kmax, double *x, int offset, double *pD, int sdd);
void ddiain_sqrt_lib(int kmax, double *x, int offset, double *pD, int sdd);
void ddiaex_lib(int kmax, int offset, double *pD, int sdd, double *x);
void ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void ddiain_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
void ddiaad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
void drowin_lib(int kmax, double *x, double *pD);
void drowex_lib(int kmax, double *pD, double *x);
void drowad_lib(int kmax, double alpha, double *x, double *pD);
void drowin_libsp(int kmax, int *idx, double *x, double *pD);
void drowad_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
void drowsw_lib(int kmax, double *pA, double *pC);
void dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
void dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
void dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);




// new routines
void dpotrf_lib_old(int m, int n, double *pD, int sdd, double *pC, int sdc, double *diag);
void dsyrk_dpotrf_lib_old(int m, int n, int k, double *pA, int sda, int alg, double *pC, int sdc, double *pD, int sdd, double *diag, int fast_rsqrt);
void dtrsv_n_lib_old(int m, int n, int inverted_diag, double *pA, int sda, double *x);
void dtrsv_t_lib_old(int m, int n, int inverted_diag, double *pA, int sda, double *x);
void dtrtri_lib(int m, double *pA, int sda, int use_inv_diag_A, double *inv_diag_A, double *pC, int sdc);
void dlauum_lib(int m, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd);
void dlauum_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, int alg, double *pC, int sdc, double *pD, int sdd, double *inv_diag_D);
