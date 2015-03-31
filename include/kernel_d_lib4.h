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

// kernel
void kernel_dgemm_nt_12x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td);
void kernel_dgemm_nt_8x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td);
void kernel_dgemm_nt_8x2_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td);
void kernel_dgemm_nt_4x4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dgemm_nt_4x2_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dgemm_nt_2x4_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dgemm_nt_2x2_lib4(int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dgemm_nt_m8x4_lib4(int m, int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td);
void kernel_dgemm_nt_m8x2_lib4(int m, int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg, int tc, int td);
void kernel_dgemm_nt_m4x4_lib4(int m, int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dgemm_nt_m4x2_lib4(int m, int kmax, double *A, double *B, double *C, double *D, int alg, int tc, int td);
void kernel_dtrmm_l_nt_12x4_lib4(int kadd, double *A0, int sda, double *B, double *D0, int sdd);
void kernel_dtrmm_l_nt_8x4_lib4(int kadd, double *A0, int sda, double *B, double *D0, int sdd);
void kernel_dtrmm_l_nt_4x4_lib4(int kadd, double *A, double *B, double *D);
void kernel_dtrmm_u_nt_8x4_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc);
void kernel_dtrmm_u_nt_8x2_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc);
void kernel_dtrmm_u_nt_4x4_lib4(int kmax, double *A, double *B, double *C);
void kernel_dtrmm_u_nt_4x2_lib4(int kmax, double *A, double *B, double *C);
void kernel_dtrmm_u_nt_2x4_lib4(int kmax, double *A, double *B, double *C);
void kernel_dtrmm_u_nt_2x2_lib4(int kmax, double *A, double *B, double *C);
void kernel_dtrmm_l_u_nt_4x4_lib4(int kmax, double *A, double *B, double *C);
void kernel_dsyrk_nt_12x4_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg);
void kernel_dsyrk_nt_8x4_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg);
void kernel_dsyrk_nt_8x2_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg);
void kernel_dsyrk_nt_4x4_lib4(int kadd, double *A, double *B, double *C, double *D, int alg);
void kernel_dsyrk_nt_4x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg);
void kernel_dsyrk_nt_2x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg);
void kernel_dsyrk_dpotrf_nt_12x4_lib4(int tri, int kadd, int ksub, double *Ap, int sdap, double *Bp, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *fact, int alg);
void kernel_dsyrk_dpotrf_nt_8x4_lib4(int tri, int kadd, int ksub, double *Ap, int sdap, double *Bp, double *Am, int sdam, double *Bm, double *C, int sdc, double *D, int sdd, double *fact, int alg);
void kernel_dsyrk_dpotrf_nt_4x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dsyrk_dpotrf_nt_4x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dsyrk_dpotrf_nt_2x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_12x4_lib4(int tri, int kadd, int ksub, double *Ap, int sdap, double *Bp, double *Am, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_8x4_lib4(int tri, int kadd, int ksub, double *Ap, int sdap, double *Bp, double *Am, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_8x2_lib4(int tri, int kadd, int ksub, double *Ap, int sdap, double *Bp, double *Am, int sdam, double *Bm, double *C0, int sdc, double *D0, int sdd, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_4x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_4x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_2x4_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dgemm_dtrsm_nt_2x2_lib4(int tri, int kadd, int ksub, double *Ap, double *Bp, double *Am, double *Bm, double *C, double *D, double *fact, int alg);
void kernel_dgemv_t_8_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_4_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_2_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_1_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_n_8_lib4(int kmax, double *A0, int sda, double *x, double *y, int alg);
void kernel_dgemv_n_4_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_2_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_1_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dtrmv_u_t_8_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dtrmv_u_t_4_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dtrmv_u_t_2_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dtrmv_u_t_1_lib4(int kmax, double *A, int sda, double *x, double *y, int alg);
void kernel_dtrmv_u_n_8_lib4(int kmax, double *A0, int sda, double *x, double *y, int alg);
void kernel_dtrmv_u_n_4_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dtrmv_u_n_2_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dtrsv_n_8_lib4(int kmax, double *A0, int sda, double *x, double *y);
void kernel_dtrsv_n_4_lib4(int kmax, int ksv, double *A, double *x, double *y);
void kernel_dtrsv_t_4_lib4(int kmax, double *A, int sda, double *x);
void kernel_dtrsv_t_3_lib4(int kmax, double *A, int sda, double *x);
void kernel_dtrsv_t_2_lib4(int kmax, double *A, int sda, double *x);
void kernel_dtrsv_t_1_lib4(int kmax, double *A, int sda, double *x);
void kernel_dsymv_4_lib4(int kmax, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg);
void kernel_dsymv_3_lib4(int kmax, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg);
void kernel_dsymv_2_lib4(int kmax, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg);
void kernel_dsymv_1_lib4(int kmax, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg);
void kernel_dgetr_4_lib4(int kmax, int kna, double *A, double *C, int sdc);
void kernel_dgetr_3_lib4(int kmax, int kna, double *A, double *C, int sdc);
void kernel_dgetr_2_lib4(int kmax, int kna, double *A, double *C, int sdc);
void kernel_dgetr_1_lib4(int kmax, int kna, double *A, double *C, int sdc);
void kernel_dtran_4_lib4(int kmax, int kna, double *A, int sda, double *C);
void kernel_dsyttmm_lu_nt_4x4_lib4(int kmax, double *A, double *C);
void kernel_dsyttmm_lu_nt_2x2_lib4(int kmax, double *A, double *C);
void kernel_dsyttmm_ul_nt_4x4_lib4(int kmax, double *A, double *C);
void kernel_dttmm_ll_nt_4x4_lib4(int kmax, double *A, double *B, double *C);
void kernel_dttmm_uu_nt_4x4_lib4(int kmax, double *A, double *B, double *C);
void kernel_dttmm_uu_nt_4x2_lib4(int kmax, double *A, double *B, double *C);
void kernel_dgema_4_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc);
void kernel_dgema_3_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc);
void kernel_dgema_2_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc);
void kernel_dgema_1_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc);
void kernel_dtrma_4_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc);
void kernel_dtrtr_u_4_lib4(int kmax, double *A, double *C, int sdc);
void kernel_dtrinv_4x4_lib4(int kmax, double *A, double *B, double *C, double *fact);
void kernel_dtrinv_4x2_lib4(int kmax, double *A, double *B, double *C, double *fact);
void kernel_dtsyrk_dpotrf_nt_4x4_lib4(int kadd, int ksub, double *Ap, double *Am, double *C, double *D, double *fact, int alg);
//void kernel_dtsyrk_dpotrf_nt_4x2_lib4(int kadd, int ksub, double *Ap, double *Am, double *C, double *D, double *fact, int alg);
//void kernel_dtsyrk_dpotrf_nt_2x2_lib4(int kadd, int ksub, double *Ap, double *Am, double *C, double *D, double *fact, int alg);
void kernel_dtrmm_dtrsm_nt_4x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg);
//void kernel_dtrmm_dtrsm_nt_4x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg);
//void kernel_dtrmm_dtrsm_nt_2x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg);
//void kernel_dtrmm_dtrsm_nt_2x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, double *fact, int alg);
// corner
void corner_dtrmm_l_nt_12x3_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_12x2_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_12x1_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_8x3_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_8x2_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_8x1_lib4(double *A0, int sda, double *B, double *C0, int sdc);
void corner_dtrmm_l_nt_4x3_lib4(double *A, double *B, double *C);
void corner_dtrmm_l_nt_4x2_lib4(double *A, double *B, double *C);
void corner_dtrmm_l_nt_4x1_lib4(double *A, double *B, double *C);
void corner_dtran_3_lib4(int kna, double *A, int sda, double *C);
void corner_dtran_2_lib4(int kna, double *A, int sda, double *C);
void corner_dttmm_ll_nt_4x4_lib4(double *A, double *B, double *C);
void corner_dttmm_uu_nt_4x4_lib4(double *A, double *B, double *C);
void corner_dttmm_uu_nt_2x2_lib4(double *A, double *B, double *C);
void corner_dtrma_3_lib4(int kna, double *A, double *C, int sdc);
void corner_dtrma_2_lib4(int kna, double *A, double *C, int sdc);
void corner_dtrinv_4x4_lib4(double *fact, double *C);
void corner_dtrinv_2x2_lib4(double *fact, double *C);
