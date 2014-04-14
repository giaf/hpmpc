/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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
void kernel_dgemm_pp_nt_8x4_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_8x2_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_8x1_avx_lib4(int kmax, double *A0, double *A1, double *B, double *C0, double *C1, int ldc, int alg);
void kernel_dgemm_pp_nt_4x4_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_4x3_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_4x2_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_4x1_avx_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(int kmax, double *A, int sda, int shf, double *L, int sdl);
void kernel_dpotrf_dtrsv_4x4_sse_lib4(int kmax, double *A, int sda);
void kernel_dpotrf_dtrsv_3x3_sse_lib4(int kmax, double *A, int sda);
void kernel_dpotrf_dtrsv_2x2_sse_lib4(int kmax, double *A, int sda);
void kernel_dpotrf_dtrsv_1x1_sse_lib4(int kmax, double *A, int sda);
void kernel_dgemv_t_8_avx_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_4_avx_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_2_avx_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_1_avx_lib4(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_n_8_avx_lib4(int kmax, double *A0, double *A1, double *x, double *y, int alg);
void kernel_dgemv_n_4_avx_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_2_avx_lib4(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_1_avx_lib4(int kmax, double *A, double *x, double *y, int alg);
// corner
void corner_dtrmm_pp_nt_8x3_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc);
void corner_dtrmm_pp_nt_8x2_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc);
void corner_dtrmm_pp_nt_8x1_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc);
void corner_dtrmm_pp_nt_4x3_avx_lib4(double *A, double *B, double *C, int ldc);
void corner_dtrmm_pp_nt_4x2_avx_lib4(double *A, double *B, double *C, int ldc);
void corner_dtrmm_pp_nt_4x1_avx_lib4(double *A, double *B, double *C, int ldc);
void corner_dpotrf_dtrsv_dcopy_3x3_sse_lib4(double *A, int sda, int shf, double *L, int sdl);
void corner_dpotrf_dtrsv_dcopy_2x2_sse_lib4(double *A, int sda, int shf, double *L, int sdl);
void corner_dpotrf_dtrsv_dcopy_1x1_sse_lib4(double *A, int sda, int shf, double *L, int sdl);

