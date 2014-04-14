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
void kernel_sgemm_pp_nt_4x4_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x3_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x2_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_sgemm_pp_nt_4x1_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg);
void kernel_spotrf_strsv_scopy_4x4_c99_lib4(int kmax, float *A, int sda, int shf, float *L, int sdl);
void kernel_spotrf_strsv_4x4_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_3x3_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_2x2_c99_lib4(int kmax, float *A, int sda);
void kernel_spotrf_strsv_1x1_c99_lib4(int kmax, float *A, int sda);
void kernel_sgemv_t_8_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_4_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_2_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_t_1_c99_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg);
void kernel_sgemv_n_8_c99_lib4(int kmax, float *A0, float *A1, float *x, float *y, int alg);
void kernel_sgemv_n_4_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
void kernel_sgemv_n_2_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
void kernel_sgemv_n_1_c99_lib4(int kmax, float *A, float *x, float *y, int alg);
//// corner
void corner_strmm_pp_nt_4x3_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_strmm_pp_nt_4x2_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_strmm_pp_nt_4x1_c99_lib4(float *A, float *B, float *C, int ldc);
void corner_spotrf_strsv_scopy_3x3_c99_lib4(float *A, int sda, int shf, float *L, int sdl);
void corner_spotrf_strsv_scopy_2x2_c99_lib4(float *A, int sda, int shf, float *L, int sdl);
void corner_spotrf_strsv_scopy_1x1_c99_lib4(float *A, int sda, int shf, float *L, int sdl);

