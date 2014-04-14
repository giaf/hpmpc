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
void kernel_dgemm_pp_nt_2x2_atom_lib2(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dgemm_pp_nt_2x1_c99_lib2(int kmax, double *A, double *B, double *C, int ldc, int alg);
void kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(int kmax, double *A, int sda, int shf, double *L, int sdl);
void kernel_dpotrf_dtrsv_2x2_c99_lib2(int kmax, double *A, int sda);
void kernel_dpotrf_dtrsv_1x1_c99_lib2(int kmax, double *A, int sda);
void kernel_dgemv_t_2_c99_lib2(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_t_1_c99_lib2(int kmax, int kna, double *A, int sda, double *x, double *y, int alg);
void kernel_dgemv_n_2_c99_lib2(int kmax, double *A, double *x, double *y, int alg);
void kernel_dgemv_n_1_c99_lib2(int kmax, double *A, double *x, double *y, int alg);
//// corner
void corner_dtrmm_pp_nt_2x1_c99_lib2(double *A, double *B, double *C, int ldc);
void corner_dpotrf_dtrsv_dcopy_1x1_c99_lib2(double *A, int sda, int shf, double *L, int sdl);

