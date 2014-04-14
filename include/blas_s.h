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

void sgemm_ppp_nt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, int alg);
void strmm_ppp_lib(int m, int n, int offset, float *pA, int sda, float *pB, int sdb, float *pC, int sdc);
//void dsyrk_ppp_lib(int n, int m, float *pA, int sda, float *pC, int sdc);
void ssyrk_ppp_lib(int n, int m, int k, float *pA, int sda, float *pC, int sdc);
void spotrf_p_lib(int m, int n, float *pC, int sdc);
void spotrf_p_scopy_p_t_lib(int n, int nna, float *pC, int sdc, float *pL, int sdl);
void sgemv_p_n_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg);
void sgemv_p_t_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg);
void strmv_p_n_lib(int m, int offset, float *pA, int sda, float *x, float *y);
void strmv_p_t_lib(int m, int offset, float *pA, int sda, float *x, float *y);
void ssymv_p_lib(int m, float *pA, int sda, float *x, float *y);
void strsv_p_n_lib(int n, float *pA, int sda, float *x);
void strsv_p_t_lib(int n, float *pA, int sda, float *x);

