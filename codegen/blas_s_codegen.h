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

void spotrf_p_scopy_p_t_code_generator(FILE *f, int n, int nna);
void spotrf_p_code_generator(FILE *f, int m, int n);
void strmm_ppp_code_generator(FILE *f, int m, int n, int offset);
void ssyrk_ppp_code_generator(FILE *f, int m, int n, int k);
void sgemv_p_n_code_generator(FILE *f, int n, int m, int offset, int alg);
void sgemv_p_t_code_generator(FILE *f, int n, int m, int offset, int alg);
void strmv_p_n_code_generator(FILE *f, int m, int offset, int alg);
void strmv_p_t_code_generator(FILE *f, int m, int offset, int alg);
void ssymv_p_code_generator(FILE *f, int m, int offset, int alg);
void strsv_p_n_code_generator(FILE *f, int n);
void strsv_p_t_code_generator(FILE *f, int n);

