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

void d_ip_box(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **lb, double **ub, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work, int *info);
void d_ip_box2(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **lb, double **ub, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work, int *info);
void dres_ip_box(int nx, int nu, int N, int nb, int sda, double **hpBAbt, double **hpQ, double **hq, double **hux, double **lb, double **ub, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu);
