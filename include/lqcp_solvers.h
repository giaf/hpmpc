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

void d_ric_sv_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi);
void d_ric_sv_mhe(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi);
void d_ric_trs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_pi, double **hpi);
void d_ric_trs_mhe(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_pi, double **hpi);
void d_res_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb);
void d_res_mhe(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb);
void s_ric_sv_mpc(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hux, float **hpL, float *work, float *diag, int compute_pi, float **hpi);
void s_ric_sv_mhe(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hux, float **hpL, float *work, float *diag, int compute_pi, float **hpi);
void s_ric_trs_mpc(int nx, int nu, int N, float **hpBAbt, float **hpL, float **hq, float **hux, float *work, int compute_pi, float **hpi);
void s_ric_trs_mhe(int nx, int nu, int N, float **hpBAbt, float **hpL, float **hq, float **hux, float *work, int compute_pi, float **hpi);
void s_res_mpc(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hq, float **hux, float **hpi, float **hrq, float **hrb);
void s_res_mhe(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hq, float **hux, float **hpi, float **hrq, float **hrb);

