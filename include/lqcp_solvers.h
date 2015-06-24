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

// work space size
int d_ric_sv_mpc_tv_work_space_size_double(int N, int *nx, int *nu, int *nb, int *ng);
int d_cond_lqcp_work_space(int N, int nx, int nu, int N2);

// MPC
void d_ric_sv_mpc_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi, int *nb, int **idxb, double **hQd, double **hQl, int *ng, double **hpDCt, double **Qx, double **qx, int fast_rsqrt);
void d_ric_trs_mpc_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpL, double **hq, double **hl, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi, int *nb, int **idxb, double **hql, int *ng, double **hpDCt, double **qx);
void d_res_mpc_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb);
void d_ric_sv_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, int update_hessian, double **hQd, double **hQl, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi, int nb, int ng, int ngN, double **hDCt, double **Qx, double **qx, int fast_rsqrt);
void d_ric_trs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi, int nb, int ng, int ngN, double **hpDCt, double **qx);
void d_res_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb);
void d_ric_diag_trf_mpc(int N, int *nx, int *nu, double **hdA, double **hpBt, double **hpR, double **hpSt, double **hpQ, double **hpL, double *pK, double **hpP, double *work);
void d_ric_diag_trs_mpc(int N, int *nx, int *nu, double **hdA, double **hpBt, double **hpL, double **hpP, double **hb, double **hrq, double **hux, int compute_Pb, double **hPb, int compute_pi, double **hp, double *work);
void d_res_diag_mpc(int N, int *nx, int *nu, double **hdA, double **hpBt, double **hpR, double **hpSt, double **hpQ, double **hb, double **hrq, double **hux, double **hpi, double **hres_rq, double **hres_b, double *work);
void s_ric_sv_mpc(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hux, float **hpL, float *work, float *diag, int compute_pi, float **hpi);
void s_ric_trs_mpc(int nx, int nu, int N, float **hpBAbt, float **hpL, float **hq, float **hux, float *work, int compute_Pb, float ** hPb, int compute_pi, float **hpi);
void s_res_mpc(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hq, float **hux, float **hpi, float **hrq, float **hrb);

// MHE
void d_ric_trf_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hdLp, double **hpQ, double **hpR, double **hpLe, double *work);
int d_ric_trs_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hdLp, double **hpQ, double **hpR, double **hpLe, double **hq, double **hr, double **hf, double **hxp, double **hxe, double **hw, double **hy, int smooth, double **hlam,  double *work);
void d_ric_trf_mhe_end(int nx, int nw, int ny, int N, double **hpCA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double *work);
void d_ric_trs_mhe_end(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double **hq, double **hr, double **hf, double **hxp, double **hxe, double **hy, double *work);
int d_ric_trf_mhe_if(int nx, int nw, int ndN, int N, double **hpQRAG, int diag_R, double **hpLe, double **hpLAG, double *Ld, double *work);
void d_ric_trs_mhe_if(int nx, int nw, int ndN, int N, double **hpLe, double **hpLAG, double *Ld, double **hr, double **hq, double **hf, double **hxp, double **hx, double **hw, double **hlam, double *work);
void d_res_mhe_if(int nx, int nw, int ndN, int N, double **hpQA, double **hpRG, double *L0_inv, double **hq, double **hr, double **hf, double *p0, double **hx, double **hw, double **hlam, double **hrq, double **hrr, double **hrf, double *work);

// condensing

void d_cond_Q(int N, int nx, int nu, double **pA, int diag_Q, int nzero_Q_N, double **pQ, double **pL, int compute_Gamma_0, double **pGamma_0, double **pGamma_0_Q, double *pH_Q, double *work);
void d_cond_R(int N, int nx, int nu, int N2_cond, double **pA, double **pAt, double **pBt, int diag_Q, int nzero_Q_N, double **pQ, double **pL, int nzero_S, double **pS, double **pR, int compute_Gamma_u, double **pGamma_u, double **pGamma_u_Q, double **pGamma_u_Q_A, double *pH_R);
void d_cond_St(int N, int nx, int nu, int nzero_S, double **pS, int nzero_Q_N, double **pGamma_0, int use_pGamma_0_Q, double **pGamma_0_Q, double **pGamma_u_Q, double *pH_St);
void d_cond_q(int N, int nx, int nu, double **pA, double **b, int diag_Q, int nzero_Q_N, double **pQ, double **q, double **pGamma_0, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_q);
void d_cond_r(int N, int nx, int nu, double **pA, double **b, int diag_Q, int nzero_Q_N, double **pQ, int nzero_S, double **pS, double **q, double **r, double **pGamma_u, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_r);
void d_cond_A(int N, int nx, int nu, double **pA, int compute_Gamma_0, double **pGamma_0, double *pH_A);
void d_cond_B(int N, int nx, int nu, double **pA, double **pB, int compute_Gamma_u, double **pGamma_u, double *pH_B);
void d_cond_b(int N, int nx, int nu, double **pA, double **b, int compute_Gamma_b, double **Gamma_b, double *H_b);
void d_cond_lqcp(int N, int nx, int nu, double **hpA, double **hpAt, double **hpBt, double **hb, double **hpR, int nzero_S, double **hpS, int diag_Q, double **hpQ, double **hr, double **hq, int N2, int *nx2, int *nu2, double **hpA2, double **hpB2, double **hb2, double **hpR2, double **hpSt2, double **hpQ2, double **hr2, double **hq2, double *work_double, int N2_cond);
