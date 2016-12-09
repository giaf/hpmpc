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

//
// backward Riccati recursion
//

// work space size
int d_back_ric_rec_sv_tv_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int *ng);
// memory space size
int d_back_ric_rec_sv_tv_memory_space_size_bytes(int N, int *nx, int *nu, int *nb, int *ng);
// backward Riccati recursion: factorization and solution
void d_back_ric_rec_sv_tv_res(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int update_b, double **hpBAbt, double **b, int update_q, double **hpQ, double **q, double **bd, double **hpDCt, double **Qx, double **qx, double **hux, int compute_pi, double **hpi, int compute_Pb, double **hPb, double *memory, double *work);
#ifdef BLASFEO
void d_back_ric_rec_sv_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_q, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat, struct d_strvec *hswork_vec);
void d_back_ric_rec_sv_back_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_q, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat, struct d_strvec *hswork_vec);
void d_back_ric_rec_sv_forw_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_q, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat, struct d_strvec *hswork_vec);
#endif
// backward Riccati recursion: factorization 
void d_back_ric_rec_trf_tv_res(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **hpBAbt, double **hpQ, double **hpDCt, double **Qx, double **bd, double *memory, double *work);
#ifdef BLASFEO
void d_back_ric_rec_trf_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat);
#endif
// tree IPM
#if defined(TREE_MPC)
#ifdef BLASFEO
void d_tree_back_ric_rec_trf_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat);
#endif
#endif
// backward Riccati recursion: solution
void d_back_ric_rec_trs_tv_res(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **hpBAbt, double **hb, double **hq, double **hpDCt, double **qx, double **hux, int compute_pi, double **hpi, int compute_Pb, double **hPb, double *memory, double *work);
#ifdef BLASFEO
void d_back_ric_rec_trs_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hswork_vec);
#endif
#if defined(TREE_MPC)
#ifdef BLASFEO
void d_tree_back_ric_rec_trs_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hswork_vec);
#endif
#endif

//
// (partial) condensing
//

// condense state space
void d_cond_BAbt(int N, int *nx, int *nu, double **hpBAbt, double *work, double **hpGamma, double *pBAbt2);
// condense Hessian and gradient (N^2 n_x^3 algorithm)
void d_cond_RSQrq(int N, int *nx, int *nu, double **hpBAbt, double **hpRSQrq, double **hpGamma, double *work, double *pRSQrq2);
// condense constraints (TODO general constraints)
void d_cond_DCtd(int N, int *nx, int *nu, int *nb, int **hidxb, double **hd, double **hpGamma, double *pDCt2, double *d2, int *idxb2);
// computes problem size (not hidxb2)
void d_part_cond_compute_problem_size(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2);
// work space for partially condensing routine
int d_part_cond_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2);
// work space for partially condensing routine
int d_part_cond_memory_space_size_bytes(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2);
// partial condensing routine
void d_part_cond(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, double **hpBAbt, double **hpRSQrq, double **hpDCt, double **hd, int N2, int *nx2, int *nu2, int *nb2, int **hidxb2, int *ng2, double **hpBAbt2, double **hpRSQrq2, double **hpDCt2, double **hd2, void *memory, void *work);
// work space for partial expand
int d_part_expand_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int *ng);
// partial expand routine (recovers the solution to the full space problem)
void d_part_expand_solution(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, double **hpBAbt, double **hb, double **hpRSQrq, double **hrq, double **hpDCt, double **hux, double **hpi, double **hlam, double **ht, int N2, int *nx2, int *nu2, int *nb2, int **hidxb2, int *ng2, double **hux2, double **hpi2, double **hlam2, double **ht2, void *work);
