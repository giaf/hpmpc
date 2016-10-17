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

#include <stdlib.h>
#include <stdio.h>

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_i_aux.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>
#include <blasfeo_d_blas.h>



void d_back_ric_rec_sv_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_q, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strvec *hsdRSQ, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat, struct d_strvec *hswork_vec)
	{

	int nn;

	// factorization and backward substitution

	// last stage
	if(update_q)
		{
		drowin_libstr(nx[N], 1.0, &hsrq[N], 0, &hsRSQrq[N], nx[N], 0);
		}
	if(nb[N]>0)
		{
		ddiaadin_libspstr(nb[N], hidxb[N], 1.0, &hsQx[N], 0, &hsdRSQ[N], 0, &hsRSQrq[N], 0, 0);
		drowad_libspstr(nb[N], hidxb[N], 1.0, &hsqx[N], 0, &hsRSQrq[N], nx[N], 0);
		}
	if(ng[N]>0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	else
		{
		dpotrf_l_libstr(nx[N]+1, nx[N], &hsRSQrq[N], 0, 0, &hsL[N], 0, 0);
		}
	dtrtr_l_libstr(nx[N], &hsL[N], 0, 0, &hsLxt[N], 0, 0);

	// middle stages
	for(nn=0; nn<N; nn++)
		{
		if(update_b)
			{
			drowin_libstr(nx[N-nn], 1.0, &hsb[N-nn], 0, &hsBAbt[N-nn], nu[N-nn-1]+nu[N-nn-1], 0);
			}
		dtrmm_rutn_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], 1.0, &hsBAbt[N-nn], 0, 0, &hsLxt[N-nn], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
		if(compute_Pb)
			{
			drowex_libstr(nx[N-nn], 1.0, &hswork_mat[0], nu[N-nn-1]+nx[N-nn-1], 0, &hswork_vec[0], 0);
			dtrmv_utn_libstr(nx[N-nn], &hsLxt[N-nn], 0, 0, &hswork_vec[0], 0, &hsPb[N-nn], 0);
			}
		dgead_libstr(1, nx[N-nn], 1.0, &hsL[N-nn], nu[N-nn]+nx[N-nn], nu[N-nn], &hswork_mat[0], nu[N-nn-1]+nx[N-nn-1], 0);
		if(update_q)
			{
			drowin_libstr(nu[N-nn-1]+nx[N-nn-1], 1.0, &hsrq[N-nn-1], 0, &hsRSQrq[N-nn-1], nu[N-nn-1]+nx[N-nn-1], 0);
			}
		if(nb[N-nn-1]>0)
			{
			ddiaadin_libspstr(nb[N-nn-1], hidxb[N-nn-1], 1.0, &hsQx[N-nn-1], 0, &hsdRSQ[N-nn-1], 0, &hsRSQrq[N-nn-1], 0, 0);
			drowad_libspstr(nb[N-nn-1], hidxb[N-nn-1], 1.0, &hsqx[N-nn-1], 0, &hsRSQrq[N-nn-1], nu[N-nn-1]+nx[N-nn-1], 0);
			}
		if(ng[N-nn-1]>0)
			{
			printf("\nfeature not implemented yet\n\n");
			exit(1);
			}
		else
			{
			dsyrk_dpotrf_ln_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], nx[N-nn], &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, &hsRSQrq[N-nn-1], 0, 0, &hsL[N-nn-1], 0, 0);
			}
		dtrtr_l_libstr(nx[N-nn-1], &hsL[N-nn-1], nu[N-nn-1], nu[N-nn-1], &hsLxt[N-nn-1], 0, 0);
		}
	
	// forward substitution

	// first stage
	nn = 0;
	drowex_libstr(nu[nn]+nx[nn], -1.0, &hsL[nn], nu[nn]+nx[nn], 0, &hsux[nn], 0);
	dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn]+nx[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
	drowex_libstr(nx[nn+1], 1.0, &hsBAbt[nn+1], nu[nn]+nx[nn], 0, &hsux[nn+1], nu[nn+1]);
	dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn+1], 0, 0, &hsux[nn], 0, 1.0, &hsux[nn+1], nu[nn+1], &hsux[nn+1], nu[nn+1]);
	if(compute_pi)
		{
		dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hspi[nn+1], 0);
		drowex_libstr(nx[nn+1], 1.0, &hsL[nn+1], nu[nn+1]+nx[nn+1], nu[nn+1], &hswork_vec[0], 0);
		dtrmv_unn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hspi[nn+1], 0, &hspi[nn+1], 0);
		daxpy_libstr(nx[nn+1], 1.0, &hswork_vec[0], 0, &hspi[nn+1], 0);
		dtrmv_utn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hspi[nn+1], 0, &hspi[nn+1], 0);
		}

	// middle stages
	for(nn=1; nn<N; nn++)
		{
		drowex_libstr(nu[nn], -1.0, &hsL[nn], nu[nn]+nx[nn], 0, &hsux[nn], 0);
		dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
		drowex_libstr(nx[nn+1], 1.0, &hsBAbt[nn+1], nu[nn]+nx[nn], 0, &hsux[nn+1], nu[nn+1]);
		dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn+1], 0, 0, &hsux[nn], 0, 1.0, &hsux[nn+1], nu[nn+1], &hsux[nn+1], nu[nn+1]);
		if(compute_pi)
			{
			dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hspi[nn+1], 0);
			drowex_libstr(nx[nn+1], 1.0, &hsL[nn+1], nu[nn+1]+nx[nn+1], nu[nn+1], &hswork_vec[0], 0);
			dtrmv_unn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hspi[nn+1], 0, &hspi[nn+1], 0);
			daxpy_libstr(nx[nn+1], 1.0, &hswork_vec[0], 0, &hspi[nn+1], 0);
			dtrmv_utn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hspi[nn+1], 0, &hspi[nn+1], 0);
			}
		}

	return;

	}



void d_back_ric_trf_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat)
	{

	int nn;

	// factorization

	// last stage
	if(nb[N]>0)
		{
		ddiaadin_libspstr(nb[N], hidxb[N], 1.0, &hsQx[N], 0, &hsdRSQ[N], 0, &hsRSQrq[N], 0, 0);
		}
	if(ng[N]>0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	else
		{
		dpotrf_l_libstr(nx[N], nx[N], &hsRSQrq[N], 0, 0, &hsL[N], 0, 0);
		}
	dtrtr_l_libstr(nx[N], &hsL[N], 0, 0, &hsLxt[N], 0, 0);

	// middle stages
	for(nn=0; nn<N; nn++)
		{
		dtrmm_rutn_libstr(nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &hsBAbt[N-nn], 0, 0, &hsLxt[N-nn], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
		if(nb[N-nn-1]>0)
			{
			ddiaadin_libspstr(nb[N-nn-1], hidxb[N-nn-1], 1.0, &hsQx[N-nn-1], 0, &hsdRSQ[N-nn-1], 0, &hsRSQrq[N-nn-1], 0, 0);
			}
		if(ng[N-nn-1]>0)
			{
			printf("\nfeature not implemented yet\n\n");
			exit(1);
			}
		else
			{
			dsyrk_dpotrf_ln_libstr(nu[N-nn-1]+nx[N-nn-1], nu[N-nn-1]+nx[N-nn-1], nx[N-nn], &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, &hsRSQrq[N-nn-1], 0, 0, &hsL[N-nn-1], 0, 0);
			}
		dtrtr_l_libstr(nx[N-nn-1], &hsL[N-nn-1], nu[N-nn-1], nu[N-nn-1], &hsLxt[N-nn-1], 0, 0);
		}
	
	return;

	}



void d_back_ric_rec_trs_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hswork_vec)
	{

	int nn;

	// backward substitution

	// last stage
	dveccp_libstr(nu[N]+nx[N], 1.0, &hsrq[N], 0, &hsux[N], 0);
	if(nb[N]>0)
		{
		dvecad_libspstr(nb[N], idxb[N], 1.0, &hsqx[N], 0, &hsux[N], 0);
		}
	// general constraints
	if(ng[N]>0)
		{
		dgemv_n_libstr(nx[N], ng[N], 1.0, &hsDCt[N], 0, 0, &hsqx[N], nb[N], 1.0, &hsux[N], 0, &hsux[N], 0);
		}

	// middle stages
	for(nn=0; nn<N-1; nn++)
		{
		if(compute_Pb)
			{
			dtrmv_unn_libstr(nx[N-nn], &hsLxt[N-nn], 0, 0, &hsb[N-nn], 0, &hsPb[N-nn], 0);
			dtrmv_utn_libstr(nx[N-nn], &hsLxt[N-nn], 0, 0, &hsPb[N-nn], 0, &hsPb[N-nn], 0);
			}
		dveccp_libstr(nu[N-nn-1]+nx[N-nn-1], 1.0, &hsrq[N-nn-1], 0, &hsux[N-nn-1], 0);
		if(nb[N-nn-1]>0)
			{
			dvecad_libspstr(nb[N-nn-1], idxb[N-nn-1], 1.0, &hsqx[N-nn-1], 0, &hsux[N-nn-1], 0);
			}
		if(ng[N-nn-1]>0)
			{
			dgemv_n_libstr(nu[N-nn-1]+nx[N-nn-1], ng[N-nn-1], 1.0, &hsDCt[N-nn-1], 0, 0, &hsqx[N-nn-1], nb[N-nn-1], 1.0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);
			}
		dveccp_libstr(nx[N-nn], 1.0, &hsPb[N-nn-1], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx[N-nn], 1.0, &hsux[N-nn], nu[N-nn], &hswork_vec[0], 0);
		dgemv_n_libstr(nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &hsBAbt[N-nn], 0, 0, &hswork_vec[0], 0, 1.0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);
		dtrsv_lnn_libstr(nu[N-nn-1]+nx[N-nn-1], nu[N-nn-1], &hsL[N-nn-1], 0, 0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);
		}

	// first stage
	nn = N-1;
	if(compute_Pb)
		{
		dtrmv_unn_libstr(nx[N-nn], &hsLxt[N-nn], 0, 0, &hsb[N-nn], 0, &hsPb[N-nn], 0);
		dtrmv_utn_libstr(nx[N-nn], &hsLxt[N-nn], 0, 0, &hsPb[N-nn], 0, &hsPb[N-nn], 0);
		}
	dveccp_libstr(nu[N-nn-1]+nx[N-nn-1], 1.0, &hsrq[N-nn-1], 0, &hsux[N-nn-1], 0);
	if(nb[N-nn-1]>0)
		{
		dvecad_libspstr(nb[N-nn-1], idxb[N-nn-1], 1.0, &hsqx[N-nn-1], 0, &hsux[N-nn-1], 0);
		}
	if(ng[N-nn-1]>0)
		{
		dgemv_n_libstr(nu[N-nn-1]+nx[N-nn-1], ng[N-nn-1], 1.0, &hsDCt[N-nn-1], 0, 0, &hsqx[N-nn-1], nb[N-nn-1], 1.0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);
		}
	dveccp_libstr(nx[N-nn], 1.0, &hsPb[N-nn], 0, &hswork_vec[0], 0);
	daxpy_libstr(nx[N-nn], 1.0, &hsux[N-nn], nu[N-nn], &hswork_vec[0], 0);
	dgemv_n_libstr(nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &hsBAbt[N-nn], 0, 0, &hswork_vec[0], 0, 1.0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);
	dtrsv_lnn_libstr(nu[N-nn-1]+nx[N-nn-1], nu[N-nn-1]+nx[N-nn-1], &hsL[N-nn-1], 0, 0, &hsux[N-nn-1], 0, &hsux[N-nn-1], 0);

	// forward substitution

	// first stage
	nn = 0;
	if(compute_pi)
		{
		dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hspi[nn+1], 0);
		}
	dveccp_libstr(nu[nn]+nx[nn], -1.0, &hsux[nn], 0, &hsux[nn], 0);
	dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn]+nx[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
	dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn+1], 0, 0, &hsux[nn], 0, 1.0, &hsb[nn+1], 0, &hsux[nn+1], nu[nn+1]);
	if(compute_pi)
		{
		dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hswork_vec[0], 0);
		dtrmv_unn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		dtrmv_utn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx[nn+1], 1.0, &hswork_vec[0], 0, &hspi[nn+1], 0);
		}

	// middle stages
	for(nn=1; nn<N; nn++)
		{
		if(compute_pi)
			{
			dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hspi[nn+1], 0);
			}
		dveccp_libstr(nu[nn], -1.0, &hsux[nn], 0, &hsux[nn], 0);
		dtrsv_ltn_libstr(nu[nn]+nx[nn], nu[nn], &hsL[nn], 0, 0, &hsux[nn], 0, &hsux[nn], 0);
		dgemv_t_libstr(nu[nn]+nx[nn], nx[nn+1], 1.0, &hsBAbt[nn+1], 0, 0, &hsux[nn], 0, 1.0, &hsb[nn+1], 0, &hsux[nn+1], nu[nn+1]);
		if(compute_pi)
			{
			dveccp_libstr(nx[nn+1], 1.0, &hsux[nn+1], nu[nn+1], &hswork_vec[0], 0);
			dtrmv_unn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			dtrmv_utn_libstr(nx[nn+1], &hsLxt[nn+1], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			daxpy_libstr(nx[nn+1], 1.0, &hswork_vec[0], 0, &hspi[nn+1], 0);
			}
		}

	return;

	}



