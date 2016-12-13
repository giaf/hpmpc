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

#ifdef BLASFEO

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_i_aux.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>
#include <blasfeo_d_blas.h>

#include "../include/tree.h"



// help routines

void d_back_ric_trf_funnel1_libstr(int nkids, int nx0, int nx1, int nu0, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt0, struct d_strmat *hsLxt1, struct d_strmat *hswork_mat)
	{

	int ii;

	// first kid: initialize with hessian
	ii = 0;
	dtrmm_rutn_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hsLxt1[ii], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
	if(nb0>0)
		{
		ddiaadin_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsdRSQ[0], 0, &hsRSQrq[0], 0, 0);
		}
	if(ng0>0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	else
		{
		dsyrk_ln_libstr(nu0+nx0, nu0+nx0, nx1, 1.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, 1.0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		}

	// other kids: update
	for(ii=1; ii<nkids; ii++)
		{
		dtrmm_rutn_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hsLxt1[ii], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
		dsyrk_ln_libstr(nu0+nx0, nu0+nx0, nx1, 1.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, 1.0, &hsL[0], 0, 0, &hsL[0], 0, 0);
		}
	
	// factorize at the end
	dpotrf_l_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsL[0], 0, 0);
	dtrtr_l_libstr(nx0, 1.0, &hsL[0], nu0, nu0, &hsLxt0[0], 0, 0);

	return;

	}



void d_back_ric_trf_leg1_libstr(int nx0, int nx1, int nu0, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt0, struct d_strmat *hsLxt1, struct d_strmat *hswork_mat)
	{


	dtrmm_rutn_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[0], 0, 0, &hsLxt1[0], 0, 0, 0.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0);
	if(nb0>0)
		{
		ddiaadin_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsdRSQ[0], 0, &hsRSQrq[0], 0, 0);
		}
	if(ng0>0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	else
		{
		dsyrk_ln_libstr(nu0+nx0, nu0+nx0, nx1, 1.0, &hswork_mat[0], 0, 0, &hswork_mat[0], 0, 0, 1.0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		dpotrf_l_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsL[0], 0, 0);
		}
	dtrtr_l_libstr(nx0, 1.0, &hsL[0], nu0, nu0, &hsLxt0[0], 0, 0);

	return;

	}



void d_back_ric_trf_legN_libstr(int nx0, int nb0, int *hidxb0, int ng0, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt)
	{

	if(nb0>0)
		{
		ddiaadin_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsdRSQ[0], 0, &hsRSQrq[0], 0, 0);
		}
	if(ng0>0)
		{
		printf("\nfeature not implemented yet\n\n");
		exit(1);
		}
	else
		{
		dpotrf_l_libstr(nx0, nx0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		}
	dtrtr_l_libstr(nx0, 1.0, &hsL[0], 0, 0, &hsLxt[0], 0, 0);


	return;

	}



void d_back_ric_trs_back_leg0_libstr(int nx0, int nx1, int nu0, int nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL, struct d_strmat *hsLxt, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, struct d_strvec *hswork_vec)
	{

	if(compute_Pb)
		{
		dtrmv_unn_libstr(nx1, &hsLxt[0], 0, 0, &hsb[0], 0, &hsPb[0], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[0], 0, 0, &hsPb[0], 0, &hsPb[0], 0);
		}
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	dveccp_libstr(nx1, 1.0, &hsPb[0], 0, &hswork_vec[0], 0);
	daxpy_libstr(nx1, 1.0, &hsux1[0], nu1, &hswork_vec[0], 0);
	dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[0], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_lnn_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



void d_back_ric_trs_back_leg1_libstr(int nx0, int nx1, int nu0, int nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL, struct d_strmat *hsLxt, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, struct d_strvec *hswork_vec)
	{

	if(compute_Pb)
		{
		dtrmv_unn_libstr(nx1, &hsLxt[0], 0, 0, &hsb[0], 0, &hsPb[0], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[0], 0, 0, &hsPb[0], 0, &hsPb[0], 0);
		}
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	dveccp_libstr(nx1, 1.0, &hsPb[0], 0, &hswork_vec[0], 0);
	daxpy_libstr(nx1, 1.0, &hsux1[0], nu1, &hswork_vec[0], 0);
	dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[0], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_lnn_libstr(nu0+nx0, nu0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



void d_back_ric_trs_back_legN_libstr(int nx0, int nu0, int nb0, int *hidxb0, int ng0, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux)
	{

	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux[0], 0, &hsux[0], 0);
		}

	return;

	}



void d_back_ric_trs_back_funnel0_libstr(int nkids, int nx0, int nx1, int nu0, int nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL, struct d_strmat *hsLxt, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, struct d_strvec *hswork_vec)
	{

	int ii;

	// first kid: initialize with gradient
	ii = 0;
	if(compute_Pb)
		{
		dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hsb[ii], 0, &hsPb[ii], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hsPb[ii], 0, &hsPb[ii], 0);
		}
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	dveccp_libstr(nx1, 1.0, &hsPb[ii], 0, &hswork_vec[0], 0);
	daxpy_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
	dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);

	// other kids: update
	for(ii=1; ii<nkids; ii++)
		{
		if(compute_Pb)
			{
			dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hsb[ii], 0, &hsPb[ii], 0);
			dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hsPb[ii], 0, &hsPb[ii], 0);
			}
//		dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
		dveccp_libstr(nx1, 1.0, &hsPb[ii], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
		dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}

	// solve at the end
	dtrsv_lnn_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



void d_back_ric_trs_back_funnel1_libstr(int nkids, int nx0, int nx1, int nu0, int nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL, struct d_strmat *hsLxt, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, struct d_strvec *hswork_vec)
	{

	int ii;

	// first kid: initialize with gradient
	ii = 0;
	if(compute_Pb)
		{
		dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hsb[ii], 0, &hsPb[ii], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hsPb[ii], 0, &hsPb[ii], 0);
		}
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	dveccp_libstr(nx1, 1.0, &hsPb[ii], 0, &hswork_vec[0], 0);
	daxpy_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
	dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);

	// other kids: update
	for(ii=1; ii<nkids; ii++)
		{
		if(compute_Pb)
			{
			dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hsb[ii], 0, &hsPb[ii], 0);
			dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hsPb[ii], 0, &hsPb[ii], 0);
			}
//		dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);
		dveccp_libstr(nx1, 1.0, &hsPb[ii], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
		dgemv_n_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hswork_vec[0], 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}

	// solve at the end
	dtrsv_lnn_libstr(nu0+nx0, nu0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



void d_back_ric_trs_forw_leg0_libstr(int nx0, int nx1, int nu0, int nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, struct d_strvec *hswork_vec)
	{

	if(compute_pi)
		{
		dveccp_libstr(nx1, 1.0, &hsux1[0], nu1, &hspi[0], 0);
		}
	dveccp_libstr(nu0+nx0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	dgemv_t_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[0], 0, 0, &hsux0[0], 0, 1.0, &hsb[0], 0, &hsux1[0], nu1);
	if(compute_pi)
		{
		dveccp_libstr(nx1, 1.0, &hsux1[0], nu1, &hswork_vec[0], 0);
		dtrmv_unn_libstr(nx1, &hsLxt[0], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[0], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx1, 1.0, &hswork_vec[0], 0, &hspi[0], 0);
		}

	return;

	}



void d_back_ric_trs_forw_leg1_libstr(int nx0, int nx1, int nu0, int nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, struct d_strvec *hswork_vec)
	{

	if(compute_pi)
		{
		dveccp_libstr(nx1, 1.0, &hsux1[0], nu1, &hspi[0], 0);
		}
	dveccp_libstr(nu0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_libstr(nu0+nx0, nu0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	dgemv_t_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[0], 0, 0, &hsux0[0], 0, 1.0, &hsb[0], 0, &hsux1[0], nu1);
	if(compute_pi)
		{
		dveccp_libstr(nx1, 1.0, &hsux1[0], nu1, &hswork_vec[0], 0);
		dtrmv_unn_libstr(nx1, &hsLxt[0], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		dtrmv_utn_libstr(nx1, &hsLxt[0], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
		daxpy_libstr(nx1, 1.0, &hswork_vec[0], 0, &hspi[0], 0);
		}

	return;

	}



void d_back_ric_trs_forw_funnel0_libstr(int nkids, int nx0, int nx1, int nu0, int nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, struct d_strvec *hswork_vec)
	{

	int ii;

	if(compute_pi)
		{
		for(ii=0; ii<nkids; ii++)
			{
			dveccp_libstr(nx1, 1.0, &hsux1[ii], nu1, &hspi[ii], 0);
			}
		}
	dveccp_libstr(nu0+nx0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_libstr(nu0+nx0, nu0+nx0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		dgemv_t_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsb[ii], 0, &hsux1[ii], nu1);
		if(compute_pi)
			{
			dveccp_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
			dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			daxpy_libstr(nx1, 1.0, &hswork_vec[0], 0, &hspi[ii], 0);
			}
		}

	return;

	}



void d_back_ric_trs_forw_funnel1_libstr(int nkids, int nx0, int nx1, int nu0, int nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, struct d_strvec *hswork_vec)
	{

	int ii;

	if(compute_pi)
		{
		for(ii=0; ii<nkids; ii++)
			{
			dveccp_libstr(nx1, 1.0, &hsux1[ii], nu1, &hspi[ii], 0);
			}
		}
	dveccp_libstr(nu0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_libstr(nu0+nx0, nu0, &hsL[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		dgemv_t_libstr(nu0+nx0, nx1, 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsb[ii], 0, &hsux1[ii], nu1);
		if(compute_pi)
			{
			dveccp_libstr(nx1, 1.0, &hsux1[ii], nu1, &hswork_vec[0], 0);
			dtrmv_unn_libstr(nx1, &hsLxt[ii], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			dtrmv_utn_libstr(nx1, &hsLxt[ii], 0, 0, &hswork_vec[0], 0, &hswork_vec[0], 0);
			daxpy_libstr(nx1, 1.0, &hswork_vec[0], 0, &hspi[ii], 0);
			}
		}

	return;

	}



// Riccati recursion routines

void d_tree_back_ric_rec_trf_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strvec *hsdRSQ, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strmat *hswork_mat)
	{

	int nn;

	int dad, stage, nkids, idxkid, real;

	// factorization

	// process one node at the time, starting from the last one
	for(nn=Nn-1; nn>=0; nn--)
		{
//		stage = tree[nn].stage;
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		real = tree[nn].real;
		if(nkids>1) // has many kids => funnel
			{
			idxkid = tree[nn].kids[0];
			d_back_ric_trf_funnel1_libstr(nkids, nx[nn], nx[idxkid], nu[nn], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsRSQrq[nn], &hsdRSQ[nn], &hsQx[nn], &hsL[nn], &hsLxt[nn], &hsLxt[idxkid], hswork_mat);
			}
		else // has at most one kid => leg
			{
			if(nkids==0) // has no kids: last stage
				{
				d_back_ric_trf_legN_libstr(nx[nn], nb[nn], hidxb[nn], ng[nn], &hsRSQrq[nn], &hsdRSQ[nn], &hsQx[nn], &hsL[nn], &hsLxt[nn]);
				}
			else // has one kid: middle stages
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trf_leg1_libstr(nx[nn], nx[idxkid], nu[nn], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsRSQrq[nn], &hsdRSQ[nn], &hsQx[nn], &hsL[nn], &hsLxt[nn], &hsLxt[idxkid], hswork_mat);
				}
			}
		}

	return;

	}



void d_tree_back_ric_rec_trs_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, struct d_strmat *hsLxt, struct d_strvec *hswork_vec)
	{

	int nn;

	int dad, stage, nkids, idxkid, real;

	// backward substitution

	// process one node at the time, starting from the last one
	for(nn=Nn-1; nn>=0; nn--)
		{
//		stage = tree[nn].stage;
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		real = tree[nn].real;
		if(dad<0) // root
			{
			if(nkids>1) // has many kids => funnel
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_funnel0_libstr(nkids, nx[nn], nx[idxkid], nu[nn], nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsb[idxkid], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsLxt[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], hswork_vec);
				}
			else if(nkids==1) // has one kid => leg
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_leg0_libstr(nx[nn], nx[idxkid], nu[nn], nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsb[idxkid], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsLxt[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], hswork_vec);
				}
			else // has no kids: last stage
				{
				// TODO
				}
			}
		else // kid
			{
			if(nkids>1) // has many kids => funnel
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_funnel1_libstr(nkids, nx[nn], nx[idxkid], nu[nn], nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsb[idxkid], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsLxt[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], hswork_vec);
				}
			else if(nkids==1)// has one kid => leg
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_leg1_libstr(nx[nn], nx[idxkid], nu[nn], nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid], &hsb[idxkid], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsLxt[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], hswork_vec);
				}
			else // has no kids: last stage
				{
				d_back_ric_trs_back_legN_libstr(nx[nn], nu[nn], nb[nn], hidxb[nn], ng[nn], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsux[nn]);
				}
			}
		}

	// forward substitution

	// process one node at the time, starting from the first one
	for(nn=0; nn<Nn; nn++)
		{
//		stage = tree[nn].stage;
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		real = tree[nn].real;
		if(dad<0) // root
			{
			if(nkids>1) // has many kids: funnel
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_funnel0_libstr(nkids, nx[nn], nx[idxkid], nu[nn], nu[idxkid], &hsBAbt[idxkid], &hsb[idxkid], &hsL[nn], &hsLxt[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], hswork_vec);
				}
			else if(nkids==1) // has one kid: leg
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_leg0_libstr(nx[nn], nx[idxkid], nu[nn], nu[idxkid], &hsBAbt[idxkid], &hsb[idxkid], &hsL[nn], &hsLxt[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], hswork_vec);
				}
			else // no kids
				{
				// TODO
				}
			}
		else // kids
			{
			if(nkids>1) // has many kids: funnel
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_funnel1_libstr(nkids, nx[nn], nx[idxkid], nu[nn], nu[idxkid], &hsBAbt[idxkid], &hsb[idxkid], &hsL[nn], &hsLxt[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], hswork_vec);
				}
			else if(nkids==1) // has one kid: leg
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_leg1_libstr(nx[nn], nx[idxkid], nu[nn], nu[idxkid], &hsBAbt[idxkid], &hsb[idxkid], &hsL[nn], &hsLxt[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], hswork_vec);
				}
			else // has no kids: last stage
				{
				// nothing to do
				}
			}
		}

	return;

	}



#endif
