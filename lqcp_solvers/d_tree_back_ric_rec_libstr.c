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
#include <blasfeo_d_aux.h>
#include <blasfeo_d_kernel.h>
#include <blasfeo_d_blas.h>

#include "../include/tree.h"



// work space
int d_tree_back_ric_rec_work_space_size_bytes_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int *ng)
	{

	int ii, jj;

	int idxkid, tmp;

	int size = 0;

	// max sizes
	int nxM   = 0;
	int nuxM  = 0;
	int nxgM  = 0;
	for(ii=0; ii<Nn; ii++)
		{
		nxM = nx[ii]>nxM ? nx[ii] : nxM;
		nuxM = nu[ii]+nx[ii]>nuxM ? nu[ii]+nx[ii] : nuxM;
		tmp = 0;
		for(jj=0; jj<tree[ii].nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			tmp += nx[idxkid];
			}
		nxgM = tmp>nxgM ? tmp : nxgM;
		nxgM = ng[ii]>nxgM ? ng[ii] : nxgM;
		}

	size += d_size_strmat(nuxM+1, nxgM); // ric_work_mat[0]
	size += d_size_strvec(nxM); // ric_work_vec[0]

	return size;
	}



// help routines

static void d_back_ric_sv_back_1_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, int nb0, int *hidxb0, int ng0, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_rq, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL0, struct d_strmat *hsL1, void *work)
	{

	char *c_ptr;

	struct d_strmat hswork_mat_0;
	struct d_strvec hswork_vec_0;

	int ii;

	int nx1t, tmp;

	nx1t = 0;
	for(ii=0; ii<nkids; ii++)
		nx1t += nx1[ii];

	c_ptr = (char *) work;
	d_create_strmat(nu0+nx0+1, nx1t, &hswork_mat_0, (void *) c_ptr);
	c_ptr += hswork_mat_0.memory_size;
	d_create_strvec(nx1t, &hswork_vec_0, (void *) c_ptr);
	c_ptr += hswork_vec_0.memory_size;
		
	tmp = 0;
	for(ii=0; ii<nkids; ii++)
		{
		if(update_b)
			{
			drowin_libstr(nx1[ii], 1.0, &hsb[ii], 0, &hsBAbt[ii], nu0+nx0, 0);
			}
		dtrmm_rlnn_libstr(nu0+nx0+1, nx1[ii], 1.0, &hsL1[ii], nu1[ii], nu1[ii], &hsBAbt[ii], 0, 0, &hswork_mat_0, 0, tmp);
		if(compute_Pb)
			{
			drowex_libstr(nx1[ii], 1.0, &hswork_mat_0, nu0+nx0, tmp, &hswork_vec_0, 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hswork_vec_0, 0, &hsPb[ii], 0);
			}
		dgead_libstr(1, nx1[ii], 1.0, &hsL1[ii], nu1[ii]+nx1[ii], nu1[ii], &hswork_mat_0, nu0+nx0, tmp);
		tmp += nx1[ii];
		}
	if(nb0>0 | ng0>0 | update_rq)
		{
		// initialize with hessian
		dtrcp_l_libstr(nu0+nx0, 1.0, &hsRSQrq[0], 0, 0, &hsL0[0], 0, 0);
		if(update_rq)
			{
			drowin_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsL0[0], nu0+nx0, 0);
			}
		else
			{
			dgecp_libstr(1, nu0+nx0, 1.0, &hsRSQrq[0], nu0+nx0, 0, &hsL0[0], nu0+nx0, 0);
			}
		// update with box constraints
		if(nb0>0)
			{
			ddiaad_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsL0[0], 0, 0);
			drowad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsL0[0], nu0+nx0, 0);
			}
		// update with general constraints and factorize at the end
		if(ng0>0)
			{
			dsyrk_ln_libstr(nu0+nx0+1, nu0+nx0, nx1t, 1.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, 1.0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			d_create_strmat(nu0+nx0, ng0, &hswork_mat_0, work);
			dgemm_r_diag_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsQx[0], nb0, 0.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0);
			drowin_libstr(ng0, 1.0, &hsqx[0], nb0, &hswork_mat_0, nu0+nx0, 0);
			dsyrk_dpotrf_ln_libstr(nu0+nx0+1, nu0+nx0, ng0, &hswork_mat_0, 0, 0, &hsDCt[0], 0, 0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			}
		else
			{
			dsyrk_dpotrf_ln_libstr(nu0+nx0+1, nu0+nx0, nx1t, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			}
		}
	else
		{
		dsyrk_dpotrf_ln_libstr(nu0+nx0+1, nu0+nx0, nx1t, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, &hsRSQrq[0], 0, 0, &hsL0[0], 0, 0);
		}

	return;

	}



static void d_back_ric_sv_back_N_libstr(int nx0, int nu0, int nb0, int *hidxb0, int ng0, int update_rq, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strmat *hsL, void *work)
	{

	struct d_strmat hswork_mat_0;

	if(nb0>0 | ng0>0 | update_rq)
		{
		dtrcp_l_libstr(nu0+nx0, 1.0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		if(update_rq)
			{
			drowin_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsL[0], nu0+nx0, 0);
			}
		else
			{
			dgecp_libstr(1, nu0+nx0, 1.0, &hsRSQrq[0], nu0+nx0, 0, &hsL[0], nu0+nx0, 0);
			}
		if(nb0>0)
			{
			ddiaad_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsL[0], 0, 0);
			drowad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsL[0], nu0+nx0, 0);
			}
		if(ng0>0)
			{
			d_create_strmat(nu0+nx0+1, ng0, &hswork_mat_0, work);
			dgemm_r_diag_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsQx[0], nb0, 0.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0);
			drowin_libstr(ng0, 1.0, &hsqx[0], nb0, &hswork_mat_0, nu0+nx0, 0);
			dsyrk_dpotrf_ln_libstr(nu0+nx0+1, nu0+nx0, ng0, &hswork_mat_0, 0, 0, &hsDCt[0], 0, 0, &hsL[0], 0, 0, &hsL[0], 0, 0);
			}
		else
			{
			dpotrf_l_mn_libstr(nu0+nx0+1, nu0+nx0, &hsL[0], 0, 0, &hsL[0], 0, 0);
			}
		}
	else
		{
		dpotrf_l_mn_libstr(nu0+nx0+1, nu0+nx0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		}

	return;

	}



static void d_back_ric_sv_forw_0_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, struct d_strmat *hsBAbt, struct d_strmat *hsL0, struct d_strmat *hsL1, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, void *work)
	{

	struct d_strvec hswork_vec_0;

	int ii;

	drowex_libstr(nu0+nx0, -1.0, &hsL0[0], nu0+nx0, 0, &hsux0[0], 0);
	dtrsv_ltn_mn_libstr(nu0+nx0, nu0+nx0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		drowex_libstr(nx1[ii], 1.0, &hsBAbt[ii], nu0+nx0, 0, &hsux1[ii], nu1[ii]);
		dgemv_t_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsux1[ii], nu1[ii], &hsux1[ii], nu1[ii]);
		if(compute_pi)
			{
			d_create_strvec(nx1[ii], &hswork_vec_0, work);
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hspi[ii], 0);
			drowex_libstr(nx1[ii], 1.0, &hsL1[ii], nu1[ii]+nx1[ii], nu1[ii], &hswork_vec_0, 0);
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hspi[ii], 0, &hspi[ii], 0);
			daxpy_libstr(nx1[ii], 1.0, &hswork_vec_0, 0, &hspi[ii], 0, &hspi[ii], 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hspi[ii], 0, &hspi[ii], 0);
			}
		}

	return;

	}



static void d_back_ric_sv_forw_1_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, struct d_strmat *hsBAbt, struct d_strmat *hsL0, struct d_strmat *hsL1, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, void *work)
	{

	struct d_strvec hswork_vec_0;

	int ii;

	drowex_libstr(nu0, -1.0, &hsL0[0], nu0+nx0, 0, &hsux0[0], 0);
	dtrsv_ltn_mn_libstr(nu0+nx0, nu0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		drowex_libstr(nx1[ii], 1.0, &hsBAbt[ii], nu0+nx0, 0, &hsux1[ii], nu1[ii]);
		dgemv_t_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsux1[ii], nu1[ii], &hsux1[ii], nu1[ii]);
		if(compute_pi)
			{
			d_create_strvec(nx1[ii], &hswork_vec_0, work);
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hspi[ii], 0);
			drowex_libstr(nx1[ii], 1.0, &hsL1[ii], nu1[ii]+nx1[ii], nu1[ii], &hswork_vec_0, 0);
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hspi[ii], 0, &hspi[ii], 0);
			daxpy_libstr(nx1[ii], 1.0, &hswork_vec_0, 0, &hspi[ii], 0, &hspi[ii], 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hspi[ii], 0, &hspi[ii], 0);
			}
		}

	return;

	}



static void d_back_ric_trf_1_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL0, struct d_strmat *hsL1, void *work)
	{

	struct d_strmat hswork_mat_0;

	int ii;

	int nx1t, tmp;

	nx1t = 0;
	for(ii=0; ii<nkids; ii++)
		nx1t += nx1[ii];

	d_create_strmat(nu0+nx0, nx1t, &hswork_mat_0, work);
		
	// all kids: update
	tmp = 0;
	for(ii=0; ii<nkids; ii++)
		{
		dtrmm_rlnn_libstr(nu0+nx0, nx1[ii], 1.0, &hsL1[ii], nu1[ii], nu1[ii], &hsBAbt[ii], 0, 0, &hswork_mat_0, 0, tmp);
		tmp += nx1[ii];
		}
	if(nb0>0 | ng0>0)
		{
		// initialize with hessian
		dtrcp_l_libstr(nu0+nx0, 1.0, &hsRSQrq[0], 0, 0, &hsL0[0], 0, 0);
		// update with box constraints
		if(nb0>0)
			{
			ddiaad_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsL0[0], 0, 0);
			}
		// update with general constraints and factorize at the end
		if(ng0>0)
			{
			dsyrk_ln_libstr(nu0+nx0, nu0+nx0, nx1t, 1.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, 1.0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			d_create_strmat(nu0+nx0, ng0, &hswork_mat_0, work);
			dgemm_r_diag_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsQx[0], nb0, 0.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0);
			dsyrk_dpotrf_ln_libstr(nu0+nx0, nu0+nx0, ng0, &hsDCt[0], 0, 0, &hswork_mat_0, 0, 0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			}
		else
			{
			dsyrk_dpotrf_ln_libstr(nu0+nx0, nu0+nx0, nx1t, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, &hsL0[0], 0, 0, &hsL0[0], 0, 0);
			}
		}
	else
		{
		dsyrk_dpotrf_ln_libstr(nu0+nx0, nu0+nx0, nx1t, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0, &hsRSQrq[0], 0, 0, &hsL0[0], 0, 0);
		}

	return;

	}



static void d_back_ric_trf_N_libstr(int nx0, int nu0, int nb0, int *hidxb0, int ng0, struct d_strmat *hsRSQrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, void *work)
	{

	char *c_ptr;

	struct d_strmat hswork_mat_0;

	if(nb0>0 | ng0>0)
		{
		dtrcp_l_libstr(nu0+nx0, 1.0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		if(nb0>0)
			{
			ddiaad_libspstr(nb0, hidxb0, 1.0, &hsQx[0], 0, &hsL[0], 0, 0);
			}
		if(ng0>0)
			{
			d_create_strmat(nu0+nx0, ng0, &hswork_mat_0, work);
			dgemm_r_diag_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsQx[0], nb0, 0.0, &hswork_mat_0, 0, 0, &hswork_mat_0, 0, 0);
			dsyrk_dpotrf_ln_libstr(nu0+nx0, nu0+nx0, ng0, &hswork_mat_0, 0, 0, &hsDCt[0], 0, 0, &hsL[0], 0, 0, &hsL[0], 0, 0);
			}
		else
			{
			dpotrf_l_libstr(nu0+nx0, &hsL[0], 0, 0, &hsL[0], 0, 0);
			}
		}
	else
		{
		dpotrf_l_libstr(nu0+nx0, &hsRSQrq[0], 0, 0, &hsL[0], 0, 0);
		}

	return;

	}



static void d_back_ric_trs_back_N_libstr(int nx0, int nu0, int nb0, int *hidxb0, int ng0, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux)
	{

	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux[0], 0);
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux[0], 0);
		}
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux[0], 0, &hsux[0], 0);
		}

	return;

	}



static void d_back_ric_trs_back_0_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL0, struct d_strmat *hsL1, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, void *work)
	{

	int ii;

	struct d_strvec hswork_vec_0;

	// initialize with gradient
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);

	// all kids: update
	for(ii=0; ii<nkids; ii++)
		{
		d_create_strvec(nx1[ii], &hswork_vec_0, work);
		if(compute_Pb)
			{
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hsb[ii], 0, &hsPb[ii], 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hsPb[ii], 0, &hsPb[ii], 0);
			}
		daxpy_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hsPb[ii], 0, &hswork_vec_0, 0);
		dgemv_n_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hswork_vec_0, 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}

	// update with box constraints
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	// update with general constraints
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	// solve at the end
	dtrsv_lnn_mn_libstr(nu0+nx0, nu0+nx0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



static void d_back_ric_trs_back_1_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, int nb0, int *hidxb0, int ng0, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strmat *hsL0, struct d_strmat *hsL1, int compute_Pb, struct d_strvec *hsPb, struct d_strvec *hsux0, struct d_strvec *hsux1, void *work)
	{

	int ii;

	struct d_strvec hswork_vec_0;

	// initialize with gradient
	dveccp_libstr(nu0+nx0, 1.0, &hsrq[0], 0, &hsux0[0], 0);

	// all kids: update
	for(ii=0; ii<nkids; ii++)
		{
		d_create_strvec(nx1[ii], &hswork_vec_0, work);
		if(compute_Pb)
			{
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hsb[ii], 0, &hsPb[ii], 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hsPb[ii], 0, &hsPb[ii], 0);
			}
		daxpy_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hsPb[ii], 0, &hswork_vec_0, 0);
		dgemv_n_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hswork_vec_0, 0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}

	// update with box constraints
	if(nb0>0)
		{
		dvecad_libspstr(nb0, hidxb0, 1.0, &hsqx[0], 0, &hsux0[0], 0);
		}
	// update with general constraints
	if(ng0>0)
		{
		dgemv_n_libstr(nu0+nx0, ng0, 1.0, &hsDCt[0], 0, 0, &hsqx[0], nb0, 1.0, &hsux0[0], 0, &hsux0[0], 0);
		}
	// solve at the end
	dtrsv_lnn_mn_libstr(nu0+nx0, nu0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);

	return;

	}



static void d_back_ric_trs_forw_0_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL0, struct d_strmat *hsL1, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, void *work)
	{

	struct d_strvec hswork_vec_0;

	int ii;

	dveccp_libstr(nu0+nx0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_mn_libstr(nu0+nx0, nu0+nx0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		if(compute_pi)
			{
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hspi[ii], 0);
			}
		dgemv_t_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsb[ii], 0, &hsux1[ii], nu1[ii]);
		if(compute_pi)
			{
			d_create_strvec(nx1[ii], &hswork_vec_0, work);
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hswork_vec_0, 0);
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hswork_vec_0, 0, &hswork_vec_0, 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hswork_vec_0, 0, &hswork_vec_0, 0);
			daxpy_libstr(nx1[ii], 1.0, &hswork_vec_0, 0, &hspi[ii], 0, &hspi[ii], 0);
			}
		}

	return;

	}



static void d_back_ric_trs_forw_1_libstr(int nkids, int nx0, int *nx1, int nu0, int *nu1, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsL0, struct d_strmat *hsL1, struct d_strvec *hsux0, struct d_strvec *hsux1, int compute_pi, struct d_strvec *hspi, void *work)
	{

	struct d_strvec hswork_vec_0;

	int ii;

	dveccp_libstr(nu0, -1.0, &hsux0[0], 0, &hsux0[0], 0);
	dtrsv_ltn_mn_libstr(nu0+nx0, nu0, &hsL0[0], 0, 0, &hsux0[0], 0, &hsux0[0], 0);
	for(ii=0; ii<nkids; ii++)
		{
		if(compute_pi)
			{
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hspi[ii], 0);
			}
		dgemv_t_libstr(nu0+nx0, nx1[ii], 1.0, &hsBAbt[ii], 0, 0, &hsux0[0], 0, 1.0, &hsb[ii], 0, &hsux1[ii], nu1[ii]);
		if(compute_pi)
			{
			d_create_strvec(nx1[ii], &hswork_vec_0, work);
			dveccp_libstr(nx1[ii], 1.0, &hsux1[ii], nu1[ii], &hswork_vec_0, 0);
			dtrmv_ltn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hswork_vec_0, 0, &hswork_vec_0, 0);
			dtrmv_lnn_libstr(nx1[ii], nx1[ii], &hsL1[ii], nu1[ii], nu1[ii], &hswork_vec_0, 0, &hswork_vec_0, 0);
			daxpy_libstr(nx1[ii], 1.0, &hswork_vec_0, 0, &hspi[ii], 0, &hspi[ii], 0);
			}
		}

	return;

	}



// Riccati recursion routines

void d_tree_back_ric_rec_sv_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, int update_b, struct d_strmat *hsBAbt, struct d_strvec *hsb, int update_rq, struct d_strmat *hsRSQrq, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, void *work)
	{

	int nn, ii;

	int nkids, idxkid, dad;

	struct d_strmat hswork_mat_0;

	// factorization and backward substitution

	// process one node at the time, starting from the last one
	for(nn=Nn-1; nn>=0; nn--)
		{

		nkids = tree[nn].nkids;

		if(nkids==0) // has no kids: last stage
			{
			d_back_ric_sv_back_N_libstr(nx[nn], nu[nn], nb[nn], hidxb[nn], ng[nn], update_rq, &hsRSQrq[nn], &hsrq[nn], &hsDCt[nn], &hsQx[nn], &hsqx[nn], &hsL[nn], work);
			}
		else // has at least one kid
			{
			idxkid = tree[nn].kids[0];
			d_back_ric_sv_back_1_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], nb[nn], hidxb[nn], ng[nn], update_b, &hsBAbt[idxkid-1], &hsb[idxkid-1], update_rq, &hsRSQrq[nn], &hsrq[nn], &hsDCt[nn], &hsQx[nn], &hsqx[nn], compute_Pb, &hsPb[idxkid], &hsL[nn], &hsL[idxkid], work);
			}
		}

	// forward substitution

	// process one node at the time, starting from the first one
	for(nn=0; nn<Nn; nn++)
		{
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		if(dad<0) // root
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_sv_forw_0_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], &hsBAbt[idxkid-1], &hsL[nn], &hsL[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], work);
				}
			else // no kids
				{
				// TODO
				}
			}
		else // kids
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_sv_forw_1_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], &hsBAbt[idxkid-1], &hsL[nn], &hsL[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], work);
				}
			else // has no kids: last stage
				{
				// nothing to do
				}
			}
		}

	return;

	}



void d_tree_back_ric_rec_trf_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsDCt, struct d_strvec *hsQx, struct d_strmat *hsL, void *work)
	{

	int nn, ii;

	int nkids, idxkid;

	struct d_strmat hswork_mat_0;

	// factorization

	// process one node at the time, starting from the last one
	for(nn=Nn-1; nn>=0; nn--)
		{

		nkids = tree[nn].nkids;

		if(nkids==0) // has no kids: last stage
			{
			d_back_ric_trf_N_libstr(nx[nn], nu[nn], nb[nn], hidxb[nn], ng[nn], &hsRSQrq[nn], &hsDCt[nn], &hsQx[nn], &hsL[nn], work);
			}
		else // has at least one kid
			{
			idxkid = tree[nn].kids[0];
			d_back_ric_trf_1_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid-1], &hsRSQrq[nn], &hsDCt[nn], &hsQx[nn], &hsL[nn], &hsL[idxkid], work);
			}
		}

	return;

	}



void d_tree_back_ric_rec_trs_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strvec *hsrq, struct d_strmat *hsDCt, struct d_strvec *hsqx, struct d_strvec *hsux, int compute_pi, struct d_strvec *hspi, int compute_Pb, struct d_strvec *hsPb, struct d_strmat *hsL, void *work)
	{

	int nn;

	int dad, nkids, idxkid;

	// backward substitution

	// process one node at the time, starting from the last one
	for(nn=Nn-1; nn>=0; nn--)
		{
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		if(dad<0) // root
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_0_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid-1], &hsb[idxkid-1], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsL[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], work);
				}
			else // has no kids: last stage
				{
				// TODO
				}
			}
		else // kid
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_back_1_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], nb[nn], hidxb[nn], ng[nn], &hsBAbt[idxkid-1], &hsb[idxkid-1], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsL[nn], &hsL[idxkid], compute_Pb, &hsPb[idxkid], &hsux[nn], &hsux[idxkid], work);
				}
			else // has no kids: last stage
				{
				d_back_ric_trs_back_N_libstr(nx[nn], nu[nn], nb[nn], hidxb[nn], ng[nn], &hsrq[nn], &hsDCt[nn], &hsqx[nn], &hsux[nn]);
				}
			}
		}

	// forward substitution

	// process one node at the time, starting from the first one
	for(nn=0; nn<Nn; nn++)
		{
		dad = tree[nn].dad;
		nkids = tree[nn].nkids;
		if(dad<0) // root
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_0_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], &hsBAbt[idxkid-1], &hsb[idxkid-1], &hsL[nn], &hsL[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], work);
				}
			else // no kids
				{
				// TODO
				}
			}
		else // kids
			{
			if(nkids>0) // has kids
				{
				idxkid = tree[nn].kids[0];
				d_back_ric_trs_forw_1_libstr(nkids, nx[nn], &nx[idxkid], nu[nn], &nu[idxkid], &hsBAbt[idxkid-1], &hsb[idxkid-1], &hsL[nn], &hsL[idxkid], &hsux[nn], &hsux[idxkid], compute_pi, &hspi[idxkid], work);
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
