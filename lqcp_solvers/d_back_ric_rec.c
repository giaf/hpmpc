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

#include <math.h>

#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/block_size.h"
#include "../include/lqcp_aux.h"



int d_ric_sv_mpc_tv_work_space_size_double(int N, int *nx, int *nu, int *nb, int *ng)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int nzM  = 0;
	for(ii=0; ii<=N; ii++)
		{
		if(nu[ii]+nx[ii]+1>nzM) nzM = nu[ii]+nx[ii]+1;
		}

	int nxgM = ng[N];
	for(ii=0; ii<N; ii++)
		{
		if(nx[ii+1]+ng[ii]>nxgM) nxgM = nx[ii+1]+ng[ii];
		}
	
	int size = ((nzM+bs-1)/bs*bs) * ((nxgM+ncl-1)/ncl*ncl);

	return size;
	}



/* version tailored for mpc (x0 fixed) ; version supporting time-variant nx, nu, nb, ng */
void d_back_ric_sv_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hux, double **hpL, double **hdL, double *work, double *diag, int compute_Pb, double **hPb, int compute_pi, double **hpi, int *nb, int **idxb, double **hQd, double **hQl, int *ng, double **hpDCt, double **Qx, double **qx)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	
	int nu0, nu1, nx0, nx1, nz0, nb0, ng0, cnx0, cnx1, cnl0, cnl1, cng0, cnxg0, pnx1, nux0, nux1, cnux0, cnux1;

	int ii, jj, ll, nn;

	double temp;

	// factorization and backward substitution 

	// final stage 
	nu0 = nu[N];
	nx0 = nx[N];
	nb0 = nb[N];
	ng0 = ng[N];
	nux0 = nu0+nx0;
	nz0 = nux0+1;
	cnx0 = (nx0+ncl-1)/ncl*ncl;
//	cnz0 = (nz0+ncl-1)/ncl*ncl; // XXX
	cnux0 = (nux0+ncl-1)/ncl*ncl; // XXX
	cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;

	if(ng0>0)
		{
		cng0 = (ng0+ncl-1)/ncl*ncl;
		dgemm_diag_right_lib(nux0, ng0, hpDCt[N], cng0, Qx[N], 0, work, cng0, work, cng0);
		drowin_lib(ng0, qx[N], &work[(nux0)/bs*cng0*bs+(nux0)%bs]);
		}
	if(nb0>0)
		{
		ddiain_libsp(nb0, idxb[N], hQd[N], hpQ[N], cnux0);
		drowin_libsp(nb0, idxb[N], hQl[N], hpQ[N]+(nux0)/bs*bs*cnux0+(nux0)%bs);
		}

	dsyrk_dpotrf_lib(nz0, nux0, ng0, work, cng0, work, cng0, 1, hpQ[N], cnux0, hpL[N], cnl0, hdL[N]);
//	d_print_pmat(nz0, nux0, bs, hpL[N], cnl0);
//	exit(2);

	dtrtr_l_lib(nx0, nu0, hpL[N]+nu0/bs*bs*cnl0+nu0%bs+nu0*bs, cnl0, 0, hpL[N]+ncl*bs, cnl0);	



	// middle stages 
	for(nn=0; nn<N; nn++)
		{	
		nu1 = nu0;
		nx1 = nx0;
		nux1 = nux0;
		cnx1 = cnx0;
		cnl1 = cnl0;
		nu0 = nu[N-nn-1];
		nx0 = nx[N-nn-1];
		nb0 = nb[N-nn-1];
		ng0 = ng[N-nn-1];
		nux0 = nu0+nx0;
		nz0 = nux0+1;
		cnx0 = (nx0+ncl-1)/ncl*ncl;
//		cnz0 = (nz0+ncl-1)/ncl*ncl; // XXX
		cnux0 = (nux0+ncl-1)/ncl*ncl; // XXX
		cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;
		cnxg0 = (nx1+ng0+ncl-1)/ncl*ncl;

		dtrmm_nt_u_lib(nz0, nx1, hpBAbt[N-nn-1], cnx1, hpL[N-nn]+ncl*bs, cnl1, work, cnxg0);

		if(compute_Pb)
			{
			for(jj=0; jj<nx1; jj++) diag[jj] = work[(nux0)/bs*bs*cnxg0+(nux0)%bs+jj*bs];
			dtrmv_u_t_lib(nx1, hpL[N-nn]+ncl*bs, cnl1, diag, 0, hPb[N-nn-1]); // L*(L'*b)
			}
		dgead_lib(1, nx1, 1.0, nux1, hpL[N-nn]+(nux1)/bs*bs*cnl1+(nux1)%bs+nu1*bs,cnl1, nux0, work+(nux0)/bs*bs*cnxg0+(nux0)%bs, cnxg0);
		if(ng0>0)
			{
			cng0 = (ng0+ncl-1)/ncl*ncl;
			dgemm_diag_right_lib(nux0, ng0, hpDCt[N-nn-1], cng0, Qx[N-nn-1], 0, work+nx1*bs, cnxg0, work+nx1*bs, cnxg0);
			drowin_lib(ng0, qx[N-nn-1], &work[(nux0)/bs*cnxg0*bs+(nux0)%bs+nx1*bs]);
			}
		if(nb0>0)
			{
			ddiain_libsp(nb0, idxb[N-nn-1], hQd[N-nn-1], hpQ[N-nn-1], cnux0);
			drowin_libsp(nb0, idxb[N-nn-1], hQl[N-nn-1], hpQ[N-nn-1]+(nux0)/bs*bs*cnux0+(nux0)%bs);
			}

		dsyrk_dpotrf_lib(nz0, nux0, nx1+ng0, work, cnxg0, work, cnxg0, 1, hpQ[N-nn-1], cnux0, hpL[N-nn-1], cnl0, hdL[N-nn-1]);

		dtrtr_l_lib(nx0, nu0, hpL[N-nn-1]+nu0/bs*bs*cnl0+nu0%bs+nu0*bs, cnl0, 0, hpL[N-nn-1]+ncl*bs, cnl0);	

//	d_print_pmat(nz0, nux0, bs, hpL[N-nn-1], cnl0);
		}

//	exit(2);


	// forward substitution 

	nu1 = nu[0];
	nx1 = nx[0];
	nux1 = nu1+nx1;
	cnx1 = (nx1+ncl-1)/ncl*ncl;
//	cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
	cnux1 = (nux1+ncl-1)/ncl*ncl;
	cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
	// first stage
	nn = 0;
	nu0 = nu1;
	nx0 = nx1;
	nux0 = nux1;
	cnx0 = cnx1;
//	cnz0 = cnz1;
	cnl0 = cnl1;
	nu1 = nu[nn+1];
	nx1 = nx[nn+1];
	nux1 = nu1+nx1;
	cnx1 = (nx1+ncl-1)/ncl*ncl;
//	cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
	cnux1 = (nux1+ncl-1)/ncl*ncl;
	cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
	pnx1  = (nx1+ncl-1)/ncl*ncl;
	for(jj=0; jj<nux0; jj++) hux[nn][jj] = - hpL[nn][(nux0)/bs*bs*cnl0+(nux0)%bs+bs*jj];
	dtrsv_t_lib(nux0, nux0, hpL[nn], cnl0, 1, hdL[nn], &hux[nn][0], &hux[nn][0]);
	for(jj=0; jj<nx1; jj++) hux[nn+1][nu1+jj] = hpBAbt[nn][(nux0)/bs*bs*cnx1+(nux0)%bs+bs*jj];
	dgemv_t_lib(nux0, nx1, hpBAbt[nn], cnx1, &hux[nn][0], 1, &hux[nn+1][nu1], &hux[nn+1][nu1]);
	if(compute_pi)
		{
		for(jj=0; jj<nx1; jj++) work[pnx1+jj] = hux[nn+1][nu1+jj]; // copy x into aligned memory
		for(jj=0; jj<nx1; jj++) work[jj] = hpL[nn+1][(nux1)/bs*bs*cnl1+(nux1)%bs+bs*(nu1+jj)]; // work space
		dtrmv_u_n_lib(nx1, hpL[nn+1]+(ncl)*bs, cnl1, &work[pnx1], 1, &work[0]);
		dtrmv_u_t_lib(nx1, hpL[nn+1]+(ncl)*bs, cnl1, &work[0], 0, &hpi[nn+1][0]); // L*(L'*b) + p
		}
	// moddle stages
	for(nn=1; nn<N; nn++)
		{
		nu0 = nu1;
		nx0 = nx1;
		nux0 = nux1;
		cnx0 = cnx1;
//		cnz0 = cnz1;
		cnl0 = cnl1;
		nu1 = nu[nn+1];
		nx1 = nx[nn+1];
		nux1 = nu1+nx1;
		cnx1 = (nx1+ncl-1)/ncl*ncl;
//		cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
		cnux1 = (nux1+ncl-1)/ncl*ncl;
		cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
		pnx1  = (nx1+ncl-1)/ncl*ncl;
		for(jj=0; jj<nu0; jj++) hux[nn][jj] = - hpL[nn][(nux0)/bs*bs*cnl0+(nux0)%bs+bs*jj];
		dtrsv_t_lib(nux0, nu0, hpL[nn], cnl0, 1, hdL[nn], &hux[nn][0], &hux[nn][0]);
		for(jj=0; jj<nx1; jj++) hux[nn+1][nu1+jj] = hpBAbt[nn][(nux0)/bs*bs*cnx1+(nux0)%bs+bs*jj];
		dgemv_t_lib(nux0, nx1, hpBAbt[nn], cnx1, &hux[nn][0], 1, &hux[nn+1][nu1], &hux[nn+1][nu1]);
		if(compute_pi)
			{
			for(jj=0; jj<nx1; jj++) work[pnx1+jj] = hux[nn+1][nu1+jj]; // copy x into aligned memory
			for(jj=0; jj<nx1; jj++) work[jj] = hpL[nn+1][(nux1)/bs*bs*cnl1+(nux1)%bs+bs*(nu1+jj)]; // work space
			dtrmv_u_n_lib(nx1, hpL[nn+1]+(ncl)*bs, cnl1, &work[pnx1], 1, &work[0]);
			dtrmv_u_t_lib(nx1, hpL[nn+1]+(ncl)*bs, cnl1, &work[0], 0, &hpi[nn+1][0]); // L*(L'*b) + p
			}
		}
	
	}



void d_back_ric_trf_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hpL, double **hdL, double *work, int *nb, int **idxb, double **hQd, int *ng, double **hpDCt, double **Qx)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	
	int nu0, nu1, nx0, nx1, nz0, nb0, ng0, cnx0, cnx1, cnl0, cnl1, cng0, cnxg0, pnx1, nux0, nux1, cnux0, cnux1;

	int ii, jj, ll, nn;

	double temp;

	// factorization and backward substitution 

	// final stage 
	nu0 = nu[N];
	nx0 = nx[N];
	nb0 = nb[N];
	ng0 = ng[N];
	nux0 = nu0+nx0;
	nz0 = nux0+1;
	cnx0 = (nx0+ncl-1)/ncl*ncl;
//	cnz0 = (nz0+ncl-1)/ncl*ncl; // XXX
	cnux0 = (nux0+ncl-1)/ncl*ncl; // XXX
	cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;

	if(ng0>0)
		{
		cng0 = (ng0+ncl-1)/ncl*ncl;
		dgemm_diag_right_lib(nux0, ng0, hpDCt[N], cng0, Qx[N], 0, work, cng0, work, cng0);
		}
	if(nb0>0)
		{
		ddiain_libsp(nb0, idxb[N], hQd[N], hpQ[N], cnux0);
		}

	dsyrk_dpotrf_lib(nux0, nux0, ng0, work, cng0, work, cng0, 1, hpQ[N], cnux0, hpL[N], cnl0, hdL[N]);

	dtrtr_l_lib(nx0, nu0, hpL[N]+nu0/bs*bs*cnl0+nu0%bs+nu0*bs, cnl0, 0, hpL[N]+ncl*bs, cnl0);	



	// middle stages 
	for(nn=0; nn<N; nn++)
		{	
		nu1 = nu0;
		nx1 = nx0;
		nux1 = nux0;
		cnx1 = cnx0;
		cnl1 = cnl0;
		nu0 = nu[N-nn-1];
		nx0 = nx[N-nn-1];
		nb0 = nb[N-nn-1];
		ng0 = ng[N-nn-1];
		nux0 = nu0+nx0;
		nz0 = nux0+1;
		cnx0 = (nx0+ncl-1)/ncl*ncl;
//		cnz0 = (nz0+ncl-1)/ncl*ncl; // XXX
		cnux0 = (nux0+ncl-1)/ncl*ncl; // XXX
		cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;
		cnxg0 = (nx1+ng0+ncl-1)/ncl*ncl;

		dtrmm_nt_u_lib(nux0, nx1, hpBAbt[N-nn-1], cnx1, hpL[N-nn]+ncl*bs, cnl1, work, cnxg0);

		if(ng0>0)
			{
			cng0 = (ng0+ncl-1)/ncl*ncl;
			dgemm_diag_right_lib(nux0, ng0, hpDCt[N-nn-1], cng0, Qx[N-nn-1], 0, work+nx1*bs, cnxg0, work+nx1*bs, cnxg0);
			}
		if(nb0>0)
			{
			ddiain_libsp(nb0, idxb[N-nn-1], hQd[N-nn-1], hpQ[N-nn-1], cnux0);
			}

		dsyrk_dpotrf_lib(nux0, nux0, nx1+ng0, work, cnxg0, work, cnxg0, 1, hpQ[N-nn-1], cnux0, hpL[N-nn-1], cnl0, hdL[N-nn-1]);

		dtrtr_l_lib(nx0, nu0, hpL[N-nn-1]+nu0/bs*bs*cnl0+nu0%bs+nu0*bs, cnl0, 0, hpL[N-nn-1]+ncl*bs, cnl0);	

		}

	}



void d_back_ric_trs_tv(int N, int *nx, int *nu, double **hpBAbt, double **hb, double **hpL, double **hdL, double **hq, double **hl, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi, int *nb, int **idxb, double **hql, int *ng, double **hpDCt, double **qx)
	{
	
	const int bs  = D_MR;
	const int ncl = D_NCL;

	int nu0, nu1, nx0, nx1, nb0, ng0, pnx0, pnx1, cnx0, cnx1, cng0, cnl0, cnl1, nux0, nux1, cnux0, cnux1;

	int ii, jj, nn;
	
	// backward substitution 

	// final stage
	nu0 = nu[N];
	nx0 = nx[N];
	nb0 = nb[N];
	ng0 = ng[N];
	nux0 = nu0+nx0;
	pnx0 = (nx0+bs-1)/bs*bs;
	cnx0 = (nx0+ncl-1)/ncl*ncl;
//	cnz0 = (nu0+nx0+1+ncl-1)/ncl*ncl;
	cnux0 = (nux0+ncl-1)/ncl*ncl;
	cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;
	// copy q in l
	for(ii=0; ii<nu0+nx0; ii++) hl[N][ii] = hq[N][ii];
	// box constraints
	if(nb0>0)
		{
		d_update_vector_sparse(nb[N], idxb[N], hl[N], hql[N]);
		}
	// general constraints
	if(ng0>0)
		{
		cng0 = (ng0+ncl-1)/ncl*ncl;
		dgemv_n_lib(nux0, ng0, hpDCt[N], cng0, qx[N], 1, hl[N], hl[N]);
		}

	// middle stages
	for(nn=0; nn<N; nn++)
		{
		nu1 = nu0;
		nx1 = nx0;
		nux1 = nux0;
		pnx1 = pnx0;
		cnx1 = cnx0;
		cnl1 = cnl0;
		nu0 = nu[N-nn-1];
		nx0 = nx[N-nn-1];
		nb0 = nb[N-nn-1];
		ng0 = ng[N-nn-1];
		nux0 = nu0+nx0;
		pnx0 = (nx0+bs-1)/bs*bs;
		cnx0 = (nx0+ncl-1)/ncl*ncl;
//		cnz0 = (nu0+nx0+1+ncl-1)/ncl*ncl;
		cnux0 = (nux0+ncl-1)/ncl*ncl;
		cnl0 = cnux0<cnx0+ncl ? cnx0+ncl : cnux0;
		if(compute_Pb)
			{
//			for(jj=0; jj<nx1; jj++) work[jj] = hux[N-nn][nu1+jj]; // copy b in aligned memory
//			dtrmv_u_n_lib(nx1, hpL[N-nn]+ncl*bs, cnl1, work, 0, work+pnx1);
			dtrmv_u_n_lib(nx1, hpL[N-nn]+ncl*bs, cnl1, hb[N-nn-1], 0, work);
			dtrmv_u_t_lib(nx1, hpL[N-nn]+ncl*bs, cnl1, work, 0, hPb[N-nn-1]); // L*(L'*b)
			}
		// copy q in l
		for(ii=0; ii<nux0; ii++) hl[N-nn-1][ii] = hq[N-nn-1][ii];
		// box constraints
		if(nb0>0)
			{
			d_update_vector_sparse(nb[N-nn-1], idxb[N-nn-1], hl[N-nn-1], hql[N-nn-1]);
			}
		// general constraints
		if(ng0>0)
			{
			cng0 = (ng0+ncl-1)/ncl*ncl;
			dgemv_n_lib(nux0, ng0, hpDCt[N-nn-1], cng0, qx[N-nn-1], 1, hl[N-nn-1], hl[N-nn-1]);
			}
		for(jj=0; jj<nx1; jj++) work[jj] = hPb[N-nn-1][jj] + hl[N-nn][nu1+jj]; // add p
		dgemv_n_lib(nux0, nx1, hpBAbt[N-nn-1], cnx1, work, 1, hl[N-nn-1], hl[N-nn-1]);
		dtrsv_n_lib(nux0, nu0, hpL[N-nn-1], cnl0, 1, hdL[N-nn-1], hl[N-nn-1], hl[N-nn-1]);
		}


	// forward substitution 

	nu1 = nu[0];
	nx1 = nx[0];
	nux1 = nu1+nx1;
	cnx1 = (nx1+ncl-1)/ncl*ncl;
//	cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
	cnux1 = (nux1+ncl-1)/ncl*ncl;
	cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
	// first stage
	nn = 0;
	nu0 = nu1;
	nx0 = nx1;
	cnx0 = cnx1;
	cnux0 = cnux1;
//	cnz0 = cnz1;
	cnl0 = cnl1;
	nu1 = nu[nn+1];
	nx1 = nx[nn+1];
	nux1 = nu1+nx1;
	pnx1  = (nx1+bs-1)/bs*bs;
	cnx1 = (nx1+ncl-1)/ncl*ncl;
//	cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
	cnux1 = (nux1+ncl-1)/ncl*ncl;
	cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
	for(jj=0; jj<nu0; jj++) hux[nn][jj] = - hl[nn][jj];
	dtrsv_t_lib(nux0, nux0, hpL[nn], cnl0, 1, hdL[nn], &hux[nn][0], &hux[nn][0]);
	dgemv_t_lib(nux0, nx1, hpBAbt[nn], cnx1, &hux[nn][0], 1, &hb[nn][0], &hux[nn+1][nu1]);
	if(compute_pi)
		{
		for(jj=0; jj<nx1; jj++) work[pnx1+jj] = hux[nn+1][nu1+jj]; // copy x into aligned memory
		dtrmv_u_n_lib(nx1, hpL[nn+1]+ncl*bs, cnl1, &work[pnx1], 0, &work[0]);
		dtrmv_u_t_lib(nx1, hpL[nn+1]+ncl*bs, cnl1, &work[0], 0, &hpi[nn+1][0]); // L*(L'*b) + p
		for(jj=0; jj<nx1; jj++) hpi[nn+1][jj] += hl[nn+1][nu1+jj];
		}
	// middle stages
	for(nn=1; nn<N; nn++)
		{
		nu0 = nu1;
		nx0 = nx1;
		nux0 = nux1;
		cnx0 = cnx1;
//		cnz0 = cnz1;
		cnl0 = cnl1;
		nu1 = nu[nn+1];
		nx1 = nx[nn+1];
		nux1 = nu1+nx1;
		pnx1  = (nx1+bs-1)/bs*bs;
		cnx1 = (nx1+ncl-1)/ncl*ncl;
//		cnz1 = (nu1+nx1+1+ncl-1)/ncl*ncl;
		cnux1 = (nux1+ncl-1)/ncl*ncl;
		cnl1 = cnux1<cnx1+ncl ? cnx1+ncl : cnux1;
		for(jj=0; jj<nu0; jj++) hux[nn][jj] = - hl[nn][jj];
		dtrsv_t_lib(nux0, nu0, hpL[nn], cnl0, 1, hdL[nn], &hux[nn][0], &hux[nn][0]);
		dgemv_t_lib(nux0, nx1, hpBAbt[nn], cnx1, &hux[nn][0], 1, &hb[nn][0], &hux[nn+1][nu1]);
		if(compute_pi)
			{
			for(jj=0; jj<nx1; jj++) work[pnx1+jj] = hux[nn+1][nu1+jj]; // copy x into aligned memory
			dtrmv_u_n_lib(nx1, hpL[nn+1]+ncl*bs, cnl1, &work[pnx1], 0, &work[0]);
			dtrmv_u_t_lib(nx1, hpL[nn+1]+ncl*bs, cnl1, &work[0], 0, &hpi[nn+1][0]); // L*(L'*b) + p
			for(jj=0; jj<nx1; jj++) hpi[nn+1][jj] += hl[nn+1][nu1+jj];
			}
		}

	}




