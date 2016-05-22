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



int d_back_ric_rec_sv_tv_work_space_size_doubles(int N, int *nx, int *nu, int *nb, int *ng)
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
	
	int pnzM = (nzM+bs-1)/bs*bs;
	int cnxgM = (nxgM+ncl-1)/ncl*ncl;
	
	int size = pnzM*cnxgM + pnzM;

	return size;
	}



/* backward riccati recursion, factorization & solution */
/* version supporting stage-variant nx, nu, nb, ng */
void d_back_ric_rec_sv_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hux, double **hpL, double **hdL, double *work, int compute_Pb, double **hPb, int compute_pi, double **hpi, int *nb, int **idxb, double **hQd, double **hQl, int *ng, double **hpDCt, double **Qx, double **qx)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	
	int ii, jj, ll, nn;

	double *work0, *work1;

	// compute sizes of matrices TODO pass them instead of compute them ???
	int nux[N+1];
	int nz[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cnl[N+1];
	int cng[N+1];
	int cnxg[N+1];
	int pnx[N+1];

	for(nn=0; nn<N; nn++)
		{
		nux[nn] = nu[nn]+nx[nn];
		nz[nn] = nux[nn]+1;
		cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
		cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
		cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];
		cnxg[nn] = (nx[nn+1]+ng[nn]+ncl-1)/ncl*ncl;
		pnx[nn] = (nx[nn]+bs-1)/bs*bs;
		}
	nn = N;
	nux[nn] = nu[nn]+nx[nn];
	nz[nn] = nux[nn]+1;
	cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
	cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
	cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];
	pnx[nn] = (nx[nn]+bs-1)/bs*bs;



	// factorization and backward substitution 

	// final stage 
	
	work0 = work;
	
	if(ng[N]>0)
		{
		cng[N] = (ng[N]+ncl-1)/ncl*ncl;
		dgemm_diag_right_lib(nux[N], ng[N], hpDCt[N], cng[N], Qx[N], 0, work0, cng[N], work0, cng[N]);
		drowin_lib(ng[N], qx[N], work0+nux[N]/bs*cng[N]*bs+nux[N]%bs);
		}
	if(nb[N]>0)
		{
		ddiain_libsp(nb[N], idxb[N], hQd[N], hpQ[N], cnux[N]);
		drowin_libsp(nb[N], idxb[N], hQl[N], hpQ[N]+nux[N]/bs*bs*cnux[N]+nux[N]%bs);
		}

#ifdef BLASFEO
	dsyrk_dpotrf_ntnn_l_lib(nz[N], nux[N], ng[N], work0, cng[N], work0, cng[N], 1, hpQ[N], cnux[N], hpL[N], cnl[N], hdL[N]);
#else
	dsyrk_dpotrf_lib(nz[N], nux[N], ng[N], work0, cng[N], work0, cng[N], 1, hpQ[N], cnux[N], hpL[N], cnl[N], hdL[N]);
#endif

	dtrtr_l_lib(nx[N], nu[N], hpL[N]+nu[N]/bs*bs*cnl[N]+nu[N]%bs+nu[N]*bs, cnl[N], 0, hpL[N]+ncl*bs, cnl[N]);	



	// middle stages 
	for(nn=0; nn<N; nn++)
		{	

		work1 = work;
		work0 = work + pnx[N-nn];

#ifdef BLASFEO
		dtrmm_ntnn_lu_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], 0, work0, cnxg[N-nn-1], work0, cnxg[N-nn-1]);
#else
		dtrmm_nt_u_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work0, cnxg[N-nn-1]);
#endif

		if(compute_Pb)
			{
			for(jj=0; jj<nx[N-nn]; jj++) work1[jj] = work0[nux[N-nn-1]/bs*bs*cnxg[N-nn-1]+nux[N-nn-1]%bs+jj*bs];
#ifdef BLASFEO
			dtrmv_ut_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work1, 0, hPb[N-nn-1], hPb[N-nn-1]); // L*(L'*b)
#else
			dtrmv_u_t_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work1, 0, hPb[N-nn-1]); // L*(L'*b)
#endif
			}
		dgead_lib(1, nx[N-nn], 1.0, nux[N-nn], hpL[N-nn]+nux[N-nn]/bs*bs*cnl[N-nn]+nux[N-nn]%bs+nu[N-nn]*bs, cnl[N-nn], nux[N-nn-1], work0+nux[N-nn-1]/bs*bs*cnxg[N-nn-1]+nux[N-nn-1]%bs, cnxg[N-nn-1]);
		if(ng[N-nn-1]>0)
			{
			cng[N-nn-1] = (ng[N-nn-1]+ncl-1)/ncl*ncl;
			dgemm_diag_right_lib(nux[N-nn-1], ng[N-nn-1], hpDCt[N-nn-1], cng[N-nn-1], Qx[N-nn-1], 0, work0+nx[N-nn]*bs, cnxg[N-nn-1], work0+nx[N-nn]*bs, cnxg[N-nn-1]);
			drowin_lib(ng[N-nn-1], qx[N-nn-1], work0+nux[N-nn-1]/bs*cnxg[N-nn-1]*bs+nux[N-nn-1]%bs+nx[N-nn]*bs);
			}
		if(nb[N-nn-1]>0)
			{
			ddiain_libsp(nb[N-nn-1], idxb[N-nn-1], hQd[N-nn-1], hpQ[N-nn-1], cnux[N-nn-1]);
			drowin_libsp(nb[N-nn-1], idxb[N-nn-1], hQl[N-nn-1], hpQ[N-nn-1]+nux[N-nn-1]/bs*bs*cnux[N-nn-1]+nux[N-nn-1]%bs);
			}

#ifdef BLASFEO
		dsyrk_dpotrf_ntnn_l_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn]+ng[N-nn-1], work0, cnxg[N-nn-1], work0, cnxg[N-nn-1], 1, hpQ[N-nn-1], cnux[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], hdL[N-nn-1]);
#else
		dsyrk_dpotrf_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn]+ng[N-nn-1], work0, cnxg[N-nn-1], work0, cnxg[N-nn-1], 1, hpQ[N-nn-1], cnux[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], hdL[N-nn-1]);
#endif

		dtrtr_l_lib(nx[N-nn-1], nu[N-nn-1], hpL[N-nn-1]+nu[N-nn-1]/bs*bs*cnl[N-nn-1]+nu[N-nn-1]%bs+nu[N-nn-1]*bs, cnl[N-nn-1], 0, hpL[N-nn-1]+ncl*bs, cnl[N-nn-1]);	

		}



	// forward substitution 

	work1 = work;

	// first stage
	nn = 0;
	for(jj=0; jj<nux[nn]; jj++) hux[nn][jj] = - hpL[nn][nux[nn]/bs*bs*cnl[nn]+nux[nn]%bs+bs*jj];
#ifdef BLASFEO
	dtrsv_lt_inv_lib(nux[nn], nux[nn], hpL[nn], cnl[nn], hdL[nn], hux[nn], hux[nn]);
#else
	dtrsv_t_lib(nux[nn], nux[nn], hpL[nn], cnl[nn], 1, hdL[nn], hux[nn], hux[nn]);
#endif
	for(jj=0; jj<nx[nn+1]; jj++) hux[nn+1][nu[nn+1]+jj] = hpBAbt[nn][nux[nn]/bs*bs*cnx[nn+1]+nux[nn]%bs+bs*jj];
#ifdef BLASFEO
	dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hux[nn+1]+nu[nn+1], hux[nn+1]+nu[nn+1]);
#else
	dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hux[nn+1]+nu[nn+1], hux[nn+1]+nu[nn+1]);
#endif
	if(compute_pi) // TODO change pi index !!!!!!!!!! done
		{
		for(jj=0; jj<nx[nn+1]; jj++) work1[pnx[nn+1]+jj] = hux[nn+1][nu[nn+1]+jj]; // copy x into aligned memory
		for(jj=0; jj<nx[nn+1]; jj++) work1[jj] = hpL[nn+1][nux[nn+1]/bs*bs*cnl[nn+1]+nux[nn+1]%bs+bs*(nu[nn+1]+jj)]; // work space
#ifdef BLASFEO
		dtrmv_un_lib(nx[nn+1], hpL[nn+1]+(ncl)*bs, cnl[nn+1], work1+pnx[nn+1], 1, work1, work1);
		dtrmv_ut_lib(nx[nn+1], hpL[nn+1]+(ncl)*bs, cnl[nn+1], work1, 0, hpi[nn], hpi[nn]); // L*(L'*b) + p
#else
		dtrmv_u_n_lib(nx[nn+1], hpL[nn+1]+(ncl)*bs, cnl[nn+1], work1+pnx[nn+1], 1, work1);
		dtrmv_u_t_lib(nx[nn+1], hpL[nn+1]+(ncl)*bs, cnl[nn+1], work1, 0, hpi[nn]); // L*(L'*b) + p
#endif
		}

	// middle stages
	for(nn=1; nn<N; nn++)
		{
		for(jj=0; jj<nu[nn]; jj++) hux[nn][jj] = - hpL[nn][nux[nn]/bs*bs*cnl[nn]+nux[nn]%bs+bs*jj];
#ifdef BLASFEO
		dtrsv_lt_inv_lib(nux[nn], nu[nn], hpL[nn], cnl[nn], hdL[nn], hux[nn], hux[nn]);
#else
		dtrsv_t_lib(nux[nn], nu[nn], hpL[nn], cnl[nn], 1, hdL[nn], hux[nn], hux[nn]);
#endif
		for(jj=0; jj<nx[nn+1]; jj++) hux[nn+1][nu[nn+1]+jj] = hpBAbt[nn][nux[nn]/bs*bs*cnx[nn+1]+nux[nn]%bs+bs*jj];
#ifdef BLASFEO
		dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hux[nn+1]+nu[nn+1], hux[nn+1]+nu[nn+1]);
#else
		dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hux[nn+1]+nu[nn+1], hux[nn+1]+nu[nn+1]);
#endif
		if(compute_pi) // TODO change pi index !!!!!!!!!!
			{
			for(jj=0; jj<nx[nn+1]; jj++) work1[pnx[nn+1]+jj] = hux[nn+1][nu[nn+1]+jj]; // copy x into aligned memory
			for(jj=0; jj<nx[nn+1]; jj++) work1[jj] = hpL[nn+1][nux[nn+1]/bs*bs*cnl[nn+1]+nux[nn+1]%bs+bs*(nu[nn+1]+jj)]; // work space
#ifdef BLASFEO
			dtrmv_un_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work1+pnx[nn+1], 1, work1, work1);
			dtrmv_ut_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work1, 0, hpi[nn], hpi[nn]); // L*(L'*b) + p
#else
			dtrmv_u_n_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work1+pnx[nn+1], 1, work1);
			dtrmv_u_t_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work1, 0, hpi[nn]); // L*(L'*b) + p
#endif
			}
		}
	
	}



/* backward riccati recursion, factorization */
/* version supporting stage-variant nx, nu, nb, ng */
void d_back_ric_rec_trf_tv(int N, int *nx, int *nu, double **hpBAbt, double **hpQ, double **hpL, double **hdL, double *work, int *nb, int **idxb, double **hQd, int *ng, double **hpDCt, double **Qx)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	
	int ii, jj, ll, nn;

	// compute sizes of matrices TODO pass them instead of compute them ???
	int nux[N+1];
	int nz[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cnl[N+1];
	int cng[N+1];
	int cnxg[N+1];

	for(nn=0; nn<N; nn++)
		{
		nux[nn] = nu[nn]+nx[nn];
		nz[nn] = nux[nn]+1;
		cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
		cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
		cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];
		cnxg[nn] = (nx[nn+1]+ng[nn]+ncl-1)/ncl*ncl;
		}
	nn = N;
	nux[nn] = nu[nn]+nx[nn];
	nz[nn] = nux[nn]+1;
	cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
	cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
	cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];



	// factorization and backward substitution 

	// final stage 

	if(ng[N]>0)
		{
		cng[N] = (ng[N]+ncl-1)/ncl*ncl;
		dgemm_diag_right_lib(nux[N], ng[N], hpDCt[N], cng[N], Qx[N], 0, work, cng[N], work, cng[N]);
		}
	if(nb[N]>0)
		{
		ddiain_libsp(nb[N], idxb[N], hQd[N], hpQ[N], cnux[N]);
		}

#ifdef BLASFEO
	dsyrk_dpotrf_ntnn_l_lib(nux[N], nux[N], ng[N], work, cng[N], work, cng[N], 1, hpQ[N], cnux[N], hpL[N], cnl[N], hdL[N]);
#else
	dsyrk_dpotrf_lib(nux[N], nux[N], ng[N], work, cng[N], work, cng[N], 1, hpQ[N], cnux[N], hpL[N], cnl[N], hdL[N]);
#endif

	dtrtr_l_lib(nx[N], nu[N], hpL[N]+nu[N]/bs*bs*cnl[N]+nu[N]%bs+nu[N]*bs, cnl[N], 0, hpL[N]+ncl*bs, cnl[N]);	



	// middle stages 
	for(nn=0; nn<N; nn++)
		{	

#ifdef BLASFEO
		dtrmm_ntnn_lu_lib(nux[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], 0, work, cnxg[N-nn-1], work, cnxg[N-nn-1]);
#else
		dtrmm_nt_u_lib(nux[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work, cnxg[N-nn-1]);
#endif

		if(ng[N-nn-1]>0)
			{
			cng[N-nn-1] = (ng[N-nn-1]+ncl-1)/ncl*ncl;
			dgemm_diag_right_lib(nux[N-nn-1], ng[N-nn-1], hpDCt[N-nn-1], cng[N-nn-1], Qx[N-nn-1], 0, work+nx[N-nn]*bs, cnxg[N-nn-1], work+nx[N-nn]*bs, cnxg[N-nn-1]);
			}
		if(nb[N-nn-1]>0)
			{
			ddiain_libsp(nb[N-nn-1], idxb[N-nn-1], hQd[N-nn-1], hpQ[N-nn-1], cnux[N-nn-1]);
			}

#ifdef BLASFEO
		dsyrk_dpotrf_ntnn_l_lib(nux[N-nn-1], nux[N-nn-1], nx[N-nn]+ng[N-nn-1], work, cnxg[N-nn-1], work, cnxg[N-nn-1], 1, hpQ[N-nn-1], cnux[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], hdL[N-nn-1]);
#else
		dsyrk_dpotrf_lib(nux[N-nn-1], nux[N-nn-1], nx[N-nn]+ng[N-nn-1], work, cnxg[N-nn-1], work, cnxg[N-nn-1], 1, hpQ[N-nn-1], cnux[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], hdL[N-nn-1]);
#endif

		dtrtr_l_lib(nx[N-nn-1], nu[N-nn-1], hpL[N-nn-1]+nu[N-nn-1]/bs*bs*cnl[N-nn-1]+nu[N-nn-1]%bs+nu[N-nn-1]*bs, cnl[N-nn-1], 0, hpL[N-nn-1]+ncl*bs, cnl[N-nn-1]);	

		}

	}



/* backward riccati recursion, solution */
/* version supporting stage-variant nx, nu, nb, ng */
void d_back_ric_rec_trs_tv(int N, int *nx, int *nu, double **hpBAbt, double **hb, double **hpL, double **hdL, double **hq, double **hl, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi, int *nb, int **idxb, double **hql, int *ng, double **hpDCt, double **qx)
	{
	
	const int bs  = D_MR;
	const int ncl = D_NCL;

	int ii, jj, nn;
	
	// compute sizes of matrices TODO pass them instead of compute them ???
	int nux[N+1];
	int nz[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cnl[N+1];
	int cng[N+1];
	int cnxg[N+1];
	int pnx[N+1];

	for(nn=0; nn<N; nn++)
		{
		nux[nn] = nu[nn]+nx[nn];
		nz[nn] = nux[nn]+1;
		cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
		cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
		cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];
		cnxg[nn] = (nx[nn+1]+ng[nn]+ncl-1)/ncl*ncl;
		pnx[nn] = (nx[nn]+bs-1)/bs*bs;
		}
	nn = N;
	nux[nn] = nu[nn]+nx[nn];
	nz[nn] = nux[nn]+1;
	cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
	cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
	cnl[nn] = cnux[nn]<cnx[nn]+ncl ? cnx[nn]+ncl : cnux[nn];
	pnx[nn] = (nx[nn]+bs-1)/bs*bs;



	// backward substitution 

	// final stage
	// copy q in l
	for(ii=0; ii<nux[N]; ii++) hl[N][ii] = hq[N][ii];
	// box constraints
	if(nb[N]>0)
		{
		dvecin_libsp(nb[N], idxb[N], hql[N], hl[N]);
		}
	// general constraints
	if(ng[N]>0)
		{
		cng[N] = (ng[N]+ncl-1)/ncl*ncl;
#ifdef BLASFEO
		dgemv_n_lib(nux[N], ng[N], hpDCt[N], cng[N], qx[N], 1, hl[N], hl[N]);
#else
		dgemv_n_lib(nux[N], ng[N], hpDCt[N], cng[N], qx[N], 1, hl[N], hl[N]);
#endif
		}

	// middle stages
	for(nn=0; nn<N; nn++)
		{
		if(compute_Pb)
			{
#ifdef BLASFEO
			dtrmv_un_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], hb[N-nn-1], 0, work, work);
			dtrmv_ut_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work, 0, hPb[N-nn-1], hPb[N-nn-1]); // L*(L'*b)
#else
			dtrmv_u_n_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], hb[N-nn-1], 0, work);
			dtrmv_u_t_lib(nx[N-nn], hpL[N-nn]+ncl*bs, cnl[N-nn], work, 0, hPb[N-nn-1]); // L*(L'*b)
#endif
			}
		// copy q in l
		for(ii=0; ii<nux[N-nn-1]; ii++) hl[N-nn-1][ii] = hq[N-nn-1][ii];
		// box constraints
		if(nb[N-nn-1]>0)
			{
			dvecin_libsp(nb[N-nn-1], idxb[N-nn-1], hql[N-nn-1], hl[N-nn-1]);
			}
		// general constraints
		if(ng[N-nn-1]>0)
			{
			cng[N-nn-1] = (ng[N-nn-1]+ncl-1)/ncl*ncl;
#ifdef BLASFEO
			dgemv_n_lib(nux[N-nn-1], ng[N-nn-1], hpDCt[N-nn-1], cng[N-nn-1], qx[N-nn-1], 1, hl[N-nn-1], hl[N-nn-1]);
#else
			dgemv_n_lib(nux[N-nn-1], ng[N-nn-1], hpDCt[N-nn-1], cng[N-nn-1], qx[N-nn-1], 1, hl[N-nn-1], hl[N-nn-1]);
#endif
			}
		for(jj=0; jj<nx[N-nn]; jj++) work[jj] = hPb[N-nn-1][jj] + hl[N-nn][nu[N-nn]+jj]; // add p
#ifdef BLASFEO
		dgemv_n_lib(nux[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], work, 1, hl[N-nn-1], hl[N-nn-1]);
		dtrsv_ln_inv_lib(nux[N-nn-1], nu[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], hdL[N-nn-1], hl[N-nn-1], hl[N-nn-1]);
#else
		dgemv_n_lib(nux[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], work, 1, hl[N-nn-1], hl[N-nn-1]);
		dtrsv_n_lib(nux[N-nn-1], nu[N-nn-1], hpL[N-nn-1], cnl[N-nn-1], 1, hdL[N-nn-1], hl[N-nn-1], hl[N-nn-1]);
#endif
		}


	// forward substitution 

	// first stage
	nn = 0;
	for(jj=0; jj<nux[nn]; jj++) hux[nn][jj] = - hl[nn][jj]; 
#ifdef BLASFEO
	dtrsv_lt_inv_lib(nux[nn], nux[nn], hpL[nn], cnl[nn], hdL[nn], hux[nn], hux[nn]);
	dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hb[nn], hux[nn+1]+nu[nn+1]);
#else
	dtrsv_t_lib(nux[nn], nux[nn], hpL[nn], cnl[nn], 1, hdL[nn], hux[nn], hux[nn]);
	dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hb[nn], hux[nn+1]+nu[nn+1]);
#endif
	if(compute_pi) // TODO change pi index !!!!!!!!!!
		{
		for(jj=0; jj<nx[nn+1]; jj++) work[pnx[nn+1]+jj] = hux[nn+1][nu[nn+1]+jj]; // copy x into aligned memory
#ifdef BLASFEO
		dtrmv_un_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work+pnx[nn+1], 0, work, work);
		dtrmv_ut_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work, 0, hpi[nn], hpi[nn]); // L*(L'*b) + p
#else
		dtrmv_u_n_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work+pnx[nn+1], 0, work);
		dtrmv_u_t_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work, 0, hpi[nn]); // L*(L'*b) + p
#endif
		for(jj=0; jj<nx[nn+1]; jj++) hpi[nn][jj] += hl[nn+1][nu[nn+1]+jj];
		}
	// middle stages
	for(nn=1; nn<N; nn++)
		{
		for(jj=0; jj<nu[nn]; jj++) hux[nn][jj] = - hl[nn][jj];
#ifdef BLASFEO
		dtrsv_lt_inv_lib(nux[nn], nu[nn], hpL[nn], cnl[nn], hdL[nn], hux[nn], hux[nn]);
		dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hb[nn], hux[nn+1]+nu[nn+1]);
#else
		dtrsv_t_lib(nux[nn], nu[nn], hpL[nn], cnl[nn], 1, hdL[nn], hux[nn], hux[nn]);
		dgemv_t_lib(nux[nn], nx[nn+1], hpBAbt[nn], cnx[nn+1], hux[nn], 1, hb[nn], hux[nn+1]+nu[nn+1]);
#endif
		if(compute_pi) // TODO change pi index !!!!!!!!!!
			{
			for(jj=0; jj<nx[nn+1]; jj++) work[pnx[nn+1]+jj] = hux[nn+1][nu[nn+1]+jj]; // copy x into aligned memory
#ifdef BLASFEO
			dtrmv_un_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work+pnx[nn+1], 0, work, work);
			dtrmv_ut_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work, 0, hpi[nn], hpi[nn]); // L*(L'*b) + p
#else
			dtrmv_u_n_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work+pnx[nn+1], 0, work);
			dtrmv_u_t_lib(nx[nn+1], hpL[nn+1]+ncl*bs, cnl[nn+1], work, 0, hpi[nn]); // L*(L'*b) + p
#endif
			for(jj=0; jj<nx[nn+1]; jj++) hpi[nn][jj] += hl[nn+1][nu[nn+1]+jj];
			}
		}

	}




