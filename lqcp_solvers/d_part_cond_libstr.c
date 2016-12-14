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
#include <stdlib.h>
#include <stdio.h>

#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/block_size.h"
#include "../include/lqcp_aux.h"

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>



#if 0
// pGamma_x = [A_0' A_0'A_1' A_0'A_1'A_2' ..... A_0'...A_{N-1}'], there is not I at the beginning !!!
// pGamma[ii] has size nx[0] x nx[ii+1]
void d_cond_A(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_x0, double *pBAbt2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	

	dgecp_lib(nx[0], nx[1], nu[0], pBAbt[0]+nu[0]/bs*bs*cnx[1]+nu[0]%bs, cnx[1], 0, pGamma_x0[0], cnx[1]);

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		dgetr_lib(nx[ii], nx[ii+1], 1.0, nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		dgemm_nt_lib(nx[0], nx[ii+1], nx[ii], pGamma_x0[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_x0[ii], cnx[ii+1], pGamma_x0[ii], cnx[ii+1], 0, 0);
		}
	
	dgecp_lib(nx[0], nx[N], 0, pGamma_x0[N-1], cnx[N], nu2, pBAbt2+nu2/bs*bs*cnx[N]+nu2%bs, cnx[N]);

	}



//void d_cond_B(int N, int nx, int nu, double **pA, double **pBt, int compute_Gamma_u, double **pGamma_u, double *pH_B)
void d_cond_B(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_u, double *pBAbt2)
	{
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	int nu_tmp = 0;

	// Gamma_u
	dgecp_lib(nu[0], nx[1], 0, pBAbt[0], cnx[1], 0, pGamma_u[0], cnx[1]);
	nu_tmp += nu[0];

	for(ii=1; ii<N; ii++)
		{
		dgetr_lib(nx[ii], nx[ii+1], 1.0, nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx[ii+1], nu_tmp, nx[ii], work, cnx[ii], pGamma_u[ii-1], cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib(nu_tmp, nx[ii+1], nx[ii], pGamma_u[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 0); // Gamma_u * A^T
#endif
		dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], nu_tmp, pGamma_u[ii]+nu_tmp/bs*bs*cnx[ii+1]+nu_tmp%bs, cnx[ii+1]);
		nu_tmp += nu[ii];
		}
	
	dgecp_lib(nu_tmp, nx[N], 0, pGamma_u[N-1], cnx[N], 0, pBAbt2, cnx[N]);

	}



void d_cond_b(int N, int *nx, int *nu, double **pBAbt, double *work, double **Gamma_b, double *pBAbt2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	// Gamma_b
	ii = 0;
	dgetr_lib(1, nx[ii+1], 1.0, nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	for(ii=1; ii<N; ii++)
		{
		dgetr_lib(nx[ii], nx[ii+1], 1.0, nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		dgetr_lib(1, nx[ii+1], 1.0, nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		}
	
	dgetr_lib(nx[N], 1, 1.0, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);
	
	}
	


void d_cond_BAb(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_u, double **pGamma_x0, double **Gamma_b, double *pBAbt2)
	{

	// XXX can merge Gamma matrices ???
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	int nu_tmp = 0;

	ii = 0;
	// B
	dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], 0, pGamma_u[ii], cnx[ii+1]);
	// A
	dgecp_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, pGamma_x0[ii], cnx[ii+1]);
	// b
	dgetr_lib(1, nx[ii+1], 1.0, nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	//
	nu_tmp += nu[0];

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		dgetr_lib(nx[ii], nx[ii+1], 1.0, nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		// B
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx[ii+1], nu_tmp, nx[ii], work, cnx[ii], pGamma_u[ii-1], cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib(nu_tmp, nx[ii+1], nx[ii], pGamma_u[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 0); // Gamma_u * A^T
#endif
		dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], nu_tmp, pGamma_u[ii]+nu_tmp/bs*bs*cnx[ii+1]+nu_tmp%bs, cnx[ii+1]);
		// A
		dgemm_nt_lib(nx[0], nx[ii+1], nx[ii], pGamma_x0[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_x0[ii], cnx[ii+1], pGamma_x0[ii], cnx[ii+1], 0, 0);
		// b
		dgetr_lib(1, nx[ii+1], 1.0, nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		//
		nu_tmp += nu[ii];
		}
	
	// B
	dgecp_lib(nu_tmp, nx[N], 0, pGamma_u[N-1], cnx[N], 0, pBAbt2, cnx[N]);
	// A
	dgecp_lib(nx[0], nx[N], 0, pGamma_x0[N-1], cnx[N], nu2, pBAbt2+nu2/bs*bs*cnx[N]+nu2%bs, cnx[N]);
	// b
	dgetr_lib(nx[N], 1, 1.0, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);

	}
#endif



void d_cond_BAbt_libstr(int N, int *nx, int *nu, struct d_strmat *hsBAbt, void *work, struct d_strmat *hsGamma, struct d_strmat *sBAbt2)
	{

	int ii, jj;

	int nu_tmp;

	nu_tmp = 0;
	ii = 0;
	// B & A & b
	dgecp_libstr(nu[0]+nx[0]+1, nx[1], 1.0, &hsBAbt[0], 0, 0, &hsGamma[0], 0, 0);
	//
	nu_tmp += nu[0];
	ii++;

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy

		struct d_strmat sA;
		d_create_strmat(nx[ii+1], nx[ii], &sA, work);
		
		// pA in work space
		dgetr_libstr(nx[ii], nx[ii+1], 1.0, &hsBAbt[ii], nu[ii], 0, &sA, 0, 0); // pA in work // TODO avoid copy for LA_BLAS and LA_REFERENCE

		// Gamma * A^T
		dgemm_nt_libstr(nu_tmp+nx[0]+1, nx[ii+1], nx[ii], 1.0, &hsGamma[ii-1], 0, 0, &sA, 0, 0, 0.0, &hsGamma[ii], nu[ii], 0, &hsGamma[ii], nu[ii], 0); // Gamma * A^T

		dgecp_libstr(nu[ii], nx[ii+1], 1.0, &hsBAbt[ii], 0, 0, &hsGamma[ii], 0, 0);

		nu_tmp += nu[ii];

		dgead_libstr(1, nx[ii+1], 1.0, &hsBAbt[ii], nu[ii]+nx[ii], 0, &hsGamma[ii], nu_tmp+nx[0], 0);
		}
	
	// B & A & b
	dgecp_libstr(nu_tmp+nx[0]+1, nx[N], 1.0, &hsGamma[N-1], 0, 0, sBAbt2, 0, 0);

	return;

	}



void d_cond_RSQrq_libstr(int N, int *nx, int *nu, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsGamma, void **work, struct d_strmat *sRSQrq2)
	{

	// early return
	if(N<1)
		return;

	int nn;

	int nu2[N+1];
	int nu3[N+1];
	nu2[0]= 0; // sum
	nu3[0]= 0; // reverse sum

	for(nn=0; nn<=N; nn++)
		{
		nu2[nn+1] = nu2[nn] + nu[nn];
		nu3[nn+1] = nu3[nn] + nu[N-nn-1];
		}

	struct d_strmat sL;
	struct d_strmat sM;
	struct d_strmat sLx;
	struct d_strmat sBAbtL;



	// early return
	if(N==1)
		{
		dgecp_libstr(nu[N-1]+nx[N-1]+1, nu[N-1]+nx[N-1], 1.0, &hsRSQrq[N-1], 0, 0, sRSQrq2, 0, 0);
		return;
		}



	// final stage 
	d_create_strmat(nu[N-1]+nx[N-1]+1, nu[N-1]+nx[N-1], &sL, (void *) work[0]);
	d_create_strmat(nu[N-1], nx[N-1], &sM, (void *) work[1]);

	dgecp_libstr(nu[N-1]+nx[N-1]+1, nu[N-1]+nx[N-1], 1.0, &hsRSQrq[N-1], 0, 0, &sL, 0, 0);

	// D
	dgecp_libstr(nu[N-1], nu[N-1], 1.0, &sL, 0, 0, sRSQrq2, nu3[0], nu3[0]);

	// M
	dgetr_libstr(nx[N-1], nu[N-1], 1.0, &sL, nu[N-1], 0, &sM, 0, 0);

	dgemm_nt_libstr(nu2[N-1]+nx[0]+1, nu[N-1], nx[N-1], 1.0, &hsGamma[N-2], 0, 0, &sM, 0, 0, 0.0, sRSQrq2, nu3[1], nu3[0], sRSQrq2, nu3[1], nu3[0]);

	// m
	dgead_libstr(1, nu[N-1], 1.0, &sL, nu[N-1]+nx[N-1], 0, sRSQrq2, nu2[N]+nx[0], nu3[0]);



	// middle stages 
	for(nn=1; nn<N-1; nn++)
		{	

		d_create_strmat(nx[N-nn]+1, nx[N-nn], &sLx, (void *) work[2]);
		d_create_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], &sBAbtL, (void *) work[3]);

		dgecp_libstr(nx[N-nn]+1, nx[N-nn], 1.0, &sL, nu[N-nn], nu[N-nn], &sLx, 0, 0);

		dpotrf_l_libstr(nx[N-nn]+1, nx[N-nn], &sLx, 0, 0, &sLx, 0, 0);

		dtrtr_l_libstr(nx[N-nn], 1.0, &sLx, 0, 0, &sLx, 0, 0);

		dtrmm_rutn_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], 1.0, &hsBAbt[N-nn-1], 0, 0, &sLx, 0, 0, 0.0, &sBAbtL, 0, 0, &sBAbtL, 0, 0);

		dgead_libstr(1, nx[N-nn], 1.0, &sLx, nx[N-nn], 0, &sBAbtL, nu[N-nn-1]+nx[N-nn-1], 0);

		d_create_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], &sL, (void *) work[0]);
		d_create_strmat(nu[N-nn-1], nx[N-nn-1], &sM, (void *) work[1]);

		dsyrk_ln_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &sBAbtL, 0, 0, &sBAbtL, 0, 0, 1.0, &hsRSQrq[N-nn-1], 0, 0, &sL, 0, 0);

		// D
		dgecp_libstr(nu[N-nn-1], nu[N-nn-1], 1.0, &sL, 0, 0, sRSQrq2, nu3[nn], nu3[nn]);

		// M
		dgetr_libstr(nx[N-nn-1], nu[N-nn-1], 1.0, &sL, nu[N-nn-1], 0, &sM, 0, 0);

		dgemm_nt_libstr(nu2[N-nn-1]+nx[0]+1, nu[N-nn-1], nx[N-nn-1], 1.0, &hsGamma[N-nn-2], 0, 0, &sM, 0, 0, 0.0, sRSQrq2, nu3[nn+1], nu3[nn], sRSQrq2, nu3[nn+1], nu3[nn]);

		// m
		dgead_libstr(1, nu[N-nn-1], 1.0, &sL, nu[N-nn-1]+nx[N-nn-1], 0, sRSQrq2, nu2[N]+nx[0], nu3[nn]);

		}

	// first stage
	nn = N-1;

	d_create_strmat(nx[N-nn]+1, nx[N-nn], &sLx, (void *) work[2]);
	d_create_strmat(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], &sBAbtL, (void *) work[3]);
	
	dgecp_libstr(nx[N-nn]+1, nx[N-nn], 1.0, &sL, nu[N-nn], nu[N-nn], &sLx, 0, 0);

	dpotrf_l_libstr(nx[N-nn]+1, nx[N-nn], &sLx, 0, 0, &sLx, 0, 0);

	dtrtr_l_libstr(nx[N-nn], 1.0, &sLx, 0, 0, &sLx, 0, 0);	

	dtrmm_rutn_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nx[N-nn], 1.0, &hsBAbt[N-nn-1], 0, 0, &sLx, 0, 0, 0.0, &sBAbtL, 0, 0, &sBAbtL, 0, 0);

	dgead_libstr(1, nx[N-nn], 1.0, &sLx, nx[N-nn], 0, &sBAbtL, nu[N-nn-1]+nx[N-nn-1], 0);

	dsyrk_ln_libstr(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], nx[N-nn], 1.0, &sBAbtL, 0, 0, &sBAbtL, 0, 0, 1.0, &hsRSQrq[N-nn-1], 0, 0, &sL, 0, 0);

	// D, M, m, P, p
	dgecp_libstr(nu[0]+nx[0]+1, nu[0]+nx[0], 1.0, &sL, 0, 0, sRSQrq2, nu3[N-1], nu3[N-1]); // TODO dtrcp for 'rectangular' matrices

	return;

	}



// TODO general constraints !!!!!
void d_cond_DCtd_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, double **hd, double **hpGamma, double *pDCt2, double *d2, int *idxb2)
	{

	// early return
	if(N<1)
		return;

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int pnb[N+1];
	int cnx[N+1];
	for(ii=0; ii<=N; ii++)
		{
		pnb[ii] = (nb[ii]+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		}

	int nbb = nb[0]; // box that remain box constraints
	int nbg = 0; // box that becomes general constraints
	for(ii=1; ii<N; ii++)
		for(jj=0; jj<nb[ii]; jj++)
			if(hidxb[ii][jj]<nu[ii])
				nbb++;
			else
				nbg++;
	
	int pnbb = (nbb+bs-1)/bs*bs;
	int pnbg = (nbg+bs-1)/bs*bs;
	int cnbg = (nbg+ncl-1)/ncl*ncl;

//	int nx_tmp = 0;
//	for(ii=1; ii<N; ii++)
//		nx_tmp += nx[ii];
	
//	int cnxm2 = (nx_tmp+ncl-1)/ncl*ncl;

//	int nu2 = 0;
//	for(ii=0; ii<N; ii++)
//		nu2 += nu[ii];

//	nx_tmp = 0;

	int nu_tmp = 0;

	int idx_gammab = nx[0];
	for(ii=0; ii<N-1; ii++)
		idx_gammab += nu[ii];

	int ib = 0;
	int ig = 0;

	double tmp;
	int idx_g;

	// middle stages
	for(ii=0; ii<N-1; ii++)
		{
		nu_tmp += nu[N-1-ii];
		for(jj=0; jj<nb[N-1-ii]; jj++)
			{
			if(hidxb[N-1-ii][jj]<nu[N-1-ii]) // input: box constraint
				{
				d2[0*pnbb+ib] = hd[N-1-ii][0*pnb[N-1-ii]+jj];
				d2[1*pnbb+ib] = hd[N-1-ii][1*pnb[N-1-ii]+jj];
				idxb2[ib] = nu_tmp - nu[N-1-ii] + hidxb[N-1-ii][jj];
				ib++;
				}
			else // state: general constraint
				{
				idx_g = hidxb[N-1-ii][jj]-nu[N-1-ii];
				tmp = hpGamma[N-2-ii][idx_gammab/bs*bs*cnx[N-1-ii]+idx_gammab%bs+idx_g*bs];
				d2[2*pnbb+0*pnbg+ig] = hd[N-1-ii][0*pnb[N-1-ii]+jj] - tmp;
				d2[2*pnbb+1*pnbg+ig] = hd[N-1-ii][1*pnb[N-1-ii]+jj] - tmp;
#ifdef BLASFEO
				dgecp_lib(idx_gammab, 1, 1.0, 0, hpGamma[N-ii-2]+idx_g*bs, cnx[N-ii-1], nu_tmp, pDCt2+nu_tmp/bs*bs*cnbg+nu_tmp%bs+ig*bs, cnbg);
#else
				dgecp_lib(idx_gammab, 1, 0, hpGamma[N-ii-2]+idx_g*bs, cnx[N-ii-1], nu_tmp, pDCt2+nu_tmp/bs*bs*cnbg+nu_tmp%bs+ig*bs, cnbg);
#endif
				ig++;
				}
			}
//		d_print_pmat(idx_gammab+1, nx[N-1-ii], bs, hpGamma[N-2-ii], cnx[N-1-ii]);
//		printf("\n%d %d\n", idx_gammab, cnx[N-1-ii]);
		idx_gammab -= nu[N-2-ii];
//		printf("\n%d\n", idx_gammab);
//		return;
		}

	// initial stage: both inputs and states as box constraints
	nu_tmp += nu[0];
	for(jj=0; jj<nb[0]; jj++)
		{
		d2[0*pnbb+ib] = hd[0][0*pnb[0]+jj];
		d2[1*pnbb+ib] = hd[0][1*pnb[0]+jj];
		idxb2[ib] = nu_tmp - nu[0] + hidxb[0][jj];
		ib++;
		}

//	for(ii=0; ii<N-1; ii++)
//		{
//		nu_tmp += nu[N-ii-1];
//		dgecp_lib(nu2-nu_tmp+nx[0], nx[N-ii-1], 0, hpGamma[N-ii-2], cnx[N-ii-1], nu_tmp, pDCt2+nu_tmp/bs*bs*cnxm2+nu_tmp%bs+nx_tmp*bs, cnxm2);
//		nx_tmp += nx[N-ii-1];
//		}

	return;

	}



// XXX does not compute hidxb2
void d_part_cond_compute_problem_size_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2)
	{

	int ii, jj, kk;

	int N1 = N/N2; // (floor) horizon of small blocks
	int R1 = N - N2*N1; // the first R1 blocks have horizon N1+1
	int M1 = R1>0 ? N1+1 : N1; // (ceil) horizon of large blocks
	int T1; // horizon of current block

	int N_tmp = 0; // temporary sum of horizons
	int nbb; // box constr that remain box constr
	int nbg; // box constr that becomes general constr
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;
		nx2[ii] = nx[N_tmp+0];
		nu2[ii] = nu[N_tmp+0];
		nb2[ii] = nb[N_tmp+0];
		ng2[ii] = ng[N_tmp+0];
		for(jj=1; jj<T1; jj++)
			{
			nbb = 0;
			nbg = 0;
			for(kk=0; kk<nb[N_tmp+jj]; kk++)
				if(hidxb[N_tmp+jj][kk]<nu[N_tmp+jj])
					nbb++;
				else
					nbg++;
			nx2[ii] += 0;
			nu2[ii] += nu[N_tmp+jj];
			nb2[ii] += nbb;
			ng2[ii] += ng[N_tmp+jj] + nbg;
			}
		N_tmp += T1;
		}
	nx2[N2] = nx[N];
	nu2[N2] = nu[N];
	nb2[N2] = nb[N];
	ng2[N2] = ng[N];

	}



int d_part_cond_work_space_size_bytes_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj, kk;
	int nu_tmp;

	// packing quantities - 1
	int pnx[N+1];
	int pnu[N+1];
	int pnz[N+1];
	int cnx[N+1];
	int cnu[N+1];
	for(ii=0; ii<=N; ii++)
		{
		pnx[ii] = (nx[ii]+bs-1)/bs*bs;
		pnu[ii] = (nu[ii]+bs-1)/bs*bs;
		pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		cnu[ii] = (nu[ii]+ncl-1)/ncl*ncl;
		}

	int N1 = N/N2; // (floor) horizon of small blocks
	int R1 = N - N2*N1; // the first R1 blocks have horizon N1+1
	int M1 = R1>0 ? N1+1 : N1; // (ceil) horizon of large blocks
	int T1; // horizon of current block
	int N_tmp; // temporary sum of horizons

	// data matrices
	int d_size = 0;

	int Gamma_size;
	int pA_size;
	int buffer_size;
	int pBAbtL_size;
	int pM_size;
	int tmp_size;

	int nuM, nxM, nuxM;
	int pnzM, pnx1M, cnuxM, cnxM;

	int stage_size = 0;

	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;

		Gamma_size = 0;
		pA_size = 0;
		buffer_size = 0;
		nu_tmp = 0;
		for(jj=0; jj<T1; jj++)
			{
			// hpGamma
			Gamma_size += ((nx[N_tmp+0]+nu_tmp+nu[N_tmp+jj]+1+bs-1)/bs*bs) * cnx[N_tmp+jj+1];
			// pA
			tmp_size = pnx[N_tmp+jj] * cnx[N_tmp+jj+1];
			pA_size = tmp_size > pA_size ? tmp_size : pA_size;
			// buffer
			tmp_size += ((nx[N_tmp+0]+nu_tmp+1+bs-1)/bs*bs) * cnx[N_tmp+jj+1];
			buffer_size = tmp_size > buffer_size ? tmp_size : buffer_size;
			//
			nu_tmp += nu[N_tmp+jj];
			}

		tmp_size = Gamma_size + pA_size + buffer_size;
		stage_size = tmp_size > stage_size ? tmp_size : stage_size;

		pBAbtL_size = 0;
		buffer_size = 0;
		pM_size = 0;
		nuM = 0;
		nxM = 0;
		nuxM = 0;
		nu_tmp = 0;
		for(jj=0; jj<T1; jj++)
			{
			// nu
			nuM = nu[N_tmp+jj] > nuM ? nu[N_tmp+jj] : nuM;
			nxM = nx[N_tmp+jj] > nxM ? nx[N_tmp+jj] : nxM;
			nuxM = nu[N_tmp+jj]+nx[N_tmp+jj] > nxM ? nu[N_tmp+jj]+nx[N_tmp+jj] : nxM;
			// pBAbtL
			tmp_size = pnz[N_tmp+jj]*cnx[N_tmp+jj+1];
			pBAbtL_size = tmp_size > pBAbtL_size ? tmp_size : pBAbtL_size;
			// buffer
			tmp_size = ((nu_tmp+nx[N_tmp+0]+1+bs-1)/bs*bs) * cnu[N_tmp+jj+1];
			buffer_size = tmp_size > buffer_size ? tmp_size : buffer_size;
			// pM
			tmp_size = pnu[N_tmp+jj]*cnx[N_tmp+jj];
			pM_size = tmp_size > pM_size ? tmp_size : pM_size;
			//
			nu_tmp += nu[N_tmp+jj];
			}

		pnzM = (nuM+nxM+1+bs-1)/bs*bs;
		pnx1M = (nxM+1+bs-1)/bs*bs;
		cnuxM = (nuM+nxM+ncl-1)/ncl*ncl;
		cnxM = (nxM+ncl-1)/ncl*ncl;
		tmp_size = Gamma_size + pBAbtL_size + buffer_size + pM_size + pnzM*cnuxM + pnx1M*cnxM + pnx1M;
		stage_size = tmp_size > stage_size ? tmp_size : stage_size;

		N_tmp += T1;
		}
	
	d_size += stage_size;

	int size = d_size*sizeof(double);

	size = (size + 63) / 64 * 64; // make work space multiple of (typical) cache line size

	return size;

	}





int d_part_cond_memory_space_size_bytes_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, int N2, int *nx2, int *nu2, int *nb2, int *ng2)
	{

	// early return
	if(N2==N)
		{
		return 0;
		}

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj, kk;

	// packing quantities
	int pnz2[N2+1];
	int pnux2[N2+1];
	int pnb2[N2+1];
	int png2[N2+1];
	int cnx2[N2+1];
	int cnux2[N2+1];
	int cng2[N2+1];
	for(ii=0; ii<=N2; ii++)
		{
		pnz2[ii] = (nu2[ii]+nx2[ii]+1+bs-1)/bs*bs;
		pnux2[ii] = (nu2[ii]+nx2[ii]+bs-1)/bs*bs;
		pnb2[ii] = (nb2[ii]+bs-1)/bs*bs;
		png2[ii] = (ng2[ii]+bs-1)/bs*bs;
		cnx2[ii] = (nx2[ii]+ncl-1)/ncl*ncl;
		cnux2[ii] = (nu2[ii]+nx2[ii]+ncl-1)/ncl*ncl;
		cng2[ii] = (ng2[ii]+ncl-1)/ncl*ncl;
		}

	// data matrices
	int d_size = 0;
	for(ii=0; ii<N2; ii++)
		{
		// hpBAbt2
		d_size += pnz2[ii]*cnx2[ii+1];
		// hpRSQrq2
		d_size += pnz2[ii]*cnux2[ii];
		// hDCt2
		d_size += pnux2[ii]*cng2[ii];
		// hd2
		d_size += 2*pnb2[ii]+2*png2[ii];
		}
	// no last stage !!!!!
	int i_size = 0;
	for(ii=0; ii<N2; ii++)
		{
		// hidxb2
		i_size += nb2[ii];
		}

	int size = d_size*sizeof(double) + i_size*sizeof(int);

	size = (size + 63) / 64 * 64; // make memory space multiple of (typical) cache line size

	return size;

	}





void d_part_cond_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, double **hpBAbt, double **hpRSQrq, double **hpDCt, double **hd, int N2, int *nx2, int *nu2, int *nb2, int **hidxb2, int *ng2, double **hpBAbt2, double **hpRSQrq2, double **hpDCt2, double **hd2, void *memory, void *work)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj, kk;
	int nu_tmp;

	// early return
	if(N2==N)
		{
		for(ii=0; ii<N; ii++)
			{
			nx2[ii] = nx[ii];
			nu2[ii] = nu[ii];
			nb2[ii] = nb[ii];
			hidxb2[ii] = hidxb[ii];
			ng2[ii] = ng[ii];
			hpBAbt2[ii] = hpBAbt[ii];
			hpRSQrq2[ii] = hpRSQrq[ii];
			hpDCt2[ii] = hpDCt[ii];
			hd2[ii] = hd[ii];
			}
		ii = N;
		nx2[ii] = nx[ii];
		nu2[ii] = nu[ii];
		nb2[ii] = nb[ii];
		hidxb2[ii] = hidxb[ii];
		ng2[ii] = ng[ii];
		hpRSQrq2[ii] = hpRSQrq[ii];
		hpDCt2[ii] = hpDCt[ii];
		hd2[ii] = hd[ii];
		return;
		}
	
	// sequential update not implemented
	if(N2>N)
		{
		printf("\nError: it must be N2<=N, sequential update not implemented\n\n");
		exit(1);
		}
	
	// general constraints not implemented (can be only ng[N]>0)
	for(ii=0; ii<N; ii++)
		if(ng[ii]>0)
			{
			printf("\nError: it must be ng>0, general constraints case not implemented\n\n");
			exit(1);
			}

	// packing quantities - 1
	int cnx[N];
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		}

	int N1 = N/N2; // (floor) horizon of small blocks
	int R1 = N - N2*N1; // the first R1 blocks have horizon N1+1
	int M1 = R1>0 ? N1+1 : N1; // (ceil) horizon of large blocks
	int T1; // horizon of current block
	int N_tmp = 0; // temporary sum of horizons

	// packing quantities - 2
	int pnz2[N2+1];
	int pnux2[N2+1];
	int pnb2[N2+1];
	int png2[N2+1];
	int cnx2[N2+1];
	int cnux2[N2+1];
	int cng2[N2+1];
	for(ii=0; ii<=N2; ii++)
		{
		pnz2[ii] = (nu2[ii]+nx2[ii]+1+bs-1)/bs*bs;
		pnux2[ii] = (nu2[ii]+nx2[ii]+bs-1)/bs*bs;
		pnb2[ii] = (nb2[ii]+bs-1)/bs*bs;
		png2[ii] = (ng2[ii]+bs-1)/bs*bs;
		cnx2[ii] = (nx2[ii]+ncl-1)/ncl*ncl;
		cnux2[ii] = (nu2[ii]+nx2[ii]+ncl-1)/ncl*ncl;
		cng2[ii] = (ng2[ii]+ncl-1)/ncl*ncl;
		}

	// data matrices (memory space) no last stage !!!!!
	double *ptr = memory;
	for(ii=0; ii<N2; ii++)
		{
		hpBAbt2[ii] = ptr;
		ptr += pnz2[ii]*cnx2[ii+1];
		}
	for(ii=0; ii<N2; ii++)
		{
		hpRSQrq2[ii] = ptr;
		ptr += pnz2[ii]*cnux2[ii];
		}
	for(ii=0; ii<N2; ii++)
		{
		hpDCt2[ii] = ptr;
		ptr += pnux2[ii]*cng2[ii];
		}
	for(ii=0; ii<N2; ii++)
		{
		hd2[ii] = ptr;
		ptr += 2*pnb2[ii]+2*png2[ii];
		}
	int *i_ptr = (int *) ptr;
	for(ii=0; ii<N2; ii++)
		{
		hidxb2[ii] = i_ptr;
		i_ptr += nb2[ii];
		}

	// work space
	double *hpGamma[M1];

	// other stages
	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;
		ptr = work;
		nu_tmp = nu[N_tmp+0];
		for(jj=0; jj<T1; jj++)
			{
			hpGamma[jj] = ptr;
			ptr += ((nx[N_tmp+0]+nu_tmp+1+bs-1)/bs*bs) * cnx[N_tmp+jj+1];
			nu_tmp += nu[N_tmp+jj+1];
			}
		d_cond_BAbt(T1, &nx[N_tmp], &nu[N_tmp], &hpBAbt[N_tmp], ptr, hpGamma, hpBAbt2[ii]);
		d_cond_RSQrq(T1, &nx[N_tmp], &nu[N_tmp], &hpBAbt[N_tmp], &hpRSQrq[N_tmp], hpGamma, ptr, hpRSQrq2[ii]);
		d_cond_DCtd(T1, &nx[N_tmp], &nu[N_tmp], &nb[N_tmp], &hidxb[N_tmp], &hd[N_tmp], hpGamma, hpDCt2[ii], hd2[ii], hidxb2[ii]);
		N_tmp += T1;
		}

	// last stage
	hpRSQrq2[N2] = hpRSQrq[N];
	hpDCt2[N2] = hpDCt[N];
	hd2[N2] = hd[N];
	hidxb2[N2] = hidxb[N];

	return;

	}



int d_part_expand_work_space_size_bytes_libstr(int N, int *nx, int *nu, int *nb, int *ng)
	{

	const int bs = D_MR;

	int ii;

	int nzM = nu[0]+nx[0]+1;
	int ngM = ng[0];

	for(ii=1; ii<=N; ii++)
		{
		nzM = nu[ii]+nx[ii]+1>nzM ? nu[ii]+nx[ii]+1 : nzM;
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}
	
	int pnzM = (nzM+bs-1)/bs*bs;
	int pngM = (ngM+bs-1)/bs*bs;

	int d_size = pnzM + pngM;

	int size = d_size*sizeof(double);

	size = (size + 63) / 64 * 64; // make multiple of (typical) cache line size

	return size;

	}



void d_part_expand_solution_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, double **hpBAbt, double **hb, double **hpRSQrq, double **hrq, double **hpDCt, double **hux, double **hpi, double **hlam, double **ht, int N2, int *nx2, int *nu2, int *nb2, int **hidxb2, int *ng2, double **hux2, double **hpi2, double **hlam2, double **ht2, void *work)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj, ll;

	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cng[N+1];
	int nzM = 0;
	int ngM = 0;

	for(ii=0; ii<=N; ii++)
		{
		pnb[ii] = (nb[ii]+bs-1)/bs*bs;
		png[ii] = (ng[ii]+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;
		nzM = nu[ii]+nx[ii]+1>nzM ? nu[ii]+nx[ii]+1 : nzM;
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}
	
	int pnzM = (nzM+bs-1)/bs*bs;
	int pngM = (ngM+bs-1)/bs*bs;

	int pnb2[N2+1];
	int png2[N2+1];

	for(ii=0; ii<=N2; ii++)
		{
		pnb2[ii] = (nb2[ii]+bs-1)/bs*bs;
		png2[ii] = (ng2[ii]+bs-1)/bs*bs;
		}

	double *pi_work0;
	double *pi_work1;

	double *ptr = (double *) work;

	pi_work0 = ptr;
	ptr += pnzM;

	pi_work1 = ptr;
	ptr += pngM;


	int N1 = N/N2; // (floor) horizon of small blocks
	int R1 = N - N2*N1; // the first R1 blocks have horizion N1+1
	int M1 = R1>0 ? N1+1 : N1; // (ceil) horizon of large blocks
	int T1; // horizon of current block
	int N_tmp, nu_tmp;
	int nbb2, nbg2, nbb2_tmp, nbg2_tmp;
	int stg;

	// inputs & initial states
	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;
		nu_tmp = 0;
		// final stages: copy only input
		for(jj=0; jj<T1-1; jj++)
			{
			for(ll=0; ll<nu[N_tmp+T1-1-jj]; ll++)
				hux[N_tmp+T1-1-jj][ll] = hux2[ii][nu_tmp+ll];
			nu_tmp += nu[N_tmp+T1-1-jj];
			}
		// first stage: copy input and state
		for(ll=0; ll<nu[N_tmp+0]+nx[N_tmp+0]; ll++)
			hux[N_tmp+0][ll] = hux2[ii][nu_tmp+ll];
		//
		N_tmp += T1;
		}

	// copy final state
	for(ll=0; ll<nx[N]; ll++)
		hux[N][ll] = hux2[N2][ll];

	// compute missing states by simulation within each block
	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;
		for(jj=0; jj<T1-1; jj++) // last stage is already there !!!
			{
			for(ll=0; ll<nx[N_tmp+jj+1]; ll++)
				hux[N_tmp+jj+1][nu[N_tmp+jj+1]+ll] = hb[N_tmp+jj][ll];
#if defined(BLASFEO)
			dgemv_t_lib(nu[N_tmp+jj]+nx[N_tmp+jj], nx[N_tmp+jj+1], 1.0, hpBAbt[N_tmp+jj], cnx[N_tmp+jj+1], hux[N_tmp+jj], 1.0, hux[N_tmp+jj+1]+nu[N_tmp+jj+1], hux[N_tmp+jj+1]+nu[N_tmp+jj+1]);
#else
			dgemv_t_lib(nu[N_tmp+jj]+nx[N_tmp+jj], nx[N_tmp+jj+1], hpBAbt[N_tmp+jj], cnx[N_tmp+jj+1], hux[N_tmp+jj], 1, hux[N_tmp+jj+1]+nu[N_tmp+jj+1], hux[N_tmp+jj+1]+nu[N_tmp+jj+1]);
#endif
			}
		//
		N_tmp += T1;
		}

	// slack variables and ineq lagrange multipliers
	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		nbb2_tmp = 0;
		nbg2_tmp = 0;
		T1 = ii<R1 ? M1 : N1;
		// final stages
		for(jj=0; jj<T1-1; jj++)
			{
			nbb2 = 0;
			nbg2 = 0;
			for(ll=0; ll<nb[N_tmp+T1-1-jj]; ll++)
				if(hidxb[N_tmp+T1-1-jj][ll]<nu[N_tmp+T1-1-jj])
					nbb2++;
				else
					nbg2++;
			for(ll=0; ll<nbb2; ll++) // box as box
				{
				hlam[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+ll] = hlam2[ii][0*pnb2[ii]+nbb2_tmp+ll];
				hlam[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+ll] = hlam2[ii][1*pnb2[ii]+nbb2_tmp+ll];
				ht[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+ll] = ht2[ii][0*pnb2[ii]+nbb2_tmp+ll];
				ht[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+ll] = ht2[ii][1*pnb2[ii]+nbb2_tmp+ll];
				}
			for(ll=0; ll<nbg2; ll++) // box as general XXX change when decide where nbg are placed wrt ng
				{
				hlam[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+nbb2+ll] = hlam2[ii][2*pnb2[ii]+0*png2[ii]+nbg2_tmp+ll];
				hlam[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+nbb2+ll] = hlam2[ii][2*pnb2[ii]+1*png2[ii]+nbg2_tmp+ll];
				ht[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+nbb2+ll] = ht2[ii][2*pnb2[ii]+0*png2[ii]+nbg2_tmp+ll];
				ht[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+nbb2+ll] = ht2[ii][2*pnb2[ii]+1*png2[ii]+nbg2_tmp+ll];
				}
			nbb2_tmp += nbb2;
			nbg2_tmp += nbg2;
			}
		// first stage
		for(ll=0; ll<nb[N_tmp+0]; ll++) // all remain box
			{
			hlam[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+ll] = hlam2[ii][0*pnb2[ii]+nbb2_tmp+ll];
			hlam[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+ll] = hlam2[ii][1*pnb2[ii]+nbb2_tmp+ll];
			ht[N_tmp+T1-1-jj][0*pnb[N_tmp+T1-1-jj]+ll] = ht2[ii][0*pnb2[ii]+nbb2_tmp+ll];
			ht[N_tmp+T1-1-jj][1*pnb[N_tmp+T1-1-jj]+ll] = ht2[ii][1*pnb2[ii]+nbb2_tmp+ll];
			}
//		nbb2_tmp += nbb2;
//		nbg2_tmp += nbg2;
		//
		N_tmp += T1;
		}
	// last stage: just copy
	for(jj=0; jj<nb[N]; jj++)
		{
		hlam[N][0*pnb[N]+jj] = hlam2[N2][0*pnb2[N2]+jj];
		hlam[N][1*pnb[N]+jj] = hlam2[N2][1*pnb2[N2]+jj];
		ht[N][0*pnb[N]+jj] = ht2[N2][0*pnb2[N2]+jj];
		ht[N][1*pnb[N]+jj] = ht2[N2][1*pnb2[N2]+jj];
		}
	for(jj=0; jj<ng[N]; jj++)
		{
		hlam[N][2*pnb[N]+0*png[N]+jj] = hlam2[N2][2*pnb2[N2]+0*png2[N2]+jj];
		hlam[N][2*pnb[N]+1*png[N]+jj] = hlam2[N2][2*pnb2[N2]+1*png2[N2]+jj];
		ht[N][2*pnb[N]+0*png[N]+jj] = ht2[N2][2*pnb2[N2]+0*png2[N2]+jj];
		ht[N][2*pnb[N]+1*png[N]+jj] = ht2[N2][2*pnb2[N2]+1*png2[N2]+jj];
		}

	// lagrange multipliers of equality constraints
	N_tmp = 0;
	for(ii=0; ii<N2; ii++)
		{
		T1 = ii<R1 ? M1 : N1;
		// last stage: just copy
		for(ll=0; ll<nx[N_tmp+T1]; ll++)
			hpi[N_tmp+T1-1][ll] = hpi2[ii][ll];
		// middle stages: backward simulation
		for(jj=0; jj<T1-1; jj++)
			{
			stg = N_tmp+T1-1-jj;
			for(ll=0; ll<nu[stg]+nx[stg]; ll++)
				pi_work0[ll] = hrq[stg][ll];
			for(ll=0; ll<nb[stg]; ll++)
				pi_work0[hidxb[stg][ll]] += - hlam[stg][0*pnb[stg]+ll] + hlam[stg][1*pnb[stg]+ll];
#if defined(BLASFEO)
			dsymv_l_lib(nu[stg]+nx[stg], nu[stg]+nx[stg], 1.0, hpRSQrq[stg], cnux[stg], hux[stg], 1.0, pi_work0, pi_work0);
			dgemv_n_lib(nu[stg]+nx[stg], nx[stg+1], 1.0, hpBAbt[stg], cnx[stg+1], hpi[stg], 1.0, pi_work0, pi_work0);
#else
			dsymv_lib(nu[stg]+nx[stg], nu[stg]+nx[stg], hpRSQrq[stg], cnux[stg], hux[stg], 1, pi_work0, pi_work0);
			dgemv_n_lib(nu[stg]+nx[stg], nx[stg+1], hpBAbt[stg], cnx[stg+1], hpi[stg], 1, pi_work0, pi_work0);
#endif
			for(ll=0; ll<ng[stg]; ll++)
				pi_work1[ll] = hlam[stg][2*pnb[stg]+1*png[stg]+ll] - hlam[stg][2*pnb[stg]+0*png[stg]+ll];
#if defined(BLASFEO)
			dgemv_n_lib(nu[stg]+nx[stg], ng[stg], 1.0, hpDCt[stg], cng[stg], pi_work1, 1.0, pi_work0, pi_work0);
#else
			dgemv_n_lib(nu[stg]+nx[stg], ng[stg], hpDCt[stg], cng[stg], pi_work1, 1, pi_work0, pi_work0);
#endif
			//
			for(ll=0; ll<nx[stg]; ll++)
				hpi[stg-1][ll] = + pi_work0[nu[stg]+ll];
			}
		//
		N_tmp += T1;
		}

	return;

	}
