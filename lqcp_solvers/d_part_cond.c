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
#include "../include/lqcp_aux.h"
#include "../include/block_size.h"



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
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
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
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
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
	dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	for(ii=1; ii<N; ii++)
		{
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		}
	
	dgetr_lib(nx[N], 1, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);
	
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
	dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	//
	nu_tmp += nu[0];

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
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
		dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		//
		nu_tmp += nu[ii];
		}
	
	// B
	dgecp_lib(nu_tmp, nx[N], 0, pGamma_u[N-1], cnx[N], 0, pBAbt2, cnx[N]);
	// A
	dgecp_lib(nx[0], nx[N], 0, pGamma_x0[N-1], cnx[N], nu2, pBAbt2+nu2/bs*bs*cnx[N]+nu2%bs, cnx[N]);
	// b
	dgetr_lib(nx[N], 1, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);

	}
#endif



void d_cond_BAb(int N, int *nx, int *nu, double **hpBAbt, double *work, double **hpGamma, double *pBAbt2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nxM = 0;
	int nu2 = 0;
	for(ii=0; ii<N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nxM = nx[ii]>nxM ? nx[ii] : nxM;
		nu2 += nu[ii];
		}
	ii = N;
	cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
	nxM = nx[ii]>nxM ? nx[ii] : nxM;
	
	int pnz2 = (nu2+nx[0]+1+bs-1)/bs*bs;
	int pnxM = (nxM+bs-1)/bs*bs;
	int cnxM = (nxM+ncl-1)/ncl*ncl;
	
	double *pA = work;
	work += pnxM*cnxM;

	double *buffer = work;
	work += pnz2*cnxM;

	int nu_tmp = 0;

	ii = 0;
	// B & A & b
	dgecp_lib(nu[0]+nx[0]+1, nx[1], 0, hpBAbt[0], cnx[1], 0, hpGamma[0], cnx[1]);
	//
	nu_tmp += nu[0];
	ii++;

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		
		// pA in work space
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], hpBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, pA, cnx[ii]); // pA in work

		// Gamma * A^T
		dgemm_nt_lib(nu_tmp+nx[0]+1, nx[ii+1], nx[ii], hpGamma[ii-1], cnx[ii], pA, cnx[ii], 0, buffer, cnx[ii+1], buffer, cnx[ii+1], 0, 0); // Gamma * A^T // TODO in BLASFEO, store to unaligned !!!!!

		dgecp_lib(nu[ii], nx[ii+1], 0, hpBAbt[ii], cnx[ii+1], 0, hpGamma[ii], cnx[ii+1]);

		dgecp_lib(nu_tmp+nx[0]+1, nx[ii+1], 0, buffer, cnx[ii+1], nu[ii], hpGamma[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1]);

		nu_tmp += nu[ii];

		for(jj=0; jj<nx[ii+1]; jj++) hpGamma[ii][(nu_tmp+nx[0])/bs*bs*cnx[ii+1]+(nu_tmp+nx[0])%bs+jj*bs] += hpBAbt[ii][(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs+jj*bs];
		}
	
	// B & A & b
	dgecp_lib(nu_tmp+nx[0]+1, nx[N], 0, hpGamma[N-1], cnx[N], 0, pBAbt2, cnx[N]);

	return;

	}



void d_cond_RSQrq(int N, int *nx, int *nu, double **hpBAbt, double **hpRSQrq, double **hpGamma, double *work, double *pRSQrq2)
	{

	// early return
	if(N<1)
		return;

	const int bs = D_MR;
	const int ncl = D_NCL;
	
	int nn;

	// compute sizes of matrices TODO pass them instead of compute them ???
	// TODO check if all are needed
	int nux[N+1];
	int nz[N+1];
	int cnu[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int pnx[N+1];
	int pnz[N+1];
	int nuM = nu[0];
	int nxM = nx[0];
	int nuxM = nu[0]+nx[0];

	int nu2[N+1];
	int nu3[N+1];
	nu2[0]= 0; // sum
	nu3[0]= 0; // reverse sum

	for(nn=0; nn<N; nn++)
		{
		nux[nn] = nu[nn]+nx[nn];
		nz[nn] = nux[nn]+1;
		cnu[nn] = (nu[nn]+ncl-1)/ncl*ncl;
		cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
		cnux[nn] = (nu[nn]+nx[nn]+ncl-1)/ncl*ncl;
		pnx[nn] = (nx[nn]+bs-1)/bs*bs;
		pnz[nn] = (nu[nn]+nx[nn]+1+bs-1)/bs*bs;
		nuM = nu[nn]>nuM ? nu[nn] : nuM;
		nxM = nx[nn]>nxM ? nx[nn] : nxM;
		nuxM = nu[nn]+nx[nn]>nuxM ? nu[nn]+nx[nn] : nuxM;
		nu2[nn+1] = nu2[nn] + nu[nn];
		nu3[nn+1] = nu3[nn] + nu[N-nn-1];
		}
	nn = N;
	nux[nn] = nx[nn]; //
	nz[nn] = nux[nn]+1;
	cnu[nn] = 0;
	cnx[nn] = (nx[nn]+ncl-1)/ncl*ncl;
	cnux[nn] = (nx[nn]+ncl-1)/ncl*ncl; //
	pnx[nn] = (nx[nn]+bs-1)/bs*bs;
	pnz[nn] = (nu[nn]+nx[nn]+1+bs-1)/bs*bs;
	nxM = nx[nn]>nxM ? nx[nn] : nxM;
	nuxM = nx[nn]>nuxM ? nx[nn] : nuxM; //

	int pnuM = (nuM+bs-1)/bs*bs;
	int pnzM = (nuxM+1+bs-1)/bs*bs;
	int pnx1M = (nxM+1+bs-1)/bs*bs;
	int cnuM = (nuM+ncl-1)/ncl*ncl;
	int cnxM = (nxM+ncl-1)/ncl*ncl;
	int cnuxM = (nuxM+ncl-1)/ncl*ncl;

	int pnz2 = (nu2[N]+nx[0]+1+bs-1)/bs*bs;
	int cnux2 = (nu2[N]+nx[0]+ncl-1)/ncl*ncl;

	double *pL = work;
	work += pnzM*cnuxM;

	double *pLx = work;
	work += pnx1M*cnxM;

	double *pBAbtL = work;
	work += pnzM*cnxM;

	double *pM = work;
	work += pnuM*cnxM;

	double *buffer = work;
	work += pnz2*cnuM;

	double *dLx = work;
	work += pnzM;



	// early return
	if(N==1)
		{
		dgecp_lib(nu[N-1]+nx[N-1]+1, nu[N-1]+nx[N-1], 0, hpRSQrq[N-1], cnux[N-1], 0, pRSQrq2, cnux[N-1]);
		return;
		}



	// final stage 

	dgecp_lib(nu[N-1]+nx[N-1]+1, nu[N-1]+nx[N-1], 0, hpRSQrq[N-1], cnux[N-1], 0, pL, cnux[N-1]);

	// D
	dgecp_lib(nu[N-1], nu[N-1], 0, pL, cnux[N-1], nu3[0], pRSQrq2+nu3[0]/bs*bs*cnux2+nu3[0]%bs+nu3[0]*bs, cnux2);

	// M
	dgetr_lib(nx[N-1], nu[N-1], nu[N-1], pL+nu[N-1]/bs*bs*cnux[N-1]+nu[N-1]%bs, cnux[N-1], 0, pM, cnu[N-1]);

	dgemm_nt_lib(nu2[N-1]+nx[0]+1, nu[N-1], nx[N-1], hpGamma[N-2], cnx[N-1], pM, cnu[N-1], 0, buffer, cnu[N-1], buffer, cnu[N-1], 0, 0);

	dgecp_lib(nu2[N-1]+nx[0]+1, nu[N-1], 0, buffer, cnu[N-1], nu3[1], pRSQrq2+nu3[1]/bs*bs*cnux2+nu3[1]%bs+nu3[0]*bs, cnux2);

	// m
	dgead_lib(1, nu[N-1], 1.0, nu[N-1]+nx[N-1], pL+(nu[N-1]+nx[N-1])/bs*bs*cnux[N-1]+(nu[N-1]+nx[N-1])%bs, cnux[N-1], nu2[N]+nx[0], pRSQrq2+(nu2[N]+nx[0])/bs*bs*cnux2+(nu2[N]+nx[0])%bs+nu3[0]*bs, cnux2);



	// middle stages 
	for(nn=1; nn<N-1; nn++)
		{	

		dgecp_lib(nx[N-nn]+1, nx[N-nn], nu[N-nn], pL+nu[N-nn]/bs*bs*cnux[N-nn]+nu[N-nn]%bs+nu[N-nn]*bs, cnux[N-nn], 0, pLx, cnx[N-nn]);
//		d_print_pmat(nx[N-nn]+1, nx[N-nn], bs, pLx, cnx[N-nn]);

#ifdef BLASFEO
		dpotrf_ntnn_l_lib(nx[N-nn]+1, nx[N-nn], pLx, cnx[N-nn], pLx, cnx[N-nn], dLx);
#else
		dpotrf_lib(nx[N-nn]+1, nx[N-nn], pLx, cnx[N-nn], pLx, cnx[N-nn], dLx);
#endif

		dtrtr_l_lib(nx[N-nn], 0, pLx, cnx[N-nn], 0, pLx, cnx[N-nn]);	
//		d_print_pmat(nx[N]+1, nx[N], bs, pLx, cnx[N-nn]);

#ifdef BLASFEO
		dtrmm_ntnn_ru_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], pLx, cnx[N-nn], 0, pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn]);
#else
		dtrmm_nt_u_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], pLx, cnx[N-nn], pBAbtL, cnx[N-nn]);
#endif

		dgead_lib(1, nx[N-nn], 1.0, nx[N-nn], pLx+nx[N-nn]/bs*bs*cnx[N-nn]+nx[N-nn]%bs, cnx[N-nn], nux[N-nn-1], pBAbtL+nux[N-nn-1]/bs*bs*cnx[N-nn]+nux[N-nn-1]%bs, cnx[N-nn]);

#ifdef BLASFEO
		dsyrk_ntnn_l_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn], pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn], 1, hpRSQrq[N-nn-1], cnux[N-nn-1], pL, cnux[N-nn-1]);
#else
		dsyrk_nt_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn], pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn], 1, hpRSQrq[N-nn-1], cnux[N-nn-1], pL, cnux[N-nn-1]);
#endif
//		d_print_pmat(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], bs, pL, cnux[N-nn-1]);

		// D
		dgecp_lib(nu[N-nn-1], nu[N-nn-1], 0, pL, cnux[N-nn-1], nu3[nn], pRSQrq2+nu3[nn]/bs*bs*cnux2+nu3[nn]%bs+nu3[nn]*bs, cnux2);

		// M
		dgetr_lib(nx[N-nn-1], nu[N-nn-1], nu[N-nn-1], pL+nu[N-nn-1]/bs*bs*cnux[N-nn-1]+nu[N-nn-1]%bs, cnux[N-nn-1], 0, pM, cnu[N-nn-1]);

		dgemm_nt_lib(nu2[N-nn-1]+nx[0]+1, nu[N-nn-1], nx[N-nn-1], hpGamma[N-nn-2], cnx[N-nn-1], pM, cnu[N-nn-1], 0, buffer, cnu[N-nn-1], buffer, cnu[N-nn-1], 0, 0); // add unaligned stores in BLASFEO !!!!!!

		dgecp_lib(nu2[N-nn-1]+nx[0]+1, nu[N-nn-1], 0, buffer, cnu[N-nn-1], nu3[nn+1], pRSQrq2+nu3[nn+1]/bs*bs*cnux2+nu3[nn+1]%bs+nu3[nn]*bs, cnux2);

		// m
		dgead_lib(1, nu[N-nn-1], 1.0, nu[N-nn-1]+nx[N-nn-1], pL+(nu[N-nn-1]+nx[N-nn-1])/bs*bs*cnux[N-nn-1]+(nu[N-nn-1]+nx[N-nn-1])%bs, cnux[N-nn-1], nu2[N]+nx[0], pRSQrq2+(nu2[N]+nx[0])/bs*bs*cnux2+(nu2[N]+nx[0])%bs+nu3[nn]*bs, cnux2);

//		d_print_pmat(nu2[N-nn-1]+nx[0]+1, nu[N-nn-1], bs, buffer, cnu[N-nn-1]);
//		exit(2);
//		return;
		}

	// first stage
	nn = N-1;
	
	dgecp_lib(nx[N-nn]+1, nx[N-nn], nu[N-nn], pL+nu[N-nn]/bs*bs*cnux[N-nn]+nu[N-nn]%bs+nu[N-nn]*bs, cnux[N-nn], 0, pLx, cnx[N-nn]);
//	d_print_pmat(nx[N-nn]+1, nx[N-nn], bs, pLx, cnx[N-nn]);

#ifdef BLASFEO
	dpotrf_ntnn_l_lib(nx[N-nn]+1, nx[N-nn], pLx, cnx[N-nn], pLx, cnx[N-nn], dLx);
#else
	dpotrf_lib(nx[N-nn]+1, nx[N-nn], pLx, cnx[N-nn], pLx, cnx[N-nn], dLx);
#endif

	dtrtr_l_lib(nx[N-nn], 0, pLx, cnx[N-nn], 0, pLx, cnx[N-nn]);	
//	d_print_pmat(nx[N]+1, nx[N], bs, pLx, cnx[N-nn]);

#ifdef BLASFEO
	dtrmm_ntnn_ru_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], pLx, cnx[N-nn], 0, pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn]);
#else
	dtrmm_nt_u_lib(nz[N-nn-1], nx[N-nn], hpBAbt[N-nn-1], cnx[N-nn], pLx, cnx[N-nn], pBAbtL, cnx[N-nn]);
#endif

	dgead_lib(1, nx[N-nn], 1.0, nx[N-nn], pLx+nx[N-nn]/bs*bs*cnx[N-nn]+nx[N-nn]%bs, cnx[N-nn], nux[N-nn-1], pBAbtL+nux[N-nn-1]/bs*bs*cnx[N-nn]+nux[N-nn-1]%bs, cnx[N-nn]);

#ifdef BLASFEO
	dsyrk_ntnn_l_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn], pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn], 1, hpRSQrq[N-nn-1], cnux[N-nn-1], pL, cnux[N-nn-1]);
#else
	dsyrk_nt_lib(nz[N-nn-1], nux[N-nn-1], nx[N-nn], pBAbtL, cnx[N-nn], pBAbtL, cnx[N-nn], 1, hpRSQrq[N-nn-1], cnux[N-nn-1], pL, cnux[N-nn-1]);
#endif
//	d_print_pmat(nu[N-nn-1]+nx[N-nn-1]+1, nu[N-nn-1]+nx[N-nn-1], bs, pL, cnux[N-nn-1]);

	// D, M, m, P, p
	dgecp_lib(nu[0]+nx[0]+1, nu[0]+nx[0], 0, pL, cnux[0], nu3[N-1], pRSQrq2+nu3[N-1]/bs*bs*cnux2+nu3[N-1]%bs+nu3[N-1]*bs, cnux2); // TODO dtrcp for 'rectangular' matrices

	return;

	}
