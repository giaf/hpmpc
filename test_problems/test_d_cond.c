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
#include <math.h>
#include <sys/time.h>
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
#endif

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/blas_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_solvers.h"
#include "../problem_size.h"
#include "../include/block_size.h"
#include "tools.h"
#include "test_param.h"
#include "../include/c_interface.h"



// XXX
#include "../lqcp_solvers/d_part_cond.c"



/************************************************ 
Mass-spring system: nx/2 masses connected each other with springs (in a row), and the first and the last one to walls. nu (<=nx) controls act on the first nu masses. The system is sampled with sampling time Ts. 
************************************************/
void mass_spring_system(double Ts, int nx, int nu, int N, double *A, double *B, double *b, double *x0)
	{

	int nx2 = nx*nx;

	int info = 0;

	int pp = nx/2; // number of masses
	
/************************************************
* build the continuous time system 
************************************************/
	
	double *T; d_zeros(&T, pp, pp);
	int ii;
	for(ii=0; ii<pp; ii++) T[ii*(pp+1)] = -2;
	for(ii=0; ii<pp-1; ii++) T[ii*(pp+1)+1] = 1;
	for(ii=1; ii<pp; ii++) T[ii*(pp+1)-1] = 1;

	double *Z; d_zeros(&Z, pp, pp);
	double *I; d_zeros(&I, pp, pp); for(ii=0; ii<pp; ii++) I[ii*(pp+1)]=1.0; // = eye(pp);
	double *Ac; d_zeros(&Ac, nx, nx);
	dmcopy(pp, pp, Z, pp, Ac, nx);
	dmcopy(pp, pp, T, pp, Ac+pp, nx);
	dmcopy(pp, pp, I, pp, Ac+pp*nx, nx);
	dmcopy(pp, pp, Z, pp, Ac+pp*(nx+1), nx); 
	free(T);
	free(Z);
	free(I);
	
	d_zeros(&I, nu, nu); for(ii=0; ii<nu; ii++) I[ii*(nu+1)]=1.0; //I = eye(nu);
	double *Bc; d_zeros(&Bc, nx, nu);
	dmcopy(nu, nu, I, nu, Bc+pp, nx);
	free(I);
	
/************************************************
* compute the discrete time system 
************************************************/

	double *bb; d_zeros(&bb, nx, 1);
	dmcopy(nx, 1, bb, nx, b, nx);
		
	dmcopy(nx, nx, Ac, nx, A, nx);
	dscal_3l(nx2, Ts, A);
	expm(nx, A);
	
	d_zeros(&T, nx, nx);
	d_zeros(&I, nx, nx); for(ii=0; ii<nx; ii++) I[ii*(nx+1)]=1.0; //I = eye(nx);
	dmcopy(nx, nx, A, nx, T, nx);
	daxpy_3l(nx2, -1.0, I, T);
	dgemm_nn_3l(nx, nu, nx, T, nx, Bc, nx, B, nx);
	free(T);
	free(I);
	
	int *ipiv = (int *) malloc(nx*sizeof(int));
	dgesv_3l(nx, nu, Ac, nx, ipiv, B, nx, &info);
	free(ipiv);

	free(Ac);
	free(Bc);
	free(bb);
	
			
/************************************************
* initial state 
************************************************/
	
	if(nx==4)
		{
		x0[0] = 5;
		x0[1] = 10;
		x0[2] = 15;
		x0[3] = 20;
		}
	else
		{
		int jj;
		for(jj=0; jj<nx; jj++)
			x0[jj] = 1;
		}

	}



int main()
	{
	
	printf("\n");
	printf("\n");
	printf("\n");
	printf(" HPMPC -- Library for High-Performance implementation of solvers for MPC.\n");
	printf(" Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.\n");
	printf("\n");
	printf(" HPMPC is distributed in the hope that it will be useful,\n");
	printf(" but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf(" MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
	printf(" See the GNU Lesser General Public License for more details.\n");
	printf("\n");
	printf("\n");
	printf("\n");
	
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
#endif

	int ii, jj;
	
	int rep, nrep=1000;//NREP;

	int nx = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght
//	int nb  = nu+nx; // number of box constrained inputs and states
//	int ng  = nx; //4;  // number of general constraints
//	int ngN = nx; // number of general constraints at the last stage
	printf("\nN = %d, nx = %d, nu = %d\n\n", N, nx, nu);

#define MHE 1


//	int nbu = nu<nb ? nu : nb ;
//	int nbx = nb-nu>0 ? nb-nu : 0;


	// stage-wise variant size
	int nx_v[N+1];
#if MHE==1
	nx_v[0] = nx;
#else
	nx_v[0] = 0;
#endif
	for(ii=1; ii<=N; ii++)
		nx_v[ii] = nx;

	int nu_v[N+1];
	for(ii=0; ii<N; ii++)
		nu_v[ii] = nu;
	nu_v[N] = 0;

//	int nb_v[N+1];
//	nb_v[0] = nbu;
//	for(ii=1; ii<N; ii++)
//		nb_v[ii] = nb;
//	nb_v[N] = nbx;

//	int ng_v[N+1];
//	for(ii=0; ii<N; ii++)
//		ng_v[ii] = ng;
//	ng_v[N] = ngN;
//	ng_v[M] = nx; // XXX
	


	int info = 0;
		
	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	int pnz = (nu+nx+1+bs-1)/bs*bs;
	int pnu = (nu+bs-1)/bs*bs;
	int pnu1 = (nu+1+bs-1)/bs*bs;
	int pnx = (nx+bs-1)/bs*bs;
	int pnx1 = (nx+1+bs-1)/bs*bs;
	int pnux = (nu+nx+bs-1)/bs*bs;
	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cnux = (nu+nx+ncl-1)/ncl*ncl;

//	int pnb_v[N+1]; 
//	int png_v[N+1]; 
	int pnx_v[N+1]; 
	int pnz_v[N+1]; 
	int pnux_v[N+1]; 
	int cnx_v[N+1]; 
	int cnux_v[N+1]; 
//	int cng_v[N+1]; 
	int nuM = 0;
	int nxM = 0;
	int nuxM = 0;
	int nu2 = 0;

	for(ii=0; ii<N; ii++) 
		{
//		pnb_v[ii] = (nb_v[ii]+bs-1)/bs*bs;
//		png_v[ii] = (ng_v[ii]+bs-1)/bs*bs;
		pnx_v[ii] = (nx_v[ii]+bs-1)/bs*bs;
		pnz_v[ii] = (nu_v[ii]+nx_v[ii]+1+bs-1)/bs*bs;
		pnux_v[ii] = (nu_v[ii]+nx_v[ii]+bs-1)/bs*bs;
		cnx_v[ii] = (nx_v[ii]+ncl-1)/ncl*ncl;
		cnux_v[ii] = (nu_v[ii]+nx_v[ii]+ncl-1)/ncl*ncl;
//		cng_v[ii] = (ng_v[ii]+ncl-1)/ncl*ncl;
		nuM = nu_v[ii]>nuM ? nu_v[ii] : nuM;
		nxM = nx_v[ii]>nxM ? nx_v[ii] : nxM;
		nuxM = nu_v[ii]+nx_v[ii]>nuxM ? nu_v[ii]+nx_v[ii] : nuxM;
		nu2 += nu_v[ii];
		}
	ii = N;
//	pnb_v[ii] = (nb_v[ii]+bs-1)/bs*bs;
//	png_v[ii] = (ng_v[ii]+bs-1)/bs*bs;
	pnx_v[ii] = (nx_v[ii]+bs-1)/bs*bs;
	pnz_v[ii] = (nx_v[ii]+1+bs-1)/bs*bs;
	pnux_v[ii] = (nx_v[ii]+bs-1)/bs*bs;
	cnx_v[ii] = (nx_v[ii]+ncl-1)/ncl*ncl;
	cnux_v[ii] = (nx_v[ii]+ncl-1)/ncl*ncl;
//	cng_v[ii] = (ng_v[ii]+ncl-1)/ncl*ncl;
	nxM = nx_v[ii]>nxM ? nx_v[ii] : nxM;
	nuxM = nx_v[ii]>nuxM ? nx_v[ii] : nuxM;

	int pnuM = (nuM+bs-1)/bs*bs;
	int pnxM = (nxM+bs-1)/bs*bs;
	int pnx1M = (nxM+1+bs-1)/bs*bs;
	int pnzM = (nuxM+1+bs-1)/bs*bs;
	int cnxM = (nxM+ncl-1)/ncl*ncl;
	int cnuM = (nuM+ncl-1)/ncl*ncl;
	int cnuxM = (nuxM+ncl-1)/ncl*ncl;
	int pnz2 = (nu2+nx_v[0]+1+bs-1)/bs*bs;
	int cnux2 = (nu2+nx_v[0]+ncl-1)/ncl*ncl;

/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx, nx); // states update matrix

	double *B; d_zeros(&B, nx, nu); // inputs matrix

	double *b; d_zeros_align(&b, nx, 1); // states offset
	double *x0; d_zeros_align(&x0, nx, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx, nu, N, A, B, b, x0);
	
	for(jj=0; jj<nx; jj++)
		b[jj] = 0.1;
	
	for(jj=0; jj<nx; jj++)
		x0[jj] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

#if MHE!=1
	double *pA; d_zeros_align(&pA, pnx, cnx);
	d_cvt_mat2pmat(nx, nx, A, nx, 0, pA, cnx);
	double *b0; d_zeros_align(&b0, pnx, 1);
	dgemv_n_lib(nx, nx, pA, cnx, x0, 1, b, b0);

	double *pBAbt0; 
	d_zeros_align(&pBAbt0, pnz_v[0], cnx_v[1]);
	d_cvt_tran_mat2pmat(nx_v[1], nu_v[0], B, nx_v[1], 0, pBAbt0, cnx_v[1]);
	d_cvt_tran_mat2pmat(nx_v[1], nx_v[0], A, nx_v[1], nu_v[0], pBAbt0+nu_v[0]/bs*bs*cnx_v[1]+nu_v[0]%bs, cnx_v[1]);
	d_cvt_tran_mat2pmat(nx_v[1], 1, b0, nx_v[1], nu_v[0]+nx_v[0], pBAbt0+(nu_v[0]+nx_v[0])/bs*bs*cnx_v[1]+(nu_v[0]+nx_v[0])%bs, cnx_v[1]);
	printf("\npBAbt0 = \n");
	d_print_pmat(nu_v[0]+nx_v[0]+1, nx_v[1], bs, pBAbt0, cnx_v[1]);
#endif

	double *pBAbt1; 
	if(N>1)
		{
		d_zeros_align(&pBAbt1, pnz_v[1], cnx_v[2]);
		d_cvt_tran_mat2pmat(nx_v[2], nu_v[1], B, nx_v[2], 0, pBAbt1, cnx_v[2]);
		d_cvt_tran_mat2pmat(nx_v[2], nx_v[1], A, nx_v[2], nu_v[1], pBAbt1+nu_v[1]/bs*bs*cnx_v[2]+nu_v[1]%bs, cnx_v[2]);
		d_cvt_tran_mat2pmat(nx_v[2], 1, b, nx_v[2], nu_v[1]+nx_v[1], pBAbt1+(nu_v[1]+nx_v[1])/bs*bs*cnx_v[2]+(nu_v[1]+nx_v[1])%bs, cnx_v[2]);
		printf("\npBAbt1 = \n");
		d_print_pmat(nu_v[1]+nx_v[1]+1, nx_v[2], bs, pBAbt1, cnx_v[2]);
		}
	
/************************************************
* cost function
************************************************/	

	double *R; d_zeros(&R, nu, nu);
	for(ii=0; ii<nu; ii++) R[ii*(nu+1)] = 2.0;

	double *S; d_zeros(&S, nu, nx);

	double *Q; d_zeros(&Q, nx, nx);
	for(ii=0; ii<nx; ii++) Q[ii*(nx+1)] = 1.0;

	double *r; d_zeros(&r, nu, 1);
	for(ii=0; ii<nu; ii++) r[ii] = 0.2;

	double *q; d_zeros(&q, nx, 1);
	for(ii=0; ii<nx; ii++) q[ii] = 0.1;

#if MHE!=1
	double *pS; d_zeros_align(&pS, pnu, cnx);
	d_cvt_mat2pmat(nu, nx, S, nu, 0, pS, cnx);
	double *r0; d_zeros_align(&r0, pnu, 1);
	dgemv_n_lib(nu, nx, pS, cnx, x0, 1, r, r0);

	double *pRSQrq0;
	d_zeros_align(&pRSQrq0, pnz_v[0], cnux_v[0]);
	d_cvt_mat2pmat(nu_v[0], nu_v[0], R, nu, 0, pRSQrq0, cnux_v[0]);
	d_cvt_tran_mat2pmat(nu_v[0], 1, r, nu, nu_v[0], pRSQrq0+nu_v[1]/bs*bs*cnux_v[1]+nu_v[1]%bs, cnux_v[0]);
	printf("\npRSQrq0 = \n");
	d_print_pmat(nu_v[0]+1, nu_v[0], bs, pRSQrq0, cnux_v[0]);
#endif

	double *pRSQrq1;
	if(N>1)
		{
		d_zeros_align(&pRSQrq1, pnz_v[1], cnux_v[1]);
		d_cvt_mat2pmat(nu_v[1], nu_v[1], R, nu, 0, pRSQrq1, cnux_v[1]);
		d_cvt_tran_mat2pmat(nu_v[1], nx_v[1], S, nu, nu_v[1], pRSQrq1+nu_v[1]/bs*bs*cnux_v[1]+nu_v[1]%bs, cnux_v[1]);
		d_cvt_mat2pmat(nx_v[1], nx_v[1], Q, nx, nu_v[1], pRSQrq1+nu_v[1]/bs*bs*cnux_v[1]+nu_v[1]%bs+nu_v[1]*bs, cnux_v[1]);
		d_cvt_tran_mat2pmat(nu_v[1], 1, r, nu, nu_v[1]+nx_v[1], pRSQrq1+(nu_v[1]+nx_v[1])/bs*bs*cnux_v[1]+(nu_v[1]+nx_v[1])%bs, cnux_v[1]);
		d_cvt_tran_mat2pmat(nx_v[1], 1, q, nx, nu_v[1]+nx_v[1], pRSQrq1+(nu_v[1]+nx_v[1])/bs*bs*cnux_v[1]+(nu_v[1]+nx_v[1])%bs+nu_v[1]*bs, cnux_v[1]);
		printf("\npRSQrq1 = \n");
		d_print_pmat(nu_v[1]+nx_v[1]+1, nu_v[1]+nx_v[1], bs, pRSQrq1, cnux_v[1]);
		}

	double *pRSQrqN;
	d_zeros_align(&pRSQrqN, pnz_v[N], cnux_v[N]);
	d_cvt_mat2pmat(nx_v[N], nx_v[N], Q, nx, 0, pRSQrqN, cnux_v[N]);
	d_cvt_tran_mat2pmat(nx_v[N], 1, q, nx, 0, pRSQrqN+nx_v[N]/bs*bs*cnux_v[N]+nx_v[N]%bs, cnux_v[N]);
	printf("\npRSQrqN = \n");
	d_print_pmat(nx_v[N]+1, nx_v[N], bs, pRSQrqN, cnux_v[N]);

/************************************************
* array of matrices & work space
************************************************/	

	double *hpBAbt[N];
	double *hpRSQrq[N+1];
//	double *hpGamma_x0[N];
//	double *hpGamma_u[N];
//	double *hpGamma_b[N];
	double *hpGamma[N];
	double *pBAbt2;
	double *pRSQrq2;
	double *work0;
	double *work1;

	int nu_tmp = 0;

	nu_tmp += nu_v[0];
#if MHE!=1
	hpBAbt[0] = pBAbt0;
	hpRSQrq[0] = pRSQrq0;
#else
	hpBAbt[0] = pBAbt1;
	hpRSQrq[0] = pRSQrq1;
#endif
//	d_zeros_align(&hpGamma_x0[0], pnx_v[0], cnx_v[1]);
//	d_zeros_align(&hpGamma_u[0], (nu_tmp+bs-1)/bs*bs, cnx_v[1]);
//	d_zeros_align(&hpGamma_b[0], pnx_v[1], 1);
	d_zeros_align(&hpGamma[0], (nx+1+nu_tmp+bs-1)/bs*bs, cnx_v[1]);
	for(ii=1; ii<N; ii++)
		{
		nu_tmp += nu_v[ii];
		hpBAbt[ii] = pBAbt1;
		hpRSQrq[ii] = pRSQrq1;
//		d_zeros_align(&hpGamma_x0[ii], pnx_v[0], cnx_v[ii+1]);
//		d_zeros_align(&hpGamma_u[ii], (nu_tmp+bs-1)/bs*bs, cnx_v[ii+1]);
//		d_zeros_align(&hpGamma_b[ii], pnx_v[ii+1], 1);
		d_zeros_align(&hpGamma[ii], (nx+1+nu_tmp+bs-1)/bs*bs, cnx_v[ii+1]);
		}
	hpRSQrq[N] = pRSQrqN;

	d_zeros_align(&pBAbt2, pnz2, cnx_v[N]);
	d_zeros_align(&pRSQrq2, pnz2, cnux2);

	d_zeros_align(&work0, pnxM*cnxM+pnz2*cnxM, 1);
	d_zeros_align(&work1, pnx1M*cnxM+pnzM*cnuxM+pnzM*cnxM+pnuM*cnxM+pnz2*cnuM+pnzM, 1);
	
//	for(ii=0; ii<N; ii++)
//		{
//		d_print_pmat(nu_v[ii]+nx_v[ii]+1, nx_v[ii+1], bs, hpBAbt[ii], cnx_v[ii]);
//		}
	
/************************************************
* solver call
************************************************/	

//	d_cond_B(N, nx_v, nu_v, hpBAbt, work, hpGamma_u, pBAbt2);
//	d_cond_A(N, nx_v, nu_v, hpBAbt, work, hpGamma_x0, pBAbt2);
//	d_cond_b(N, nx_v, nu_v, hpBAbt, work, hpGamma_b, pBAbt2);
//	d_cond_BAb(N, nx_v, nu_v, hpBAbt, work, hpGamma_u, hpGamma_x0, hpGamma_b, pBAbt2);
	
//	printf("\nGamma_x0\n\n");
//	for(ii=0; ii<N; ii++)
//		d_print_pmat(nx_v[0], nx_v[ii+1], bs, hpGamma_x0[ii], cnx_v[ii+1]);
	
//	printf("\nGamma_u\n\n");
//	nu_tmp = 0;
//	for(ii=0; ii<N; ii++)
//		{
//		nu_tmp += nu_v[ii];
//		d_print_pmat(nu_tmp, nx_v[ii+1], bs, hpGamma_u[ii], cnx_v[ii+1]);
//		}
	
//	printf("\nGamma_b\n\n");
//	for(ii=0; ii<N; ii++)
//		d_print_mat(1, nx_v[ii+1], hpGamma_b[ii], 1);
	
//	printf("\nBAbt2\n\n");
//	d_print_pmat(nu2+nx_v[0]+1, nx_v[N], bs, pBAbt2, cnx_v[N]);

	// routine packing the Gammas
	// zero the solution
	for(ii=0; ii<pnz2*cnx_v[N]; ii++) pBAbt2[ii] = 0;

	d_cond_BAb(N, nx_v, nu_v, hpBAbt, work0, hpGamma, pBAbt2);
	
	printf("\nGamma\n\n");
	nu_tmp = 0;
	for(ii=0; ii<N; ii++)
		{
		nu_tmp += nu_v[ii];
		d_print_pmat(nx+1+nu_tmp, nx_v[ii+1], bs, hpGamma[ii], cnx_v[ii+1]);
		}
	
	printf("\nBAbt2\n\n");
	d_print_pmat(nu2+nx_v[0]+1, nx_v[N], bs, pBAbt2, cnx_v[N]);

	d_cond_RSQrq(N, nx_v, nu_v, hpBAbt, hpRSQrq, hpGamma, work1, pRSQrq2);

	printf("\nRSQrq2\n\n");
	d_print_pmat(nu2+nx_v[0]+1, nu2+nx_v[0], bs, pRSQrq2, cnux2);

/************************************************
* free memory
************************************************/	
	
#if MHE!=1
	free(pBAbt0);
	free(b0);
	free(pA);
	free(pRSQrq0);
#endif
	if(N>1) free(pBAbt1);
	if(N>1) free(pRSQrq1);
	free(pRSQrqN);
	free(pBAbt2);
	free(work0);
	free(work1);
	for(ii=0; ii<N; ii++)
		{
	//	free(hpGamma_x0[ii]);
	//	free(hpGamma_u[ii]);
	//	free(hpGamma_b[ii]);
		free(hpGamma[ii]);
		}

/************************************************
* return
************************************************/	

	return 0;

	}
