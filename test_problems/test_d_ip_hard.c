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



#if 0

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
/*	printf("\nflush subnormals to zero\n");*/
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
#endif

	int ii, jj, idx;
	
	int rep, nrep=NREP;

	int nx = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght
#if 1
	int nb  = NB; // number of box constrained inputs and states
	int ng  = 0; //4;  // number of general constraints
	int ngN = 0;//nx;
#else
	int nb  = 0;//NB; // number of box constrained inputs and states
	int ng  = nx+nu;//0; //4;  // number of general constraints
	int ngN = nx;//0; //4;  // number of general constraints at last stage
#endif

	int nbu = nu<nb ? nu : nb ;
	int ngu = nu<ng ? nu : ng ; // TODO remove when not needed any longer in tests

	printf(" Test problem: mass-spring system with %d masses and %d controls.\n", nx/2, nu);
	printf("\n");
	printf(" MPC problem size: %d states, %d inputs, %d horizon length, %d two-sided box constraints, %d two-sided general constraints, %d two-sided general constraints on the last stage.\n", nx, nu, N, nb, ng, ngN);
	printf("\n");
#if IP == 1
	printf(" IP method parameters: primal-dual IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", K_MAX, MU_TOL);
#elif IP == 2
	printf(" IP method parameters: predictor-corrector IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", K_MAX, MU_TOL);
#else
	printf(" Wrong value for IP solver choice: %d\n", IP);
#endif

	int info = 0;
		
	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	const int nz   = nx+nu+1;
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	const int cnxg = ncl*((ng+nx+ncl-1)/ncl);
	//const int pnb = bs*((2*nb+bs-1)/bs); // packed number of box constraints
	const int pnb  = bs*((nb+bs-1)/bs); // simd aligned number of one-sided box constraints !!!!!!!!!!!!
	const int png  = bs*((ng+bs-1)/bs); // simd aligned number of one-sided box constraints !!!!!!!!!!!!
	const int pngN = bs*((ngN+bs-1)/bs); // simd aligned number of one-sided box constraints !!!!!!!!!!!!
	const int anz  = nal*((nz+nal-1)/nal);
	const int anx  = nal*((nx+nal-1)/nal);
//	const int anb  = nal*((2*nb+nal-1)/nal); // cache aligned number of box constraints
	//const int anb = nal*((nb+nal-1)/nal); // cache aligned number of one-sided box constraints !!!!!!!!!!!!
	//const int ang = nal*((ng+nal-1)/nal); // cache aligned number of one-sided box constraints !!!!!!!!!!!!

//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	
/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx, nx); // states update matrix

	double *B; d_zeros(&B, nx, nu); // inputs matrix

	double *b; d_zeros(&b, nx, 1); // states offset
	double *x0; d_zeros(&x0, nx, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx, nu, N, A, B, b, x0);
	
	for(jj=0; jj<nx; jj++)
		b[jj] = 0.1;
	
	for(jj=0; jj<nx; jj++)
		x0[jj] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;
	
//	d_print_mat(nx, nx, A, nx);
//	d_print_mat(nx, nu, B, nx);
//	d_print_mat(nx, 1, b, nx);
//	d_print_mat(nx, 1, x0, nx);
	
	/* packed */
/*	double *BAb; d_zeros(&BAb, nx, nz);*/

/*	dmcopy(nx, nu, B, nx, BAb, nx);*/
/*	dmcopy(nx, nx, A, nx, BAb+nu*nx, nx);*/
/*	dmcopy(nx, 1 , b, nx, BAb+(nu+nx)*nx, nx);*/
	
	/* transposed */
/*	double *BAbt; d_zeros_align(&BAbt, pnz, pnz);*/
/*	for(ii=0; ii<nx; ii++)*/
/*		for(jj=0; jj<nz; jj++)*/
/*			{*/
/*			BAbt[jj+pnz*ii] = BAb[ii+nx*jj];*/
/*			}*/

	/* packed into contiguous memory */
	double *pBAbt; d_zeros_align(&pBAbt, pnz, cnx);
/*	d_cvt_mat2pmat(nz, nx, BAbt, pnz, 0, pBAbt, cnx);*/
/*	d_cvt_tran_mat2pmat(nx, nz, BAb, nx, 0, pBAbt, cnx);*/

	d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, pBAbt, cnx);
	d_cvt_tran_mat2pmat(nx, nx, A, nx, nu, pBAbt+nu/bs*cnx*bs+nu%bs, cnx);
	for (jj = 0; jj<nx; jj++)
		pBAbt[(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[jj];

/*	d_print_pmat (nz, nx, bs, pBAbt, cnx);*/
/*	exit(1);*/

/************************************************
* box & general constraints
************************************************/	

	double *d; d_zeros_align(&d, 2*pnb+2*png, 1);
	for(jj=0; jj<nbu; jj++)
		{
		d[jj]      = - 0.5;   //   umin
		d[pnb+jj]  = - 0.5;   // - umax
		}
	for(; jj<nb; jj++)
		{
		d[jj]      = - 4.0;   //   xmin
		d[pnb+jj]  = - 4.0;   // - xmax
		}
	for(jj=0; jj<ngu; jj++)
		{
		d[2*pnb+jj]     = - 0.5;   //   umin
		d[2*pnb+png+jj] = - 0.5;   // - umax
		}
	for(; jj<ng; jj++)
		{
		d[2*pnb+jj]     = - 4.0;   //   xmin
		d[2*pnb+png+jj] = - 4.0;   // - xmax
		}

	double *dN; d_zeros_align(&dN, 2*pnb+2*pngN, 1);
	for(jj=nu; jj<nb; jj++)
		{
		dN[jj]      = - 4.0;   //   xmin
		dN[pnb+jj]  = - 4.0;   // - xmax
		}
	for(jj=0; jj<ngN; jj++)
		{
		dN[2*pnb+jj]      = - 4.0;   //   xmin
		dN[2*pnb+pngN+jj] = - 4.0;   // - xmax
		}
	//d_print_mat(1, 2*pnb+2*png, d, 1);
	//d_print_mat(1, 2*pnb+2*pngN, dN, 1);
	//exit(1);
	
	double *D; d_zeros(&D, ng, nu);
	for(jj=0; jj<ngu; jj++)
		D[jj*(ng+1)] = 1.0;
	double *C; d_zeros(&C, ng, nx);
	for(; jj<ng; jj++)
		C[jj*(ng+1)-ngu*ng] = 1.0;
	double *CN; d_zeros(&CN, ngN, nx);
	for(jj=0; jj<ngN; jj++)
		CN[jj*(ngN+1)] = 1.0;
	//d_print_mat(ng, nu, D, ng);
	//d_print_mat(ng, nx, C, ng);
	//d_print_mat(ngN, nx, CN, ngN);
	//exit(1);

	// first stage
	double *pDCt0; d_zeros_align(&pDCt0, pnz, cng);
	d_cvt_tran_mat2pmat(ng, nu, D, ng, 0, pDCt0, cng);
	// middle stages
	double *pDCtn; d_zeros_align(&pDCtn, pnz, cng);
	d_cvt_tran_mat2pmat(ng, nu, D, ng, 0, pDCtn, cng);
	d_cvt_tran_mat2pmat(ng, nx, C, ng, nu, pDCtn+nu/bs*cng*bs+nu%bs, cng);
	// last stages
	double *pDCtN; d_zeros_align(&pDCtN, pnz, cngN);
	d_cvt_tran_mat2pmat(ngN, nx, CN, ngN, nu, pDCtN+nu/bs*cngN*bs+nu%bs, cngN);
	//d_print_pmat(nu+nx, ng, bs, pDCt0, cng);
	//d_print_pmat(nu+nx, ng, bs, pDCtn, cng);
	//d_print_pmat(nu+nx, ngN, bs, pDCtN, cngN);
	//exit(1);
	// TODO arrived here working on general constraints

/************************************************
* cost function
************************************************/	
	
	double *Q0; d_zeros(&Q0, nx, nx);
	for(ii=0; ii<nx; ii++) Q0[ii*(nx+1)] = 1.0;

	double *R0; d_zeros(&R0, nu, nu);
	for(ii=0; ii<nu; ii++) R0[ii*(nu+1)] = 2.0;

	double *S0; d_zeros(&S0, nu, nx);

	double *St0; d_zeros(&St0, nu, nx);
	for(ii=0; ii<nx; ii++)
		for(jj=0; jj<nu; jj++)
			St0[ii+nx*jj] = S0[jj+nu*ii];

	double *q0; d_zeros(&q0, nx, 1);
	for(ii=0; ii<nx; ii++) q0[ii] = 0.1;

	double *r0; d_zeros(&r0, nu, 1);
	for(ii=0; ii<nu; ii++) r0[ii] = 0.1;

	// packed matrix (symmetric)
	double *Q; d_zeros_align(&Q, nz, nz);
	//for(ii=0; ii<nu; ii++) Q[ii*(pnz+1)] = 2.0;
	//for(; ii<pnz; ii++) Q[ii*(pnz+1)] = 1.0;
	//for(ii=0; ii<nu+nx; ii++) Q[nx+nu+ii*pnz] = 0.1;
/*	Q[(nx+nu)*(pnz+1)] = 1e35; // large enough (not needed any longer) */
	d_copy_mat(nu, nu, R0, nu, Q, nz);
	d_copy_mat(nx, nx, Q0, nx, Q+nu*(nz+1), nz);
	d_copy_mat(nu, nx, S0, nu, Q+nu*nz, nz);
	d_copy_mat(nx, nu, St0, nx, Q+nu, nz);
	d_copy_mat(nu, 1, r0, nu, Q+(nu+nx)*nz, nz);
	d_copy_mat(1, nu, r0, 1, Q+nu+nx, nz);
	d_copy_mat(nx, 1, q0, nx, Q+nu+(nu+nx)*nz, nz);
	d_copy_mat(1, nx, q0, 1, Q+nu+nx+nu*nz, nz);
	//d_print_mat(nz, nz, Q, nz);
	
	/* packed into contiguous memory */
	double *pQ; d_zeros_align(&pQ, pnz, cnz);
	d_cvt_mat2pmat(nz, nz, Q, nz, 0, pQ, cnz);

	// linear part copied on another vector
	double *q; d_zeros_align(&q, anz, 1);
	for(ii=0; ii<nu+nx; ii++) q[ii] = Q[nx+nu+ii*nz];

	// maximum element in cost functions
	double mu0 = 1.0;
	for(ii=0; ii<nu+nx; ii++)
		for(jj=0; jj<nu+nx; jj++)
			mu0 = fmax(mu0, Q[jj+nz*ii]);
	//mu0 = 1;
	//printf("\n mu0 = %f\n", mu0);

/************************************************
* high-level interface: repmat
************************************************/	


	double *rA; d_zeros(&rA, nx, N*nx);
	d_rep_mat(N, nx, nx, A, nx, rA, nx);

	double *rB; d_zeros(&rB, nx, N*nu);
	d_rep_mat(N, nx, nu, B, nx, rB, nx);

	double *rC; d_zeros(&rC, ng, (N+1)*nx);
	d_rep_mat(N, ng, nx, C, ng, rC+nx*ng, ng);

	double *rD; d_zeros(&rD, ng, N*nu);
	d_rep_mat(N, ng, nu, D, ng, rD, ng);

	double *rb; d_zeros(&rb, nx, N*1);
	d_rep_mat(N, nx, 1, b, nx, rb, nx);

	double *rQ; d_zeros(&rQ, nx, N*nx);
	d_rep_mat(N, nx, nx, Q0, nx, rQ, nx);

	double *rQf; d_zeros(&rQf, nx, nx);
	d_copy_mat(nx, nx, Q0, nx, rQf, nx);

	double *rS; d_zeros(&rS, nu, N*nx);
	d_rep_mat(N, nu, nx, S0, nu, rS, nu);

	double *rR; d_zeros(&rR, nu, N*nu);
	d_rep_mat(N, nu, nu, R0, nu, rR, nu);

	double *rq; d_zeros(&rq, nx, N);
	d_rep_mat(N, nx, 1, q0, nx, rq, nx);

	double *rqf; d_zeros(&rqf, nx, 1);
	d_copy_mat(nx, 1, q0, nx, rqf, nx);

	double *rr; d_zeros(&rr, nu, N);
	d_rep_mat(N, nu, 1, r0, nu, rr, nu);

	double *lb; d_zeros(&lb, nb, 1);
	for(ii=0; ii<nb; ii++)
		lb[ii] = d[ii];
	double *rlb; d_zeros(&rlb, nb, N+1);
	d_rep_mat(N+1, nb, 1, lb, nb, rlb, nb);
	//d_print_mat(nb, N+1, rlb, nb);

	double *lg; d_zeros(&lg, ng, 1);
	for(ii=0; ii<ng; ii++)
		lg[ii] = d[2*pnb+ii];
	double *rlg; d_zeros(&rlg, ng, N);
	d_rep_mat(N, ng, 1, lg, ng, rlg, ng);
	//d_print_mat(ng, N, rlg, ng);

	double *lgN; d_zeros(&lgN, ngN, 1);
	for(ii=0; ii<ngN; ii++)
		lgN[ii] = dN[2*pnb+ii];
	//d_print_mat(ngN, 1, lgN, ngN);

	double *ub; d_zeros(&ub, nb, 1);
	for(ii=0; ii<nb; ii++)
		ub[ii] = - d[pnb+ii];
	double *rub; d_zeros(&rub, nb, N+1);
	d_rep_mat(N+1, nb, 1, ub, nb, rub, nb);
	//d_print_mat(nb, N+1, rub, nb);

	double *ug; d_zeros(&ug, ng, 1);
	for(ii=0; ii<ng; ii++)
		ug[ii] = - d[2*pnb+png+ii];
	double *rug; d_zeros(&rug, ng, N);
	d_rep_mat(N, ng, 1, ug, ng, rug, ng);
	//d_print_mat(ng, N, rug, ng);

	double *ugN; d_zeros(&ugN, ngN, 1);
	for(ii=0; ii<ngN; ii++)
		ugN[ii] = - dN[2*pnb+pngN+ii];
	//d_print_mat(ngN, 1, ugN, ngN);
	//exit(1);

	double *rx; d_zeros(&rx, nx, N+1);

	double *ru; d_zeros(&ru, nu, N);

	double *rpi; d_zeros(&rpi, nx, N+1);

	double *rlam; d_zeros(&rlam, N*2*(nb+ng)+2*(nb+ngN), 1);

	double *rt; d_zeros(&rt, N*2*(nb+ng)+2*(nb+ngN), 1);


	//d_print_mat(nb+ng, 3, rlb, nb+ng);
	//d_print_mat(nb+ng, 3, rub, nb+ng);
	//exit(1);

/************************************************
* low-level interface: series of panel format of packed matrices
************************************************/	

	double *hpQ[N+1];
	double *hq[N+1];
	double *hux[N+1];
	double *hpi[N+1];
	double *hlam[N+1];
	double *ht[N+1];
	double *hpBAbt[N];
	double *hd[N+1];
	double *hpDCt[N+1];
	double *hrb[N];
	double *hrq[N+1];
	double *hrd[N+1];

	for(jj=0; jj<N; jj++)
		{
		//d_zeros_align(&hpQ[jj], pnz, cnz);
		hpQ[jj] = pQ;
		//d_zeros_align(&hq[jj], anz, 1);
		hq[jj] = q;
		d_zeros_align(&hux[jj], anz, 1);
		d_zeros_align(&hpi[jj], anx, 1);
		d_zeros_align(&hlam[jj],2*pnb+2*png, 1);
		d_zeros_align(&ht[jj], 2*pnb+2*png, 1);
		hpBAbt[jj] = pBAbt;
		hd[jj] = d;
		hpDCt[jj] = pDCtn;
		d_zeros_align(&hrb[jj], anx, 1);
		d_zeros_align(&hrq[jj], anz, 1);
		d_zeros_align(&hrd[jj], 2*pnb+2*png, 1);
		}
	//d_zeros_align(&hpQ[N], pnz, cnz);
	hpQ[N] = pQ;
	//d_zeros_align(&hq[N], anz, 1);
	hq[N] = q;
	d_zeros_align(&hux[N], anz, 1);
	d_zeros_align(&hpi[N], anx, 1);
	d_zeros_align(&hlam[N], 2*pnb+2*pngN, 1);
	d_zeros_align(&ht[N], 2*pnb+2*pngN, 1);
	hd[N] = dN;
	hpDCt[0] = pDCt0;
	hpDCt[N] = pDCtN;
	d_zeros_align(&hrq[N], anz, 1);
	d_zeros_align(&hrd[N], 2*pnb+2*pngN, 1); // TODO pnb
	
	// starting guess
	for(jj=0; jj<nx; jj++) hux[0][nu+jj]=x0[jj];

/************************************************
* riccati-like iteration
************************************************/

	//double *work; d_zeros_align(&work, (N+1)*(pnz*cnl + 4*anz + 4*anb + 2*anx) + 3*anz, 1); // work space
	//double *work; d_zeros_align(&work, (N+1)*(pnz*cnl + 5*anz + 4*anb + 2*anx) + 3*anz, 1); // work space TODO change work space on other files !!!!!!!!!!!!!
	//double *work; d_zeros_align(&work, (N+1)*(pnz*cnl + 5*anz + 4*anb + 2*anx) + anz + pnz*cnx, 1); // work space TODO change work space on other files !!!!!!!!!!!!!
	double *work; d_zeros_align(&work, (N+1)*(pnz*cnl + pnz + 5*anz + 10*(pnb+png) + 2*anx) + 10*(pngN-png) + anz + pnz*(cngN<cnxg ? cnxg : cngN), 1); // work space TODO change work space on other files !!!!!!!!!!!!!
/*	for(jj=0; jj<( (N+1)*(pnz*cnl + 4*anz + 4*anb + 2*anx) + 3*anz ); jj++) work[jj] = -1.0;*/
	int kk = 0; // acutal number of iterations
	int rkk = 0; // acutal number of iterations
/*	char prec = PREC; // double/single precision*/
/*	double sp_thr = SP_THR; // threshold to switch between double and single precision*/
	int k_max = K_MAX; // maximum number of iterations in the IP method
	double mu_tol = MU_TOL; // tolerance in the duality measure
	double alpha_min = ALPHA_MIN; // minimum accepted step length
	double sigma[] = {0.4, 0.3, 0.01}; // control primal-dual IP behaviour
	double *stat; d_zeros(&stat, 5, k_max); // stats from the IP routine
	int compute_mult = COMPUTE_MULT;
	int warm_start = WARM_START;
	double mu = -1.0;
	int hpmpc_status;
	
	double *rwork; d_zeros(&rwork, hpmpc_ip_hard_mpc_dp_work_space_tv(N, nx, nu, nb, ng, ngN), 1);
	double *rstat; d_zeros(&rstat, 5, k_max); // stats from the IP routine
	int compute_res = 1; // flag to control the computation of residuals on exit (high-level interface only)
	double inf_norm_res[4] = {}; // infinity norm of residuals: rq, rb, rd, mu


	/* initizile the cost function */
//	for(ii=0; ii<N; ii++)
//		{
//		for(jj=0; jj<pnz*cnz; jj++) hpQ[ii][jj]=pQ[jj];
//		}
//	for(jj=0; jj<pnz*cnz; jj++) hpQ[N][jj]=pQ[jj];



	// solution of unconstrained problem
//	double *(hpL[N+1]);
//	double *(hPb[N]);
//	for(jj=0; jj<=N; jj++)
//		{
//		d_zeros_align(&hpL[jj], pnz, cnl);
//		d_zeros_align(&hPb[jj], anx, 1);
//		}
//	double *diag; d_zeros_align(&diag, pnz, 1);
//	int update_hessian = 0;
//	double **Qd;
//	double **Ql;
//	d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, update_hessian, Qd, Ql, hux, hpL, work, diag, COMPUTE_MULT, hpi);



	// initial states
	double xx0[] = {2.5, 2.5, 3.66465, 2.15833, 1.81327, -0.94207, 1.86531, -2.35760, 2.91534, 1.79890, -1.49600, -0.76600, -2.60268, 1.92456, 1.66630, -2.28522, 3.12038, 1.83830, 1.93519, -1.87113};



	/* warm up */

	// initialize states and inputs
	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx+nu; jj++)
			hux[ii][jj] = 0;

	hux[0][nu+0] = 2.5; //xx0[0];
	hux[0][nu+1] = 2.5; //xx0[1];

//	// solution of unconstrained problem as warm start
//	warm_start = 1;
//	d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi);
//	//d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, update_hessian, Qd, Ql, hux, hpL, work, diag, COMPUTE_MULT, hpi);

	// call the IP solver
//	if(FREE_X0==0)
//		{
		if(IP==1)
			hpmpc_status = d_ip_hard_mpc(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
		else
			hpmpc_status = d_ip2_hard_mpc(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
//		}
//	else
//		{
//		if(IP==1)
//			hpmpc_status = d_ip_box_mhe_old(&kk, k_max, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
//		else
//			hpmpc_status = d_ip2_box_mhe_old(&kk, k_max, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
//		}



	int kk_avg = 0;

	/* timing */
	struct timeval tv0, tv1, tv2;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{

		idx = rep%10;
		x0[0] = xx0[2*idx];
		x0[1] = xx0[2*idx+1];

		// initialize states and inputs
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx+nu; jj++)
				hux[ii][jj] = 0;

		hux[0][nu+0] = 2.5; //xx0[2*idx];
		hux[0][nu+1] = 2.5; //xx0[2*idx+1];

//		// solution of unconstrained problem as warm start
//		warm_start = 1;
//		d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi);
//		//d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, update_hessian, Qd, Ql, hux, hpL, work, diag, COMPUTE_MULT, hpi);

		// call the IP solver
//		if(FREE_X0==0)
//			{
			if(IP==1)
				hpmpc_status = d_ip_hard_mpc(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
			else
				hpmpc_status = d_ip2_hard_mpc(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
//			}
//		else
//			{
//			if(IP==1)
//				hpmpc_status = d_ip_box_mhe_old(&kk, k_max, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
//			else
//				hpmpc_status = d_ip2_box_mhe_old(&kk, k_max, mu_tol, alpha_min, warm_start, sigma, stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
//			}

		kk_avg += kk;

		}
	
	gettimeofday(&tv1, NULL); // stop

	int rkk_avg = 0;

	for(rep=0; rep<nrep; rep++)
		{

		// initialize states and inputs to zero
		for(ii=0; ii<(nx*(N+1)); ii++)
			rx[ii] = 0;
		for(ii=0; ii<(nu*N); ii++)
			ru[ii] = 0;

		idx = rep%10;
		rx[0] = xx0[2*idx];
		rx[1] = xx0[2*idx+1];

		hpmpc_status = fortran_order_ip_hard_mpc_tv(&rkk, k_max, mu0, mu_tol, 'd', N, nx, nu, nb, ng, ngN, rA, rB, rb, rQ, rQf, rS, rR, rq, rqf, rr, rlb, rub, rC, rD, rlg, rug, CN, lgN, ugN, rx, ru, rwork, rstat, compute_res, inf_norm_res, compute_mult, rpi, rlam, rt);

		rkk_avg += rkk;

		}
	
	gettimeofday(&tv2, NULL); // stop
	


	double time = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	double rtime = (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
	
/*	printf("\nnx\tnu\tN\tkernel\n\n");*/
/*	printf("\n%d\t%d\t%d\t%e\n\n", nx, nu, N, time);*/
	
	printf("\n");
	printf(" Average number of iterations over %d runs: %5.1f (low-level interface)\n", nrep, kk_avg / (double) nrep);
	printf(" Average number of iterations over %d runs: %5.1f (high-level interface)\n", nrep, rkk_avg / (double) nrep);
	printf("\n");
	printf(" Average solution time over %d runs: %5.2e seconds (low-level interface)\n", nrep, time);
	printf(" Average solution time over %d runs: %5.2e seconds (high-level interface)\n", nrep, rtime);
	printf("\n");



	// restore linear part of cost function 
//	for(ii=0; ii<N; ii++)
//		{
//		for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+nz*jj];
//		}
//	for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+nz*jj];

	// residuals computation
//	if(FREE_X0==0)
		d_res_ip_hard_mpc(nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);
//	else
//		d_res_ip_box_mhe_old(nx, nu, N, nb, hpBAbt, hpQ, hq, hux, hdb, hpi, hlam, ht, hrq, hrb, hrd, &mu);


	if(PRINTSTAT==1)
		{

		printf("\n");
		printf("\n");
		printf(" Print IP statistics of the last run\n");
		printf("\n");

		for(jj=0; jj<kk; jj++)
			printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
		printf("\n");
		
		for(jj=0; jj<rkk; jj++)
			printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, rstat[5*jj], rstat[5*jj+1], rstat[5*jj+2], rstat[5*jj+2], rstat[5*jj+3], rstat[5*jj+4], rstat[5*jj+4]);
		printf("\n");

		if(compute_res)
			printf("\n\nInfinity norm of residuals: %e (rq), %e (rb), %e (rd), %e (mu)\n", inf_norm_res[0], inf_norm_res[1], inf_norm_res[2], inf_norm_res[3]);
		
		}

	if(PRINTRES==1)
		{

		printf("\n");
		printf("\n");
		printf(" Print solution\n");
		printf("\n");

		printf("\nu = \n\n");
		for(ii=0; ii<N; ii++)
			d_print_mat(1, nu, hux[ii], 1);
		
		printf("\nx = \n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nx, hux[ii]+nu, 1);
		
		printf("\nlam = \n\n");
		for(ii=0; ii<N; ii++)
			d_print_mat(1, 2*pnb+2*png, hlam[ii], 1);
		d_print_mat(1, 2*pnb+2*pngN, hlam[N], 1);
		
		printf("\nru = \n\n");
		d_print_mat(nu, N, ru, nu);

		printf("\nrx = \n\n");
		d_print_mat(nx, N+1, rx, nx);
		
		}

	if(PRINTRES==1 && COMPUTE_MULT==1)
		{
		// print result 
		// print result 
		printf("\n");
		printf("\n");
		printf(" Print residuals\n\n");
		printf("\n");
		printf("\n");
		printf("rq = \n\n");
//		if(FREE_X0==0)
//			{
			d_print_mat(1, nu, hrq[0], 1);
			for(ii=1; ii<=N; ii++)
/*				d_print_mat_e(1, nx+nu, hrq[ii], 1);*/
				d_print_mat(1, nx+nu, hrq[ii], 1);
//			}
//		else
//			{
//			for(ii=0; ii<=N; ii++)
///*				d_print_mat_e(1, nx+nu, hrq[ii], 1);*/
//				d_print_mat(1, nx+nu, hrq[ii], 1);
//			}
		printf("\n");
		printf("\n");
		printf("rb = \n\n");
		for(ii=0; ii<N; ii++)
/*			d_print_mat_e(1, nx, hrb[ii], 1);*/
			d_print_mat(1, nx, hrb[ii], 1);
		printf("\n");
		printf("\n");
		printf("rd = \n\n");
		for(ii=0; ii<N; ii++)
/*			d_print_mat_e(1, 2*nb, hrd[ii], 1);*/
			d_print_mat(1, 2*pnb+2*png, hrd[ii], 1);
		d_print_mat(1, 2*pnb+2*pngN, hrd[N], 1);
		printf("\n");
		printf("\n");
		printf("mu = %e\n\n", mu);
		
		}

/*	printf("\nnx\tnu\tN\tkernel\n\n");*/
/*	printf("\n%d\t%d\t%d\t%e\n\n", nx, nu, N, time);*/
	
/************************************************
* free memory and return
************************************************/

	free(A);
	free(B);
	free(C);
	free(CN);
	free(D);
	free(b);
	free(x0);
/*	free(BAb);*/
/*	free(BAbt);*/
	free(pBAbt);
	free(pDCt0);
	free(pDCtn);
	free(pDCtN);
	free(d);
	free(dN);
	free(Q);
	free(pQ);
	free(q);
	free(work);
	free(stat);
	free(Q0);
	free(S0);
	free(St0);
	free(R0);
	free(q0);
	free(r0);
	free(rA);
	free(rB);
	free(rC);
	free(rD);
	free(rb);
	free(rQ);
	free(rQf);
	free(rS);
	free(rR);
	free(rq);
	free(rqf);
	free(rr);
	free(lb);
	free(lg);
	free(lgN);
	free(ub);
	free(ug);
	free(ugN);
	free(rlb);
	free(rlg);
	free(rub);
	free(rug);
	free(rx);
	free(ru);
	free(rwork);
	for(jj=0; jj<N; jj++)
		{
		//free(hpQ[jj]);
		//free(hq[jj]);
		free(hux[jj]);
		free(hpi[jj]);
		free(hlam[jj]);
		free(ht[jj]);
		free(hrb[jj]);
		free(hrq[jj]);
		free(hrd[jj]);
		}
	//free(hpQ[N]);
	//free(hq[N]);
	free(hux[N]);
	free(hpi[N]);
	free(hlam[N]);
	free(ht[N]);
	free(hrq[N]);
	free(hrd[N]);



	return 0;

	}


#else

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
	
	int rep, nrep=NREP;

	int nx = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght
	int nb  = NB; // number of box constrained inputs and states
	int ng  = 0; //4;  // number of general constraints
	
	int nbu = nu<nb ? nu : nb ;
	int nbx = nb-nu>0 ? nb-nu : 0;


	// stage-wise variant size
	int nx_v[N+1];
	nx_v[0] = 0;
	for(ii=1; ii<=N; ii++)
		nx_v[ii] = nx;

	int nu_v[N+1];
	for(ii=0; ii<N; ii++)
		nu_v[ii] = nu;
	nu_v[N] = 0;

	int nb_v[N+1];
	nb_v[0] = nbu;
	for(ii=1; ii<N; ii++)
		nb_v[ii] = nb;
	nb_v[N] = nbx;

	int ng_v[N+1];
	for(ii=0; ii<=N; ii++)
		ng_v[ii] = 0;
	



	printf(" Test problem: mass-spring system with %d masses and %d controls.\n", nx/2, nu);
	printf("\n");
	printf(" MPC problem size: %d states, %d inputs, %d horizon length, %d two-sided box constraints, %d two-sided general constraints.\n", nx, nu, N, nb, ng);
	printf("\n");
#if IP == 1
	printf(" IP method parameters: primal-dual IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", K_MAX, MU_TOL);
#elif IP == 2
	printf(" IP method parameters: predictor-corrector IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", K_MAX, MU_TOL);
#else
	printf(" Wrong value for IP solver choice: %d\n", IP);
#endif

	int info = 0;
		
	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	int pnz = (nu+nx+1+bs-1)/bs*bs;
	int pnu = (nu+bs-1)/bs*bs;
	int pnu1 = (nu+1+bs-1)/bs*bs;
	int pnx = (nx+bs-1)/bs*bs;
	int pnx1 = (nx+1+bs-1)/bs*bs;
	int pnux = (nu+nx+bs-1)/bs*bs;
	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cnux = (nu+nx+ncl-1)/ncl*ncl;

	int pnb_v[N+1]; for(ii=0; ii<=N; ii++) pnb_v[ii] = (nb_v[ii]+bs-1)/bs*bs;
	int cnx_v[N+1]; for(ii=0; ii<=N; ii++) cnx_v[ii] = (nx_v[ii]+ncl-1)/ncl*ncl;


/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx, nx); // states update matrix

	double *B; d_zeros(&B, nx, nu); // inputs matrix

	double *b; d_zeros(&b, nx, 1); // states offset
	double *x0; d_zeros_align(&x0, nx, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx, nu, N, A, B, b, x0);
	
	for(jj=0; jj<nx; jj++)
		b[jj] = 0.1;
	
	for(jj=0; jj<nx; jj++)
		x0[jj] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

	double *pA; d_zeros_align(&pA, pnx, cnx);
	d_cvt_mat2pmat(nx, nx, A, nx, 0, pA, cnx);
//	d_print_pmat(nx, nx, bs, pA, cnx);
	double *b0; d_zeros_align(&b0, pnx, 1);
	dgemv_n_lib(nx, nx, pA, cnx, x0, 1, b, b0);
//	d_print_mat(1, nx, b0, 1);

	double *pBAbt0; d_zeros_align(&pBAbt0, pnu1, cnx);
	d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, pBAbt0, cnx);
	d_cvt_tran_mat2pmat(nx, 1, b0, nx, nu, pBAbt0+nu/bs*bs*cnx+nu%bs, cnx);
//	d_print_pmat(nu+1, nx, bs, pBAbt0, cnx);

	double *pBAbt1; d_zeros_align(&pBAbt1, pnz, cnx);
	d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, pBAbt1, cnx);
	d_cvt_tran_mat2pmat(nx, nx, A, nx, nu, pBAbt1+nu/bs*bs*cnx+nu%bs, cnx);
	d_cvt_tran_mat2pmat(nx, 1, b, nx, nu+nx, pBAbt1+(nu+nx)/bs*bs*cnx+(nu+nx)%bs, cnx);
//	d_print_pmat(nu+nx+1, nx, bs, pBAbt1, cnx);
//	exit(4);


/************************************************
* box & general constraints
************************************************/	

	int *idx0; i_zeros(&idx0, nb_v[0], 1);
	double *d0; d_zeros_align(&d0, 2*pnb_v[0], 1);
	for(jj=0; jj<nbu; jj++)
		{
		d0[jj]          = - 0.5;   //   umin
		d0[pnb_v[0]+jj] = - 0.5;   // - umax
		idx0[jj] = jj;
		}
//	i_print_mat(nbu, 1, idx0, nbu);

	int *idx1; i_zeros(&idx1, nb_v[1], 1);
	double *d1; d_zeros_align(&d1, 2*pnb_v[1], 1);
	for(jj=0; jj<nbu; jj++)
		{
		d1[jj]          = - 0.5;   //   umin
		d1[pnb_v[1]+jj] = - 0.5;   // - umax
		idx1[jj] = jj;
		}
	for(; jj<nb; jj++)
		{
		d1[jj]          = - 4.0;   //   xmin
		d1[pnb_v[1]+jj] = - 4.0;   // - xmax
		idx1[jj] = jj;
		}
//	i_print_mat(nb, 1, idx1, nb);

	int *idxN; i_zeros(&idxN, nb_v[N], 1);
	double *dN; d_zeros_align(&dN, 2*pnb_v[N], 1);
	for(jj=0; jj<nbx; jj++)
		{
		dN[jj]          = - 4.0;   //   xmin
		dN[pnb_v[N]+jj] = - 4.0;   // - xmax
		idxN[jj] = jj;
		}
	//d_print_mat(1, 2*pnb+2*png, d, 1);
	//d_print_mat(1, 2*pnb+2*pngN, dN, 1);
	//exit(1);
	
/************************************************
* cost function
************************************************/	
	
	double *Q; d_zeros(&Q, nx, nx);
	for(ii=0; ii<nx; ii++) Q[ii*(nx+1)] = 1.0;

	double *R; d_zeros(&R, nu, nu);
	for(ii=0; ii<nu; ii++) R[ii*(nu+1)] = 2.0;

	double *S; d_zeros(&S, nu, nx);

	double *q; d_zeros(&q, nx, 1);
	for(ii=0; ii<nx; ii++) q[ii] = 0.1;

	double *r; d_zeros(&r, nu, 1);
	for(ii=0; ii<nu; ii++) r[ii] = 0.2;

	double  *pQ0; d_zeros_align(&pQ0, pnu1, cnu);
	d_cvt_mat2pmat(nu, nu, R, nu, 0, pQ0, cnu);
	d_cvt_tran_mat2pmat(nu, 1, r, nu, nu, pQ0+nu/bs*bs*cnu+nu%bs, cnu);
//	d_print_pmat(nu+1, nu, bs, pQ0, cnu);

	double  *pQ1; d_zeros_align(&pQ1, pnz, cnux);
	d_cvt_mat2pmat(nu, nu, R, nu, 0, pQ1, cnux);
	d_cvt_tran_mat2pmat(nu, nx, S, nu, nu, pQ1+nu/bs*bs*cnux+nu%bs, cnux);
	d_cvt_tran_mat2pmat(nu, 1, r, nu, nu+nx, pQ1+(nu+nx)/bs*bs*cnux+(nu+nx)%bs, cnux);
	d_cvt_mat2pmat(nx, nx, Q, nx, nu, pQ1+nu/bs*bs*cnux+nu%bs+nu*bs, cnux);
	d_cvt_tran_mat2pmat(nx, 1, q, nx, nu+nx, pQ1+(nu+nx)/bs*bs*cnux+(nu+nx)%bs+nu*bs, cnux);
//	d_print_pmat(nu+nx+1, nu+nx, bs, pQ1, cnux);

	double  *pQN; d_zeros_align(&pQN, pnx1, cnx);
	d_cvt_mat2pmat(nx, nx, Q, nx, 0, pQN, cnx);
	d_cvt_tran_mat2pmat(nx, 1, q, nx, nx, pQN+(nx)/bs*bs*cnx+(nx)%bs, cnx);
//	d_print_pmat(nx+1, nx, bs, pQN, cnx);


	// maximum element in cost functions
	double mu0 = 2.0;

/************************************************
* work space
************************************************/	

	double *hpBAbt[N];
	double *hpQ[N+1];
	double *hd[N+1];
	int *idx[N+1];
	double *hux[N+1];
	double *hpi[N+1];
	double *hlam[N+1];
	double *ht[N+1];
	hpBAbt[0] = pBAbt0;
	hpQ[0] = pQ0;
	hd[0] = d0;
	idx[0] = idx0;
	d_zeros_align(&hux[0], pnu, 1);
	d_zeros_align(&hpi[0], pnx, 1);
	d_zeros_align(&hlam[0], 2*pnb_v[0], 1);
	d_zeros_align(&ht[0], 2*pnb_v[0], 1);
	for(ii=1; ii<N; ii++)
		{
		hpBAbt[ii] = pBAbt1;
		hpQ[ii] = pQ1;
		hd[ii] = d1;
		idx[ii] = idx1;
		d_zeros_align(&hux[ii], pnux, 1);
		d_zeros_align(&hpi[ii], pnx, 1);
		d_zeros_align(&hlam[ii], 2*pnb_v[ii], 1);
		d_zeros_align(&ht[ii], 2*pnb_v[ii], 1);
		}
	hpQ[N] = pQN;
	hd[N] = dN;
	idx[N] = idxN;
	d_zeros_align(&hux[N], pnx, 1);
	d_zeros_align(&hpi[N], pnx, 1);
	d_zeros_align(&hlam[N], 2*pnb_v[N], 1);
	d_zeros_align(&ht[N], 2*pnb_v[N], 1);



	double *work; d_zeros_align(&work, d_ip2_hard_mpc_tv_work_space_size_double(N, nx_v, nu_v, nb_v, ng_v), 1);

/************************************************
* call the solver
************************************************/	

	int hpmpc_status;
	int kk;
	int k_max = 10;
	double mu_tol = 1e-20;
	double alpha_min = 1e-8;
	int warm_start = 0;
	double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
	double *stat; d_zeros(&stat, k_max, 5);

	double **dummy;

//	for(ii=0; ii<N; ii++)
//		d_print_pmat(nu_v[ii]+nx_v[ii]+1, nx_v[ii+1], bs, hpBAbt[ii], cnx_v[ii+1]);
//	exit(3);

	struct timeval tv0, tv1, tv2;
	gettimeofday(&tv0, NULL); // stop

	int kk_avg = 0;

//	printf("\nsolution...\n");
	for(rep=0; rep<nrep; rep++)
		{

		hpmpc_status = d_ip2_hard_mpc_tv(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, N, nx_v, nu_v, nb_v, idx, ng_v, hpBAbt, hpQ, dummy, hd, hux, 1, hpi, hlam, ht, work);
		
		kk_avg += kk;

		}
//	printf("\ndone\n");

	gettimeofday(&tv1, NULL); // stop

	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nu_v[ii]+nx_v[ii], hux[ii], 1);

	double time = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\n");
	printf(" Average number of iterations over %d runs: %5.1f\n", nrep, kk_avg / (double) nrep);
	printf(" Average solution time over %d runs: %5.2e seconds\n", nrep, time);
	printf("\n\n");

/************************************************
* free memory
************************************************/	

	free(A);
	free(B);
	free(b);
	d_free_align(x0);
	d_free_align(pBAbt0);
	d_free_align(pBAbt1);
	d_free_align(d0);
	d_free_align(d1);
	d_free_align(dN);
	free(idx0);
	free(idx1);
	free(idxN);
	free(Q);
	free(S);
	free(R);
	free(q);
	free(r);
	d_free_align(pQ0);
	d_free_align(pQ1);
	d_free_align(pQN);
	d_free_align(work);
	free(stat);
	for(ii=0; ii<=N; ii++)
		{
		d_free_align(hux[ii]);
		}
	
	return 0;
	
	}


#endif

