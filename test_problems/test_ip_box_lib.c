/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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
#include <sys/time.h>
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_ATOM)
#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
#endif

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_solvers.h"
#include "../problem_size.h"
#include "../include/block_size.h"
#include "tools.h"
#include "test_param.h"



/*void openblas_set_num_threads(int num_threads);*/

void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
void dcopy_(int *n, double *dx, int *incx, double *dy, int *incy);
void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
void dscal_(int *n, double *da, double *dx, int *incx);



/************************************************ 
Mass-spring system: nx/2 masses connected each other with springs (in a row), and the first and the last one to walls. nu (<=nx) controls act on the first nu masses. The system is sampled with sampling time Ts=1. 
The system seem to give problems for a large number of states: the riccati procedure with OpenBLAS becomes MUCH slower for approximately nx>=1024 (double) and nx>=64 (float). ? under-flow ?
************************************************/
void mass_spring_system(double Ts, int nx, int nu, int N, double *A, double *B, double *b, double *x0)
	{

	int nx2 = nx*nx;

//	double d0 = 0;
//	double d1 = 1;
//	double dm1 = -1;
//	int i1 = 1;
	int info = 0;
//	char cn = 'n';

	int pp = nx/2; // number of masses
	int mm = nu;   // number of forces
	
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
//	dscal_(&nx2, &Ts, A, &i1);
	dscal_3l(nx2, Ts, A);
	expm(nx, A);
	
	d_zeros(&T, nx, nx);
	d_zeros(&I, nx, nx); for(ii=0; ii<nx; ii++) I[ii*(nx+1)]=1.0; //I = eye(nx);
//	dcopy_(&nx2, A, &i1, T, &i1);
	dmcopy(nx, nx, A, nx, T, nx);
//	daxpy_(&nx2, &dm1, I, &i1, T, &i1);
	daxpy_3l(nx2, -1.0, I, T);
//	dgemm_(&cn, &cn, &nx, &nu, &nx, &d1, T, &nx, Bc, &nx, &d0, B, &nx);
	dgemm_nn_3l(nx, nu, nx, T, nx, Bc, nx, B, nx);
	
	int *ipiv = (int *) malloc(nx*sizeof(int));
//	dgesv_(&nx, &nu, Ac, &nx, ipiv, B, &nx, &info);
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
	
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_ATOM)
/*	printf("\nflush subnormals to zero\n");*/
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
#endif

	int err;
	
	int i, j, ii, jj, idx;
	
	double d1=1.0, d0=0.0, dm1=-1.0;

	int rep, nrep=NREP;

	int nx = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght
	int nb = NB; // number of box constraints (it has to be even, and the first 2*nu constraints are for the inputs, the following for the states)
	
	const int dbs = D_MR; //d_get_mr();
	const int sbs = S_MR; //s_get_mr();
	
	int nz = nx+nu+1;
	int pnz = dbs*((nz+dbs-nu%dbs+dbs-1)/dbs);
	int spnz = sbs*((nz+sbs-nu%sbs+sbs-1)/sbs);
	
/*	printf("\n\n%d %d %d %d\n\n", dbs, pnz, sbs, spnz);*/

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
	x0[0] = 3.5;
	x0[1] = 3.5;
	
//	d_print_mat(nx, nx, A, nx);
//	d_print_mat(nx, nu, B, nx);
//	d_print_mat(nx, 1, b, nx);
//	d_print_mat(nx, 1, x0, nx);
	
	/* packed */
	double *BAb; d_zeros(&BAb, nx, nz);

	dmcopy(nx, nu, B, nx, BAb, nx);
	dmcopy(nx, nx, A, nx, BAb+nu*nx, nx);
	dmcopy(nx, 1 , b, nx, BAb+(nu+nx)*nx, nx);
	
//	d_print_mat(nx, nx+nu+1, BAb, nx);

	/* transposed */
	double *BAbt; d_zeros_align(&BAbt, pnz, pnz);
	for(ii=0; ii<nx; ii++)
		for(jj=0; jj<nz; jj++)
			{
			BAbt[jj+pnz*ii] = BAb[ii+nx*jj];
			}

//	d_print_mat(nz, nx+1, BAbt, pnz);
	
	/* packed into contiguous memory */
	double *pBAbt; d_zeros_align(&pBAbt, pnz, pnz);
	d_cvt_mat2pmat(nz, nx, 0, dbs, BAbt, pnz, pBAbt, pnz);

	float *psBAbt; s_zeros_align(&psBAbt, spnz, spnz);
	s_cvt_d2s_pmat(nz, nx, dbs, pBAbt, pnz, sbs, psBAbt, spnz);

/*	d_print_pmat(nz, nx, dbs, pBAbt, pnz);*/
/*	s_print_pmat(nz, nx, sbs, psBAbt, spnz);*/
/*	return 0;*/

/************************************************
* box constraints
************************************************/	

	double *db; d_zeros(&db, nb, 1);
	for(jj=0; jj<2*nu; jj++)
		db[jj] = -0.5;   // umin & -umax
	for(; jj<nb; jj++)
		db[jj] = -4.0;   // xmin & -xmax

/************************************************
* cost function
************************************************/	

	double *Q; d_zeros_align(&Q, pnz, pnz);
	for(ii=0; ii<nu; ii++) Q[ii*(pnz+1)] = 2.0;
	for(; ii<pnz; ii++) Q[ii*(pnz+1)] = 1.0;
	for(ii=0; ii<nz; ii++) Q[nx+nu+ii*pnz] = 0.1;
	Q[(nx+nu)*(pnz+1)] = 1e6;
	
	/* packed into contiguous memory */
	double *pQ; d_zeros_align(&pQ, pnz, pnz);
	d_cvt_mat2pmat(nz, nz, 0, dbs, Q, pnz, pQ, pnz);

	float *psQ; s_zeros_align(&psQ, spnz, spnz);
	s_cvt_d2s_pmat(nz, nz, dbs, pQ, pnz, sbs, psQ, spnz);
/*	d_cvt_mat2pmat(nz, nz, 0, dbs, Q, pnz, pQ, pnz);*/

/************************************************
* matrices series
************************************************/	
	double *(hpQ[N+1]);
	float *(hpsQ[N+1]);
	double *(hux[N+1]);
	double *(hpBAbt[N]);
	float *(hpsBAbt[N]);
	double *(hdb[N+1]);
	for(jj=0; jj<N; jj++)
		{
		d_zeros_align(&hpQ[jj], pnz, pnz);
		s_zeros_align(&hpsQ[jj], spnz, spnz);
		d_zeros_align(&hux[jj], nz, 1);
		hpBAbt[jj] = pBAbt;
		hpsBAbt[jj] = psBAbt;
		hdb[jj] = db;
		}
	d_zeros_align(&hpQ[N], pnz, pnz);
	s_zeros_align(&hpsQ[N], spnz, spnz);
	d_zeros_align(&hux[N], nz, 1);
	hdb[jj] = db;
	
	// starting guess
	for(jj=0; jj<nx; jj++) hux[0][nu+jj]=x0[jj];
	
/*	double *pL; d_zeros_align(&pL, pnz, pnz);*/
/*	*/
/*	double *pBAbtL; d_zeros_align(&pBAbtL, pnz, pnz);*/

/************************************************
* riccati-like iteration
************************************************/

	int kk=0; // acutal number of iterations
	char prec = PREC; // double/single precision
	double sp_thr = SP_THR; // threshold to switch between double and single precision
	int k_max = K_MAX; // maximum number of iterations in the IP method
	double tol = TOL; // tolerance in the duality measure
	double sigma[] = {0.4, 0.3, 0.01}; // control primal-dual IP behaviour
	double *info; d_zeros(&info, 5, k_max); // infos from the IP routine
	
	double *work; d_zeros_align(&work, 2*((N+1)*(pnz*pnz+pnz+5*nb)+2*pnz*pnz), 1); // work space



	/* initizile the cost function */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<pnz*pnz; jj++) hpQ[ii][jj]=pQ[jj];
		}
	for(jj=0; jj<pnz*pnz; jj++) hpQ[N][jj]=pQ[jj];

	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<spnz*spnz; jj++) hpsQ[ii][jj]=psQ[jj];
		}
	for(jj=0; jj<spnz*spnz; jj++) hpsQ[N][jj]=psQ[jj];



	// initial states
	double xx0[] = {3.5, 3.5, 3.66465, 2.15833, 1.81327, -0.94207, 1.86531, -2.35760, 2.91534, 1.79890, -1.49600, -0.76600, -2.60268, 1.92456, 1.66630, -2.28522, 3.12038, 1.83830, 1.93519, -1.87113};



	/* warm up */

	// initialize states and inputs
	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx+nu; jj++)
			hux[ii][jj] = 0;

	hux[0][nu+0] = xx0[2*idx];
	hux[0][nu+1] = xx0[2*idx+1];

	// call the IP solver
	ip_d_box(prec, sp_thr, &kk, k_max, tol, sigma, info, nx, nu, N, nb, hpBAbt, hpsBAbt, hpQ, hpsQ, hdb, hux, work);



	/* timing */
	struct timeval tv0, tv1;
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

		hux[0][nu+0] = xx0[2*idx];
		hux[0][nu+1] = xx0[2*idx+1];

		// call the IP solver
		ip_d_box(prec, sp_thr, &kk, k_max, tol, sigma, info, nx, nu, N, nb, hpBAbt, hpsBAbt, hpQ, hpsQ, hdb, hux, work);

		}
	
	gettimeofday(&tv1, NULL); // stop
	


	float time = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	
	printf("\nnx\tnu\tN\tkernel\n\n");
	printf("\n%d\t%d\t%d\t%e\n\n", nx, nu, N, time);
	
	if(PRINTSTAT==1)
		{

		for(jj=0; jj<kk; jj++)
			printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, info[5*jj], info[5*jj+1], info[5*jj+2], info[5*jj+2], info[5*jj+3], info[5*jj+4], info[5*jj+4]);
		printf("\n");
		
		}

	if(PRINTRES==1)
		{

		printf("\nu = \n\n");
		for(ii=0; ii<N; ii++)
			d_print_mat(1, nu, hux[ii], 1);
		
		}

/************************************************
* free memory and return
************************************************/

	free(A);
	free(B);
	free(b);
	free(x0);
	free(BAb);
	free(BAbt);
	free(pBAbt);
	free(psBAbt);
	free(db);
	free(Q);
	free(pQ);
/*	free(pL);*/
/*	free(pBAbtL);*/
	free(work);
	free(info);
	for(jj=0; jj<N; jj++)
		{
		free(hpQ[jj]);
		free(hux[jj]);
		}
	free(hpQ[N]);
	free(hux[N]);



	return 0;

	}



