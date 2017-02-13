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

#ifdef BLASFEO
#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_v_aux_ext_dep.h>
#include <blasfeo_d_aux_ext_dep.h>
#include <blasfeo_i_aux_ext_dep.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>
#endif

#include "../include/lqcp_solvers.h"
//#include "../include/mpc_aux.h"
#include "../include/mpc_solvers.h"
#include "../include/c_interface.h"
#include "tools.h"


#define KEEP_X0 0

// printing
#define PRINT 0

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
	
	int rep;

	int nx_ = 60; //8; // number of states (it has to be even for the mass-spring system test problem)
	int nu_ = 6; //3; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = 50; // horizon lenght

	// partial condensing horizon
	int N2 = N*nu_/(0.94224*nx_); //N/2;

	// maximum number of IPM iterations
	int k_max = 100;

	// exit tolerance in duality measure
	double mu_tol = 1e-6;

	// minimum step size lenght 
	double alpha_min = 1e-6;

	// number of calls to solver (for more accurate timings)
	int nrep=1000;



	// stage-wise variant size
	int nx[N+1];
#if KEEP_X0
	nx[0] = nx_;
#else
	nx[0] = 0;
#endif
	for(ii=1; ii<=N; ii++)
		nx[ii] = nx_;

	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_;
	nu[N] = 0;

#if 1
	int nb[N+1];
#if KEEP_X0
	nb[0] = nu[0]+nx[0];
#else
	nb[0] = nu[0];
#endif
	for(ii=1; ii<N; ii++)
		nb[ii] = nu[1]+nx[1];
	nb[N] = nx[N]/2;

	int ng[N+1];
	ng[0] = 0;
	for(ii=1; ii<N; ii++)
		ng[ii] = 0;
	ng[N] = 0;
#elif 0
	int nb[N+1];
#if KEEP_X0
	nb[0] = 0;
#else
	nb[0] = 0;
#endif
	for(ii=1; ii<N; ii++)
		nb[ii] = 0;
	nb[N] = 0;

	int ng[N+1];
#if KEEP_X0
	ng[0] = nu[0]+nx[0];
#else
	ng[0] = nu[0];
#endif
	for(ii=1; ii<N; ii++)
		ng[ii] = nu[1]+nx[1];
	ng[N] = nx[N]/2;
#elif 0
	int nb[N+1];
#if KEEP_X0
	nb[0] = nx[0]/2;
#else
	nb[0] = 0;
#endif
	for(ii=1; ii<N; ii++)
		nb[ii] = nx[1]/2;
	nb[N] = 0;

	int ng[N+1];
#if KEEP_X0
	ng[0] = nu[0]+nx[0]/2;
#else
	ng[0] = nu[0];
#endif
	for(ii=1; ii<N; ii++)
		ng[ii] = nu[1]+nx[1]/2;
	ng[N] = nx[N]/2;
#endif
	


	printf(" Test problem: mass-spring system with %d masses and %d controls.\n", nx_/2, nu_);
	printf("\n");
	printf(" MPC problem size: %d states, %d inputs, %d horizon length, %d two-sided box constraints, %d two-sided general constraints, %d condensed horizon.\n", nx[1], nu[1], N, nb[1], ng[1], N2);
	printf("\n");
	printf(" IP method parameters: predictor-corrector IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", k_max, mu_tol);



/************************************************
* dynamical system
************************************************/	

	double *A; d_zeros(&A, nx_, nx_); // states update matrix

	double *B; d_zeros(&B, nx_, nu_); // inputs matrix

	double *b; d_zeros_align(&b, nx_, 1); // states offset
	double *x0; d_zeros_align(&x0, nx_, 1); // initial state

	double Ts = 0.5; // sampling time
	mass_spring_system(Ts, nx_, nu_, N, A, B, b, x0);
	
	for(jj=0; jj<nx_; jj++)
		b[jj] = 0;  //0.1;
	
	for(jj=0; jj<nx_; jj++)
		x0[jj] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

#if PRINT
	d_print_mat(nx_, nx_, A, nx_);
	d_print_mat(nx_, nu_, B, nu_);
	d_print_mat(1, nx_, b, 1);
	d_print_mat(1, nx_, x0, 1);
#endif

	struct d_strmat sA;
	d_allocate_strmat(nx_, nx_, &sA);
	d_cvt_mat2strmat(nx_, nx_, A, nx_, &sA, 0, 0);
#if PRINT
	d_print_strmat(nx_, nx_, &sA, 0, 0);
#endif

	struct d_strvec sx0;
	d_allocate_strvec(nx_, &sx0);
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
#if PRINT
	d_print_tran_strvec(nx_, &sx0, 0);
#endif

	struct d_strvec sb0;
	d_allocate_strvec(nx_, &sb0);
	d_cvt_vec2strvec(nx_, b, &sb0, 0);
#if PRINT
	d_print_tran_strvec(nx_, &sb0, 0);
#endif
#if ! KEEP_X0
	dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb0, 0, &sb0, 0);
#endif
#if PRINT
	d_print_tran_strvec(nx_, &sb0, 0);
#endif

	struct d_strmat sBAbt0;
	d_allocate_strmat(nu[0]+nx[0]+1, nx[1], &sBAbt0);
	d_cvt_tran_mat2strmat(nx[1], nu[0], B, nx_, &sBAbt0, 0, 0);
	d_cvt_tran_mat2strmat(nx[1], nx[0], A, nx_, &sBAbt0, nu[0], 0);
	drowin_libstr(nx[1], 1.0, &sb0, 0, &sBAbt0, nu[0]+nx[0], 0);
#if PRINT
	d_print_strmat(nu[0]+nx[0]+1, nx[1], &sBAbt0, 0, 0);
#endif

	struct d_strmat sBAbt1;
	struct d_strvec sb1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1);
		d_cvt_tran_mat2strmat(nx[2], nu[1], B, nx_, &sBAbt1, 0, 0);
		d_cvt_tran_mat2strmat(nx[2], nx[1], A, nx_, &sBAbt1, nu[1], 0);
		d_cvt_tran_mat2strmat(nx[2], 1, b, nx_, &sBAbt1, nu[1]+nx[1], 0);
#if PRINT
		d_print_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1, 0, 0);
#endif
		d_allocate_strvec(nx_, &sb1);
		d_cvt_vec2strvec(nx_, b, &sb1, 0);
		}

/************************************************
* cost function
************************************************/	
	
	double *Q; d_zeros(&Q, nx_, nx_);
	for(ii=0; ii<nx_; ii++) Q[ii*(nx_+1)] = 1.0;

	double *R; d_zeros(&R, nu_, nu_);
	for(ii=0; ii<nu_; ii++) R[ii*(nu_+1)] = 2.0;

	double *S; d_zeros(&S, nu_, nx_); // S=0, so no need to update r0

	double *q; d_zeros(&q, nx_, 1);
	for(ii=0; ii<nx_; ii++) q[ii] = 0.1;

	double *r; d_zeros(&r, nu_, 1);
	for(ii=0; ii<nu_; ii++) r[ii] = 0.2;

	struct d_strmat sRSQrq0;
	struct d_strvec srq0;
	d_allocate_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0);
	d_cvt_mat2strmat(nu[0], nu[0], R, nu_, &sRSQrq0, 0, 0);
	d_cvt_tran_mat2strmat(nu[0], nx[0], S, nu_, &sRSQrq0, nu[0], 0);
	d_cvt_mat2strmat(nx[0], nx[0], Q, nx_, &sRSQrq0, nu[0], nu[0]);
	d_cvt_tran_mat2strmat(nu[0], 1, r, nu_, &sRSQrq0, nu[0]+nx[0], 0);
	d_cvt_tran_mat2strmat(nx[0], 1, q, nx_, &sRSQrq0, nu[0]+nx[0], nu[0]);
#if PRINT
	d_print_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0, 0, 0);
#endif
	d_allocate_strvec(nu[0]+nx[0], &srq0);
	d_cvt_vec2strvec(nu[0], r, &srq0, 0);
	d_cvt_vec2strvec(nx[0], q, &srq0, nu[0]);
#if PRINT
	d_print_tran_strvec(nu[0]+nx[0], &srq0, 0);
#endif

	struct d_strmat sRSQrq1;
	struct d_strvec srq1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1);
		d_cvt_mat2strmat(nu[1], nu[1], R, nu_, &sRSQrq1, 0, 0);
		d_cvt_tran_mat2strmat(nu[1], nx[1], S, nu_, &sRSQrq1, nu[1], 0);
		d_cvt_mat2strmat(nx[1], nx[1], Q, nx_, &sRSQrq1, nu[1], nu[1]);
		d_cvt_tran_mat2strmat(nu[1], 1, r, nu_, &sRSQrq1, nu[1]+nx[1], 0);
		d_cvt_tran_mat2strmat(nx[1], 1, q, nx_, &sRSQrq1, nu[1]+nx[1], nu[1]);
#if PRINT
		d_print_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1, 0, 0);
#endif
		d_allocate_strvec(nu[1]+nx[1], &srq1);
		d_cvt_vec2strvec(nu[1], r, &srq1, 0);
		d_cvt_vec2strvec(nx[1], q, &srq1, nu[1]);
#if PRINT
		d_print_tran_strvec(nu[1]+nx[1], &srq1, 0);
#endif
		}

	struct d_strmat sRSQrqN;
	struct d_strvec srqN;
	d_allocate_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN);
	d_cvt_mat2strmat(nu[N], nu[N], R, nu_, &sRSQrqN, 0, 0);
	d_cvt_tran_mat2strmat(nu[N], nx[N], S, nu_, &sRSQrqN, nu[N], 0);
	d_cvt_mat2strmat(nx[N], nx[N], Q, nx_, &sRSQrqN, nu[N], nu[N]);
	d_cvt_tran_mat2strmat(nu[N], 1, r, nu_, &sRSQrqN, nu[N]+nx[N], 0);
	d_cvt_tran_mat2strmat(nx[N], 1, q, nx_, &sRSQrqN, nu[N]+nx[N], nu[N]);
#if PRINT
	d_print_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN, 0, 0);
#endif
	d_allocate_strvec(nu[N]+nx[N], &srqN);
	d_cvt_vec2strvec(nu[N], r, &srqN, 0);
	d_cvt_vec2strvec(nx[N], q, &srqN, nu[N]);
#if PRINT
	d_print_tran_strvec(nu[N]+nx[N], &srqN, 0);
#endif

	// maximum element in cost functions
	double mu0 = 2.0;

/************************************************
* box & general constraints
************************************************/	

	int *idxb0; int_zeros(&idxb0, nb[0], 1);
	double *d0; d_zeros(&d0, 2*nb[0]+2*ng[0], 1);
	for(ii=0; ii<nb[0]; ii++)
		{
//		if(0) // input
		if(ii<nu[0]) // input
			{
			d0[ii]       = - 0.5; // umin
			d0[nb[0]+ii] =   0.5; // umax
			}
		else // state
			{
			d0[ii]       = - 4.0; // xmin
			d0[nb[0]+ii] =   4.0; // xmax
			}
//		idxb0[ii] = nu[0]+nx[0]/2+ii;
		idxb0[ii] = ii;
		}
	for(ii=0; ii<ng[0]; ii++)
		{
		if(ii<nu[0]) // input
			{
			d0[2*nb[0]+ii]       = - 0.5; // umin
			d0[2*nb[0]+ng[0]+ii] =   0.5; // umax
			}
		else // state
			{
			d0[2*nb[0]+ii]       = - 4.0; // xmin
			d0[2*nb[0]+ng[0]+ii] =   4.0; // xmax
			}
		}
#if PRINT
	int_print_mat(1, nb[0], idxb0, 1);
	d_print_mat(1, 2*nb[0]+2*ng[0], d0, 1);
#endif

	int *idxb1; int_zeros(&idxb1, nb[1], 1);
	double *d1; d_zeros(&d1, 2*nb[1]+2*ng[1], 1);
	for(ii=0; ii<nb[1]; ii++)
		{
//		if(0) // input
		if(ii<nu[1]) // input
			{
			d1[ii]       = - 0.5; // umin
			d1[nb[1]+ii] =   0.5; // umax
			}
		else // state
			{
			d1[ii]       = - 4.0; // xmin
			d1[nb[1]+ii] =   4.0; // xmax
			}
		idxb1[ii] = ii;
//		idxb1[ii] = nu[1]+nx[1]/2+ii;
		}
	for(ii=0; ii<ng[1]; ii++)
		{
		if(ii<nu[1]) // input
			{
			d1[2*nb[1]+ii]       = - 0.5; // umin
			d1[2*nb[1]+ng[1]+ii] =   0.5; // umax
			}
		else // state
			{
			d1[2*nb[1]+ii]       = - 4.0; // xmin
			d1[2*nb[1]+ng[1]+ii] =   4.0; // xmax
			}
		}
#if PRINT
	int_print_mat(1, nb[1], idxb1, 1);
	d_print_mat(1, 2*nb[1]+2*ng[1], d1, 1);
#endif

	int *idxbN; int_zeros(&idxbN, nb[N], 1);
	double *dN; d_zeros(&dN, 2*nb[N]+2*ng[N], 1);
	for(ii=0; ii<nb[N]; ii++)
		{
		dN[ii]       = - 4.0; // xmin
		dN[nb[N]+ii] =   4.0; // xmax
		idxbN[ii] = ii;
		}
	for(ii=0; ii<ng[N]; ii++)
		{
		dN[2*nb[N]+ii]       = - 4.0; // dmin
		dN[2*nb[N]+ng[N]+ii] =   4.0; // dmax
		}
#if PRINT
	int_print_mat(1, nb[N], idxbN, 1);
	d_print_mat(1, 2*nb[N]+2*ng[N], dN, 1);
#endif

	double *DC0; d_zeros(&DC0, ng[0], nu[0]+nx[0]);
	for(ii=0; ii<ng[0]; ii++)
		DC0[ii*(ng[0]+1)] = 1.0;
	double *DC1; d_zeros(&DC1, ng[1], nu[1]+nx[1]);
	for(ii=0; ii<ng[1]; ii++)
		DC1[ii*(ng[1]+1)] = 1.0;
	double *DCN; d_zeros(&DCN, ng[N], nx[N]);
	for(ii=0; ii<ng[N]; ii++)
		DCN[ii*(ng[N]+1)] = 1.0;

	struct d_strmat sDCt0;
	d_allocate_strmat(nu[0]+nx[0], ng[0], &sDCt0);
	d_cvt_tran_mat2strmat(ng[0], nu[0]+nx[0], DC0, ng[0], &sDCt0, 0, 0);
#if PRINT
	d_print_strmat(nu[0]+nx[0], ng[0], &sDCt0, 0, 0);
#endif
	struct d_strvec sd0;
	d_allocate_strvec(2*nb[0]+2*ng[0], &sd0);
	d_cvt_vec2strvec(2*nb[0]+2*ng[0], d0, &sd0, 0);
#if PRINT
	d_print_tran_strvec(2*nb[0]+2*ng[0], &sd0, 0);
#endif

	struct d_strmat sDCt1;
	d_allocate_strmat(nu[1]+nx[1], ng[1], &sDCt1);
	d_cvt_tran_mat2strmat(ng[1], nu[1]+nx[1], DC1, ng[1], &sDCt1, 0, 0);
#if PRINT
	d_print_strmat(nu[1]+nx[1], ng[1], &sDCt1, 0, 0);
#endif
	struct d_strvec sd1;
	d_allocate_strvec(2*nb[1]+2*ng[1], &sd1);
	d_cvt_vec2strvec(2*nb[1]+2*ng[1], d1, &sd1, 0);
#if PRINT
	d_print_tran_strvec(2*nb[1]+2*ng[1], &sd1, 0);
#endif

	struct d_strmat sDCtN;
	d_allocate_strmat(nx[N], ng[N], &sDCtN);
	d_cvt_tran_mat2strmat(ng[N], nx[N], DCN, ng[N], &sDCtN, 0, 0);
#if PRINT
	d_print_strmat(nx[N], ng[N], &sDCtN, 0, 0);
#endif
	struct d_strvec sdN;
	d_allocate_strvec(2*nb[N]+2*ng[N], &sdN);
	d_cvt_vec2strvec(2*nb[N]+2*ng[N], dN, &sdN, 0);
#if PRINT
	d_print_tran_strvec(2*nb[N]+2*ng[N], &sdN, 0);
#endif

/************************************************
* libstr ip2 solver
************************************************/	

	struct d_strmat hsBAbt[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsd[N+1];
	int *hidxb[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strvec hslam[N+1];
	struct d_strvec hst[N+1];

	hsBAbt[0] = sBAbt0;
	hsRSQrq[0] = sRSQrq0;
	hsDCt[0] = sDCt0;
	hsd[0] = sd0;
	hidxb[0] = idxb0;
	d_allocate_strvec(nu[0]+nx[0], &hsux[0]);
	d_allocate_strvec(nx[1], &hspi[1]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hslam[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hst[0]);
	for(ii=1; ii<N; ii++)
		{
		hsBAbt[ii] = sBAbt1;
		hsRSQrq[ii] = sRSQrq1;
		hsDCt[ii] = sDCt1;
		hsd[ii] = sd1;
		hidxb[ii] = idxb1;
		d_allocate_strvec(nu[ii]+nx[ii], &hsux[ii]);
		d_allocate_strvec(nx[ii+1], &hspi[ii+1]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hslam[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hst[ii]);
		}
	hsRSQrq[N] = sRSQrqN;
	hsDCt[N] = sDCtN;
	hsd[N] = sdN;
	hidxb[N] = idxbN;
	d_allocate_strvec(nu[N]+nx[N], &hsux[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hslam[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hst[N]);
	
	void *work_memory;
	v_zeros_align(&work_memory, d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));
	printf("\nwork space size (in bytes): %d\n", d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

	// IP options
	int kk = -1;
	int warm_start = 0;
	double stat[5*k_max];

	int hpmpc_exit;

	struct timeval tv0, tv1, tv2, tv3;
	double maxtime = 0.0;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{
		gettimeofday(&tv2, NULL); // start
		hpmpc_exit = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, 1, hspi, hslam, hst, work_memory);
		gettimeofday(&tv3, NULL); // end
		maxtime = fmax(maxtime,(tv3.tv_sec-tv2.tv_sec)+(tv3.tv_usec-tv2.tv_usec)/1e6);
		}

	gettimeofday(&tv1, NULL); // stop

	printf("\nstat = (%d iter)\n\nsigma\t\talpha1\t\tmu1\t\talpha2\t\tmu2\n\n", kk);
	d_print_e_tran_mat(5, kk, stat, 5);

#if PRINT
	printf("\nux =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi[ii], 0);

	printf("\nlam =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hslam[ii], 0);

	printf("\nt =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hst[ii], 0);
#endif

	double time_ipm = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

/************************************************
* libstr ip2 residuals
************************************************/	

	struct d_strvec hsb[N];
	struct d_strvec hsrq[N+1];
	struct d_strvec hsrrq[N+1];
	struct d_strvec hsrb[N];
	struct d_strvec hsrd[N+1];
	struct d_strvec hsrm[N+1];
	double mu;

	hsb[0] = sb0;
	hsrq[0] = srq0;
	d_allocate_strvec(nu[0]+nx[0], &hsrrq[0]);
	d_allocate_strvec(nx[1], &hsrb[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrd[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrm[0]);
	for(ii=1; ii<N; ii++)
		{
		hsb[ii] = sb1;
		hsrq[ii] = srq1;
		d_allocate_strvec(nu[ii]+nx[ii], &hsrrq[ii]);
		d_allocate_strvec(nx[ii+1], &hsrb[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii]);
		}
	hsrq[N] = srqN;
	d_allocate_strvec(nu[N]+nx[N], &hsrrq[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrd[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrm[N]);

	void *work_res;
	v_zeros_align(&work_res, d_res_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

	d_res_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hslam, hst, hsrrq, hsrb, hsrd, hsrm, &mu, work_res);

#if PRINT
	printf("\nres_rq\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(nu[ii]+nx[ii], &hsrrq[ii], 0);

	printf("\nres_b\n");
	for(ii=0; ii<N; ii++)
		d_print_e_tran_strvec(nx[ii+1], &hsrb[ii], 0);

	printf("\nres_d\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii], 0);

	printf("\nres_m\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii], 0);
#endif

	printf(" Average solution time over %d runs: %5.2e seconds (IPM)\n\n", nrep, time_ipm);
	printf(" Max time was %5.2e seconds\n", maxtime);

/************************************************
* libstr ip2 solver - resolve last kkt system for new rhs
************************************************/	
	
	// change constraints
//	dvecse_libstr(nb[0], -0.51, &sd0, 0);
//	dvecse_libstr(nb[0], 0.51, &sd0, nb[0]);

	struct d_strvec hsux2[N+1];
	struct d_strvec hspi2[N+1];
	struct d_strvec hslam2[N+1];
	struct d_strvec hst2[N+1];

	d_allocate_strvec(nu[0]+nx[0], &hsux2[0]);
	d_allocate_strvec(nx[1], &hspi2[1]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hslam2[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hst2[0]);
	for(ii=1; ii<N; ii++)
		{
		d_allocate_strvec(nu[ii]+nx[ii], &hsux2[ii]);
		d_allocate_strvec(nx[ii+1], &hspi2[ii+1]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hslam2[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hst2[ii]);
		}
	d_allocate_strvec(nu[N]+nx[N], &hsux2[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hslam2[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hst2[N]);
	
	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{

		d_kkt_solve_new_rhs_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsDCt, hsd, hsux2, 1, hspi2, hslam2, hst2, work_memory);

		}

	gettimeofday(&tv1, NULL); // stop

#if PRINT
	printf("\nux2 =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux2[ii], 0);

	printf("\npi2 =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi2[ii], 0);

	printf("\nlam2 =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hslam2[ii], 0);

	printf("\nt2 =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hst2[ii], 0);
#endif

	double time_ipm2 = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

/************************************************
* libstr ip2 residuals - resolve last kkt system for new rhs
************************************************/	

	d_res_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux2, hsDCt, hsd, hspi2, hslam2, hst2, hsrrq, hsrb, hsrd, hsrm, &mu, work_res);

#if PRINT
	printf("\nres_rq2\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(nu[ii]+nx[ii], &hsrrq[ii], 0);

	printf("\nres_b2\n");
	for(ii=0; ii<N; ii++)
		d_print_e_tran_strvec(nx[ii+1], &hsrb[ii], 0);

	printf("\nres_d2\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii], 0);

	printf("\nres_m2\n");
	for(ii=0; ii<=N; ii++)
		d_print_e_tran_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii], 0);
#endif

	printf(" Average solution time over %d runs: %5.2e seconds (IPM resolve last kkt for new rhs)\n\n", nrep, time_ipm2);


/************************************************
* high-level interface
************************************************/	
	
	// change (back) constraints
	dvecse_libstr(nb[0], -0.50, &sd0, 0);
	dvecse_libstr(nb[0], 0.50, &sd0, nb[0]);

	double *b0;
	d_zeros(&b0, nx_, 1);
	d_cvt_strvec2vec(nx_, &sb0, 0, b0);

	double *lb0;
	d_zeros(&lb0, nb[0], 1);
	d_cvt_strvec2vec(nb[0], &sd0, 0, lb0);

	double *ub0;
	d_zeros(&ub0, nb[0], 1);
	d_cvt_strvec2vec(nb[0], &sd0, nb[0], ub0);

	double *lg0;
	d_zeros(&lg0, ng[0], 1);
	d_cvt_strvec2vec(ng[0], &sd0, 2*nb[0]+0, lg0);

	double *ug0;
	d_zeros(&ug0, ng[0], 1);
	d_cvt_strvec2vec(ng[0], &sd0, 2*nb[0]+ng[0], ug0);

	double *lb1;
	d_zeros(&lb1, nb[1], 1);
	d_cvt_strvec2vec(nb[1], &sd1, 0, lb1);

	double *ub1;
	d_zeros(&ub1, nb[1], 1);
	d_cvt_strvec2vec(nb[1], &sd1, nb[1], ub1);

	double *lg1;
	d_zeros(&lg1, ng[1], 1);
	d_cvt_strvec2vec(ng[1], &sd1, 2*nb[1]+0, lg1);

	double *ug1;
	d_zeros(&ug1, ng[1], 1);
	d_cvt_strvec2vec(ng[1], &sd1, 2*nb[1]+ng[1], ug1);

	double *lbN;
	d_zeros(&lbN, nb[N], 1);
	d_cvt_strvec2vec(nb[N], &sdN, 0, lbN);

	double *ubN;
	d_zeros(&ubN, nb[N], 1);
	d_cvt_strvec2vec(nb[N], &sdN, nb[N], ubN);

	double *lgN;
	d_zeros(&lgN, ng[N], 1);
	d_cvt_strvec2vec(ng[N], &sdN, 2*nb[N]+0, lgN);

	double *ugN;
	d_zeros(&ugN, ng[N], 1);
	d_cvt_strvec2vec(ng[N], &sdN, 2*nb[N]+ng[N], ugN);

	double *D0;
	d_zeros(&D0, ng[0], nu[0]);
	d_cvt_tran_strmat2mat(nu[0], ng[0], &sDCt0, 0, 0, D0, ng[0]);

	double *C0;
	d_zeros(&C0, ng[0], nx[0]);
	d_cvt_tran_strmat2mat(nx[0], ng[0], &sDCt0, nu[0], 0, C0, ng[0]);

	double *D1;
	d_zeros(&D1, ng[1], nu[1]);
	d_cvt_tran_strmat2mat(nu[1], ng[1], &sDCt1, 0, 0, D1, ng[1]);

	double *C1;
	d_zeros(&C1, ng[1], nx[1]);
	d_cvt_tran_strmat2mat(nx[1], ng[1], &sDCt1, nu[1], 0, C1, ng[1]);

	double *CN;
	d_zeros(&CN, ng[N], nx[N]);
	d_cvt_tran_strmat2mat(nx[N], ng[N], &sDCtN, 0, 0, CN, ng[N]);

	double *hA[N];
	double *hB[N];
	double *hb[N];
	double *hR[N];
	double *hS[N];
	double *hQ[N+1];
	double *hr[N];
	double *hq[N+1];
	double *hlb[N+1];
	double *hub[N+1];
	double *hC[N+1];
	double *hD[N];
	double *hlg[N+1];
	double *hug[N+1];
	double *hx[N+1];
	double *hu[N];
	double *hpi[N];
	double *hlam[N+1];
	double inf_norm_res[5];

	ii = 0;
	hB[ii] = B;
	hb[ii] = b0;
	hC[ii] = C0;
	hD[ii] = D0;
	hR[ii] = R;
	hr[ii] = r; // XXX r0
	hlb[ii] = lb0;
	hub[ii] = ub0;
	hlg[ii] = lg0;
	hug[ii] = ug0;
	d_zeros(&hu[ii], nu[ii], 1);
	d_zeros(&hx[ii], nx[ii], 1);
	d_zeros(&hpi[ii], nx[ii+1], 1);
	d_zeros(&hlam[ii], 2*nb[ii]+2*ng[ii], 1);
	for(ii=1; ii<N; ii++)
		{
		hA[ii] = A;
		hB[ii] = B;
		hb[ii] = b;
		hC[ii] = C1;
		hD[ii] = D1;
		hR[ii] = R;
		hS[ii] = S;
		hQ[ii] = Q;
		hr[ii] = r;
		hq[ii] = q;
		hlb[ii] = lb1;
		hub[ii] = ub1;
		hlg[ii] = lg1;
		hug[ii] = ug1;
		d_zeros(&hu[ii], nu[ii], 1);
		d_zeros(&hx[ii], nx[ii], 1);
		d_zeros(&hpi[ii], nx[ii+1], 1);
		d_zeros(&hlam[ii], 2*nb[ii]+2*ng[ii], 1);
		}
	ii = N;
	hC[ii] = CN;
	hQ[ii] = Q;
	hq[ii] = q;
	hlb[ii] = lbN;
	hub[ii] = ubN;
	hlg[ii] = lgN;
	hug[ii] = ugN;
	d_zeros(&hx[ii], nx[ii], 1);
	d_zeros(&hlam[ii], 2*nb[ii]+2*ng[ii], 1);
	
	void *work_ipm_high;
	v_zeros(&work_ipm_high, hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(N, nx, nu, nb, hidxb, ng, N2));
	printf("\nwork space size 2 (in bytes): %d\n", hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(N, nx, nu, nb, hidxb, ng, N2));

	gettimeofday(&tv0, NULL); // start
	
	maxtime = 0.0;

	for(rep=0; rep<nrep; rep++)
	{
		gettimeofday(&tv2, NULL); // start
		hpmpc_exit = fortran_order_d_ip_ocp_hard_tv(&kk, k_max, mu0, mu_tol, N, nx, nu, nb, hidxb, ng, N2, warm_start, hA, hB, hb, hQ, hS, hR, hq, hr, hlb, hub, hC, hD, hlg, hug, hx, hu, hpi, hlam, inf_norm_res, work_ipm_high, stat);
		gettimeofday(&tv3, NULL); // end
		maxtime = fmax(maxtime,(tv3.tv_sec-tv2.tv_sec)+(tv3.tv_usec-tv2.tv_usec)/1e6);
	}

	gettimeofday(&tv1, NULL); // stop

	printf("\nstat = (%d iter)\n\nsigma\t\talpha1\t\tmu1\t\talpha2\t\tmu2\n\n", kk);
	d_print_e_tran_mat(5, kk, stat, 5);
#if PRINT

	printf("\nu = \n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nu[ii], hu[ii], 1);

	printf("\nx = \n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx[ii], hx[ii], 1);
#endif
	
	printf("\ninf norm res = %e, %e, %e, %e, %e\n\n", inf_norm_res[0], inf_norm_res[1], inf_norm_res[2], inf_norm_res[3], inf_norm_res[4]); 

	double time_ipm_high = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf(" Average solution time over %d runs: %5.2e seconds (IPM high)\n\n", nrep, time_ipm_high);
	printf(" Max time was %5.2e seconds\n", maxtime);
	
	printf("Exit code: %d\n", hpmpc_exit);
	
	switch(hpmpc_exit)
	{
		case 0: 
			printf("Successful solution!\n");
			break;
		case 1: 
			printf("Max number of iterations reached\n");
			break;
		case 2: 
			printf("No improvement...\n");
			break;
		case -1: 
			printf("Infeasible problem!\n");
			break;
	}

/************************************************
* high-level interface (resolve last kkt new rhs)
************************************************/	
	
	// change constraints
//	for(ii=0; ii<nb[0]; ii++) lb0[ii] = -0.51;
//	for(ii=0; ii<nb[0]; ii++) ub0[ii] =  0.51;

	// zero solution
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]; jj++)
			hu[ii][jj] = 0.0;
	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx[ii]; jj++)
			hx[ii][jj] = 0.0;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
	{

		fortran_order_d_ip_last_kkt_new_rhs_ocp_hard_libstr(N, nx, nu, nb, hidxb, ng, N2, hb, hq, hr, hlb, hub, hlg, hug, hx, hu, hpi, hlam, inf_norm_res, work_ipm_high);

	}

	gettimeofday(&tv1, NULL); // stop

#if PRINT
	printf("\nu = \n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nu[ii], hu[ii], 1);

	printf("\nx = \n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx[ii], hx[ii], 1);
#endif
	
	printf("\ninf norm res = %e, %e, %e, %e, %e\n\n", inf_norm_res[0], inf_norm_res[1], inf_norm_res[2], inf_norm_res[3], inf_norm_res[4]); 

	double time_ipm_high_kkt = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf(" Average solution time over %d runs: %5.2e seconds (IPM high kkt last rhs)\n\n", nrep, time_ipm_high_kkt);

/************************************************
* free memory
************************************************/	

	d_free(A);
	d_free(B);
	d_free(b);
	d_free(x0);
	int_free(idxb0);
	d_free(d0);
	int_free(idxb1);
	d_free(d1);
	int_free(idxbN);
	d_free(dN);

	d_free_strmat(&sA);
	d_free_strvec(&sx0);
	d_free_strmat(&sBAbt0);
	d_free_strvec(&sb0);
	d_free_strmat(&sRSQrq0);
	d_free_strvec(&srq0);
	d_free_strmat(&sRSQrqN);
	d_free_strvec(&srqN);
	d_free_strmat(&sDCt0);
	d_free_strvec(&sd0);
	d_free_strmat(&sDCt1);
	d_free_strvec(&sd1);
	d_free_strmat(&sDCtN);
	d_free_strvec(&sdN);
	if(N>1)
		{
		d_free_strmat(&sBAbt1);
		d_free_strvec(&sb1);
		d_free_strmat(&sRSQrq1);
		d_free_strvec(&srq1);
		}
	d_free_strvec(&hsux[0]);
	d_free_strvec(&hspi[1]);
	d_free_strvec(&hslam[0]);
	d_free_strvec(&hst[0]);
	d_free_strvec(&hsrrq[0]);
	d_free_strvec(&hsrb[0]);
	for(ii=1; ii<N; ii++)
		{
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii+1]);
		d_free_strvec(&hslam[ii]);
		d_free_strvec(&hst[ii]);
		d_free_strvec(&hsrrq[ii]);
		d_free_strvec(&hsrb[ii]);
		}
	d_free_strvec(&hsux[N]);
	d_free_strvec(&hslam[N]);
	d_free_strvec(&hst[N]);
	d_free_strvec(&hsrrq[N]);

/************************************************
* return
************************************************/	

	return 0;
	}

