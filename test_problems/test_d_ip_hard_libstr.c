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
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>
#endif

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/blas_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_aux.h"
#include "../include/mpc_solvers.h"
#include "../problem_size.h"
#include "../include/block_size.h"
#include "tools.h"
#include "test_param.h"
#include "../include/c_interface.h"


#define USE_IPM_RES 1
#define KEEP_X0 0

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

	int nx_ = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu_ = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght
	int nb_  = nu_+nx_/2; // number of box constrained inputs and states
	int ng_  = 0; //nx; //4;  // number of general constraints
	int ngN = nx_/2; //nx; // number of general constraints at the last stage

	// partial condensing horizon
	int N2 = N; //N/2;


	int nbu = nb_<nu_ ? nb_ : nu_;
	int nbx = nb_-nu_<nx_ ? nb_-nu_ : nx_;
	nbx = nbx<0 ? 0 : nbx;

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

	int nb[N+1];
#if KEEP_X0
	nb[0] = nb_;
#else
	nb[0] = nbu;
#endif
	for(ii=1; ii<N; ii++)
		nb[ii] = nb_;
	nb[N] = nbx;

	int ng[N+1];
	for(ii=0; ii<N; ii++)
		ng[ii] = ng_;
	ng[N] = ngN;
	


	printf(" Test problem: mass-spring system with %d masses and %d controls.\n", nx_/2, nu_);
	printf("\n");
	printf(" MPC problem size: %d states, %d inputs, %d horizon length, %d two-sided box constraints, %d two-sided general constraints.\n", nx_, nu_, N, nb_, ng_);
	printf("\n");
	printf(" IP method parameters: predictor-corrector IP, double precision, %d maximum iterations, %5.1e exit tolerance in duality measure (edit file test_param.c to change them).\n", K_MAX, MU_TOL);



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
		b[jj] = 0.1;
	
	for(jj=0; jj<nx_; jj++)
		x0[jj] = 0;
	x0[0] = 2.5;
	x0[1] = 2.5;

	d_print_mat(nx_, nx_, A, nx_);
	d_print_mat(nx_, nu_, B, nu_);
	d_print_mat(1, nx_, b, 1);
	d_print_mat(1, nx_, x0, 1);

	struct d_strmat sA;
	d_allocate_strmat(nx_, nx_, &sA);
	d_cvt_mat2strmat(nx_, nx_, A, nx_, &sA, 0, 0);
	d_print_strmat(nx_, nx_, &sA, 0, 0);

	struct d_strvec sx0;
	d_allocate_strvec(nx_, &sx0);
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
	d_print_tran_strvec(nx_, &sx0, 0);

	struct d_strvec sb0;
	d_allocate_strvec(nx_, &sb0);
	d_cvt_vec2strvec(nx_, b, &sb0, 0);
	d_print_tran_strvec(nx_, &sb0, 0);
#if ! KEEP_X0
	dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb0, 0, &sb0, 0);
#endif
	d_print_tran_strvec(nx_, &sb0, 0);

	struct d_strmat sBAbt0;
	d_allocate_strmat(nu[0]+nx[0]+1, nx[1], &sBAbt0);
	d_cvt_tran_mat2strmat(nx[1], nu[0], B, nx_, &sBAbt0, 0, 0);
	d_cvt_tran_mat2strmat(nx[1], nx[0], A, nx_, &sBAbt0, nu[0], 0);
	drowin_libstr(nx[1], 1.0, &sb0, 0, &sBAbt0, nu[0]+nx[0], 0);
	d_print_strmat(nu[0]+nx[0]+1, nx[1], &sBAbt0, 0, 0);

	struct d_strmat sBAbt1;
	struct d_strvec sb1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1);
		d_cvt_tran_mat2strmat(nx[2], nu[1], B, nx_, &sBAbt1, 0, 0);
		d_cvt_tran_mat2strmat(nx[2], nx[1], A, nx_, &sBAbt1, nu[1], 0);
		d_cvt_tran_mat2strmat(nx[2], 1, b, nx_, &sBAbt1, nu[1]+nx[1], 0);
		d_print_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1, 0, 0);
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
	d_print_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0, 0, 0);
	d_allocate_strvec(nu[0]+nx[0], &srq0);
	d_cvt_vec2strvec(nu[0], r, &srq0, 0);
	d_cvt_vec2strvec(nx[0], q, &srq0, nu[0]);
	d_print_tran_strvec(nu[0]+nx[0], &srq0, 0);

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
		d_print_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1, 0, 0);
		d_allocate_strvec(nu[1]+nx[1], &srq1);
		d_cvt_vec2strvec(nu[1], r, &srq1, 0);
		d_cvt_vec2strvec(nx[1], q, &srq1, nu[1]);
		d_print_tran_strvec(nu[1]+nx[1], &srq1, 0);
		}

	struct d_strmat sRSQrqN;
	struct d_strvec srqN;
	d_allocate_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN);
	d_cvt_mat2strmat(nu[N], nu[N], R, nu_, &sRSQrqN, 0, 0);
	d_cvt_tran_mat2strmat(nu[N], nx[N], S, nu_, &sRSQrqN, nu[N], 0);
	d_cvt_mat2strmat(nx[N], nx[N], Q, nx_, &sRSQrqN, nu[N], nu[N]);
	d_cvt_tran_mat2strmat(nu[N], 1, r, nu_, &sRSQrqN, nu[N]+nx[N], 0);
	d_cvt_tran_mat2strmat(nx[N], 1, q, nx_, &sRSQrqN, nu[N]+nx[N], nu[N]);
	d_print_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN, 0, 0);
	d_allocate_strvec(nu[N]+nx[N], &srqN);
	d_cvt_vec2strvec(nu[N], r, &srqN, 0);
	d_cvt_vec2strvec(nx[N], q, &srqN, nu[N]);
	d_print_tran_strvec(nu[N]+nx[N], &srqN, 0);

	// maximum element in cost functions
	double mu0 = 2.0;

/************************************************
* box & general constraints
************************************************/	

	int *idxb0; int_zeros(&idxb0, nb[0], 1);
	double *d0; d_zeros(&d0, 2*nb[0]+2*ng[0], 1);
	for(ii=0; ii<nb[0]; ii++)
		{
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
		idxb0[ii] = ii;
		}
	for(ii=0; ii<ng[0]; ii++)
		{
		d0[2*nb[0]+ii]       = - 100.0; // dmin
		d0[2*nb[0]+ng[0]+ii] =   100.0; // dmax
		}
	int_print_mat(1, nb[0], idxb0, 1);
	d_print_mat(1, 2*nb[0]+2*ng[0], d0, 1);

	int *idxb1; int_zeros(&idxb1, nb[1], 1);
	double *d1; d_zeros(&d1, 2*nb[1]+2*ng[1], 1);
	for(ii=0; ii<nb[1]; ii++)
		{
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
		}
	for(ii=0; ii<ng[1]; ii++)
		{
		d1[2*nb[1]+ii]       = - 100.0; // dmin
		d1[2*nb[1]+ng[1]+ii] =   100.0; // dmax
		}
	int_print_mat(1, nb[1], idxb1, 1);
	d_print_mat(1, 2*nb[1]+2*ng[1], d1, 1);

	int *idxbN; int_zeros(&idxbN, nb[N], 1);
	double *dN; d_zeros(&dN, 2*nb[N]+2*ng[N], 1);
	for(ii=0; ii<nb[N]; ii++)
		{
		if(ii<nu[N]) // input
			{
			dN[ii]       = - 0.5; // umin
			dN[nb[N]+ii] =   0.5; // umax
			}
		else // state
			{
			dN[ii]       = - 4.0; // xmin
			dN[nb[N]+ii] =   4.0; // xmax
			}
		idxbN[ii] = ii;
		}
	for(ii=0; ii<ng[N]; ii++)
		{
		dN[2*nb[N]+ii]       = - 0.0; // dmin
		dN[2*nb[N]+ng[N]+ii] =   0.0; // dmax
		}
	int_print_mat(1, nb[N], idxbN, 1);
	d_print_mat(1, 2*nb[N]+2*ng[N], dN, 1);

	double *C; d_zeros(&C, ng_, nx_);
	for(ii=0; ii<ng_; ii++)
		C[ii*(ng_+1)] = 1.0;
	double *D; d_zeros(&D, ng_, nu_);
	double *CN; d_zeros(&CN, ngN, nx_);
	for(ii=0; ii<ngN; ii++)
		CN[ii*(ngN+1)] = 1.0;

	struct d_strmat sDCt0;
	d_allocate_strmat(nu[0]+nx[0], ng[0], &sDCt0);
	d_cvt_tran_mat2strmat(ng[0], nu[0], D, ng_, &sDCt0, 0, 0);
	d_cvt_tran_mat2strmat(ng[0], nx[0], C, ng_, &sDCt0, nu[0], 0);
	d_print_strmat(nu[0]+nx[0], ng[0], &sDCt0, 0, 0);
	struct d_strvec sd0;
	d_allocate_strvec(2*nb[0]+2*ng[0], &sd0);
	d_cvt_vec2strvec(2*nb[0]+2*ng[0], d0, &sd0, 0);
	d_print_tran_strvec(2*nb[0]+2*ng[0], &sd0, 0);

	struct d_strmat sDCt1;
	d_allocate_strmat(nu[1]+nx[1], ng[1], &sDCt1);
	d_cvt_tran_mat2strmat(ng[1], nu[1], D, ng_, &sDCt1, 0, 0);
	d_cvt_tran_mat2strmat(ng[1], nx[1], C, ng_, &sDCt1, nu[1], 0);
	d_print_strmat(nu[1]+nx[1], ng[1], &sDCt1, 0, 0);
	struct d_strvec sd1;
	d_allocate_strvec(2*nb[1]+2*ng[1], &sd1);
	d_cvt_vec2strvec(2*nb[1]+2*ng[1], d1, &sd1, 0);
	d_print_tran_strvec(2*nb[1]+2*ng[1], &sd1, 0);

	struct d_strmat sDCtN;
	d_allocate_strmat(nx[N], ng[N], &sDCtN);
	d_cvt_tran_mat2strmat(ng[N], nx[N], CN, ngN, &sDCtN, 0, 0);
	d_print_strmat(nx[N], ng[N], &sDCtN, 0, 0);
	struct d_strvec sdN;
	d_allocate_strvec(2*nb[N]+2*ng[N], &sdN);
	d_cvt_vec2strvec(2*nb[N]+2*ng[N], dN, &sdN, 0);
	d_print_tran_strvec(2*nb[N]+2*ng[N], &sdN, 0);

/************************************************
* libstr ip2 solver
************************************************/	

	struct d_strmat hsBAbt[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsd[N+1];
	int *hidxb[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N];
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
	int k_max = 10;
	double mu_tol = 1e-12;
	double alpha_min = 1e-8;
	int warm_start = 0;
	double stat[5*k_max];

	int hpmpc_exit;

	struct timeval tv0, tv1;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{

		hpmpc_exit = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, 1, hspi, hslam, hst, work_memory);

		}

	gettimeofday(&tv1, NULL); // stop

	printf("\nstat =\n\nsigma\t\talpha1\t\tmu1\t\talpha2\t\tmu2\n\n");
	d_print_e_tran_mat(5, kk, stat, 5);

	printf("\nux =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi[ii], 0);

	printf("\nt =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hst[ii], 0);

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

	printf(" Average solution time over %d runs: %5.2e seconds (IPM)\n", nrep, time_ipm);

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
#if 0


/************************************************
* high level interface work space
************************************************/	

#if 0
	double *rA; d_zeros(&rA, nx, N*nx);
	d_rep_mat(N, nx, nx, A, nx, rA, nx);

	double *rB; d_zeros(&rB, nx, N*nu);
	d_rep_mat(N, nx, nu, B, nx, rB, nx);

	double *rC; d_zeros(&rC, ng, (N+1)*nx);
	d_rep_mat(N, ng, nx, C, ng, rC+nx*ng, ng);

	double *CN = DCN;

	double *rD; d_zeros(&rD, ng, N*nu);
	d_rep_mat(N, ng, nu, D, ng, rD, ng);

	double *rb; d_zeros(&rb, nx, N*1);
	d_rep_mat(N, nx, 1, b, nx, rb, nx);

	double *rQ; d_zeros(&rQ, nx, N*nx);
	d_rep_mat(N, nx, nx, Q, nx, rQ, nx);

	double *rQf; d_zeros(&rQf, nx, nx);
	d_copy_mat(nx, nx, Q, nx, rQf, nx);

	double *rS; d_zeros(&rS, nu, N*nx);
	d_rep_mat(N, nu, nx, S, nu, rS, nu);

	double *rR; d_zeros(&rR, nu, N*nu);
	d_rep_mat(N, nu, nu, R, nu, rR, nu);

	double *rq; d_zeros(&rq, nx, N);
	d_rep_mat(N, nx, 1, q, nx, rq, nx);

	double *rqf; d_zeros(&rqf, nx, 1);
	d_copy_mat(nx, 1, q, nx, rqf, nx);

	double *rr; d_zeros(&rr, nu, N);
	d_rep_mat(N, nu, 1, r, nu, rr, nu);

	double *lb; d_zeros(&lb, nb, 1);
	for(ii=0; ii<nb; ii++)
		lb[ii] = d1[ii];
	double *rlb; d_zeros(&rlb, nb, N+1);
	d_rep_mat(N+1, nb, 1, lb, nb, rlb, nb);
//	d_print_mat(nb, N+1, rlb, nb);

	double *lg; d_zeros(&lg, ng, 1);
	for(ii=0; ii<ng; ii++)
		lg[ii] = d1[2*nb_v[1]+ii];
	double *rlg; d_zeros(&rlg, ng, N);
	d_rep_mat(N, ng, 1, lg, ng, rlg, ng);
//	d_print_mat(ng, N, rlg, ng);

	double *lgN; d_zeros(&lgN, ngN, 1);
	for(ii=0; ii<ngN; ii++)
		lgN[ii] = dN[2*nb_v[N]+ii];
//	d_print_mat(ngN, 1, lgN, ngN);

	double *ub; d_zeros(&ub, nb, 1);
	for(ii=0; ii<nb; ii++)
		ub[ii] = d1[nb_v[1]+ii];
	double *rub; d_zeros(&rub, nb, N+1);
	d_rep_mat(N+1, nb, 1, ub, nb, rub, nb);
//	d_print_mat(nb, N+1, rub, nb);

	double *ug; d_zeros(&ug, ng, 1);
	for(ii=0; ii<ng; ii++)
		ug[ii] = d1[2*nb_v[1]+ng_v[1]+ii];
	double *rug; d_zeros(&rug, ng, N);
	d_rep_mat(N, ng, 1, ug, ng, rug, ng);
//	d_print_mat(ng, N, rug, ng);

	double *ugN; d_zeros(&ugN, ngN, 1);
	for(ii=0; ii<ngN; ii++)
		ugN[ii] = dN[2*nb_v[N]+ng_v[N]+ii];
//	d_print_mat(ngN, 1, ugN, ngN);

	double *rx; d_zeros(&rx, nx, N+1);
	d_copy_mat(nx, 1, x0, nx, rx, nx);

	double *ru; d_zeros(&ru, nu, N);

	double *rpi; d_zeros(&rpi, nx, N);

	double *rlam; d_zeros(&rlam, N*2*(nb+ng)+2*(nb+ngN), 1);

	double *rt; d_zeros(&rt, N*2*(nb+ng)+2*(nb+ngN), 1);

	double *rwork = (double *) malloc(hpmpc_d_ip_mpc_hard_tv_work_space_size_bytes(N, nx, nu, nb, ng, ngN));

	double inf_norm_res[4] = {}; // infinity norm of residuals: rq, rb, rd, mu
#endif

/************************************************
* low level interface work space
************************************************/	

	double *hpBAbt[N+1];
	double *hpDCt[N+1];
	double *hb[N+1];
	double *hpRSQrq[N+1];
	double *hrq[N+1];
	double *hd[N+1];
	int *hidxb[N+1];
	double *hux[N+1];
	double *hpi[N+1];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrb[N+1];
	double *hrrq[N+1];
	double *hrd[N+1];
	double *hrm[N+1];
	hpBAbt[1] = pBAbt0;
	hpDCt[0] = pDCt0;
	hb[1] = b0;
	hpRSQrq[0] = pRSQ0;
	hrq[0] = rq0;
	hd[0] = d0;
	hidxb[0] = idxb0;
	d_zeros_align(&hux[0], pnux_v[0], 1);
	d_zeros_align(&hpi[1], pnx_v[1], 1);
	d_zeros_align(&hlam[0], 2*nb_v[0]+2*ng_v[0], 1);
	d_zeros_align(&ht[0], 2*nb_v[0]+2*ng_v[0], 1);
	d_zeros_align(&hrb[1], pnx_v[1], 1);
	d_zeros_align(&hrrq[0], pnz_v[0], 1);
	d_zeros_align(&hrd[0], 2*nb_v[0]+2*ng_v[0], 1);
	d_zeros_align(&hrm[0], 2*nb_v[0]+2*ng_v[0], 1);
	for(ii=1; ii<N; ii++)
		{
		hpBAbt[ii+1] = pBAbt1;
//		d_zeros_align(&hpBAbt[ii], pnz_v[ii], cnx_v[ii+1]); for(jj=0; jj<pnz_v[ii]*cnx_v[ii+1]; jj++) hpBAbt[ii][jj] = pBAbt1[jj];
		hpDCt[ii] = pDCt1;
		hb[ii+1] = b;
		hpRSQrq[ii] = pRSQ1;
//		d_zeros_align(&hpRSQrq[ii], pnz_v[ii], cnux_v[ii]); for(jj=0; jj<pnz_v[ii]*cnux_v[ii]; jj++) hpRSQrq[ii][jj] = pRSQ1[jj];
		hrq[ii] = rq1;
		hd[ii] = d1;
		hidxb[ii] = idxb1;
		d_zeros_align(&hux[ii], pnux_v[ii], 1);
		d_zeros_align(&hpi[ii+1], pnx_v[ii+1], 1);
		d_zeros_align(&hlam[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		d_zeros_align(&ht[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		d_zeros_align(&hrb[ii+1], pnx_v[ii+1], 1);
		d_zeros_align(&hrrq[ii], pnz_v[ii], 1);
		d_zeros_align(&hrd[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		d_zeros_align(&hrm[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		}
	hpDCt[N] = pDCtN;
	hpRSQrq[N] = pRSQN;
	hrq[N] = rqN;
	hd[N] = dN;
	hidxb[N] = idxbN;
	d_zeros_align(&hux[N], pnx, 1);
	d_zeros_align(&hlam[N], 2*nb_v[N]+2*ng_v[N], 1);
	d_zeros_align(&ht[N], 2*nb_v[N]+2*ng_v[N], 1);
	d_zeros_align(&hrrq[N], pnz_v[N], 1);
	d_zeros_align(&hrd[N], 2*nb_v[N]+2*ng_v[N], 1);
	d_zeros_align(&hrm[N], 2*nb_v[N]+2*ng_v[N], 1);

//	hpDCt[M] = pDCtM;
//	hd[M] = dM;

	double mu = 0.0;

#if USE_IPM_RES
	double *work; d_zeros_align(&work, d_ip2_res_mpc_hard_tv_work_space_size_bytes(N, nx_v, nu_v, nb_v, ng_v)/sizeof(double), 1);
#else
	double *work; d_zeros_align(&work, d_ip2_mpc_hard_tv_work_space_size_bytes(N, nx_v, nu_v, nb_v, ng_v)/sizeof(double), 1);
#endif

/************************************************
* (new) high level interface work space
************************************************/	

	// box constraints
	double *lb0; d_zeros(&lb0, nb_v[0], 1);
	for(ii=0; ii<nb_v[0]; ii++)
		lb0[ii] = d0[ii];
	double *ub0; d_zeros(&ub0, nb_v[0], 1);
	for(ii=0; ii<nb_v[0]; ii++)
		ub0[ii] = d0[nb_v[0]+ii];
	double *lb1; d_zeros(&lb1, nb_v[1], 1);
	for(ii=0; ii<nb_v[1]; ii++)
		lb1[ii] = d1[ii];
	double *ub1; d_zeros(&ub1, nb_v[1], 1);
	for(ii=0; ii<nb_v[1]; ii++)
		ub1[ii] = d1[nb_v[1]+ii];
	double *lbN; d_zeros(&lbN, nb_v[N], 1);
	for(ii=0; ii<nb_v[N]; ii++)
		lbN[ii] = dN[ii];
	double *ubN; d_zeros(&ubN, nb_v[N], 1);
	for(ii=0; ii<nb_v[N]; ii++)
		ubN[ii] = dN[nb_v[N]+ii];

	// general constraints
	double *lg0; d_zeros(&lg0, ng_v[0], 1);
	for(ii=0; ii<ng_v[0]; ii++)
		lg0[ii] = d0[2*nb_v[0]+ii];
	double *ug0; d_zeros(&ug0, ng_v[0], 1);
	for(ii=0; ii<ng_v[0]; ii++)
		ug0[ii] = d0[2*nb_v[0]+ng_v[0]+ii];
	double *lg1; d_zeros(&lg1, ng_v[1], 1);
	for(ii=0; ii<ng_v[1]; ii++)
		lg1[ii] = d1[2*nb_v[1]+ii];
	double *ug1; d_zeros(&ug1, ng_v[1], 1);
	for(ii=0; ii<ng_v[1]; ii++)
		ug1[ii] = d1[2*nb_v[1]+ng_v[1]+ii];
	double *lgN; d_zeros(&lgN, ng_v[N], 1);
	for(ii=0; ii<ng_v[N]; ii++)
		lgN[ii] = dN[2*nb_v[N]+ii];
	double *ugN; d_zeros(&ugN, ng_v[N], 1);
	for(ii=0; ii<ng_v[N]; ii++)
		ugN[ii] = dN[2*nb_v[N]+ng_v[N]+ii];

	// data matrices
	double *hA[N];
	double *hB[N];
	double *hbb[N];
	double *hC[N+1];
	double *hD[N];
	double *hQ[N+1];
	double *hS[N];
	double *hR[N];
	double *hq[N+1];
	double *hr[N];
	double *hlb[N+1];
	double *hub[N+1];
	double *hlg[N+1];
	double *hug[N+1];
	double *hx[N+1];
	double *hu[N];
	double *hpi1[N];
	double *hlam1[N+1];
	double *ht1[N+1];
	double inf_norm_res[4] = {}; // infinity norm of residuals: rq, rb, rd, mu

	ii = 0;
	hA[0] = A;
	hB[0] = B;
	hbb[0] = b;
	hC[0] = C;
	hD[0] = D;
	hQ[0] = Q;
	hS[0] = S;
	hR[0] = R;
	hq[0] = q;
	hr[0] = r;
	hlb[0] = lb0;
	hub[0] = ub0;
	hlg[0] = lg0;
	hug[0] = ug0;
	d_zeros(&hx[0], nx_v[0], 1);
	d_zeros(&hu[0], nu_v[0], 1);
	d_zeros(&hpi1[0], nx_v[1], 1);
	d_zeros(&hlam1[0], 2*nb_v[0]+2*ng_v[0], 1);
	d_zeros(&ht1[0], 2*nb_v[0]+2*ng_v[0], 1);
	for(ii=1; ii<N; ii++)
		{
		hA[ii] = A;
		hB[ii] = B;
		hbb[ii] = b;
		hC[ii] = C;
		hD[ii] = D;
		hQ[ii] = Q;
		hS[ii] = S;
		hR[ii] = R;
		hq[ii] = q;
		hr[ii] = r;
		hlb[ii] = lb1;
		hub[ii] = ub1;
		hlg[ii] = lg1;
		hug[ii] = ug1;
		d_zeros(&hx[ii], nx_v[ii], 1);
		d_zeros(&hu[ii], nu_v[ii], 1);
		d_zeros(&hpi1[ii], nx_v[ii+1], 1);
		d_zeros(&hlam1[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		d_zeros(&ht1[ii], 2*nb_v[ii]+2*ng_v[ii], 1);
		}
	ii = N;
	hC[N] = C;
	hQ[N] = Q;
	hq[N] = q;
	hlb[N] = lbN;
	hub[N] = ubN;
	hlg[N] = lgN;
	hug[N] = ugN;
	d_zeros(&hx[N], nx_v[N], 1);
	d_zeros(&hlam1[N], 2*nb_v[N]+2*ng_v[N], 1);
	d_zeros(&ht1[N], 2*nb_v[N]+2*ng_v[N], 1);

	// work space
#if 0
	printf("work space in bytes: %d\n", hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(N, nx_v, nu_v, nb_v, ng_v));
	exit(3);
#endif
	void *work1 = malloc(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(N, nx_v, nu_v, nb_v, hidxb, ng_v, N2));

/************************************************
* solvers common stuff
************************************************/	

	int hpmpc_status;
	int kk, kk_avg;
	int k_max = 10;
	double mu_tol = 1e-20;
	double alpha_min = 1e-8;
	int warm_start = 0; // read initial guess from x and u
	double *stat; d_zeros(&stat, k_max, 5);
	int compute_res = 1;
	int compute_mult = 1;

	struct timeval tv0, tv1, tv2, tv3;
	double time;

	double **dummy;

/************************************************
* call the solver (high-level interface)
************************************************/	

#if 1
	int time_invariant = 0; // assume the problem to be time invariant
	int free_x0 = 0; // assume x0 as optimization variable

	gettimeofday(&tv0, NULL); // stop

	kk_avg = 0;

	for(rep=0; rep<nrep; rep++)
		{

//		hpmpc_status = fortran_order_d_ip_mpc_hard_tv(&kk, k_max, mu0, mu_tol, N, nx, nu, nb, ng, ngN, time_invariant, free_x0, warm_start, rA, rB, rb, rQ, rQf, rS, rR, rq, rqf, rr, rlb, rub, rC, rD, rlg, rug, CN, lgN, ugN, rx, ru, rpi, rlam, rt, inf_norm_res, rwork, stat);
		hpmpc_status = fortran_order_d_ip_ocp_hard_tv(&kk, k_max, mu0, mu_tol, N, nx_v, nu_v, nb_v, hidxb, ng_v, N2, warm_start, hA, hB, hbb, hQ, hS, hR, hq, hr, hlb, hub, hC, hD, hlg, hug, hx, hu, hpi1, hlam1, /*ht1,*/ inf_norm_res, work1, stat);

		kk_avg += kk;

		}
	
	gettimeofday(&tv1, NULL); // stop

	printf("\nsolution from high-level interface\n\n");
//	d_print_mat(nx, N+1, rx, nx);
//	d_print_mat(nu, N, ru, nu);
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx_v[ii], hx[ii], 1);
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nu_v[ii], hu[ii], 1);

	printf("\ninfinity norm of residuals\n\n");
	d_print_mat_e(1, 4, inf_norm_res, 1);

	time = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\n");
	printf(" Average number of iterations over %d runs: %5.1f\n", nrep, kk_avg / (double) nrep);
	printf(" Average solution time over %d runs: %5.2e seconds\n", nrep, time);
	printf("\n\n");

	gettimeofday(&tv0, NULL); // stop

	kk_avg = 0;

	for(rep=0; rep<nrep; rep++)
		{

//		fortran_order_d_solve_kkt_new_rhs_mpc_hard_tv(N, nx, nu, nb, ng, ngN, time_invariant, free_x0, rA, rB, rb, rQ, rQf, rS, rR, rq, rqf, rr, rlb, rub, rC, rD, rlg, rug, CN, lgN, ugN, rx, ru, rpi, rlam, rt, inf_norm_res, rwork);
		fortran_order_d_solve_kkt_new_rhs_ocp_hard_tv(N, nx_v, nu_v, nb_v, hidxb, ng_v, hA, hB, hbb, hQ, hS, hR, hq, hr, hlb, hub, hC, hD, hlg, hug, hx, hu, hpi1, hlam1, /*ht1,*/ inf_norm_res, work1);

		kk_avg += kk;

		}
	
	gettimeofday(&tv1, NULL); // stop

	printf("\nsolution from high-level interface (resolve final kkt)\n\n");
//	d_print_mat(nx, N+1, rx, nx);
//	d_print_mat(nu, N, ru, nu);
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nx_v[ii], hx[ii], 1);
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nu_v[ii], hu[ii], 1);

	printf("\ninfinity norm of residuals\n\n");
	d_print_mat_e(1, 4, inf_norm_res, 1);

	time = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf(" Average solution time over %d runs: %5.2e seconds\n", nrep, time);
#endif

/************************************************
* call the solver (low-level interface)
************************************************/	

//	for(ii=0; ii<N; ii++)
//		d_print_pmat(nu_v[ii]+nx_v[ii]+1, nx_v[ii+1], bs, hpBAbt[ii], cnx_v[ii+1]);
//	exit(3);

	gettimeofday(&tv0, NULL); // stop

	kk_avg = 0;

	printf("\nsolution...\n");
	for(rep=0; rep<nrep; rep++)
		{

#if USE_IPM_RES
//		hpmpc_status = d_ip2_res_mpc_hard_tv(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx_v, nu_v, nb_v, hidxb, ng_v, hpBAbt, hpRSQrq, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
		hpmpc_status = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx_v, nu_v, nb_v, hidxb, ng_v, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, compute_mult, hspi, hslam, hst, work);
#else
		hpmpc_status = d_ip2_mpc_hard_tv(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx_v, nu_v, nb_v, hidxb, ng_v, hpBAbt, hpRSQrq, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
#endif
		
		kk_avg += kk;

		}
	printf("\ndone\n");

	gettimeofday(&tv1, NULL); // stop

	printf("\nsolution from low-level interface (original problem)\n\n");
	printf("\nux\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nu_v[ii]+nx_v[ii], hux[ii], 1);
	printf("\npi\n\n");
	for(ii=1; ii<=N; ii++)
		d_print_mat(1, nx_v[ii], hpi[ii], 1);
//	printf("\nux\n\n");
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, 2*nb_v[ii]+2*ng_v[ii], hlam[ii], 1);
//	printf("\nux\n\n");
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, 2*nb_v[ii]+2*ng_v[ii], ht[ii], 1);
	
	// residuals
	if(compute_res)
		{

		int pnzM = (nx+bs-1)/bs*bs;

		struct d_strmat hsBAbt[N+1];
		struct d_strvec hsb[N+1];
		struct d_strmat hsRSQrq[N+1];
		struct d_strvec hsrq[N+1];
		struct d_strmat hsDCt[N+1];
		struct d_strvec hsd[N+1];
		struct d_strvec hsux[N+1];
		struct d_strvec hspi[N+1];
		struct d_strvec hst[N+1];
		struct d_strvec hslam[N+1];
		struct d_strvec hsres_q[N+1];
		struct d_strvec hsres_b[N+1];
		struct d_strvec hsres_d[N+1];
		struct d_strvec hsres_m[N+1];
		struct d_strvec hsres_work[2];

		for(ii=0; ii<=N; ii++)
			{
			d_create_strmat(nu_v[ii]+nx_v[ii]+1, nx_v[ii+1], &hsBAbt[ii], (void *) hpBAbt[ii]);
			hsBAbt[ii].cn = cnx_v[ii];
			d_create_strvec(nx_v[ii], &hsb[ii], (void *) hb[ii]);
			d_create_strmat(nu_v[ii]+nx_v[ii]+1, nu_v[ii]+nx_v[ii], &hsRSQrq[ii], (void *) hpRSQrq[ii]);
			hsRSQrq[ii].cn = cnux_v[ii];
			d_create_strvec(nu_v[ii]+nx_v[ii], &hsrq[ii], (void *) hrq[ii]);
			d_create_strvec(nb_v[ii]+ng_v[ii], &hsd[ii], (void *) hd[ii]);
			d_create_strvec(nu_v[ii]+nx_v[ii], &hsux[ii], (void *) hux[ii]);
			d_create_strvec(nx_v[ii], &hspi[ii], (void *) hpi[ii]);
			d_create_strvec(2*nb_v[ii]+2*ng_v[ii], &hst[ii], (void *) ht[ii]);
			d_create_strvec(2*nb_v[ii]+2*ng_v[ii], &hslam[ii], (void *) hlam[ii]);
			d_create_strvec(nu_v[ii]+nx_v[ii], &hsres_q[ii], (void *) hrrq[ii]);
			d_create_strvec(nx_v[ii], &hsres_b[ii], (void *) hrb[ii]);
			d_create_strvec(2*nb_v[ii]+2*ng_v[ii], &hsres_d[ii], (void *) hrd[ii]);
			d_create_strvec(2*nb_v[ii]+2*ng_v[ii], &hsres_m[ii], (void *) hrm[ii]);
			}
		d_allocate_strvec(pnzM, &hsres_work[0]);
		d_allocate_strvec(pnzM, &hsres_work[1]);

		// compute residuals
		d_res_res_mpc_hard_libstr(N, nx_v, nu_v, nb_v, hidxb, ng_v, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hslam, hst, hsres_work, hsres_q, hsres_b, hsres_d, hsres_m, &mu);

		// print residuals
		printf("\nhrrq\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nu_v[ii]+nx_v[ii], hrrq[ii], 1);

		printf("\nhrb\n\n");
		for(ii=1; ii<=N; ii++)
			d_print_mat_e(1, nx_v[ii], hrb[ii], 1);

		printf("\nhrd low\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nb_v[ii], hrd[ii], 1);

		printf("\nhrd up\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nb_v[ii], hrd[ii]+nb_v[ii], 1);

		}


	// zero the solution again
	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nu_v[ii]+nx_v[ii]; jj++) hux[ii][jj] = 0.0;

	// modify constraints
#if 0
	for(jj=0; jj<nbx; jj++)
		{
		dN[jj]          = - 4.0;   //   xmin
		dN[nb_v[N]+jj] =   4.0;   //   xmax
		idxbN[jj] = jj;
		}
	for(jj=0; jj<ng_v[N]; jj++)
		{
		dN[2*nb_v[N]+jj]          =   0.1;   //   xmin
		dN[2*nb_v[N]+ng_v[N]+jj] =   0.1;   //   xmax
		}
#endif

#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu_v[ii]+nx_v[ii]+1, nu_v[ii]+nx_v[ii], bs, hpRSQrq[ii], cnux_v[ii]);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu_v[ii]+nx_v[ii], hrq[ii], 1);
exit(1);
#endif

	gettimeofday(&tv2, NULL); // stop

	printf("\nsolution...\n");
	for(rep=0; rep<nrep; rep++)
		{

#if USE_IPM_RES
//		d_kkt_solve_new_rhs_res_mpc_hard_tv(N, nx_v, nu_v, nb_v, hidxb, ng_v, hpBAbt, hb, hpRSQrq, hrq, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
#else
//		d_kkt_solve_new_rhs_mpc_hard_tv(N, nx_v, nu_v, nb_v, hidxb, ng_v, hpBAbt, hb, hpRSQrq, hrq, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
#endif

		}
	printf("\ndone\n");

	gettimeofday(&tv3, NULL); // stop

#if 0
	printf("\nsolution from low-level interface (resolve final kkt)\n\n");
	printf("\nux\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nu_v[ii]+nx_v[ii], hux[ii], 1);
	printf("\npi\n\n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nx_v[ii+1], hpi[ii], 1);
//	printf("\nux\n\n");
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, 2*nb_v[ii]+2*ng_v[ii], hlam[ii], 1);
//	printf("\nux\n\n");
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, 2*nb_v[ii]+2*ng_v[ii], ht[ii], 1);
#endif

	// residuals
	if(compute_res)
		{
		// compute residuals
//		d_res_mpc_hard_tv(N, nx_v, nu_v, nb_v, hidxb, ng_v, hpBAbt, hb, hpRSQrq, hrq, hux, hpDCt, hd, hpi, hlam, ht, hrrq, hrb, hrd, &mu);

#if 0
		// print residuals
		printf("\nhrrq\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nu_v[ii]+nx_v[ii], hrrq[ii], 1);

		printf("\nhrb\n\n");
		for(ii=0; ii<N; ii++)
			d_print_mat_e(1, nx_v[ii+1], hrb[ii], 1);

		printf("\nhrd low\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nb_v[ii], hrd[ii], 1);

		printf("\nhrd up\n\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat_e(1, nb_v[ii], hrd[ii]+nb_v[ii], 1);
#endif

		}

	double time_ipm = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
	double time_final = (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\n");
	printf(" Average number of iterations over %d runs: %5.1f\n", nrep, kk_avg / (double) nrep);
	printf(" Average solution time over %d runs: %5.2e seconds (IPM)\n", nrep, time_ipm);
	printf(" Average solution time over %d runs: %5.2e seconds (resolve final kkt)\n", nrep, time_final);
	printf("\n\n");

/************************************************
* compute residuals
************************************************/	

/************************************************
* free memory
************************************************/	

	// problem data
	free(A);
	free(B);
	d_free_align(b);
	d_free_align(x0);
	free(C);
	free(D);
	free(Q);
	free(S);
	free(R);
	free(q);
	free(r);

	// low level interface
	d_free_align(pA);
	d_free_align(b0);
	d_free_align(pBAbt0);
	d_free_align(pBAbt1);
	d_free_align(d0);
	d_free_align(d1);
	d_free_align(dN);
	d_free_align(pDCt0);
	d_free_align(pDCt1);
	free(DCN);
	d_free_align(pDCtN);
	free(idxb0);
	free(idxb1);
	free(idxbN);
	d_free_align(pRSQ0);
	d_free_align(pRSQ1);
	d_free_align(pRSQN);
	d_free_align(rq0);
	d_free_align(rq1);
	d_free_align(rqN);
	d_free_align(work);
	free(stat);
	for(ii=0; ii<N; ii++)
		{
		d_free_align(hux[ii]);
		d_free_align(hpi[ii+1]);
		d_free_align(hlam[ii]);
		d_free_align(ht[ii]);
		d_free_align(hrb[ii]);
		d_free_align(hrrq[ii]);
		d_free_align(hrd[ii]);
		}
	d_free_align(hux[N]);
	d_free_align(hlam[N]);
	d_free_align(ht[N]);
	d_free_align(hrrq[N]);
	d_free_align(hrd[N]);
	
#if 0
	// high level interface
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
	free(rlb);
	free(lg);
	free(rlg);
	free(lgN);
	free(ub);
	free(rub);
	free(ug);
	free(rug);
	free(ugN);
	free(rx);
	free(ru);
	free(rpi);
	free(rlam);
	free(rt);
	free(rwork);
#endif
	
	// new high level interface
	free(lb0);
	free(ub0);
	free(lb1);
	free(ub1);
	free(lbN);
	free(ubN);
	free(lg0);
	free(ug0);
	free(lg1);
	free(ug1);
	free(work1);
	for(ii=0; ii<N; ii++)
		{
		free(hx[ii]);
		free(hu[ii]);
		free(hpi1[ii]);
		free(hlam1[ii]);
		free(ht1[ii]);
		}
	free(hx[N]);
	free(hlam1[N]);
	free(ht1[N]);

	return 0;
	
	}

#endif
