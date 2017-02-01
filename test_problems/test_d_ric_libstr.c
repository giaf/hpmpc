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
#include "../include/lqcp_solvers.h"
#include "../include/mpc_aux.h"
#include "../include/mpc_solvers.h"
#include "tools.h"


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

	int nx_ = 8; // number of states (it has to be even for the mass-spring system test problem)
	int nu_ = 3; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = 10; // horizon lenght


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
	for(ii=0; ii<=N; ii++)
		nb[ii] = 0;

	int ng[N+1];
	for(ii=0; ii<=N; ii++)
		ng[ii] = 0;
	

	// max sizes
	int ngM = 0;
	for(ii=0; ii<=N; ii++)
		{
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}

	int nzM  = 0;
	for(ii=0; ii<=N; ii++)
		{
		nzM = nu[ii]+nx[ii]+1>nzM ? nu[ii]+nx[ii]+1 : nzM;
		}

	int nxgM = ng[N];
	for(ii=0; ii<N; ii++)
		{
		nxgM = nx[ii+1]+ng[ii]>nxgM ? nx[ii+1]+ng[ii] : nxgM;
		}
	

	printf(" Test problem: mass-spring system with %d masses and %d controls.\n", nx_/2, nu_);
	printf("\n");
	printf(" MPC problem size: %d states, %d inputs, %d horizon length.\n", nx_, nu_, N);
	printf("\n");
	printf(" Backward Riccati recursion\n");



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

/************************************************
* libstr riccati solver
************************************************/	

	struct d_strmat *hsmatdummy;
	struct d_strvec *hsvecdummy;

	struct d_strmat hsBAbt[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strmat hsDCt[N+1];
	struct d_strvec hsd[N+1];
	int *hidxb[N+1];
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strvec hsPb[N+1];
	struct d_strmat hsL[N+1];


	hsBAbt[0] = sBAbt0;
	hsRSQrq[0] = sRSQrq0;
	d_allocate_strvec(nu[0]+nx[0], &hsux[0]);
	d_allocate_strvec(nx[1], &hspi[1]);
	d_allocate_strvec(nx[1], &hsPb[1]);
	d_allocate_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &hsL[0]);
	for(ii=1; ii<N; ii++)
		{
		hsBAbt[ii] = sBAbt1;
		hsRSQrq[ii] = sRSQrq1;
		d_allocate_strvec(nu[ii]+nx[ii], &hsux[ii]);
		d_allocate_strvec(nx[ii+1], &hspi[ii+1]);
		d_allocate_strvec(nx[ii+1], &hsPb[ii+1]);
		d_allocate_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii]);
		}
	hsRSQrq[N] = sRSQrqN;
	d_allocate_strvec(nu[N]+nx[N], &hsux[N]);
	d_allocate_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &hsL[N]);
	
	// riccati work space
	void *work_ric;
	v_zeros_align(&work_ric, d_back_ric_rec_work_space_size_bytes_libstr(N, nx, nu, nb, ng));



	int M = 3;

	int nxM;

	struct d_strmat hstmpmat0;



	struct timeval tv0, tv1;

	gettimeofday(&tv0, NULL); // start

	for(rep=0; rep<nrep; rep++)
		{

		d_back_ric_rec_sv_libstr(N, nx, nu, nb, hidxb, ng, 0, hsBAbt, hsvecdummy, 0, hsRSQrq, hsvecdummy, hsmatdummy, hsvecdummy, hsvecdummy, hsux, 1, hspi, 1, hsPb, hsL, work_ric);

		}

	gettimeofday(&tv1, NULL); // stop

	printf("\nux =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi[ii], 0);

	printf("\nL =\n\n");
	for(ii=0; ii<=N; ii++)
		d_print_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii], 0, 0);

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

	d_res_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hsvecdummy, hsvecdummy, hsrrq, hsrb, hsrd, hsrm, &mu, work_res);

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

	d_free_strmat(&sA);
	d_free_strvec(&sx0);
	d_free_strmat(&sBAbt0);
	d_free_strvec(&sb0);
	d_free_strmat(&sRSQrq0);
	d_free_strvec(&srq0);
	d_free_strmat(&sRSQrqN);
	d_free_strvec(&srqN);
	if(N>1)
		{
		d_free_strmat(&sBAbt1);
		d_free_strvec(&sb1);
		d_free_strmat(&sRSQrq1);
		d_free_strvec(&srq1);
		}
	d_free_strvec(&hsux[0]);
	d_free_strvec(&hspi[1]);
	d_free_strvec(&hsrrq[0]);
	d_free_strvec(&hsrb[0]);
	for(ii=1; ii<N; ii++)
		{
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii+1]);
		d_free_strvec(&hsrrq[ii]);
		d_free_strvec(&hsrb[ii]);
		}
	d_free_strvec(&hsux[N]);
	d_free_strvec(&hsrrq[N]);

/************************************************
* return
************************************************/	

	return 0;
	}
