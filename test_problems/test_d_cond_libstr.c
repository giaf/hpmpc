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
#include "../include/mpc_solvers.h"
#include "../problem_size.h"
#include "../include/block_size.h"
#include "tools.h"
#include "test_param.h"
#include "../include/c_interface.h"



// XXX
//#include "../lqcp_solvers/d_part_cond.c"



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
	
	int rep, nrep=1; //000;//NREP;

	int nx_ = 8;//NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu_ = 3;//NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = 5;//NN; // horizon lenght
//	int nb  = nu+nx; // number of box constrained inputs and states
//	int ng  = nx; //4;  // number of general constraints
//	int ngN = nx; // number of general constraints at the last stage
	printf("\nN = %d, nx = %d, nu = %d\n\n", N, nx_, nu_);

#define MHE 0


//	int nbu = nu<nb ? nu : nb ;
//	int nbx = nb-nu>0 ? nb-nu : 0;


	// stage-wise variant size
	int nx[N+1];
#if MHE==1
	nx[0] = nx_;
#else
	nx[0] = 0;
#endif
	for(ii=1; ii<=N; ii++)
		nx[ii] = nx_;

	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_;
	nu[N] = 0; // XXX

	int nb[N+1];
	nb[0] = nu[0] + nx[0]/2;
	for(ii=1; ii<N; ii++)
		nb[ii] = nu[1] + nx[ii]/2;
	nb[N] = nu[N] + nx[N]/2;

	int ng[N+1];
	for(ii=0; ii<N; ii++)
		ng[ii] = 0; //ng;
	ng[N] = 0; //ngN;
//	ng[M] = nx_; // XXX
	


	int info = 0;
		
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

#if MHE!=1
	struct d_strvec sx0;
	d_allocate_strvec(nx_, &sx0);
	d_cvt_vec2strvec(nx_, x0, &sx0, 0);
	struct d_strvec sb;
	d_allocate_strvec(nx_, &sb);
	d_cvt_vec2strvec(nx_, b, &sb, 0);
	struct d_strmat sA;
	d_allocate_strmat(nx_, nx_, &sA);
	d_cvt_mat2strmat(nx_, nx_, A, nx_, &sA, 0, 0);
	struct d_strvec sb0;
	d_allocate_strvec(nx_, &sb0);
	dgemv_n_libstr(nx_, nx_, 1.0, &sA, 0, 0, &sx0, 0, 1.0, &sb, 0, &sb0, 0);

	struct d_strmat sBAbt0;
	d_allocate_strmat(nu[0]+1, nx[1], &sBAbt0);
	d_cvt_tran_mat2strmat(nx_, nu_, B, nx_, &sBAbt0, 0, 0);
	drowin_libstr(nx[1], 1.0, &sb0, 0, &sBAbt0, nu[0], 0);
	d_print_strmat(nu[0]+1, nx[1], &sBAbt0, 0, 0);
#endif

	struct d_strmat sBAbt1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1);
		d_cvt_tran_mat2strmat(nx_, nu_, B, nx_, &sBAbt1, 0, 0);
		d_cvt_tran_mat2strmat(nx_, nx_, A, nx_, &sBAbt1, nu[1], 0);
		d_cvt_tran_mat2strmat(nx_, 1, b, nx_, &sBAbt1, nu[1]+nx[1], 0);
		d_print_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1, 0, 0);
		}
	
/************************************************
* cost function
************************************************/	

	double *R; d_zeros(&R, nu_, nu_);
	for(ii=0; ii<nu_; ii++) R[ii*(nu_+1)] = 2.0;

	double *S; d_zeros(&S, nu_, nx_);

	double *Q; d_zeros(&Q, nx_, nx_);
	for(ii=0; ii<nx_; ii++) Q[ii*(nx_+1)] = 1.0;

	double *r; d_zeros(&r, nu_, 1);
	for(ii=0; ii<nu_; ii++) r[ii] = 0.2;

	double *q; d_zeros(&q, nx_, 1);
	for(ii=0; ii<nx_; ii++) q[ii] = 0.1;

#if MHE!=1
	struct d_strvec sr;
	d_allocate_strvec(nu_, &sr);
	d_cvt_vec2strvec(nu_, r, &sr, 0);
	struct d_strmat sS;
	d_allocate_strmat(nu_, nx_, &sS);
	d_cvt_mat2strmat(nu_, nx_, S, nu_, &sS, 0, 0);
	struct d_strvec sr0;
	d_allocate_strvec(nu_, &sr0);
	dgemv_n_libstr(nu_, nx_, 1.0, &sS, 0, 0, &sx0, 0, 1.0, &sr, 0, &sr0, 0);

	struct d_strmat sRSQrq0;
	d_allocate_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0);
	d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRSQrq0, 0, 0);
	drowin_libstr(nu[0], 1.0, &sr0, 0, &sRSQrq0, nu[0], 0);
	d_print_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0, 0, 0);

	struct d_strvec srq0;
	d_allocate_strvec(nu[0]+nx[0], &srq0);
	dveccp_libstr(nu[0], 1.0, &sr0, 0, &srq0, 0);
#endif

	struct d_strmat sRSQrq1;
	struct d_strvec srq1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1);
		d_cvt_mat2strmat(nu_, nu_, R, nu_, &sRSQrq1, 0, 0);
		d_cvt_tran_mat2strmat(nu_, nx_, S, nu_, &sRSQrq1, nu[1], 0);
		d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sRSQrq1, nu[1], nu[1]);
		d_cvt_tran_mat2strmat(nu_, 1, r, nu_, &sRSQrq1, nu[1]+nx[1], 0);
		d_cvt_tran_mat2strmat(nx_, 1, q, nx_, &sRSQrq1, nu[1]+nx[1], nu[1]);
		d_print_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1, 0, 0);

		d_allocate_strvec(nu[1]+nx[1], &srq1);
		d_cvt_vec2strvec(nu_, r, &srq1, 0);
		d_cvt_vec2strvec(nx_, q, &srq1, nu[1]);
		}

	struct d_strmat sRSQrqN;
	d_allocate_strmat(nx[N]+1, nx[N], &sRSQrqN);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sRSQrqN, 0, 0);
	d_cvt_tran_mat2strmat(nx_, 1, q, nx_, &sRSQrqN, nx[1], 0);
	d_print_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN, 0, 0);

	struct d_strvec srqN;
	d_allocate_strvec(nx[N], &srqN);
	d_cvt_vec2strvec(nx_, q, &srqN, 0);

/************************************************
* constraints
************************************************/	

#if MHE!=1
	double *d0; d_zeros_align(&d0, 2*nb[0], 1);
	int *idxb0; i_zeros(&idxb0, nb[0], 1);
	// inputs
	for(ii=0; ii<nu[0]; ii++)
		{
		d0[0*nb[0]+ii] = - 0.5; // u_min
		d0[1*nb[0]+ii] = + 0.5; // u_max
		idxb0[ii] = ii;
		}
	// states
	for( ; ii<nb[0]; ii++)
		{
		d0[0*nb[0]+ii] = - 800.0; // x_min
		d0[1*nb[0]+ii] = + 800.0; // x_max
		idxb0[ii] = ii;
		}
#endif

	double *d1; 
	int *idxb1; 
	if(N>1)
		{
		d_zeros_align(&d1, 2*nb[1], 1);
		i_zeros(&idxb1, nb[1], 1);
		// inputs
		for(ii=0; ii<nu[1]; ii++)
			{
			d1[0*nb[1]+ii] = - 0.5; // u_min
			d1[1*nb[1]+ii] = + 0.5; // u_max
			idxb1[ii] = ii;
			}
		// states
		for( ; ii<nb[1]; ii++)
			{
			d1[0*nb[1]+ii] = - 800.0; // x_min
			d1[1*nb[1]+ii] = + 800.0; // x_max
			idxb1[ii] = ii;
			}
		}

	double *dN; d_zeros_align(&dN, 2*nb[N], 1);
	int *idxbN; i_zeros(&idxbN, nb[N], 1);
	// no inputs
	// states
	for(ii=0 ; ii<nb[N]; ii++)
		{
		dN[0*nb[N]+ii] = - 800.0; // x_min
		dN[1*nb[N]+ii] = + 800.0; // x_max
		idxbN[ii] = ii;
		}

	struct d_strvec sd0;
	d_allocate_strvec(2*nb[0], &sd0);
	d_cvt_vec2strvec(2*nb[0], d0, &sd0, 0);
	d_print_tran_strvec(2*nb[0], &sd0, 0);

	struct d_strvec sd1;
	d_allocate_strvec(2*nb[1], &sd1);
	d_cvt_vec2strvec(2*nb[1], d1, &sd1, 0);
	d_print_tran_strvec(2*nb[1], &sd1, 0);

	struct d_strvec sdN;
	d_allocate_strvec(2*nb[N], &sdN);
	d_cvt_vec2strvec(2*nb[N], dN, &sdN, 0);
	d_print_tran_strvec(2*nb[N], &sdN, 0);

/************************************************
* array of matrices & work space
************************************************/	

	// original MPC
	struct d_strmat hsBAbt[N];
	struct d_strvec hsb[N];
	struct d_strmat hsRSQrq[N+1];
	struct d_strvec hsrq[N+1];
	struct d_strmat hsDCt[N+1]; // XXX
	struct d_strvec hsd[N+1];
	int *hidxb[N+1];

	ii = 0;
#if MHE!=1
	hsBAbt[ii] = sBAbt0;
	hsb[ii] = sb0;
	hsRSQrq[ii] = sRSQrq0;
	hsrq[ii] = srq0;
	hsd[ii] = sd0;
	hidxb[0] = idxb0;
#else
	hsBAbt[ii] = sBAbt1;
	hsb[ii] = sb;
	hsRSQrq[ii] = sRSQrq1;
	hsrq[ii] = srq1;
	hsd[ii] = sd1;
	hidxb[0] = idxb1;
#endif

	for(ii=1; ii<N; ii++)
		{
		hsBAbt[ii] = sBAbt1;
		hsb[ii] = sb;
		hsRSQrq[ii] = sRSQrq1;
		hsrq[ii] = srq1;
		hsd[ii] = sd1;
		hidxb[ii] = idxb1;
		}
	hsRSQrq[ii] = sRSQrqN;
	hsrq[ii] = srqN;
	hsd[ii] = sdN;
	hidxb[N] = idxbN;


/************************************************
* solve full spase system using Riccati / IPM
************************************************/	

	// IPM stuff
	int hpmpc_status;
	int kk = -1;
	int k_max = 10;
	double mu0 = 2.0;
	double mu_tol = 1e-20;
	double alpha_min = 1e-8;
	int warm_start = 0; // read initial guess from x and u
	double *stat; d_zeros(&stat, k_max, 5);
	int compute_res = 1;
	int compute_mult = 1;

	// result vectors
	struct d_strvec hsux[N+1];
	struct d_strvec hspi[N+1];
	struct d_strvec hslam[N+1];
	struct d_strvec hst[N+1];
	for(ii=0; ii<=N; ii++)
		{
		d_allocate_strvec(nu[ii]+nx[ii], &hsux[ii]);
		d_allocate_strvec(nx[ii], &hspi[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hslam[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hst[ii]);
		}

	// work space
	void *work_memory;
	v_zeros_align(&work_memory, d_ip2_res_mpc_hard_tv_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

	struct timeval tv0, tv1;

	printf("\nsolving... (full space system)\n");

	gettimeofday(&tv0, NULL); // stop

	for(rep=0; rep<nrep; rep++)
		{
//		hpmpc_status = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, 1, hspi, hslam, hst, work_memory);
		}

	gettimeofday(&tv1, NULL); // stop

	float time_ipm_full = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\nux =\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

	printf("\npi =\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nx[ii], &hspi[ii], 0);

	printf("\nlam =\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hslam[ii], 0);

	printf("\nt =\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hst[ii], 0);

	// residuals vectors
	struct d_strvec hsrrq[N+1];
	struct d_strvec hsrb[N+1];
	struct d_strvec hsrd[N+1];
	struct d_strvec hsrm[N+1];
	double mu;

	d_allocate_strvec(nu[0]+nx[0], &hsrrq[0]);
	d_allocate_strvec(nx[1], &hsrb[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrd[0]);
	d_allocate_strvec(2*nb[0]+2*ng[0], &hsrm[0]);
	for(ii=1; ii<N; ii++)
		{
		d_allocate_strvec(nu[ii]+nx[ii], &hsrrq[ii]);
		d_allocate_strvec(nx[ii+1], &hsrb[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrd[ii]);
		d_allocate_strvec(2*nb[ii]+2*ng[ii], &hsrm[ii]);
		}
	d_allocate_strvec(nu[N]+nx[N], &hsrrq[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrd[N]);
	d_allocate_strvec(2*nb[N]+2*ng[N], &hsrm[N]);

	int ngM = ng[0];
	for(ii=1; ii<=N; ii++)
		{
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}
	struct d_strvec hswork[2];
	d_allocate_strvec(ngM, &hswork[0]);
	d_allocate_strvec(ngM, &hswork[1]);

	d_res_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hslam, hst, hswork, hsrrq, hsrb, hsrd, hsrm, &mu);

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

/************************************************
* full condensing
************************************************/	

	int N2 = 1;

	int nu2[N2+1];
	nu2[0] = 0;
	for(ii=0; ii<N; ii++) 
		{
		nu2[0] += nu[ii];
		}
	nu2[N2] = nu[N];

	int nx2[N2+1];
	nx2[0] = nx[0];
	nx2[N2] = nx[N];

	int nu_tmp = 0;

	int nbb = nb[0]; // box that remain box constraints
	int nbg = 0; // box that become general constraints
	for(ii=1; ii<N; ii++)
		for(jj=0; jj<nb[ii]; jj++)
			if(hidxb[ii][jj]<nu[ii])
				nbb++;
			else
				nbg++;
	
	int nb2[N2+1];
	nb2[0] = nbb;
	nb2[N2] = nb[N];

	int ng2[N2+1];
	ng2[0] = nbg; // XXX
	ng2[N2] = ng[N];

	// condensed MPC
	struct d_strmat hsBAbt2[N2];
	struct d_strvec hsb2[N2];
	struct d_strmat hsRSQrq2[N2+1];
	struct d_strvec hsrq2[N2+1];
	struct d_strmat hsDCt2[N2+1];
	struct d_strvec hsd2[N2+1];
	int *hidxb2[N2+1];

	struct d_strmat hsGamma[N];

	int i_tmp, size_sA, size_sL, size_sM, size_sLx, size_sBAbtL;

	ii = 0;
	size_sA = d_size_strmat(nx[ii+1], nx[ii]);
	size_sL = d_size_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]);
	size_sM = d_size_strmat(nu[ii], nx[ii]);
	size_sLx = d_size_strmat(nx[ii]+1, nx[ii]);
	size_sBAbtL = d_size_strmat(nu[ii]+nx[ii]+1, nx[ii+1]);
	nu_tmp = nu[ii];
	d_allocate_strmat(nu_tmp+nx[0]+1, nx[ii+1], &hsGamma[ii]);
	for(ii=1; ii<N; ii++)
		{
		i_tmp = d_size_strmat(nx[ii+1], nx[ii]);
		size_sA = i_tmp > size_sA ? i_tmp : size_sA;
		i_tmp = d_size_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]);
		size_sL = i_tmp > size_sL ? i_tmp : size_sL;
		i_tmp = d_size_strmat(nu[ii], nx[ii]);
		size_sM = i_tmp > size_sM ? i_tmp : size_sM;
		i_tmp = d_size_strmat(nx[ii]+1, nx[ii]);
		size_sLx = i_tmp > size_sLx ? i_tmp : size_sLx;
		i_tmp = d_size_strmat(nu[ii]+nx[ii]+1, nx[ii+1]);
		size_sBAbtL = i_tmp > size_sBAbtL ? i_tmp : size_sBAbtL;
		nu_tmp += nu[ii];
		d_allocate_strmat(nu_tmp+nx[0]+1, nx[ii+1], &hsGamma[ii]);
		}
	ii = N;
	i_tmp = d_size_strmat(nx[ii]+1, nx[ii]);
	size_sL = i_tmp > size_sL ? i_tmp : size_sL;
	i_tmp = d_size_strmat(nx[ii]+1, nx[ii]);
	size_sLx = i_tmp > size_sLx ? i_tmp : size_sLx;

	d_allocate_strmat(nu2[0]+nx2[0]+1, nx2[1], &hsBAbt2[0]);
	d_allocate_strvec(nx2[1], &hsb2[0]);
	d_allocate_strmat(nu2[0]+nx2[0]+1, nu2[0]+nx2[0], &hsRSQrq2[0]);
	hsRSQrq2[N2] = hsRSQrq[N];
	d_allocate_strvec(nu2[0]+nx2[0], &hsrq2[0]);
	hsrq2[N2] = hsrq[N];
	d_allocate_strmat(nu2[0]+nx2[0]+1, ng2[0], &hsDCt2[0]);
	hsDCt2[N2] = hsDCt[N];
	d_allocate_strvec(2*nbb+2*nbg, &hsd2[0]);
	hsd2[N2] = hsd[N];
	i_zeros(&hidxb2[0], nbb, 1);
	hidxb2[N2] = hidxb[N];

	void *work_sA; v_zeros_align(&work_sA, size_sA);
	void *work_d_cond_RSQrq_libstr[4]; 
	v_zeros_align(&work_d_cond_RSQrq_libstr[0], size_sL);
	v_zeros_align(&work_d_cond_RSQrq_libstr[1], size_sM);
	v_zeros_align(&work_d_cond_RSQrq_libstr[2], size_sLx);
	v_zeros_align(&work_d_cond_RSQrq_libstr[3], size_sBAbtL);

	d_cond_BAbt_libstr(N, nx, nu, hsBAbt, work_sA, hsGamma, &hsBAbt2[0]);
	drowex_libstr(nx2[1], 1.0, &hsBAbt2[0], nu2[0]+nx2[0], 0, &hsb2[0], 0);

	printf("\nGamma\n\n");
	for(ii=0; ii<N; ii++)
		d_print_strmat(hsGamma[ii].m, hsGamma[ii].n, &hsGamma[ii], 0, 0);

	printf("\nhBAbt2\n\n");
	for(ii=0; ii<N2; ii++)
		d_print_strmat(nu2[ii]+nx2[ii]+1, nx2[ii+1], &hsBAbt2[ii], 0, 0);

	printf("\nhb2\n\n");
	for(ii=0; ii<N2; ii++)
		d_print_tran_strvec(nx2[ii+1], &hsb2[ii], 0);

	d_cond_RSQrq_libstr(N, nx, nu, hsBAbt, hsRSQrq, hsGamma, work_d_cond_RSQrq_libstr, &hsRSQrq2[0]);
	drowex_libstr(nu2[0]+nx2[0], 1.0, &hsRSQrq2[0], nu2[0]+nx2[0], 0, &hsrq2[0], 0);

	printf("\nhRSQrq2\n\n");
	for(ii=0; ii<=N2; ii++)
		d_print_strmat(nu2[ii]+nx2[ii]+1, nu2[ii]+nx2[ii], &hsRSQrq2[ii], 0, 0);

	printf("\nhrq2\n\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(nu2[ii]+nx2[ii], &hsrq2[ii], 0);

	d_cond_DCtd_libstr(N, nx, nu, nb, hidxb, hsd, hsGamma, &hsDCt2[0], &hsd2[0], hidxb2[0]);

	printf("\nhDCt2\n\n");
	for(ii=0; ii<=N2; ii++)
		d_print_strmat(nu2[ii]+nx2[ii], ng2[ii], &hsDCt2[ii], 0, 0);

	printf("\nhd2\n\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(2*nb2[ii]+2*ng2[ii], &hsd2[ii], 0);

	printf("\nhidxb2\n\n");
	for(ii=0; ii<=N2; ii++)
		i_print_mat(1, nb2[ii], hidxb2[ii], 1);

/************************************************
* solve condensed system using Riccati / IPM
************************************************/	
	
	// TODO general constraints in IPM !!!!!!!!!!!!!!!!!!!!!!!!!!!!
//	ng2[0] = 0; // XXX !!!!!!!!!!!!!!!!!!!!

	// result vectors
	struct d_strvec hsux2[N2+1];
	struct d_strvec hspi2[N2+1];
	struct d_strvec hslam2[N2+1];
	struct d_strvec hst2[N2+1];
	for(ii=0; ii<=N2; ii++)
		{
		d_allocate_strvec(nu2[ii]+nx2[ii], &hsux2[ii]);
		d_allocate_strvec(nx2[ii], &hspi2[ii]);
		d_allocate_strvec(2*nb2[ii]+2*ng2[ii], &hslam2[ii]);
		d_allocate_strvec(2*nb2[ii]+2*ng2[ii], &hst2[ii]);
		}

	// work space
	void *work_memory2;
	v_zeros_align(&work_memory2, d_ip2_res_mpc_hard_tv_work_space_size_bytes_libstr(N2, nx2, nu2, nb2, ng2));

	printf("\nsolving... (condensed system)\n");

	gettimeofday(&tv0, NULL); // stop

	for(rep=0; rep<nrep; rep++)
		{
		hpmpc_status = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N2, nx2, nu2, nb2, hidxb2, ng2, hsBAbt2, hsRSQrq2, hsDCt2, hsd2, hsux2, 1, hspi2, hslam2, hst2, work_memory2);
		}

	gettimeofday(&tv1, NULL); // stop

	float time_ipm_full2 = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\nux2 =\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(nu2[ii]+nx2[ii], &hsux2[ii], 0);

	printf("\npi2 =\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(nx2[ii], &hspi2[ii], 0);

	printf("\nlam2 =\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(2*nb2[ii]+2*ng2[ii], &hslam2[ii], 0);

	printf("\nt2 =\n");
	for(ii=0; ii<=N2; ii++)
		d_print_tran_strvec(2*nb2[ii]+2*ng2[ii], &hst2[ii], 0);

	// residuals vectors
	struct d_strvec hsrrq2[N2+1];
	struct d_strvec hsrb2[N2];
	struct d_strvec hsrd2[N2+1];
	struct d_strvec hsrm2[N2+1];

	d_allocate_strvec(nu2[0]+nx2[0], &hsrrq2[0]);
	d_allocate_strvec(nx2[1], &hsrb2[0]);
	d_allocate_strvec(2*nb2[0]+2*ng2[0], &hsrd2[0]);
	d_allocate_strvec(2*nb2[0]+2*ng2[0], &hsrm2[0]);
	for(ii=1; ii<N2; ii++)
		{
		d_allocate_strvec(nu2[ii]+nx2[ii], &hsrrq2[ii]);
		d_allocate_strvec(nx2[ii+1], &hsrb2[ii]);
		d_allocate_strvec(2*nb2[ii]+2*ng2[ii], &hsrd2[ii]);
		d_allocate_strvec(2*nb2[ii]+2*ng2[ii], &hsrm2[ii]);
		}
	d_allocate_strvec(nu2[N2]+nx2[N2], &hsrrq2[N2]);
	d_allocate_strvec(2*nb2[N2]+2*ng2[N2], &hsrd2[N2]);
	d_allocate_strvec(2*nb2[N2]+2*ng2[N2], &hsrm2[N2]);

	int ngM2 = ng2[0];
	for(ii=1; ii<=N; ii++)
		{
		ngM2 = ng2[ii]>ngM2 ? ng2[ii] : ngM2;
		}
	struct d_strvec hswork2[2];
	d_allocate_strvec(ngM2, &hswork2[0]);
	d_allocate_strvec(ngM2, &hswork2[1]);

	// compute residuals on condensed system
	d_res_res_mpc_hard_libstr(N2, nx2, nu2, nb2, hidxb2, ng2, hsBAbt2, hsb2, hsRSQrq2, hsrq2, hsux2, hsDCt2, hsd2, hspi2, hslam2, hst2, hswork2, hsrrq2, hsrb2, hsrd2, hsrm2, &mu);

	printf("\nres_rq2\n");
	for(ii=0; ii<=N2; ii++)
		d_print_e_tran_strvec(nu2[ii]+nx2[ii], &hsrrq2[ii], 0);

	printf("\nres_b2\n");
	for(ii=0; ii<N2; ii++)
		d_print_e_tran_strvec(nx2[ii+1], &hsrb2[ii], 0);

	printf("\nres_d2\n");
	for(ii=0; ii<=N2; ii++)
		d_print_e_tran_strvec(2*nb2[ii]+2*ng2[ii], &hsrd2[ii], 0);

	printf("\nres_m2\n");
	for(ii=0; ii<=N2; ii++)
		d_print_e_tran_strvec(2*nb2[ii]+2*ng2[ii], &hsrm2[ii], 0);

	// convert result vectors to full space formulation (using simulation)
	for(ii=0; ii<=N; ii++)
		dvecse_libstr(nu[ii]+nx[ii], 0.0, &hsux[ii], 0);

	int nu_sum = 0;
	for(ii=0; ii<N; ii++)
		{
		dveccp_libstr(nu[N-ii-1], 1.0, &hsux2[0], nu_sum, &hsux[N-ii-1], 0);
		nu_sum += nu[N-ii-1];
		}
	dveccp_libstr(nx[0], 1.0, &hsux2[0], nu2[0], &hsux[0], nu[0]);

	for(ii=0; ii<N; ii++)
		{
		drowex_libstr(nx[ii+1], 1.0, &hsBAbt[ii], nu[ii]+nx[ii], 0, &hsux[ii+1], nu[ii+1]);
		dgemv_t_libstr(nu[ii]+nx[ii], nx[ii+1], 1.0, &hsBAbt[ii], 0, 0, &hsux[ii], 0, 1.0, &hsux[ii+1], nu[ii+1], &hsux[ii+1], nu[ii+1]);
		}

	// TODO compute lagrangian multipliers 

	printf("\nux =\n");
	for(ii=0; ii<=N; ii++)
		d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);

/************************************************
* free memory
************************************************/	

	// TODO

/************************************************
* return
************************************************/	

	return 0;

	}
