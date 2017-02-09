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

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/blas_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_aux.h"
#include "../include/mpc_solvers.h"
#include "../include/block_size.h"
#include "tools.h"
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
	
	int rep, nrep=1000; //000;//NREP;

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
//	d_print_strmat(nu[0]+1, nx[1], &sBAbt0, 0, 0);
#endif

	struct d_strmat sBAbt1;
	if(N>1)
		{
		d_allocate_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1);
		d_cvt_tran_mat2strmat(nx_, nu_, B, nx_, &sBAbt1, 0, 0);
		d_cvt_tran_mat2strmat(nx_, nx_, A, nx_, &sBAbt1, nu[1], 0);
		d_cvt_tran_mat2strmat(nx_, 1, b, nx_, &sBAbt1, nu[1]+nx[1], 0);
//		d_print_strmat(nu[1]+nx[1]+1, nx[2], &sBAbt1, 0, 0);
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
//	d_print_strmat(nu[0]+nx[0]+1, nu[0]+nx[0], &sRSQrq0, 0, 0);

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
//		d_print_strmat(nu[1]+nx[1]+1, nu[1]+nx[1], &sRSQrq1, 0, 0);

		d_allocate_strvec(nu[1]+nx[1], &srq1);
		d_cvt_vec2strvec(nu_, r, &srq1, 0);
		d_cvt_vec2strvec(nx_, q, &srq1, nu[1]);
		}

	struct d_strmat sRSQrqN;
	d_allocate_strmat(nx[N]+1, nx[N], &sRSQrqN);
	d_cvt_mat2strmat(nx_, nx_, Q, nx_, &sRSQrqN, 0, 0);
	d_cvt_tran_mat2strmat(nx_, 1, q, nx_, &sRSQrqN, nx[1], 0);
//	d_print_strmat(nu[N]+nx[N]+1, nu[N]+nx[N], &sRSQrqN, 0, 0);

	struct d_strvec srqN;
	d_allocate_strvec(nx[N], &srqN);
	d_cvt_vec2strvec(nx_, q, &srqN, 0);

/************************************************
* constraints
************************************************/	

#if MHE!=1
	double *d0; d_zeros(&d0, 2*nb[0], 1);
	int *idxb0; int_zeros(&idxb0, nb[0], 1);
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
		d0[0*nb[0]+ii] = - 4.0; // x_min
		d0[1*nb[0]+ii] = + 4.0; // x_max
		idxb0[ii] = ii;
		}
#endif

	double *d1; 
	int *idxb1; 
	if(N>1)
		{
		d_zeros(&d1, 2*nb[1], 1);
		int_zeros(&idxb1, nb[1], 1);
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
			d1[0*nb[1]+ii] = - 4.0; // x_min
			d1[1*nb[1]+ii] = + 4.0; // x_max
			idxb1[ii] = ii;
			}
		}

	double *dN; d_zeros(&dN, 2*nb[N], 1);
	int *idxbN; int_zeros(&idxbN, nb[N], 1);
	// no inputs
	// states
	for(ii=0 ; ii<nb[N]; ii++)
		{
		dN[0*nb[N]+ii] = - 4.0; // x_min
		dN[1*nb[N]+ii] = + 4.0; // x_max
		idxbN[ii] = ii;
		}

	struct d_strvec sd0;
	d_allocate_strvec(2*nb[0], &sd0);
	d_cvt_vec2strvec(2*nb[0], d0, &sd0, 0);
//	d_print_tran_strvec(2*nb[0], &sd0, 0);

	struct d_strvec sd1;
	d_allocate_strvec(2*nb[1], &sd1);
	d_cvt_vec2strvec(2*nb[1], d1, &sd1, 0);
//	d_print_tran_strvec(2*nb[1], &sd1, 0);

	struct d_strvec sdN;
	d_allocate_strvec(2*nb[N], &sdN);
	d_cvt_vec2strvec(2*nb[N], dN, &sdN, 0);
//	d_print_tran_strvec(2*nb[N], &sdN, 0);

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
	void *work_space_ipm;
	v_zeros_align(&work_space_ipm, d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

	struct timeval tv0, tv1;

	printf("\nsolving... (full space system)\n");

	gettimeofday(&tv0, NULL); // stop

	for(rep=0; rep<nrep; rep++)
		{
		hpmpc_status = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, hsux, 1, hspi, hslam, hst, work_space_ipm);
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

	for(ii=0; ii<N; ii++)
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

	void *work_space_res;
	v_zeros_align(&work_space_res, d_res_res_mpc_hard_work_space_size_bytes_libstr(N, nx, nu, nb, ng));

	d_res_res_mpc_hard_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hslam, hst, hsrrq, hsrb, hsrd, hsrm, &mu, work_space_res);

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

	// condensed problem size
	int N2 = 1;

/************************************************
* solve condensed system using IPM
************************************************/	
	
/************************************************
* partial condensing
************************************************/	

	int N3 = 3;

	// compute problem size
	int nx3[N3+1];
	int nu3[N3+1];
	int nb3[N3+1];
	int ng3[N3+1];

	d_part_cond_compute_problem_size_libstr(N, nx, nu, nb, hidxb, ng, N3, nx3, nu3, nb3, ng3);
	
	printf("\npartial condensing, problem size (N=%d)\n", N3);
	for(ii=0; ii<=N3; ii++)
		printf("\n%d %d %d %d\n", nx3[ii], nu3[ii], nb3[ii], ng3[ii]);

	int work_space_sizes[4];
	int work_space_size_part_cond = d_part_cond_work_space_size_bytes_libstr(N, nx, nu, nb, hidxb, ng, N3, nx3, nu3, nb3, ng3, work_space_sizes);
	void *work_space_part_cond;
	v_zeros_align(&work_space_part_cond, work_space_size_part_cond);
//	printf("\n%d %d %d %d %d\n", work_space_size_part_cond, work_space_sizes[0], work_space_sizes[1], work_space_sizes[2], work_space_sizes[3]); 

	int memory_space_size_part_cond = d_part_cond_memory_space_size_bytes_libstr(N, nx, nu, nb, hidxb, ng, N3, nx3, nu3, nb3, ng3);
	void *memory_space_part_cond;
	v_zeros_align(&memory_space_part_cond, memory_space_size_part_cond);
//	printf("\n%d\n", memory_space_size_part_cond);

	struct d_strmat hsBAbt3[N3];
	struct d_strvec hsb3[N3];
	struct d_strmat hsRSQrq3[N3+1];
	struct d_strvec hsrq3[N3+1];
	struct d_strmat hsDCt3[N3+1];
	struct d_strvec hsd3[N3+1];
	int *hidxb3[N3+1];

	d_part_cond_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsRSQrq, hsDCt, hsd, N3, nx3, nu3, nb3, hidxb3, ng3, hsBAbt3, hsRSQrq3, hsDCt3, hsd3, memory_space_part_cond, work_space_part_cond, work_space_sizes);

//	printf("\nhBAbt3\n\n");
//	for(ii=0; ii<N3; ii++)
//		d_print_strmat(nu3[ii]+nx3[ii]+1, nx3[ii+1], &hsBAbt3[ii], 0, 0);
	
	for(ii=0; ii<N3; ii++)
		{
		d_allocate_strvec(nx3[ii+1], &hsb3[ii]);
		drowex_libstr(nx3[ii+1], 1.0, &hsBAbt3[ii], nu3[ii]+nx3[ii], 0, &hsb3[ii], 0);
		}

//	printf("\nhb3\n\n");
//	for(ii=0; ii<N3; ii++)
//		d_print_strvec(nx3[ii+1], &hsb3[ii], 0);
	
//	printf("\nhRSQrq3\n\n");
//	for(ii=0; ii<=N3; ii++)
//		d_print_strmat(nu3[ii]+nx3[ii]+1, nu3[ii]+nx3[ii], &hsRSQrq3[ii], 0, 0);

	for(ii=0; ii<=N3; ii++)
		{
		d_allocate_strvec(nu3[ii]+nx3[ii], &hsrq3[ii]);
		drowex_libstr(nu3[ii]+nx3[ii], 1.0, &hsRSQrq3[ii], nu3[ii]+nx3[ii], 0, &hsrq3[ii], 0);
		}

//	printf("\nhrq3\n\n");
//	for(ii=0; ii<=N3; ii++)
//		d_print_strvec(nu3[ii]+nx3[ii], &hsrq3[ii], 0);
	
//	printf("\nhDCt3\n\n");
//	for(ii=0; ii<=N3; ii++)
//		d_print_strmat(nu3[ii]+nx3[ii], ng3[ii], &hsDCt3[ii], 0, 0);

//	printf("\nhd3\n\n");
//	for(ii=0; ii<=N3; ii++)
//		d_print_tran_strvec(2*nb3[ii]+2*ng3[ii], &hsd3[ii], 0);

//	printf("\nhidxb3\n\n");
//	for(ii=0; ii<=N3; ii++)
//		int_print_mat(1, nb3[ii], hidxb3[ii], 1);

/************************************************
* solve partially condensed system using IPM
************************************************/	
	
	// result vectors
	struct d_strvec hsux3[N3+1];
	struct d_strvec hspi3[N3+1];
	struct d_strvec hslam3[N3+1];
	struct d_strvec hst3[N3+1];
	for(ii=0; ii<=N3; ii++)
		{
		d_allocate_strvec(nu3[ii]+nx3[ii], &hsux3[ii]);
		d_allocate_strvec(nx3[ii], &hspi3[ii]);
		d_allocate_strvec(2*nb3[ii]+2*ng3[ii], &hslam3[ii]);
		d_allocate_strvec(2*nb3[ii]+2*ng3[ii], &hst3[ii]);
		}

	// work space
	void *work_space_ipm3;
	v_zeros_align(&work_space_ipm3, d_ip2_res_mpc_hard_work_space_size_bytes_libstr(N3, nx3, nu3, nb3, ng3));

	printf("\nsolving... (partially condensed system)\n");

	gettimeofday(&tv0, NULL); // stop

	for(rep=0; rep<nrep; rep++)
		{
		hpmpc_status = d_ip2_res_mpc_hard_libstr(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N3, nx3, nu3, nb3, hidxb3, ng3, hsBAbt3, hsRSQrq3, hsDCt3, hsd3, hsux3, 1, hspi3, hslam3, hst3, work_space_ipm3);
		}

	gettimeofday(&tv1, NULL); // stop

	float time_ipm_part_cond = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

	printf("\nstatistics from last run\n\n");
	for(jj=0; jj<kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\talpha = %f\tmu = %f\tmu = %e\n", jj, stat[5*jj], stat[5*jj+1], stat[5*jj+2], stat[5*jj+2], stat[5*jj+3], stat[5*jj+4], stat[5*jj+4]);
	printf("\n");
	
	printf("\nux3 =\n");
	for(ii=0; ii<=N3; ii++)
		d_print_tran_strvec(nu3[ii]+nx3[ii], &hsux3[ii], 0);

	printf("\npi3 =\n");
	for(ii=0; ii<=N3; ii++)
		d_print_tran_strvec(nx3[ii], &hspi3[ii], 0);

	printf("\nlam3 =\n");
	for(ii=0; ii<=N3; ii++)
		d_print_tran_strvec(2*nb3[ii]+2*ng3[ii], &hslam3[ii], 0);

	printf("\nt3 =\n");
	for(ii=0; ii<=N3; ii++)
		d_print_tran_strvec(2*nb3[ii]+2*ng3[ii], &hst3[ii], 0);

	// residuals vectors
	struct d_strvec hsrrq3[N3+1];
	struct d_strvec hsrb3[N3];
	struct d_strvec hsrd3[N3+1];
	struct d_strvec hsrm3[N3+1];

	d_allocate_strvec(nu3[0]+nx3[0], &hsrrq3[0]);
	d_allocate_strvec(nx3[1], &hsrb3[0]);
	d_allocate_strvec(2*nb3[0]+2*ng3[0], &hsrd3[0]);
	d_allocate_strvec(2*nb3[0]+2*ng3[0], &hsrm3[0]);
	for(ii=1; ii<N3; ii++)
		{
		d_allocate_strvec(nu3[ii]+nx3[ii], &hsrrq3[ii]);
		d_allocate_strvec(nx3[ii+1], &hsrb3[ii]);
		d_allocate_strvec(2*nb3[ii]+2*ng3[ii], &hsrd3[ii]);
		d_allocate_strvec(2*nb3[ii]+2*ng3[ii], &hsrm3[ii]);
		}
	d_allocate_strvec(nu3[N3]+nx3[N3], &hsrrq3[N3]);
	d_allocate_strvec(2*nb3[N3]+2*ng3[N3], &hsrd3[N3]);
	d_allocate_strvec(2*nb3[N3]+2*ng3[N3], &hsrm3[N3]);

	int ngM3 = ng3[0];
	for(ii=1; ii<=N; ii++)
		{
		ngM3 = ng3[ii]>ngM3 ? ng3[ii] : ngM3;
		}

	void *work_space_res3;
	v_zeros_align(&work_space_res3, d_res_res_mpc_hard_work_space_size_bytes_libstr(N3, nx3, nu3, nb3, ng3));

	// compute residuals on condensed system
	d_res_res_mpc_hard_libstr(N3, nx3, nu3, nb3, hidxb3, ng3, hsBAbt3, hsb3, hsRSQrq3, hsrq3, hsux3, hsDCt3, hsd3, hspi3, hslam3, hst3, hsrrq3, hsrb3, hsrd3, hsrm3, &mu, work_space_res3);

	printf("\nres_rq3\n");
	for(ii=0; ii<=N3; ii++)
		d_print_e_tran_strvec(nu3[ii]+nx3[ii], &hsrrq3[ii], 0);

	printf("\nres_b3\n");
	for(ii=0; ii<N3; ii++)
		d_print_e_tran_strvec(nx3[ii+1], &hsrb3[ii], 0);

	printf("\nres_d3\n");
	for(ii=0; ii<=N3; ii++)
		d_print_e_tran_strvec(2*nb3[ii]+2*ng3[ii], &hsrd3[ii], 0);

	printf("\nres_m3\n");
	for(ii=0; ii<=N3; ii++)
		d_print_e_tran_strvec(2*nb3[ii]+2*ng3[ii], &hsrm3[ii], 0);

	// convert result vectors to full space formulation
	void *work_space_part_expand;
	int work_space_sizes_part_expand[2];
	v_zeros_align(&work_space_part_expand, d_part_expand_work_space_size_bytes_libstr(N, nx, nu, nb, ng, work_space_sizes_part_expand));

	for(ii=0; ii<=N; ii++)
		dvecse_libstr(nu[ii]+nx[ii], 0.0, &hsux[ii], 0);

	for(ii=0; ii<=N; ii++)
		dvecse_libstr(nx[ii], 0.0, &hspi[ii], 0);

	for(ii=0; ii<=N; ii++)
		dvecse_libstr(2*nb[ii]+2*ng[ii], 0.0, &hslam[ii], 0);

	for(ii=0; ii<=N; ii++)
		dvecse_libstr(2*nb[ii]+2*ng[ii], 0.0, &hst[ii], 0);

	printf("\nexpanding solution...\n");

	d_part_expand_solution_libstr(N, nx, nu, nb, hidxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsDCt, hsux, hspi, hslam, hst, N3, nx3, nu3, nb3, hidxb3, ng3, hsux3, hspi3, hslam3, hst3, work_space_part_expand, work_space_sizes_part_expand);

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

/************************************************
* free memory partial condensing
************************************************/	

	v_free_align(work_space_part_cond);
	v_free_align(memory_space_part_cond);
	v_free_align(work_space_ipm3);
	v_free_align(work_space_part_expand);
	v_free_align(work_space_res3);
	for(ii=0; ii<N3; ii++)
		{
		d_free_strvec(&hsb3[ii]);
		d_free_strvec(&hsrq3[ii]);
		d_free_strvec(&hsux3[ii]);
		d_free_strvec(&hspi3[ii]);
		d_free_strvec(&hslam3[ii]);
		d_free_strvec(&hst3[ii]);
		d_free_strvec(&hsrrq3[ii]);
		d_free_strvec(&hsrb3[ii]);
		d_free_strvec(&hsrd3[ii]);
		d_free_strvec(&hsrm3[ii]);
		}
	ii = N3;
	d_free_strvec(&hsrq3[ii]);
	d_free_strvec(&hsux3[ii]);
	d_free_strvec(&hspi3[ii]);
	d_free_strvec(&hslam3[ii]);
	d_free_strvec(&hst3[ii]);
	d_free_strvec(&hsrrq3[ii]);
	d_free_strvec(&hsrd3[ii]);
	d_free_strvec(&hsrm3[ii]);

/************************************************
* free memory full space
************************************************/	

	// TODO
	d_free(A);
	d_free(B);
	d_free(b);
	d_free(x0);
	d_free(R);
	d_free(S);
	d_free(Q);
	d_free(r);
	d_free(q);
	d_free(d0);
	int_free(idxb0);
	d_free(d1);
	int_free(idxb1);
	d_free(dN);
	int_free(idxbN);

	v_free_align(work_space_ipm);

	d_free_strvec(&sx0);
	d_free_strvec(&sb);
	d_free_strmat(&sA);
	d_free_strvec(&sb0);
	d_free_strmat(&sBAbt0);
	if(N>1)
		d_free_strmat(&sBAbt1);
	d_free_strvec(&sr);
	d_free_strmat(&sS);
	d_free_strvec(&sr0);
	d_free_strmat(&sRSQrq0);
	d_free_strvec(&srq0);
	if(N>1)
		d_free_strmat(&sRSQrq1);
	if(N>1)
		d_free_strvec(&srq1);
	d_free_strmat(&sRSQrqN);
	d_free_strvec(&srqN);
	d_free_strvec(&sd0);
	d_free_strvec(&sd1);
	d_free_strvec(&sdN);
	for(ii=0; ii<N; ii++)
		{
		d_free_strvec(&hsux[ii]);
		d_free_strvec(&hspi[ii]);
		d_free_strvec(&hslam[ii]);
		d_free_strvec(&hst[ii]);
		d_free_strvec(&hsrrq[ii]);
		d_free_strvec(&hsrb[ii]);
		d_free_strvec(&hsrd[ii]);
		d_free_strvec(&hsrm[ii]);
		}
	ii = N;
	d_free_strvec(&hsux[ii]);
	d_free_strvec(&hspi[ii]);
	d_free_strvec(&hslam[ii]);
	d_free_strvec(&hst[ii]);
	d_free_strvec(&hsrrq[ii]);
	d_free_strvec(&hsrd[ii]);
	d_free_strvec(&hsrm[ii]);

	v_free_align(work_space_res);

/************************************************
* print timings
************************************************/	

	printf("\ntime ipm full      (in sec): %e", time_ipm_full);
	printf("\ntime ipm part cond (in sec): %e\n\n", time_ipm_part_cond);

/************************************************
* return
************************************************/	

	return 0;

	}
