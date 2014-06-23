/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical University of Denmark. All rights reserved.                     *
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
#endif

// to throw floating-point exception
/*#define _GNU_SOURCE*/
/*#include <fenv.h>*/

#include "test_param.h"
#include "../problem_size.h"
#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"
#include "tools.h"



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



int main()
	{
	
	
	printf("\n");
	printf("\n");
	printf("\n");
	printf(" HPMPC -- Library for High-Performance implementation of solvers for MPC.\n");
	printf(" Copyright (C) 2014 by Technical University of Denmark. All rights reserved.\n");
	printf("\n");
	printf(" HPMPC is distributed in the hope that it will be useful,\n");
	printf(" but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf(" MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
	printf(" See the GNU Lesser General Public License for more details.\n");
	printf("\n");
	printf("\n");
	printf("\n");

	printf("Riccati solver performance test - single precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = 2.9; //3.6; //2.9;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit test_dricposv.c to modify this value).\n", GHz_max);
	printf("\n");

	// maximum flops per cycle, single precision
#if defined(TARGET_X64_AVX)
	const float flops_max = 16;
	printf("Testing solvers for AVX instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_SSE3) || defined(TARGET_AMD_SSE3)
	const float flops_max = 8;
	printf("Testing solvers for SSE3 instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_CORTEXA9)
	const float flops_max = 4;
	printf("Testing solvers for ARMv7a NEON instruction set: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X86_ATOM)
	const float flops_max = 4;
	printf("Testing solvers for SSE3 instruction set, 32 bit, optimized for Intel Atom: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_POWERPC_G2)
	const float flops_max = 2;
	printf("Testing solvers for POWERPC instruction set, 32 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_C99_4X4)
	const float flops_max = 2;
	printf("Testing reference solvers, 4x4 kernel: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_C99_2X2)
	const float flops_max = 2;
	printf("Testing reference solvers, 2x2 kernel: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#endif

	printf("\n");
	printf("Tested solvers:\n");
	printf("-sv : Riccati factorization and system solution (prediction step in IP methods)\n");
	printf("-trs: system solution after a previous call to Riccati factorization (correction step in IP methods)\n");
	printf("\n");
	printf("\n");

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
	printf("\nflush to zero on\n");
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
#endif

	// to throw floating-point exception
/*#ifndef __APPLE__*/
/*    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);*/
/*#endif*/

	int err;
	
	int i, j, ii, jj, idx;
	
	const int bsd = D_MR; //d_get_mr();
	const int bss = S_MR; //s_get_mr();
	
	int info = 0;

	int nn[] = {4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 4000, 4000, 2000, 2000, 1000, 1000, 400, 400, 400, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
	
	int vnx[] = {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024};
	int vnrep[] = {100, 100, 100, 100, 100, 100, 50, 50, 50, 20, 10, 10};
	int vN[] = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256};

	int ll;
	for(ll=0; ll<77; ll++)
/*	for(ll=0; ll<1; ll++)*/

		{

		int nx = nn[ll];//NX;//16;//nn[ll]; // number of states (it has to be even for the mass-spring system test problem)
		int nu = 2;//NU;//5; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
		int N  = 10;//NN;//10; // horizon lenght
		int nrep = nnrep[ll];
/*		int nx = NX;//16;//nn[ll]; // number of states (it has to be even for the mass-spring system test problem)*/
/*		int nu = NU;//5; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)*/
/*		int N  = NN;//10; // horizon lenght*/
/*		int nrep = NREP;*/

		int rep;
	
		int nz = nx+nu+1;
		int pnz = bss*((nz+bss-nu%bss+bss-1)/bss);
	
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
//	s_print_mat(nz, nx+1, sBAbt, pnz);
//	return 0;
	
	/* packed into contiguous memory */
		double *pBAbt; d_zeros_align(&pBAbt, pnz, pnz);
		d_cvt_mat2pmat(nz, nx, 0, bsd, BAbt, pnz, pBAbt, pnz);

		float *psBAbt; s_zeros_align(&psBAbt, pnz, pnz);
		s_cvt_d2s_pmat(nz, nx, bsd, pBAbt, pnz, bss, psBAbt, pnz);

//	d_print_pmat(nz, nx, bsd, pBAbt, pnz);
//	s_print_pmat(nz, nx, bss, spBAbt, pnz);

/************************************************
* cost function
************************************************/	

		double *Q; d_zeros_align(&Q, pnz, pnz);
		for(ii=0; ii<nu; ii++) Q[ii*(pnz+1)] = 2.0;
		for(; ii<pnz; ii++) Q[ii*(pnz+1)] = 1.0;
		for(ii=0; ii<nz; ii++) Q[nx+nu+ii*pnz] = 1.0;
		Q[(nx+nu)*(pnz+1)] = 1e6;

		/* packed into contiguous memory */
		float *pQ; s_zeros_align(&pQ, pnz, pnz);
		cvt_d2s_mat2pmat(nz, nz, 0, bss, Q, pnz, pQ, pnz);

	/* matrices series */
		float *(hpQ[N+1]);
		float *(hq[N+1]);
		float *(hux[N+1]);
		float *(hpi[N+1]);
		float *(hpBAbt[N]);
		float *(hrb[N]);
		float *(hrq[N+1]);
		for(jj=0; jj<N; jj++)
			{
			s_zeros_align(&hpQ[jj], pnz, pnz);
			s_zeros_align(&hq[jj], pnz, 1);
			s_zeros_align(&hux[jj], pnz, 1);
			s_zeros_align(&hpi[jj], nx, 1);
			hpBAbt[jj] = psBAbt;
			s_zeros_align(&hrb[jj], nx, 1);
			s_zeros_align(&hrq[jj], nx+nu, 1);
			}
		s_zeros_align(&hpQ[N], pnz, pnz);
		s_zeros_align(&hq[N], pnz, 1);
		s_zeros_align(&hux[N], pnz, 1);
		s_zeros_align(&hpi[N], nx, 1);
		s_zeros_align(&hrq[N], nx+nu, 1);
	
		// starting guess
		for(jj=0; jj<nx; jj++) hux[0][nu+jj] = (float) x0[jj];
	
		float *pL; s_zeros_align(&pL, pnz, pnz);
	
		float *pBAbtL; s_zeros_align(&pBAbtL, pnz, pnz);

/************************************************
* riccati-like iteration
************************************************/

		// predictor

		// restore cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<pnz*pnz; jj++) hpQ[ii][jj]=pQ[jj];
			}
		for(jj=0; jj<pnz*pnz; jj++) hpQ[N][jj]=pQ[jj];

		// call the solver
		sricposv_mpc(nx, nu, N, pnz, hpBAbt, hpQ, hux, pL, pBAbtL, COMPUTE_MULT, hpi, &info);

		if(PRINTRES==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				s_print_mat(1, nu, hux[ii], 1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1)
			{
			// print result 
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				s_print_mat(1, nx, hpi[ii+1], 1);
			}

		// corrector
	
		// clear solution 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nu; jj++) hux[ii][jj] = 0;
			for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = 0;
			}

		// restore linear part of cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
			}
		for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

		// call the solver 
		sricpotrs_mpc(nx, nu, N, pnz, hpBAbt, hpQ, hq, hux, pBAbtL, COMPUTE_MULT, hpi);

		if(PRINTRES==1)
			{
			// print result 
			printf("\n\ntrs\n\n");
			for(ii=0; ii<N; ii++)
				s_print_mat(1, nu, hux[ii], 1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1)
			{
			// print result 
			printf("\n\ntrs\n\n");
			for(ii=0; ii<N; ii++)
				s_print_mat(1, nx, hpi[ii+1], 1);
			}

		// restore cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<pnz*pnz; jj++) hpQ[ii][jj]=pQ[jj];
			}
		for(jj=0; jj<pnz*pnz; jj++) hpQ[N][jj]=pQ[jj];

		// restore linear part of cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
			}
		for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

		// residuals computation
		sres(nx, nu, N, pnz, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);

		if(PRINTRES==1 && COMPUTE_MULT==1)
			{
			// print result 
			printf("\n\nres\n\n");
			for(ii=0; ii<+N; ii++)
				s_print_mat(1, nx+nu, hrq[ii], 1);
			for(ii=0; ii<N; ii++)
				s_print_mat(1, nx, hrb[ii], 1);
			}



		// timing 
		struct timeval tv0, tv1, tv2;

		gettimeofday(&tv0, NULL); // start

		// double precision
		for(rep=0; rep<nrep; rep++)
			{
			// restore cost function 
			for(ii=0; ii<N; ii++)
				{
				for(jj=0; jj<pnz*pnz; jj++) hpQ[ii][jj]=pQ[jj];
				}
			for(jj=0; jj<pnz*pnz; jj++) hpQ[N][jj]=pQ[jj];

			// call the solver 
			sricposv_mpc(nx, nu, N, pnz, hpBAbt, hpQ, hux, pL, pBAbtL, COMPUTE_MULT, hpi, &info);
			}
			
		gettimeofday(&tv1, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			// clear solution 
			for(ii=0; ii<N; ii++)
				{
				for(jj=0; jj<nu; jj++) hux[ii][jj] = 0;
				for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = 0;
				}

			// restore linear part of cost function 
			for(ii=0; ii<N; ii++)
				{
				for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
				}
			for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

			// call the solver 
			sricpotrs_mpc(nx, nu, N, pnz, hpBAbt, hpQ, hq, hux, pBAbtL, COMPUTE_MULT, hpi);
			}
		
		gettimeofday(&tv2, NULL); // start



		float time_sv = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float flop_sv = (1.0/3.0*nx*nx*nx+3.0/2.0*nx*nx) + N*(7.0/3.0*nx*nx*nx+4.0*nx*nx*nu+2.0*nx*nu*nu+1.0/3.0*nu*nu*nu+13.0/2.0*nx*nx+9.0*nx*nu+5.0/2.0*nu*nu);
		if(COMPUTE_MULT==1)
			flop_sv += N*2*nx*nx;
		float Gflops_sv = 1e-9*flop_sv/time_sv;
	
		float time_trs = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float flop_trs = N*(8.0*nx*nx+8.0*nx*nu+2.0*nu*nu);
		if(COMPUTE_MULT==1)
			flop_trs += N*2*nx*nx;
		float Gflops_trs = 1e-9*flop_trs/time_trs;
		
		float Gflops_max = flops_max * GHz_max;

		if(ll==0)
			printf("\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv \%\t\ttrs time\ttrs Gflops\ttrs \%\n\n");
		printf("%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_sv, Gflops_sv, 100.0*Gflops_sv/Gflops_max, time_trs, Gflops_trs, 100.0*Gflops_trs/Gflops_max);

/************************************************
* return
************************************************/

		free(A);
		free(B);
		free(b);
		free(x0);
		free(BAb);
		free(BAbt);
		free(pBAbt);
		free(Q);
		free(pQ);
		free(pL);
		free(pBAbtL);
		for(jj=0; jj<N; jj++)
			{
			free(hpQ[jj]);
			free(hq[jj]);
			free(hux[jj]);
			free(hpi[jj]);
			}
		free(hpQ[N]);
		free(hq[N]);
		free(hux[N]);
		free(hpi[N]);
	


		} // increase size

	printf("\n");
	printf("\n");
	printf("\n");

	return 0;

	}



