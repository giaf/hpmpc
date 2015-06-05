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
#include <sys/time.h>
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
#include <xmmintrin.h> // needed to flush to zero sub-normals with _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON); in the main()
#endif

// to throw floating-point exception
/*#define _GNU_SOURCE*/
/*#include <fenv.h>*/

#include "test_param.h"
#include "../problem_size.h"
#include "../include/aux_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_solvers.h"
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
	printf(" Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.\n");
	printf("\n");
	printf(" HPMPC is distributed in the hope that it will be useful,\n");
	printf(" but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf(" MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
	printf(" See the GNU Lesser General Public License for more details.\n");
	printf("\n");
	printf("\n");
	printf("\n");

	printf("Riccati solver performance test - double precision\n");
	printf("\n");

	// maximum frequency of the processor
	const float GHz_max = GHZ_MAX;
	printf("Frequency used to compute theoretical peak: %5.1f GHz (edit test_param.h to modify this value).\n", GHz_max);
	printf("\n");

	// maximum flops per cycle, double precision
#if defined(TARGET_X64_AVX2)
	const float flops_max = 16;
	printf("Testing solvers for AVX & FMA3 instruction sets, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_AVX)
	const float flops_max = 8;
	printf("Testing solvers for AVX instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X64_SSE3) || defined(TARGET_AMD_SSE3)
	const float flops_max = 4;
	printf("Testing solvers for SSE3 instruction set, 64 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_CORTEX_A15)
	const float flops_max = 2;
	printf("Testing solvers for ARMv7a VFPv3 instruction set, oprimized for Cortex A15: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_CORTEX_A9)
	const float flops_max = 1;
	printf("Testing solvers for ARMv7a VFPv3 instruction set, oprimized for Cortex A9: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_CORTEX_A7)
	const float flops_max = 0.5;
	printf("Testing solvers for ARMv7a VFPv3 instruction set, oprimized for Cortex A7: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_X86_ATOM)
	const float flops_max = 1;
	printf("Testing solvers for SSE3 instruction set, 32 bit, optimized for Intel Atom: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_POWERPC_G2)
	const float flops_max = 1;
	printf("Testing solvers for POWERPC instruction set, 32 bit: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_C99_4X4)
	const float flops_max = 2;
	printf("Testing reference solvers, 4x4 kernel: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_C99_4X4_PREFETCH)
	const float flops_max = 2;
	printf("Testing reference solvers, 4x4 kernel with register prefetch: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#elif defined(TARGET_C99_2X2)
	const float flops_max = 2;
	printf("Testing reference solvers, 2x2 kernel: theoretical peak %5.1f Gflops\n", flops_max*GHz_max);
#endif
	
	FILE *f;
	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_AVX2)
	fprintf(f, "C = 'd_x64_avx2';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X64_AVX)
	fprintf(f, "C = 'd_x64_avx';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X64_SSE3) || defined(TARGET_AMD_SSE3)
	fprintf(f, "C = 'd_x64_sse3';\n");
	fprintf(f, "\n");
#elif defined(TARGET_CORTEX_A9)
	fprintf(f, "C = 'd_ARM_cortex_A9';\n");
	fprintf(f, "\n");
#elif defined(TARGET_CORTEX_A7)
	fprintf(f, "C = 'd_ARM_cortex_A7';\n");
	fprintf(f, "\n");
#elif defined(TARGET_CORTEX_A15)
	fprintf(f, "C = 'd_ARM_cortex_A15';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X86_ATOM)
	fprintf(f, "C = 'd_x86_atom';\n");
	fprintf(f, "\n");
#elif defined(TARGET_POWERPC_G2)
	fprintf(f, "C = 'd_PowerPC_G2';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_4X4)
	fprintf(f, "C = 'd_c99_4x4';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_4X4_PREFETCH)
	fprintf(f, "C = 'd_c99_4x4';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_2X2)
	fprintf(f, "C = 'd_c99_2x2';\n");
	fprintf(f, "\n");
#endif

	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
	fprintf(f, "\n");

	fprintf(f, "B = [\n");
	

	const int LTI = 1;

	printf("\n");
	printf("Tested solvers:\n");
	printf("-sv : Riccati factorization and system solution (prediction step in IP methods)\n");
	printf("-trs: system solution after a previous call to Riccati factorization (correction step in IP methods)\n");
	printf("\n");
	if(LTI==1)
		printf("\nTest for linear time-invariant systems\n");
	else
		printf("\nTest for linear time-variant systems\n");
	printf("\n");
	
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_X86_ATOM) || defined(TARGET_AMD_SSE3)
/*	printf("\nflush to zero on\n");*/
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); // flush to zero subnormals !!! works only with one thread !!!
#endif

	// to throw floating-point exception
/*#ifndef __APPLE__*/
/*    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);*/
/*#endif*/
	
	int ii, jj;

	double **dummy;

	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	int nn[] = {4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 4000, 4000, 2000, 2000, 1000, 1000, 400, 400, 400, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
	
	int vnx[] = {8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024};
	int vnrep[] = {100, 100, 100, 100, 100, 100, 50, 50, 50, 20, 10, 10};
	int vN[] = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256};

	int nx, nu, N, nrep;

	int *nx_v, *nu_v, *nb_v, *ng_v;

	int ll;
//	int ll_max = 77;
	int ll_max = 1;
	for(ll=0; ll<ll_max; ll++)
		{
		

		if(ll_max==1)
			{
			nx = NX; // number of states (it has to be even for the mass-spring system test problem)
			nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = 5;//NN; // horizon lenght
			nrep = NREP;
			//nx = 25;
			//nu = 1;
			//N = 11;
			}
		else
			{
			nx = nn[ll]; // number of states (it has to be even for the mass-spring system test problem)
			nu = 2; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = 10; // horizon lenght
			nrep = 2*nnrep[ll];
			}

		int rep;
	
		int nz = nx+nu+1;
		int anz = (nz+nal-1)/nal*nal;
		int anx = (nx+nal-1)/nal*nal;
		int pnz = (nz+bs-1)/bs*bs;
		int pnx = (nx+bs-1)/bs*bs;
		int pnu = (nu+bs-1)/bs*bs;
		int pNnu = (N*nu+bs-1)/bs*bs;
		int cnz = (nx+nu+1+ncl-1)/ncl*ncl;
		int cnx = (nx+ncl-1)/ncl*ncl;
		int cnu = (nu+ncl-1)/ncl*ncl;
		//int cNnu = ((N-1)*nu+cnu+ncl-1)/ncl*ncl;
		int cNnu = (N*nu+ncl-1)/ncl*ncl;
		int cNnx = (N*nx+ncl-1)/ncl*ncl;

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
	
//		d_print_mat(nx, nx, A, nx);
//		d_print_mat(nx, nu, B, nx);
//		d_print_mat(nx, 1, b, nx);
//		d_print_mat(nx, 1, x0, nx);
//		exit(1);

		double *pA; d_zeros_align(&pA, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, A, nx, 0, pA, cnx);
		//d_print_pmat(nx, nx, bs, pA, cnx);

		double *pAt; d_zeros_align(&pAt, pnx, cnx);
		d_cvt_tran_mat2pmat(nx, nx, A, nx, 0, pAt, cnx);
		//d_print_pmat(nx, nx, bs, pA, cnx);

		double *pBt; d_zeros_align(&pBt, pnu, cnx);
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, pBt, cnx);
		//d_print_pmat(nu, nx, bs, pBt, cnx);

/************************************************
* cost function
************************************************/

		double *Q; d_zeros(&Q, nx, nx);
		for(ii=0; ii<nx; ii++)
			Q[ii*(nx+1)] = 2.0;
		//d_print_mat(nx, nx, Q, nx);

		double *R; d_zeros(&R, nu, nu);
		for(ii=0; ii<nu; ii++)
			R[ii*(nu+1)] = 1.0;
		//d_print_mat(nu, nu, R, nu);

		double *S; d_zeros(&S, nu, nx);
		for(ii=0; ii<nu; ii++)
			S[ii*(nu+1)] = 0.0;
		//d_print_mat(nu, nx, S, nu);

		double *pQ; d_zeros_align(&pQ, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, Q, nx, 0, pQ, cnx);
		//d_print_pmat(nx, nx, bs, pQ, cnx);

		double *dQ; d_zeros_align(&dQ, pnx, 1);
		for(ii=0; ii<nx; ii++) dQ[ii] = Q[ii*(nx+1)];
		//d_print_mat(1, nx, dQ, 1);

		double *pR; d_zeros_align(&pR, pnu, cnu);
		d_cvt_mat2pmat(nu, nu, R, nu, 0, pR, cnu);
		//d_print_pmat(nu, nu, bs, pR, cnu);

		double *pS; d_zeros_align(&pS, pnu, cnx);
		d_cvt_mat2pmat(nu, nx, S, nu, 0, pS, cnx);
		//d_print_pmat(nu, nx, bs, pS, cnx);

/************************************************
* matrix series
************************************************/

		double *(hpA[N]);
		double *(hpAt[N]);
		double *(hpBt[N]);
		double *(hpGamma_u[N]);
		double *(hpGamma_u_Q[N]);
		double *(hpGamma_u_Q_A[N]);
		double *(hpGamma_0[N]);
		double *(hpGamma_0_Q[N]);
		double *(hpQ[N+1]);
		double *(hdQ[N+1]);
		double *(hpR[N]);
		double *(hpS[N]);
		double *(hpL[N+1]);
		double *pH_R;
		double *pH_Q;
		double *pH_S;
		double *work;
		double *pGamma_u;
		double *pGamma_u_Q;
		double *pGamma_u_Q_A;
		double *pGamma_0;
		double *pGamma_0_Q;

		d_zeros_align(&pH_R, pNnu, cNnu);
		d_zeros_align(&pH_Q, pnx, cnx);
		d_zeros_align(&pH_S, pnx, cNnu);
		d_zeros_align(&work, pnx, 1);
		d_zeros_align(&pGamma_u, pNnu, cNnx);
		d_zeros_align(&pGamma_u_Q, pNnu, cNnx);
		d_zeros_align(&pGamma_u_Q_A, pNnu, cNnx);
		d_zeros_align(&pGamma_0, pnx, cNnx);
		d_zeros_align(&pGamma_0_Q, pnx, cNnx);
		for(ii=0; ii<N; ii++)
			{
			hpA[ii] = pA;
			hpAt[ii] = pAt;
			hpBt[ii] = pBt;
			//d_zeros_align(&hpGamma_u[ii], ((ii+1)*nu+bs-1)/bs*bs, cnx);
			//d_zeros_align(&hpGamma_u_Q[ii], ((ii+1)*nu+bs-1)/bs*bs, cnx);
			hpGamma_u[ii] = pGamma_u+ii*nx*bs;
			hpGamma_u_Q[ii] = pGamma_u_Q+ii*nx*bs;
			hpGamma_u_Q_A[ii] = pGamma_u_Q_A+ii*nx*bs;
			//d_zeros_align(&hpGamma_0[ii], pnx, cnx);
			//d_zeros_align(&hpGamma_0_Q[ii], pnx, cnx);
			hpGamma_0[ii] = pGamma_0+ii*nx*bs;
			hpGamma_0_Q[ii] = pGamma_0_Q+ii*nx*bs;
			hpQ[ii] = pQ;
			hdQ[ii] = dQ;
			hpR[ii] = pR;
			hpS[ii] = pS;
			d_zeros_align(&hpL[ii], pnx, cnx);;
			}
		hpQ[N] = pQ;
		hdQ[N] = dQ;
		d_zeros_align(&hpL[N], pnx, cnx);;

/************************************************
* condensing
************************************************/
		
		struct timeval tv0, tv1;

		gettimeofday(&tv0, NULL); // start

		nrep = 100000;
		for(rep=0; rep<nrep; rep++)
			{

			//d_cond_Q(N, nx, nu, hpA, 0, hpQ, hpL, 1, hpGamma_0, hpGamma_0_Q, pH_Q, work);
			d_cond_Q(N, nx, nu, hpA, 1, hdQ, hpL, 1, hpGamma_0, hpGamma_0_Q, pH_Q, work);
			
			//d_cond_R(N, nx, nu, hpA, hpAt, hpBt, 0, hpQ, 0, hpS, hpR, 1, hpGamma_u, hpGamma_u_Q, pH_R);
			d_cond_R(N, nx, nu, 0, hpA, hpAt, hpBt, 1, hdQ, 0, hpS, hpR, 1, hpGamma_u, hpGamma_u_Q, hpGamma_u_Q_A, pH_R);

			d_cond_S(N, nx, nu, 0, hpS, hpGamma_0, hpGamma_u_Q, pH_S);

			}

		gettimeofday(&tv1, NULL); // start

		double time = (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);

#if 1
		for(ii=0; ii<N; ii++)	
			d_print_pmat(nx, nx, bs, hpGamma_0[ii], cNnx);

		for(ii=0; ii<N; ii++)	
			d_print_pmat(nx, nx, bs, hpGamma_0_Q[ii], cNnx);

		d_print_pmat(nx, nx, bs, pH_Q, cnx);
#endif

#if 1
		for(ii=0; ii<N; ii++)	
			d_print_pmat(nu*(ii+1), nx, bs, hpGamma_u[ii], cNnx);

		for(ii=0; ii<N; ii++)	
			d_print_pmat(nu*(ii+1), nx, bs, hpGamma_u_Q[ii], cNnx);

		d_print_pmat(N*nu, N*nu, bs, pH_R, cNnu);
#endif

#if 1
		d_print_pmat(nx, N*nu, bs, pH_S, cNnu);
#endif

		printf("\ntime = %e seconds\n", time);

/************************************************
* return
************************************************/

		free(A);
		free(B);
		free(b);
		free(x0);
		free(Q);
		free(pA);
		free(pAt);
		free(pBt);
		free(pQ);
		free(dQ);
		free(pR);
		free(pS);
		free(pH_R);
		free(pH_Q);
		free(pH_S);
		free(work);
		free(pGamma_u);
		free(pGamma_u_Q);
		free(pGamma_u_Q_A);
		free(pGamma_0);
		free(pGamma_0_Q);

		for(ii=0; ii<N; ii++)
			{
			//free(hpGamma_u[ii]);
			//free(hpGamma_u_Q[ii]);
			//free(hpGamma_0[ii]);
			//free(hpGamma_0_Q[ii]);
			free(hpL[ii]);
			}
		free(hpL[N]);

		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



