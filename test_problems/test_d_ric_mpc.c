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

	int ll;
	int ll_max = 77;
//	int ll_max = 1;
	for(ll=0; ll<ll_max; ll++)
		{
		

		if(ll_max==1)
			{
			nx = NX; // number of states (it has to be even for the mass-spring system test problem)
			nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = NN; // horizon lenght
			nrep = NREP;
			//nx = 24;
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
		int anz = nal*((nz+nal-1)/nal);
		int anx = nal*((nx+nal-1)/nal);
		int pnz = bs*((nz+bs-1)/bs);
		int pnx = bs*((nx+bs-1)/bs);
		int pnu = bs*((nu+bs-1)/bs);
		int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		int cnx = ncl*((nx+ncl-1)/ncl);
		int cnu = ncl*((nu+ncl-1)/ncl);

//		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		//const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
		int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	
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
		
		for(ii=0; ii<nx*nx; ii++) A[ii] = 0.0;
		for(ii=0; ii<nx; ii++) A[ii*(nx+1)] = 1.0;
	
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
		double *pBAbt; d_zeros_align(&pBAbt, pnz, cnx);
		d_cvt_mat2pmat(nz, nx, 0, bs, BAbt, pnz, pBAbt, cnx);

//	d_print_pmat(nz, nx, bs, pBAbt, cnx);

/************************************************
* cost function
************************************************/	

		const int ncx = nx;

		double *Q; d_zeros_align(&Q, pnz, pnz);
		for(ii=0; ii<nu; ii++) Q[ii*(pnz+1)] = 2.0;
		for(; ii<nu+ncx; ii++) Q[ii*(pnz+1)] = 1.0;
		for(ii=0; ii<nu+ncx; ii++) Q[nx+nu+ii*pnz] = 1.0;
/*		Q[(nx+nu)*(pnz+1)] = 1e35; // large enough (not needed any longer) */
		double *q; d_zeros_align(&q, pnz, 1);
		for(ii=0; ii<nu+ncx; ii++) q[ii] = Q[nx+nu+ii*pnz];

		/* packed into contiguous memory */
		double *pQ; d_zeros_align(&pQ, pnz, cnz);
		d_cvt_mat2pmat(nz, nz, 0, bs, Q, pnz, pQ, cnz);

//	d_print_pmat(nz, nz, bs, pQ, cnz);

	/* matrices series */
		double *(hpQ[N+1]);
		double *(hpL[N+1]);
		double *(hq[N+1]);
		double *(hux[N+1]);
		double *(hpi[N+1]);
		double *(hpBAbt[N]);
		double *(hrb[N]);
		double *(hrq[N+1]);
		double *(hPb[N]);
		for(jj=0; jj<N; jj++)
			{
			if(LTI==1)
				{
				hpBAbt[jj] = pBAbt; // LTI
				hpQ[jj] = pQ;
				}
			else // LTV
				{
				d_zeros_align(&hpBAbt[jj], pnz, cnx);
				for(ii=0; ii<pnz*cnx; ii++) hpBAbt[jj][ii] = pBAbt[ii];
				d_zeros_align(&hpQ[jj], pnz, cnz);
				for(ii=0; ii<pnz*cnz; ii++) hpQ[jj][ii] = pQ[ii];
				}
			d_zeros_align(&hq[jj], pnz, 1); // it has to be pnz !!!
			d_zeros_align(&hpL[jj], pnz, cnl);
			d_zeros_align(&hux[jj], pnz, 1); // it has to be pnz !!!
			d_zeros_align(&hpi[jj], pnx, 1);
			d_zeros_align(&hrb[jj], pnx, 1);
			d_zeros_align(&hrq[jj], pnz, 1);
			d_zeros_align(&hPb[jj], pnx, 1);
			}
		if(LTI==1)
			{
			hpQ[N] = pQ;
			}
		else
			{
			d_zeros_align(&hpQ[N], pnz, cnz);
			for(ii=0; ii<pnz*cnz; ii++) hpQ[N][ii] = pQ[ii]; // LTV
			}
		d_zeros_align(&hpL[N], pnz, cnl);
		d_zeros_align(&hq[N], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hux[N], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hpi[N], pnx, 1);
		d_zeros_align(&hrq[N], pnz, 1);
	
		// starting guess
		for(jj=0; jj<nx; jj++) hux[0][nu+jj]=x0[jj];
	
		double *diag; d_zeros_align(&diag, pnz, 1);
		
		//double *work; d_zeros_align(&work, 2*anz, 1);
		double *work; d_zeros_align(&work, pnz, cnx);

/************************************************
* test of riccati_eye / diag
************************************************/
		
#if 0
		
		#include "test_matrices_variable_nx.h"

		// horizon length
		N = 11;

		// base nx
		int nx0 = 2;
		int nu0 = 1;

		// size-varing
		int nxx[N+1];
		for(ii=0; ii<=N; ii++) nxx[ii] = (N+1-ii)*nx0;

		int pnxx[N+1];
		for(ii=0; ii<=N; ii++) pnxx[ii] = (nxx[ii]+bs-1)/bs*bs;

		int cnxx[N+1];
		for(ii=0; ii<=N; ii++) cnxx[ii] = (nxx[ii]+ncl-1)/ncl*ncl;

		int nuu[N];
		for(ii=0; ii<N; ii++) nuu[ii] = nu0;

		int pnuu[N];
		for(ii=0; ii<N; ii++) pnuu[ii] = (nuu[ii]+bs-1)/bs*bs;

		int cnuu[N];
		for(ii=0; ii<N; ii++) cnuu[ii] = (nuu[ii]+ncl-1)/ncl*ncl;

		//for(ii=0; ii<=N; ii++) printf("\n%d %d %d\n", nxx[ii], pnxx[ii], cnxx[ii]);
		//for(ii=0; ii<N; ii++)  printf("\n%d %d %d\n", nuu[ii], pnuu[ii], cnuu[ii]);


		// data memory space
		double *(hdA[N]);
		double *(hpBt[N]);
		double *(hpR[N]);
		double *(hpS[N]);
		double *(hpQx[N+1]);
		double *(hpLK[N]);
		double *(hpP[N+1]);
		double *pK;

		for(ii=0; ii<N; ii++)
			{
			d_zeros_align(&hdA[ii], pnxx[ii], 1);
			d_zeros_align(&hpBt[ii], pnuu[ii], cnxx[ii+1]);
			d_zeros_align(&hpR[ii], pnuu[ii], cnuu[ii]);
			d_zeros_align(&hpS[ii], pnxx[ii], cnuu[ii]);
			d_zeros_align(&hpQx[ii], pnxx[ii], cnxx[ii]);
			d_zeros_align(&hpLK[ii], pnuu[ii]+pnxx[ii], cnuu[ii]);
			d_zeros_align(&hpP[ii], pnxx[ii], cnxx[ii]);
			}
		d_zeros_align(&hpQx[N], pnxx[N], cnxx[N]);
		d_zeros_align(&hpP[N], pnxx[N], cnxx[N]);
		d_zeros_align(&pK, pnxx[0], cnuu[0]); // max(nx) x nax(nu)

		// A
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nxx[ii+1]; jj++)
				hdA[ii][jj] = 1.0;

		//d_print_mat(1, cnxx[1], hdA[0], 1);

		// B
		double *ptrB = BBB;
		for(ii=0; ii<N; ii++)
			{
			d_cvt_tran_mat2pmat(nxx[ii+1], nuu[ii], 0, bs, ptrB, nxx[ii+1], hpBt[ii], pnuu[ii]);
			ptrB += nxx[ii+1];
			}

		//d_print_pmat(pnuu[0], cnxx[1], bs, hpBt[0], cnxx[0]);
		//d_print_pmat(pnuu[1], cnxx[2], bs, hpBt[1], cnxx[1]);
		//d_print_pmat(pnuu[2], cnxx[3], bs, hpBt[2], cnxx[2]);
		//d_print_pmat(pnuu[N-1], cnxx[N], bs, hpBt[N-1], cnxx[N-1]);

		// R
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nuu[ii]; jj++)
				hpR[ii][jj/bs*bs*cnuu[ii]+jj%bs+jj*bs] = 1.0;

		//d_print_pmat(pnuu[0], cnuu[0], bs, hpR[0], pnuu[0]);

		// S

		// Q
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nxx[ii]; jj++)
				hpQx[ii][jj/bs*bs*cnxx[ii]+jj%bs+jj*bs] = 1.0;

		//d_print_pmat(pnxx[0], cnxx[0], bs, hpQx[0], cnxx[0]);
		//d_print_pmat(pnxx[1], cnxx[1], bs, hpQx[1], cnxx[1]);
		//d_print_pmat(pnxx[N-1], cnxx[N-1], bs, hpQx[N-1], cnxx[N-1]);
		//d_print_pmat(pnxx[N], cnxx[N], bs, hpQx[N], cnxx[N]);


		d_ric_diag_trf_mpc(nxx, nuu, N, hdA, hpBt, hpR, hpS, hpQx, hpLK, pK, hpP, diag);

		d_print_pmat(nxx[0], nxx[0], bs, hpP[0], cnxx[0]);
		//d_print_pmat(nxx[1], nxx[1], bs, hpP[1], cnxx[1]);
		//d_print_pmat(nxx[N-2], nxx[N-2], bs, hpP[N-2], cnxx[N-2]);
		//d_print_pmat(nxx[N-1], nxx[N-1], bs, hpP[N-1], cnxx[N-1]);
		//d_print_pmat(nxx[N], nxx[N], bs, hpP[N], cnxx[N]);

		d_print_pmat(pnuu[0]+nxx[0], nuu[0], bs, hpLK[0], cnuu[0]);
		//d_print_pmat(pnuu[1]+nxx[1], nuu[1], bs, hpLK[1], cnuu[1]);
		//d_print_pmat(pnuu[N-2]+nxx[N-2], nuu[N-2], bs, hpLK[N-2], cnuu[N-2]);
		//d_print_pmat(pnuu[N-1]+nxx[N-1], nuu[N-1], bs, hpLK[N-1], cnuu[N-1]);

		for(ii=0; ii<N; ii++)
			{
			free(hdA[ii]);
			free(hpBt[ii]);
			free(hpR[ii]);
			free(hpS[ii]);
			free(hpQx[ii]);
			free(hpLK[ii]);
			free(hpP[ii]);
			}
		free(hpQx[N]);
		free(hpP[N]);
		free(pK);




		// reference using time-invariant code
		nx = 24;
		nu = 1;
		N  = 11;

		nz  = nx+nu+1;
		pnz = (nx+nu+1+bs-1)/bs*bs;
		cnx = (nx+ncl-1)/ncl*ncl;
		cnz = (nz+ncl-1)/ncl*ncl;

		double *BAb_temp; d_zeros(&BAb_temp, nx, nu+nx+1);
		double *(hpBAbt2[N]);

		ptrB = BBB;
		for(ii=0; ii<N; ii++)
			{
			//printf("\n%d\n", ii);
			d_zeros_align(&hpBAbt2[ii], pnz, cnx);
			for(jj=0; jj<nx*(nx+nu+1); jj++) BAb_temp[jj] = 0.0;
			d_copy_mat(nxx[ii+1], nuu[ii], ptrB, nxx[ii+1], BAb_temp, nx);
			ptrB += nxx[ii+1];
			for(jj=0; jj<nxx[ii+1]; jj++) BAb_temp[nuu[ii]*nx+jj*(nx+1)] = 1.0;
			//d_print_mat(nx, nu+nx+1, BAb_temp, nx);
			d_cvt_tran_mat2pmat(nx, nx+nu+1, 0, bs, BAb_temp, nx, hpBAbt2[ii], cnx);
			//d_print_pmat(nx+nu+1, nx, bs, hpBAbt2[ii], cnx);
			}

		double *RSQ; d_zeros(&RSQ, nz, nz);
		double *(hpRSQ[N+1]);

		for(ii=0; ii<=N; ii++)
			{
			d_zeros_align(&hpRSQ[ii], pnz, cnz);
			for(jj=0; jj<nz*nz; jj++) RSQ[jj] = 0.0;
			for(jj=0; jj<nu+nxx[ii]; jj++) RSQ[jj*(nz+1)] = 1.0;
			d_cvt_mat2pmat(nz, nz, 0, bs, RSQ, nz, hpRSQ[ii], cnz);
			//d_print_pmat(nz, nz, bs, hpRSQ[ii], cnz);
			}

		d_ric_sv_mpc(nx, nu, N, hpBAbt2, hpRSQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 0);
		d_print_pmat(pnz, cnl, bs, hpL[0], cnl);
		//d_print_pmat(pnz, cnl, bs, hpL[1], cnl);
		//d_print_pmat(pnz, cnl, bs, hpL[N-2], cnl);
		//d_print_pmat(pnz, cnl, bs, hpL[N-1], cnl);
		//d_print_pmat(pnz, cnl, bs, hpL[N], cnl);
		
		free(BAb_temp);
		for(ii=0; ii<N; ii++)
			{
			free(hpBAbt2[ii]);
			free(hpRSQ[ii]);
			}
		free(hpRSQ[N]);

		exit(1);




		// problem data
		//double *dA; d_zeros_align(&dA, pnx, 1);
		//for(ii=0; ii<nx; ii++) dA[ii] = 1.0;

		//double *pBt; d_zeros_align(&pBt, pnu, cnx);
		//d_cvt_tran_mat2pmat(nx, nu, 0, bs, B, nx, pBt, cnx);
		//d_print_mat(nx, nu, B, nx);
		//d_print_pmat(nu, nx, bs, pBt, cnx);

		double *pR; d_zeros_align(&pR, pnx, cnu);
		d_cvt_tran_mat2pmat(nu, nu, 0, bs, Q, pnz, pR, cnu);
		//d_print_pmat(nu, nu, bs, pR, cnu);

		double *pQx; d_zeros_align(&pQx, pnx, cnx);
		d_cvt_tran_mat2pmat(nx, nx, 0, bs, Q+nu*(pnz+1), pnz, pQx, cnx);
		//d_print_pmat(nx, nx, bs, pQx, cnx);

		double *pS; d_zeros_align(&pS, pnu, cnx);
		//for(ii=0; ii<pnu*cnx; ii++) pS[ii] = 1.0;

		double *r; d_zeros_align(&r, pnu, 1);
		for(ii=0; ii<nu; ii++) r[ii] = Q[nx+nu+ii*pnz];

		double *qx; d_zeros_align(&qx, pnx, 1);
		for(ii=0; ii<nx; ii++) qx[ii] = Q[nx+nu+(nu+ii)*pnz];

		//double *pK; d_zeros_align(&pK, pnx, cnu);

		double *work_eye; d_zeros_align(&work_eye, pnu*cnx+pnx*cnu+pnx*cnu, 1);
		double *work_diag; d_zeros_align(&work_diag, pnx, 1);


		double *(hr[N]);
		double *(hqx[N+1]);
		double *(hu[N]);
		double *(hx[N+1]);

		for(ii=0; ii<N; ii++)
			{
			d_zeros_align(&hdA[ii], pnxx[ii], 1);
			d_zeros_aling(&hpBt[ii], pnu, cnxx[ii]);
			d_zeros_align(&hpP[ii], pnx, cnx);
			d_zeros_align(&hpLK[ii], pnu+pnx, cnu);
			hpR[ii] = pR;
			hpQx[ii] = pQx;
			hpS[ii] = pS;
			d_zeros_align(&hr[ii], pnu, 1);;
			d_zeros_align(&hqx[ii], pnx, 1);;
			d_zeros_align(&hu[ii], pnu, 1);
			d_zeros_align(&hx[ii], pnx, 1);
			}
		d_zeros_align(&hpP[N], pnx, cnx);
		hpQx[N] = pQx;
		d_zeros_align(&hqx[N], pnx, 1);;
		d_zeros_align(&hx[N], pnx, 1);

		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nu; jj++)
				hr[ii][jj] = r[jj];

		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				hqx[ii][jj] = qx[jj];

		hx[0][0] = 3.5;
		hx[0][1] = 3.5;
		for(ii=1; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				hx[ii][jj] = b[jj];

		//d_ric_eye_sv_mpc(nx, nu, N, hpBt, hpR, hpS, hpQx, hpL, hpP, work_eye, diag);
		d_ric_diag_trf_mpc(nxx, nuu, N, hdA, hpBt, hpR, hpS, hpQx, hpLK, pK, hpP, diag);

		d_print_pmat(nx, nx, bs, hpP[0], cnx);
		d_print_pmat(nx, nx, bs, hpP[1], cnx);
		d_print_pmat(nx, nx, bs, hpP[N-1], cnx);
		d_print_pmat(nx, nx, bs, hpP[N], cnx);

		d_print_pmat(pnu+nx, nu, bs, hpLK[0], cnu);
		d_print_pmat(pnu+nx, nu, bs, hpLK[1], cnu);
		d_print_pmat(pnu+nx, nu, bs, hpLK[N-1], cnu);

		d_ric_diag_trs_mpc(nxx, nuu, N, hdA, hpBt, hpLK, hpP, hr, hqx, hu, hx, work_diag, 1, hPb, 1, hpi);

		for(ii=0; ii<N; ii++)
			d_print_mat(1, nu, hu[ii], 1);

		for(ii=0; ii<N; ii++)
			d_print_mat(1, nx, hx[ii], 1);

		exit(0);
#endif
		
/************************************************
* riccati-like iteration
************************************************/
		
		// predictor

		// restore cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<pnz*cnz; jj++) hpQ[ii][jj]=pQ[jj];
			}
		for(jj=0; jj<pnz*cnz; jj++) hpQ[N][jj]=pQ[jj];

	
		// call the solver
//		if(FREE_X0==0)
			d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 0);
//		else
//			d_ric_sv_mhe_old(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, COMPUTE_MULT, hpi);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nu+nx, hux[ii], 1);
			exit(1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\npi\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hpi[ii+1], 1);
			}

		// restore linear part of cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
			}
		for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

		// residuals computation
//		if(FREE_X0==0)
			d_res_mpc(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);
//		else
//			d_res_mhe_old(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);

		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\nres\n\n");
//			if(FREE_X0==0)
//				{
				d_print_mat(1, nu, hrq[0], 1);
				for(ii=1; ii<=N; ii++)
					d_print_mat(1, nx+nu, hrq[ii], 1);
//				}
//			else
//				{
//				for(ii=0; ii<=N; ii++)
//					d_print_mat(1, nx+nu, hrq[ii], 1);
//				}
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hrb[ii], 1);
			}




		// corrector
	
		// clear solution 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nu; jj++) hux[ii][jj] = 0;
			//for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = 0;
			}

		// put b into x
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = b[jj];
			}

		// restore linear part of cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
			}
		for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

		// call the solver 
//		if(FREE_X0==0)
			d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
//		else
//			d_ric_trs_mhe_old(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi);

		if(PRINTRES==1 && ll_max==1)
			{
			// print result 
			printf("\n\ntrs\n\n");
			printf("\n\nu\n\n");
			for(ii=0; ii<=N; ii++)
				d_print_mat(1, nu, hux[ii], 1);
			printf("\n\nx\n\n");
			for(ii=0; ii<=N; ii++)
				d_print_mat(1, nx, hux[ii]+nu, 1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\npi\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hpi[ii+1], 1);
			}

		// restore linear part of cost function 
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx+nu; jj++) hq[ii][jj] = Q[nx+nu+pnz*jj];
			}
		for(jj=0; jj<nx+nu; jj++) hq[N][jj] = Q[nx+nu+pnz*jj];

		// residuals computation
//		if(FREE_X0==0)
			d_res_mpc(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);
//		else
//			d_res_mhe_old(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);

		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\nres\n\n");
//			if(FREE_X0==0)
//				{
				d_print_mat(1, nu, hrq[0], 1);
				for(ii=1; ii<=N; ii++)
					d_print_mat(1, nx+nu, hrq[ii], 1);
//				}
//			else
//				{
//				for(ii=0; ii<=N; ii++)
//					d_print_mat(1, nx+nu, hrq[ii], 1);
//				}
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hrb[ii], 1);
			}





		// timing 
		struct timeval tv0, tv1, tv2, tv3, tv4, tv5;

		// double precision
		gettimeofday(&tv0, NULL); // start

		// factorize & solve
		for(rep=0; rep<nrep; rep++)
			{
//			if(FREE_X0==0)
				d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 0);
//			else
//				d_ric_sv_mhe_old(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, COMPUTE_MULT, hpi);
			}
			
		gettimeofday(&tv1, NULL); // start

		// solve
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
//			if(FREE_X0==0)
				d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
//			else
//				d_ric_trs_mhe_old(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi);
			}
		
		gettimeofday(&tv2, NULL); // start

		// residuals
		for(rep=0; rep<nrep; rep++)
			{
//			if(FREE_X0==0)
				d_res_mpc(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);
//			else
//				d_res_mhe_old(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);
			}

		gettimeofday(&tv3, NULL); // start

		// solve for ADMM
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
//			if(FREE_X0==0)
				d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work, 0, hPb, 0, hpi, 0, 0, 0, dummy, dummy);
//			else
//				d_ric_trs_mhe_old(nx, nu, N, hpBAbt, hpL, hq, hux, work, 0, hPb, 0, hpi);
			}
		
		gettimeofday(&tv4, NULL); // start


		// factorize & solve (fast rsqrt)
		for(rep=0; rep<nrep; rep++)
			{
//			if(FREE_X0==0)
				d_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, 0, dummy, dummy, hux, hpL, work, diag, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy, 2);
//			else
//				d_ric_sv_mhe_old(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, COMPUTE_MULT, hpi);
			}
			
		gettimeofday(&tv5, NULL); // start



		float time_sv = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float flop_sv = (1.0/3.0*nx*nx*nx+3.0/2.0*nx*nx) + N*(7.0/3.0*nx*nx*nx+4.0*nx*nx*nu+2.0*nx*nu*nu+1.0/3.0*nu*nu*nu+13.0/2.0*nx*nx+9.0*nx*nu+5.0/2.0*nu*nu) - (nx*(nx+nu)+1.0/3.0*nx*nx*nx+3.0/2.0*nx*nx);
		if(COMPUTE_MULT==1)
			flop_sv += N*2*nx*nx;
		float Gflops_sv = 1e-9*flop_sv/time_sv;
	
		float time_trs = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float flop_trs = N*(6.0*nx*nx+8.0*nx*nu+2.0*nu*nu);
		if(COMPUTE_MULT==1)
			flop_trs += N*2*nx*nx;
		float Gflops_trs = 1e-9*flop_trs/time_trs;
		
		float time_trs_admm = (float) (tv4.tv_sec-tv3.tv_sec)/(nrep+0.0)+(tv4.tv_usec-tv3.tv_usec)/(nrep*1e6);
		float flop_trs_admm = N*(4.0*nx*nx+8.0*nx*nu+2.0*nu*nu);
		float Gflops_trs_admm = 1e-9*flop_trs_admm/time_trs_admm;

		float Gflops_max = flops_max * GHz_max;

		float time_sv_fast = (float) (tv5.tv_sec-tv4.tv_sec)/(nrep+0.0)+(tv5.tv_usec-tv4.tv_usec)/(nrep*1e6);
		float Gflops_sv_fast = 1e-9*flop_sv/time_sv_fast;

		if(ll==0)
			{
			printf("\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv %%\t\ttrs time\ttrs Gflops\ttrs %%\n\n");
//			fprintf(f, "\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv %%\t\ttrs time\ttrs Gflops\ttrs %%\n\n");
			}
		printf("%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_sv, Gflops_sv, 100.0*Gflops_sv/Gflops_max, time_trs, Gflops_trs, 100.0*Gflops_trs/Gflops_max, time_trs_admm, Gflops_trs_admm, 100.0*Gflops_trs_admm/Gflops_max, time_sv_fast, Gflops_sv_fast, 100.0*Gflops_sv_fast/Gflops_max);
		fprintf(f, "%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_sv, Gflops_sv, 100.0*Gflops_sv/Gflops_max, time_trs, Gflops_trs, 100.0*Gflops_trs/Gflops_max, time_trs_admm, Gflops_trs_admm, 100.0*Gflops_trs_admm/Gflops_max, time_sv_fast, Gflops_sv_fast, 100.0*Gflops_sv_fast/Gflops_max);
	

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
		free(q);
		free(diag);
		free(work);
		for(jj=0; jj<N; jj++)
			{
			if(LTI!=1)
				{
				free(hpQ[jj]);
				free(hq[jj]);
				free(hpBAbt[jj]);
				}
			free(hpL[jj]);
			free(hux[jj]);
			free(hpi[jj]);
			free(hrq[jj]);
			free(hrb[jj]);
			free(hPb[jj]);
			}
		if(LTI!=1)
			{
			free(hpQ[N]);
			free(hq[N]);
			}
		free(hpL[N]);
		free(hux[N]);
		free(hpi[N]);
		free(hrq[N]);
	


		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



