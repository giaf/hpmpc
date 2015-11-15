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
#include "../include/blas_d.h"
#include "../include/lqcp_solvers.h"
#include "../include/mpc_solvers.h"
#include "../include/block_size.h"
#include "../include/reference_code.h"
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
#if 1
	int ll_max = 77;
#else
	int ll_max = 1;
#endif
	for(ll=0; ll<ll_max; ll++)
		{
		

		if(ll_max==1)
			{
			nx = NX; // number of states (it has to be even for the mass-spring system test problem)
			nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = NN; // horizon lenght
			nrep = NREP;
			//nx = 25;
			//nu = 1;
			//N = 11;
			}
		else
			{
			nx = nn[ll]; // number of states (it has to be even for the mass-spring system test problem)
			nu = nx/2; //2; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = 10; // horizon lenght
			nrep = 2*nnrep[ll];
//			nrep = nnrep[ll]/4;
			}



		// define time-varian problem size
		nx_v = (int *) malloc((N+1)*sizeof(int));
		nx_v[0] = 0;
		for(ii=1; ii<N; ii++) nx_v[ii] = nx;
		nx_v[N] = nx;

		nu_v = (int *) malloc((N+1)*sizeof(int));
		nu_v[0] = nu;
		for(ii=1; ii<N; ii++) nu_v[ii] = nu;
		nu_v[N] = 0;

		nb_v = (int *) malloc((N+1)*sizeof(int));
		for(ii=0; ii<=N; ii++) nb_v[ii] = 0;

		ng_v = (int *) malloc((N+1)*sizeof(int));
		for(ii=0; ii<=N; ii++) ng_v[ii] = 0;



		int rep;
	
		int nz = nx+nu+1;
		int anz = nal*((nz+nal-1)/nal);
		int anx = nal*((nx+nal-1)/nal);
		int pnz = bs*((nz+bs-1)/bs);
		int pnx = bs*((nx+bs-1)/bs);
		int pnx1 = bs*((nx+1+bs-1)/bs);
		int pnu = bs*((nu+bs-1)/bs);
		int pnu1 = bs*((nu+1+bs-1)/bs);
		int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		int cnx = ncl*((nx+ncl-1)/ncl);
		int cnx1 = ncl*((nx+1+ncl-1)/ncl);
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
			b[jj] = 0.0;
	
		for(jj=0; jj<nx; jj++)
			x0[jj] = 0;
		x0[0] = 3.5;
		x0[1] = 3.5;
	
//	d_print_mat(nx, nx, A, nx);
//	d_print_mat(nx, nu, B, nx);
//	d_print_mat(nx, 1, b, nx);
//	d_print_mat(nx, 1, x0, nx);
		
		//for(ii=0; ii<nx*nx; ii++) A[ii] = 0.0;
		//for(ii=0; ii<nx; ii++) A[ii*(nx+1)] = 1.0;
	
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
		d_cvt_mat2pmat(nz, nx, BAbt, pnz, 0, pBAbt, cnx);

//	d_print_pmat(nz, nx, bs, pBAbt, cnx);

		// matrices for size-variant solver
		double *pBAbt0; d_zeros_align(&pBAbt0, pnu1, cnx);
		d_cvt_mat2pmat(nu, nx, BAbt, pnz, 0, pBAbt0, cnx);
		d_cvt_mat2pmat(1, nx, BAbt+nu+nx, pnz, nu, pBAbt0+nu/bs*bs*cnx+nu%bs, cnx);
		double *pA; d_zeros_align(&pA, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, A, nx, 0, pA, cnx);
		double *b0; d_zeros_align(&b0, pnx, 1);
		dgemv_n_lib(nx, nx, pA, cnx, x0, 1, b, b0);
		d_copy_mat(1, nx, b0, 1, pBAbt0+nu/bs*bs*cnx+nu%bs, bs);
		double *pBAbt1; d_zeros_align(&pBAbt1, pnz, cnx);
		d_cvt_mat2pmat(nz, nx, BAbt, pnz, 0, pBAbt1, cnx);

//		d_print_pmat(nu+1, nx, bs, pBAbt0, cnx);
//		d_print_pmat(nu+nx+1, nx, bs, pBAbt1, cnx);
//		d_print_pmat(nx, nx, bs, pA, cnx);
//		d_print_mat(1, nx, b0, 1);
//		exit(2);
		
#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_MKL) || defined(REF_BLAS_NETLIB)
		
		double *BAbt0; d_zeros(&BAbt0, nu+1, nx);
		d_tran_mat(nx, nu, B, nx, BAbt0, nu+1);
		d_tran_mat(nx, 1, b0, nx, BAbt0+nu, nu+1);
		double *BAbt1; d_zeros(&BAbt1, nu+nx+1, nx);
		d_tran_mat(nx, nu, B, nx, BAbt1, nu+nx+1);
		d_tran_mat(nx, nx, A, nx, BAbt1+nu, nu+nx+1);
		d_tran_mat(nx, 1, b, nx, BAbt1+nu+nx, nu+nx+1);

//		d_print_mat(nu+1, nx, BAbt0, nu+1);
//		d_print_mat(nu+nx+1, nx, BAbt1, nu+nx+1);
//		exit(1);

#endif

/************************************************
* cost function
************************************************/	

		const int ncx = nx;

		double *Q; d_zeros_align(&Q, pnz, pnz);
		for(ii=0; ii<nu; ii++) Q[ii*(pnz+1)] = 2.0;
		for(; ii<nu+ncx; ii++) Q[ii*(pnz+1)] = 1.0;
		for(ii=0; ii<nu; ii++) Q[nx+nu+ii*pnz] = 0.0;
		for(; ii<nu+ncx; ii++) Q[nx+nu+ii*pnz] = 0.0;
/*		Q[(nx+nu)*(pnz+1)] = 1e35; // large enough (not needed any longer) */
		double *q; d_zeros_align(&q, pnz, 1);
		for(ii=0; ii<nu+ncx; ii++) q[ii] = Q[nx+nu+ii*pnz];

		/* packed into contiguous memory */
		double *pQ; d_zeros_align(&pQ, pnz, cnz);
		d_cvt_mat2pmat(nz, nz, Q, pnz, 0, pQ, cnz);

		// matrices for size-variant solver
//		int cnu = (nu+ncl-1)/ncl*ncl;
//		int cnx = (nx+ncl-1)/ncl*ncl;
		int cnux = (nu+nx+ncl-1)/ncl*ncl;
		int pnux = (nu+nx+bs-1)/bs*bs;
		double *pQ0; d_zeros_align(&pQ0, pnu1, cnu);
		d_cvt_mat2pmat(nu, nu, Q, pnz, 0, pQ0, cnu);
		d_cvt_mat2pmat(1, nu, Q+nu+nx, pnz, nu, pQ0+nu/bs*bs*cnu+nu%bs, cnu);
		double *pQ1; d_zeros_align(&pQ1, pnz, cnux);
		d_cvt_mat2pmat(nu+nx+1, nu+nx, Q, pnz, 0, pQ1, cnux);
		double *pQN; d_zeros_align(&pQN, pnx1, cnx);
		d_cvt_mat2pmat(nx, nx, Q+nu*(pnz+1), pnz, 0, pQN, cnx);
		d_cvt_mat2pmat(1, nx, Q+nu*(pnz+1)+nx, pnz, nx, pQN+nx/bs*bs*cnx+nx%bs, cnx);

		double *q1; d_zeros_align(&q1, pnux, 1);

//		d_print_pmat(nu+1, nu, bs, pQ0, cnu);
//		d_print_pmat(nu+nx+1, nu+nx, bs, pQ1, cnux);
//		d_print_pmat(nx+1, nx, bs, pQN, cnx);
//		exit(2);

#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_MKL) || defined(REF_BLAS_NETLIB)

		double *Q0; d_zeros(&Q0, nu+1, nu+1);
		d_copy_mat(nu, nu, Q, pnz, Q0, nu+1);
		d_copy_mat(1, nu, Q+nu+nx, pnz, Q0+nu, nu+1);
		Q0[nu*(nu+2)] = 1e35; // large enough
		double *Q1; d_zeros(&Q1, nu+nx+1, nu+nx+1);
		d_copy_mat(nu+nx+1, nu+nx, Q, pnz, Q1, nu+nx+1);
		Q1[(nu+nx)*(nu+nx+2)] = 1e35; // large enough
		double *QN; d_zeros(&QN, nx+1, nx+1);
		d_copy_mat(nx, nx, Q+nu*(pnz+1), pnz, QN, nx+1);
		d_copy_mat(1, nx, Q+nu*(pnz+1)+nx, pnz, QN+nx, nx+1);
		QN[(nx)*(nx+2)] = 1e35; // large enough

//		d_print_mat(nu+1, nu+1, Q0, nu+1);
//		d_print_mat(nu+nx+1, nu+nx+1, Q1, nu+nx+1);
//		d_print_mat(nx+1, nx+1, QN, nx+1);
//		exit(2);

#endif

//	d_print_pmat(nz, nz, bs, pQ, cnz);

	/* matrices series */
		double *(hpQ[N+1]);
		double *(hpQ_tv[N+1]);
		double *(hpL[N+1]);
		double *(hdL[N+1]);
		double *(hl[N+1]);
		double *(hq[N+1]);
		double *(hq_tv[N+1]);
		double *(hux[N+1]);
		double *(hpi[N+1]);
		double *(hpBAbt[N]);
		double *(hpBAbt_tv[N]);
		double *(hb_tv[N]);
		double *(hrb[N]);
		double *(hrq[N+1]);
		double *(hPb[N]);
		for(jj=0; jj<N; jj++)
			{
			if(LTI==1)
				{
				hpBAbt[jj] = pBAbt; // LTI
				hpQ[jj] = pQ;
//				hpQ_tv[jj] = pQ;
				}
			else // LTV
				{
				d_zeros_align(&hpBAbt[jj], pnz, cnx);
				for(ii=0; ii<pnz*cnx; ii++) hpBAbt[jj][ii] = pBAbt[ii];
				d_zeros_align(&hpQ[jj], pnz, cnz);
				for(ii=0; ii<pnz*cnz; ii++) hpQ[jj][ii] = pQ[ii];
//				d_zeros_align(&hpQ_tv[jj], pnz, cnz);
//				for(ii=0; ii<pnz*cnz; ii++) hpQ_tv[jj][ii] = pQ[ii];
				}
			d_zeros_align(&hq[jj], pnz, 1); // it has to be pnz !!!
			d_zeros_align(&hpL[jj], pnz, cnl);
			d_zeros_align(&hdL[jj], pnz, 1);
			d_zeros_align(&hl[jj], pnz, 1);
			d_zeros_align(&hux[jj], pnz, 1); // it has to be pnz !!!
			d_zeros_align(&hpi[jj], pnx, 1);
			d_zeros_align(&hrb[jj], pnx, 1);
			d_zeros_align(&hrq[jj], pnz, 1);
			d_zeros_align(&hPb[jj], pnx, 1);
			}
		if(LTI==1)
			{
			hpQ[N] = pQ;
//			hpQ_tv[N] = pQ_N;
			}
		else
			{
			d_zeros_align(&hpQ[N], pnz, cnz);
			for(ii=0; ii<pnz*cnz; ii++) hpQ[N][ii] = pQ[ii]; // LTV
			d_zeros_align(&hpQ_tv[N], pnx1, cnx1);
//			for(ii=0; ii<pnx1*cnx1; ii++) hpQ_tv[N][ii] = pQ_N[ii]; // LTV
			}
		d_zeros_align(&hpL[N], pnz, cnl);
		d_zeros_align(&hdL[N], pnz, 1);
		d_zeros_align(&hl[N], pnz, 1);
		d_zeros_align(&hq[N], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hux[N], pnz, 1); // it has to be pnz !!!
		d_zeros_align(&hpi[N], pnx, 1);
		d_zeros_align(&hrq[N], pnz, 1);
		// size-variant matrices
		if(LTI==1)
			{
			hpQ_tv[0] = pQ0;
			hq_tv[0] = q1;
			hpBAbt_tv[0] = pBAbt0;
			hb_tv[0] = b0;
			for(ii=1; ii<N; ii++)
				{
				hpQ_tv[ii] = pQ1;
				hpBAbt_tv[ii] = pBAbt1;
				hb_tv[ii] = b;
				hq_tv[ii] = q1;
				}
			hpQ_tv[N] = pQN;
			hq_tv[N] = q1;
			}
		else
			{
			// TODO
			}
	
		// starting guess
		for(jj=0; jj<nx; jj++) hux[0][nu+jj]=x0[jj];
	
		double *work1; d_zeros_align(&work1, pnz, 2);
		
		//double *work; d_zeros_align(&work, 2*anz, 1);
		double *work0; d_zeros_align(&work0, pnz, cnx);

#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_MKL) || defined(REF_BLAS_NETLIB)

		double *(hBAbt[N]);
		double *(hQ[N+1]);
		double *(hL[N+1]);
		hBAbt[0] = BAbt0;
		hQ[0] = Q0;
		d_zeros(&hL[0], nu+1, nu+1);
		for(ii=1; ii<N; ii++)
			{
			hBAbt[ii] = BAbt1;
			hQ[ii] = Q1;
			d_zeros(&hL[ii], nu+nx+1, nu+nx+1);
			}
		hQ[N] = QN;
		d_zeros(&hL[N], nx+1, nx+1);

		double *BAbtL; d_zeros(&BAbtL, nu+nx+1, nx);

#endif

/************************************************
* test of riccati_eye / diag
************************************************/
		
#if 0
		float time_ip_diag, time_sv_diag, time_sv_full;

		#include "test_matrices_variable_nx.h"

		// horizon length
		N = 11;

		// base nx
		int nx0 = 2;
		int nu0 = 1;

		// size-varing
		int nxx[N+1];
		for(ii=0; ii<=N; ii++) nxx[ii] = (N+1-ii)*nx0 + nu0;

		int pnxx[N+1];
		for(ii=0; ii<=N; ii++) pnxx[ii] = (nxx[ii]+bs-1)/bs*bs;

		int cnxx[N+1];
		for(ii=0; ii<=N; ii++) cnxx[ii] = (nxx[ii]+ncl-1)/ncl*ncl;

		int nuu[N+1];
		for(ii=0; ii<N; ii++) nuu[ii] = nu0;
		nuu[N] = 0; // !!!!!

		int pnuu[N+1];
		for(ii=0; ii<N; ii++) pnuu[ii] = (nuu[ii]+bs-1)/bs*bs;
		pnuu[N] = 0; // !!!!!

		int cnuu[N+1];
		for(ii=0; ii<N; ii++) cnuu[ii] = (nuu[ii]+ncl-1)/ncl*ncl;
		cnuu[N] = 0; // !!!!!

		//for(ii=0; ii<=N; ii++) printf("\n%d %d %d\n", nxx[ii], pnxx[ii], cnxx[ii]);
		//for(ii=0; ii<N; ii++)  printf("\n%d %d %d\n", nuu[ii], pnuu[ii], cnuu[ii]);


		// data memory space
		double *(hdA[N]);
		double *(hpBt[N]);
		double *(hpR[N]);
		double *(hpS[N]);
		double *(hpQ2[N+1]);
		double *(hpLK[N]);
		double *(hpP[N+1]);
		double *pK;

		for(ii=0; ii<N; ii++)
			{
			d_zeros_align(&hdA[ii], pnxx[ii], 1);
			d_zeros_align(&hpBt[ii], pnuu[ii], cnxx[ii+1]);
			d_zeros_align(&hpR[ii], pnuu[ii], cnuu[ii]);
			d_zeros_align(&hpS[ii], pnxx[ii], cnuu[ii]);
			d_zeros_align(&hpQ2[ii], pnxx[ii], cnxx[ii]);
			d_zeros_align(&hpLK[ii], pnuu[ii]+pnxx[ii], cnuu[ii]);
			d_zeros_align(&hpP[ii], pnxx[ii], cnxx[ii]);
			}
		d_zeros_align(&hpQ2[N], pnxx[N], cnxx[N]);
		d_zeros_align(&hpP[N], pnxx[N], cnxx[N]);
		d_zeros_align(&pK, pnxx[0], cnuu[0]); // max(nx) x nax(nu)

		// A
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nxx[ii+1]; jj++)
				hdA[ii][jj] = 1.0;

		//d_print_mat(1, cnxx[1], hdA[0], 1);

		// B
		double *eye_nu0; d_zeros(&eye_nu0, nu0, nu0);
		for(jj=0; jj<nu0; jj++) eye_nu0[jj*(nu0+1)] = 1.0;
		double *ptrB = BBB;
		for(ii=0; ii<N; ii++)
			{
			d_cvt_mat2pmat(nuu[ii], nuu[ii], eye_nu0, nuu[ii], 0, hpBt[ii], cnxx[ii]);
			d_cvt_tran_mat2pmat(nxx[ii+1]-nuu[ii], nuu[ii], ptrB, nxx[ii+1]-nuu[ii], 0, hpBt[ii]+nuu[ii]*bs, cnxx[ii]);
			ptrB += nxx[ii+1] - nuu[ii];
			}
		free(eye_nu0);

		//d_print_pmat(pnuu[0], cnxx[1], bs, hpBt[0], cnxx[0]);
		//d_print_pmat(pnuu[1], cnxx[2], bs, hpBt[1], cnxx[1]);
		//d_print_pmat(pnuu[2], cnxx[3], bs, hpBt[2], cnxx[2]);
		//d_print_pmat(pnuu[N-1], cnxx[N-1], bs, hpBt[N-2], cnxx[N-2]);
		//d_print_pmat(pnuu[N-1], cnxx[N], bs, hpBt[N-1], cnxx[N-1]);

		// R
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nuu[ii]; jj++)
				hpR[ii][jj/bs*bs*cnuu[ii]+jj%bs+jj*bs] = 0.0;

		//for(ii=0; ii<N; ii++)
		//	d_print_pmat(pnuu[ii], cnuu[ii], bs, hpR[ii], pnuu[ii]);
		//d_print_pmat(pnuu[0], cnuu[0], bs, hpR[0], pnuu[0]);

		// S

		// Q
		for(ii=0; ii<=N; ii++)
			{
			for(jj=0; jj<nu0; jj++)
				hpQ2[ii][jj/bs*bs*cnxx[ii]+jj%bs+jj*bs] = 1.0;
			for(jj=nxx[ii]-nx0; jj<nxx[ii]; jj++) 
				hpQ2[ii][jj/bs*bs*cnxx[ii]+jj%bs+jj*bs] = 1.0;
			}

		//for(ii=0; ii<=N; ii++)
		//	d_print_pmat(pnxx[ii], cnxx[ii], bs, hpQ2[ii], cnxx[ii]);
		//d_print_pmat(pnxx[0], cnxx[0], bs, hpQ2[0], cnxx[0]);
		//d_print_pmat(pnxx[1], cnxx[1], bs, hpQ2[1], cnxx[1]);
		//d_print_pmat(pnxx[N-1], cnxx[N-1], bs, hpQ2[N-1], cnxx[N-1]);
		//d_print_pmat(pnxx[N], cnxx[N], bs, hpQ2[N], cnxx[N]);
		//exit(1);


		double **ptr_dummy;


		printf("\nfactorize\n");
		d_ric_diag_trf_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hpLK, pK, hpP, diag, 0, ptr_dummy);
		printf("\nfactorize done\n");

#if 1
		//d_print_pmat(nxx[0], nxx[0], bs, hpP[0], cnxx[0]);
		//d_print_pmat(nxx[1], nxx[1], bs, hpP[1], cnxx[1]);
		//d_print_pmat(nxx[N-2], nxx[N-2], bs, hpP[N-2], cnxx[N-2]);
		//d_print_pmat(nxx[N-1], nxx[N-1], bs, hpP[N-1], cnxx[N-1]);
		//d_print_pmat(nxx[N], nxx[N], bs, hpP[N], cnxx[N]);

		//for(ii=0; ii<=N; ii++)
		//	d_print_pmat(pnuu[ii]+nxx[ii], nuu[ii], bs, hpLK[ii], cnuu[ii]);
		//d_print_pmat(pnuu[0]+nxx[0], nuu[0], bs, hpLK[0], cnuu[0]);
		//d_print_pmat(pnuu[1]+nxx[1], nuu[1], bs, hpLK[1], cnuu[1]);
		//d_print_pmat(pnuu[2]+nxx[2], nuu[2], bs, hpLK[2], cnuu[2]);
		//d_print_pmat(pnuu[N-3]+nxx[N-3], nuu[N-3], bs, hpLK[N-3], cnuu[N-3]);
		//d_print_pmat(pnuu[N-2]+nxx[N-2], nuu[N-2], bs, hpLK[N-2], cnuu[N-2]);
		//d_print_pmat(pnuu[N-1]+nxx[N-1], nuu[N-1], bs, hpLK[N-1], cnuu[N-1]);
#endif



		// data memory space
		double *(hrq2[N+1]);
		//double *(hu2[N]);
		//double *(hx2[N+1]);
		double *(hux2[N+1]);
		double *(hpi2[N+1]);
		double *(hPb2[N]);

		double *(hb2[N]);
		//double *(hres_r2[N]);
		//double *(hres_q2[N+1]);
		double *(hres_rq2[N+1]);
		double *(hres_b2[N]);

		for(ii=0; ii<N; ii++)
			{
			d_zeros_align(&hrq2[ii], pnuu[ii]+pnxx[ii], 1);
			//d_zeros_align(&hu2[ii], pnuu[ii], 1);
			//d_zeros_align(&hx2[ii], pnxx[ii], 1);
			d_zeros_align(&hux2[ii], pnuu[ii]+pnxx[ii], 1);
			d_zeros_align(&hpi2[ii], pnxx[ii], 1);
			d_zeros_align(&hPb2[ii], pnxx[ii+1], 1);

			d_zeros_align(&hb2[ii], pnxx[ii+1], 1);
			//d_zeros_align(&hres_r2[ii], pnuu[ii], 1);
			//d_zeros_align(&hres_q2[ii], pnxx[ii], 1);
			d_zeros_align(&hres_rq2[ii], pnuu[ii]+pnxx[ii], 1);
			d_zeros_align(&hres_b2[ii], pnxx[ii+1], 1);
			}
		d_zeros_align(&hrq2[N], pnuu[N]+pnxx[N], 1);
		//d_zeros_align(&hx2[N], pnxx[N], 1);
		d_zeros_align(&hux2[N], pnuu[N]+pnxx[N], 1);
		d_zeros_align(&hpi2[N], pnxx[N], 1);

		d_zeros_align(&hres_rq2[N], pnuu[N]+pnxx[N], 1);

		double *work_diag2; d_zeros_align(&work_diag2, pnxx[0], 1);

		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nuu[ii]; jj++)
				hrq2[ii][jj] = 0.0;

		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nxx[ii]; jj++)
				hrq2[ii][nuu[ii]+jj] = 0.0;

		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nxx[ii+1]; jj++)
				hb2[ii][jj] = 0.0;

		// x0
		//hux2[0][2*nuu[0]+0] =  5.0;
		//hux2[0][2*nuu[0]+1] = -5.0;
		for(jj=0; jj<nuu[0]; jj++)
			{
			hux2[0][jj] = 0.0;
			}
		for(; jj<nuu[0]+nu; jj++)
			{
			hux2[0][jj] = 7.5097;
			}
		for(; jj<nxx[0]; jj+=2)
			{
			hux2[0][jj+0] = 15.01940;
			hux2[0][jj+1] =  0.0;
			}
		d_print_mat(1, nuu[0]+nxx[0], hux2[0], 1);


		printf("\nsolve\n");
		d_ric_diag_trs_mpc(N, nxx, nuu, hdA, hpBt, hpLK, hpP, hb2, hrq2, hux2, 1, hPb2, 1, hpi2, work_diag2);
		printf("\nsolve done\n");

#if 1
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nuu[ii]+nxx[ii], hux2[ii], 1);
#endif



		// residuals

#if 1
		printf("\nresudyals\n");
		d_res_diag_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hb2, hrq2, hux2, hpi2, hres_rq2, hres_b2, work_diag2);
		printf("\nresiduals done\n");

		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nuu[ii]+nxx[ii], hres_rq2[ii], 1);

		for(ii=0; ii<N; ii++)
			d_print_mat(1, nxx[ii+1], hres_b2[ii], 1);
#endif





		// timing
		struct timeval tv20, tv21;

#if 1
		gettimeofday(&tv20, NULL); // start

		nrep = 10000;
		for(ii=0; ii<nrep; ii++)
			{
			d_ric_diag_trf_mpc(N, nxx, nuu, hdA, hpBt, hpR, hpS, hpQ2, hpLK, pK, hpP, diag, 0, ptr_dummy);
			d_ric_diag_trs_mpc(N, nxx, nuu, hdA, hpBt, hpLK, hpP, hb2, hrq2, hux2, 1, hPb2, 1, hpi2, work_diag2);
			}

		gettimeofday(&tv21, NULL); // start

		time_sv_diag = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);
#endif





#if 1
		// IPM
		int kk = -1;
		int kmax = 50;
		double mu0 = 1;
		double mu_tol = 1e-8;
		double alpha_min = 1e-12;
		double sigma_par[] = {0.4, 0.3, 0.01};
		double stat[5*50] = {};

		int nbb[N+1];
		for(ii=0; ii<=N; ii++) nbb[ii] = nuu[ii] + nxx[ii];
		nbb[0] = nuu[0]; // XXX !!!!!!!!!!!!!!

		int pnbb[N+1];
		for(ii=0; ii<=N; ii++) pnbb[ii] = (nbb[ii]+bs-1)/bs*bs;

		double *(hd2[N+1]);
		double *(hlam2[N+1]);
		double *(ht2[N+1]);
		double *(hres_d2[N+1]);
		for(ii=0; ii<=N; ii++)
			{
			d_zeros_align(&hd2[ii], 2*pnbb[ii], 1);
			d_zeros_align(&hlam2[ii], 2*pnbb[ii], 1);
			d_zeros_align(&ht2[ii], 2*pnbb[ii], 1);
			d_zeros_align(&hres_d2[ii], 2*pnbb[ii], 1);
			}

		double mu2 = -1;

		printf("\nbounds\n");
		ii = 0; // initial stage
		for(jj=0; jj<nuu[ii]; jj++)
			{
			hd2[ii][jj]                  = -20.5;
			hd2[ii][pnbb[ii]+jj]         = -20.5;
			}
		d_print_mat(1, 2*pnbb[ii], hd2[ii], 1);
		for(ii=1; ii<=N; ii++)
			{
			for(jj=0; jj<nuu[ii]; jj++)
				{
				hd2[ii][jj]          = -20.5;
				hd2[ii][pnbb[ii]+jj] = -20.5;
				}
			for(; jj<nuu[ii]+nu0; jj++)
				{
				hd2[ii][jj]          = - 2.5;
				hd2[ii][pnbb[ii]+jj] = -10.0;
				}
			for(; jj<nbb[ii]-nx0; jj++)
			//for(; jj<nbb[ii]; jj++)
				{
				hd2[ii][jj]          = -100.0;
				hd2[ii][pnbb[ii]+jj] = -100.0;
				}
			hd2[ii][jj+0]          = - 0.0; //   0
			hd2[ii][pnbb[ii]+jj+0] = -20.0; // -20
			hd2[ii][jj+1]          = -10.0; // -10
			hd2[ii][pnbb[ii]+jj+1] = -10.0; // -10
			d_print_mat(1, 2*pnbb[ii], hd2[ii], 1);
			}

		for(jj=0; jj<nuu[0]; jj++)
			{
			hux2[0][jj] = 0.0;
			}
		for(; jj<nuu[0]+nu; jj++)
			{
			hux2[0][jj] = 7.5097;
			}
		for(; jj<nxx[0]; jj+=2)
			{
			hux2[0][jj+0] = 15.01940;
			hux2[0][jj+1] =  0.0;
			}
		d_print_mat(1, nuu[0]+nxx[0], hux2[0], 1);


		int pnxM = pnxx[0];
		int pnuM = pnuu[0];
		int cnuM = cnuu[0];

		int anxx[N+1];
		for(ii=0; ii<=N; ii++) anxx[ii] = (nxx[ii]+nal-1)/nal*nal;

		int anuu[N+1];
		for(ii=0; ii<=N; ii++) anuu[ii] = (nuu[ii]+nal-1)/nal*nal;

		int work_space_ip_double = 0;
		for(ii=0; ii<=N; ii++)
			work_space_ip_double += anuu[ii] + 3*anxx[ii] + (pnuu[ii]+pnxx[ii])*cnuu[ii] + pnxx[ii]*cnxx[ii] + 3*pnxx[ii] + 3*pnuu[ii] + 8*pnbb[ii];
		work_space_ip_double += pnxM*cnuM + pnxM + pnuM;
		int work_space_ip_int = (N+1)*7*sizeof(int);
		work_space_ip_int = (work_space_ip_int+63)/64*64;
		work_space_ip_int /= sizeof(int);
		printf("\nwork space: %d double + %d int\n", work_space_ip_double, work_space_ip_int);
		double *work_space_ip; d_zeros_align(&work_space_ip, work_space_ip_double+(work_space_ip_int+1)/2, 1); // XXX assume sizeof(double) = 2 * sizeof(int) !!!!!


		printf("\nIPM\n");
		d_ip2_diag_mpc(&kk, kmax, mu0, mu_tol, alpha_min, 0, sigma_par, stat, N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hd2, hrq2, hux2, 1, hpi2, hlam2, ht2, work_space_ip);
		printf("\nIPM done\n");


		printf("\nux\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nuu[ii]+nxx[ii], hux2[ii], 1);

		printf("\nlam\n");
		for(ii=0; ii<=N; ii++)
			{
			d_print_mat(1, nbb[ii], hlam2[ii], 1);
			d_print_mat(1, nbb[ii], hlam2[ii]+pnbb[ii], 1);
			}

		printf("\nt\n");
		for(ii=0; ii<=N; ii++)
			{
			d_print_mat(1, nbb[ii], ht2[ii], 1);
			d_print_mat(1, nbb[ii], ht2[ii]+pnbb[ii], 1);
			}

		printf("\nstat\n\n");
		for(ii=0; ii<kk; ii++)
			printf("%f %f %f %f %f\n", stat[5*ii+0], stat[5*ii+1], stat[5*ii+2], stat[5*ii+3], stat[5*ii+4]);
		printf("\n\n");


		// residuals
		printf("\nresuduals IPM\n");
		d_res_ip_diag_mpc(N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hrq2, hd2, hux2, hpi2, hlam2, ht2, hres_rq2, hres_b2, hres_d2, &mu2, work_diag2);
		printf("\nresiduals IPM done\n");

		printf("\nres_rq\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nuu[ii]+nxx[ii], hres_rq2[ii], 1);

		printf("\nres_b\n");
		for(ii=0; ii<N; ii++)
			d_print_mat(1, nxx[ii+1], hres_b2[ii], 1);

		printf("\nres_d\n");
		for(ii=0; ii<=N; ii++)
			{
			d_print_mat(1, nbb[ii], hres_d2[ii], 1);
			d_print_mat(1, nbb[ii], hres_d2[ii]+pnbb[ii], 1);
			}

		printf("\nres_mu\n");
		d_print_mat(1, 1, &mu2, 1);


		// timing
		gettimeofday(&tv20, NULL); // start

		for(ii=0; ii<nrep; ii++)
			{
			d_ip2_diag_mpc(&kk, kmax, mu0, mu_tol, alpha_min, 0, sigma_par, stat, N, nxx, nuu, nbb, hdA, hpBt, hpR, hpS, hpQ2, hb2, hd2, hrq2, hux2, 1, hpi2, hlam2, ht2, work_space_ip);
			}

		gettimeofday(&tv21, NULL); // start

		time_ip_diag = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);


		for(ii=0; ii<=N; ii++)
			{
			free(hd2[ii]);
			free(hlam2[ii]);
			free(ht2[ii]);
			}
		free(work_space_ip);
#endif





		for(ii=0; ii<N; ii++)
			{
			free(hdA[ii]);
			free(hpBt[ii]);
			free(hpR[ii]);
			free(hpS[ii]);
			free(hpQ2[ii]);
			free(hpLK[ii]);
			free(hpP[ii]);
			free(hrq2[ii]);
			//free(hx2[ii]);
			//free(hu2[ii]);
			free(hux2[ii]);
			free(hpi2[ii]);
			free(hPb2[ii]);

			free(hb2[ii]);
			//free(hres_r2[ii]);
			//free(hres_q2[ii]);
			free(hres_rq2[ii]);
			free(hres_b2[ii]);
			}
		free(hpQ2[N]);
		free(hpP[N]);
		free(pK);
		free(hrq2[N]);
		//free(hx2[N]);
		free(hux2[N]);
		free(hpi2[N]);
		free(work_diag2);

		free(hres_rq2[N]);



		// reference using time-invariant code
		nx = 25;
		nu = 1;
		N  = 11;

		nz  = nx+nu+1;
		pnx = (nx+bs-1)/bs*bs;
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
			for(jj=0; jj<nu; jj++) BAb_temp[jj*(nx+1)] = 1.0;
			d_copy_mat(nxx[ii+1]-1, nuu[ii], ptrB, nxx[ii+1]-1, BAb_temp+1, nx);
			ptrB += nxx[ii+1]-1;
			for(jj=0; jj<nxx[ii+1]; jj++) BAb_temp[nuu[ii]*nx+jj*(nx+1)] = 1.0;
			//for(jj=0; jj<nxx[ii+1]; jj++) BAb_temp[(nuu[ii]+nxx[ii+1])*nx+jj] = 1.0;
			//d_print_mat(nx, nu+nx+1, BAb_temp, nx);
			d_cvt_tran_mat2pmat(nx, nx+nu+1, BAb_temp, nx, 0, hpBAbt2[ii], cnx);
			//d_print_pmat(nx+nu+1, nx, bs, hpBAbt2[ii], cnx);
			}

		double *RSQ; d_zeros(&RSQ, nz, nz);
		double *(hpRSQ[N+1]);

		for(ii=0; ii<=N; ii++)
			{
			//printf("\n%d\n", ii);
			d_zeros_align(&hpRSQ[ii], pnz, cnz);
			for(jj=0; jj<nz*nz; jj++) RSQ[jj] = 0.0;
			for(jj=nu; jj<2*nu; jj++) RSQ[jj*(nz+1)] = 1.0;
			for(jj=nu+nxx[ii]-nx0; jj<nu+nxx[ii]; jj++) RSQ[jj*(nz+1)] = 1.0;
			d_cvt_mat2pmat(nz, nz, RSQ, nz, 0, hpRSQ[ii], cnz);
			//d_print_pmat(nz, nz, bs, hpRSQ[ii], cnz);
			}

		for(jj=0; jj<nx+nu; jj++) hux[0][jj] = 0.0;
		//hux[0][2*nu+0] =  5.0;
		//hux[0][2*nu+1] = -5.0;
		for(jj=0; jj<nu; jj++)
			{
			hux[0][nu+jj] = 7.5097;
			}
		for(; jj<nx; jj+=2)
			{
			hux[0][nu+jj+0] = 15.01940;
			hux[0][nu+jj+1] =  0.0;
			}

		d_back_ric_sv_new(N, nx, nu, hpBAbt2, hpRSQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);
		//for(ii=0; ii<=N; ii++)
		//	d_print_pmat(pnz, cnl-3, bs, hpL[ii], cnl);
		//d_print_pmat(pnz, nu, bs, hpL[0], cnl);
		//d_print_pmat(pnz, cnl-3, bs, hpL[1], cnl);
		//d_print_pmat(pnz, cnl-3, bs, hpL[2], cnl);
		//d_print_pmat(pnz, cnl-3, bs, hpL[N-3], cnl);
		//d_print_pmat(pnz, cnl-3, bs, hpL[N-2], cnl);
		//d_print_pmat(pnz, cnl-3, bs, hpL[N-1], cnl);
		//d_print_pmat(pnz, cnl, bs, hpL[N], cnl);
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nx+nu, hux[ii], 1);

		
		double *(hq3[N+1]);
		double *(hrq3[N+1]);
		double *(hrb3[N]);

		for(ii=0; ii<N; ii++)
			{
			d_zeros_align(&hq3[ii], pnz, 1);
			d_zeros_align(&hrq3[ii], pnz, 1);
			d_zeros_align(&hrb3[ii], pnx, 1);
			}
		d_zeros_align(&hq3[N], pnz, 1);
		d_zeros_align(&hrq3[N], pnz, 1);
		

		d_res_mpc(nx, nu, N, hpBAbt2, hpRSQ, hq3, hux, hpi, hrq3, hrb3);

		printf("\nresiduals\n");
		for(ii=0; ii<=N; ii++)
			d_print_mat(1, nx+nu, hrq3[ii], 1);

		for(ii=0; ii<N; ii++)
			d_print_mat(1, nx, hrb3[ii], 1);


		for(ii=0; ii<N; ii++)
			{
			free(hq3[ii]);
			free(hrq3[ii]);
			free(hrb3[ii]);
			}
		free(hq3[N]);
		free(hrq3[N]);


		// timing
		//struct timeval tv20, tv21;

#if 1
		gettimeofday(&tv20, NULL); // start

		for(ii=0; ii<nrep; ii++)
			{
			d_back_ric_sv_new(N, nx, nu, hpBAbt2, hpRSQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);
			}

		gettimeofday(&tv21, NULL); // start

		time_sv_full = (float) (tv21.tv_sec-tv20.tv_sec)/(nrep+0.0)+(tv21.tv_usec-tv20.tv_usec)/(nrep*1e6);

		printf("\ndiag time = %e\t\tfull time = %e\t\tip diag time = %e\n\n", time_sv_diag, time_sv_full, time_ip_diag);
#endif



		free(BAb_temp);
		for(ii=0; ii<N; ii++)
			{
			free(hpBAbt2[ii]);
			free(hpRSQ[ii]);
			}
		free(hpRSQ[N]);

		exit(1);


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
		d_back_ric_sv_new(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nu+nx, hux[ii], 1);
//			exit(1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\npi\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hpi[ii+1], 1);
			}

#if 0
		// call the solver
		d_back_ric_sv_new(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, hl, work, diag, 0, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nu+nx, hux[ii], 1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\npi\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hpi[ii+1], 1);
			}
#if 1
		// restore linear part
		for(ii=0; ii<=N; ii++)
			drowex_lib(nu+nx, hpQ[ii]+(nu+nx)/bs*bs*cnz+(nu+nx)%bs, hq[ii]);
			
		// call the solver
		d_back_ric_trs_new(N, nx, nu, hpBAbt, hq, 1, hux, hpL, hl, diag, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nu+nx, hux[ii], 1);
			}
		if(PRINTRES==1 && COMPUTE_MULT==1 && ll_max==1)
			{
			// print result 
			printf("\n\npi\n\n");
			for(ii=0; ii<N; ii++)
				d_print_mat(1, nx, hpi[ii+1], 1);
			}
#endif
			exit(1);
#endif

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
			//d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work1, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
			d_back_ric_trs_new(N, nx, nu, hpBAbt, hq, 1, hux, hpL, hdL, work1, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
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
		struct timeval tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10;

		// double precision
		gettimeofday(&tv0, NULL); // start

		// factorize & solve
		for(rep=0; rep<nrep; rep++)
			{
//			d_back_ric_sv_new(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);
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
				//d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work1, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
//				d_back_ric_trs_new(N, nx, nu, hpBAbt, hq, 1, hux, hpL, hdL, work1, 1, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy);
//			else
//				d_ric_trs_mhe_old(nx, nu, N, hpBAbt, hpL, hq, hux, work, 1, hPb, COMPUTE_MULT, hpi);
			}
		
		gettimeofday(&tv2, NULL); // start

		// residuals
		for(rep=0; rep<nrep; rep++)
			{
//			if(FREE_X0==0)
//				d_res_mpc(nx, nu, N, hpBAbt, hpQ, hq, hux, hpi, hrq, hrb);
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
				//d_ric_trs_mpc(nx, nu, N, hpBAbt, hpL, hq, hux, work1, 0, hPb, 0, hpi, 0, 0, 0, dummy, dummy);
//				d_back_ric_trs_new(N, nx, nu, hpBAbt, hq, 1, hux, hpL, hdL, work1, 0, hPb, 0, hpi, 0, 0, 0, dummy, dummy);
//			else
//				d_ric_trs_mhe_old(nx, nu, N, hpBAbt, hpL, hq, hux, work, 0, hPb, 0, hpi);
			}
		
		gettimeofday(&tv4, NULL); // start


		// factorize & solve (fast rsqrt)
		for(rep=0; rep<nrep; rep++)
			{
//			d_back_ric_sv_new(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, hPb, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);
			//d_back_ric_sv(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, work, diag, 0, dummy, COMPUTE_MULT, hpi, 0, 0, 0, dummy, dummy, dummy);
			}
			
		gettimeofday(&tv5, NULL); // start

#if 1

		// size-variant code

		for(rep=0; rep<nrep; rep++)
			{
//			d_back_ric_sv_tv(N, nx_v, nu_v, hpBAbt_tv, hpQ_tv, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, nb_v, 0, dummy, dummy, ng_v, dummy, dummy, dummy);
			d_back_ric_trf_tv(N, nx_v, nu_v, hpBAbt_tv, hpQ_tv, hpL, hdL, work0, work1, nb_v, 0, dummy, ng_v, dummy, dummy);
			}

		gettimeofday(&tv6, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			d_back_ric_trs_tv(N, nx_v, nu_v, hpBAbt_tv, hb_tv, hpL, hdL, hq_tv, hl, hux, work1, 1, hPb, 1, hpi, nb_v, 0, dummy, ng_v, dummy, dummy);
			}

		gettimeofday(&tv7, NULL); // start

//		for(ii=0; ii<=N; ii++)
//			printf("\n%d %d\n", nu_v[ii], nx_v[ii]);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<=N; ii++)
				d_print_mat(1, nu_v[ii]+nx_v[ii], hux[ii], 1);
			printf("\n");
			for(ii=1; ii<=N; ii++)
				d_print_mat(1, nx, hpi[ii], 1);
//			exit(1);
			}
#endif


#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_MKL) || defined(REF_BLAS_NETLIB)

		// size-variant code
		gettimeofday(&tv8, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			d_back_ric_trf_tv_blas(N, nx_v, nu_v, hBAbt, hQ, hL, BAbtL);
			}

		gettimeofday(&tv9, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			d_back_ric_trs_tv_blas(N, nx_v, nu_v, hBAbt, hb_tv, hL, hq_tv, hl, hux, work1, 1, hPb, 1, hpi);
			}

		gettimeofday(&tv10, NULL); // start

//		for(ii=0; ii<=N; ii++)
//			printf("\n%d %d\n", nu_v[ii], nx_v[ii]);

		if(PRINTRES==1 && ll_max==1)
			{
			/* print result */
			printf("\n\nsv\n\n");
			for(ii=0; ii<=N; ii++)
				d_print_mat(1, nu_v[ii]+nx_v[ii], hux[ii], 1);
			printf("\n");
			for(ii=1; ii<=N; ii++)
				d_print_mat(1, nx, hpi[ii], 1);
//			exit(1);
			}

#endif


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

		float flop_trf_tv = (1.0/3.0*nx*nx*nx) + (N-1)*(7.0/3.0*nx*nx*nx+4.0*nx*nx*nu+2.0*nx*nu*nu+1.0/3.0*nu*nu*nu) + (1.0*nx*nx*nu+1.0*nx*nu*nu+1.0/3.0*nu*nu*nu);
		float time_trf_tv = (float) (tv6.tv_sec-tv5.tv_sec)/(nrep+0.0)+(tv6.tv_usec-tv5.tv_usec)/(nrep*1e6);
		float Gflops_trf_tv = 1e-9*flop_trf_tv/time_trf_tv;
	
		float flop_trs_tv = (N)*(8*nx*nx+8.0*nx*nu+2.0*nu*nu);
		float time_trs_tv = (float) (tv7.tv_sec-tv6.tv_sec)/(nrep+0.0)+(tv7.tv_usec-tv6.tv_usec)/(nrep*1e6);
		float Gflops_trs_tv = 1e-9*flop_trs_tv/time_trs_tv;
	
		float time_trf_tv_blas = (float) (tv9.tv_sec-tv8.tv_sec)/(nrep+0.0)+(tv9.tv_usec-tv8.tv_usec)/(nrep*1e6);
		float Gflops_trf_tv_blas = 1e-9*flop_trf_tv/time_trf_tv_blas;

		float time_trs_tv_blas = (float) (tv10.tv_sec-tv9.tv_sec)/(nrep+0.0)+(tv10.tv_usec-tv9.tv_usec)/(nrep*1e6);
		float Gflops_trs_tv_blas = 1e-9*flop_trs_tv/time_trs_tv_blas;
	
		if(ll==0)
			{
			printf("\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv %%\t\ttrs time\ttrs Gflops\ttrs %%\n\n");
//			fprintf(f, "\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv %%\t\ttrs time\ttrs Gflops\ttrs %%\n\n");
			}
//		printf("%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_sv, Gflops_sv, 100.0*Gflops_sv/Gflops_max, time_trs, Gflops_trs, 100.0*Gflops_trs/Gflops_max, time_trs_admm, Gflops_trs_admm, 100.0*Gflops_trs_admm/Gflops_max, time_sv_fast, Gflops_sv_fast, 100.0*Gflops_sv_fast/Gflops_max, time_sv_tv, Gflops_sv_tv, 100.0*Gflops_sv_tv/Gflops_max);
//		fprintf(f, "%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_sv, Gflops_sv, 100.0*Gflops_sv/Gflops_max, time_trs, Gflops_trs, 100.0*Gflops_trs/Gflops_max, time_trs_admm, Gflops_trs_admm, 100.0*Gflops_trs_admm/Gflops_max, time_sv_fast, Gflops_sv_fast, 100.0*Gflops_sv_fast/Gflops_max, time_sv_tv, Gflops_sv_tv, 100.0*Gflops_sv_tv/Gflops_max);
	
		printf("%d\t%d\t%d\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\t%e\t%f\t%f\n", nx, nu, N, time_trf_tv, Gflops_trf_tv, 100.0*Gflops_trf_tv/Gflops_max, time_trs_tv, Gflops_trs_tv, 100.0*Gflops_trs_tv/Gflops_max, time_trf_tv_blas, Gflops_trf_tv_blas, 100.0*Gflops_trf_tv_blas/Gflops_max, time_trs_tv_blas, Gflops_trs_tv_blas, 100.0*Gflops_trs_tv_blas/Gflops_max);

/************************************************
* return
************************************************/

		free(nx_v);
		free(nu_v);
		free(nb_v);
		free(ng_v);

		free(A);
		free(pA);
		free(B);
		free(b);
		free(b0);
		free(x0);
		free(BAb);
		free(BAbt);
		free(pBAbt);
		free(pBAbt0);
		free(pBAbt1);
		free(Q);
		free(pQ);
		free(pQ0);
		free(pQ1);
		free(pQN);
		free(q);
		free(q1);
		free(work0);
		free(work1);
		for(jj=0; jj<N; jj++)
			{
			if(LTI!=1)
				{
				free(hpQ[jj]);
				free(hq[jj]);
				free(hpBAbt[jj]);
				}
			free(hpL[jj]);
			free(hdL[jj]);
			free(hl[jj]);
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
		free(hdL[N]);
		free(hl[N]);
		free(hux[N]);
		free(hpi[N]);
		free(hrq[N]);
	
#if defined(REF_BLAS_OPENBLAS) || defined(REF_BLAS_MKL) || defined(REF_BLAS_NETLIB)
		free(BAbt0);
		free(BAbt1);
		free(Q0);
		free(Q1);
		free(QN);
		free(BAbtL);
		for(jj=0; jj<N; jj++)
			{
			free(hL[jj]);
			}
#endif


		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



