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
	
	
	// problem size
	int nx = NX; // number of states (it has to be even for the mass-spring system test problem)
	int nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
	int N  = NN; // horizon lenght

	// number of problem instances solution for better timing
	int nrep = NREP; 


	int N_v[N+1];
	int nx_v[N+1];
	int nu_v[N+1];
	int nb_v[N+1]; for(ii=0; ii<=N; ii++) nb_v[ii] = 0;
	int ng_v[N+1]; for(ii=0; ii<=N; ii++) ng_v[ii] = 0;


	int ll;
	for(ll=0; ll<N; ll++)
		{


		int Np = ll+1; // horizon length in the partially condensed problem

		int Nc = N/Np; // minimum horizon length within each stage group

		int rN = N - Np*Nc; // number of stage groups with horizon Nc+1 


		ii = 0;
		for(; ii<rN; ii++)
			N_v[ii] = Nc+1;
		for(; ii<Np; ii++)
			N_v[ii] = Nc;

//		printf("\n\t%d:\t", Np);
//		for(ii=0; ii<Np; ii++)
//			printf("%d\t", N_v[ii]);
//		printf("\n\n");

		nx_v[0] = 0;
		for(ii=1; ii<=Np; ii++)
			nx_v[ii] = nx;

//		printf("\n\t%d:\t", Np);
//		for(ii=0; ii<=Np; ii++)
//			printf("%d\t", nx_v[ii]);
//		printf("\n\n");

		for(ii=0; ii<Np; ii++)
			nu_v[ii] = nu*N_v[ii];
		nu_v[Np] = 0;

//		printf("\n\t%d:\t", Np);
//		for(ii=0; ii<=Np; ii++)
//			printf("%d\t", nu_v[ii]);
//		printf("\n\n");
		


		// matrix size
		int pnux_v[Np+1];
		int cnx_v[Np+1];
		int cnux_v[Np+1];
		for(ii=0; ii<=Np; ii++) 
			{
			pnux_v[ii] = (nu_v[ii]+nx_v[ii]+bs-1)/bs*bs;
			cnx_v[ii] = (nx_v[ii]+ncl-1)/ncl*ncl;
			cnux_v[ii] = (nu_v[ii]+nx_v[ii]+ncl-1)/ncl*ncl;
			}
		


		// dynamic system matrices
		double *pBA0;
		double *pBA1;
		double *pBANm1;

		d_zeros_align(&pBA0, pnux_v[0], cnx_v[1]);
		if(Np>1) 
			d_zeros_align(&pBA1, pnux_v[1], cnx_v[2]);
		if(Np>1)
			d_zeros_align(&pBANm1, pnux_v[Np-1], cnx_v[Np]);

		double *(hpBA[Np]);
		hpBA[0] = pBA0;
		for(ii=1; ii<rN; ii++) hpBA[ii] = pBA1;
		for(; ii<Np; ii++) hpBA[ii] = pBANm1;
//		for(ii=0; ii<Np; ii++)
//			d_print_pmat(nu_v[ii]+nx_v[ii], nx_v[ii+1], bs, hpBA[ii], cnx_v[ii]);


		// cost function matrices
		double *pQ0;
		double *pQ1;
		double *pQNm1;
		double *pQN;

		d_zeros_align(&pQ0, pnux_v[0], cnux_v[0]);
		for(ii=0; ii<nu_v[0]+nx_v[0]; ii++)
			pQ0[ii/bs*bs*cnux_v[0]+ii%bs+ii*bs] = 1.0;
//		d_print_pmat(nu_v[0]+nx_v[0], nu_v[0]+nx_v[0], bs, pQ0, cnux_v[0]);
		if(Np>1)
			{
			d_zeros_align(&pQ1, pnux_v[1], cnux_v[1]);
			for(ii=0; ii<nu_v[1]+nx_v[1]; ii++)
				pQ1[ii/bs*bs*cnux_v[1]+ii%bs+ii*bs] = 1.0;
//			d_print_pmat(nu_v[1]+nx_v[1], nu_v[1]+nx_v[1], bs, pQ1, cnux_v[1]);
			}
		if(Np>1)
			{
			d_zeros_align(&pQNm1, pnux_v[Np-1], cnux_v[Np-1]);
			for(ii=0; ii<nu_v[Np-1]+nx_v[Np-1]; ii++)
				pQNm1[ii/bs*bs*cnux_v[Np-1]+ii%bs+ii*bs] = 1.0;
//			d_print_pmat(nu_v[Np-1]+nx_v[Np-1], nu_v[Np-1]+nx_v[Np-1], bs, pQNm1, cnux_v[Np-1]);
			}
		d_zeros_align(&pQN, pnux_v[Np], cnux_v[Np]);
		for(ii=0; ii<nu_v[Np]+nx_v[Np]; ii++)
			pQN[ii/bs*bs*cnux_v[Np]+ii%bs+ii*bs] = 1.0;
//		d_print_pmat(nu_v[Np]+nx_v[Np], nu_v[Np]+nx_v[Np], bs, pQN, cnux_v[Np]);

		double *(hpQ[Np+1]);
		hpQ[0] = pQ0;
		for(ii=1; ii<rN; ii++) hpQ[ii] = pQ1;
		for(; ii<Np; ii++) hpQ[ii] = pQNm1;
		hpQ[Np] = pQN;
//		for(ii=0; ii<=Np; ii++)
//			d_print_pmat(nu_v[ii]+nx_v[ii], nu_v[ii]+nx_v[ii], bs, hpQ[ii], cnux_v[ii]);



		// work space
		int cnl_v;
		double *(hpL[Np+1]);
		for(ii=0; ii<=Np; ii++)
			{
			cnl_v = cnux_v[ii]<cnx_v[ii]+ncl ? cnx_v[ii]+ncl : cnux_v[ii];
			d_zeros_align(&hpL[ii], pnux_v[ii], cnl_v);
			}
		double *(hdL[Np+1]);
		for(ii=0; ii<=Np; ii++)
			d_zeros_align(&hdL[ii], pnux_v[ii], 1);
		double *pBAL;
		int pnuxM = pnux_v[0];
		for(ii=1; ii<Np; ii++)
			pnuxM = pnux_v[ii]>pnuxM ? pnux_v[ii] : pnuxM;
		int cnxM = cnx_v[1];
		for(ii=2; ii<=Np; ii++)
			cnxM = cnx_v[ii]>cnxM ? cnx_v[ii] : cnxM;
		d_zeros_align(&pBAL, pnuxM, cnxM);



		// time the KKT matrix factorization
		struct timeval tv0, tv1;
		double **dummy;
		int rep;

		nrep = 400;

		gettimeofday(&tv0, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			d_back_ric_trf_tv(Np, nx_v, nu_v, hpBA, hpQ, hpL, hdL, pBAL, nb_v, 0, dummy, ng_v, dummy, dummy);
			}

		gettimeofday(&tv1, NULL); // start

		float time_trf_tv = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		printf("%d\t%e\n", Np, time_trf_tv);


		// free data
		free(pBA0);
		if(Np>1)
			free(pBA1);
		if(Np>1)
			free(pBANm1);
		free(pQ0);
		if(Np>1)
			free(pQ1);
		if(Np>1)
			free(pQNm1);
		free(pQN);
		for(ii=0; ii<=Np; ii++)
			free(hpL[ii]);
		for(ii=0; ii<=Np; ii++)
			free(hdL[ii]);
		free(pBAL);


		}

	return 0;




	for(ll=0; ll<1; ll++)
		{
		

		// define time-varian problem size
		nx_v[0] = 0;
		for(ii=1; ii<N; ii++) nx_v[ii] = nx;
		nx_v[N] = nx;

		nu_v[0] = nu;
		for(ii=1; ii<N; ii++) nu_v[ii] = nu;
		nu_v[N] = 0;

		for(ii=0; ii<=N; ii++) nb_v[ii] = 0;

		for(ii=0; ii<=N; ii++) ng_v[ii] = 0;



		int rep;
	
		int nz = nx+nu+1;
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



/************************************************
* riccati-like iteration
************************************************/
		

		// timing 
		struct timeval tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10;

		// double precision
		gettimeofday(&tv0, NULL); // start

		gettimeofday(&tv1, NULL); // start

		gettimeofday(&tv2, NULL); // start

		gettimeofday(&tv3, NULL); // start

		gettimeofday(&tv4, NULL); // start

		gettimeofday(&tv5, NULL); // start

		// size-variant code

		for(rep=0; rep<nrep; rep++)
			{
//			d_back_ric_sv_tv(N, nx_v, nu_v, hpBAbt_tv, hpQ_tv, hux, hpL, hdL, work0, work1, 0, dummy, COMPUTE_MULT, hpi, nb_v, 0, dummy, dummy, ng_v, dummy, dummy, dummy);
			d_back_ric_trf_tv(N, nx_v, nu_v, hpBAbt_tv, hpQ_tv, hpL, hdL, work0, nb_v, 0, dummy, ng_v, dummy, dummy);
			}

		gettimeofday(&tv6, NULL); // start

		for(rep=0; rep<nrep; rep++)
			{
			d_back_ric_trs_tv(N, nx_v, nu_v, hpBAbt_tv, hb_tv, hpL, hdL, hq_tv, hl, hux, work1, 1, hPb, 1, hpi, nb_v, 0, dummy, ng_v, dummy, dummy);
			}

		gettimeofday(&tv7, NULL); // start

		if(PRINTRES==1)
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



		float flop_trf_tv = (1.0/3.0*nx*nx*nx) + (N-1)*(7.0/3.0*nx*nx*nx+4.0*nx*nx*nu+2.0*nx*nu*nu+1.0/3.0*nu*nu*nu) + (1.0*nx*nx*nu+1.0*nx*nu*nu+1.0/3.0*nu*nu*nu);
		float time_trf_tv = (float) (tv6.tv_sec-tv5.tv_sec)/(nrep+0.0)+(tv6.tv_usec-tv5.tv_usec)/(nrep*1e6);
		float Gflops_trf_tv = 1e-9*flop_trf_tv/time_trf_tv;
	
		float flop_trs_tv = (N)*(8*nx*nx+8.0*nx*nu+2.0*nu*nu);
		float time_trs_tv = (float) (tv7.tv_sec-tv6.tv_sec)/(nrep+0.0)+(tv7.tv_usec-tv6.tv_usec)/(nrep*1e6);
		float Gflops_trs_tv = 1e-9*flop_trs_tv/time_trs_tv;
	
/************************************************
* return
************************************************/

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
	


		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



