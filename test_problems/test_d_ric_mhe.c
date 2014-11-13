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
	printf(" Copyright (C) 2014 by Technical University of Denmark. All rights reserved.\n");
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
	fprintf(f, "C = 'd_c99_2x2';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_2X2)
	fprintf(f, "C = 'd_c99_4x4';\n");
	fprintf(f, "\n");
#endif

	fprintf(f, "A = [%f %f];\n", GHz_max, flops_max);
	fprintf(f, "\n");

	fprintf(f, "B = [\n");
	

	printf("\n");
	printf("Tested solvers:\n");
	printf("-sv : Riccati factorization and system solution (prediction step in IP methods)\n");
	printf("-trs: system solution after a previous call to Riccati factorization (correction step in IP methods)\n");
	printf("\n");
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
//	int ll_max = 77;
	int ll_max = 1;
	for(ll=0; ll<ll_max; ll++)
		{
		

		if(ll_max==1)
			{
			nx = NX; // number of states (it has to be even for the mass-spring system test problem)
			nu = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = NN; // horizon lenght
			nrep = NREP;
			}
		else
			{
			nx = nn[ll]; // number of states (it has to be even for the mass-spring system test problem)
			nu = 2; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			N  = 10; // horizon lenght
			nrep = nnrep[ll];
			}

		int rep;
		
		const int ny = nx/2; // size of measurements vector
	
		const int nz = nx+nu+1; // TODO delete
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);
		const int anu = nal*((nu+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pnu = bs*((nu+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnu = ncl*((nu+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	
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
			x0[jj] = 0.0;
		x0[0] = 3.5;
		x0[1] = 3.5;
	
		double *C; d_zeros(&C, ny, nx); // inputs matrix
		for(jj=0; jj<ny; jj++)
			C[jj*(ny+1)] = 1.0;

//		d_print_mat(nx, nx, A, nx);
//		d_print_mat(nx, nu, B, nx);
//		d_print_mat(ny, nx, C, ny);
//		d_print_mat(nx, 1, b, nx);
//		d_print_mat(nx, 1, x0, nx);
	
		/* packed into contiguous memory */
		double *pA; d_zeros_align(&pA, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, 0, bs, A, nx, pA, cnx);

		double *pG; d_zeros_align(&pG, pnx, cnu);
		d_cvt_mat2pmat(nx, nu, 0, bs, B, nx, pG, cnu);
		
		double *pC; d_zeros_align(&pC, pny, cnx);
		d_cvt_mat2pmat(ny, nx, 0, bs, C, ny, pC, cnx);
		
//		d_print_pmat(nx, nx, bs, pA, cnx);
//		d_print_pmat(nx, nu, bs, pG, cnu);
//		d_print_pmat(ny, nx, bs, pC, cnx);

/************************************************
* cost function
************************************************/	

		double *Q; d_zeros(&Q, nu, nu);
		for(jj=0; jj<nu; jj++)
			Q[jj*(nu+1)] = 1.0;

		double *R; d_zeros(&R, ny, ny);
		for(jj=0; jj<ny; jj++)
			R[jj*(ny+1)] = 1.0;

		double *L0; d_zeros(&L0, nx, nx);
		for(jj=0; jj<nx; jj++)
			L0[jj*(nx+1)] = 2.0;

		/* packed into contiguous memory */
		double *pQ; d_zeros_align(&pQ, pnu, cnu);
		d_cvt_mat2pmat(nu, nu, 0, bs, Q, nu, pQ, cnu);

		double *pR; d_zeros_align(&pR, pny, cny);
		d_cvt_mat2pmat(ny, ny, 0, bs, R, ny, pR, cny);

//		d_print_pmat(nu, nu, bs, pQ, cnu);
//		d_print_pmat(ny, ny, bs, pR, cny);

/************************************************
* series of matrices
************************************************/	

		double *(hpA[N]);
		double *(hpG[N]);
		double *(hpC[N]);
		double *(hpQ[N]);
		double *(hpR[N]);
		double *(hpL[N+1]);

		d_zeros_align(&hpL[0], pnx, cnx);
		d_cvt_mat2pmat(nx, nx, 0, bs, L0, nx, hpL[0], cnx);

		for(jj=0; jj<N; jj++)
			{
			hpA[jj] = pA;
			hpG[jj] = pG;
			hpC[jj] = pC;
			hpQ[jj] = pQ;
			hpR[jj] = pR;
			d_zeros_align(&hpL[jj+1], pnx, cnx);
			}


/************************************************
* call the solver
************************************************/	

		d_ric_trf_mhe(nx, nu, ny, N, hpA, hpG, hpC, hpL, hpQ, hpR);

/************************************************
* return
************************************************/

		free(A);
		free(B);
		free(C);
		free(b);
		free(x0);
		free(Q);
		free(R);
		free(L0);
		free(pA);
		free(pG);
		free(pC);
		free(pQ);
		free(pR);
		free(hpL[0]);
		for(jj=0; jj<N; jj++)
			{
			free(hpL[jj+1]);
			}



		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



