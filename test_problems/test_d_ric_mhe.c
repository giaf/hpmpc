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
#include "../include/blas_d.h"
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

	int nx, nw, ny, N, nrep, Ns;

	int ll;
//	int ll_max = 77;
	int ll_max = 1;
	for(ll=0; ll<ll_max; ll++)
		{
		

		FILE* fid;
		double* yy;
		float* yy_temp;

		if(1)
			{
			fid = fopen("./test_problems/mhe_measure.dat", "r");
			if(fid==NULL)
				exit(-1);
			//printf("\nhola\n");
			int dummy_int = fscanf(fid, "%d %d %d %d", &nx, &nw, &ny, &Ns);
			//printf("\n%d %d %d %d\n", nx, nw, ny, Ns);
			yy_temp = (float*) malloc(ny*Ns*sizeof(float));
			yy = (double*) malloc(ny*Ns*sizeof(double));
			for(jj=0; jj<ny*Ns; jj++)
				{
				dummy_int = fscanf(fid, "%e", &yy_temp[jj]);
				yy[jj] = (double) yy_temp[jj];
				//printf("\n%f", yy[jj]);
				}
			//printf("\n");
			fclose(fid);
			N = Ns-1; // NN;
			nrep = NREP;
			//nx = 10;
			}
		else if(ll_max==1)
			{
			nx = NX; // number of states (it has to be even for the mass-spring system test problem)
			nw = NU; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			ny = nx/2; // size of measurements vector
			N  = NN; // horizon lenght
			nrep = NREP;
			}
		else
			{
			nx = nn[ll]; // number of states (it has to be even for the mass-spring system test problem)
			nw = 2; // number of inputs (controllers) (it has to be at least 1 and at most nx/2 for the mass-spring system test problem)
			ny = nx/2; // size of measurements vector
			N  = 10; // horizon lenght
			nrep = nnrep[ll];
			}

		int rep;
		
	
		const int nz = nx+ny; // TODO delete
		const int nwx = nw+nx;
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);
		const int anw = nal*((nw+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pnw = bs*((nw+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int pnx2 = bs*((2*nx+bs-1)/bs);
		const int pnwx = bs*((nw+nx+bs-1)/bs);
		const int cnz = ncl*((nz+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnw = ncl*((nw+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);
		const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));
		const int cnwx = ncl*((nw+nx+ncl-1)/ncl);
		const int cnf = cnz<cnx+ncl ? cnx+ncl : cnz;

		const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
		const int cnl = nx+nw+pad+cnx;
		const int pad2 = (ncl-(nx)%ncl)%ncl; // packing between AGL & P
		const int cnl2 = cnz<cnx+ncl ? nx+pad2+cnx+ncl : nx+pad2+cnz;
	
/************************************************
* dynamical system
************************************************/	

		double *A; d_zeros(&A, nx, nx); // states update matrix

		double *B; d_zeros(&B, nx, nw); // inputs matrix

		double *b; d_zeros(&b, nx, 1); // states offset
		double *x0; d_zeros(&x0, nx, 1); // initial state

		double Ts = 0.5; // sampling time
		mass_spring_system(Ts, nx, nw, N, A, B, b, x0);
	
		for(jj=0; jj<nx; jj++)
			b[jj] = 0.0;
	
		for(jj=0; jj<nx; jj++)
			x0[jj] = 0.0;
		x0[0] = 3.5;
		x0[1] = 3.5;
	
		double *C; d_zeros(&C, ny, nx); // inputs matrix
		for(jj=0; jj<ny; jj++)
			C[jj*(ny+1)] = 1.0;

//		d_print_mat(nx, nx, A, nx);
//		d_print_mat(nx, nw, B, nx);
//		d_print_mat(ny, nx, C, ny);
//		d_print_mat(nx, 1, b, nx);
//		d_print_mat(nx, 1, x0, nx);
	
		/* packed into contiguous memory */
		double *pA; d_zeros_align(&pA, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, 0, bs, A, nx, pA, cnx);

		double *pG; d_zeros_align(&pG, pnx, cnw);
		d_cvt_mat2pmat(nx, nw, 0, bs, B, nx, pG, cnw);
		
		double *pC; d_zeros_align(&pC, pny, cnx);
		d_cvt_mat2pmat(ny, nx, 0, bs, C, ny, pC, cnx);
		
		double *pCA; d_zeros_align(&pCA, pnz, cnx);
		d_cvt_mat2pmat(ny, nx, 0, bs, C, ny, pCA, cnx);
		d_cvt_mat2pmat(nx, nx, ny, bs, A, nx, pCA+(ny/bs)*bs+ny%bs, cnx);

//		d_print_pmat(nx, nx, bs, pA, cnx);
//		d_print_pmat(nx, nw, bs, pG, cnw);
//		d_print_pmat(ny, nx, bs, pC, cnx);

/************************************************
* cost function
************************************************/	

		double *Q; d_zeros(&Q, nw, nw);
		for(jj=0; jj<nw; jj++)
			Q[jj*(nw+1)] = 1.0;

		double *R; d_zeros(&R, ny, ny);
		for(jj=0; jj<ny; jj++)
			R[jj*(ny+1)] = 1.0;

		double *L0; d_zeros(&L0, nx, nx);
		for(jj=0; jj<nx; jj++)
			L0[jj*(nx+1)] = 1.0;

		double *r; d_zeros_align(&r, any, 1);
		for(jj=0; jj<ny; jj++)
			r[jj] = 0.0;

		double *q; d_zeros_align(&q, anw, 1);
		for(jj=0; jj<nw; jj++)
			q[jj] = 0.0;

		double *f; d_zeros_align(&f, anx, 1);
		for(jj=0; jj<nx; jj++)
			f[jj] = b[jj]; //1.0;

		/* packed into contiguous memory */
		double *pQ; d_zeros_align(&pQ, pnw, cnw);
		d_cvt_mat2pmat(nw, nw, 0, bs, Q, nw, pQ, cnw);

		double *pR; d_zeros_align(&pR, pny, cny);
		d_cvt_mat2pmat(ny, ny, 0, bs, R, ny, pR, cny);

//		d_print_pmat(nw, nw, bs, pQ, cnw);
//		d_print_pmat(ny, ny, bs, pR, cny);

/************************************************
* compound quantities
************************************************/	
		
		double *pQG; d_zeros_align(&pQG, pnwx, cnw);
		d_cvt_mat2pmat(nw, nw, 0, bs, Q, nw, pQG, cnw);
		d_cvt_mat2pmat(nx, nw, nw, bs, B, nx, pQG+(nw/bs)*bs*cnw+nw%bs, cnw);
		//d_print_pmat(nw+nx, nw, bs, pQG, cnw);

		double *pRA; d_zeros_align(&pRA, pnx2, cnx);
		d_cvt_mat2pmat(ny, ny, 0, bs, R, ny, pRA, cnx);
		d_cvt_mat2pmat(nx, nx, nx, bs, A, nx, pRA+(nx/bs)*bs*cnx+nx%bs, cnx);
		//d_print_pmat(2*nx, cnx, bs, pRA, cnx);
		//exit(1);

/************************************************
* series of matrices
************************************************/	

		double *(hpA[N]);
		double *(hpCA[N]);
		double *(hpG[N]);
		double *(hpC[N+1]);
		double *(hpQ[N]);
		double *(hpR[N+1]);
		double *(hpLp[N+1]);
		double *(hdLp[N+1]);
		double *(hpLp2[N+1]);
		double *(hpLe[N+1]);
		double *(hq[N]);
		double *(hr[N+1]);
		double *(hf[N]);
		double *(hxe[N+1]);
		double *(hxp[N+1]);
		double *(hw[N]);
		double *(hy[N+1]);
		double *(hlam[N]);

		double *(hpQG[N]);
		double *(hpRA[N+1]);
		double *(hpGLq[N]);
		double *(hpALe[N+1]);
		double *(hqq[N]);
		double *(hrr[N+1]);
		double *(hff[N]);
		double *p_hqq; d_zeros_align(&p_hqq, anw, N);
		double *p_hrr; d_zeros_align(&p_hrr, anx, N+1);
		double *p_hff; d_zeros_align(&p_hff, anx, N);

		double *p_hxe; d_zeros_align(&p_hxe, anx, N+1);
		double *p_hxp; d_zeros_align(&p_hxp, anx, N+1);
		double *p_hw; d_zeros_align(&p_hw, anw, N);
		double *p_hy; d_zeros_align(&p_hy, any, N+1);
		double *p_hlam; d_zeros_align(&p_hlam, anx, N);

		for(jj=0; jj<N; jj++)
			{
			hpA[jj] = pA;
			hpCA[jj] = pCA;
			hpG[jj] = pG;
			hpC[jj] = pC;
			hpQ[jj] = pQ;
			hpR[jj] = pR;
			d_zeros_align(&hpLp[jj], pnx, cnl);
			d_zeros_align(&hdLp[jj], anx, 1);
			d_zeros_align(&hpLp2[jj], pnz, cnl2);
			d_zeros_align(&hpLe[jj], pnz, cnf);
			hq[jj] = q;
			hr[jj] = r;
			hf[jj] = f;

			hpQG[jj] = pQG;
			hpRA[jj] = pRA;
			d_zeros_align(&hpGLq[jj], pnwx, cnw);
			d_zeros_align(&hpALe[jj], pnx2, cnx2);
			hqq[jj] = p_hqq+jj*anw;
			hrr[jj] = p_hrr+jj*anx;
			hff[jj] = p_hff+jj*anx;

			hxe[jj] = p_hxe+jj*anx; //d_zeros_align(&hxe[jj], anx, 1);
			hxp[jj] = p_hxp+jj*anx; //d_zeros_align(&hxp[jj], anx, 1);
			hw[jj] = p_hw+jj*anw; //d_zeros_align(&hw[jj], anw, 1);
			hy[jj] = p_hy+jj*any; //d_zeros_align(&hy[jj], any, 1);
			hlam[jj] = p_hlam+jj*anx; //d_zeros_align(&hlambda[jj], anx, 1);
			}

		hpC[N] = pC;
		hpR[N] = pR;
		d_zeros_align(&hpLp[N], pnx, cnl);
		d_zeros_align(&hdLp[N], anx, 1);
		d_zeros_align(&hpLp2[N], pnz, cnl2);
		d_zeros_align(&hpLe[N], pnz, cnf);
		hr[N] = r;

		double *pCtRC; d_zeros_align(&pCtRC, pnx, cnx);
		d_cvt_mat2pmat(ny, ny, 0, bs, R, ny, pCtRC, cnx);
		hpRA[N] = pCtRC; // there is not A_N
		d_zeros_align(&hpALe[N], pnx, cnx2); // there is not A_N: pnx not pnx2
		hrr[N] = p_hrr+N*anx;

		hxe[N] = p_hxe+N*anx; //d_zeros_align(&hxe[N], anx, 1);
		hxp[N] = p_hxp+N*anx; //d_zeros_align(&hxp[N], anx, 1);
		hy[N] = p_hy+N*any; //d_zeros_align(&hy[N], any, 1);

		// initialize hpLp[0] with the cholesky factorization of /Pi_p
		d_cvt_mat2pmat(nx, nx, 0, bs, L0, nx, hpLp[0]+(nx+nw+pad)*bs, cnl);
		for(ii=0; ii<nx; ii++) hdLp[0][ii] = 1.0/L0[ii*(nx+1)];
		d_cvt_mat2pmat(nx, nx, ny, bs, L0, nx, hpLp2[0]+(ny/bs)*bs+ny%bs+(nx+pad2+ny)*bs, cnl2);
		dtrtr_l_lib(nx, ny, hpLp2[0]+(ny/bs)*bs*cnl2+ny%bs+(nx+pad2+ny)*bs, cnl2, hpLp2[0]+(nx+pad2+ncl)*bs, cnl2);	
		//d_print_pmat(nx, cnl, bs, hpLp[0], cnl);
		//d_print_pmat(nz, cnl2, bs, hpLp2[0], cnl2);

		// buffer for L0
		double *pL0; d_zeros_align(&pL0, pnx, cnx);
		d_cvt_mat2pmat(nx, nx, 0, bs, L0, nx, pL0, cnx);
		// invert L0 in hpALe[0]
		dtrinv_lib(nx, pL0, cnx, hpALe[0], cnx2);
		//d_print_pmat(nx, nx, bs, pL0, cnx);
		//d_print_pmat(pnx2, cnx2, bs, hpALe[0], cnx2);
		//exit(1);

		//double *work; d_zeros_align(&work, pny*cnx+pnz*cnz+anz+pnz*cnf+pnw*cnw, 1);
		double *work; d_zeros_align(&work, 2*pny*cnx+anz+pnw*cnw+pnx*cnx, 1);
		double *work2; d_zeros_align(&work2, 2*pny*cnx+pnw*cnw+pnx*cnw+2*pnx*cnx+anz, 1);

		double *work3; d_zeros_align(&work3, pnx*cnl+anx, 1);
//		for(jj=0; jj<2*pny*cnx+anz+pnw*cnw+pnx*cnx; jj++)
//			work[jj] = -100.0;

		// measurements
		for(jj=0; jj<=N; jj++)
			for(ii=0; ii<ny; ii++)
				hy[jj][ii] = yy[jj*ny+ii];

		//d_print_mat(ny, N+1, hy[0], any);

		// initial guess
		for(ii=0; ii<nx; ii++)
			hxp[0][ii] = 0.0;
		hxp[0][0] = 0.0;
		hxp[0][1] = 0.0;

/************************************************
* call the solver
************************************************/	

		//d_print_mat(nx, nx, A, nx);
		//d_print_mat(nx, nw, B, nx);

		//d_ric_trf_mhe_test(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hpQ, hpR, hpLe, work);
		d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);

		// estimation
		d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, 0, hlam, work);

		if(PRINTRES)
			{
			// print solution
			printf("\nx_e\n");
			d_print_mat(nx, N+1, hxe[0], anx);
			}
	
		// smooth estimation
		d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, 1, hlam, work);

		//d_print_pmat(nx, nx, bs, hpLp[N-1]+(nx+nw+pad)*bs, cnl);
		//d_print_pmat(nx, nx, bs, hpLp[N]+(nx+nw+pad)*bs, cnl);
		//d_print_pmat(nx, nx, bs, hpLe[N-1]+ncl*bs, cnf);
		//d_print_pmat(nx, nx, bs, hpLe[N]+ncl*bs, cnf);

		//d_print_mat(nx, N+1, hxp[0], anx);
		//d_print_mat(nx, N+1, hxe[0], anx);
		//d_print_mat(nx, N, hlam[0], anx);
		d_print_mat(nw, N, hw[0], anw);

		// information filter - factorization
		d_ric_trf_mhe_if(nx, nw, N, hpRA, hpQG, hpALe, hpGLq, work3);

		// information filter - solution
		double *y_temp; d_zeros_align(&y_temp, any, 1);
		for(ii=0; ii<N; ii++) for(jj=0; jj<nw; jj++) hqq[ii][jj] = -q[jj];
		for(ii=0; ii<N; ii++) for(jj=0; jj<nx; jj++) hff[ii][jj] = -f[jj];
		for(ii=0; ii<=N; ii++) 
			{
			for(jj=0; jj<ny; jj++) y_temp[jj] = - r[jj];
			//d_print_mat(1, ny, y_temp, 1);
			dsymv_lib(ny, 0, hpR[ii], cny, hy[ii], y_temp, -1);
			//d_print_mat(1, ny, y_temp, 1);
			dgemv_t_lib(ny, nx, 0, hpC[ii], cnx, y_temp, hrr[ii], 0);
			//d_print_mat(1, nx, hrr[ii], 1);
			//if(ii==9)
			//exit(1);
			}
		d_ric_trs_mhe_if(nx, nw, N, hpALe, hpGLq, hrr, hqq, hff, hxp, hxe, hw, hlam, work3);

		//d_print_pmat(nx, nx, bs, hpALe[N-1], cnx2);
		//d_print_pmat(nx, nx, bs, hpALe[N], cnx2);
		//d_print_pmat(nx, nx, bs, hpALe[N-2]+cnx*bs, cnx2);
		//d_print_pmat(nx, nx, bs, hpALe[N-1]+cnx*bs, cnx2);
		//d_print_pmat(nx, nx, bs, hpALe[N]+cnx*bs, cnx2);
		//d_print_pmat(nx, nx, bs, hpRA[N], cnx);

		//d_print_mat(nx, N+1, hxp[0], anx);
		//d_print_mat(nx, N+1, hxe[0], anx);
		//d_print_mat(nx, N, hlam[0], anx);
		d_print_mat(nw, N, hw[0], anw);
		//exit(1);

		if(PRINTRES)
			{
			// print solution
			printf("\nx_p\n");
			d_print_mat(nx, N+1, hxp[0], anx);
			printf("\nx_s\n");
			d_print_mat(nx, N+1, hxe[0], anx);
			printf("\nw\n");
			d_print_mat(nw, N+1, hw[0], anw);
			//printf("\nL_p\n");
			d_print_pmat(nx, nx, bs, hpLp[0]+(nx+nw+pad)*bs, cnl);
			d_print_mat(1, nx, hdLp[0], 1);
			d_print_pmat(nx, nx, bs, hpLp[1]+(nx+nw+pad)*bs, cnl);
			d_print_mat(1, nx, hdLp[1], 1);
			d_print_pmat(nx, nx, bs, hpLp[2]+(nx+nw+pad)*bs, cnl);
			d_print_mat(1, nx, hdLp[2], 1);
			d_print_pmat(nx, nx, bs, hpLp[N]+(nx+nw+pad)*bs, cnl);
			d_print_mat(1, nx, hdLp[N], 1);
			//printf("\nL_p\n");
			//d_print_pmat(nz, nz, bs, hpLp2[0]+(nx+pad2)*bs, cnl2);
			//d_print_pmat(nz, nz, bs, hpLp2[1]+(nx+pad2)*bs, cnl2);
			//d_print_pmat(nz, nz, bs, hpLp2[2]+(nx+pad2)*bs, cnl2);
			//printf("\nL_e\n");
			//d_print_pmat(nz, nz, bs, hpLe[0], cnf);
			//d_print_pmat(nz, nz, bs, hpLe[1], cnf);
			//d_print_pmat(nz, nz, bs, hpLe[2], cnf);
			//d_print_pmat(nx, nx, bs, hpA[0], cnx);
			}


		// timing 
		struct timeval tv0, tv1, tv2, tv3, tv4, tv5, tv6;

		// double precision
		gettimeofday(&tv0, NULL); // start

		// factorize
		for(rep=0; rep<nrep; rep++)
			{
			//d_ric_trf_mhe_test(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hpQ, hpR, hpLe, work);
			d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);
			}

		gettimeofday(&tv1, NULL); // start

		// solve
		for(rep=0; rep<nrep; rep++)
			{
			d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, 1, hlam, work);
			}

		gettimeofday(&tv2, NULL); // start

		// factorize
		for(rep=0; rep<nrep; rep++)
			{
			//d_print_pmat(nx, nx, bs, hpLe[N]+(ncl)*bs, cnf);
			//d_print_pmat(nx, nx, bs, hpLp[N]+(nx+nw+pad)*bs, cnl);
			//d_ric_trf_mhe_test(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hpQ, hpR, hpLe, work);
			d_ric_trf_mhe_end(nx, nw, ny, N, hpCA, hpG, hpC, hpLp2, hpQ, hpR, hpLe, work2);
			}

		gettimeofday(&tv3, NULL); // start

		// solve
		for(rep=0; rep<nrep; rep++)
			{
			d_ric_trs_mhe_end(nx, nw, ny, N, hpA, hpG, hpC, hpLp2, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hy, work2);
			}

		gettimeofday(&tv4, NULL); // start

		// factorize information filter
		for(rep=0; rep<nrep; rep++)
			{
			d_ric_trf_mhe_if(nx, nw, N, hpRA, hpQG, hpALe, hpGLq, work3);
			}

		gettimeofday(&tv5, NULL); // start

		// factorize information filter
		for(rep=0; rep<nrep; rep++)
			{
			d_ric_trs_mhe_if(nx, nw, N, hpALe, hpGLq, hrr, hqq, hff, hxp, hxe, hw, hlam, work3);
			}

		gettimeofday(&tv6, NULL); // start



		float time_trf = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float time_trs = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float time_trf_end = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);
		float time_trs_end = (float) (tv4.tv_sec-tv3.tv_sec)/(nrep+0.0)+(tv4.tv_usec-tv3.tv_usec)/(nrep*1e6);
		float time_trf_if = (float) (tv5.tv_sec-tv4.tv_sec)/(nrep+0.0)+(tv5.tv_usec-tv4.tv_usec)/(nrep*1e6);
		float time_trs_if = (float) (tv6.tv_sec-tv5.tv_sec)/(nrep+0.0)+(tv6.tv_usec-tv5.tv_usec)/(nrep*1e6);

		if(ll==0)
			{
			printf("\nnx\tnw\tny\tN\ttrf time\ttrs time\ttrf_e time\ttrs_e time\ttrf_if time\ttrs_if time\n\n");
//			fprintf(f, "\nnx\tnu\tN\tsv time\t\tsv Gflops\tsv %%\t\ttrs time\ttrs Gflops\ttrs %%\n\n");
			}
		printf("%d\t%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\n\n", nx, nw, ny, N, time_trf, time_trs, time_trf_end, time_trs_end, time_trf_if, time_trs_if);




		// moving horizon test

		// window size
		N = 20;

		double *(hhxe[N+1]);
		double *(hhxp[N+1]);
		double *(hhw[N]);
		double *(hhy[N+1]);
		double *(hhlam[N]);

		double *p_hhxe; d_zeros_align(&p_hhxe, anx, N+1);
		double *p_hhxp; d_zeros_align(&p_hhxp, anx, N+1);
		double *p_hhw; d_zeros_align(&p_hhw, anw, N);
		double *p_hhlam; d_zeros_align(&p_hhlam, anx, N);

		// shift measurements and initial prediction
		for(ii=0; ii<N; ii++)
			{
			hhxe[ii] = p_hhxe+ii*anx; //d_zeros_align(&hxe[jj], anx, 1);
			hhxp[ii] = p_hhxp+ii*anx; //d_zeros_align(&hxp[jj], anx, 1);
			hhw[ii] = p_hhw+ii*anw; //d_zeros_align(&hw[jj], anw, 1);
			hhy[ii] = hy[ii]; //d_zeros_align(&hy[jj], any, 1);
			hhlam[ii] = p_hhlam+ii*anx; //d_zeros_align(&hlam[jj], anx, 1);
			}
		hhxe[N] = p_hhxe+N*anx; //d_zeros_align(&hxe[jj], anx, 1);
		hhxp[N] = p_hhxp+N*anx; //d_zeros_align(&hxp[jj], anx, 1);
		hhy[N] = hy[N]; //d_zeros_align(&hy[jj], any, 1);

		// shift initial prediction covariance
		//for(ii=0; ii<pnx*cnl; ii++)
		//	hpLp[0][ii] = hpLp[1][ii];

		d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);
		d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hhxp, hhxe, hhw, hhy, 1, hhlam, work);

		// zero data
		for(ii=0; ii<Ns*anx; ii++)
			hxe[0][ii] = 0.0;

		for(ii=anx; ii<Ns*anx; ii++)
			hxp[0][ii] = 0.0;

		for(ii=0; ii<(Ns-1)*anw; ii++)
			hw[0][ii] = 0.0;

		for(ii=0; ii<(Ns-1)*anx; ii++)
			hlam[0][ii] = 0.0;

		// save data
		for(ii=0; ii<(N+1); ii++)
			for(jj=0; jj<nx; jj++)
				hxe[ii][jj] = hhxe[ii][jj];

		for(ii=0; ii<(N+1); ii++)
			for(jj=0; jj<nx; jj++)
				hxp[ii][jj] = hhxp[ii][jj];

		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nw; jj++)
				hw[ii][jj] = hhw[ii][jj];
		//d_print_mat(nw, N, hw[0], anw);

		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++)
				hlam[ii][jj] = hhlam[ii][jj];



		for(jj=1; jj<Ns-N; jj++)
			{

			//break;
			
			// shift measurements and initial prediction
			for(ii=0; ii<=N; ii++)
				{
				hhy[ii] = hy[ii+jj];
				}

			// shift initial prediction and relative covariance
			for(ii=0; ii<nx; ii++)
				hhxp[0][ii] = hhxp[1][ii];
			for(ii=0; ii<pnx*cnl; ii++)
				hpLp[0][ii] = hpLp[1][ii];

			//d_print_mat(nx, N+1, hhxp[0], anx);

			//d_print_pmat(nx, nx, bs, hpLp[1]+(nx+nw+pad)*bs, cnl);
			//d_print_pmat(nz, nz, bs, hpLe[1], cnf);
			//d_print_pmat(nx, nx, bs, hpLp[2]+(nx+nw+pad)*bs, cnl);
			//d_print_pmat(nz, nz, bs, hpLe[2], cnf);

			d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);
			d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hhxp, hhxe, hhw, hhy, 1, hhlam, work);

			//d_print_mat(nx, N+1, hhxp[0], anx);

			//d_print_pmat(nx, nx, bs, hpLp[0]+(nx+nw+pad)*bs, cnl);
			//d_print_pmat(nz, nz, bs, hpLe[0], cnf);
			//d_print_pmat(nx, nx, bs, hpLp[1]+(nx+nw+pad)*bs, cnl);
			//d_print_pmat(nz, nz, bs, hpLe[1], cnf);

			// save data
			for(ii=0; ii<nx; ii++)
				hxe[N+jj][ii] = hhxe[N][ii];

			for(ii=0; ii<nx; ii++)
				hxp[N+jj][ii] = hhxp[N][ii];

			if(jj<Ns-N-1)
				for(ii=0; ii<nw; ii++)
					hw[N+jj][ii] = hhw[N-1][ii];

			if(jj<Ns-N-1)
				for(ii=0; ii<nx; ii++)
					hlam[N+jj][ii] = hhlam[N-1][ii];

			//break;

			}

		// print solution
		if(PRINTRES)
			{
			printf("\nx_p\n");
			d_print_mat(nx, Ns, hxp[0], anx);
			printf("\nx_e\n");
			d_print_mat(nx, Ns, hxe[0], anx);
			//printf("\nL_e\n");
			//d_print_pmat(nx, nx, bs, hpLp[Ns-1]+(nx+nw+pad)*bs, cnl);
			}

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
		free(q);
		free(r);
		free(f);
		free(L0);
		free(pA);
		free(pG);
		free(pC);
		free(pQ);
		free(pR);
		free(work);
		free(p_hxe);
		free(p_hxp);
		free(p_hy);
		free(p_hw);
		free(p_hlam);
		free(p_hhxe);
		free(p_hhxp);
		free(p_hhw);
		free(p_hhlam);
		free(hpLp[0]);
		free(hdLp[0]);
		free(hpLe[0]);
		for(jj=0; jj<N; jj++)
			{
			free(hpLp[jj+1]);
			free(hdLp[jj+1]);
			free(hpLe[jj+1]);
			}



		} // increase size

	fprintf(f, "];\n");
	fclose(f);


	return 0;

	}



