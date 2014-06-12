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

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/blas_d.h"
#include "../include/blas_s.h"
#include "../include/block_size.h"



int main()
	{
		
	// maximum frequency of the processor
	const float GHz_max = 2.3; //3.6; //2.9;

	// maximum flops per cycle, single precision
#if defined(TARGET_X64_AVX)
	const float d_flops_max = 16;
#elif defined(TARGET_X64_SSE3) || defined(TARGET_AMD_SSE3)
	const float d_flops_max = 8;
#elif defined(TARGET_CORTEX_A9)
	const float d_flops_max = 4;
#elif defined(TARGET_CORTEX_A15)
	const float d_flops_max = 8;
#elif defined(TARGET_X86_ATOM)
	const float d_flops_max = 4;
#elif defined(TARGET_POWERPC_G2)
	const float d_flops_max = 2;
#elif defined(TARGET_C99_4X4)
	const float d_flops_max = 2;
#elif defined(TARGET_C99_2X2)
	const float d_flops_max = 2;
#endif
	
	FILE *f;
	f = fopen("./test_problems/results/test_blas.m", "w"); // a

#if defined(TARGET_X64_AVX)
	fprintf(f, "C = 's_x64_avx';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X64_SSE3) || defined(TARGET_AMD_SSE3)
	fprintf(f, "C = 's_x64_sse3';\n");
	fprintf(f, "\n");
#elif defined(TARGET_CORTEX_A9)
	fprintf(f, "C = 's_ARM_cortex_A9';\n");
	fprintf(f, "\n");
#elif defined(TARGET_CORTEX_A15)
	fprintf(f, "C = 's_ARM_cortex_A15';\n");
	fprintf(f, "\n");
#elif defined(TARGET_X86_ATOM)
	fprintf(f, "C = 's_x86_atom';\n");
	fprintf(f, "\n");
#elif defined(TARGET_POWERPC_G2)
	fprintf(f, "C = 's_PowerPC_G2';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_4X4)
	fprintf(f, "C = 's_c99_2x2';\n");
	fprintf(f, "\n");
#elif defined(TARGET_C99_2X2)
	fprintf(f, "C = 's_c99_4x4';\n");
	fprintf(f, "\n");
#endif

	fprintf(f, "A = [%f %f];\n", GHz_max, d_flops_max);
	fprintf(f, "\n");

	fprintf(f, "B = [\n");



	int i, j, rep, ll;
	
	const int bsd = D_MR; //d_get_mr();
	const int bss = S_MR; //s_get_mr();
	
	int info = 0;
	
	printf("\nn\tGflops dgemm %%\tGflops dsyrk %%\tGflops dtrmm %%\tGflops dpotrf %%\tGflops dgemv_n%%\tGflops dgemv_t%%\tGflops dsymv %%\tGflops dtrmv_n%%\tGflops dtrmv_t%%\tGflops dmvmv%%\n\n");
	
/*	int nn[] = {4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 144, 160, 192, 256};*/
/*	int nnrep[] = {10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100, 100};*/
/*	*/
/*	for(ll=0; ll<17; ll++)*/
	
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
	
	for(ll=0; ll<75; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll];
	
		double *A; d_zeros(&A, n, n);
		double *B; d_zeros(&B, n, n);
		double *C; d_zeros(&C, n, n);
		float *sA; s_zeros(&sA, n, n);
		float *sB; s_zeros(&sB, n, n);
		float *sC; s_zeros(&sC, n, n);
	
		for(i=0; i<n*n; i++)
			A[i] = i;
	
		for(i=0; i<n; i++)
			B[i*(n+1)] = 1;
	
		for(i=0; i<n*n; i++)
			sA[i] = i;
	
		for(i=0; i<n; i++)
			sB[i*(n+1)] = 1;
	
		int pns = ((n+bss-1)/bss)*bss;	

		float *pA; s_zeros_align(&pA, pns, pns);
		float *pB; s_zeros_align(&pB, pns, pns);
		float *pC; s_zeros_align(&pC, pns, pns);
		float *pD; s_zeros_align(&pD, pns, pns);
		float *pL; s_zeros_align(&pL, pns, pns);
		float *x; s_zeros_align(&x, pns, 1);
		float *y; s_zeros_align(&y, pns, 1);
		float *x2; s_zeros_align(&x2, pns, 1);
		float *y2; s_zeros_align(&y2, pns, 1);
	
		s_cvt_mat2pmat(n, n, 0, bss, sA, n, pA, pns);
		s_cvt_mat2pmat(n, n, 0, bss, sB, n, pB, pns);
		s_cvt_mat2pmat(n, n, 0, bss, sB, n, pD, pns);
	
		for(i=0; i<pns*pns; i++) pC[i] = -1;
		
		for(i=0; i<pns; i++) x[i] = 1;
		for(i=0; i<pns; i++) x2[i] = 1;

		/* timing */
		struct timeval tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			sgemm_ppp_nt_lib(n, n, n, pA, pns, pB, pns, pC, pns, 0);
			}

		gettimeofday(&tv0, NULL); // start
	
		for(rep=0; rep<nrep; rep++)
			{

			sgemm_ppp_nt_lib(n, n, n, pA, pns, pB, pns, pC, pns, 0);

			}
	
		gettimeofday(&tv1, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			ssyrk_ppp_lib(n, n, n, pA, pns, pC, pns);*/

			}
	
		gettimeofday(&tv2, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			strmm_ppp_lib(n, n, 0, pA, pns, pB, pns, pC, pns);*/

			}
	
		gettimeofday(&tv3, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			spotrf_p_lib(n, 0, pD, pns, &info);*/

			}
	
		gettimeofday(&tv4, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			sgemv_p_n_lib(n, n, 0, pA, pns, x, y, 0);*/

			}
	
		gettimeofday(&tv5, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			sgemv_p_t_lib(n, n, 0, pA, pns, x, y, 0);*/

			}
	
		gettimeofday(&tv6, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			strmv_p_n_lib(n, 0, pA, pns, x, y, 0);*/

			}
	
		gettimeofday(&tv7, NULL); // stop


		for(rep=0; rep<nrep; rep++)
			{

/*			strmv_p_t_lib(n, 0, pA, pns, x, y, 0);*/

			}
	
		gettimeofday(&tv8, NULL); // stop


		for(rep=0; rep<nrep; rep++)
			{

/*			strsv_p_n_lib(n, pB, pns, x);*/

			}
	
		gettimeofday(&tv9, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			strsv_p_t_lib(n, pB, pns, x);*/

			}
	
		gettimeofday(&tv10, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			ssymv_p_lib(n, 0, pA, pns, x, y, 0);*/

			}
	
		gettimeofday(&tv11, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

/*			smvmv_p_lib(n, n, 0, pA, pns, x, y, x2, y2, 0);*/

			}
	
		gettimeofday(&tv12, NULL); // stop



		float Gflops_max = d_flops_max * GHz_max;

		float time_dgemm = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float flop_dgemm = 2.0*n*n*n;
		float Gflops_dgemm = 1e-9*flop_dgemm/time_dgemm;

		float time_dsyrk = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float flop_dsyrk = 1.0*n*n*n;
		float Gflops_dsyrk = 1e-9*flop_dsyrk/time_dsyrk;

		float time_dtrmm = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);
		float flop_dtrmm = 1.0*n*n*n;
		float Gflops_dtrmm = 1e-9*flop_dtrmm/time_dtrmm;
	
		float time_dpotrf = (float) (tv4.tv_sec-tv3.tv_sec)/(nrep+0.0)+(tv4.tv_usec-tv3.tv_usec)/(nrep*1e6);
		float flop_dpotrf = 1.0/3.0*n*n*n;
		float Gflops_dpotrf = 1e-9*flop_dpotrf/time_dpotrf;

		float time_dgemv_n = (float) (tv5.tv_sec-tv4.tv_sec)/(nrep+0.0)+(tv5.tv_usec-tv4.tv_usec)/(nrep*1e6);
		float flop_dgemv_n = 2.0*n*n;
		float Gflops_dgemv_n = 1e-9*flop_dgemv_n/time_dgemv_n;

		float time_dgemv_t = (float) (tv6.tv_sec-tv5.tv_sec)/(nrep+0.0)+(tv6.tv_usec-tv5.tv_usec)/(nrep*1e6);
		float flop_dgemv_t = 2.0*n*n;
		float Gflops_dgemv_t = 1e-9*flop_dgemv_t/time_dgemv_t;

		float time_dtrmv_n = (float) (tv7.tv_sec-tv6.tv_sec)/(nrep+0.0)+(tv7.tv_usec-tv6.tv_usec)/(nrep*1e6);
		float flop_dtrmv_n = 1.0*n*n;
		float Gflops_dtrmv_n = 1e-9*flop_dtrmv_n/time_dtrmv_n;

		float time_dtrmv_t = (float) (tv8.tv_sec-tv7.tv_sec)/(nrep+0.0)+(tv8.tv_usec-tv7.tv_usec)/(nrep*1e6);
		float flop_dtrmv_t = 1.0*n*n;
		float Gflops_dtrmv_t = 1e-9*flop_dtrmv_t/time_dtrmv_t;

		float time_dtrsv_n = (float) (tv9.tv_sec-tv8.tv_sec)/(nrep+0.0)+(tv9.tv_usec-tv8.tv_usec)/(nrep*1e6);
		float flop_dtrsv_n = 1.0*n*n;
		float Gflops_dtrsv_n = 1e-9*flop_dtrsv_n/time_dtrsv_n;

		float time_dtrsv_t = (float) (tv10.tv_sec-tv9.tv_sec)/(nrep+0.0)+(tv10.tv_usec-tv9.tv_usec)/(nrep*1e6);
		float flop_dtrsv_t = 1.0*n*n;
		float Gflops_dtrsv_t = 1e-9*flop_dtrsv_t/time_dtrsv_t;

		float time_dsymv = (float) (tv11.tv_sec-tv10.tv_sec)/(nrep+0.0)+(tv11.tv_usec-tv10.tv_usec)/(nrep*1e6);
		float flop_dsymv = 2.0*n*n;
		float Gflops_dsymv = 1e-9*flop_dsymv/time_dsymv;

		float time_dmvmv = (float) (tv12.tv_sec-tv11.tv_sec)/(nrep+0.0)+(tv12.tv_usec-tv11.tv_usec)/(nrep*1e6);
		float flop_dmvmv = 4.0*n*n;
		float Gflops_dmvmv = 1e-9*flop_dmvmv/time_dmvmv;

		printf("%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_dgemm, 100.0*Gflops_dgemm/Gflops_max, Gflops_dsyrk, 100.0*Gflops_dsyrk/Gflops_max, Gflops_dtrmm, 100.0*Gflops_dtrmm/Gflops_max, Gflops_dpotrf, 100.0*Gflops_dpotrf/Gflops_max, Gflops_dgemv_n, 100.0*Gflops_dgemv_n/Gflops_max, Gflops_dgemv_t, 100.0*Gflops_dgemv_t/Gflops_max, Gflops_dtrmv_n, 100.0*Gflops_dtrmv_n/Gflops_max, Gflops_dtrmv_t, 100.0*Gflops_dtrmv_t/Gflops_max, Gflops_dtrsv_n, 100.0*Gflops_dtrsv_n/Gflops_max, Gflops_dtrsv_t, 100.0*Gflops_dtrsv_t/Gflops_max, Gflops_dsymv, 100.0*Gflops_dsymv/Gflops_max, Gflops_dmvmv, 100.0*Gflops_dmvmv/Gflops_max);

	fprintf(f, "%d\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\t%7.2f\n", n, Gflops_dgemm, 100.0*Gflops_dgemm/Gflops_max, Gflops_dsyrk, 100.0*Gflops_dsyrk/Gflops_max, Gflops_dtrmm, 100.0*Gflops_dtrmm/Gflops_max, Gflops_dpotrf, 100.0*Gflops_dpotrf/Gflops_max, Gflops_dgemv_n, 100.0*Gflops_dgemv_n/Gflops_max, Gflops_dgemv_t, 100.0*Gflops_dgemv_t/Gflops_max, Gflops_dtrmv_n, 100.0*Gflops_dtrmv_n/Gflops_max, Gflops_dtrmv_t, 100.0*Gflops_dtrmv_t/Gflops_max, Gflops_dtrsv_n, 100.0*Gflops_dtrsv_n/Gflops_max, Gflops_dtrsv_t, 100.0*Gflops_dtrsv_t/Gflops_max, Gflops_dsymv, 100.0*Gflops_dsymv/Gflops_max, Gflops_dmvmv, 100.0*Gflops_dmvmv/Gflops_max);

		free(A);
		free(B);
		free(pA);
		free(pB);
		free(pC);
		free(pD);
		free(pL);
		free(x);
		free(y);
		free(x2);
		free(y2);
		
		}

	printf("\n");

	fprintf(f, "];\n");
	fclose(f);

	return 0;
	
	}
