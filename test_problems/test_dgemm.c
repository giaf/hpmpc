#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "../include/aux_d.h"
/*#include "../include/aux_s.h"*/
#include "../include/blas_d.h"
/*#include "../include/blas_lib_s.h"*/
#include "../include/block_size.h"



int main()
	{
		
	// maximum frequency of the processor
	const float GHz_max = 2.9; //3.6; //2.9;
	// maximum flops per cycle, double precision
	const float d_flops_max = 8; //4; //2;
	// maximum flops per cycle, single precision
	const float s_flops_max = 8; //16; //8; //2;
	
	int i, j, rep, ll;
	
	const int bsd = D_MR; //d_get_mr();
/*	const int bss = S_MR; //s_get_mr();*/
	
	printf("\nn\tGflops c99\t\t%%\tGflops d\t\t%%\tGflops s\t\t%%\n\n");
	
/*	int nn[] = {4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 144, 160, 192, 256};*/
/*	int nnrep[] = {10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100, 100};*/
/*	*/
/*	for(ll=0; ll<17; ll++)*/
	
	int nn[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332};
	int nnrep[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
	
	for(ll=0; ll<75; ll++)

		{

		int n = nn[ll];
		int nrep = nnrep[ll];
	
		double *A; d_zeros(&A, n, n);
		double *B; d_zeros(&B, n, n);
		double *C; d_zeros(&C, n, n);
/*		float *sA; s_zeros(&sA, n, n);*/
/*		float *sB; s_zeros(&sB, n, n);*/
/*		float *sC; s_zeros(&sC, n, n);*/
	
		for(i=0; i<n*n; i++)
			A[i] = i;
	
		for(i=0; i<n; i++)
			B[i*(n+1)] = 1;
	
/*		for(i=0; i<n*n; i++)*/
/*			sA[i] = i;*/
/*	*/
/*		for(i=0; i<n; i++)*/
/*			sB[i*(n+1)] = 1;*/
	
		int pnd = ((n+bsd-1)/bsd)*bsd;	
/*		int pns = ((n+bss-1)/bss)*bss;	*/

		double *pA; d_zeros_align(&pA, pnd, pnd);
		double *pB; d_zeros_align(&pB, pnd, pnd);
		double *pC; d_zeros_align(&pC, pnd, pnd);
		double *pD; d_zeros_align(&pD, pnd, pnd);
		double *pL; d_zeros_align(&pL, pnd, pnd);
/*		float *spA; s_zeros_align(&spA, pns, pns);*/
/*		float *spB; s_zeros_align(&spB, pns, pns);*/
/*		float *spC; s_zeros_align(&spC, pns, pns);*/
	
		d_cvt_mat2pmat(n, n, 0, bsd, A, n, pA, pnd);
		d_cvt_mat2pmat(n, n, 0, bsd, B, n, pB, pnd);
		d_cvt_mat2pmat(n, n, 0, bsd, B, n, pD, pnd);
/*		s_cvt_mat2pmat(n, n, 0, bss, sA, n, spA, pns);*/
/*		s_cvt_mat2pmat(n, n, 0, bss, sB, n, spB, pns);*/
	
		for(i=0; i<pnd*pnd; i++) pC[i] = -1;
/*		for(i=0; i<pns*pns; i++) spC[i] = -1;*/
		
/*		openblas_set_num_threads(1);*/
/*		char cn = 'n'; double alpha = 1.0; double beta = 0.0; float salpha = 1.0; float sbeta = 0.0;*/
	
		/* timing */
		struct timeval tv0, tv1, tv2, tv3, tv4;

		/* warm up */
		for(rep=0; rep<nrep; rep++)
			{
			dgemm_ppp_nt_lib(n, n, n, pA, pnd, pB, pnd, pC, pnd, 0);
/*			dgemm_pup_nn_lib(n, n, n, pA, pnd, B, n, pC, pnd, 0);*/
			}

		gettimeofday(&tv0, NULL); // start
	
		for(rep=0; rep<nrep; rep++)
			{

//			dgemm_nt_c99_lib2(n, pB, pA, pC, pnd);
//			dgemm_nt_lib_blk(n, n, n, pA, pnd, pB, pnd, pC, pnd, 0);
			dgemm_ppp_nt_lib(n, n, n, pA, pnd, pB, pnd, pC, pnd, 0);
/*			dgemm_pup_nn_lib(n, n, n, pA, pnd, B, n, pC, pnd, 0);*/
//			dgemm_(&cn, &cn, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);

			}
	
		gettimeofday(&tv1, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

//			dgemm_nt_lib2(n, pB, pA, pC, pnd);
/*			dgemm_ppp_nt_lib(n, n, n, pA, pnd, pB, pnd, pC, pnd, 0);*/
/*			dgemm_pup_nn_lib(n, n, n, pA, pnd, B, n, pC, pnd, 0);*/
//			dgemm_(&cn, &cn, &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
			dsyrk_ppp_lib(n, n, pA, pnd, pC, pnd);

			}
	
		gettimeofday(&tv2, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

//			sgemm_nt_lib4(n, spB, spA, spC, pns);
/*			sgemm_nt_lib(n, n, n, spA, pns, spB, pns, spC, pns, 0);*/
//			sgemm_(&cn, &cn, &n, &n, &n, &salpha, sA, &n, sB, &n, &sbeta, sC, &n);
/*			dtrmm_pup_nn_lib(n, n, pA, pnd, B, n, pC, pnd);*/
/*			dgemm_pup_nn_lib(n, n, n, pA, pnd, B, n, pC, pnd, 0);*/
			dtrmm_ppp_lib(n, n, 0, pA, pnd, pB, pnd, pC, pnd);
			}
	
		gettimeofday(&tv3, NULL); // stop

		for(rep=0; rep<nrep; rep++)
			{

//			sgemm_nt_lib4(n, spB, spA, spC, pns);
/*			sgemm_nt_lib(n, n, n, spA, pns, spB, pns, spC, pns, 0);*/
//			sgemm_(&cn, &cn, &n, &n, &n, &salpha, sA, &n, sB, &n, &sbeta, sC, &n);
/*			dtrmm_pup_nn_lib(n, n, pA, pnd, B, n, pC, pnd);*/
			dpotrf_p_dcopy_p_t_lib(n, 0, pD, pnd, pL, pnd);

			}
	
		gettimeofday(&tv4, NULL); // stop

		float time_d_c99 = (float) (tv1.tv_sec-tv0.tv_sec)/(nrep+0.0)+(tv1.tv_usec-tv0.tv_usec)/(nrep*1e6);
		float flop_d_c99 = 2.0*n*n*n;
		float Gflops_d_c99 = 1e-9*flop_d_c99/time_d_c99;
		float Gflops_max_d_c99 = d_flops_max * GHz_max;

		float time_d = (float) (tv2.tv_sec-tv1.tv_sec)/(nrep+0.0)+(tv2.tv_usec-tv1.tv_usec)/(nrep*1e6);
		float flop_d = 1.0*n*n*n;
		float Gflops_d = 1e-9*flop_d/time_d;
		float Gflops_max_d = d_flops_max * GHz_max;

		float time_s = (float) (tv3.tv_sec-tv2.tv_sec)/(nrep+0.0)+(tv3.tv_usec-tv2.tv_usec)/(nrep*1e6);
		float flop_s = 1.0*n*n*n;
		float Gflops_s = 1e-9*flop_s/time_s;
		float Gflops_max_s = s_flops_max * GHz_max;
	
		float time_s_asm = (float) (tv4.tv_sec-tv3.tv_sec)/(nrep+0.0)+(tv4.tv_usec-tv3.tv_usec)/(nrep*1e6);
		float flop_s_asm = 1.0/3.0*n*n*n;
		float Gflops_s_asm = 1e-9*flop_s_asm/time_s_asm;
		float Gflops_max_s_asm = s_flops_max * GHz_max;

		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", n, Gflops_d_c99, 100.0*Gflops_d_c99/Gflops_max_d_c99, Gflops_d, 100.0*Gflops_d/Gflops_max_d, Gflops_s, 100.0*Gflops_s/Gflops_max_s, Gflops_s_asm, 100.0*Gflops_s_asm/Gflops_max_s_asm);

		free(A);
		free(B);
		free(pA);
		free(pB);
		free(pC);
/*		free(sA);*/
/*		free(sB);*/
/*		free(spA);*/
/*		free(spB);*/
/*		free(spC);*/
		
		}

	printf("\n");

	return 0;
	
	}
