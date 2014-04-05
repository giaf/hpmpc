#include <stdlib.h>
#include <stdio.h>

#include "../include/blas_d.h"
#include "../include/kernel_d_avx.h"

void dricposv(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	if(!(nx==9 && nu==5 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	double *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* initial Cholesky factorization */
	/* dpotrf */
	pC = hpQ[10];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(11, 0, &pC[0], 20);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[80], &pC[160], &pC[80], &pC[96], &pC[176], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[240], &pC[80], &pC[256], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(7, 0, &pC[96], 20, 3, &pL[0], 20);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[160], &pC[240], &pC[160], &pC[192], &pC[272], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(3, 0, &pC[192], 20, 3, &pL[96], 20);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[240], &pC[240], &pC[288], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(-1, 0, &pC[288], 20, 3, &pL[192], 20);
	
	/* factorization and backward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+96;
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[0], &pA[80], &pB[0], &pC[0], &pC[80], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(5, &pA[16], &pA[96], &pB[96], &pC[16], &pC[96], 4, 0);
	corner_dtrmm_pp_nt_8x1_avx_lib4(&pA[32], &pA[112], &pB[192], &pC[32], &pC[112], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[160], &pA[240], &pB[0], &pC[160], &pC[240], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(5, &pA[176], &pA[256], &pB[96], &pC[176], &pC[256], 4, 0);
	corner_dtrmm_pp_nt_8x1_avx_lib4(&pA[192], &pA[272], &pB[192], &pC[192], &pC[272], 4);
	pC[242] += pB[36];
	pC[246] += pB[37];
	pC[250] += pB[38];
	pC[254] += pB[39];
	pC[258] += pB[116];
	pC[262] += pB[117];
	pC[266] += pB[118];
	pC[270] += pB[119];
	pC[274] += pB[196];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[0], &pA[80], &pA[0], &pC[0], &pC[80], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(9, &pA[80], &pA[80], &pC[96], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[160], &pA[240], &pA[0], &pC[160], &pC[240], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[160], &pA[240], &pA[80], &pC[176], &pC[256], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(9, &pA[160], &pA[240], &pA[160], &pC[192], &pC[272], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(9, &pA[240], &pA[240], &pC[288], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(11, 0, &pC[0], 20);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[80], &pC[160], &pC[80], &pC[96], &pC[176], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[240], &pC[80], &pC[256], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(7, 0, &pC[96], 20, 3, &pL[0], 20);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[160], &pC[240], &pC[160], &pC[192], &pC[272], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(3, 0, &pC[192], 20, 3, &pL[96], 20);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[240], &pC[240], &pC[288], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(-1, 0, &pC[288], 20, 3, &pL[192], 20);
		
		}
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<2; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][242+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][242+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][242+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][242+4*(jj+3)];
			}
		hux[ii][4] = hpQ[ii][258];
		
		/* dgemv */
		pA = &hpQ[ii][81];
		x = &hux[ii][5];
		y = &hux[ii][0];
	kernel_dgemv_t_4_avx_lib4(9, 3, pA+0, 20, x, y+0, 1);
	kernel_dgemv_t_1_avx_lib4(9, 3, pA+16, 20, x, y+4, 1);
		
		/* dtrsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 96;
	ptrx = x + 4;
	kernel_dgemv_t_1_avx_lib4(0, 0, &ptrA[1], 20, &ptrx[1], &ptrx[0], -1);
	ptrx[0] = (ptrx[0]) / ptrA[0];
	ptrA = pA + 0;
	ptrx = x  + 0;
	kernel_dgemv_t_4_avx_lib4(1, 0, ptrA+80, sda, ptrx+4, ptrx, -1);
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
		
		/* */
		for(jj=0; jj<2; jj+=4)
			{
			hux[ii][jj+0] = - hux[ii][jj+0];
			hux[ii][jj+1] = - hux[ii][jj+1];
			hux[ii][jj+2] = - hux[ii][jj+2];
			hux[ii][jj+3] = - hux[ii][jj+3];
			}
		hux[ii][4] = - hux[ii][4];
		
		/* */
		for(jj=0; jj<6; jj+=4)
			{
			hux[ii+1][jj+5] = hpBAbt[ii][242+4*(jj+0)];
			hux[ii+1][jj+6] = hpBAbt[ii][242+4*(jj+1)];
			hux[ii+1][jj+7] = hpBAbt[ii][242+4*(jj+2)];
			hux[ii+1][jj+8] = hpBAbt[ii][242+4*(jj+3)];
			}
		hux[ii+1][13] = hpBAbt[ii][274];
		
		/* dgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][5];
	kernel_dgemv_t_8_avx_lib4(14, 0, pA+0, 20, x, y+0, 1);
	kernel_dgemv_t_1_avx_lib4(14, 0, pA+32, 20, x, y+8, 1);
		
		
		}
	
	}
	
