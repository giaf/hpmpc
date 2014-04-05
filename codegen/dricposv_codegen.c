#include <stdlib.h>
#include <stdio.h>

#include "../include/blas_d.h"
#include "../include/kernel_d_avx.h"

void dricposv(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	if(!(nx==16 && nu==5 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	double *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* initial Cholesky factorization */
	/* dpotrf */
	pC = hpQ[10];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(18, 0, &pC[0], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[112], &pC[224], &pC[112], &pC[128], &pC[240], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[336], &pC[448], &pC[112], &pC[352], &pC[464], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[560], &pC[112], &pC[576], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(14, 0, &pC[128], 28, 3, &pL[0], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[224], &pC[336], &pC[224], &pC[256], &pC[368], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[448], &pC[560], &pC[224], &pC[480], &pC[592], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(10, 0, &pC[256], 28, 3, &pL[128], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[336], &pC[448], &pC[336], &pC[384], &pC[496], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[560], &pC[336], &pC[608], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(6, 0, &pC[384], 28, 3, &pL[256], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[448], &pC[560], &pC[448], &pC[512], &pC[624], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(2, 0, &pC[512], 28, 3, &pL[384], 28);
	kernel_dgemm_pp_nt_4x2_avx_lib4(20, &pC[560], &pC[560], &pC[640], 4, -1);
	corner_dpotrf_dtrsv_dcopy_2x2_sse_lib4(&pC[640], 28, 3, &pL[512], 28);
	
	/* factorization and backward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+128;
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[0], &pA[112], &pB[0], &pC[0], &pC[112], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pA[16], &pA[128], &pB[128], &pC[16], &pC[128], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pA[32], &pA[144], &pB[256], &pC[32], &pC[144], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pA[48], &pA[160], &pB[384], &pC[48], &pC[160], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[224], &pA[336], &pB[0], &pC[224], &pC[336], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pA[240], &pA[352], &pB[128], &pC[240], &pC[352], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pA[256], &pA[368], &pB[256], &pC[256], &pC[368], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pA[272], &pA[384], &pB[384], &pC[272], &pC[384], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pB[0], &pC[448], &pC[560], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pA[464], &pA[576], &pB[128], &pC[464], &pC[576], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pA[480], &pA[592], &pB[256], &pC[480], &pC[592], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pA[496], &pA[608], &pB[384], &pC[496], &pC[608], 4, 0);
	pC[561] += pB[64];
	pC[565] += pB[65];
	pC[569] += pB[66];
	pC[573] += pB[67];
	pC[577] += pB[176];
	pC[581] += pB[177];
	pC[585] += pB[178];
	pC[589] += pB[179];
	pC[593] += pB[288];
	pC[597] += pB[289];
	pC[601] += pB[290];
	pC[605] += pB[291];
	pC[609] += pB[400];
	pC[613] += pB[401];
	pC[617] += pB[402];
	pC[621] += pB[403];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[0], &pA[112], &pA[0], &pC[0], &pC[112], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(16, &pA[112], &pA[112], &pC[128], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[224], &pA[336], &pA[0], &pC[224], &pC[336], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[224], &pA[336], &pA[112], &pC[240], &pC[352], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[224], &pA[336], &pA[224], &pC[256], &pC[368], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(16, &pA[336], &pA[336], &pC[384], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pA[0], &pC[448], &pC[560], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pA[112], &pC[464], &pC[576], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pA[224], &pC[480], &pC[592], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pA[336], &pC[496], &pC[608], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pA[448], &pA[560], &pA[448], &pC[512], &pC[624], 4, 1);
	kernel_dgemm_pp_nt_4x2_avx_lib4(16, &pA[560], &pA[560], &pC[640], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(18, 0, &pC[0], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[112], &pC[224], &pC[112], &pC[128], &pC[240], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[336], &pC[448], &pC[112], &pC[352], &pC[464], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[560], &pC[112], &pC[576], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(14, 0, &pC[128], 28, 3, &pL[0], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[224], &pC[336], &pC[224], &pC[256], &pC[368], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[448], &pC[560], &pC[224], &pC[480], &pC[592], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(10, 0, &pC[256], 28, 3, &pL[128], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[336], &pC[448], &pC[336], &pC[384], &pC[496], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[560], &pC[336], &pC[608], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(6, 0, &pC[384], 28, 3, &pL[256], 28);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[448], &pC[560], &pC[448], &pC[512], &pC[624], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(2, 0, &pC[512], 28, 3, &pL[384], 28);
	kernel_dgemm_pp_nt_4x2_avx_lib4(20, &pC[560], &pC[560], &pC[640], 4, -1);
	corner_dpotrf_dtrsv_dcopy_2x2_sse_lib4(&pC[640], 28, 3, &pL[512], 28);
		
		}
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<2; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][561+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][561+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][561+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][561+4*(jj+3)];
			}
		hux[ii][4] = hpQ[ii][577];
		
		/* dgemv */
		pA = &hpQ[ii][113];
		x = &hux[ii][5];
		y = &hux[ii][0];
	kernel_dgemv_t_4_avx_lib4(16, 3, pA+0, 28, x, y+0, 1);
	kernel_dgemv_t_1_avx_lib4(16, 3, pA+16, 28, x, y+4, 1);
		
		/* dtrsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 128;
	ptrx = x + 4;
	kernel_dgemv_t_1_avx_lib4(0, 0, &ptrA[1], 28, &ptrx[1], &ptrx[0], -1);
	ptrx[0] = (ptrx[0]) / ptrA[0];
	ptrA = pA + 0;
	ptrx = x  + 0;
	kernel_dgemv_t_4_avx_lib4(1, 0, ptrA+112, sda, ptrx+4, ptrx, -1);
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
		for(jj=0; jj<13; jj+=4)
			{
			hux[ii+1][jj+5] = hpBAbt[ii][561+4*(jj+0)];
			hux[ii+1][jj+6] = hpBAbt[ii][561+4*(jj+1)];
			hux[ii+1][jj+7] = hpBAbt[ii][561+4*(jj+2)];
			hux[ii+1][jj+8] = hpBAbt[ii][561+4*(jj+3)];
			}
		
		/* dgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][5];
	kernel_dgemv_t_8_avx_lib4(21, 0, pA+0, 28, x, y+0, 1);
	kernel_dgemv_t_8_avx_lib4(21, 0, pA+32, 28, x, y+8, 1);
		
		
		}
	
	}
	
