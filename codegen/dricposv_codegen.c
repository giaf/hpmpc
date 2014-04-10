#include <stdlib.h>
#include <stdio.h>

#include "../include/blas_d.h"
#include "../include/kernel_d_avx.h"

void dricposv_mpc(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	if(!(nx==8 && nu==3 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	double *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* factorization and backward substitution */
	
	/* final stage */
	
	/* dpotrf */
	pC = hpQ[10];
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(8, &pC[0], 16, 1, &pL[0], 16);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pC[64], &pC[64], &pC[80], 4, -1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pC[128], &pC[64], &pC[144], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(4, &pC[80], 16, 1, &pL[80], 16);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pC[128], &pC[128], &pC[160], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(0, &pC[160], 16, 1, &pL[160], 16);
	
	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{
		
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+80;
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[16], &pB[80], &pC[16], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[64], &pB[0], &pC[64], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[80], &pB[80], &pC[80], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[128], &pB[0], &pC[128], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[144], &pB[80], &pC[144], 4, 0);
	pC[131] += pB[32];
	pC[135] += pB[33];
	pC[139] += pB[34];
	pC[143] += pB[35];
	pC[147] += pB[96];
	pC[151] += pB[97];
	pC[155] += pB[98];
	pC[159] += pB[99];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[64], &pA[0], &pC[64], 4, 1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[64], &pA[64], &pC[80], 4, 1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[128], &pA[0], &pC[128], 4, 1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[128], &pA[64], &pC[144], 4, 1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[128], &pA[128], &pC[160], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(8, &pC[0], 16, 1, &pL[0], 16);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pC[64], &pC[64], &pC[80], 4, -1);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pC[128], &pC[64], &pC[144], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(4, &pC[80], 16, 1, &pL[80], 16);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pC[128], &pC[128], &pC[160], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_c99_lib4(0, &pC[160], 16, 1, &pL[160], 16);
		
		}
	
	/* initial stage */
	
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+80;
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[16], &pB[80], &pC[16], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[64], &pB[0], &pC[64], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[80], &pB[80], &pC[80], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(8, &pA[128], &pB[0], &pC[128], 4, 0);
	kernel_dgemm_pp_nt_4x4_c99_lib4(4, &pA[144], &pB[80], &pC[144], 4, 0);
	pC[131] += pB[32];
	pC[135] += pB[33];
	pC[139] += pB[34];
	pC[143] += pB[35];
	pC[147] += pB[96];
	pC[151] += pB[97];
	pC[155] += pB[98];
	pC[159] += pB[99];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_4x3_c99_lib4(8, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_dgemm_pp_nt_4x3_c99_lib4(8, &pA[64], &pA[0], &pC[64], 4, 1);
	kernel_dgemm_pp_nt_4x3_c99_lib4(8, &pA[128], &pA[0], &pC[128], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_4x3_c99_lib4(0, &pC[0], &pC[0], &pC[0], 4, -1);
	kernel_dgemm_pp_nt_4x3_c99_lib4(0, &pC[64], &pC[0], &pC[64], 4, -1);
	kernel_dgemm_pp_nt_4x3_c99_lib4(0, &pC[128], &pC[0], &pC[128], 4, -1);
	kernel_dpotrf_dtrsv_3x3_c99_lib4(9, &pC[0], 16);
	
	
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<0; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][131+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][131+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][131+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][131+4*(jj+3)];
			}
		hux[ii][0] = hpQ[ii][131];
		hux[ii][1] = hpQ[ii][135];
		hux[ii][2] = hpQ[ii][139];
		
		/* dgemv */
		pA = &hpQ[ii][3];
		x = &hux[ii][3];
		y = &hux[ii][0];
	kernel_dgemv_t_2_c99_lib4(8, 1, pA+0, 16, x, y+0, 1);
	kernel_dgemv_t_1_c99_lib4(8, 1, pA+8, 16, x, y+2, 1);
		
		/* dtrsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 0;
	ptrx = x + 0;
	kernel_dgemv_t_1_c99_lib4(0, 0, &ptrA[11], 16, &ptrx[3], &ptrx[2], -1);
	ptrx[2] = (ptrx[2]) / ptrA[10];
	kernel_dgemv_t_2_c99_lib4(1, 1, &ptrA[2], 16, &ptrx[2], &ptrx[0], -1);
	ptrx[1] = (ptrx[1]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[1]*ptrx[1]) / ptrA[0];
		
		/* */
		for(jj=0; jj<0; jj+=4)
			{
			hux[ii][jj+0] = - hux[ii][jj+0];
			hux[ii][jj+1] = - hux[ii][jj+1];
			hux[ii][jj+2] = - hux[ii][jj+2];
			hux[ii][jj+3] = - hux[ii][jj+3];
			}
		hux[ii][0] = - hux[ii][0];
		hux[ii][1] = - hux[ii][1];
		hux[ii][2] = - hux[ii][2];
		
		/* */
		for(jj=0; jj<5; jj+=4)
			{
			hux[ii+1][jj+3] = hpBAbt[ii][131+4*(jj+0)];
			hux[ii+1][jj+4] = hpBAbt[ii][131+4*(jj+1)];
			hux[ii+1][jj+5] = hpBAbt[ii][131+4*(jj+2)];
			hux[ii+1][jj+6] = hpBAbt[ii][131+4*(jj+3)];
			}
		
		/* dgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][3];
	kernel_dgemv_t_8_c99_lib4(11, 0, pA+0, 16, x, y+0, 1);
		
		
		}
	
	}
	
