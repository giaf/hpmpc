#include <stdlib.h>
#include <stdio.h>

#include "../include/kernel_d_avx.h"

void sricposv_mpc(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL)
	{
	if(!(nx==4 && nu==1 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	float *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* factorization and backward substitution */
	
	/* final stage */
	
	/*spotrf */
	pC = hpQ[10];
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(2, &pC[0], 12, 3, &pL[0], 12);
	kernel_sgemm_pp_nt_4x2_atom_lib4(4, &pC[48], &pC[48], &pC[64], 4, -1);
	corner_spotrf_strsv_scopy_2x2_c99_lib4(&pC[64], 12, 3, &pL[64], 12);
	
	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{
		
		/* strmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+64;
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[48], &pB[0], &pC[48], 4, 0);
	pC[49] += pB[16];
	pC[53] += pB[17];
	pC[57] += pB[18];
	pC[61] += pB[19];
		
		/* ssyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_sgemm_pp_nt_4x2_atom_lib4(4, &pA[0], &pA[48], &pC[16], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[48], &pA[0], &pC[48], 4, 1);
	kernel_sgemm_pp_nt_4x2_atom_lib4(4, &pA[48], &pA[48], &pC[64], 4, 1);
		
		/* spotrf */
		pC = hpQ[9-ii];
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(2, &pC[0], 12, 3, &pL[0], 12);
	kernel_sgemm_pp_nt_4x2_atom_lib4(4, &pC[48], &pC[48], &pC[64], 4, -1);
	corner_spotrf_strsv_scopy_2x2_c99_lib4(&pC[64], 12, 3, &pL[64], 12);
		
		}
	
	/* initial stage */
	
		/* strmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+64;
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pA[48], &pB[0], &pC[48], 4, 0);
	pC[49] += pB[16];
	pC[53] += pB[17];
	pC[57] += pB[18];
	pC[61] += pB[19];
		
		/* ssyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x1_c99_lib4(4, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_sgemm_pp_nt_4x1_c99_lib4(4, &pA[48], &pA[0], &pC[48], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x1_c99_lib4(0, &pC[0], &pC[0], &pC[0], 4, -1);
	kernel_sgemm_pp_nt_4x1_c99_lib4(0, &pC[48], &pC[0], &pC[48], 4, -1);
	kernel_spotrf_strsv_1x1_c99_lib4(5, &pC[0], 12);
	
	
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<-2; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][49+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][49+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][49+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][49+4*(jj+3)];
			}
		hux[ii][0] = hpQ[ii][49];
		
		/* sgemv */
		pA = &hpQ[ii][1];
		x = &hux[ii][1];
		y = &hux[ii][0];
	kernel_sgemv_t_1_sse_lib4(4, 3, pA+0, 12, x, y+0, 1);
		
		/* strsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 0;
	ptrx = x + 0;
	kernel_sgemv_t_1_sse_lib4(0, 0, &ptrA[1], 12, &ptrx[1], &ptrx[0], -1);
	ptrx[0] = (ptrx[0]) / ptrA[0];
		
		/* */
		for(jj=0; jj<-2; jj+=4)
			{
			hux[ii][jj+0] = - hux[ii][jj+0];
			hux[ii][jj+1] = - hux[ii][jj+1];
			hux[ii][jj+2] = - hux[ii][jj+2];
			hux[ii][jj+3] = - hux[ii][jj+3];
			}
		hux[ii][0] = - hux[ii][0];
		
		/* */
		for(jj=0; jj<1; jj+=4)
			{
			hux[ii+1][jj+1] = hpBAbt[ii][49+4*(jj+0)];
			hux[ii+1][jj+2] = hpBAbt[ii][49+4*(jj+1)];
			hux[ii+1][jj+3] = hpBAbt[ii][49+4*(jj+2)];
			hux[ii+1][jj+4] = hpBAbt[ii][49+4*(jj+3)];
			}
		
		/* sgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][1];
	kernel_sgemv_t_4_atom_lib4(5, 0, pA+0, 12, x, y+0, 1);
		
		
		}
	
	}
	
