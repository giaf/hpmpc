#include <stdlib.h>
#include <stdio.h>

#include "../include/blas_d.h"
#include "../include/kernel_d_avx.h"

void dricposv_mpc(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	if(!(nx==4 && nu==1 && N==10))
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
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(4, &pC[0], 8, 1, &pL[0], 8);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pC[16], &pC[16], &pC[20], 2, -1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pC[32], &pC[16], &pC[36], 2, -1);
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(2, &pC[20], 8, 1, &pL[20], 8);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pC[32], &pC[32], &pC[40], 2, -1);
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(0, &pC[40], 8, 1, &pL[40], 8);
	
	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{
		
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+20;
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[0], &pB[0], &pC[0], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[4], &pB[20], &pC[4], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[16], &pB[0], &pC[16], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[20], &pB[20], &pC[20], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[32], &pB[0], &pC[32], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[36], &pB[20], &pC[36], 2, 0);
	pC[33] += pB[8];
	pC[35] += pB[9];
	pC[37] += pB[24];
	pC[39] += pB[25];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[0], &pA[0], &pC[0], 2, 1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[16], &pA[0], &pC[16], 2, 1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[16], &pA[16], &pC[20], 2, 1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[32], &pA[0], &pC[32], 2, 1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[32], &pA[16], &pC[36], 2, 1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[32], &pA[32], &pC[40], 2, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(4, &pC[0], 8, 1, &pL[0], 8);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pC[16], &pC[16], &pC[20], 2, -1);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pC[32], &pC[16], &pC[36], 2, -1);
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(2, &pC[20], 8, 1, &pL[20], 8);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pC[32], &pC[32], &pC[40], 2, -1);
	kernel_dpotrf_dtrsv_dcopy_2x2_c99_lib2(0, &pC[40], 8, 1, &pL[40], 8);
		
		}
	
	/* initial stage */
	
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+20;
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[0], &pB[0], &pC[0], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[4], &pB[20], &pC[4], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[16], &pB[0], &pC[16], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[20], &pB[20], &pC[20], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(4, &pA[32], &pB[0], &pC[32], 2, 0);
	kernel_dgemm_pp_nt_2x2_atom_lib2(2, &pA[36], &pB[20], &pC[36], 2, 0);
	pC[33] += pB[8];
	pC[35] += pB[9];
	pC[37] += pB[24];
	pC[39] += pB[25];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_2x1_c99_lib2(4, &pA[0], &pA[0], &pC[0], 2, 1);
	kernel_dgemm_pp_nt_2x1_c99_lib2(4, &pA[16], &pA[0], &pC[16], 2, 1);
	kernel_dgemm_pp_nt_2x1_c99_lib2(4, &pA[32], &pA[0], &pC[32], 2, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_2x1_c99_lib2(0, &pC[0], &pC[0], &pC[0], 2, -1);
	kernel_dgemm_pp_nt_2x1_c99_lib2(0, &pC[16], &pC[0], &pC[16], 2, -1);
	kernel_dgemm_pp_nt_2x1_c99_lib2(0, &pC[32], &pC[0], &pC[32], 2, -1);
	kernel_dpotrf_dtrsv_1x1_c99_lib2(5, &pC[0], 8);
	
	
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<-2; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][33+2*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][33+2*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][33+2*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][33+2*(jj+3)];
			}
		hux[ii][0] = hpQ[ii][33];
		
		/* dgemv */
		pA = &hpQ[ii][1];
		x = &hux[ii][1];
		y = &hux[ii][0];
	kernel_dgemv_t_1_c99_lib2(4, 1, pA+0, 8, x, y+0, 1);
		
		/* dtrsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 0;
	ptrx = x + 0;
	kernel_dgemv_t_1_c99_lib2(0, 0, &ptrA[1], 8, &ptrx[1], &ptrx[0], -1);
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
			hux[ii+1][jj+1] = hpBAbt[ii][33+2*(jj+0)];
			hux[ii+1][jj+2] = hpBAbt[ii][33+2*(jj+1)];
			hux[ii+1][jj+3] = hpBAbt[ii][33+2*(jj+2)];
			hux[ii+1][jj+4] = hpBAbt[ii][33+2*(jj+3)];
			}
		
		/* dgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][1];
	kernel_dgemv_t_2_c99_lib2(5, 0, pA+0, 8, x, y+0, 1);
	kernel_dgemv_t_2_c99_lib2(5, 0, pA+4, 8, x, y+2, 1);
		
		
		}
	
	}
	
