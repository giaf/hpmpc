/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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

#include <stdio.h>

#include "../problem_size.h"
#include "../include/block_size.h"

#define NZ NX+NU+1
#define PNZ D_MR*((NZ+D_MR-NU%D_MR+D_MR-1)/D_MR);



/* computes the lower triangular Cholesky factor of pC, */
/* and copies its transposed in pL                      */
void dpotrf_p_dcopy_p_t_code_generator(FILE *f, int n, int nna)
	{

	int i, j;
	
	const int bs = 2;
	
	const int sdc = PNZ;
	const int sdl = PNZ;
	
	j = 0;
	if(j<nna-1)
		{
fprintf(f, "	kernel_dpotrf_dtrsv_2x2_lib2(%d, &pC[%d], %d, info);\n", n-j-2, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
	            	j += 2;     
	            	for(; j<nna-1; j+=2)
	            		{     
	            		i = j;     
	            		for(; i<n; i+=2)     
	            			{     
fprintf(f, "	kernel_dgemm_pp_nt_2x2_lib2(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
	            			}     
fprintf(f, "	kernel_dpotrf_dtrsv_2x2_lib2(%d, &pC[%d], %d, info);\n", n-j-2, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
	            		}     
		}
	int j0 = j;
	if(j==0) // assume that n>0
		{
fprintf(f, "	kernel_dpotrf_dtrsv_dcopy_2x2_lib2(%d, &pC[%d], %d, %d, &pL[%d], %d, info);\n", n-j-2, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
		j += 2;
		}
	for(; j<n-1; j+=2)
		{
		i = j;
		for(; i<n; i+=2)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x2_lib2(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_dpotrf_dtrsv_dcopy_2x2_lib2(%d, &pC[%d], %d, %d, &pL[%d], %d, info);\n", n-j-2, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
		}
	if(n-j==1)
		{
		i = j;
fprintf(f, "	kernel_dgemm_pp_nt_2x1_lib2(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_dpotrf_dtrsv_dcopy_1x1_lib2(&pC[%d], %d, %d, &pL[%d], %d, info);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
		}

	}



/* computes an mxn band of the lower triangular Cholesky factor of pC, supposed to be aligned */
void dpotrf_p_code_generator(FILE *f, int m, int n)
	{

	int i, j;
	
	const int bs = 2;
	
	const int sdc = PNZ;

	j = 0;
	for(; j<n-1; j+=2)
		{
		i = j;
		for(; i<m; i+=2)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x2_lib2(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_dpotrf_dtrsv_2x2_lib2(%d, &pC[%d], %d, info);\n", m-j-2, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
		}
	if(n-j==1)
		{
		i = j;
		for(; i<m; i+=2)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x1_lib2(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_dpotrf_dtrsv_1x1_lib2(%d, &pC[%d], %d, info);\n", m-j-1, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 2,    */
/* and B is upper triangular                         */
void dtrmm_ppp_code_generator(FILE *f, int m, int n, int offset)
	{
	
	int i, j;
	
	const int bs = 2;
	
	const int sda = PNZ;
	const int sdb = PNZ;
	const int sdc = PNZ;

	if(offset%bs!=0)
fprintf(f, "	pB = pB+%d;\n", bs*sdb+bs*bs);
	
	i = 0;
	for(; i<m; i+=2)
		{
		j = 0;
		for(; j<n-1; j+=2)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x2_lib2(%d, &pA[%d], &pB[%d], &pC[%d], %d, 0);\n", n-j, j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
			}
		if(n-j==1)
			{
fprintf(f, "	corner_dtrmm_pp_nt_2x1_lib2(&pA[%d], &pB[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
			}
		}

	// add to the last row
	for(j=0; j<n; j++)
		{
fprintf(f, "	pC[%d] += pB[%d];\n", (m-1)%bs+j*bs+((m-1)/bs)*bs*sdc, j%bs+n*bs+(j/bs)*bs*sdb);
		
		}

/*	fprintf(f, "	\n");*/

	}



/* preforms                                          */
/* C  = A * A'                                       */
/* where A, C are packed with block size 2           */
void dsyrk_ppp_code_generator(FILE *f, int m, int n, int k)
	{
	
	int i, j, j_end;
	
	const int bs = 2;
	
	const int sda = PNZ;
/*	const int sdb = PNZ;*/
	const int sdc = PNZ;
	
	i = 0;
	for(; i<m; i+=2)
		{
		j = 0;
		j_end = i+2;
		if(n-1<j_end)
			j_end = n-1;
		for(; j<j_end; j+=2)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x2_lib2(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
			}
		if(n-j==1)
			{
fprintf(f, "	kernel_dgemm_pp_nt_2x1_lib2(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
			}
		}

/*	fprintf(f, "	\n");*/

	}



void dgemv_p_n_code_generator(FILE *f, int n, int m, int offset, int alg)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int j;

	int idxA = 0;
	int idxy = 0;

	int nna = (bs-offset%bs)%bs;

	j = 0;
	if(nna==1)
		{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += 1 + (sda-1)*bs;
		idxy += 1;
		j++;
		}
	for(; j<n-1; j+=2)
		{
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += sda*bs;
		idxy += bs;
		}
	for(; j<n; j++)
		{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += 1;
		idxy += 1;
		}

/*	fprintf(f, "	\n");*/
	}



void dgemv_p_t_code_generator(FILE *f, int n, int m, int offset, int alg)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int nna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
	for(; j<m-1; j+=2)
		{
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, %d, &pA[%d], %d, x, &y[%d], %d);\n", n, nna, j*bs, sda, j, alg);
		}
	for(; j<m; j++)
		{
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, %d, &pA[%d], %d, x, &y[%d], %d);\n", n, nna, j*bs, sda, j, alg);
		}

/*	fprintf(f, "	\n");*/
	}



void dtrmv_p_n_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	int idxA = 0;
	int idxy = 0;

	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
			idxA += 1 + (sda-1)*bs;
			idxy += 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
			idxA += sda*bs;
			idxy += bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
			idxA += 1;
			idxy += 1;
			}
		}
	else
		{
		j=0;
		if(mna==1)
				{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
			idxA += 1 + (sda-1)*bs;
			idxy += 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
			idxA += sda*bs;
			idxy += bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
			idxA += 1;
			idxy += 1;
			}
		}

	}



// !!! x and y can not be the same vector !!!
void dtrmv_p_t_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;
	int mmax = m;
	
	int j;
	
	int idxA = 0;
	int idxx = 0;
	int idxy = 0;

	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax, idxA, sda, idxx, idxy, alg);
			idxA += 1 + sda*bs;
			idxx += 1;
			idxy += 1;
			mmax -= 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-1, idxA+1, sda, idxx+1, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy, idxA, idxx);
			idxA += bs*sda + bs*bs;
			idxx += bs;
			idxy += bs;
			mmax -= bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax, mmax, idxA, sda, idxx, idxy, alg);
			idxA += 1 + bs;
			idxx += 1;
			idxy += 1;
			mmax -= 1;
			}
		}
	else
		{
		j=0;
		if(mna==1)
			{
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 1, &pA[%d], %d, &x[%d], &y[%d], -1);\n", mmax, idxA, sda, idxx, idxy);
			idxA += 1 + sda*bs;
			idxx += 1;
			idxy += 1;
			mmax -= 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, 1, &pA[%d], %d, &x[%d], &y[%d], -1);\n", mmax-1, idxA+1, sda, idxx+1, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy, idxA, idxx);
			idxA += bs*sda + bs*bs;
			idxx += bs;
			idxy += bs;
			mmax -= bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, %d, &pA[%d], %d, &x[%d], &y[%d], -1);\n", mmax, mmax, idxA, sda, idxx, idxy);
			idxA += 1 + bs;
			idxx += 1;
			idxy += 1;
			mmax -= 1;
			}
		}

	}



void dsymv_p_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 2;

	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;

	int j;
	
	int idxA = 0;
	int idxx = 0;
	int idxy = 0;

/*	double *ptrA, *ptrx;*/
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], &x[%d], &y[%d], %d);\n", j, idxA, idxx, idxy, alg);
fprintf(f, "	y[%d] += pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j);
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], 1);\n", m-j-1, idxA+j*bs+1, sda, idxx+j+1, idxy);
			idxA += 1 + (sda-1)*bs;
			idxy += 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], &x[%d], &y[%d], %d);\n", j, idxA, idxx, idxy, alg);
fprintf(f, "	y[%d] += pA[%d]*x[%d] + pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j, idxA+j*bs+1, idxx+j+1);
fprintf(f, "	y[%d] += pA[%d]*x[%d];\n", idxy+1, idxA+j*bs+1, idxx+j, idxA+j*bs+1+bs*1, idxx+j+1);
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], 1);\n", m-j-2, idxA+j*bs+sda*bs, sda, idxx+j+2, idxy);
			idxA += sda*bs;
			idxy += bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], &x[%d], &y[%d], %d);\n", j, idxA, idxx, idxy, alg);
fprintf(f, "	y[%d] += pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j);
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], 1);\n", m-j-1, idxA+j*bs+1, sda, idxx+j+1, idxy);
			idxA += 1;
			idxy += 1;
			}
		}
	else // alg==-1
		{
		j=0;
		if(mna==1)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], &x[%d], &y[%d], -1);\n", j, idxA, idxx, idxy);
fprintf(f, "	y[%d] -= pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j);
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], -1);\n", m-j-1, idxA+j*bs+1, sda, idxx+j+1, idxy);
			idxA += 1 + (sda-1)*bs;
			idxy += 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], &x[%d], &y[%d], -1);\n", j, idxA, idxx, idxy);
fprintf(f, "	y[%d] -= pA[%d]*x[%d] + pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j, idxA+j*bs+1, idxx+j+1);
fprintf(f, "	y[%d] -= pA[%d]*x[%d];\n", idxy+1, idxA+j*bs+1, idxx+j, idxA+j*bs+1+bs*1, idxx+j+1);
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], -1);\n", m-j-2, idxA+j*bs+sda*bs, sda, idxx+j+2, idxy);
			idxA += sda*bs;
			idxy += bs;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], &x[%d], &y[%d], -1);\n", j, idxA, idxx, idxy);
fprintf(f, "	y[%d] -= pA[%d]*x[%d];\n", idxy, idxA+j*bs, idxx+j);
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, 0, &pA[%d], %d, &x[%d], &y[%d], -1);\n", m-j-1, idxA+j*bs+1, sda, idxx+j+1, idxy);
			idxA += 1;
			idxy += 1;
			}
		}

	}



void dtrsv_p_n_code_generator(FILE *f, int n)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int j;
	
	int idxA = 0;
	int idxAd = 0;
	int idxx = 0;

/*	double *ptrA, *ptrAd, *ptrx;*/

	// blocks of 2 (pA is supposed to be properly aligned)
/*	ptrA  = pA;*/
/*	ptrAd = pA;*/
/*	ptrx  = x;*/

	j = 0;
	for(; j<n-1; j+=2)
		{
		// correct
fprintf(f, "	kernel_dgemv_n_2_lib2(%d, &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+1, idxx+1, idxx, idxAd+1, idxAd+1+bs*1);
		
		idxA  += bs*sda;
		idxAd += bs*(sda+bs);
		idxx  += bs;
		}
	for(; j<n; j++)
		{
		// correct
fprintf(f, "	kernel_dgemv_n_1_lib2(%d, &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
		
		idxA  += 1;
		idxAd += bs+1;
		idxx  += 1;
		}

	}



// TODO idxA & idxx
void dtrsv_p_t_code_generator(FILE *f, int n)
	{
	
	const int bs = 2;
	
	const int sda = PNZ;

	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	int ri, qi;
	
	// clean up stuff at the end
	j = 0;

	if(rn==1)
		{
		i = rn-1-j;
fprintf(f, "	kernel_dgemv_t_1_lib2(%d, %d, &pA[%d], %d, &x[%d], &x[%d], -1);\n", j, j, i+1+bs*i+qn*bs*(sda+bs), sda, i+1+qn*bs, i+qn*bs);
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", i+qn*bs, i+qn*bs, i+bs*i+qn*bs*(sda+bs) );
		}

	// blocks of 2
	for(; j<qn; j++)
		{
		
		// first 2 rows
fprintf(f, "	kernel_dgemv_t_2_lib2(%d, 0, pA+%d, sda, x+%d+2, x+%d, -1);\n", rn+j*bs, bs*sda+(qn-j-1)*bs*(sda+bs), (qn-j-1)*bs, (qn-j-1)*bs);
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", (qn-j-1)*bs+1, (qn-j-1)*bs+1, 1+bs*1+(qn-j-1)*bs*(sda+bs));
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d]) / pA[%d];\n", (qn-j-1)*bs, (qn-j-1)*bs, 1+(qn-j-1)*bs*(sda+bs), (qn-j-1)*bs+1, (qn-j-1)*bs*(sda+bs));

		}

/*	fprintf(f, "	\n");*/

	}
