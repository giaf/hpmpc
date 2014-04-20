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
#define PNZ S_MR*((NZ+S_MR-NU%S_MR+S_MR-1)/S_MR);



/* computes the lower triangular Cholesky factor of pC, */
/* and copies its transposed in pL                      */
void spotrf_p_scopy_p_t_code_generator(FILE *f, int n, int nna)
	{

	int i, j;
	
	const int bs = 4;
	
	const int sdc = PNZ;
	const int sdl = PNZ;
	
	j = 0;
	if(j<nna-3)
		{
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d);\n", n-j-4, j*bs+j*sdc, sdc);
	            	j += 4;     
	            	for(; j<nna-3; j+=4)
	            		{     
	            		i = j;     
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
	            		for(; i<n-4; i+=8)     
	            			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
	            			}     
#endif
	            		for(; i<n; i+=4)     
	            			{     
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
	            			}     
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d);\n", n-j-4, j*bs+j*sdc, sdc);
	            		}     
		}
	int j0 = j;
	if(j==0) // assume that n>0
		{
fprintf(f, "	kernel_spotrf_strsv_scopy_4x4_lib4(%d, &pC[%d], %d, %d, &pL[%d], %d);\n", n-j-4, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
		j += 4;
		}
	for(; j<n-3; j+=4)
		{
		i = j;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
		for(; i<n-4; i+=8)
			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
#endif
		for(; i<n; i+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_spotrf_strsv_scopy_4x4_lib4(%d, &pC[%d], %d, %d, &pL[%d], %d);\n", n-j-4, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_1x1_lib4(&pC[%d], %d, %d, &pL[%d], %d);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
			}
		else if(n-j==2)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_2x2_lib4(&pC[%d], %d, %d, &pL[%d], %d);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
			}
		else if(n-j==3)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_3x3_lib4(&pC[%d], %d, %d, &pL[%d], %d);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
			}
		}

	}



/* computes an mxn band of the lower triangular Cholesky factor of pC, supposed to be aligned */
void spotrf_p_code_generator(FILE *f, int m, int n)
	{

	int i, j;
	
	const int bs = 4;
	
	const int sdc = PNZ;

	j = 0;
	for(; j<n-3; j+=4)
		{
		i = j;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
		for(; i<m-4; i+=8)
			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
#endif
		for(; i<m; i+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d);\n", m-j-4, j*bs+j*sdc, sdc);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
			for(; i<m-4; i+=8)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
#endif
			for(; i<m; i+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
				}
fprintf(f, "	kernel_spotrf_strsv_1x1_lib4(%d, &pC[%d], %d);\n", m-j-1, j*bs+j*sdc, sdc);
			}
		else if(n-j==2)
			{
			i = j;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
			for(; i<m-4; i+=8)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
#endif
			for(; i<m; i+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
				}
fprintf(f, "	kernel_spotrf_strsv_2x2_lib4(%d, &pC[%d], %d);\n", m-j-2, j*bs+j*sdc, sdc);
			}
		else if(n-j==3)
			{
			i = j;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
			for(; i<m-4; i+=8)
				{
/*				kernel_sgemm_pp_nt_8x3_lib4(j, &pC[0+i*sdc], &pC[0+(i+4)*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], &pC[0+j*bs+(i+4)*sdc], bs, -1);*/
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, (i+4)*sdc, j*sdc, j*bs+(i+4)*sdc, bs);
				}
#endif
			for(; i<m; i+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
				}
fprintf(f, "	kernel_spotrf_strsv_3x3_lib4(%d, &pC[%d], %d);\n", m-j-3, j*bs+j*sdc, sdc);
			}
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is upper triangular                         */
void strmm_ppp_code_generator(FILE *f, int m, int n, int offset)
	{
	
	int i, j;
	
	const int bs = 4;
	
	const int sda = PNZ;
	const int sdb = PNZ;
	const int sdc = PNZ;

	if(offset%bs!=0)
	fprintf(f, "	pB = pB+%d;\n", bs*sdb+bs*bs);
	
	i = 0;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], %d, 0);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
		if(n-j==1)
			{
fprintf(f, "	corner_strmm_pp_nt_8x1_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
		else if(n-j==2)
			{
fprintf(f, "	corner_strmm_pp_nt_8x2_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
		else if(n-j==3)
			{
fprintf(f, "	corner_strmm_pp_nt_8x3_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pA[%d], &pB[%d], &pC[%d], %d, 0);\n", n-j, j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
			}
		if(n-j==1)
			{
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
			}
		else if(n-j==2)
			{
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
			}
		else if(n-j==3)
			{
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d], %d);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc, bs);
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
/* where A, C are packed with block size 4           */
void ssyrk_ppp_code_generator(FILE *f, int m, int n, int k)
	{
	
	int i, j, j_end;
	
	const int bs = 4;
	
	const int sda = PNZ;
/*	const int sdb = PNZ;*/
	const int sdc = PNZ;
	
	i = 0;
#if defined(TARGET_AVX) || defined(TARGET_SSE4) || defined(TARGET_NEON)
	if(m==n)
		{
		for(; i<m-4; i+=8)
			{
			j = 0;
			j_end = i+4;
			if(n-3<j_end)
				j_end = n-3;
			for(; j<j_end; j+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
			if(j<n-3)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, (i+4)*sda, j*sda, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==1)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, (i+4)*sda, j*sda, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==2)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, (i+4)*sda, j*sda, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==3)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, (i+4)*sda, j*sda, j*bs+(i+4)*sdc, bs);
				}
			}
		}
	else
		{
		for(; i<m-4; i+=8)
			{
			j = 0;
			j_end = i+4;
			if(n-3<j_end)
				j_end = n-3;
			for(; j<j_end; j+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
			if(j<n-3)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==1)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x1_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==2)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x2_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
			else if(n-j==3)
				{
/*fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], %d, 1);\n", k, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);*/
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, (i+4)*sda, j*sda, j*bs+(i+4)*sdc, bs);
				}
			}
		}
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
			}
		if(j<n)
			{
			if(n-j==1)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			else if(n-j==2)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			else if(n-j==3)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			}
		}
#else
	for(; i<m; i+=4)
		{
		j = 0;
		j_end = i+4;
		if(j_end>n)
			{
			for(; j<n-3; j+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			if(n-j==1)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			else if(n-j==2)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			else if(n-j==3)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			}
		else
			{
			for(; j<j_end; j+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pA[%d], &pA[%d], &pC[%d], %d, 1);\n", k, i*sda, j*sda, j*bs+i*sdc, bs);
				}
			}
		}
#endif

/*	fprintf(f, "	\n");*/

	}



void sgemv_p_t_code_generator(FILE *f, int n, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int nna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
#if !defined(TARGET_ATOM)
	for(; j<m-7; j+=8)
		{
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, %d, pA+%d, %d, x, y+%d, %d);\n", n, nna, j*bs, sda, j, alg);
		}
#endif
	for(; j<m-3; j+=4)
		{
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, %d, pA+%d, %d, x, y+%d, %d);\n", n, nna, j*bs, sda, j, alg);
		}
	for(; j<m-1; j+=2)
		{
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, pA+%d, %d, x, y+%d, %d);\n", n, nna, j*bs, sda, j, alg);
		}
	for(; j<m; j++)
		{
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, pA+%d, %d, x, y+%d, %d);\n", n, nna, j*bs, sda, j, alg);
		}

/*	fprintf(f, "	\n");*/
	}



void strsv_p_t_code_generator(FILE *f, int n)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	int ri, qi;
	
/*	fprintf(f, "	double *ptrA, *ptrx;\n");*/
	
	// clean up stuff at the end
	j = 0;
fprintf(f, "	ptrA = pA + %d;\n", qn*bs*(sda+bs));
fprintf(f, "	ptrx = x + %d;\n", qn*bs);

	for(; j<rn%2; j++)
		{
		i = rn-1-j;
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &ptrA[%d], %d, &ptrx[%d], &ptrx[%d], -1);\n", j, j, i+1+bs*(i+0), sda, i+1, i);
fprintf(f, "	ptrx[%d] = (ptrx[%d]) / ptrA[%d];\n", i+0, i+0, i+0+bs*(i+0));
		}
	for(; j<rn; j+=2)
		{
		i = rn-2-j;
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, &ptrA[%d], %d, &ptrx[%d], &ptrx[%d], -1);\n", j, j, i+2+bs*(i+0), sda, i+2, i);
fprintf(f, "	ptrx[%d] = (ptrx[%d]) / ptrA[%d];\n", i+1, i+1, (i+1)+bs*(i+1));
fprintf(f, "	ptrx[%d] = (ptrx[%d] - ptrA[%d]*ptrx[%d]) / ptrA[%d];\n", i+0, i+0, (i+1)+bs*(i+0), i+1, (i+0)+bs*(i+0));
		}

	// blocks of 8
	j = 0;
#if !defined(TARGET_ATOM)
	for(; j<qn-1; j+=2)
		{
		
		// all 4 rows
fprintf(f, "	ptrA = pA + %d;\n", (qn-j-2)*bs*(sda+bs));
fprintf(f, "	ptrx = x  + %d;\n", (qn-j-2)*bs);
		

		// correct
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, 0, ptrA+%d, sda, ptrx+8, ptrx, -1);\n", rn+j*bs, 2*bs*sda);
		

		// last 4 rows
fprintf(f, "	ptrA = pA + %d;\n", (qn-j-1)*bs*(sda+bs));
fprintf(f, "	ptrx = x  + %d;\n", (qn-j-1)*bs);

		// solve
fprintf(f, "	ptrx[3] = (ptrx[3]) / ptrA[%d];\n", 3+bs*3);
fprintf(f, "	ptrx[2] = (ptrx[2] - ptrA[%d]*ptrx[3]) / ptrA[%d];\n", 3+bs*2, 2+bs*2);
fprintf(f, "	ptrx[1] = (ptrx[1] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2]) / ptrA[%d];\n", 3+bs*1, 2+bs*1, 1+bs*1);
fprintf(f, "	ptrx[0] = (ptrx[0] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2] - ptrA[%d]*ptrx[1]) / ptrA[%d];\n", 3+bs*0, 2+bs*0, 1+bs*0, 0+bs*0);

		// first 4 rows
fprintf(f, "	ptrA = pA + %d;\n", (qn-j-2)*bs*(sda+bs));
fprintf(f, "	ptrx = x  + %d;\n", (qn-j-2)*bs);

		// correct
fprintf(f, "	kernel_sgemv_t_4_lib4(4, 0, ptrA+%d, sda, ptrx+4, ptrx, -1);\n", bs*sda);

		// solve
fprintf(f, "	ptrx[3] = (ptrx[3]) / ptrA[%d];\n", 3+bs*3);
fprintf(f, "	ptrx[2] = (ptrx[2] - ptrA[%d]*ptrx[3]) / ptrA[%d];\n", 3+bs*2, 2+bs*2);
fprintf(f, "	ptrx[1] = (ptrx[1] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2]) / ptrA[%d];\n", 3+bs*1, 2+bs*1, 1+bs*1);
fprintf(f, "	ptrx[0] = (ptrx[0] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2] - ptrA[%d]*ptrx[1]) / ptrA[%d];\n", 3+bs*0, 2+bs*0, 1+bs*0, 0+bs*0);

		}
#endif	
	// blocks of 4
	for(; j<qn; j++)
		{
		
		// first 4 rows
fprintf(f, "	ptrA = pA + %d;\n", (qn-j-1)*bs*(sda+bs));
fprintf(f, "	ptrx = x  + %d;\n", (qn-j-1)*bs);
		
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 0, ptrA+%d, sda, ptrx+4, ptrx, -1);\n", rn+j*bs, bs*sda);
fprintf(f, "	ptrx[3] = (ptrx[3]) / ptrA[%d];\n", 3+bs*3);
fprintf(f, "	ptrx[2] = (ptrx[2] - ptrA[%d]*ptrx[3]) / ptrA[%d];\n", 3+bs*2, 2+bs*2);
fprintf(f, "	ptrx[1] = (ptrx[1] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2]) / ptrA[%d];\n", 3+bs*1, 2+bs*1, 1+bs*1);
fprintf(f, "	ptrx[0] = (ptrx[0] - ptrA[%d]*ptrx[3] - ptrA[%d]*ptrx[2] - ptrA[%d]*ptrx[1]) / ptrA[%d];\n", 3+bs*0, 2+bs*0, 1+bs*0, 0+bs*0);

		}

/*	fprintf(f, "	\n");*/

	}
