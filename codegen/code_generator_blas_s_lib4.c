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
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d, info);\n", n-j-4, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
	            	j += 4;     
	            	for(; j<nna-3; j+=4)
	            		{     
	            		i = j;     
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
	            		for(; i<n-4; i+=8)     
	            			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
	            			}     
#endif
	            		for(; i<n; i+=4)     
	            			{     
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
	            			}     
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d, info);\n", n-j-4, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
	            		}     
		}
	int j0 = j;
	if(j==0) // assume that n>0
		{
fprintf(f, "	kernel_spotrf_strsv_scopy_4x4_lib4(%d, &pC[%d], %d, %d, &pL[%d], %d, info);\n", n-j-4, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
		j += 4;
		}
	for(; j<n-3; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
		for(; i<n-4; i+=8)
			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
#endif
		for(; i<n; i+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_spotrf_strsv_scopy_4x4_lib4(%d, &pC[%d], %d, %d, &pL[%d], %d, info);\n", n-j-4, j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_1x1_lib4(&pC[%d], %d, %d, &pL[%d], %d, info);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
			}
		else if(n-j==2)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_2x2_lib4(&pC[%d], %d, %d, &pL[%d], %d, info);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
			}
		else if(n-j==3)
			{
			i = j;
fprintf(f, "	kernel_sgemm_pp_nt_4x3_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
fprintf(f, "	corner_spotrf_strsv_scopy_3x3_lib4(&pC[%d], %d, %d, &pL[%d], %d, info);\n", j*bs+j*sdc, sdc, (bs-nna%bs)%bs, (j-j0)*bs+((j-j0)/bs)*bs*sdc, sdl);
fprintf(f, "	if(*info!=0) return;\n");
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
		for(; i<m-4; i+=8)
			{
fprintf(f, "	kernel_sgemm_pp_nt_8x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
			}
#endif
		for(; i<m; i+=4)
			{
fprintf(f, "	kernel_sgemm_pp_nt_4x4_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
			}
fprintf(f, "	kernel_spotrf_strsv_4x4_lib4(%d, &pC[%d], %d, info);\n", m-j-4, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
		}
	if(j<n)
		{
		if(n-j==1)
			{
			i = j;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
			for(; i<m-4; i+=8)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
#endif
			for(; i<m; i+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x1_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
				}
fprintf(f, "	kernel_spotrf_strsv_1x1_lib4(%d, &pC[%d], %d, info);\n", m-j-1, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
			}
		else if(n-j==2)
			{
			i = j;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
			for(; i<m-4; i+=8)
				{
fprintf(f, "	kernel_sgemm_pp_nt_8x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, (i+4)*sdc, j*sdc, j*bs+i*sdc, j*bs+(i+4)*sdc, bs);
				}
#endif
			for(; i<m; i+=4)
				{
fprintf(f, "	kernel_sgemm_pp_nt_4x2_lib4(%d, &pC[%d], &pC[%d], &pC[%d], %d, -1);\n", j, i*sdc, j*sdc, j*bs+i*sdc, bs);
				}
fprintf(f, "	kernel_spotrf_strsv_2x2_lib4(%d, &pC[%d], %d, info);\n", m-j-2, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
			}
		else if(n-j==3)
			{
			i = j;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
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
fprintf(f, "	kernel_spotrf_strsv_3x3_lib4(%d, &pC[%d], %d, info);\n", m-j-3, j*bs+j*sdc, sdc);
fprintf(f, "	if(*info!=0) return;\n");
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_SSE3) || defined(TARGET_NEON)
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



void sgemv_p_n_code_generator(FILE *f, int n, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int j;

	int idxA = 0;
	int idxy = 0;

	int nna = (bs-offset%bs)%bs;

	if(nna>n) // it is always nna < bs , thus n<bs !!!!!
		{
		if(nna%2==1)
			{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
			idxA += 1;
			idxy  += 1;
			n  -= 1;
			}
		j = 0;
		for(; j<n-1; j+=2)
			{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
			idxA += 2;
			idxy  += 2;
			}
		for(; j<n; j++)
			{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
			idxA += 1;
			idxy  += 1;
			}
		return;
		}
	j=0;
	if(nna>0) // it can be nna = {1, 2, 3}
		{
		if(nna%2==1)
			{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
			idxA += 1;
			idxy  += 1;
			j++;
			}
		if(nna%4>=2)
			{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
			idxA += 2;
			idxy  += 2;
			j+=2;
			}
		idxA += (sda-1)*bs;
		}
#if !defined(TARGET_X86_ATOM)
	for(; j<n-7; j+=8)
		{
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], x, &y[%d], %d);\n", m, idxA, idxA+sda*bs, idxy, alg);
		idxA += 2*sda*bs;
		idxy  += 2*bs;
		}
#endif
	for(; j<n-3; j+=4)
		{
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += sda*bs;
		idxy  += bs;
		}
	for(; j<n-1; j+=2)
		{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += 2;
		idxy  += 2;
		}
	for(; j<n; j++)
		{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", m, idxA, idxy, alg);
		idxA += 1;
		idxy  += 1;
		}

	}



void sgemv_p_t_code_generator(FILE *f, int n, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int nna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
#if !defined(TARGET_X86_ATOM)
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



void strmv_p_n_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;
	
	int idxA = 0;
	int idxy = 0;

	int j;
	
	if(alg==0 || alg==1)
		{
		if(mna>m) // it is always mna < bs , thus m<bs !!!!!
			{
			if(mna%2==1)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
				idxA += 1;
				idxy  += 1;
				m  -= 1;
				}
			j = 0;
			for(; j<m-1; j+=2)
				{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
				idxA += 2;
				idxy  += 2;
				}
			for(; j<m; j++)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
				idxA += 1;
				idxy  += 1;
				}
			return;
			}
		j=0;
		if(mna>0)
			{
			for(; j<mna%2; j++)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
				idxA += 1;
				idxy  += 1;
				}
			for(; j<mna; j+=2)
				{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
				idxA += 2;
				idxy  += 2;
				}
			idxA += (sda-1)*bs;
			}
#if !defined(TARGET_X86_ATOM)
		for(; j<m-7; j+=8)
			{
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxA+sda*bs, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+1)*bs+2, j+1, idxA+(j+2)*bs+2, j+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+1)*bs+3, j+1, idxA+(j+2)*bs+3, j+2, idxA+(j+3)*bs+3, j+3);
			idxA += sda*bs;
			idxy  += bs;

fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[%d], &y[%d], 1);\n", 4, idxA+bs*(j+1), j+1, idxy);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+5)*bs+1, j+5);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+5)*bs+2, j+5, idxA+(j+6)*bs+2, j+6);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+5)*bs+3, j+5, idxA+(j+6)*bs+3, j+6, idxA+(j+7)*bs+3, j+7);

			idxA += sda*bs;
			idxy  += bs;
			}
#endif
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+1)*bs+2, j+1, idxA+(j+2)*bs+2, j+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+1)*bs+3, j+1, idxA+(j+2)*bs+3, j+2, idxA+(j+3)*bs+3, j+3);
			idxA += sda*bs;
			idxy  += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
			idxA += 2;
			idxy  += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], %d);\n", j+1, idxA, idxy, alg);
			idxA += 1;
			idxy  += 1;
			}
		}
	else
		{
		if(mna>m) // it is always mna < bs , thus m<bs !!!!!
			{
			if(mna%2==1)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
				idxA += 1;
				idxy  += 1;
				m  -= 1;
				}
			j = 0;
			for(; j<m-1; j+=2)
				{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
				idxA += 2;
				idxy  += 2;
				}
			for(; j<m; j++)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
				idxA += 1;
				idxy  += 1;
				}
			return;
			}
		j=0;
		if(mna>0)
			{
			for(; j<mna%2; j++)
				{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
				idxA += 1;
				idxy  += 1;
				}
			for(; j<mna; j+=2)
				{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
				idxA += 2;
				idxy  += 2;
				}
			idxA += (sda-1)*bs;
			}
#if !defined(TARGET_X86_ATOM)
		for(; j<m-7; j+=8)
			{
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxA+sda*bs, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+1)*bs+2, j+1, idxA+(j+2)*bs+2, j+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+1)*bs+3, j+1, idxA+(j+2)*bs+3, j+2, idxA+(j+3)*bs+3, j+3);
			idxA += sda*bs;
			idxy  += bs;

fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[%d], &y[%d], -1);\n", 4, idxA+bs*(j+1), j+1, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+5)*bs+1, j+5);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+5)*bs+2, j+5, idxA+(j+6)*bs+2, j+6);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+5)*bs+3, j+5, idxA+(j+6)*bs+3, j+6, idxA+(j+7)*bs+3, j+7);

			idxA += sda*bs;
			idxy  += bs;
			}
#endif
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+2, idxA+(j+1)*bs+2, j+1, idxA+(j+2)*bs+2, j+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+3, idxA+(j+1)*bs+3, j+1, idxA+(j+2)*bs+3, j+2, idxA+(j+3)*bs+3, j+3);
			idxA += sda*bs;
			idxy  += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+1, idxA+(j+1)*bs+1, j+1);
			idxA += 2;
			idxy  += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], x, &y[%d], -1);\n", j+1, idxA, idxy);
			idxA += 1;
			idxy  += 1;
			}
		}

	}



// !!! x and y can not be the same vector !!!
void strmv_p_t_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
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
		if(mna>0)
			{
			for(; j<mna; j++)
				{
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-j, mna-j, idxA+j*bs+j, sda, idxx+j, idxy+j, alg);
				}
			idxA += j + (sda-1)*bs + j*bs;
			idxx  += j;
			idxy  += j;
			mmax -= j;
			}
#if !defined(TARGET_X86_ATOM)
		for(; j<m-7; j+=8)
			{
			idxA += bs*sda;
			
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-7, idxA+3, sda, idxx+7, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+4, idxA+0+bs*4, idxx+4, idxA+1+bs*4, idxx+5, idxA+2+bs*4, idxx+6);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+5, idxA+1+bs*5, idxx+5, idxA+2+bs*5, idxx+6);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+6, idxA+2+bs*6, idxx+6);

			idxA -= bs*sda;

fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", 4, idxA+3, sda, idxx+3, idxy, 1);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+0, idxA+0+bs*0, idxx+0, idxA+1+bs*0, idxx+1, idxA+2+bs*0, idxx+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+1, idxA+1+bs*1, idxx+1, idxA+2+bs*1, idxx+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+2, idxA+2+bs*2, idxx+2);

			idxA += 2*bs*sda + 2*bs*bs;
			idxx  += 2*bs;
			idxy  += 2*bs;
			mmax -= 2*bs;
			}
#endif
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-3, idxA+3, sda, idxx+3, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+0, idxA+0+bs*0, idxx+0, idxA+1+bs*0, idxx+1, idxA+2+bs*0, idxx+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+1, idxA+1+bs*1, idxx+1, idxA+2+bs*1, idxx+2);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy+2, idxA+2+bs*2, idxx+2);
			idxA += bs*sda + bs*bs;
			idxx  += bs;
			idxy  += bs;
			mmax -= bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-1, mmax-1, idxA+1, sda, idxx+1, idxy, alg);
fprintf(f, "	y[%d] += pA[%d] * x[%d];\n", idxy, idxA, idxx);
			idxA += 2 + 2*bs;
			idxx  += 2;
			idxy  += 2;
			mmax -= 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax, mmax, idxA, sda, idxx, idxy, alg);
			idxA += 1 + bs;
			idxx  += 1;
			idxy  += 1;
			mmax -= 1;
			}
		}
	else
		{
		j=0;
		if(mna>0)
			{
			for(; j<mna; j++)
				{
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-j, mna-j, idxA+j*bs+j, sda, idxx+j, idxy+j, -1);
				}
			idxA += j + (sda-1)*bs + j*bs;
			idxx  += j;
			idxy  += j;
			mmax -= j;
			}
#if !defined(TARGET_X86_ATOM)
		for(; j<m-7; j+=8)
			{
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", 4, idxA+3, sda, idxx+3, idxy, -1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+0, idxA+0+bs*0, idxx+0, idxA+1+bs*0, idxx+1, idxA+2+bs*0, idxx+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+1, idxA+1+bs*1, idxx+1, idxA+2+bs*1, idxx+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+2, idxA+2+bs*2, idxx+2);

			idxA += bs*sda;
			
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-7, idxA+3, sda, idxx+7, idxy, -1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+4, idxA+0+bs*4, idxx+4, idxA+1+bs*4, idxx+5, idxA+2+bs*4, idxx+6);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+5, idxA+1+bs*5, idxx+5, idxA+2+bs*5, idxx+6);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+6, idxA+2+bs*6, idxx+6);

			idxA += bs*sda + 2*bs*bs;
			idxx  += 2*bs;
			idxy  += 2*bs;
			mmax -= 2*bs;
			}
#endif
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 1, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-3, idxA+3, sda, idxx+3, idxy, -1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+0, idxA+0+bs*0, idxx+0, idxA+1+bs*0, idxx+1, idxA+2+bs*0, idxx+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d] + pA[%d] * x[%d];\n", idxy+1, idxA+1+bs*1, idxx+1, idxA+2+bs*1, idxx+2);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy+2, idxA+2+bs*2, idxx+2);
			idxA += bs*sda + bs*bs;
			idxx  += bs;
			idxy  += bs;
			mmax -= bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax-1, mmax-1, idxA+1, sda, idxx+1, idxy, -1);
fprintf(f, "	y[%d] -= pA[%d] * x[%d];\n", idxy, idxA, idxx);
			idxA += 2 + 2*bs;
			idxx  += 2;
			idxy  += 2;
			mmax -= 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], %d);\n", mmax, mmax, idxA, sda, idxx, idxy, -1);
			idxA += 1 + bs;
			idxx  += 1;
			idxy  += 1;
			mmax -= 1;
			}
		}

	}



void ssymv_p_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;
	if(m<mna)
		mna = m;
	
	int j, jj;
	
	int idxA = 0;
	int idxx_n = 0;
	int idxy_n = 0;
	int idxx_t = 0;
	int idxy_t = 0;
	
	if(alg==0)
		{
fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", m-3);
fprintf(f, "		{\n");
fprintf(f, "		y[jj+%d] = 0.0;\n", 0);
fprintf(f, "		y[jj+%d] = 0.0;\n", 1);
fprintf(f, "		y[jj+%d] = 0.0;\n", 2);
fprintf(f, "		y[jj+%d] = 0.0;\n", 3);
fprintf(f, "		}\n");
		for(jj=0; jj<m%4; jj++)
			{
fprintf(f, "	y[%d] = 0.0;\n", (m/4)*4+jj );
			}
fprintf(f, "	\n");
		}
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna>0)
			{
			for( ; j<mna; j++)
				{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, 1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
				idxA  += 1;
				idxy_n += 1;
				idxx_t += 1;
				}
			idxA  += (sda-1)*bs;
			}
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_ssymv_4_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, 1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += sda*bs;
			idxy_n += bs;
			idxx_t += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_ssymv_2_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, 1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 2;
			idxy_n += 2;
			idxx_t += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, 1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 1;
			idxy_n += 1;
			idxx_t += 1;
			}
		}
	else // alg==-1
		{
		j=0;
		if(mna>0)
			{
			for( ; j<mna; j++)
				{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, -1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
				idxA  += 1;
				idxy_n += 1;
				idxx_t += 1;
				}
			idxA  += (sda-1)*bs;
			}
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_ssymv_4_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, -1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += sda*bs;
			idxy_n += bs;
			idxx_t += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_ssymv_2_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, -1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 2;
			idxy_n += 2;
			idxx_t += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x[%d], &y[%d], &x[%d], &y[%d], 1, -1);\n", j, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 1;
			idxy_n += 1;
			idxx_t += 1;
			}
		}

	}



void smvmv_p_code_generator(FILE *f, int m, int n, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int mna = (bs-offset%bs)%bs;
	if(m<mna)
		mna = m;
	
	int j, jj;
	
	int idxA = 0;
	int idxx_n = 0;
	int idxy_n = 0;
	int idxx_t = 0;
	int idxy_t = 0;

	if(alg==0)
		{
fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", m-3);
fprintf(f, "		{\n");
fprintf(f, "		y_n[jj+%d] = 0.0;\n", 0);
fprintf(f, "		y_n[jj+%d] = 0.0;\n", 1);
fprintf(f, "		y_n[jj+%d] = 0.0;\n", 2);
fprintf(f, "		y_n[jj+%d] = 0.0;\n", 3);
fprintf(f, "		}\n");
		for(jj=0; jj<m%4; jj++)
			{
fprintf(f, "	y_n[%d] = 0.0;\n", (m/4)*4+jj );
			}
fprintf(f, "	\n");
fprintf(f, "	for(jj=0; jj<%d; jj+=4)\n", n-3);
fprintf(f, "		{\n");
fprintf(f, "		y_t[jj+%d] = 0.0;\n", 0);
fprintf(f, "		y_t[jj+%d] = 0.0;\n", 1);
fprintf(f, "		y_t[jj+%d] = 0.0;\n", 2);
fprintf(f, "		y_t[jj+%d] = 0.0;\n", 3);
fprintf(f, "		}\n");
		for(jj=0; jj<n%4; jj++)
			{
fprintf(f, "	y_t[%d] = 0.0;\n", (n/4)*4+jj );
			}
fprintf(f, "	\n");
		}
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna>0)
			{
			for( ; j<mna; j++)
				{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, 1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
				idxA  += 1;
				idxy_n += 1;
				idxx_t += 1;
				}
			idxA  += (sda-1)*bs;
			}
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_ssymv_4_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, 1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += sda*bs;
			idxy_n += bs;
			idxx_t += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_ssymv_2_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, 1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 2;
			idxy_n += 2;
			idxx_t += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, 1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 1;
			idxy_n += 1;
			idxx_t += 1;
			}
		}
	else // alg==-1
		{
		j=0;
		if(mna>0)
			{
			for( ; j<mna; j++)
				{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, -1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
				idxA  += 1;
				idxy_n += 1;
				idxx_t += 1;
				}
			idxA  += (sda-1)*bs;
			}
		for(; j<m-3; j+=4)
			{
fprintf(f, "	kernel_ssymv_4_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, -1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += sda*bs;
			idxy_n += bs;
			idxx_t += bs;
			}
		for(; j<m-1; j+=2)
			{
fprintf(f, "	kernel_ssymv_2_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, -1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 2;
			idxy_n += 2;
			idxx_t += 2;
			}
		for(; j<m; j++)
			{
fprintf(f, "	kernel_ssymv_1_lib4(%d, &pA[%d], &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, -1);\n", n, idxA, idxx_n, idxy_n, idxx_t, idxy_t);
			idxA  += 1;
			idxy_n += 1;
			idxx_t += 1;
			}
		}

	}



void strsv_p_n_code_generator(FILE *f, int n)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int j;
	
	// blocks of 4 (pA is supposed to be properly aligned)
	int idxA = 0;
	int idxAd = 0;
	int idxx = 0;

	j = 0;
#if !defined(TARGET_X86_ATOM)
	for(; j<n-7; j+=8)
		{
		// correct
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxA+bs*sda, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+1, idxx+1, idxx, idxAd+1, idxAd+1+bs*1);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+2, idxx+2, idxx, idxAd+2, idxx+1, idxAd+2+bs*1, idxAd+2+bs*2);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+3, idxx+3, idxx, idxAd+3, idxx+1, idxAd+3+bs*1, idxx+2, idxAd+3+bs*2, idxAd+3+bs*3);

		// correct
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[%d], &x[%d], -1);\n", 4, idxA+bs*sda, idxx, idxx+4);

		idxA  += bs*sda;
		idxAd += bs*(sda+bs);
		idxx  += bs;

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+1, idxx+1, idxx, idxAd+1, idxAd+1+bs*1);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+2, idxx+2, idxx, idxAd+2, idxx+1, idxAd+2+bs*1, idxAd+2+bs*2);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+3, idxx+3, idxx, idxAd+3, idxx+1, idxAd+3+bs*1, idxx+2, idxAd+3+bs*2, idxAd+3+bs*3);

		idxA  += bs*sda;
		idxAd += bs*(sda+bs);
		idxx  += bs;

		}
#endif
	// blocks of 4
	for(; j<n-3; j+=4)
		{
		// correct
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+1, idxx+1, idxx, idxAd+1, idxAd+1+bs*1);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+2, idxx+2, idxx, idxAd+2, idxx+1, idxAd+2+bs*1, idxAd+2+bs*2);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d] - x[%d] * pA[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+3, idxx+3, idxx, idxAd+3, idxx+1, idxAd+3+bs*1, idxx+2, idxAd+3+bs*2, idxAd+3+bs*3);

		idxA  += bs*sda;
		idxAd += bs*(sda+bs);
		idxx  += bs;

		}
	for(; j<n-1; j+=2)
		{
		// correct
fprintf(f, "	kernel_sgemv_n_2_lib4(%d, &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
fprintf(f, "	x[%d] = (x[%d] - x[%d] * pA[%d]) / pA[%d];\n", idxx+1, idxx+1, idxx, idxAd+1, idxAd+1+bs*1);
		
		idxA  += 2;
		idxAd += 2*bs+2;
		idxx  += 2;
		}
	for(; j<n; j++)
		{
		// correct
fprintf(f, "	kernel_sgemv_n_1_lib4(%d, &pA[%d], &x[0], &x[%d], -1);\n", j, idxA, idxx);

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx, idxx, idxAd);
		
		idxA  += 1;
		idxAd += bs+1;
		idxx  += 1;
		}

	}



void strsv_p_t_code_generator(FILE *f, int n)
	{
	
	const int bs = 4;
	
	const int sda = PNZ;

	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	int ri, qi;
	
	int idxA = 0;
	int idxx = 0;

	// clean up stuff at the end
	j = 0;
	idxA = qn*bs*(sda+bs);
	idxx = qn*bs;

	for(; j<rn%2; j++)
		{
		i = rn-1-j;
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &x[%d], -1);\n", j, j, i+1+bs*(i+0)+idxA, sda, i+1+qn*bs, i+qn*bs);
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", i+idxx, i+idxx, i+0+bs*(i+0)+idxA );
		}
	for(; j<rn; j+=2)
		{
		i = rn-2-j;
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, &pA[%d], %d, &x[%d], &x[%d], -1);\n", j, j, i+2+bs*(i+0)+idxA, sda, i+2+idxx, i+idxx);
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", i+1+idxx, i+1+idxx, (i+1)+bs*(i+1)+idxA );
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d]) / pA[%d];\n", i+idxx, i+idxx, (i+1)+bs*(i+0)+idxA, i+1+qn*bs, (i+0)+bs*(i+0)+idxA );
		}

	// blocks of 8
	j = 0;
#if !defined(TARGET_X86_ATOM)
	for(; j<qn-1; j+=2)
		{
		
		// all 4 rows
	idxA = (qn-j-2)*bs*(sda+bs);
	idxx = (qn-j-2)*bs;
		

		// correct
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, 0, &pA[%d], %d, &x[%d], &x[%d], -1);\n", rn+j*bs, 2*bs*sda+idxA, sda, idxx+8, idxx );
		

		// last 4 rows
	idxA = (qn-j-1)*bs*(sda+bs);
	idxx = (qn-j-1)*bs;

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx+3, idxx+3, 3+bs*3+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+2, idxx+2, 3+bs*2+idxA, idxx+3, 2+bs*2+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+1, idxx+1, 3+bs*1+idxA, idxx+3, 2+bs*1+idxA, idxx+2, 1+bs*1+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx, idxx, 3+idxA, idxx+3, 2+idxA, idxx+2, 1+idxA, idxx+1, idxA);

		// first 4 rows
	idxA = (qn-j-2)*bs*(sda+bs);
	idxx = (qn-j-2)*bs;

		// correct
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 0, &pA[%d], %d, &x[%d], &x[%d], -1);\n", 4, bs*sda+idxA, sda, idxx+4, idxx );

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx+3, idxx+3, 3+bs*3+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+2, idxx+2, 3+bs*2+idxA, idxx+3, 2+bs*2+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+1, idxx+1, 3+bs*1+idxA, idxx+3, 2+bs*1+idxA, idxx+2, 1+bs*1+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx, idxx, 3+idxA, idxx+3, 2+idxA, idxx+2, 1+idxA, idxx+1, idxA);

		}
#endif	
	// blocks of 4
	for(; j<qn; j++)
		{
		
		// first 4 rows
	idxA = (qn-j-1)*bs*(sda+bs);
	idxx = (qn-j-1)*bs;
		
		// correct
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, 0, &pA[%d], %d, &x[%d], &x[%d], -1);\n", rn+j*bs, bs*sda+idxA, sda, idxx+4, idxx );

		// solve
fprintf(f, "	x[%d] = (x[%d]) / pA[%d];\n", idxx+3, idxx+3, 3+bs*3+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+2, idxx+2, 3+bs*2+idxA, idxx+3, 2+bs*2+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx+1, idxx+1, 3+bs*1+idxA, idxx+3, 2+bs*1+idxA, idxx+2, 1+bs*1+idxA);
fprintf(f, "	x[%d] = (x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d] - pA[%d]*x[%d]) / pA[%d];\n", idxx, idxx, 3+idxA, idxx+3, 2+idxA, idxx+2, 1+idxA, idxx+1, idxA);

		}

/*	fprintf(f, "	\n");*/

	}
