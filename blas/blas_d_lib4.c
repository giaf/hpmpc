/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
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

#include "../include/kernel_d_lib4.h"
#include "../include/block_size.h"



// test for the performance of the dgemm kernel
void dgemm_kernel_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg, int tc, int td)
	{

	const int bs = 4;

	int i, j, jj;
	
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	__m256i mask;
#endif

	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_nt_12x4_lib4(k, &pA[0], sda, &pB[0], &pC[0], sdc, &pD[0], sdd, alg, tc, td);
			}
		}
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_nt_8x4_vs_lib4(8, 4, k, &pA[0], sda, &pB[0], &pC[0], sdc, &pD[0], sdd, alg, tc, td);
			}
		}
#endif
#if defined(TARGET_X64_AVX)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_nt_8x4_lib4(k, &pA[0], sda, &pB[0], &pC[0], sdc, &pD[0], sdd, alg, tc, td);
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_nt_4x4_vs_lib4(4, 4, k, &pA[0], &pB[0], &pC[0], &pD[0], alg, tc, td);
			}
		}


	}



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg, int tc, int td)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j;

#if 0
	i = 0;
//	for( ; i<m-4; i+=8)
//		{
//		for(j=0; j<n; j+=4)
//			{
//			kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
//			}
//		}
	for( ; i<m; i+=4)
		{
		j = 0;
		for( ; j<n; j+=4)
			{
			kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
			//kernel_dgemm_nt_4x3_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
			//kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
			//kernel_dgemm_nt_4x1_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
			}
		}
	return;
#endif
	
	if(tc==0)
		{
		if(td==0) // tc==0, td==0
			{



			i = 0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
#if defined(TARGET_X64_AVX2)
			for(; i<m-11; i+=12)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_12x4_vs_lib4(12, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
						}
					}
				}
			if(i<m-8)
				{
				goto left_00_12;
				}
			if(i<m-4)
				{
				goto left_00_8;
				}
			if(i<m-2)
				{
				goto left_00_4;
				}
			if(i<m)
				{
				goto left_00_2;
				}
#endif
#if defined(TARGET_X64_AVX)
			for(; i<m-10; i+=8)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_8x4_vs_lib4(8, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
						}
					}
				}
			if(m-i<3)
				{
				if(m-i==0)
					return;
				else
					goto left_00_2;
				}
			else
				{
				if(m-i<7)
					{
					if(m-i<5)
						goto left_00_4;
					else
						goto left_00_6;
					}
				else
					{
					if(m-i<9)
						goto left_00_8;
					else
						goto left_00_10;
					}
				}
#endif
#else
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
#if defined(CORTEX_A15) || defined(CORTEX_A9) || defined(CORTEX_A7)
					kernel_dgemm_nt_4x4_nn_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
#else
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
#endif
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
						}
					else
						{
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					goto left_00_4;
					}
				else // m-i==2 || m-i==1
					{
					goto left_00_2;
					}
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions
#if defined(TARGET_X64_AVX2)
			left_00_12:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_12x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_nt_4x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_00_10:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_10x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_nt_2x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			left_00_8:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_8x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_00_6:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_6x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			return;
#endif

			left_00_4:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			return;

			left_00_2:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x2_vs_lib4(n-j, m-i, k, &pB[j*sdb], &pA[i*sda], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			return;



			}
		else // tc==0, td==1
			{



			i = 0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
#if defined(TARGET_X64_AVX2)
			for(; i<m-11; i+=12)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_12x4_vs_lib4(12, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[(i+8)*bs+j*sdd], alg, 0, 1);
						}
					}
				}
			if(i<m-8)
				{
				goto left_01_12;
				}
			if(i<m-4)
				{
				goto left_01_8;
				}
			if(i<m)
				{
				goto left_01_4;
				}
#endif
#if defined(TARGET_X64_AVX)
			for(; i<m-10; i+=8)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_8x4_vs_lib4(8, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
						}
					else
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
						}
					}
				}
			if(m-i<3)
				{
				if(m-i==0)
					return;
				else
					goto left_01_2;
				}
			else
				{
				if(m-i<7)
					{
					if(m-i<5)
						goto left_01_4;
					else
						goto left_01_6;
					}
				else
					{
					if(m-i<9)
						goto left_01_8;
					else
						goto left_01_10;
					}
				}
#endif
#else
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
						}
					else
						{
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					goto left_01_4;
					}
				else // m-i==2 || m-i==1
					{
					goto left_01_2;
					}
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions
#if defined(TARGET_X64_AVX2)
			left_01_12:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_12x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				kernel_dgemm_nt_4x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[(i+8)*bs+j*sdd], alg, 0, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_01_10:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_10x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				kernel_dgemm_nt_2x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[(i+8)*bs+j*sdd], alg, 0, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			left_01_8:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_8x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_01_6:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_6x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 0, 1);
				}
			return;
#endif

			left_01_4:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
				}
			return;

			left_01_2:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x2_vs_lib4(n-j, m-i, k, &pB[j*sdb], &pA[i*sda], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 1, 0); 
				}
			if(j<n)
				{
				kernel_dgemm_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, 0, 1);
				}
			return;



			}
		}
	else // tc==1
		{
		if(td==0) // tc==1, td==0
			{



			i = 0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
#if defined(TARGET_X64_AVX2)
			for(; i<m-11; i+=12)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_12x4_vs_lib4(12, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[j*bs+(i+8)*sdd], alg, 1, 0);
						}
					}
				}
			if(i<m-8)
				{
				goto left_10_12;
				}
			if(i<m-4)
				{
				goto left_10_8;
				}
			if(i<m-2)
				{
				goto left_10_4;
				}
			if(i<m)
				{
				goto left_10_2;
				}
#endif
#if defined(TARGET_X64_AVX)
			for(; i<m-7; i+=8)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_8x4_vs_lib4(8, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
						}
					else // n-j==2 || n-j==1
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
						}
					}
				}
			if(m-i<3)
				{
				if(m-i==0)
					return;
				else
					goto left_10_2;
				}
			else
				{
				if(m-i<7)
					{
					if(m-i<5)
						goto left_10_4;
					else
						goto left_10_6;
					}
				else
					{
					if(m-i<9)
						goto left_10_8;
					else
						goto left_10_10;
					}
				}
#endif
#else
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
						}
					else
						{
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					goto left_10_4;
					}
				else // m-i==2 || m-i==1
					{
					goto left_10_2;
					}
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions
#if defined(TARGET_X64_AVX2)
			left_10_12:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_12x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				kernel_dgemm_nt_4x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[j*bs+(i+8)*sdd], alg, 1, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_10_10:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_10x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				kernel_dgemm_nt_2x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[j*bs+(i+8)*sdd], alg, 1, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			left_10_8:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_8x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_10_6:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_6x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 1, 0);
				}
			return;
#endif

			left_10_4:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
				}
			return;

			left_10_2:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x2_vs_lib4(n-j, m-i, k, &pB[j*sdb], &pA[i*sda], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 0, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, 1, 0);
				}
			return;



			}
		else // tc==1, td==1
			{



			i = 0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
#if defined(TARGET_X64_AVX2)
			for(; i<m-11; i+=12)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_12x4_vs_lib4(12, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[(i+8)*bs+j*sdd], alg, 1, 1);
						}
					}
				}
			if(i<m-8)
				{
				goto left_11_12;
				}
			if(i<m-4)
				{
				goto left_11_8;
				}
			if(i<m-2)
				{
				goto left_11_4;
				}
			if(i<m)
				{
				goto left_11_2;
				}
#endif
#if defined(TARGET_X64_AVX)
			for(; i<m-7; i+=8)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_8x4_vs_lib4(8, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
						}
					else // n-j==1 || n-j==2
						{
						kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
						}
					}
				}
			if(m-i<3)
				{
				if(m-i==0)
					return;
				else
					goto left_11_2;
				}
			else
				{
				if(m-i<7)
					{
					if(m-i<5)
						goto left_11_4;
					else
						goto left_11_6;
					}
				else
					{
					if(m-i<9)
						goto left_11_8;
					else
						goto left_11_10;
					}
				}
#endif
#else
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nt_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
						}
					else
						{
						kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					goto left_11_4;
					}
				else // m-i==2 || m-i==1
					{
					goto left_11_2;
					}
				}
#endif

			// common return if i==m
			return;

			// clean up loops definitions
#if defined(TARGET_X64_AVX2)
			left_11_12:
			j = 0;
			for(; j<n-2; j+=4)
				{
					kernel_dgemm_nt_12x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				kernel_dgemm_nt_4x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[(i+8)*bs+j*sdd], alg, 1, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_11_10:
			j = 0;
			for(; j<n-2; j+=4)
				{
					kernel_dgemm_nt_10x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				kernel_dgemm_nt_2x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[(i+8)*bs+j*sdc], &pD[(i+8)*bs+j*sdd], alg, 1, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX)
			left_11_8:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_8x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			return;
#endif

#if defined(TARGET_X64_AVX)
			left_11_6:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_6x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, 1, 1);
				}
			return;
#endif

			left_11_4:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
				}
			if(j<n)
				{
				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
				}
			return;

			left_11_2:
			j = 0;
			for(; j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x2_vs_lib4(n-j, m-i, k, &pB[j*sdb], &pA[i*sda], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, 1, 1);
				}
			return;



			}
		}
	}



/* preforms                                          */
/* C  = A * B (alg== 0)                             */
/* C += A * B (alg== 1)                             */
/* C -= A * B (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_nn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg, int tc, int td)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int i, j, jj;
	
	if(tc==0)
		{
		if(td==0) // not transpose D
			{
			i = 0;
#if 0 //defined(TARGET_X64_AVX2)
			for(; i<m-8; i+=12)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, &pD[(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+(i+8)*sdc], &pD[(j+jj)*bs+(i+8)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
#if 0 //defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, &pD[(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nn_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					else
						{
						kernel_dgemm_nn_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_4x4_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_4x2_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				else // m-i==2 || m-i==1
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_2x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				}
			}
		else // td==1
			{
			i = 0;
#if 0 // defined(TARGET_X64_AVX2)
			for(; i<m-8; i+=12)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, &pD[i*bs+(j+jj)*sdd], sdd, alg, tc, td);
					kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+(i+8)*sdc], &pD[(i+8)*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
#if 0 // defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				j = 0;
				for(; j<n-2; j+=4)
				//for(; j<n; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, &pD[i*bs+(j+jj)*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nn_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					else
						{
						kernel_dgemm_nn_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_4x4_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_4x2_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				else // m-i==2 || m-i==1
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_2x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				}
			}
		}
	else // tc==1
		{
		if(td==0) // not transpose D
			{
			i = 0;
#if 0 && defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_m8x4_lib4(m-i, k, &pA[0+i*sda], sda, &pB[0+j*sdb], &pC[i*bs+j*sdc], sdc, &pD[0+(j+0)*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_m8x2_lib4(m-i, k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], sdc, &pD[0+(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
			if(i<m)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_m4x4_lib4(m-i, k, &pA[0+i*sda], &pB[0+j*sdb], &pC[i*bs+j*sdc], &pD[0+(j+0)*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_m4x2_lib4(m-i, k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], &pD[0+(j+jj)*bs+i*sdd], alg, tc, td);
					}
				}
#else
#if 0 // defined(TARGET_X64_AVX2)
			for(; i<m-8; i+=12)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], sdc, &pD[(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[jj+j*sdb], &pC[(i+8)*bs+(j+jj)*sdc], &pD[(j+jj)*bs+(i+8)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
#if 0 // defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], sdc, &pD[(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nn_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					else
						{
						kernel_dgemm_nn_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_4x4_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_4x2_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				else // m-i==2 || m-i==1
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_2x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
						}
					}
				}
#endif
			}
		else // td==1
			{
			i = 0;
#if 0 // defined(TARGET_X64_AVX2)
			for(; i<m-8; i+=12)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], sdc, &pD[i*bs+(j+jj)*sdd], sdd, alg, tc, td);
					kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[jj+j*sdb], &pC[(i+8)*bs+(j+jj)*sdc], &pD[(i+8)*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
#if 0 // defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				j = 0;
				for(; j<n-2; j+=4)
				//for(; j<n; j+=4)
					{
					kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[i*bs+j*sdc], sdc, &pD[i*bs+j*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], sdc, &pD[i*bs+(j+jj)*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[0+i*sda], sda, &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
#endif
			for(; i<m-3; i+=4)
				{
				j = 0;
				for(; j<n-3; j+=4)
					{
					kernel_dgemm_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					}
				if(j<n)
					{
					if(n-j==3)
						{
						kernel_dgemm_nn_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					else
						{
						kernel_dgemm_nn_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				}
			if(m>i)
				{
				if(m-i==3)
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_4x4_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_4x2_vs_lib4(3, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				else // m-i==2 || m-i==1
					{
					j = 0;
					for(; j<n-2; j+=4)
						{
						kernel_dgemm_nn_2x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					if(j<n)
						{
						kernel_dgemm_nn_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*bs], sdb, &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
						}
					}
				}
			}
		}
	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is upper triangular                         */
void dtrmm_nt_u_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_nt_u_12x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		if(n-j==1)
			{
			corner_dtrmm_nt_u_12x1_lib4(&pA[j*bs+(i+0)*sda], sda, &pB[j*bs+j*sdb], &pC[j*bs+(i+0)*sdc], sdc);
			}
		else if(n-j==2)
			{
			corner_dtrmm_nt_u_12x2_lib4(&pA[j*bs+(i+0)*sda], sda, &pB[j*bs+j*sdb], &pC[j*bs+(i+0)*sdc], sdc);
			}
		else if(n-j==3)
			{
			corner_dtrmm_nt_u_12x3_lib4(&pA[j*bs+(i+0)*sda], sda, &pB[j*bs+j*sdb], &pC[j*bs+(i+0)*sdc], sdc);
			}
		}
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_nt_u_8x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		if(n-j==1)
			{
			corner_dtrmm_nt_u_8x1_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		else if(n-j==2)
			{
			corner_dtrmm_nt_u_8x2_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		else if(n-j==3)
			{
			corner_dtrmm_nt_u_8x3_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_nt_u_4x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		if(n-j==1)
			{
			corner_dtrmm_nt_u_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==2)
			{
			corner_dtrmm_nt_u_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==3)
			{
			corner_dtrmm_nt_u_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is lower triangular                         */
void dtrmm_nt_l_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;
	
	i=0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for( ; i<m-4; i+=8)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_nt_l_8x4_lib4(j+4, &pA[i*sda], sda, &pB[j*sdb], &pC[i*sdc+j*bs], sdc);
			}
		if(j<n)
			{
			kernel_dtrmm_nt_l_8x2_lib4(j+2, &pA[i*sda], sda, &pB[j*sdb], &pC[i*sdc+j*bs], sdc);
			}
		}
	for( ; i<m; i+=4)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_nt_l_4x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_nt_l_4x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
#else
	for( ; i<m-2; i+=4)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_nt_l_4x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_nt_l_4x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
	if(i<m)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_nt_l_2x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_nt_l_2x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
#endif

	}



void dsyrk_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;
	
	int i, j;
	
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	double d_temp;

	__m256i mask;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
#endif

/*	int n = m;*/
	
	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-11; i+=12)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(n-j==3)
				{
				kernel_dgemm_nt_12x4_vs_lib4(12, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_12x4_vs_lib4(12, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-10)
					{
					kernel_dsyrk_nt_8x8_vs_lib4(8, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], sdb, &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				else if(j<n-6)
					{
					kernel_dsyrk_nt_8x4_vs_lib4(8, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					//if(j<n-10)
					//	{
					//	kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
					//	}
					//else 
					if(j<n-8)
						{
						kernel_dsyrk_nt_4x2_vs_lib4(4, mask, n-j-8, k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
						}
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_8x2_vs_lib4(8, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_4x2_vs_lib4(4, mask, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[(i+4)*sda], sda, &pB[j*sdb], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, alg, 0, 0);
				}
			}
		}
	if(i<m-8)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nt_12x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_nt_4x2_vs_lib4(m-i-8, n-j, k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_12x4_vs_lib4(m-i, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-10)
					{
					kernel_dsyrk_nt_8x8_vs_lib4(m-i-4, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], sdb, &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				else if(j<n-6)
					{
					kernel_dsyrk_nt_8x4_vs_lib4(m-i-4, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					//if(j<n-10)
					//	{
					//	kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
					//	}
					//else 
					if(j<n-8)
						{
						kernel_dsyrk_nt_4x2_vs_lib4(m-i-8, mask, n-j-8, k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
						}
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_8x2_vs_lib4(m-i-4, mask, n-j-4, k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_4x2_vs_lib4(4, _mm256_set_epi64x( -1, -1, -1, -1 ), n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dgemm_nt_8x2_vs_lib4(m-i-4, n-j, k, &pA[(i+4)*sda], sda, &pB[j*sdb], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, alg, 0, 0);
				}
			}
		i += 12;
		}
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-7; i+=8)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
#if defined(TARGET_X64_AVX)
			kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
#endif
#if defined(TARGET_X64_AVX2)
			kernel_dgemm_nt_8x4_vs_lib4(8, 4, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
#endif
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				if(j==n-3)
					{
					kernel_dgemm_nt_8x4_vs_lib4(8, 3, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
					}
				else
					{
					kernel_dgemm_nt_8x2_vs_lib4(8, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
					}
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_8x4_vs_lib4(8, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-6)
					{
					kernel_dsyrk_nt_4x4_vs_lib4(4, mask, n-j-4, k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_4x2_vs_lib4(4, mask, n-j-4, k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_8x2_vs_lib4(8, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				}
			}
		}
	if(i<m-4)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nt_8x4_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nt_8x2_vs_lib4(m-i, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_8x4_vs_lib4(m-i, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-6)
					{
					kernel_dsyrk_nt_4x4_vs_lib4(m-i-4, mask, n-j-4, k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_4x2_vs_lib4(m-i-4, mask, n-j-4, k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_8x2_vs_lib4(m-i, mask, n-j, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				}
			}
		i += 8;
		}
	if(i<m)
		{
		d_temp = m-i-0.0;
		mask = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &d_temp ) ) );
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_4x4_vs_lib4(m-i, mask, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_4x2_vs_lib4(m-i, mask, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			}
		//i += 4;
		}

#else
	for(; i<m-3; i+=4)
		{
		j = 0;
		for(; j<i && j<n-3; j+=4)
			{
			kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				if(n-j==3)
					{
					kernel_dgemm_nt_4x4_vs_lib4(4, 3, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
					}
				else
					{
					kernel_dgemm_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
					}
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_4x4_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_4x2_vs_lib4(4, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			}
		}
	if(i<m)
		{
		if(m-i==3)
			{
			j = 0;
			for(; j<i && j<n-2; j+=4)
				{
				kernel_dgemm_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			if(j<i) // dgemm
				{
				if(j<n)
					{
					kernel_dgemm_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
					}
				}
			else // dsyrk
				{
				if(j<n-2)
					{
					kernel_dsyrk_nt_4x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
					}
				else if(j<n)
					{
					kernel_dsyrk_nt_4x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
					}
				}
			// i += 4;
			}
		else // m-i==2 || m-i==1
			{
			j = 0;
			for(; j<i && j<n-2; j+=4)
				{
				kernel_dgemm_nt_2x4_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			if(j<i) // dgemm
				{
				if(j<n)
					{
					kernel_dgemm_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
					}
				}
			else // dsyrk
				{
				if(j<n)
					{
					kernel_dsyrk_nt_2x2_vs_lib4(m-i, n-j, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
					}
				}
			// i += 2;
			}
		}
#endif

	}



void dsyrk_nn_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;
	
	int i, j;
	
/*	int n = m;*/
	
#if 1
	i = 0;
#if 0 && defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				j += 2;
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-10)
					{
					kernel_dsyrk_nt_8x8_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], sdb, &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				else if(j<n-6)
					{
					kernel_dsyrk_nt_8x4_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					//if(j<n-10)
					//	{
					//	kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
					//	}
					//else 
					if(j<n-8)
						{
						kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
						}
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dgemm_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[j*sdb], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, alg, 0, 0);
				}
			}
		}
#endif
#if 0 // defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				j += 2;
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				if(j<n-6)
					{
					kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				else if(j<n-4)
					{
					kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					}
				}
			else if(j<n)
				{
				kernel_dsyrk_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				}
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			kernel_dgemm_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
			}
		if(j<i) // dgemm
			{
			if(j<n)
				{
				kernel_dgemm_nn_4x2_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				j += 2;
				}
			}
		else // dsyrk
			{
			if(j<n-2)
				{
				kernel_dsyrk_nn_4x4_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			else if(j<n)
				{
				kernel_dsyrk_nn_4x2_lib4(k, &pA[i*sda], &pB[j*bs], sdb, &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				}
			}
		}
#else
	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX2)
		if(i<m-8)
			{
			kernel_dsyrk_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
			i += 12;
			for(; i<m-8; i+=12)
				{
				kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			}
		else if(i<m-4)
			{
			kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
			i += 8;
			}
#else
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX)
			kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
			i += 8;
#else
			kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdD], alg);
			i += 4;
#endif
#if defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			}
#endif
		else //if(i<m)
			{
			kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
			//i += 4;
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				}
			}
		else //if(i<m)
			{
			kernel_dsyrk_nt_2x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
			}
		}
#endif

	}



void dpotrf_lib(int m, int n, double *pD, int sdd, double *pC, int sdc, double *diag)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	double *dummy;

#if 1
	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pC[1+(j+0)*bs+j*sdc];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pC[2+(j+0)*bs+j*sdc];
			fact[4] = pC[2+(j+1)*bs+j*sdc];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pC[3+(j+0)*bs+j*sdc];
			fact[7] = pC[3+(j+1)*bs+j*sdc];
			fact[8] = pC[3+(j+2)*bs+j*sdc];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			kernel_dgemm_dtrsm_nt_12x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pC[1+(j+0)*bs+j*sdc];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				//kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[(i+8)*sdc], &pC[j*sdc], &pD[j*bs+(i+8)*sdd], &pC[j*bs+(i+8)*sdc], fact, 1);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-2)
				{
				//kernel_dsyrk_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				if(j<n-10)
					{
					kernel_dsyrk_dpotrf_nt_8x8_lib4(m-i, n-j, 0, 0, j+4, dummy, 0, dummy, 0, &pC[(i+4)*sdc], sdc, &pC[(j+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, &pC[(j+4)*bs+(i+4)*sdc], sdc, fact, 1, 0);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					diag[j+6] = fact[5];
					diag[j+7] = fact[9];
					diag[j+8] = fact[10];
					diag[j+9] = fact[12];
					diag[j+10] = fact[15];
					diag[j+11] = fact[19];
					}
				else if(j<n-6)
					{
					//kernel_dsyrk_nt_8x4_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, 0, j+4, dummy, 0, dummy, &pC[(i+4)*sdc], sdc, &pC[(j+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], sdd, &pC[(j+4)*bs+(i+4)*sdc], sdc, fact, 1, 0);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					diag[j+6] = fact[5];
					diag[j+7] = fact[9];
					//if(j<n-10)
					//	{
					//	//kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
					//	kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j+8, dummy, dummy, &pC[(i+8)*sdc], &pC[(j+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], fact, 1, 0);
					//	diag[j+8] = fact[0];
					//	diag[j+9] = fact[2];
					//	diag[j+10] = fact[5];
					//	diag[j+11] = fact[9];
					//	}
					//else 
					if(j<n-8)
						{
						//kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
						kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j+8, dummy, dummy, &pC[(i+8)*sdc], &pC[(j+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], fact, 1, 0);
						diag[j+8] = fact[0];
						diag[j+9] = fact[2];
						}
					}
				else if(j<n-4)
					{
					//kernel_dsyrk_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j+4, dummy, dummy, &pC[(i+4)*sdc], &pC[(j+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], fact, 1, 0);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					if(fact[0]==0.0) fact[0]=1.0;
					if(fact[2]==0.0) fact[2]=1.0;
					kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j+4, dummy, dummy, &pC[(i+8)*sdc], &pC[(j+4)*sdc], &pD[(j+4)*bs+(i+8)*sdd], &pC[(j+4)*bs+(i+8)*sdc], fact, 1);
					}
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				if(fact[0]==0.0) fact[0]=1.0;
				if(fact[2]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[j*sdb], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, alg, 0, 0);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, 0, j, dummy, 0, dummy, &pC[(i+4)*sdc], sdc, &pC[j*sdc], &pD[j*bs+(i+4)*sdd], sdd, &pC[j*bs+(i+4)*sdc], sdc, fact, 1);
				}
			}
		}
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pC[1+(j+0)*bs+j*sdc];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pC[2+(j+0)*bs+j*sdc];
			fact[4] = pC[2+(j+1)*bs+j*sdc];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pC[3+(j+0)*bs+j*sdc];
			fact[7] = pC[3+(j+1)*bs+j*sdc];
			fact[8] = pC[3+(j+2)*bs+j*sdc];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			kernel_dgemm_dtrsm_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pC[1+(j+0)*bs+j*sdc];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-2)
				{
				//kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				if(j<n-6)
					{
					//kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j+4, dummy, dummy, &pC[(i+4)*sdc], &pC[(j+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], fact, 1, 0);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					diag[j+6] = fact[5];
					diag[j+7] = fact[9];
					}
				else if(j<n-4)
					{
					//kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j+4, dummy, dummy, &pC[(i+4)*sdc], &pC[(j+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], fact, 1, 0);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					}
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				if(fact[0]==0.0) fact[0]=1.0;
				if(fact[2]==0.0) fact[2]=1.0;
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[(i+4)*sdc], &pC[j*sdc], &pD[j*bs+(i+4)*sdd], &pC[j*bs+(i+4)*sdc], fact, 1);
				}
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pC[1+(j+0)*bs+j*sdc];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pC[2+(j+0)*bs+j*sdc];
			fact[4] = pC[2+(j+1)*bs+j*sdc];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pC[3+(j+0)*bs+j*sdc];
			fact[7] = pC[3+(j+1)*bs+j*sdc];
			fact[8] = pC[3+(j+2)*bs+j*sdc];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
			kernel_dgemm_dtrsm_nt_4x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pC[1+(j+0)*bs+j*sdc];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-2)
				{
				//kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				}
			}
		}
#else

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX2)
		if(i<m-8)
			{
			kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, 0);
			i += 12;

			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
		
			for(; i<m-8; i+=12)
				{
				kernel_dgemm_dtrsm_nt_12x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				}
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			}
		else if(i<m-4)
			{
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, 0);
			i += 8;
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
#else
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, 0);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
			i += 4;
#endif
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
#if defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			}
#endif
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x2_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
				}
			}
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_2x2_lib4(m-i, n-j, 0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}
#endif

	}



void dsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, double *diag, int alg, int fast_rsqrt)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#if 0
	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pD[1+(j+0)*bs+j*sdd];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pD[2+(j+0)*bs+j*sdd];
			fact[4] = pD[2+(j+1)*bs+j*sdd];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pD[3+(j+0)*bs+j*sdd];
			fact[7] = pD[3+(j+1)*bs+j*sdd];
			fact[8] = pD[3+(j+2)*bs+j*sdd];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			//kernel_dgemm_dtrsm_nt_12x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
			kernel_dgemm_dtrsm_nt_12x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pC[1+(j+0)*bs+j*sdc];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				//kernel_dgemm_dtrsm_nt_8x2_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//kernel_dgemm_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[j*sdb], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], alg, 0, 0);
				//kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[(i+8)*sdc], &pC[j*sdc], &pD[j*bs+(i+8)*sdd], &pC[j*bs+(i+8)*sdc], fact, 1);
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j, &pA[(i+8)*sda], &pA[j*sda], &pD[(i+8)*sdd], &pD[j*sdd], &pC[j*bs+(i+8)*sdc], &pD[j*bs+(i+8)*sdd], fact, alg);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-2)
				{
				//kernel_dsyrk_nt_12x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				//kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1, fast_rsqrt);
				kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				if(j<n-10)
					{
					kernel_dsyrk_dpotrf_nt_8x8_lib4(m-i, n-j, 0, k, j+4, &pA[(i+4)*sda], sda, &pA[(j+4)*sda], sda, &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], sdd, &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, fact, 1, fast_rsqrt);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					diag[j+6] = fact[5];
					diag[j+7] = fact[9];
					diag[j+8] = fact[10];
					diag[j+9] = fact[12];
					diag[j+10] = fact[15];
					diag[j+11] = fact[19];
					}
				else if(j<n-6)
					{
					//kernel_dsyrk_nt_8x4_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j+4, &pA[(i+4)*sda], sda, &pA[(j+4)*sda], &pD[(i+4)*sdd], sdd, &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, fact, 1, fast_rsqrt);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					diag[j+6] = fact[5];
					diag[j+7] = fact[9];
					//if(j<n-10)
					//	{
					//	//kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
					//	kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, 0, j+8, dummy, dummy, &pC[(i+8)*sdc], &pC[(j+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], fact, 1, fast_rsqrt);
					//	diag[j+8] = fact[0];
					//	diag[j+9] = fact[2];
					//	diag[j+10] = fact[5];
					//	diag[j+11] = fact[9];
					//	}
					//else 
					if(j<n-8)
						{
						//kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+8)*sda], &pB[(j+8)*sdb], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], alg);
						kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j+8, &pA[(i+8)*sda], &pA[(j+8)*sda], &pD[(i+8)*sdd], &pD[(j+8)*sdd], &pC[(j+8)*bs+(i+8)*sdc], &pD[(j+8)*bs+(i+8)*sdd], fact, 1, fast_rsqrt);
						diag[j+8] = fact[0];
						diag[j+9] = fact[2];
						}
					}
				else if(j<n-4)
					{
					//kernel_dsyrk_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], sdc, &pD[(j+4)*bs+(i+4)*sdd], sdd, alg);
					kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j+4, &pA[(i+4)*sda], &pA[(j+4)*sda], &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], fact, 1, fast_rsqrt);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					if(fact[0]==0.0) fact[0]=1.0;
					if(fact[2]==0.0) fact[2]=1.0;
					kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j+4, &pA[(i+8)*sda], &pA[(j+4)*sda], &pD[(i+8)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+8)*sdc], &pD[(j+4)*bs+(i+8)*sdd], fact, 1);
					}
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				if(fact[0]==0.0) fact[0]=1.0;
				if(fact[2]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[(i+4)*sda], sda, &pB[j*sdb], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, alg, 0, 0);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, k, j, &pA[(i+4)*sda], sda, &pA[(j+4)*sda], &pD[(i+4)*sdd], sdd, &pD[j*sdd], &pC[j*bs+(i+4)*sdc], sdc, &pD[j*bs+(i+4)*sdd], sdd, fact, 1);
				}
			}
		}
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pD[1+(j+0)*bs+j*sdd];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pD[2+(j+0)*bs+j*sdd];
			fact[4] = pD[2+(j+1)*bs+j*sdd];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pD[3+(j+0)*bs+j*sdd];
			fact[7] = pD[3+(j+1)*bs+j*sdd];
			fact[8] = pD[3+(j+2)*bs+j*sdd];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
			kernel_dgemm_dtrsm_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, 1);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pD[1+(j+0)*bs+j*sdd];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, 0, 0);
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, 1);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-6)
				{
				kernel_dsyrk_dpotrf_nt_8x8_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], sda, &pD[i*sdc], sdd, &pD[j*sdd], sdd, &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				diag[j+4] = fact[10];
				diag[j+5] = fact[12];
				diag[j+6] = fact[15];
				diag[j+7] = fact[19];
				}
			else if(j<n-2)
				{
				//kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdc], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				//if(j<n-6)
				//	{
				//	//kernel_dsyrk_nt_4x4_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
				//	kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j+4, &pA[(i+4)*sda], &pA[(j+4)*sda], &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], fact, 1, fast_rsqrt);
				//	diag[j+4] = fact[0];
				//	diag[j+5] = fact[2];
				//	diag[j+6] = fact[5];
				//	diag[j+7] = fact[9];
				//	}
				//else 
				if(j<n-4)
					{
					//kernel_dsyrk_nt_4x2_lib4(k, &pA[(i+4)*sda], &pB[(j+4)*sdb], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], alg);
					kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j+4, &pA[(i+4)*sda], &pA[(j+4)*sda], &pD[(i+4)*sdd], &pD[(j+4)*sdd], &pC[(j+4)*bs+(i+4)*sdc], &pD[(j+4)*bs+(i+4)*sdd], fact, 1, fast_rsqrt);
					diag[j+4] = fact[0];
					diag[j+5] = fact[2];
					}
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				if(fact[0]==0.0) fact[0]=1.0;
				if(fact[2]==0.0) fact[2]=1.0;
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j, &pA[(i+4)*sda], &pA[j*sda], &pD[(i+4)*sdd], &pD[j*sdd], &pC[j*bs+(i+4)*sdc], &pD[j*bs+(i+4)*sdd], fact, 1);
				}
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<i && j<n-2; j+=4)
			{
			fact[0] = diag[j+0]; if(fact[0]==0.0) fact[0]=1.0;
			fact[1] = pD[1+(j+0)*bs+j*sdd];
			fact[2] = diag[j+1]; if(fact[2]==0.0) fact[2]=1.0;
			fact[3] = pD[2+(j+0)*bs+j*sdd];
			fact[4] = pD[2+(j+1)*bs+j*sdd];
			fact[5] = diag[j+2]; if(fact[5]==0.0) fact[5]=1.0;
			fact[6] = pD[3+(j+0)*bs+j*sdd];
			fact[7] = pD[3+(j+1)*bs+j*sdd];
			fact[8] = pD[3+(j+2)*bs+j*sdd];
			fact[9] = diag[j+3]; if(fact[9]==0.0) fact[9]=1.0;
			//kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
			kernel_dgemm_dtrsm_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1);
			}
		if(j<i) // dtrsm
			{
			if(j<n)
				{
				fact[0] = diag[j+0]; if(fact[0]==0.0) fact[2]=1.0;
				fact[1] = pD[1+(j+0)*bs+j*sdd];
				fact[2] = diag[j+1]; if(fact[0]==0.0) fact[2]=1.0;
				//kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, 0, 0);
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1);
				j += 2;
				}
			}
		else // dpotrf
			{
			if(j<n-2)
				{
				//kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				diag[j+2] = fact[5];
				diag[j+3] = fact[9];
				}
			else if(j<n)
				{
				//kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg);
				kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, 1, fast_rsqrt);
				diag[j+0] = fact[0];
				diag[j+1] = fact[2];
				}
			}
		}
#else

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX2)
		if(i<m-8)
			{
			kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, fast_rsqrt);
			i += 12;

			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			for(; i<m-8; i+=12)
				{
				kernel_dgemm_dtrsm_nt_12x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
		else if(i<m-4)
			{
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, fast_rsqrt);
			i += 8;
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
#else
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, fast_rsqrt);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, fast_rsqrt);
			i += 4;
#endif
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
#if defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
#endif
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, fast_rsqrt);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, fast_rsqrt);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
		else if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_2x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, fast_rsqrt);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}
#endif

	}



// TODO modify kernels instead
void dgemv_n_lib(int m, int n, double *pA, int sda, double *x, double *y, double *z, int alg) // pA has to be aligned !!!
	{

	// early return
	if(m<=0  || n<=0)
		return;
	
	const int bs = 4;
	
	int i, j;

	j=0;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX)
	for(; j<m-11; j+=12)
		{
		kernel_dgemv_n_12_lib4(n, pA, sda, x, y, z, alg);
		pA += 3*sda*bs;
		y  += 3*bs;
		z  += 3*bs;
		}
	if(j<m-8)
		{
		kernel_dgemv_n_12_vs_lib4(m-j, n, pA, sda, x, y, z, alg);
		pA += 3*sda*bs;
		y  += 3*bs;
		z  += 3*bs;
		//j  += 12;
		return;
		}
#else
	for(; j<m-7; j+=8)
		{
		kernel_dgemv_n_8_lib4(n, pA, sda, x, y, z, alg);
		pA += 2*sda*bs;
		y  += 2*bs;
		z  += 2*bs;
		}
#endif
	if(j<m-4)
		{
		kernel_dgemv_n_8_vs_lib4(m-j, n, pA, sda, x, y, z, alg);
		pA += 2*sda*bs;
		y  += 2*bs;
		z  += 2*bs;
		//j  += 8;
		return;
		}
	if(j<m)
		{
		kernel_dgemv_n_4_vs_lib4(m-j, n, pA, x, y, z, alg);
		pA += sda*bs;
		y  += bs;
		z  += bs;
		//j  += 4;
		return;
		}

	}



void dgemv_t_lib(int m, int n, double *pA, int sda, double *x, double *y, double *z, int alg)
	{
	
	// early return
	if(m<=0  || n<=0)
		return;
	
	const int bs = 4;
	
	int j;
	
	j=0;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX)
	for(; j<n-11; j+=12)
		{
		kernel_dgemv_t_12_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		}
#endif
	for(; j<n-7; j+=8)
		{
		kernel_dgemv_t_8_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		}
	for(; j<n-3; j+=4)
		{
		kernel_dgemv_t_4_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		}
	if(n>j)
		{
		if(n-j==1)
			kernel_dgemv_t_1_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		else if(n-j==2)
			kernel_dgemv_t_2_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		else if(n-j==3)
			kernel_dgemv_t_3_lib4(m, pA+j*bs, sda, x, y+j, z+j, alg);
		}

	}



void dtrmv_u_n_lib(int m, double *pA, int sda, double *x, double *y, int alg)
	{

	if(m<=0)
		return;

	const int bs = 4;
	
	int j;
	
	j=0;
#if defined(TARGET_X64_AVX2)
	for(; j<m-11; j+=12)
		{
		kernel_dtrmv_u_n_12_lib4(m-j, pA, sda, x, y, alg);
		pA += 3*sda*bs + 3*4*bs;
		x  += 3*bs;
		y  += 3*bs;
		}
#endif
	for(; j<m-7; j+=8)
		{
		kernel_dtrmv_u_n_8_lib4(m-j, pA, sda, x, y, alg);
		pA += 2*sda*bs + 2*4*bs;
		x  += 2*bs;
		y  += 2*bs;
		}
	for(; j<m-3; j+=4)
		{
		kernel_dtrmv_u_n_4_lib4(m-j, pA, x, y, alg);
		pA += sda*bs + 4*bs;
		x  += bs;
		y  += bs;
		}
	for(; j<m-1; j+=2)
		{
		kernel_dtrmv_u_n_2_lib4(m-j, pA, x, y, alg);
		pA += 2 + 2*bs;
		x  += 2;
		y  += 2;
		}
	if(j<m)
		{
		if(alg==0)
			y[0] = pA[0+bs*0]*x[0];
		else if(alg==1)
			y[0] += pA[0+bs*0]*x[0];
		else
			y[0] -= pA[0+bs*0]*x[0];
		}

	}



void dtrmv_u_t_lib(int m, double *pA, int sda, double *x, double *y, int alg)
	{

	if(m<=0)
		return;

	const int bs = 4;
	
	int j;
	
	double *ptrA;
	
	j=0;
#if defined(TARGET_X64_AVX2)
	for(; j<m-11; j+=12)
		{
		kernel_dtrmv_u_t_12_lib4(j, pA, sda, x, y, alg);
		pA += 3*4*bs;
		y  += 3*bs;
		}
#endif
	for(; j<m-7; j+=8)
		{
		kernel_dtrmv_u_t_8_lib4(j, pA, sda, x, y, alg);
		pA += 2*4*bs;
		y  += 2*bs;
		}
	for(; j<m-3; j+=4)
		{
		kernel_dtrmv_u_t_4_lib4(j, pA, sda, x, y, alg);
		pA += 4*bs;
		y  += bs;
		}
	for(; j<m-1; j+=2) // keep for !!!
		{
		kernel_dtrmv_u_t_2_lib4(j, pA, sda, x, y, alg);
		pA += 2*bs;
		y  += 2;
		}
	if(j<m)
		{
		kernel_dtrmv_u_t_1_lib4(j, pA, sda, x, y, alg);
		}

	}



// it moves vertically across block // TODO allow rectangular matrices
void dsymv_lib(int m, int n, double *pA, int sda, double *x, double *y, double *z, int alg)
	{

	if(m<=0 || n<=0)
		return;

	// TODO better way to do 4-ways ???
	
	const int bs = 4;
	
	if(m<n)
		n = m;
	
	int j, j0;
	
	if(alg==0)
		{
		for(j=0; j<m; j++)
			z[j] = 0.0;
		alg = 1;
		}
	else
		{
		if(y!=z)
			for(j=0; j<m; j++)
				z[j] = y[j];
		}
	
	j=0;
	for(; j<n-3; j+=4)
		{
		kernel_dsymv_4_lib4(m-j, pA+j*sda+j*bs, sda, x+j, z+j, z+j, x+j, z+j, z+j, 1, alg);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			kernel_dsymv_1_lib4(m-j, pA+j*sda+j*bs, sda, x+j, z+j, z+j, x+j, z+j, z+j, 1, alg);
			}
		else if(n-j==2)
			{
			kernel_dsymv_2_lib4(m-j, pA+j*sda+j*bs, sda, x+j, z+j, z+j, x+j, z+j, z+j, 1, alg);
			}
		else // if(n-j==3)
			{
			kernel_dsymv_3_lib4(m-j, pA+j*sda+j*bs, sda, x+j, z+j, z+j, x+j, z+j, z+j, 1, alg);
			}
		}

	}



// it moves vertically across block
void dmvmv_lib(int m, int n, double *pA, int sda, double *x_n, double *y_n, double *z_n, double *x_t, double *y_t, double *z_t, int alg)
	{

	if(m<=0 || n<=0)
		return;

	// TODO better way to do 4-ways ???
	
	const int bs = 4;

	int j;
	
	if(alg==0)
		{
		for(j=0; j<m; j++)
			z_n[j] = 0.0;
		for(j=0; j<n; j++)
			z_t[j] = 0.0;
		alg = 1;
		}
	else
		{
		if(y_n!=z_n)
			for(j=0; j<m; j++)
				z_n[j] = y_n[j];
		if(y_t!=z_t)
			for(j=0; j<n; j++)
				z_t[j] = y_t[j];
		}
	
	j=0;
	for(; j<n-3; j+=4)
		{
		kernel_dsymv_4_lib4(m, pA+j*bs, sda, x_n+j, z_n, z_n, x_t, z_t+j, z_t+j, 0, alg);
		}
	for(; j<n-1; j+=2)
		{
		kernel_dsymv_2_lib4(m, pA+j*bs, sda, x_n+j, z_n, z_n, x_t, z_t+j, z_t+j, 0, alg);
		}
	for(; j<n; j++)
		{
		kernel_dsymv_1_lib4(m, pA+j*bs, sda, x_n+j, z_n, z_n, x_t, z_t+j, z_t+j, 0, alg);
		}

	}



void dtrsv_n_lib(int m, int n, int inverted_diag, double *pA, int sda, double *x)
	{

	if(m<=0 || n<=0)
		return;

	// suppose m>=n
	if(m<n)
		m = n;
	
	const int bs = 4;
	
	int j;
	
	double *y;

	// blocks of 4 (pA is supposed to be properly aligned)
	y  = x;

	j = 0;
	for(; j<n-7; j+=8)
		{

		kernel_dtrsv_n_8_lib4(j, inverted_diag, pA, sda, x, y);

		pA += 2*bs*sda;
		y  += 2*bs;

		}
	if(j<n-3)
		{

		kernel_dtrsv_n_4_lib4(j, inverted_diag, pA, x, y);

		pA += bs*sda;
		y  += bs;
		j  += 4;

		}
	if(j<n)
		{

		kernel_dtrsv_n_4_vs_lib4(m-j, n-j, j, inverted_diag, pA, x, y);

		pA += bs*sda;
		y  += bs;
		j  += 4;

		}
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX)
	for(; j<m-11; j+=12)
		{
		kernel_dgemv_n_12_lib4(n, pA, sda, x, y, y, -1);
		pA += 3*sda*bs;
		y  += 3*bs;
		}
	if(j<m-8)
		{
		kernel_dgemv_n_12_vs_lib4(m-j, n, pA, sda, x, y, y, -1);
		pA += 3*sda*bs;
		y  += 3*bs;
		j  += 12;
		}
#else
	for(; j<m-7; j+=8)
		{
		kernel_dgemv_n_8_lib4(n, pA, sda, x, y, y, -1);
		pA += 2*sda*bs;
		y  += 2*bs;
		}
#endif
	if(j<m-4)
		{
		kernel_dgemv_n_8_vs_lib4(m-j, n, pA, sda, x, y, y, -1);
		pA += 2*sda*bs;
		y  += 2*bs;
		j  += 8;
		}
	if(j<m)
		{
		kernel_dgemv_n_4_vs_lib4(m-j, n, pA, x, y, y, -1);
		pA += sda*bs;
		y  += bs;
		j  += 4;
		}

	}



void dtrsv_t_lib(int m, int n, int inverted_diag, double *pA, int sda, double *x)
	{

	if(m<=0 || n<=0)
		return;
	
	const int bs = 4;
	
	int j;
	
/*	double *y;*/
	
	j=0;
	if(n%4==1)
		{
		kernel_dtrsv_t_1_lib4(m-n+j+1, inverted_diag, pA+(n/bs)*bs*sda+(n-1)*bs, sda, x+n-j-1);
		j++;
		}
	else if(n%4==2)
		{
		kernel_dtrsv_t_2_lib4(m-n+j+2, inverted_diag, pA+(n/bs)*bs*sda+(n-j-2)*bs, sda, x+n-j-2);
		j+=2;
		}
	else if(n%4==3)
		{
		kernel_dtrsv_t_3_lib4(m-n+j+3, inverted_diag, pA+(n/bs)*bs*sda+(n-j-3)*bs, sda, x+n-j-3);
		j+=3;
		}
	for(; j<n-3; j+=4)
		{
		kernel_dtrsv_t_4_lib4(m-n+j+4, inverted_diag, pA+((n-j-4)/bs)*bs*sda+(n-j-4)*bs, sda, x+n-j-4);
		}

	}



// transpose & align lower triangular matrix
void dtrtr_l_lib(int m, int offset, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0)
		return;
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
#if defined(TARGET_X64_AVX2)
	for(; j<m-7; j+=8)
		{
		kernel_dtrtr_l_8_lib4(m-j, mna, pA, sda, pC, sdc);
		pA += 2*bs*(sda+bs);
		pC += 2*bs*(sdc+bs);
		}
#endif
	for(; j<m-3; j+=4)
		{
		kernel_dtrtr_l_4_lib4(m-j, mna, pA, sda, pC);
		pA += bs*(sda+bs);
		pC += bs*(sdc+bs);
		}
	if(j==m)
		{
		return;
		}
	else if(m-j==1)
		{
		pC[0] = pA[0];
		}
	else if(m-j==2)
		{
		corner_dtrtr_l_2_lib4(mna, pA, sda, pC);
		}
	else // if(m-j==3)
		{
		corner_dtrtr_l_3_lib4(mna, pA, sda, pC);
		}
	
	}



// copies a packed matrix into a packed matrix
void dgecp_lib(int m, int n, int offsetA, double *A, int sda, int offsetB, double *B, int sdb)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = D_MR;

	int mna, ii;

	int offA = offsetA%bs;
	int offB = offsetB%bs;

	// A at the beginning of the block
	A -= offA;

	// A at the beginning of the block
	B -= offB;

	// same alignment
	if(offA==offB)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_align_panel_2_0_lib4(n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_align_panel_2_0_lib4(n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_align_panel_3_0_lib4(n, A+offA, B+offB);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
		for(; ii<m-7; ii+=8)
			{
			kernel_align_panel_8_0_lib4(n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_align_panel_4_0_lib4(n, A, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_align_panel_1_0_lib4(n, A, B);
			else if(m-ii==2)
				kernel_align_panel_2_0_lib4(n, A, B);
			else // if(m-ii==3)
				kernel_align_panel_3_0_lib4(n, A, B);
			}
		}
	// skip one element of A
	else if(offA==(offB+1)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna) // mna<=3  ==>  m = { 1, 2 }
				{
				if(m==1)
					{
					kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
					return;
					}
				else //if(m==2 && mna==3)
					{
					kernel_align_panel_2_0_lib4(n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
				//A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_align_panel_2_3_lib4(n, A, sda, B+2);
				A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_align_panel_3_2_lib4(n, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
		for( ; ii<m-7; ii+=8)
			{
			kernel_align_panel_8_1_lib4(n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for( ; ii<m-3; ii+=4)
			{
			kernel_align_panel_4_1_lib4(n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_align_panel_1_0_lib4(n, A+1, B);
			else if(m-ii==2)
				kernel_align_panel_2_0_lib4(n, A+1, B);
			else // if(m-ii==3)
				kernel_align_panel_3_0_lib4(n, A+1, B);
			}
		}
	// skip 2 elements of A
	else if(offA==(offB+2)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_align_panel_2_3_lib4(n, A, sda, B+1);
					return;
					}
				}
			if(mna==1)
				{
				kernel_align_panel_1_0_lib4(n, A+1, B+3);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_align_panel_2_0_lib4(n, A, B+2);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_align_panel_3_3_lib4(n, A, sda, B+1);
				A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
		for(; ii<m-7; ii+=8)
			{
			kernel_align_panel_8_2_lib4(n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_align_panel_4_2_lib4(n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_align_panel_1_0_lib4(n, A+2, B);
			else if(m-ii==2)
				kernel_align_panel_2_0_lib4(n, A+2, B);
			else // if(m-ii==3)
				kernel_align_panel_3_2_lib4(n, A, sda, B);
			}
		}
	// skip 3 elements of A
	else // if(offA==(offB+3)%bs)
		{
		ii = 0;
		// clean up at the beginning
		mna = (4-offB)%bs;
		if(mna>0)
			{
			if(m<mna)
				{
				if(m==1)
					{
					kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
					return;
					}
				else // if(m==2 && mna==3)
					{
					kernel_align_panel_2_0_lib4(n, A+offA, B+offB);
					return;
					}
				}
			if(mna==1)
				{
				kernel_align_panel_1_0_lib4(n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 1;
				}
			else if(mna==2)
				{
				kernel_align_panel_2_0_lib4(n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 2;
				}
			else // if(mna==3)
				{
				kernel_align_panel_3_0_lib4(n, A+offA, B+offB);
				// A += 4*sda;
				B += 4*sdb;
				ii += 3;
				}
			}
		// main loop
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
		for(; ii<m-7; ii+=8)
			{
			kernel_align_panel_8_3_lib4(n, A, sda, B, sdb);
			A += 8*sda;
			B += 8*sdb;
			}
#endif
		for(; ii<m-3; ii+=4)
			{
			kernel_align_panel_4_3_lib4(n, A, sda, B);
			A += 4*sda;
			B += 4*sdb;
			}
		// clean up at the end
		if(ii<m)
			{
			if(m-ii==1)
				kernel_align_panel_1_0_lib4(n, A+3, B);
			else if(m-ii==2)
				kernel_align_panel_2_3_lib4(n, A, sda, B);
			else // if(m-ii==3)
				kernel_align_panel_3_3_lib4(n, A, sda, B);
			}
		}

	}



//#if defined(TARGET_C99_4X4)
// transpose & align general matrix; m and n are referred to the original matrix
void dgetr_lib(int m, int n, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna = (bs-offsetA%bs)%bs;
	int nna = (bs-offsetC%bs)%bs;
	
	int ii;

	ii = 0;

	if(mna>0)
		{
		if(mna==1)
			kernel_dgetr_1_lib4(n, nna, pA, pC, sdc);
		else if(mna==2)
			kernel_dgetr_2_lib4(n, nna, pA, pC, sdc);
		else //if(mna==2)
			kernel_dgetr_3_lib4(n, nna, pA, pC, sdc);
		ii += mna;
		pA += mna + bs*(sda-1);
		pC += mna*bs;
		}
#if defined(TARGET_X64_AVX2)
	for( ; ii<m-7; ii+=8)
		{
		kernel_dgetr_8_lib4(n, nna, pA, sda, pC, sdc);
		pA += 2*bs*sda;
		pC += 2*bs*bs;
		}
#endif
	for( ; ii<m-3; ii+=4)
//	for( ; ii<m; ii+=4)
		{
		kernel_dgetr_4_lib4(n, nna, pA, pC, sdc);
		pA += bs*sda;
		pC += bs*bs;
		}

	// clean-up at the end using smaller kernels
	if(ii==m)
		return;
	
	if(m-ii==1)
		kernel_dgetr_1_lib4(n, nna, pA, pC, sdc);
	else if(m-ii==2)
		kernel_dgetr_2_lib4(n, nna, pA, pC, sdc);
	else if(m-ii==3)
		kernel_dgetr_3_lib4(n, nna, pA, pC, sdc);
		
	return;
	
	}	
//#endif



// transpose an aligned upper triangular matrix into an aligned lower triangular matrix
void dtrtr_u_lib(int m, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii;

	for(ii=0; ii<m-3; ii+=4)
		{
		kernel_dtrtr_u_4_lib4(m-ii, pA+ii*sda+ii*bs, pC+ii*sdc+ii*bs, sdc);
		}
	
	if(ii<m)
		{
		if(m-ii==1)
			{
			pC[ii*sdc+ii*bs] = pA[ii*sda+ii*bs];
			}
		else if(m-ii==2)
			{
			pC[ii*sdc+(ii+0)*bs+0] = pA[ii*sda+(ii+0)*bs+0];
			pC[ii*sdc+(ii+1)*bs+0] = pA[ii*sda+(ii+0)*bs+1];
			pC[ii*sdc+(ii+1)*bs+1] = pA[ii*sda+(ii+1)*bs+1];
			}
		else // if(m-ii==3)
			{
			pC[ii*sdc+(ii+0)*bs+0] = pA[ii*sda+(ii+0)*bs+0];
			pC[ii*sdc+(ii+1)*bs+0] = pA[ii*sda+(ii+0)*bs+1];
			pC[ii*sdc+(ii+2)*bs+0] = pA[ii*sda+(ii+0)*bs+2];
			pC[ii*sdc+(ii+1)*bs+1] = pA[ii*sda+(ii+1)*bs+1];
			pC[ii*sdc+(ii+2)*bs+1] = pA[ii*sda+(ii+1)*bs+2];
			pC[ii*sdc+(ii+2)*bs+2] = pA[ii*sda+(ii+2)*bs+2];
			}

		}

	}



//#if defined(TARGET_C99_4X4)
void dsyttmm_lu_lib(int m, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
	for( ; ii<m-2; ii+=4)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dtrmm_nt_l_4x4_lib4(4+jj, pA+ii*sda, pA+jj*sda, pC+ii*sdc+jj*bs);
			}
		// diagonal
		kernel_dsyttmm_lu_nt_4x4_lib4(ii+4, pA+ii*sda, pC+ii*sdc+ii*bs);
		}
	for( ; ii<m; ii+=2)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii-2; jj+=4)
			{
			kernel_dtrmm_nt_l_2x4_lib4(4+jj, pA+ii*sda, pA+jj*sda, pC+ii*sdc+jj*bs);
			}
		// diagonal
		kernel_dsyttmm_lu_nt_2x2_lib4(ii+2, pA+ii*sda, pC+ii*sdc+ii*bs);
		}

	}
//#endif



//#if defined(TARGET_C99_4X4)
void dsyttmm_ul_lib(int m, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX)
//	for( ; ii<m-2; ii+=4)
	for( ; ii<m-7; ii+=8)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dtrmm_l_u_nt_8x4_lib4(m-ii, pA+ii*sda+ii*bs, sda, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs, sdc, pD+ii*sdd+jj*bs, sdd, alg);
			}
		// diagonal
		kernel_dsyttmm_ul_nt_8x4_lib4(m-ii, pA+ii*sda+ii*bs, sda, pA+ii*sda+ii*bs, pC+ii*sdc+ii*bs, sdc, pD+ii*sdd+ii*bs, sdd, alg);
		kernel_dsyttmm_ul_nt_4x4_lib4(m-ii-4, pA+(ii+4)*sda+(ii+4)*bs, pA+(ii+4)*sda+(ii+4)*bs, pC+(ii+4)*sdc+(ii+4)*bs, pD+(ii+4)*sdd+(ii+4)*bs, alg);
		}
#endif
	for( ; ii<m-3; ii+=4)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dtrmm_l_u_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs, pD+ii*sdd+jj*bs, alg);
			}
		// diagonal
		kernel_dsyttmm_ul_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+ii*sda+ii*bs, pC+ii*sdc+ii*bs, pD+ii*sdd+ii*bs, alg);
		}
	if(ii<m)
		{
		if(m-ii==1)
			{
			// off-diagonal
			jj = 0;
			for( ; jj<ii; jj+=4)
				{
				kernel_dtrmm_l_u_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs, pD+ii*sdd+jj*bs, alg);
				}
			// diagonal

			double
				a_0,
				c_00=0;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 = a_0*a_0;

			if(alg==0)
				{
				pD[ii*sdd+0+(ii+0)*bs] = c_00;
				}
			else
				{
				if(alg==1)
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] + c_00;
					}
				else
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] - c_00;
					}
				}
					
			}
		else if(m-ii==2)
			{
			// off-diagonal
			jj = 0;
			for( ; jj<ii; jj+=4)
				{
				kernel_dtrmm_l_u_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs, pD+ii*sdd+jj*bs, alg);
				}
			// diagonal

			double
				a_0, a_1,
				c_00=0,
				c_10=0, c_11=0;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 = a_0 * a_0;

			a_0 = pA[ii*sda+0+(ii+1)*bs];
			a_1 = pA[ii*sda+1+(ii+1)*bs];

			c_00 += a_0 * a_0;
			c_10 = a_1 * a_0;

			c_11 = a_1 * a_1;

			if(alg==0)
				{
				pD[ii*sdd+0+(ii+0)*bs] = c_00;
				pD[ii*sdd+1+(ii+0)*bs] = c_10;
				pD[ii*sdd+1+(ii+1)*bs] = c_11;
				}
			else
				{
				if(alg==1)
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] + c_00;
					pD[ii*sdd+1+(ii+0)*bs] = pC[ii*sdc+1+(ii+0)*bs] + c_10;
					pD[ii*sdd+1+(ii+1)*bs] = pC[ii*sdc+1+(ii+1)*bs] + c_11;
					}
				else
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] - c_00;
					pD[ii*sdd+1+(ii+0)*bs] = pC[ii*sdc+1+(ii+0)*bs] - c_10;
					pD[ii*sdd+1+(ii+1)*bs] = pC[ii*sdc+1+(ii+1)*bs] - c_11;
					}
				}

			}
		else //if(m-ii==3)
			{
			// off-diagonal
			jj = 0;
			for( ; jj<ii; jj+=4)
				{
				kernel_dtrmm_l_u_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs, pD+ii*sdd+jj*bs, alg);
				}
			// diagonal

			double
				a_0, a_1, a_2,
				c_00=0,
				c_10=0, c_11=0,
				c_20=0, c_21=0, c_22=0;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 = a_0 * a_0;

			a_0 = pA[ii*sda+0+(ii+1)*bs];
			a_1 = pA[ii*sda+1+(ii+1)*bs];

			c_00 += a_0 * a_0;
			c_10 = a_1 * a_0;

			c_11 = a_1 * a_1;

			a_0 = pA[ii*sda+0+(ii+2)*bs];
			a_1 = pA[ii*sda+1+(ii+2)*bs];
			a_2 = pA[ii*sda+2+(ii+2)*bs];

			c_00 += a_0 * a_0;
			c_10 += a_1 * a_0;
			c_20 = a_2 * a_0;

			c_11 += a_1 * a_1;
			c_21 = a_2 * a_1;

			c_22 = a_2 * a_2;
	
			if(alg==0)
				{
				pD[ii*sdd+0+(ii+0)*bs] = c_00;
				pD[ii*sdd+1+(ii+0)*bs] = c_10;
				pD[ii*sdd+2+(ii+0)*bs] = c_20;
				pD[ii*sdd+1+(ii+1)*bs] = c_11;
				pD[ii*sdd+2+(ii+1)*bs] = c_21;
				pD[ii*sdd+2+(ii+2)*bs] = c_22;
				}
			else
				{
				if(alg==1)
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] + c_00;
					pD[ii*sdd+1+(ii+0)*bs] = pC[ii*sdc+1+(ii+0)*bs] + c_10;
					pD[ii*sdd+2+(ii+0)*bs] = pC[ii*sdc+2+(ii+0)*bs] + c_20;
					pD[ii*sdd+1+(ii+1)*bs] = pC[ii*sdc+1+(ii+1)*bs] + c_11;
					pD[ii*sdd+2+(ii+1)*bs] = pC[ii*sdc+2+(ii+1)*bs] + c_21;
					pD[ii*sdd+2+(ii+2)*bs] = pC[ii*sdc+2+(ii+2)*bs] + c_22;
					}
				else
					{
					pD[ii*sdd+0+(ii+0)*bs] = pC[ii*sdc+0+(ii+0)*bs] - c_00;
					pD[ii*sdd+1+(ii+0)*bs] = pC[ii*sdc+1+(ii+0)*bs] - c_10;
					pD[ii*sdd+2+(ii+0)*bs] = pC[ii*sdc+2+(ii+0)*bs] - c_20;
					pD[ii*sdd+1+(ii+1)*bs] = pC[ii*sdc+1+(ii+1)*bs] - c_11;
					pD[ii*sdd+2+(ii+1)*bs] = pC[ii*sdc+2+(ii+1)*bs] - c_21;
					pD[ii*sdd+2+(ii+2)*bs] = pC[ii*sdc+2+(ii+2)*bs] - c_22;
					}
				}

			}
		}
	
	}
//#endif



//#if defined(TARGET_C99_4X4)
void dttmm_ll_lib(int m, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
	for( ; ii<m; ii+=4)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dttmm_ll_nt_4x4_lib4(4+ii-jj, pA+ii*sda+jj*bs, pB+jj*sdb+jj*bs, pC+ii*sdc+jj*bs);
			}
		// diagonal
		corner_dttmm_ll_nt_4x4_lib4(pA+ii*sda+ii*bs, pB+ii*sdb+ii*bs, pC+ii*sdc+ii*bs);
		}

	}
//#endif



//#if defined(TARGET_C99_4X4)
void dttmm_uu_lib(int m, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
	for( ; ii<m-2; ii+=4)
		{
		// diagonal
		corner_dttmm_uu_nt_4x4_lib4(pA+ii*sda+ii*bs, pB+ii*sdb+ii*bs, pC+ii*sdc+ii*bs);
		// off-diagonal
		jj = ii+4;
		for( ; jj<m-2; jj+=4)
			{
			kernel_dttmm_uu_nt_4x4_lib4(4+jj-ii, pA+ii*sda+ii*bs, pB+jj*sdb+ii*bs, pC+ii*sdc+jj*bs);
			}
		if(jj<m)
			{
			kernel_dttmm_uu_nt_4x2_lib4(4+jj-ii, pA+ii*sda+ii*bs, pB+jj*sdb+ii*bs, pC+ii*sdc+jj*bs);
			}
		}
	if(ii<m)
		{
		// diagonal
		corner_dttmm_uu_nt_2x2_lib4(pA+ii*sda+ii*bs, pB+ii*sdb+ii*bs, pC+ii*sdc+ii*bs);
		}

	}
//#endif



//#if defined(TARGET_C99_4X4)
void dgema_lib(int m, int n, int offset, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int mna = (bs-offset%bs)%bs;

	int i;

	i=0;
	for( ; i<n-3; i+=4)
		{
		kernel_dgema_4_lib4(m-i, mna, &pA[i*sda+i*bs], sda, &pC[i*sdc+i*bs], sdc);
		}
	if(n-i==0)
		{
		return;
		}
	else if(n-i==1)
		{
		kernel_dgema_1_lib4(m-i, mna, &pA[i*sda+i*bs], sda, &pC[i*sdc+i*bs], sdc);
		return;
		}
	else if(n-i==2)
		{
		kernel_dgema_2_lib4(m-i, mna, &pA[i*sda+i*bs], sda, &pC[i*sdc+i*bs], sdc);
		return;
		}
	else //if(n-i==3)
		{
		kernel_dgema_3_lib4(m-i, mna, &pA[i*sda+i*bs], sda, &pC[i*sdc+i*bs], sdc);
		return;
		}

	
	}
//#endif



//#if defined(TARGET_C99_4X4)
void dtrma_lib(int m, int offset, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int mna = (bs-offset%bs)%bs;

	int i;

	i=0;
	for( ; i<m-3; i+=4)
		{
		kernel_dtrma_4_lib4(m-i, mna, &pA[i*sda+i*bs], sda, &pC[i*sdc+i*bs], sdc);
		}
	if(m-i==0)
		{
		return;
		}
	else if(m-i==1)
		{
		pC[i*sdc+i*bs] = pA[i*sda+i*bs];
		return;
		}
	else if(m-i==2)
		{
		corner_dtrma_2_lib4(mna, &pA[i*sda+i*bs], &pC[i*sdc+i*bs], sdc);
		return;
		}
	else //if(m-i==3)
		{
		corner_dtrma_3_lib4(mna, &pA[i*sda+i*bs], &pC[i*sdc+i*bs], sdc);
		return;
		}

	
	}
//#endif



#if 0
void dtrinv_lib_old(int m, double *pA, int sda, double *pC, int sdc)
	{

	const int bs = 4;

	int ii, jj;

	for(ii=0; ii<m; ii+=4)
		{
		corner_dtrinv_4x4_lib4_old(pA+ii*sda+ii*bs, pC+ii*sdc+ii*bs);
		for(jj=0; jj<ii; jj+=4)
			{
			kernel_dtrinv_4x4_lib4_old(ii-jj, pC+jj*sdc+jj*bs, pA+ii*sda+jj*bs, pC+ii*sdc+ii*bs, pC+jj*sdc+ii*bs, sdc);
			}
		}
	
	return;

	}
#endif



void dtrinv_lib(int m, double *pA, int sda, double *pC, int sdc)
	{

	if(m<=0)
		return;

	const int bs = 4;

	static double fact[10] = {};

	double *ptr;

	int ii, jj;

	jj=0;
	for(; jj<m-2; jj+=4)
		{
		// convert diagonal block
		ptr = pA+jj*sda+jj*bs;
		fact[0] = 1.0/ptr[0+bs*0];
		fact[1] = ptr[1+bs*0];
		fact[2] = 1.0/ptr[1+bs*1];
		fact[3] = ptr[2+bs*0];
		fact[4] = ptr[2+bs*1];
		fact[5] = 1.0/ptr[2+bs*2];
		fact[6] = ptr[3+bs*0];
		fact[7] = ptr[3+bs*1];
		fact[8] = ptr[3+bs*2];
		if(ptr[3+bs*3]!=0.0)
			fact[9] = 1.0/ptr[3+bs*3];
		else
			fact[9] = 0.0;
		ii = 0;
#if defined(TARGET_X64_AVX) // TODO avx2 !!!!!!!!!
		for(; ii<jj-4; ii+=8)
			{
			kernel_dtrinv_8x4_lib4(jj-ii, &pC[ii*sdc+bs*ii], sdc, &pA[jj*sda+ii*bs], &pC[ii*sdc+jj*bs], sdc, fact);
			}
#endif
		for(; ii<jj; ii+=4)
			{
			kernel_dtrinv_4x4_lib4(jj-ii, &pC[ii*sdc+bs*ii], &pA[jj*sda+ii*bs], &pC[ii*sdc+jj*bs], fact);
			}
		corner_dtrinv_4x4_lib4(fact, pC+jj*sdc+jj*bs);
		}
	for(; jj<m; jj+=2)
		{
		// convert diagonal block
		ptr = pA+jj*sda+jj*bs;
		fact[0] = 1.0/ptr[0+bs*0];
		fact[1] = ptr[1+bs*0];
		if(ptr[1+bs*1]!=0.0)
			fact[2] = 1.0/ptr[1+bs*1];
		else
			fact[2] = 0.0;
		for(ii=0; ii<jj; ii+=4)
			{
			kernel_dtrinv_4x2_lib4(jj-ii, &pC[ii*sdc+bs*ii], &pA[jj*sda+ii*bs], &pC[ii*sdc+jj*bs], fact);
			}
		corner_dtrinv_2x2_lib4(fact, pC+jj*sdc+jj*bs);
		}
	
	return;

	}



void dsyrk_dpotrf_dtrinv_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, double *pE, int sde, double *diag, int alg)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;
	const int d_ncl = D_NCL;
	//const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX2)
		if(i<m-4)
			{
			kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			i += 12;

			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			for(; i<m-8; i+=12)
				{
				kernel_dgemm_dtrsm_nt_12x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
		else if(i<m-4)
			{
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			i += 8;
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
#else
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			i += 4;
#endif
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
#if defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
#endif
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
		// dtrinv
		if(diag[j+0]==0.0) fact[0]=0.0;
		if(diag[j+1]==0.0) fact[2]=0.0;
		if(diag[j+2]==0.0) fact[5]=0.0;
		if(diag[j+3]==0.0) fact[9]=0.0;
		i = 0;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) // TODO add avx2 !!!!!!!!!!!!!!!
		for(; i<j-4; i+=8)
			{
			kernel_dtrinv_8x4_lib4(j-i, &pE[i*sde+bs*i], sde, &pD[j*sdd+i*bs], &pE[i*sde+j*bs], sde, fact);
			}
#endif
		for(; i<j; i+=4)
			{
			kernel_dtrinv_4x4_lib4(j-i, &pE[i*sde+bs*i], &pD[j*sdd+i*bs], &pE[i*sde+j*bs], fact);
			}
		corner_dtrinv_4x4_lib4(fact, pE+j*sde+j*bs);
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x2_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_2x2_lib4(m-i, n-j, 0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		// dtrinv
		if(diag[j+0]==0.0) fact[0]=0.0;
		if(diag[j+1]==0.0) fact[2]=0.0;
		for(i=0; i<j; i+=4)
			{
			kernel_dtrinv_4x2_lib4(j-i, &pE[i*sde+bs*i], &pD[j*sdd+i*bs], &pE[i*sde+j*bs], fact);
			}
		corner_dtrinv_2x2_lib4(fact, pE+j*sde+j*bs);
		}

	}



// TODO add m2 !!!
void dtsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, double *diag, int alg)
	{

	if(m<=0 || n<=0)
		return;

	//printf("\n m = %d, n = %d, k = %d\n", m, n, k);
		
	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	double *dummy;

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_X64_AVX2)
		if(i<m-8)
			{
			//d_print_mat(4, k-i, &pA[(i+4)*sda+i*bs], 4);
			//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
			kernel_dsyrk_dpotrf_nt_12x4_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], sda, &pA[i*sda+i*bs], &pD[i*sdd], sdd, &pD[i*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
			i += 12;

			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;

			for(; i<m-8 && i<k; i+=12)
				{
				//printf("\n%d\n", k-i);
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				//d_print_mat(k-i, 4, &pA[j*sda+i*bs], 4);
				kernel_dgemm_dtrsm_nt_12x4_lib4(1, k-i, j, &pA[i*sda+i*bs], sda, &pA[j*sda+i*bs], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
			for(; i<m-4 && i<k; i+=8)
				{
				//printf("\n%d\n", k-i);
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				//d_print_mat(k-i, 4, &pA[j*sda+i*bs], 4);
				kernel_dgemm_dtrsm_nt_8x4_lib4(1, k-i, j, &pA[i*sda+i*bs], sda, &pA[j*sda+i*bs], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
			for(; i<m-2 && i<k; i+=4)
				{
				//printf("\n%d\n", k-i);
				//d_print_mat(k-i, 4, &pA[i*sda+i*bs], 4);
				//d_print_mat(k-i, 4, &pA[j*sda+i*bs], 4);
				kernel_dgemm_dtrsm_nt_4x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 4, &pA[(k0+k+j)*bs+i*sda], 4);
				//kernel_dtrmm_dtrsm_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pC[j*bs+i*sdc], &pA[(k+k0+j)*bs+i*sda], fact, alg);
				}
			for(; i<m && i<k; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			for(; i<m-8; i+=12)
				{
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				kernel_dgemm_dtrsm_nt_12x4_lib4(0, 0, j, dummy, 0, dummy, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
			for(; i<m-4; i+=8)
				{
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 4, &pA[(k0+k+j)*bs+i*sda], 4);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}		
		else if(i<m-4)
			{
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], sda, &pA[i*sda+i*bs], &pD[i*sdd], sdd, &pD[i*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			i += 8;
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
#else
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX)
			//d_print_mat(4, k-i, &pA[(i+4)*sda+i*bs], 4);
			//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
			kernel_dsyrk_dpotrf_nt_8x4_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], sda, &pA[i*sda+i*bs], &pD[i*sdd], sdd, &pD[i*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg, 0);
			//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			//kernel_dtsyrk_dpotrf_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			i += 4;
#endif
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;

#if defined(TARGET_X64_AVX)
			for(; i<m-4 && i<k; i+=8)
				{
				//printf("\n%d\n", k-i);
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				//d_print_mat(k-i, 4, &pA[j*sda+i*bs], 4);
				kernel_dgemm_dtrsm_nt_8x4_lib4(1, k-i, j, &pA[i*sda+i*bs], sda, &pA[j*sda+i*bs], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
#endif
			for(; i<m-2 && i<k; i+=4)
				{
				//printf("\n%d\n", k-i);
				//d_print_mat(k-i, 4, &pA[i*sda+i*bs], 4);
				//d_print_mat(k-i, 4, &pA[j*sda+i*bs], 4);
				kernel_dgemm_dtrsm_nt_4x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 4, &pA[(k0+k+j)*bs+i*sda], 4);
				//kernel_dtrmm_dtrsm_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pC[j*bs+i*sdc], &pA[(k+k0+j)*bs+i*sda], fact, alg);
				}
			for(; i<m && i<k; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
#if defined(TARGET_X64_AVX)
			for(; i<m-4; i+=8)
				{
				//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
				kernel_dgemm_dtrsm_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x4_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 4, &pA[(k0+k+j)*bs+i*sda], 4);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x4_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
#endif
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			//kernel_dtsyrk_dpotrf_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			if(fact[5]==0.0) fact[5]=1.0;
			if(fact[9]==0.0) fact[9]=1.0;
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x2_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			i += 4;

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4 && i<k; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x2_lib4(1, k-i, j, &pA[i*sda+i*bs], sda, &pA[j*sda+i*bs], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 2, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
#endif
			for(; i<m-2 && i<k; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x2_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 2, &pA[(k0+k+j)*bs+i*sda], 4);
				}
			for(; i<m && i<k; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x2_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[j*sda+i*bs], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_dtrsm_nt_8x2_lib4(1, 0, j, dummy, 0, dummy, &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
				//d_print_pmat(8, 2, bs, &pA[(k0+k+j)*bs+i*sda], sda);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_dtrsm_nt_4x2_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				//d_print_mat(4, 2, &pA[(k0+k+j)*bs+i*sda], 4);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_dtrsm_nt_2x2_lib4(0, 0, j, dummy, dummy, &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
				}
			}
		else //if(i<m)
			{
			kernel_dsyrk_dpotrf_nt_2x2_lib4(m-i, n-j, 1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg, 0);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}

	}



void dgemm_diag_left_lib(int m, int n, double *dA, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int ii;

	ii = 0;
	for( ; ii<m-3; ii+=4)
		{
		kernel_dgemm_diag_left_4_lib4(n, &dA[ii], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		}
	if(m-ii>0)
		{
		if(m-ii==1)
			kernel_dgemm_diag_left_1_lib4(n, &dA[ii], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		else if(m-ii==2)
			kernel_dgemm_diag_left_2_lib4(n, &dA[ii], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		else // if(m-ii==3)
			kernel_dgemm_diag_left_3_lib4(n, &dA[ii], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		}
	
	}



void dgemm_diag_right_lib(int m, int n, double *pA, int sda, double *dB, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0 || n<=0)
		return;

	const int bs = 4;

	int ii;

	ii = 0;
	for( ; ii<n-3; ii+=4)
		{
		kernel_dgemm_diag_right_4_lib4(m, &pA[ii*bs], sda, &dB[ii], &pC[ii*bs], sdc, &pD[ii*bs], sdd, alg);
		}
	if(n-ii>0)
		{
		if(n-ii==1)
			kernel_dgemm_diag_right_1_lib4(m, &pA[ii*bs], sda, &dB[ii], &pC[ii*bs], sdc, &pD[ii*bs], sdd, alg);
		else if(n-ii==2)
			kernel_dgemm_diag_right_2_lib4(m, &pA[ii*bs], sda, &dB[ii], &pC[ii*bs], sdc, &pD[ii*bs], sdd, alg);
		else // if(n-ii==3)
			kernel_dgemm_diag_right_3_lib4(m, &pA[ii*bs], sda, &dB[ii], &pC[ii*bs], sdc, &pD[ii*bs], sdd, alg);
		}
	
	}



void dsyrk_diag_left_right_lib(int m, double *dAl, double *dAr, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg)
	{

	if(m<=0)
		return;

	const int bs = 4;

	int ii;

	for(ii=0; ii<m-3; ii+=4)
		{
		kernel_dsyrk_diag_left_right_4_lib4(ii+4, &dAl[ii], &dAr[0], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		}
	if(m-ii>0)
		{
		if(m-ii==1)
			kernel_dsyrk_diag_left_right_1_lib4(ii+1, &dAl[ii], &dAr[0], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		else if(m-ii==2)
			kernel_dsyrk_diag_left_right_2_lib4(ii+2, &dAl[ii], &dAr[0], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		else // if(m-ii==3)
			kernel_dsyrk_diag_left_right_3_lib4(ii+3, &dAl[ii], &dAr[0], &pB[ii*sdb], &pC[ii*sdc], &pD[ii*sdd], alg);
		}
	
	}
			


void dgemv_diag_lib(int m, double *dA, double *x, double *y, double *z, int alg)
	{

	if(m<=0)
		return;
	
	kernel_dgemv_diag_lib4(m, dA, x, y, z, alg);

	}

	

