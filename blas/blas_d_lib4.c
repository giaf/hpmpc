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



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, double *pD, int sdd, int alg, int tc, int td)
	{

	const int bs = 4;

	int i, j, jj;
	
	if(tc==0)
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
					kernel_dgemm_nt_m8x4_lib4(m-i, k, &pA[i*sda], sda, &pB[j*sdb], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_m8x2_lib4(m-i, k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, &pD[(j+jj)*bs+i*sdd], sdd, alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_8x1_lib4(k, &pA[i*sda], sda, &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], sdc, alg);*/
		/*			}*/
				}
			if(i<m)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_m4x4_lib4(m-i, k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_m4x2_lib4(m-i, k, &pA[i*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], &pD[(j+jj)*bs+i*sdd], alg, tc, td);
					}
				}
#else
#if defined(TARGET_X64_AVX2)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
			for(; i<m-2; i+=4)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], &pD[(j+jj)*bs+i*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
			for(; i<m; i+=2)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], &pD[(j+jj)*bs+i*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
#endif
			}
		else // td==1
			{
			i = 0;
#if defined(TARGET_X64_AVX2)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
			for(; i<m-2; i+=4)
			//for(; i<m; i+=4)
				{
				j = 0;
				for(; j<n-2; j+=4)
				//for(; j<n; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					//d_print_mat(4, 4, &pD[i*bs+j*sdd], 4);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], &pD[i*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
			for(; i<m; i+=2)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[j*bs+i*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[(j+jj)*bs+i*sdc], &pD[i*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
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
#if defined(TARGET_X64_AVX2)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
			for(; i<m-2; i+=4)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], &pD[(j+jj)*bs+i*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
			for(; i<m; i+=2)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[j*bs+i*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], &pD[(j+jj)*bs+i*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
#endif
			}
		else // td==1
			{
			i = 0;
#if defined(TARGET_X64_AVX2)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
			for(; i<m-2; i+=4)
			//for(; i<m; i+=4)
				{
				j = 0;
				for(; j<n-2; j+=4)
				//for(; j<n; j+=4)
					{
					kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					//d_print_mat(4, 4, &pD[i*bs+j*sdd], 4);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], &pD[i*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
			for(; i<m; i+=2)
				{
				j = 0;
				for(; j<n-2; j+=4)
					{
					kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pC[i*bs+j*sdc], &pD[i*bs+j*sdd], alg, tc, td);
					}
				jj = 0;
		/*		for(; jj<n-j-1; jj+=2)*/
				for(; jj<n-j; jj+=2)
					{
					kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[jj+j*sdb], &pC[i*bs+(j+jj)*sdc], &pD[i*bs+(j+jj)*sdd], alg, tc, td);
					}
		/*		for(; jj<n-j; jj++)*/
		/*			{*/
		/*			kernel_dgemm_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], alg);*/
		/*			}*/
				}
			}
		}
	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is upper triangular                         */
void dtrmm_l_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
#if defined(TARGET_X64_AVX2)
	for(; i<m-8; i+=12)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_l_nt_12x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		if(n-j==1)
			{
			corner_dtrmm_l_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			corner_dtrmm_l_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==2)
			{
			corner_dtrmm_l_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			corner_dtrmm_l_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==3)
			{
			corner_dtrmm_l_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			corner_dtrmm_l_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		}
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_l_nt_8x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		if(n-j==1)
			{
			corner_dtrmm_l_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		else if(n-j==2)
			{
			corner_dtrmm_l_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		else if(n-j==3)
			{
			corner_dtrmm_l_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], sda, &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], sdc);
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_l_nt_4x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		if(n-j==1)
			{
			corner_dtrmm_l_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==2)
			{
			corner_dtrmm_l_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		else if(n-j==3)
			{
			corner_dtrmm_l_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc]);
			}
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is lower triangular                         */
void dtrmm_u_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	i=0;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
	for( ; i<m-4; i+=8)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_u_nt_8x4_lib4(j+4, &pA[i*sda], sda, &pB[j*sdb], &pC[i*sdc+j*bs], sdc);
			}
		if(j<n)
			{
			kernel_dtrmm_u_nt_8x2_lib4(j+2, &pA[i*sda], sda, &pB[j*sdb], &pC[i*sdc+j*bs], sdc);
			}
		}
	for( ; i<m; i+=4)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_u_nt_4x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_u_nt_4x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
#else
	for( ; i<m-2; i+=4)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_u_nt_4x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_u_nt_4x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
	if(i<m)
		{
		j=0;
		for( ; j<n-2; j+=4)
			{
			kernel_dtrmm_u_nt_2x4_lib4(j+4, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		if(j<n)
			{
			kernel_dtrmm_u_nt_2x2_lib4(j+2, &pA[i*sda], &pB[j*sdb], &pC[i*sdc+j*bs]);
			}
		}
#endif

	}



void dsyrk_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pD, int sdd, double *pC, int sdc, int alg)
	{
	const int bs = 4;
	
	int i, j;
	
/*	int n = m;*/
	
	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			kernel_dsyrk_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, alg);
			i += 8;
#else
			kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg);
			i += 4;
#endif
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_nt_8x4_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, alg, 0, 0);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg, 0, 0);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_nt_2x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg, 0, 0);
				}
			}
		else //if(i<m-2)
			{
			kernel_dsyrk_nt_4x4_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg);
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg);
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
				kernel_dgemm_nt_8x2_lib4(k, &pA[i*sda], sda, &pB[j*sdb], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, alg, 0, 0);
				}
#endif
			for(; i<m-2; i+=4)
				{
				kernel_dgemm_nt_4x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg, 0, 0);
				}
			for(; i<m; i+=2)
				{
				kernel_dgemm_nt_2x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg, 0, 0);
				}
			}
		else //if(i<m)
			{
			kernel_dsyrk_nt_2x2_lib4(k, &pA[i*sda], &pB[j*sdb], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], alg);
			}
		}

	}



void dpotrf_lib(int m, int n, double *pD, int sdd, double *pC, int sdc, double *diag)
	{
	const int bs = 4;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	double *dummy;

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(0, 0, j, dummy, 0, dummy, &pC[i*sdc], sdc, &pC[j*sdc], &pD[j*bs+i*sdd], sdd, &pC[j*bs+i*sdc], sdc, fact, 1);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
		else //if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
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
			kernel_dsyrk_dpotrf_nt_4x2_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
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
			kernel_dsyrk_dpotrf_nt_2x2_lib4(0, 0, j, dummy, dummy, &pC[i*sdc], &pC[j*sdc], &pD[j*bs+i*sdd], &pC[j*bs+i*sdc], fact, 1);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}

	}



// TODO invert k and n !!!!!!!!!!! DONE
void dsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, double *diag, int alg)
	{
	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
		else //if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dsyrk_dpotrf_nt_4x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dsyrk_dpotrf_nt_2x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}

	}



// TODO modify kernels instead
void dgemv_n_lib(int m, int n, double *pA, int sda, double *x, double *y, int alg) // pA has to be aligned !!!
	{
	
	const int bs = 4;
	
	int i, j;

//	static double y_temp[8] = {};

	j=0;
/*	for(; j<n-7; j+=8)*/
	for(; j<m-4; j+=8)
		{
//		y_temp[0] = y[0];
//		y_temp[1] = y[1];
//		y_temp[2] = y[2];
//		y_temp[3] = y[3];
//		y_temp[4] = y[4];
//		y_temp[5] = y[5];
//		y_temp[6] = y[6];
//		y_temp[7] = y[7];
//		kernel_dgemv_n_8_lib4(n, pA, pA+sda*bs, x, y_temp, alg);
		kernel_dgemv_n_8_lib4(n, pA, sda, x, y, alg);
//		y[0] = y_temp[0];
//		y[1] = y_temp[1];
//		y[2] = y_temp[2];
//		y[3] = y_temp[3];
//		if(m-j<8)
//			{
//			for(i=4; i<m-j-4; i++)
//				y[i] = y_temp[i];
//			}
//		else
//			{
//			y[4] = y_temp[4];
//			y[5] = y_temp[5];
//			y[6] = y_temp[6];
//			y[7] = y_temp[7];
//			}
		pA += 2*sda*bs;
		y  += 2*bs;
		}
/*	for(; j<n-3; j+=4)*/
	for(; j<m; j+=4)
		{
//		y_temp[0] = y[0];
//		y_temp[1] = y[1];
//		y_temp[2] = y[2];
//		y_temp[3] = y[3];
//		kernel_dgemv_n_4_lib4(n, pA, x, y_temp, alg);
		kernel_dgemv_n_4_lib4(n, pA, x, y, alg);
//		if(m-j<4)
//			{
//			for(i=0; i<m-j; i++)
//				y[i] = y_temp[i];
//			}
//		else
//			{
//			y[0] = y_temp[0];
//			y[1] = y_temp[1];
//			y[2] = y_temp[2];
//			y[3] = y_temp[3];
//			}
		pA += sda*bs;
		y  += bs;
		}
/*	for(; j<m-1; j+=2)*/
/*		{*/
/*		kernel_dgemv_n_2_lib4(n, pA, x, y, alg);*/
/*		pA += 2;*/
/*		y  += 2;*/
/*		}*/
/*	for(; j<m; j++)*/
/*		{*/
/*		kernel_dgemv_n_1_lib4(n, pA, x, y, alg);*/
/*		pA += 1;*/
/*		y  += 1;*/
/*		}*/

	}



void dgemv_t_lib(int m, int n, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int j;
	
	j=0;
	for(; j<n-7; j+=8)
		{
		kernel_dgemv_t_8_lib4(m, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n-3; j+=4)
		{
		kernel_dgemv_t_4_lib4(m, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n-1; j+=2)
		{
		kernel_dgemv_t_2_lib4(m, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n; j++)
		{
		kernel_dgemv_t_1_lib4(m, pA+j*bs, sda, x, y+j, alg);
		}

	}



void dtrmv_u_n_lib(int m, double *pA, int sda, double *x, double *y, int alg)
	{

	const int bs = 4;
	
	int j;
	
	j=0;
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

	const int bs = 4;
	
	int j;
	
	double *ptrA;
	
	j=0;
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
void dsymv_lib(int m, int n, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	if(m<n)
		n = m;
	
	int j, j0;
	
	if(alg==0)
		{
		for(j=0; j<m; j++)
			y[j] = 0.0;
		alg = 1;
		}
	
	j=0;
	for(; j<n-3; j+=4)
		{
		kernel_dsymv_4_lib4(m-j, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
		}
	if(j<n)
		{
		if(n-j==1)
			{
			kernel_dsymv_1_lib4(m-j, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
			}
		else if(n-j==2)
			{
			kernel_dsymv_2_lib4(m-j, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
			}
		else // if(n-j==3)
			{
			kernel_dsymv_3_lib4(m-j, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
			}
		}

	}



// it moves vertically across block
void dmvmv_lib(int m, int n, double *pA, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int alg)
	{
	
	const int bs = 4;

	int j;
	
	if(alg==0)
		{
		for(j=0; j<m; j++)
			y_n[j] = 0.0;
		for(j=0; j<n; j++)
			y_t[j] = 0.0;
		alg = 1;
		}
	
	j=0;
	for(; j<n-3; j+=4)
		{
		kernel_dsymv_4_lib4(m, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}
	for(; j<n-1; j+=2)
		{
		kernel_dsymv_2_lib4(m, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}
	for(; j<n; j++)
		{
		kernel_dsymv_1_lib4(m, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}

	}



// the diagonal is inverted !!!
void dtrsv_dgemv_n_lib(int m, int n, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int j;
	
	double *y;

	// blocks of 4 (pA is supposed to be properly aligned)
	y  = x;

	j = 0;
	for(; j<m-7; j+=8)
		{

		kernel_dtrsv_n_8_lib4(j, pA, sda, x, y); // j+8 !!!

		pA += 2*bs*sda;
		y  += 2*bs;

		}
	if(j<m-3)
		{

		kernel_dtrsv_n_4_lib4(j, 4, pA, x, y); // j+4 !!!

		pA += bs*sda;
		y  += bs;
		j+=4;

		}
	if(j<m) // !!! suppose that there are enough nx after !!! => x padded with enough zeros at the end !!!
		{

		kernel_dtrsv_n_4_lib4(j, m-j, pA, x, y); // j+4 !!!

		pA += bs*sda;
		y  += bs;
		j+=4;

		}
/*	for(; j<n-7; j+=8)*/
	for(; j<n-4; j+=8)
		{

		kernel_dgemv_n_8_lib4(m, pA, sda, x, y, -1);

		pA += 2*sda*bs;
		y  += 2*bs;

		}
/*	for(; j<n-3; j+=4)*/
	for(; j<n; j+=4)
		{

		kernel_dgemv_n_4_lib4(m, pA, x, y, -1);

		pA += sda*bs;
		y  += bs;

		}
/*	for(; j<n-1; j+=2)*/
/*		{*/

/*		kernel_dgemv_n_2_lib4(m, pA, x, y, -1);*/

/*		pA += 2;*/
/*		y  += 2;*/

/*		}*/
/*	for(; j<n; j+=1)*/
/*		{*/

/*		kernel_dgemv_n_1_lib4(m, pA, x, y, -1);*/

/*		pA += 1;*/
/*		y  += 1;*/

/*		}*/

	}



// the diagonal is inverted !!!
void dtrsv_dgemv_t_lib(int n, int m, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int j;
	
/*	double *y;*/
	
	j=0;
	if(n%4==1)
		{
		kernel_dtrsv_t_1_lib4(m-n+j+1, pA+(n/bs)*bs*sda+(n-1)*bs, sda, x+n-j-1);
		j++;
		}
	else if(n%4==2)
		{
		kernel_dtrsv_t_2_lib4(m-n+j+2, pA+(n/bs)*bs*sda+(n-j-2)*bs, sda, x+n-j-2);
		j+=2;
		}
	else if(n%4==3)
		{
		kernel_dtrsv_t_3_lib4(m-n+j+3, pA+(n/bs)*bs*sda+(n-j-3)*bs, sda, x+n-j-3);
		j+=3;
		}
	for(; j<n-3; j+=4)
		{
		kernel_dtrsv_t_4_lib4(m-n+j+4, pA+((n-j-4)/bs)*bs*sda+(n-j-4)*bs, sda, x+n-j-4);
		}

	}



// transpose & align lower triangular matrix
void dtrtr_l_lib(int m, int offset, double *pA, int sda, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
	for(; j<m-3; j+=4)
		{
		kernel_dtran_4_lib4(m-j, mna, pA, sda, pC);
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
		corner_dtran_2_lib4(mna, pA, sda, pC);
		}
	else // if(m-j==3)
		{
		corner_dtran_3_lib4(mna, pA, sda, pC);
		}
	
	}



//#if defined(TARGET_C99_4X4)
// transpose & align general matrix; m and n are referred to the original matrix
void dgetr_lib(int m, int mna, int n, int offset, double *pA, int sda, double *pC, int sdc)
	{

	const int bs = 4;

	int nna = (bs-offset%bs)%bs;
	
	int ii;

	ii = 0;

	if(mna>0)
		{
		// TODO using smaller kernels
		kernel_dgetr_1_lib4(n, nna, pA, pC, sdc);
		ii += mna;
		pA += mna + bs*(sda-1);
		pC += mna*bs;
		}
	
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

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
	for( ; ii<m-2; ii+=4)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dtrmm_u_nt_4x4_lib4(4+jj, pA+ii*sda, pA+jj*sda, pC+ii*sdc+jj*bs);
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
			kernel_dtrmm_u_nt_2x4_lib4(4+jj, pA+ii*sda, pA+jj*sda, pC+ii*sdc+jj*bs);
			}
		// diagonal
		kernel_dsyttmm_lu_nt_2x2_lib4(ii+2, pA+ii*sda, pC+ii*sdc+ii*bs);
		}

	}
//#endif



//#if defined(TARGET_C99_4X4)
void dsyttmm_ul_lib(int m, double *pA, int sda, double *pC, int sdc)
	{

	const int bs = 4;

	int ii, jj;
	
	ii = 0;
//	for( ; ii<m-2; ii+=4)
	for( ; ii<m-3; ii+=4)
		{
		// off-diagonal
		jj = 0;
		for( ; jj<ii; jj+=4)
			{
			kernel_dtrmm_l_u_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pA+jj*sda+ii*bs, pC+ii*sdc+jj*bs);
			}
		// diagonal
		kernel_dsyttmm_ul_nt_4x4_lib4(m-ii, pA+ii*sda+ii*bs, pC+ii*sdc+ii*bs);
		}
	if(ii<m)
		{
		if(m-ii==1)
			{
			double
				a_0,
				c_00;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 = a_0*a_0;

			pC[ii*sdc+0+(ii+0)*bs] = c_00;
			}
		else if(m-ii==2)
			{
			double
				a_0, a_1,
				c_00,
				c_10, c_11;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 += a_0 * a_0;

			a_0 = pA[ii*sda+0+(ii+1)*bs];
			a_1 = pA[ii*sda+1+(ii+1)*bs];

			c_00 += a_0 * a_0;
			c_10 += a_1 * a_0;

			c_11 += a_1 * a_1;

			pC[ii*sdc+0+(ii+0)*bs] = c_00;
			pC[ii*sdc+1+(ii+0)*bs] = c_10;
			pC[ii*sdc+1+(ii+1)*bs] = c_11;
			}
		else //if(m-ii==3)
			{
			double
				a_0, a_1, a_2,
				c_00,
				c_10, c_11,
				c_20, c_21, c_22;

			a_0 = pA[ii*sda+0+(ii+0)*bs];

			c_00 += a_0 * a_0;

			a_0 = pA[ii*sda+0+(ii+1)*bs];
			a_1 = pA[ii*sda+1+(ii+1)*bs];

			c_00 += a_0 * a_0;
			c_10 += a_1 * a_0;

			c_11 += a_1 * a_1;

			a_0 = pA[ii*sda+0+(ii+2)*bs];
			a_1 = pA[ii*sda+1+(ii+2)*bs];
			a_2 = pA[ii*sda+2+(ii+2)*bs];

			c_00 += a_0 * a_0;
			c_10 += a_1 * a_0;
			c_20 += a_2 * a_0;

			c_11 += a_1 * a_1;
			c_21 += a_2 * a_1;

			c_22 += a_2 * a_2;
	
			pC[ii*sdc+0+(ii+0)*bs] = c_00;
			pC[ii*sdc+1+(ii+0)*bs] = c_10;
			pC[ii*sdc+2+(ii+0)*bs] = c_20;
			pC[ii*sdc+1+(ii+1)*bs] = c_11;
			pC[ii*sdc+2+(ii+1)*bs] = c_21;
			pC[ii*sdc+2+(ii+2)*bs] = c_22;
			}
		}
	
	}
//#endif



//#if defined(TARGET_C99_4X4)
void dttmm_ll_lib(int m, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{

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
		for(ii=0; ii<jj; ii+=4)
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
	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			kernel_dsyrk_dpotrf_nt_8x4_lib4(0, k, j, &pA[i*sda], sda, &pA[j*sda], &pD[i*sdd], sdd, &pD[j*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
		else //if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
		for(i=0; i<j; i+=4)
			{
			kernel_dtrinv_4x4_lib4(j-i, &pE[i*sde+bs*i], &pA[j*sda+(k0+k+i)*bs], &pE[i*sde+j*bs], fact);
			}
		corner_dtrinv_4x4_lib4(fact, pE+j*sde+j*bs);
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dsyrk_dpotrf_nt_2x2_lib4(0, k, j, &pA[i*sda], &pA[j*sda], &pD[i*sdd], &pD[j*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dtrinv_4x2_lib4(j-i, &pE[i*sde+bs*i], &pA[j*sda+(k0+k+i)*bs], &pE[i*sde+j*bs], fact);
			}
		corner_dtrinv_2x2_lib4(fact, pE+j*sde+j*bs);
		}

	}



// TODO add m2 !!!
void dtsyrk_dpotrf_lib(int m, int n, int k, double *pA, int sda, double *pC, int sdc, double *pD, int sdd, double *diag, int alg)
	{

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
		if(i<m-4)
			{
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			//d_print_mat(4, k-i, &pA[(i+4)*sda+i*bs], 4);
			//d_print_pmat(8, k-i, bs, &pA[i*sda+i*bs], sda);
			kernel_dsyrk_dpotrf_nt_8x4_lib4(1, k-i, j, &pA[i*sda+i*bs], sda, &pA[i*sda+i*bs], &pD[i*sdd], sdd, &pD[i*sdd], &pC[j*bs+i*sdc], sdc, &pD[j*bs+i*sdd], sdd, fact, alg);
			//d_print_pmat(8, 4, bs, &pA[(k0+k+j)*bs+i*sda], sda);
			i += 8;
#else
			kernel_dsyrk_dpotrf_nt_4x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
			//kernel_dtsyrk_dpotrf_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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

#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
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
		else //if(i<m-2)
			{
			kernel_dsyrk_dpotrf_nt_4x4_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
			//kernel_dtsyrk_dpotrf_nt_4x4_lib4(k-i, j, &pA[i*sda+i*bs], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dsyrk_dpotrf_nt_4x2_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
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
			kernel_dsyrk_dpotrf_nt_2x2_lib4(1, k-i, j, &pA[i*sda+i*bs], &pA[i*sda+i*bs], &pD[i*sdd], &pD[i*sdd], &pC[j*bs+i*sdc], &pD[j*bs+i*sdd], fact, alg);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			if(fact[0]==0.0) fact[0]=1.0;
			if(fact[2]==0.0) fact[2]=1.0;
			}
		}

	}




