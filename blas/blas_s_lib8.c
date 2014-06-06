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

#include "../include/kernel_s_lib8.h"



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void sgemm_ppp_nt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, int alg)
	{

	const int bs = 8;

	int i, j, jj;
	
	i = 0;
	for(; i<m-8; i+=16)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_pp_nt_12x4_lib8(k, &pA[0+i*sda], &pA[0+(i+8)*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+8)*sdc], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+8)*sdc], bs, alg);
			kernel_sgemm_pp_nt_12x4_lib8(k, &pA[0+i*sda], &pA[0+(i+8)*sda], &pB[4+j*sdb], &pC[0+(j+4)*bs+i*sdc], &pC[0+(j+4)*bs+(i+8)*sdc], &pC[0+(j+4)*bs+i*sdc], &pC[0+(j+4)*bs+(i+8)*sdc], bs, alg);
			}
		jj = 0;
		for(; jj<n-j-3; jj+=4)
			{
			kernel_sgemm_pp_nt_16x4_lib8(k, &pA[0+i*sda], &pA[0+(i+8)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+8)*sdc], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+8)*sdc], bs, alg);
			}
/*		for(; jj<n-j-1; jj+=2)*/
/*			{*/
/*			kernel_sgemm_pp_nt_16x2_lib8(k, &pA[0+i*sda], &pA[0+(i+8)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+8)*sdc], bs, alg);*/
/*			}*/
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_sgemm_pp_nt_16x1_lib8(k, &pA[0+i*sda], &pA[0+(i+8)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+8)*sdc], bs, alg);*/
/*			}*/
		}
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_pp_nt_8x8_lib8(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
/*			kernel_sgemm_pp_nt_8x4_lib8(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, alg);*/
/*			kernel_sgemm_pp_nt_8x4_lib8(k, &pA[0+i*sda], &pB[4+j*sdb], &pC[0+(j+4)*bs+i*sdc], bs, alg);*/
			}
		jj = 0;
		for(; jj<n-j-3; jj+=4)
			{
			kernel_sgemm_pp_nt_8x4_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+i*sdc], bs, alg);
			}
/*		for(; jj<n-j-1; jj+=2)*/
/*			{*/
/*			kernel_sgemm_pp_nt_8x2_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_sgemm_pp_nt_8x1_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
		}
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-7; j+=8)
			{
			kernel_sgemm_pp_nt_4x8_lib8(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
/*			kernel_sgemm_pp_nt_4x4_lib8(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);*/
/*			kernel_sgemm_pp_nt_4x4_lib8(k, &pA[0+i*sda], &pB[4+j*sdb], &pC[0+(j+4)*bs+i*sdc], &pC[0+(j+4)*bs+i*sdc], bs, alg);*/
			}
		jj = 0;
		for(; jj<n-j-3; jj+=4)
			{
			kernel_sgemm_pp_nt_4x4_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+i*sdc], bs, alg);
			}
/*		for(; jj<n-j-1; jj+=2)*/
/*			{*/
/*			kernel_sgemm_pp_nt_4x2_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_sgemm_pp_nt_4x1_lib8(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
		}
	
	}

