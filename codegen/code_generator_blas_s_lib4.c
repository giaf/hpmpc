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
#define PNZ S_MR*((NZ+S_MR-1)/S_MR)
#define PNX S_MR*((NX+S_MR-1)/S_MR)
#define CNZ S_NCL*((NZ+S_NCL-1)/S_NCL)
#define CNX S_NCL*((NX+S_NCL-1)/S_NCL)
#define PAD (S_NCL-NX%S_NCL)%S_NCL
#define CNL NX+PAD+CNZ



void strmm_code_generator(FILE *f, int m, int n)
	{
	
	int i, j;
	
	const int bs = 4;
	
	const int sda = CNX;
	const int sdb = CNL;
	const int sdc = CNL;

	i = 0;
#if defined(TARGET_CORTEX_A15)
	for(; i<m-16; i+=12)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
//			kernel_strmm_pp_nt_12x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	kernel_strmm_pp_nt_12x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc);
			}
		if(n-j==1)
			{
/*			corner_strmm_pp_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		else if(n-j==2)
			{
/*			corner_strmm_pp_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		else if(n-j==3)
			{
/*			corner_strmm_pp_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		}
	if(m-i>12)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
//			kernel_strmm_pp_nt_8x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	kernel_strmm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
			}
		if(n-j==1)
			{
/*			corner_strmm_pp_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==2)
			{
/*			corner_strmm_pp_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==3)
			{
/*			corner_strmm_pp_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		i+=8;
		j = 0;
		for(; j<n-3; j+=4)
			{
//			kernel_strmm_pp_nt_8x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	kernel_strmm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
			}
		if(n-j==1)
			{
/*			corner_strmm_pp_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==2)
			{
/*			corner_strmm_pp_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==3)
			{
/*			corner_strmm_pp_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		i+=8;
		}
	if(i<m-8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
//			kernel_strmm_pp_nt_12x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	kernel_strmm_pp_nt_12x4_lib4(%d, &pA[%d], &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc);
			}
		if(n-j==1)
			{
/*			corner_strmm_pp_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		else if(n-j==2)
			{
/*			corner_strmm_pp_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		else if(n-j==3)
			{
/*			corner_strmm_pp_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);*/
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
//			corner_strmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+(i+8)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+(i+8)*sdc], bs);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+8)*sda, j*bs+j*sdb, j*bs+(i+8)*sdc);
			}
		i+=12;
		}
#endif
#if defined(TARGET_CORTEX_A9) || defined(TARGET_CORTEX_A7) || defined(TARGET_CORTEX_A15) || defined(TARGET_X64_SSE3)
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
fprintf(f, "	kernel_strmm_pp_nt_8x4_lib4(%d, &pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
			}
		if(n-j==1)
			{
//fprintf(f, "	corner_strmm_pp_nt_8x1_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==2)
			{
//fprintf(f, "	corner_strmm_pp_nt_8x2_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		else if(n-j==3)
			{
//fprintf(f, "	corner_strmm_pp_nt_8x3_lib4(&pA[%d], &pA[%d], &pB[%d], &pC[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+i*sdc, j*bs+(i+4)*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+(i+4)*sda, j*bs+j*sdb, j*bs+(i+4)*sdc);
			}
		}
#endif
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
fprintf(f, "	kernel_strmm_pp_nt_4x4_lib4(%d, &pA[%d], &pB[%d], &pC[%d], 4);\n", n-j, j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
			}
		if(n-j==1)
			{
fprintf(f, "	corner_strmm_pp_nt_4x1_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
			}
		else if(n-j==2)
			{
fprintf(f, "	corner_strmm_pp_nt_4x2_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
			}
		else if(n-j==3)
			{
fprintf(f, "	corner_strmm_pp_nt_4x3_lib4(&pA[%d], &pB[%d], &pC[%d]);\n", j*bs+i*sda, j*bs+j*sdb, j*bs+i*sdc);
			}
		}

	}



void ssyrk_spotrf_code_generator(FILE *f, int m, int k, int n)
//void dsyrk_dpotrf_pp_lib(int m, int k, int n, double *pA, int sda, double *pC, int sdc, double *diag)
	{
	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
	const int sda = CNL;
	const int sdc = CNZ;

	int i, j;
	
/*	int n = m;*/
	
//	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
#if defined(TARGET_CORTEX_A15)
		if(i<m-8)
			{
//			kernel_spotrf_pp_nt_12x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[(i+8)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pC[j*bs+(i+8)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], &pA[(k0+k+j)*bs+(i+8)*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_12x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, (i+8)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, (k0+k+j)*bs+(i+8)*sda, bs);
			i += 12;
//			diag[j+0] = fact[0];
//			diag[j+1] = fact[2];
//			diag[j+2] = fact[5];
//			diag[j+3] = fact[9];
fprintf(f, "	diag[%d] = fact[0];\n", j+0);
fprintf(f, "	diag[%d] = fact[2];\n", j+1);
fprintf(f, "	diag[%d] = fact[5];\n", j+2);
fprintf(f, "	diag[%d] = fact[9];\n", j+3);
			for(; i<m-16; i+=12)
				{
//				kernel_strsm_pp_nt_12x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[(i+8)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pC[j*bs+(i+8)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], &pA[(k0+k+j)*bs+(i+8)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_12x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, (i+8)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, (k0+k+j)*bs+(i+8)*sda, bs);
				}
			if(m-i>12)
				{
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				i+=8;
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				i+=8;
				}
			if(i<m-8)
				{
//				kernel_strsm_pp_nt_12x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[(i+8)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pC[j*bs+(i+8)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], &pA[(k0+k+j)*bs+(i+8)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_12x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, (i+8)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, (k0+k+j)*bs+(i+8)*sda, bs);
				i+=12;
				}
			for(; i<m-4; i+=8)
				{
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				}
			for(; i<m; i+=4)
				{
//				kernel_strsm_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_4x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
			}
		else if(i<m-4)
#else
		if(i<m-4)
#endif
			{
#if defined(TARGET_CORTEX_A9) || defined(TARGET_CORTEX_A7) || defined(TARGET_CORTEX_A15) || defined(TARGET_X64_SSE3)
//			kernel_spotrf_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
			i += 8;
#else
//			kernel_spotrf_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_4x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
			i += 4;
#endif
//			diag[j+0] = fact[0];
//			diag[j+1] = fact[2];
//			diag[j+2] = fact[5];
//			diag[j+3] = fact[9];
fprintf(f, "	diag[%d] = fact[0];\n", j+0);
fprintf(f, "	diag[%d] = fact[2];\n", j+1);
fprintf(f, "	diag[%d] = fact[5];\n", j+2);
fprintf(f, "	diag[%d] = fact[9];\n", j+3);
#if defined(TARGET_CORTEX_A15)
			for(; i<m-16; i+=12)
				{
//				kernel_strsm_pp_nt_12x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[(i+8)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pC[j*bs+(i+8)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], &pA[(k0+k+j)*bs+(i+8)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_12x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, (i+8)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, (k0+k+j)*bs+(i+8)*sda, bs);
				}
			if(m-i>12)
				{
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				i+=8;
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				i+=8;
				}
			if(i<m-8)
				{
//				kernel_strsm_pp_nt_12x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[(i+8)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pC[j*bs+(i+8)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], &pA[(k0+k+j)*bs+(i+8)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_12x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, (i+8)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, j*bs+(i+8)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, (k0+k+j)*bs+(i+8)*sda, bs);
				i+=12;
				}
#endif
#if defined(TARGET_CORTEX_A9) || defined(TARGET_CORTEX_A7) || defined(TARGET_CORTEX_A15) || defined(TARGET_X64_SSE3)
			for(; i<m-4; i+=8)
				{
//				kernel_strsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x4_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				}
#endif
#if defined(TARGET_CORTEX_A9) || defined(TARGET_CORTEX_A7) || defined(TARGET_CORTEX_A15) || defined(TARGET_X64_SSE3)
			for(; i<m; i+=4)
				{
//				kernel_strsm_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_4x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
#else
			for(; i<m-2; i+=4)
				{
//				kernel_strsm_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_4x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
			for(; i<m; i+=2)
				{
//				kernel_strsm_pp_nt_2x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_2x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
#endif
			}
		else //if(i<m-2)
			{
//			kernel_spotrf_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_4x4_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
//			diag[j+0] = fact[0];
//			diag[j+1] = fact[2];
//			diag[j+2] = fact[5];
//			diag[j+3] = fact[9];
fprintf(f, "	diag[%d] = fact[0];\n", j+0);
fprintf(f, "	diag[%d] = fact[2];\n", j+1);
fprintf(f, "	diag[%d] = fact[5];\n", j+2);
fprintf(f, "	diag[%d] = fact[9];\n", j+3);
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
//			kernel_spotrf_pp_nt_4x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_4x2_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
//			diag[j+0] = fact[0];
//			diag[j+1] = fact[2];
fprintf(f, "	diag[%d] = fact[0];\n", j+0);
fprintf(f, "	diag[%d] = fact[2];\n", j+1);
			i += 4;
#if defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
			for(; i<m-4; i+=8)
				{
//				kernel_strsm_pp_nt_8x2_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_8x2_lib4(%d, %d, &pA[%d], &pA[%d], &pA[%d], &pC[%d], &pC[%d], &pA[%d], &pA[%d], %d, fact);\n", k, j, i*sda, (i+4)*sda, j*sda, j*bs+i*sdc, j*bs+(i+4)*sdc, (k0+k+j)*bs+i*sda, (k0+k+j)*bs+(i+4)*sda, bs);
				}
#endif
			for(; i<m-2; i+=4)
				{
//				kernel_strsm_pp_nt_4x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_4x2_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
			for(; i<m; i+=2)
				{
//				kernel_strsm_pp_nt_2x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_strsm_pp_nt_2x2_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
				}
			}
		else //if(i<m)
			{
//			kernel_spotrf_pp_nt_2x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
fprintf(f, "	kernel_spotrf_pp_nt_2x2_lib4(%d, %d, &pA[%d], &pA[%d], &pC[%d], &pA[%d], %d, fact);\n", k, j, i*sda, j*sda, j*bs+i*sdc, (k0+k+j)*bs+i*sda, bs);
//			diag[j+0] = fact[0];
//			diag[j+1] = fact[2];
fprintf(f, "	diag[%d] = fact[0];\n", j+0);
fprintf(f, "	diag[%d] = fact[2];\n", j+1);
			}
		}

	}




void sgemv_n_code_generator(FILE *f, int m, int n, int alg)
	{
	
	const int bs = 4;
	
	int j;
	
	const int sda = CNX;
	
	int idxA = 0;
	int idxy = 0;

	j=0;
	for(; j<m-4; j+=8)
		{
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], &x[0], &y[%d], %d);\n", n, idxA, idxA+sda*bs, idxy, alg);
		//kernel_dgemv_n_8_lib4(n, pA, pA+sda*bs, x, y, alg);
		idxA += 2*sda*bs;
		idxy += 2*bs;
		}
	for(; j<m; j+=4)
		{
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[0], &y[%d], %d);\n", n, idxA, idxy, alg);
		//kernel_dgemv_n_4_lib4(n, pA, x, y, alg);
		idxA += sda*bs;
		idxy += bs;
		}

	}



void sgemv_t_code_generator(FILE *f, int m, int n, int offset, int sda, int alg)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
	for(; j<n-7; j+=8)
		{
fprintf(f, "	kernel_sgemv_t_8_lib4(%d, %d, &pA[%d], %d, &x[0], &y[%d], %d);\n", m, mna, j*bs, sda, j, alg);
		//kernel_dgemv_t_8_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n-3; j+=4)
		{
fprintf(f, "	kernel_sgemv_t_4_lib4(%d, %d, &pA[%d], %d, &x[0], &y[%d], %d);\n", m, mna, j*bs, sda, j, alg);
		//kernel_dgemv_t_4_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}
	if(j<n)
		{
		if(j==n-1)
			{
//			kernel_sgemv_t_1_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
fprintf(f, "	kernel_sgemv_t_1_lib4(%d, %d, &pA[%d], %d, &x[0], &y[%d], %d);\n", m, mna, j*bs, sda, j, alg);
			}
		if(j==n-2)
			{
//			kernel_sgemv_t_2_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
fprintf(f, "	kernel_sgemv_t_2_lib4(%d, %d, &pA[%d], %d, &x[0], &y[%d], %d);\n", m, mna, j*bs, sda, j, alg);
			}
		if(j==n-3)
			{
//			kernel_sgemv_t_3_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
fprintf(f, "	kernel_sgemv_t_3_lib4(%d, %d, &pA[%d], %d, &x[0], &y[%d], %d);\n", m, mna, j*bs, sda, j, alg);
			}
		}

	}



void strmv_u_n_code_generator(FILE *f, int m, int alg)
	{

	const int bs = 4;
	
	int j;
	
	const int sda = CNL;
	
	int idxA = 0;
	int idxx = 0;
	int idxy = 0;

	j=0;
	for(; j<m-7; j+=8)
		{
fprintf(f, "	kernel_strmv_u_n_8_lib4(%d, &pA[%d], &pA[%d], &x[%d], &y[%d], %d);\n", m-j, idxA, idxA+sda*bs, idxx, idxy, alg);
		//kernel_dtrmv_u_n_8_lib4(m-j, pA, pA+sda*bs, x, y, alg);
		idxA += 2*sda*bs + 2*4*bs;
		idxx += 2*bs;
		idxy += 2*bs;
		}
	for(; j<m-3; j+=4)
		{
fprintf(f, "	kernel_strmv_u_n_4_lib4(%d, &pA[%d], &x[%d], &y[%d], %d);\n", m-j, idxA, idxx, idxy, alg);
		//kernel_dtrmv_u_n_4_lib4(m-j, pA, x, y, alg);
		idxA += sda*bs + 4*bs;
		idxx += bs;
		idxy += bs;
		}
	if(j<m)
		{
		if(j==m-1)
			{
			if(alg==0)
fprintf(f, "	y[0] = pA[0]*x[0];\n");
			else if(alg==1)
fprintf(f, "	y[0] += pA[0]*x[0];\n");
			else
fprintf(f, "	y[0] -= pA[0]*x[0];\n");
			}
		else if(j==m-2)
			{
			if(alg==0)
				{
fprintf(f, "	y[0] = pA[0]*x[0] + pA[4]*x[1];\n");
fprintf(f, "	y[1] = pA[5]*x[1];\n");
				}
			else if(alg==1)
				{
fprintf(f, "	y[0] += pA[0]*x[0] + pA[4]*x[1];\n");
fprintf(f, "	y[1] += pA[5]*x[1];\n");
				}
			else
				{
fprintf(f, "	y[0] -= pA[0]*x[0] + pA[4]*x[1];\n");
fprintf(f, "	y[1] -= pA[5]*x[1];\n");
				}
			}
		else if(j==m-3)
			{
			if(alg==0)
				{
fprintf(f, "	y[0] = pA[0]*x[0] + pA[4]*x[1] + pA[8]*x[2];\n");
fprintf(f, "	y[1] = pA[5]*x[1] + pA[9]*x[2];\n");
fprintf(f, "	y[2] = pA[10]*x[2];\n");
				}
			else if(alg==1)
				{
fprintf(f, "	y[0] += pA[0]*x[0] + pA[4]*x[1] + pA[8]*x[2];\n");
fprintf(f, "	y[1] += pA[5]*x[1] + pA[9]*x[2];\n");
fprintf(f, "	y[2] += pA[10]*x[2];\n");
				}
			else
				{
fprintf(f, "	y[0] -= pA[0]*x[0] + pA[4]*x[1] + pA[8]*x[2];\n");
fprintf(f, "	y[1] -= pA[5]*x[1] + pA[9]*x[2];\n");
fprintf(f, "	y[2] -= pA[10]*x[2];\n");
				}
			}
		}

	}



void strmv_u_t_code_generator(FILE *f, int m, int alg)
	{

	const int bs = 4;
	
	int j;
	
	const int sda = CNL;
	
	int idxA = 0;
	int idxy = 0;
	
	j=0;
	for(; j<m-7; j+=8)
		{
fprintf(f, "	kernel_strmv_u_t_8_lib4(%d, &pA[%d], %d, &x[0], &y[%d], %d);\n", j, idxA, sda, idxy, alg);
		//kernel_dtrmv_u_t_8_lib4(j, pA, sda, x, y, alg);
		idxA += 2*4*bs;
		idxy += 2*bs;
		}
	for(; j<m-3; j+=4)
		{
fprintf(f, "	kernel_strmv_u_t_4_lib4(%d, &pA[%d], %d, &x[0], &y[%d], %d);\n", j, idxA, sda, idxy, alg);
		//kernel_dtrmv_u_t_4_lib4(j, pA, sda, x, y, alg);
		idxA += 4*bs;
		idxy += bs;
		}
	if(j<m)
		{
		if(j==m-1)
			{
//			kernel_strmv_u_t_1_lib4(j, pA, sda, x, y, alg);
fprintf(f, "	kernel_strmv_u_t_1_lib4(%d, &pA[%d], %d, &x[0], &y[%d], %d);\n", j, idxA, sda, idxy, alg);
			}
		else if(j==m-2)
			{
//			kernel_strmv_u_t_2_lib4(j, pA, sda, x, y, alg);
fprintf(f, "	kernel_strmv_u_t_2_lib4(%d, &pA[%d], %d, &x[0], &y[%d], %d);\n", j, idxA, sda, idxy, alg);
			}
		else if(j==m-3)
			{
//			kernel_strmv_u_t_3_lib4(j, pA, sda, x, y, alg);
fprintf(f, "	kernel_strmv_u_t_3_lib4(%d, &pA[%d], %d, &x[0], &y[%d], %d);\n", j, idxA, sda, idxy, alg);
			}
		}

	}



void ssymv_code_generator(FILE *f, int m, int offset, int alg)
	{
	
	const int bs = 4;
	
	const int sda = CNZ;

	int mna = (bs-offset%bs)%bs;
	int ma = m - mna;

	int j, jj, j0;
	
	int idxA = 0;
	int idxx = 0;
	int idxy = 0;

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
		alg = 1;
fprintf(f, "	\n");
		}
	
	if(mna>0)
		{
		j=0;
		for(; j<mna; j++)
			{
/*			kernel_dsymv_1_lib4(m-j, mna-j, pA+j+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);*/
fprintf(f, "	kernel_ssymv_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], &x[%d], &y[%d], 1, %d);\n", m-j, mna-j, idxA+j+j*bs, sda, idxx+j, idxy+j, idxx+j, idxy+j, alg);
			}
		idxA += j + (sda-1)*bs + j*bs;
		idxx += j;
		idxy += j;
		}
	j=0;
	for(; j<ma-3; j+=4)
		{
/*		kernel_dsymv_4_lib4(ma-j, 0, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);*/
fprintf(f, "	kernel_ssymv_4_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], &x[%d], &y[%d], 1, %d);\n", ma-j, 0, idxA+j*sda+j*bs, sda, idxx+j, idxy+j, idxx+j, idxy+j, alg);
		}
	j0 = j;
	for(; j<ma-1; j+=2)
		{
/*		kernel_dsymv_2_lib4(ma-j, ma-j, pA+(j-j0)+j0*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);*/
fprintf(f, "	kernel_ssymv_2_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], &x[%d], &y[%d], 1, %d);\n", ma-j, ma-j, idxA+(j-j0)+j0*sda+j*bs, sda, idxx+j, idxy+j, idxx+j, idxy+j, alg);
		}
	for(; j<ma; j++)
		{
/*		kernel_dsymv_1_lib4(ma-j, ma-j, pA+(j-j0)+j0*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);*/
fprintf(f, "	kernel_ssymv_1_lib4(%d, %d, &pA[%d], %d, &x[%d], &y[%d], &x[%d], &y[%d], 1, %d);\n", ma-j, ma-j, idxA+(j-j0)+j0*sda+j*bs, sda, idxx+j, idxy+j, idxx+j, idxy+j, alg);
		}

	}



void smvmv_code_generator(FILE *f, int m, int n, int offset, int alg)
	{
	
	const int bs = 4;

	const int sda = CNX;

	int mna = (bs-offset%bs)%bs;

	int j, jj;
	
	int idxA = 0;
	int idxx = 0;
	int idxy = 0;

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
		alg = 1;
fprintf(f, "	\n");
		}
	
	j=0;
	for(; j<n-3; j+=4)
		{
/*		kernel_dsymv_4_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);*/
fprintf(f, "	kernel_ssymv_4_lib4(%d, %d, &pA[%d], %d, &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, %d);\n", m, mna, idxA+j*bs, sda, idxx+j, idxy, idxx, idxy+j, alg);
		}
	for(; j<n-1; j+=2)
		{
/*		kernel_dsymv_2_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);*/
fprintf(f, "	kernel_ssymv_2_lib4(%d, %d, &pA[%d], %d, &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, %d);\n", m, mna, idxA+j*bs, sda, idxx+j, idxy, idxx, idxy+j, alg);
		}
	for(; j<n; j++)
		{
/*		kernel_dsymv_1_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);*/
fprintf(f, "	kernel_ssymv_1_lib4(%d, %d, &pA[%d], %d, &x_n[%d], &y_n[%d], &x_t[%d], &y_t[%d], 0, %d);\n", m, mna, idxA+j*bs, sda, idxx+j, idxy, idxx, idxy+j, alg);
		}

	}



// the diagonal is inverted !!!
void strsv_sgemv_n_code_generator(FILE *f, int m, int n)
	{
	
	const int bs = 4;
	
	int j;
	
	const int sda = CNL;
	
/*	double *y;*/

	// blocks of 4 (pA is supposed to be properly aligned)
/*	y  = x;*/
	
	int idxA = 0;
	int idxy = 0;

	j = 0;
	for(; j<m-7; j+=8)
		{
fprintf(f, "	kernel_strsv_n_8_lib4(%d, &pA[%d], &pA[%d], &x[0], &x[%d]);\n", j, idxA, idxA+bs*sda, idxy);
		//kernel_dtrsv_n_8_lib4(j, pA, pA+bs*sda, x, y); // j+8 !!!

		idxA += 2*bs*sda;
		idxy += 2*bs;

		}
	if(j<m-3)
		{
fprintf(f, "	kernel_strsv_n_4_lib4(%d, 4, &pA[%d], &x[0], &x[%d]);\n", j, idxA, idxy);
		//kernel_dtrsv_n_4_lib4(j, 4, pA, x, y); // j+4 !!!

		idxA += bs*sda;
		idxy += bs;
		j+=4;

		}
	if(j<m) // !!! suppose that there are enough nx after !!! => x padded with enough zeros at the end !!!
		{
fprintf(f, "	kernel_strsv_n_4_lib4(%d, %d, &pA[%d], &x[0], &x[%d]);\n", j, m-j, idxA, idxy);
		//kernel_dtrsv_n_4_lib4(j, m-j, pA, x, y); // j+4 !!!

		idxA += bs*sda;
		idxy += bs;
		j+=4;

		}
	for(; j<n-4; j+=8)
		{
fprintf(f, "	kernel_sgemv_n_8_lib4(%d, &pA[%d], &pA[%d], &x[0], &x[%d], -1);\n", m, idxA, idxA+bs*sda, idxy);
		//kernel_dgemv_n_8_lib4(m, pA, pA+sda*bs, x, y, -1);

		idxA += 2*sda*bs;
		idxy += 2*bs;

		}
	for(; j<n; j+=4)
		{
fprintf(f, "	kernel_sgemv_n_4_lib4(%d, &pA[%d], &x[0], &x[%d], -1);\n", m, idxA, idxy);
		//kernel_dgemv_n_4_lib4(m, pA, x, y, -1);

		idxA += sda*bs;
		idxy += bs;

		}

	}



// the diagonal is inverted !!!
void strsv_sgemv_t_code_generator(FILE *f, int m, int n)
	{
	
	const int bs = 4;
	
	const int sda = CNL;

	int j;
	
/*	double *y;*/
	
	j=0;
	if(n%4==1)
		{
fprintf(f, "	kernel_strsv_t_1_lib4(%d, &pA[%d], %d, &x[%d]);\n", m-n+j+1, (n/bs)*bs*sda+(n-j-1)*bs, sda, n-j-1);
		//kernel_dtrsv_t_1_lib4(m-n+j+1, pA+(n/bs)*bs*sda+(n-1)*bs, sda, x+n-j-1);
		j++;
		}
	else if(n%4==2)
		{
fprintf(f, "	kernel_strsv_t_2_lib4(%d, &pA[%d], %d, &x[%d]);\n", m-n+j+2, (n/bs)*bs*sda+(n-j-2)*bs, sda, n-j-2);
		//kernel_dtrsv_t_2_lib4(n-m+j+2, pA+(m/bs)*bs*sda+(m-j-2)*bs, sda, x+m-j-2);
		j+=2;
		}
	else if(n%4==3)
		{
fprintf(f, "	kernel_strsv_t_3_lib4(%d, &pA[%d], %d, &x[%d]);\n", m-n+j+3, (n/bs)*bs*sda+(n-j-3)*bs, sda, n-j-3);
		//kernel_dtrsv_t_3_lib4(n-m+j+3, pA+(m/bs)*bs*sda+(m-j-3)*bs, sda, x+m-j-3);
		j+=3;
		}
	for(; j<n-3; j+=4)
		{
fprintf(f, "	kernel_strsv_t_4_lib4(%d, &pA[%d], %d, &x[%d]);\n", m-n+j+4, ((n-j-4)/bs)*bs*sda+(n-j-4)*bs, sda, n-j-4);
		//kernel_dtrsv_t_4_lib4(n-m+j+4, pA+((m-j-4)/bs)*bs*sda+(m-j-4)*bs, sda, x+m-j-4);
		}

	}


// transpose & align lower triangular matrix
void strtr_l_code_generator(FILE *f, int m, int offset)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	const int sda = CNL;
	const int sdc = CNL;
	
	int idxA = 0;
	int idxC = 0;

	int j;
	
	j=0;
	for(; j<m-3; j+=4)
		{
fprintf(f, "	kernel_stran_pp_4_lib4(%d, %d, &pA[%d], %d, &pC[%d]);\n", m-j, mna, idxA, sda, idxC);
		idxA += bs*(sda+bs);
		idxC += bs*(sdc+bs);
		}
	if(j==m)
		{
		return;
		}
	else if(m-j==1)
		{
fprintf(f, "	pC[0] = pA[0];\n");
		}
	else if(m-j==2)
		{
fprintf(f, "	corner_stran_pp_2_lib4(%d, &pA[%d], %d, &pC[%d]);\n", mna, idxA, sda, idxC);
		}
	else // if(m-j==3)
		{
fprintf(f, "	corner_stran_pp_3_lib4(%d, &pA[%d], %d, &pC[%d]);\n", mna, idxA, sda, idxC);
		}
	
	}

