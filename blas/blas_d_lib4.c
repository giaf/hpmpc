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

#include "../include/kernel_d_lib4.h"
#include "../include/block_size.h"



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg)
	{

	const int bs = 4;

	int i, j, jj;
	
	i = 0;
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_8x4_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, alg);
			}
		jj = 0;
/*		for(; jj<n-j-1; jj+=2)*/
		for(; jj<n-j; jj+=2)
			{
			kernel_dgemm_pp_nt_8x2_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+4)*sdc], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+4)*sdc], bs, alg);
			}
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_dgemm_pp_nt_8x1_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+4)*sdc], bs, alg);*/
/*			}*/
		}
	for(; i<m-2; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_lib4(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
		jj = 0;
/*		for(; jj<n-j-1; jj+=2)*/
		for(; jj<n-j; jj+=2)
			{
			kernel_dgemm_pp_nt_4x2_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_dgemm_pp_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
		}
	for(; i<m; i+=2)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_2x4_lib4(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
		jj = 0;
/*		for(; jj<n-j-1; jj+=2)*/
		for(; jj<n-j; jj+=2)
			{
			kernel_dgemm_pp_nt_2x2_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
/*		for(; jj<n-j; jj++)*/
/*			{*/
/*			kernel_dgemm_pp_nt_4x1_lib4(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);*/
/*			}*/
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 4,    */
/* and B is upper triangular                         */
void dtrmm_ppp_lib(int m, int n, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_pp_nt_8x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
			}
		if(n-j==1)
			{
			corner_dtrmm_pp_nt_8x1_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
			}
		else if(n-j==2)
			{
			corner_dtrmm_pp_nt_8x2_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
			}
		else if(n-j==3)
			{
			corner_dtrmm_pp_nt_8x3_lib4(&pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs);
			}
		}
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dtrmm_pp_nt_4x4_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		if(n-j==1)
			{
			corner_dtrmm_pp_nt_4x1_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		else if(n-j==2)
			{
			corner_dtrmm_pp_nt_4x2_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		else if(n-j==3)
			{
			corner_dtrmm_pp_nt_4x3_lib4(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		}

	}



void dsyrk_dpotrf_pp_lib(int m, int k, int n, double *pA, int sda, double *pC, int sdc, double *diag)
	{
	const int bs = 4;
	const int d_ncl = D_NCL;
	const int k0 = (d_ncl-k%d_ncl)%d_ncl;
	
/*printf("\nk0 = %d\n", k0);*/

	int i, j;
	
/*	int n = m;*/
	
	double fact[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	j = 0;
	for(; j<n-2; j+=4)
		{
		i = j;
		if(i<m-4)
			{
/*printf("\n8x4\n");*/
			kernel_dpotrf_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			i += 8;
			for(; i<m-4; i+=8)
				{
				kernel_dtrsm_pp_nt_8x4_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dtrsm_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
				}
			for(; i<m; i+=2)
				{
				kernel_dtrsm_pp_nt_2x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
				}
			}
		else //if(i<m-2)
			{
/*printf("\n4x4\n");*/
			kernel_dpotrf_pp_nt_4x4_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			diag[j+2] = fact[5];
			diag[j+3] = fact[9];
			}
		}
	for(; j<n; j+=2)
		{
		i = j;
		if(i<m-2)
			{
/*printf("\n4x2\n");*/
			kernel_dpotrf_pp_nt_4x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			i += 4;
			for(; i<m-4; i+=8)
				{
				kernel_dtrsm_pp_nt_8x2_lib4(k, j, &pA[i*sda], &pA[(i+4)*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pC[j*bs+(i+4)*sdc], &pA[(k0+k+j)*bs+i*sda], &pA[(k0+k+j)*bs+(i+4)*sda], bs, fact);
				}
			for(; i<m-2; i+=4)
				{
				kernel_dtrsm_pp_nt_4x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
				}
			for(; i<m; i+=2)
				{
				kernel_dtrsm_pp_nt_2x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
				}
			}
		else //if(i<m)
			{
/*printf("\n2x2\n");*/
			kernel_dpotrf_pp_nt_2x2_lib4(k, j, &pA[i*sda], &pA[j*sda], &pC[j*bs+i*sdc], &pA[(k0+k+j)*bs+i*sda], bs, fact);
			diag[j+0] = fact[0];
			diag[j+1] = fact[2];
			}
		}
/*printf("\nend\n");*/

	}



/*void dgemv_p_n_lib(int n, int m, int offset, double *pA, int sda, double *x, double *y, int alg)*/
void dgemv_p_n_lib(int n, int m, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int j;

/*	int nna = (bs-offset%bs)%bs;*/

/*	if(nna>n) // it is always nna < bs , thus n<bs !!!!!*/
/*		{*/
/*		j = 0;*/
/*		if(nna%2==1)*/
/*			{*/
/*			kernel_dgemv_n_1_lib4(m, pA, x, y, alg);*/
/*			pA += 1;*/
/*			y  += 1;*/
/*			j  += 1;*/
/*			}*/
/*		for(; j<n-1; j+=2)*/
/*			{*/
/*			kernel_dgemv_n_2_lib4(m, pA, x, y, alg);*/
/*			pA += 2;*/
/*			y  += 2;*/
/*			}*/
/*		for(; j<n; j++)*/
/*			{*/
/*			kernel_dgemv_n_1_lib4(m, pA, x, y, alg);*/
/*			pA += 1;*/
/*			y  += 1;*/
/*			}*/
/*		return;*/
/*		}*/
	j=0;
/*	if(nna>0) // it can be nna = {1, 2, 3}*/
/*		{*/
/*		if(nna%2==1)*/
/*			{*/
/*			kernel_dgemv_n_1_lib4(m, pA, x, y, alg);*/
/*			pA += 1;*/
/*			y  += 1;*/
/*			j++;*/
/*			}*/
/*		if(nna%4>=2)*/
/*			{*/
/*			kernel_dgemv_n_2_lib4(m, pA, x, y, alg);*/
/*			pA += 2;*/
/*			y  += 2;*/
/*			j+=2;*/
/*			}*/
/*		pA += (sda-1)*bs;*/
/*		}*/
/*	for(; j<n-7; j+=8)*/
	for(; j<n-4; j+=8)
		{
		kernel_dgemv_n_8_lib4(m, pA, pA+sda*bs, x, y, alg);
		pA += 2*sda*bs;
		y  += 2*bs;
		}
/*	for(; j<n-3; j+=4)*/
	for(; j<n; j+=4)
		{
		kernel_dgemv_n_4_lib4(m, pA, x, y, alg);
		pA += sda*bs;
		y  += bs;
		}
/*	for(; j<n-1; j+=2)*/
/*		{*/
/*		kernel_dgemv_n_2_lib4(m, pA, x, y, alg);*/
/*		pA += 2;*/
/*		y  += 2;*/
/*		}*/
/*	for(; j<n; j++)*/
/*		{*/
/*		kernel_dgemv_n_1_lib4(m, pA, x, y, alg);*/
/*		pA += 1;*/
/*		y  += 1;*/
/*		}*/

	}



void dgemv_p_t_lib(int m, int n, int offset, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
	for(; j<n-7; j+=8)
		{
		kernel_dgemv_t_8_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n-3; j+=4)
		{
		kernel_dgemv_t_4_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n-1; j+=2)
		{
		kernel_dgemv_t_2_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<n; j++)
		{
		kernel_dgemv_t_1_lib4(m, mna, pA+j*bs, sda, x, y+j, alg);
		}

	}



void dtrmv_p_n_lib(int m, int offset, double *pA, int sda, double *x, double *y, int alg)
	{

	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	if(alg==0 || alg==1)
		{
		if(mna>m) // it is always mna < bs , thus m<bs !!!!!
			{
			j = 0;
			if(mna%2==1)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, alg);
				pA += 1;
				y  += 1;
				j  += 1;
				}
			for(; j<m-1; j+=2)
				{
				kernel_dgemv_n_2_lib4(j+1, pA, x, y, alg);
				y[1] += pA[(j+1)*bs+1] * x[j+1];
				pA += 2;
				y  += 2;
				}
			for(; j<m; j++)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, alg);
				pA += 1;
				y  += 1;
				}
			return;
			}
		j=0;
		if(mna>0)
			{
			for(; j<mna%2; j++)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, alg);
				pA += 1;
				y  += 1;
				}
			for(; j<mna; j+=2)
				{
				kernel_dgemv_n_2_lib4(j+1, pA, x, y, alg);
				y[1] += pA[(j+1)*bs+1] * x[j+1];
				pA += 2;
				y  += 2;
				}
			pA += (sda-1)*bs;
			}
		for(; j<m-7; j+=8)
			{
			kernel_dgemv_n_8_lib4(j+1, pA, pA+sda*bs, x, y, alg);
			y[1] += pA[1+bs*(j+1)] * x[j+1];
			y[2] += pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
			y[3] += pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
			pA += sda*bs;
			y  += bs;

			kernel_dgemv_n_4_lib4(4, pA+bs*(j+1), x+j+1, y, 1); // 1 !!!
			y[1] += pA[1+bs*(j+5)] * x[j+5];
			y[2] += pA[2+bs*(j+5)] * x[j+5] + pA[2+bs*(j+6)] * x[j+6];
			y[3] += pA[3+bs*(j+5)] * x[j+5] + pA[3+bs*(j+6)] * x[j+6] + pA[3+bs*(j+7)] * x[j+7];

			pA += sda*bs;
			y  += bs;
			}
		for(; j<m-3; j+=4)
			{
			kernel_dgemv_n_4_lib4(j+1, pA, x, y, alg);
			y[1] += pA[1+bs*(j+1)] * x[j+1];
			y[2] += pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
			y[3] += pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m-1; j+=2)
			{
			kernel_dgemv_n_2_lib4(j+1, pA, x, y, alg);
			y[1] += pA[(j+1)*bs+1] * x[j+1];
			pA += 2;
			y  += 2;
			}
		for(; j<m; j++)
			{
			kernel_dgemv_n_1_lib4(j+1, pA, x, y, alg);
			pA += 1;
			y  += 1;
			}
		}
	else
		{
		if(mna>m) // it is always mna < bs , thus m<bs !!!!!
			{
			j = 0;
			if(mna%2==1)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, -1);
				pA += 1;
				y  += 1;
				j  += 1;
				}
			for(; j<m-1; j+=2)
				{
				kernel_dgemv_n_2_lib4(j+1, pA, x, y, -1);
				y[1] -= pA[(j+1)*bs+1] * x[j+1];
				pA += 2;
				y  += 2;
				}
			for(; j<m; j++)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, -1);
				pA += 1;
				y  += 1;
				}
			return;
			}
		j=0;
		if(mna>0)
			{
			for(; j<mna%2; j++)
				{
				kernel_dgemv_n_1_lib4(j+1, pA, x, y, -1);
				pA += 1;
				y  += 1;
				}
			for(; j<mna; j+=2)
				{
				kernel_dgemv_n_2_lib4(j+1, pA, x, y, -1);
				y[1] -= pA[(j+1)*bs+1] * x[j+1];
				pA += 2;
				y  += 2;
				}
			pA += (sda-1)*bs;
			}
		for(; j<m-7; j+=8)
			{
			kernel_dgemv_n_8_lib4(j+1, pA, pA+sda*bs, x, y, -1);
			y[1] -= pA[1+bs*(j+1)] * x[j+1];
			y[2] -= pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
			y[3] -= pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
			pA += sda*bs;
			y  += bs;

			kernel_dgemv_n_4_lib4(4, pA+bs*(j+1), x+j+1, y, -1);
			y[1] -= pA[1+bs*(j+5)] * x[j+5];
			y[2] -= pA[2+bs*(j+5)] * x[j+5] + pA[2+bs*(j+6)] * x[j+6];
			y[3] -= pA[3+bs*(j+5)] * x[j+5] + pA[3+bs*(j+6)] * x[j+6] + pA[3+bs*(j+7)] * x[j+7];

			pA += sda*bs;
			y  += bs;
			}
		for(; j<m-3; j+=4)
			{
			kernel_dgemv_n_4_lib4(j+1, pA, x, y, -1);
			y[1] -= pA[1+bs*(j+1)] * x[j+1];
			y[2] -= pA[2+bs*(j+1)] * x[j+1] + pA[2+bs*(j+2)] * x[j+2];
			y[3] -= pA[3+bs*(j+1)] * x[j+1] + pA[3+bs*(j+2)] * x[j+2] + pA[3+bs*(j+3)] * x[j+3];
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m-1; j+=2)
			{
			kernel_dgemv_n_2_lib4(j+1, pA, x, y, -1);
			y[1] -= pA[(j+1)*bs+1] * x[j+1];
			pA += 2;
			y  += 2;
			}
		for(; j<m; j++)
			{
			kernel_dgemv_n_1_lib4(j+1, pA, x, y, -1);
			pA += 1;
			y  += 1;
			}
		}

	}



void dtrmv_p_u_n_lib(int m, double *pA, int sda, double *x, double *y, int alg)
	{

	const int bs = 4;
	
	int j;
	
	double *ptrA;
	
	j=0;
	for(; j<m-7; j+=8)
		{
/*		y[0] += pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2] + pA[0+bs*3]*x[3];*/
/*		y[1] +=                   pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2] + pA[1+bs*3]*x[3];*/
/*		y[2] +=                                     pA[2+bs*2]*x[2] + pA[2+bs*3]*x[3];*/
/*		y[3] +=                                                       pA[3+bs*3]*x[3];*/
/*		kernel_dgemv_n_4_lib4(4, pA+4*bs, x+4, y, 1);*/
/*		ptrA = pA + sda*bs + 4*bs;*/
/*		y[4] += ptrA[0+bs*0]*x[4] + ptrA[0+bs*1]*x[5] + ptrA[0+bs*2]*x[6] + ptrA[0+bs*3]*x[7];*/
/*		y[5] +=                     ptrA[1+bs*1]*x[5] + ptrA[1+bs*2]*x[6] + ptrA[1+bs*3]*x[7];*/
/*		y[6] +=                                         ptrA[2+bs*2]*x[6] + ptrA[2+bs*3]*x[7];*/
/*		y[7] +=                                                             ptrA[3+bs*3]*x[7];*/
/*		kernel_dgemv_n_8_lib4(m-j-8, pA+8*bs, pA+8*bs+sda*bs, x+8, y, 1);*/
		kernel_dtrmv_u_n_8_lib4(m-j, pA, pA+sda*bs, x, y, alg);
		pA += 2*sda*bs + 2*4*bs;
		x  += 2*bs;
		y  += 2*bs;
		}
	for(; j<m-3; j+=4)
		{
/*		y[0] += pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1] + pA[0+bs*2]*x[2] + pA[0+bs*3]*x[3];*/
/*		y[1] +=                   pA[1+bs*1]*x[1] + pA[1+bs*2]*x[2] + pA[1+bs*3]*x[3];*/
/*		y[2] +=                                     pA[2+bs*2]*x[2] + pA[2+bs*3]*x[3];*/
/*		y[3] +=                                                       pA[3+bs*3]*x[3];*/
/*		kernel_dgemv_n_4_lib4(m-j-4, pA+4*bs, x+4, y, 1);*/
		kernel_dtrmv_u_n_4_lib4(m-j, pA, x, y, alg);
		pA += sda*bs + 4*bs;
		x  += bs;
		y  += bs;
		}
	for(; j<m-1; j+=2)
		{
		kernel_dtrmv_u_n_2_lib4(m-j, pA, x, y, alg);
/*		kernel_dgemv_n_2_lib4(m-j-2, pA+2*bs, x+2, y, alg);*/
/*		y[0] += pA[0+bs*0]*x[0] + pA[0+bs*1]*x[1];*/
/*		y[1] +=                   pA[1+bs*1]*x[1];*/
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



void dtrmv_p_u_t_lib(int m, double *pA, int sda, double *x, double *y, int alg)
	{

	const int bs = 4;
	
	int j;
	
	double *ptrA;
	
	j=0;
/*	for(; j<m-7; j+=8)*/ // TODO kernel 8 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/*		{*/
/*		kernel_dtrmv_u_n_8_lib4(m-j, pA, pA+sda*bs, x, y, alg);*/
/*		pA += 2*sda*bs + 2*4*bs;*/
/*		x  += 2*bs;*/
/*		y  += 2*bs;*/
/*		}*/
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



// !!! x and y can not be the same vector !!!
void dtrmv_p_t_lib(int m, int offset, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	int mmax = m;
	
	int j;
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna>0)
			{
			for(; j<mna; j++)
				{
				kernel_dgemv_t_1_lib4(mmax-j, mna-j, pA+j*bs+j, sda, x+j, y+j, alg);
				}
			pA += j + (sda-1)*bs + j*bs;
			x  += j;
			y  += j;
			mmax -= j;
			}
		for(; j<m-7; j+=8)
			{
			pA += bs*sda;
			
			kernel_dgemv_t_8_lib4(mmax-7, 1, pA+3, sda, x+7, y, alg);
			y[4] += pA[0+bs*4] * x[4] + pA[1+bs*4] * x[5] + pA[2+bs*4] * x[6];
			y[5] += pA[1+bs*5] * x[5] + pA[2+bs*5] * x[6];
			y[6] += pA[2+bs*6] * x[6];

			pA -= bs*sda;

			kernel_dgemv_t_4_lib4(4, 1, pA+3, sda, x+3, y, 1); // !!! 1
			y[0] += pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
			y[1] += pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
			y[2] += pA[2+bs*2] * x[2];

			pA += 2*bs*sda + 2*bs*bs;
			x  += 2*bs;
			y  += 2*bs;
			mmax -= 2*bs;
			}
		for(; j<m-3; j+=4)
			{
			kernel_dgemv_t_4_lib4(mmax-3, 1, pA+3, sda, x+3, y, alg);
			y[0] += pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
			y[1] += pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
			y[2] += pA[2+bs*2] * x[2];
			pA += bs*sda + bs*bs;
			x  += bs;
			y  += bs;
			mmax -= bs;
			}
		for(; j<m-1; j+=2)
			{
			kernel_dgemv_t_2_lib4(mmax-1, mmax-1, pA+1, sda, x+1, y, alg);
			y[0] += pA[0+bs*0] * x[0];
			pA += 2 + 2*bs;
			x  += 2;
			y  += 2;
			mmax -= 2;
			}
		for(; j<m; j++)
			{
			kernel_dgemv_t_1_lib4(mmax, mmax, pA, sda, x, y, alg);
			pA += 1 + bs;
			x  += 1;
			y  += 1;
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
				kernel_dgemv_t_1_lib4(mmax-j, mna-j, pA+j*bs+j, sda, x+j, y+j, -1);
				}
			pA += j + (sda-1)*bs + j*bs;
			x  += j;
			y  += j;
			mmax -= j;
			}
		for(; j<m-7; j+=8)
			{
			kernel_dgemv_t_4_lib4(4, 1, pA+3, sda, x+3, y, -1);
			y[0] -= pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
			y[1] -= pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
			y[2] -= pA[2+bs*2] * x[2];

			pA += bs*sda;
			
			kernel_dgemv_t_8_lib4(mmax-7, 1, pA+3, sda, x+7, y, -1);
			y[4] -= pA[0+bs*4] * x[4] + pA[1+bs*4] * x[5] + pA[2+bs*4] * x[6];
			y[5] -= pA[1+bs*5] * x[5] + pA[2+bs*5] * x[6];
			y[6] -= pA[2+bs*6] * x[6];

			pA += bs*sda + 2*bs*bs;
			x  += 2*bs;
			y  += 2*bs;
			mmax -= 2*bs;
			}
		for(; j<m-3; j+=4)
			{
			kernel_dgemv_t_4_lib4(mmax-3, 1, pA+3, sda, x+3, y, -1);
			y[0] -= pA[0+bs*0] * x[0] + pA[1+bs*0] * x[1] + pA[2+bs*0] * x[2];
			y[1] -= pA[1+bs*1] * x[1] + pA[2+bs*1] * x[2];
			y[2] -= pA[2+bs*2] * x[2];
			pA += bs*sda + bs*bs;
			x  += bs;
			y  += bs;
			mmax -= bs;
			}
		for(; j<m-1; j+=2)
			{
			kernel_dgemv_t_2_lib4(mmax-1, mmax-1, pA+1, sda, x+1, y, -1);
			y[0] -= pA[0+bs*0] * x[0];
			pA += 2 + 2*bs;
			x  += 2;
			y  += 2;
			mmax -= 2;
			}
		for(; j<m; j++)
			{
			kernel_dgemv_t_1_lib4(mmax, mmax, pA, sda, x, y, -1);
			pA += 1 + bs;
			x  += 1;
			y  += 1;
			mmax -= 1;
			}
		}

	}



// it moves vertically across block
void dsymv_p_lib(int m, int offset, double *pA, int sda, double *x, double *y, int alg)
	{
	
	const int bs = 4;
	
	int mna = (bs-offset%bs)%bs;
	int ma = m - mna;

	int j, j0;
	
	if(alg==0)
		{
		for(j=0; j<m; j++)
			y[j] = 0.0;
		alg = 1;
		}
	
	if(mna>0)
		{
		j=0;
		for(; j<mna; j++)
			{
			kernel_dsymv_1_lib4(m-j, mna-j, pA+j+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
			}
		pA += j + (sda-1)*bs + j*bs;
		x += j;
		y += j;
		}
	j=0;
	for(; j<ma-3; j+=4)
		{
		kernel_dsymv_4_lib4(ma-j, 0, pA+j*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
		}
	j0 = j;
	for(; j<ma-1; j+=2)
		{
		kernel_dsymv_2_lib4(ma-j, ma-j, pA+(j-j0)+j0*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
		}
	for(; j<ma; j++)
		{
		kernel_dsymv_1_lib4(ma-j, ma-j, pA+(j-j0)+j0*sda+j*bs, sda, x+j, y+j, x+j, y+j, 1, alg);
		}

	}



// it moves vertically across block
void dmvmv_p_lib(int m, int n, int offset, double *pA, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int alg)
	{
	
	const int bs = 4;

	int mna = (bs-offset%bs)%bs;

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
		kernel_dsymv_4_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}
	for(; j<n-1; j+=2)
		{
		kernel_dsymv_2_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}
	for(; j<n; j++)
		{
		kernel_dsymv_1_lib4(m, mna, pA+j*bs, sda, x_n+j, y_n, x_t, y_t+j, 0, alg);
		}

	}



// the diagonal is inverted !!!
void dtrsv_p_n_lib(int n, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int j;
	
	double *ptrA, *ptrAd, *ptrx;

	// blocks of 4 (pA is supposed to be properly aligned)
	ptrA  = pA;
	ptrAd = pA;
	ptrx  = x;

	j = 0;
	for(; j<n-7; j+=8)
		{

		kernel_dtrsv_n_8_lib4(j, ptrA, ptrA+bs*sda, x, ptrx); // j+8 !!!

		ptrA  += 2*bs*sda;
		ptrAd += 2*bs*(sda+bs);
		ptrx  += 2*bs;

		}

	// clean up stuff at the end
	for(; j<n-3; j+=4)
		{

		kernel_dtrsv_n_4_lib4(j, 4, ptrA, x, ptrx); // j+4 !!!

		ptrA  += bs*sda;
		ptrAd += bs*(sda+bs);
		ptrx  += bs;

		}
	for(; j<n-1; j+=2)
		{
		// correct
		kernel_dgemv_n_2_lib4(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) * ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) * ptrAd[1+bs*1];
		
		ptrA  += 2;
		ptrAd += 2*bs+2;
		ptrx  += 2;
		}
	for(; j<n; j++)
		{
		// correct
		kernel_dgemv_n_1_lib4(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) * ptrAd[0+bs*0];
		
		ptrA  += 1;
		ptrAd += bs+1;
		ptrx  += 1;
		}

	}



// the diagonal is inverted !!!
void dtrsv_dgemv_p_n_lib(int n, int m, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int j;
	
	double *y;

	// blocks of 4 (pA is supposed to be properly aligned)
	y  = x;

	j = 0;
	for(; j<n-7; j+=8)
		{

		kernel_dtrsv_n_8_lib4(j, pA, pA+bs*sda, x, y); // j+8 !!!

		pA += 2*bs*sda;
		y  += 2*bs;

		}
	if(j<n-3)
		{

		kernel_dtrsv_n_4_lib4(j, 4, pA, x, y); // j+4 !!!

		pA += bs*sda;
		y  += bs;
		j+=4;

		}
	if(j<n) // !!! suppose that there are enough nx after !!! => x padded with enough zeros at the end !!!
		{

		kernel_dtrsv_n_4_lib4(j, n-j, pA, x, y); // j+4 !!!

		pA += bs*sda;
		y  += bs;
		j+=4;

		}
/*	for(; j<m-7; j+=8)*/
	for(; j<m-4; j+=8)
		{

		kernel_dgemv_n_8_lib4(n, pA, pA+sda*bs, x, y, -1);

		pA += 2*sda*bs;
		y  += 2*bs;

		}
/*	for(; j<m-3; j+=4)*/
	for(; j<m; j+=4)
		{

		kernel_dgemv_n_4_lib4(n, pA, x, y, -1);

		pA += sda*bs;
		y  += bs;

		}
/*	for(; j<m-1; j+=2)*/
/*		{*/

/*		kernel_dgemv_n_2_lib4(n, pA, x, y, -1);*/

/*		pA += 2;*/
/*		y  += 2;*/

/*		}*/
/*	for(; j<m; j+=1)*/
/*		{*/

/*		kernel_dgemv_n_1_lib4(n, pA, x, y, -1);*/

/*		pA += 1;*/
/*		y  += 1;*/

/*		}*/

	}



// the diagonal is inverted !!!
void dtrsv_p_t_lib(int n, double *pA, int sda, double *x)
	{
	
	const int bs = 4;
	
	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	
	double *ptrA, *ptrx;
	
	// clean up stuff at the end
	j = 0;
	ptrA = pA + qn*bs*(sda+bs);
	ptrx = x + qn*bs;

	for(; j<rn%2; j++)
		{
		i = rn-1-j;
		kernel_dgemv_t_1_lib4(j, j, &ptrA[i+1+bs*(i+0)], sda, &ptrx[i+1], &ptrx[i], -1);
		ptrx[i+0] = (ptrx[i+0]) * ptrA[i+0+bs*(i+0)];
		}
	for(; j<rn; j+=2)
		{
		i = rn-2-j;
		kernel_dgemv_t_2_lib4(j, j, &ptrA[i+2+bs*(i+0)], sda, &ptrx[i+2], &ptrx[i], -1);
		ptrx[i+1] = (ptrx[i+1]) * ptrA[(i+1)+bs*(i+1)];
		ptrx[i+0] = (ptrx[i+0] - ptrA[(i+1)+bs*(i+0)]*ptrx[i+1]) * ptrA[(i+0)+bs*(i+0)];
		}

	// blocks of 8
	j = 0;
	for(; j<qn-1; j+=2)
		{
		
		// all 4 rows
		ptrA = pA + (qn-j-2)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-2)*bs          ;

		// correct
		kernel_dgemv_t_8_lib4(rn+j*bs, 0, ptrA+2*bs*sda, sda, ptrx+8, ptrx, -1);

		// last 4 rows
		ptrA = pA + (qn-j-1)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-1)*bs          ;

		// solve
		ptrx[3] = (ptrx[3]) * ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) * ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) * ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) * ptrA[0+bs*0];

		// first 4 rows
		ptrA = pA + (qn-j-2)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-2)*bs          ;

		// correct
		kernel_dgemv_t_4_lib4(4, 0, ptrA+bs*sda, sda, ptrx+4, ptrx, -1);

		// solve
		ptrx[3] = (ptrx[3]) * ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) * ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) * ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) * ptrA[0+bs*0];

		}
	
	// blocks of 4
	for(; j<qn; j++)
		{
		
		// first 4 rows
		ptrA = pA + (qn-j-1)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-1)*bs          ;
		
		kernel_dgemv_t_4_lib4(rn+j*bs, 0, ptrA+bs*sda, sda, ptrx+4, ptrx, -1);
		ptrx[3] = (ptrx[3]) * ptrA[3+bs*3];
		ptrx[2] = (ptrx[2] - ptrA[3+bs*2]*ptrx[3]) * ptrA[2+bs*2];
		ptrx[1] = (ptrx[1] - ptrA[3+bs*1]*ptrx[3] - ptrA[2+bs*1]*ptrx[2]) * ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[3+bs*0]*ptrx[3] - ptrA[2+bs*0]*ptrx[2] - ptrA[1+bs*0]*ptrx[1]) * ptrA[0+bs*0];

		}

	}
