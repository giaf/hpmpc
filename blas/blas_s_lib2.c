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

#include "../include/kernel_s_lib2.h"



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 2     */
void sgemm_ppp_nt_lib(int m, int n, int k, float *pA, int sda, float *pB, int sdb, float *pC, int sdc, int alg)
	{

	const int bs = 2;

	int i, j, jj;
	
	i = 0;
	for(; i<m; i+=2)
		{
		j = 0;
		for(; j<n-1; j+=2)
			{
			kernel_sgemm_pp_nt_2x2_lib2(k, &pA[0+i*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, alg);
			}
		jj = 0;
		for(; jj<n-j; jj++)
			{
			kernel_sgemm_pp_nt_2x1_lib2(k, &pA[0+i*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], bs, alg);
			}
		}

	}



/* preforms                                          */
/* C  = A * B'                                       */
/* where A, B and C are packed with block size 2,    */
/* and B is upper triangular                         */
void strmm_ppp_lib(int m, int n, int offset, float *pA, int sda, float *pB, int sdb, float *pC, int sdc)
	{
	
	const int bs = 2;
	
	int i, j;
	
	if(offset%bs!=0)
		pB = pB+bs*sdb+bs*bs; // shift to the next block
	
	i = 0;
	for(; i<m; i+=2)
		{
		j = 0;
		for(; j<n-1; j+=2)
			{
			kernel_sgemm_pp_nt_2x2_lib2(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, 0);
			}
		if(n-j==1)
			{
			corner_strmm_pp_nt_2x1_lib2(&pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs);
			}
		}

	// add to the last row
	for(j=0; j<n; j++)
		{
		pC[(m-1)%bs+j*bs+((m-1)/bs)*bs*sdc] += pB[j%bs+n*bs+(j/bs)*bs*sdb];
		}

	}



/* computes the mxn panel of                         */
/* C  = A * A'                                       */
/* where A, C are packed with block size 2           */
void ssyrk_ppp_lib(int m, int n, int k, float *pA, int sda, float *pC, int sdc)
	{
	
	const int bs = 2;
	
	int i, j, j_end;
	
	i = 0;
	for(; i<m; i+=2)
		{
		j = 0;
		j_end = i+2;
		if(n-1<j_end)
			j_end = n-1;
		for(; j<j_end; j+=2)
			{
			kernel_sgemm_pp_nt_2x2_lib2(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		if(n-j==1)
			{
			kernel_sgemm_pp_nt_2x1_lib2(k, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		}

	}



/* computes the lower triangular Cholesky factor of pC, */
/* and copies its transposed in pL                      */
void spotrf_p_scopy_p_t_lib(int n, int nna, float *pC, int sdc, float *pL, int sdl, int *info)
	{

	const int bs = 2;
	
	int
		i, j, jj;

	j = 0;
	if(j<nna-1)
		{
		kernel_spotrf_strsv_2x2_lib2(n-j-2, &pC[0+j*bs+j*sdc], sdc, info);
		if(*info!=0) return;
		j += 2;
		for(; j<nna-1; j+=2)
			{
			i = j;
			for(; i<n; i+=2)
				{
				kernel_sgemm_pp_nt_2x2_lib2(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
				}
			kernel_spotrf_strsv_2x2_lib2(n-j-2, &pC[0+j*bs+j*sdc], sdc, info);
			if(*info!=0) return;
			}
		}
	int j0 = j;
	if(j==0) // assume that n>0
		{
		kernel_spotrf_strsv_scopy_2x2_lib2(n-j-2, &pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl, info);
		if(*info!=0) return;
		j += 2;
		}
	for(; j<n-1; j+=2)
		{
		i = j;
		for(; i<n; i+=2)
			{
			kernel_sgemm_pp_nt_2x2_lib2(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			}
		kernel_spotrf_strsv_scopy_2x2_lib2(n-j-2, &pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl, info);
		if(*info!=0) return;
		}
	if(n-j==1)
		{
		i = j;
		kernel_sgemm_pp_nt_2x1_lib2(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
		corner_spotrf_strsv_scopy_1x1_lib2(&pC[0+j*bs+j*sdc], sdc, (bs-nna%bs)%bs, &pL[0+(j-j0)*bs+((j-j0)/bs)*bs*sdc], sdl, info);
		if(*info!=0) return;
		}

	}



/* computes an mxn band of the lower triangular Cholesky factor of pC, supposed to be aligned */
void spotrf_p_lib(int m, int n, float *pC, int sdc, int *info)
	{

	const int bs = 2;
	
	int i, j;

	j = 0;
	for(; j<n-1; j+=2)
		{
		i = j;
		for(; i<m; i+=2)
			{
			kernel_sgemm_pp_nt_2x2_lib2(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			}
		kernel_spotrf_strsv_2x2_lib2(m-j-2, &pC[0+j*bs+j*sdc], sdc, info);
		if(*info!=0) return;
		}
	if(n-j==1)
		{
		i = j;
		for(; i<m; i+=2)
			{
			kernel_sgemm_pp_nt_2x1_lib2(j, &pC[0+i*sdc], &pC[0+j*sdc], &pC[0+j*bs+i*sdc], bs, -1);
			}
		kernel_spotrf_strsv_1x1_lib2(m-j-1, &pC[0+j*bs+j*sdc], sdc, info);
		if(*info!=0) return;
		}

	}



void sgemv_p_n_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg)
	{
	
	const int bs = 2;
	
	int j;

	int nna = (bs-offset%bs)%bs;

	j=0;
	if(nna==1)
		{
		kernel_sgemv_n_1_lib2(m, pA, x, y, alg);
		pA += 1;
		y  += 1;
		j++;
		pA += (sda-1)*bs;
		}
/*	for(; j<n-3; j+=4)*/
/*		{*/
/*		kernel_sgemv_n_4_lib2(m, pA, x, y, alg);*/
/*		pA += 2*sda*bs;*/
/*		y  += 2*bs;*/
/*		}*/
	for(; j<n-1; j+=2)
		{
		kernel_sgemv_n_2_lib2(m, pA, x, y, alg);
		pA += sda*bs;
		y  += bs;
		}
	for(; j<n; j++)
		{
		kernel_sgemv_n_1_lib2(m, pA, x, y, alg);
		pA += 1;
		y  += 1;
		}

	}



void sgemv_p_t_lib(int n, int m, int offset, float *pA, int sda, float *x, float *y, int alg)
	{
	
	const int bs = 2;
	
	int nna = (bs-offset%bs)%bs;
	
	int j;
	
	j=0;
/*	for(; j<m-3; j+=4)*/
/*		{*/
/*		kernel_sgemv_t_4_lib2(n, nna, pA+j*bs, sda, x, y+j, alg);*/
/*		}*/
	for(; j<m-1; j+=2)
		{
		kernel_sgemv_t_2_lib2(n, nna, pA+j*bs, sda, x, y+j, alg);
		}
	for(; j<m; j++)
		{
		kernel_sgemv_t_1_lib2(n, nna, pA+j*bs, sda, x, y+j, alg);
		}

	}



void strmv_p_n_lib(int m, int offset, float *pA, int sda, float *x, float *y, int alg)
	{
	
	const int bs = 2;
	
	int mna = (bs-offset%bs)%bs;
	
	int j;
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_n_1_lib2(j+1, pA, x, y, alg);
			pA += 1;
			y  += 1;
			pA += (sda-1)*bs;
			j  += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_n_2_lib2(j+1, pA, x, y, alg);
			y[1] += pA[(j+1)*bs+1] * x[j+1];
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_n_1_lib2(j+1, pA, x, y, alg);
			pA += 1;
			y  += 1;
			}
		}
	else
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_n_1_lib2(j+1, pA, x, y, -1);
			pA += 1;
			y  += 1;
			pA += (sda-1)*bs;
			j  += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_n_2_lib2(j+1, pA, x, y, -1);
			y[1] -= pA[(j+1)*bs+1] * x[j+1];
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_n_1_lib2(j+1, pA, x, y, -1);
			pA += 1;
			y  += 1;
			}
		}

	}



// !!! x and y can not be the same vector !!!
void strmv_p_t_lib(int m, int offset, float *pA, int sda, float *x, float *y, int alg)
	{
	
	const int bs = 2;
	
	int mna = (bs-offset%bs)%bs;
	int mmax = m;
	
	int j;
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_t_1_lib2(mmax, 1, pA, sda, x, y, alg);
			pA += 1 + sda*bs;
			x  += 1;
			y  += 1;
			mmax -= 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_t_2_lib2(mmax-1, 1, pA+1, sda, x+1, y, alg);
			y[0] += pA[0+bs*0] * x[0];
			pA += bs*sda + bs*bs;
			x  += bs;
			y  += bs;
			mmax -= bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_t_1_lib2(mmax, mmax, pA, sda, x, y, alg);
			pA += 1 + bs;
			x  += 1;
			y  += 1;
			mmax -= 1;
			}
		}
	else
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_t_1_lib2(mmax, 1, pA, sda, x, y, -1);
			pA += 1 + sda*bs;
			x  += 1;
			y  += 1;
			mmax -= 1;
			j += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_t_2_lib2(mmax-1, 1, pA+1, sda, x+1, y, -1);
			y[0] -= pA[0+bs*0] * x[0];
			pA += bs*sda + bs*bs;
			x  += bs;
			y  += bs;
			mmax -= bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_t_1_lib2(mmax, mmax, pA, sda, x, y, -1);
			pA += 1 + bs;
			x  += 1;
			y  += 1;
			mmax -= 1;
			}
		}

	}



void ssymv_p_lib(int m, int offset, float *pA, int sda, float *x, float *y, int alg)
	{
	
	const int bs = 2;

	int mna = (bs-offset%bs)%bs;

	int j;
	
	float *ptrA, *ptrx;
	
	if(alg==0 || alg==1)
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_n_1_lib2(j, pA, x, y, alg);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] += ptrA[0+bs*0]*ptrx[0];
			kernel_sgemv_t_1_lib2(m-j-1, 0, ptrA+1, sda, ptrx+1, y, 1); // !!! 1
			pA += 1 + (sda-1)*bs;
			y  += 1;
			j  += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_n_2_lib2(j, pA, x, y, alg);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] += ptrA[0+bs*0]*ptrx[0] + ptrA[1+bs*0]*ptrx[1];
			y[1] += ptrA[1+bs*0]*ptrx[0] + ptrA[1+bs*1]*ptrx[1];
			kernel_sgemv_t_2_lib2(m-j-2, 0, ptrA+sda*bs, sda, ptrx+2, y, 1); // !!! 1
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_n_1_lib2(j, pA, x, y, alg);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] += ptrA[0+bs*0]*ptrx[0];
			kernel_sgemv_t_1_lib2(m-j-1, 0, ptrA+1, sda, ptrx+1, y, 1); // !!! 1
			pA += 1;
			y  += 1;
			}
		}
	else // alg==-1
		{
		j=0;
		if(mna==1)
			{
			kernel_sgemv_n_1_lib2(j, pA, x, y, -1);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] -= ptrA[0+bs*0]*ptrx[0];
			kernel_sgemv_t_1_lib2(m-j-1, 0, ptrA+1, sda, ptrx+1, y, -1);
			pA += 1 + (sda-1)*bs;
			y  += 1;
			j  += 1;
			}
		for(; j<m-1; j+=2)
			{
			kernel_sgemv_n_2_lib2(j, pA, x, y, -1);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] -= ptrA[0+bs*0]*ptrx[0] + ptrA[1+bs*0]*ptrx[1];
			y[1] -= ptrA[1+bs*0]*ptrx[0] + ptrA[1+bs*1]*ptrx[1];
			kernel_sgemv_t_2_lib2(m-j-2, 0, ptrA+sda*bs, sda, ptrx+2, y, -1);
			pA += sda*bs;
			y  += bs;
			}
		for(; j<m; j++)
			{
			kernel_sgemv_n_1_lib2(j, pA, x, y, -1);
			ptrA = pA + j*bs;
			ptrx =  x + j;
			y[0] -= ptrA[0+bs*0]*ptrx[0];
			kernel_sgemv_t_1_lib2(m-j-1, 0, ptrA+1, sda, ptrx+1, y, -1);
			pA += 1;
			y  += 1;
			}
		}

	}



void strsv_p_n_lib(int n, float *pA, int sda, float *x)
	{
	
	const int bs = 2;
	
	int j;
	
	float *ptrA, *ptrAd, *ptrx;

	// blocks of 2 (pA is supposed to be properly aligned)
	ptrA  = pA;
	ptrAd = pA;
	ptrx  = x;

	j = 0;
/*	for(; j<n-3; j+=4)*/
/*		{*/
/*		// correct*/
/*		kernel_sgemv_n_4_lib2(j, ptrA, x, ptrx, -1);*/

/*		// solve*/
/*		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];*/
/*		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];*/
/*		ptrx[2] = (ptrx[2] - ptrx[0] * ptrAd[2+bs*0] - ptrx[1] * ptrAd[2+bs*1]) / ptrAd[2+bs*2];*/
/*		ptrx[3] = (ptrx[3] - ptrx[0] * ptrAd[3+bs*0] - ptrx[1] * ptrAd[3+bs*1] - ptrx[2] * ptrAd[3+bs*2]) / ptrAd[3+bs*3];*/

/*		ptrA  += 2*bs*sda;*/
/*		ptrAd += 2*bs*(sda+bs);*/
/*		ptrx  += 2*bs;*/

/*		}*/
	for(; j<n-1; j+=2)
		{
		// correct
		kernel_sgemv_n_2_lib2(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		ptrx[1] = (ptrx[1] - ptrx[0] * ptrAd[1+bs*0]) / ptrAd[1+bs*1];
		
		ptrA  += bs*sda;
		ptrAd += bs*(sda+bs);
		ptrx  += bs;
		}
	for(; j<n; j++)
		{
		// correct
		kernel_sgemv_n_1_lib2(j, ptrA, x, ptrx, -1);

		// solve
		ptrx[0] = (ptrx[0]) / ptrAd[0+bs*0];
		
		ptrA  += 1;
		ptrAd += bs+1;
		ptrx  += 1;
		}

	}



void strsv_p_t_lib(int n, float *pA, int sda, float *x)
	{
	
	const int bs = 2;
	
	int i, j;
	
	int rn = n%bs;
	int qn = n/bs;
	int ri, qi;
	
	float *ptrA, *ptrx;
	
	// clean up stuff at the end
	j = 0;
	ptrA = pA + qn*bs*(sda+bs);
	ptrx = x + qn*bs;

	if(rn==1)
		{
		i = rn-1-j;
		kernel_sgemv_t_1_lib2(j, j, &ptrA[i+1+bs*(i+0)], sda, &ptrx[i+1], &ptrx[i], -1);
		ptrx[i+0] = (ptrx[i+0]) / ptrA[i+0+bs*(i+0)];
/*		j++;*/
		}

	// blocks of 2
	for(; j<qn; j++)
		{
		
		// first 2 rows
		ptrA = pA + (qn-j-1)*bs*(sda+bs) ;
		ptrx = x  + (qn-j-1)*bs          ;
		
		kernel_sgemv_t_2_lib2(rn+j*bs, 0, ptrA+bs*sda, sda, ptrx+2, ptrx, -1);
		ptrx[1] = (ptrx[1]) / ptrA[1+bs*1];
		ptrx[0] = (ptrx[0] - ptrA[1+bs*0]*ptrx[1]) / ptrA[0+bs*0];

		}

	}
