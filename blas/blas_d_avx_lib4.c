#include <math.h>
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include "../include/kernel_d_avx.h"



/* preforms                                          */
/* C  = A * B' (alg== 0)                             */
/* C += A * B' (alg== 1)                             */
/* C -= A * B' (alg==-1)                             */
/* where A, B and C are packed with block size 4     */
void dgemm_ppp_nt_lib(int m, int n, int k, double *pA, int sda, double *pB, int sdb, double *pC, int sdc, int alg)
	{

	const int bs = 4;

	int i, ii, j, jj;
	
	i = 0;
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_8x4_avx_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[0+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, alg);
			}
		jj = 0;
		for(; jj<n-j-1; jj+=2)
			{
			kernel_dgemm_pp_nt_8x2_avx_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+4)*sdc], bs, alg);
			}
		for(; jj<n-j; jj++)
			{
			kernel_dgemm_pp_nt_8x1_avx_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &pB[jj+j*sdb], &pC[0+(j+jj)*bs+i*sdc], &pC[0+(j+jj)*bs+(i+4)*sdc], bs, alg);
			}
		}
	ii = 0;
	for(; ii<m-i; ii+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_avx_lib4(k, &pA[ii+i*sda], &pB[0+j*sdb], &pC[ii+(j+0)*bs+i*sdc], bs, alg);
			}
		jj = 0;
		for(; jj<n-j-1; jj+=2)
			{
			kernel_dgemm_pp_nt_4x2_avx_lib4(k, &pA[ii+i*sda], &pB[jj+j*sdb], &pC[ii+(j+jj)*bs+i*sdc], bs, alg);
			}
		for(; jj<n-j; jj++)
			{
			kernel_dgemm_pp_nt_4x1_avx_lib4(k, &pA[ii+i*sda], &pB[jj+j*sdb], &pC[ii+(j+jj)*bs+i*sdc], bs, alg);
			}
		}
	}



/* preforms                                          */
/* C  = A * B                                        */
/* where A, B and C are packed with block size 4,    */
/* and B is lower triangular                         */
void dtrmm_ppp_lib(int m, int n, int offset, double *pA, int sda, double *pB, int sdb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	if(offset%bs!=0)
		pB = pB+bs*sdb+bs*bs; // shift to the next block
	
	i = 0;
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_pp_nt_8x4_avx_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, 0);
			}
		}
	for(; i<m; i+=4)
		{
		j = 0;
		for(; j<n; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_avx_lib4(n-j-0, &pA[0+(j+0)*bs+i*sda], &pB[0+(j+0)*bs+j*sdb], &pC[0+(j+0)*bs+i*sdc], bs, 0);
			}
		}

	// add to the last row
	for(j=0; j<n; j++)
		{
		pC[(m-1)%bs+j*bs+((m-1)/bs)*bs*sdc] += pB[j%bs+n*bs+(j/bs)*bs*sdb];
		}

	}



/* preforms                                          */
/* C  = A * A'                                       */
/* where A, C are packed with block size 4           */
void dsyrk_ppp_lib(int n, int m, double *pA, int sda, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	i = 0;
	for(; i<n-4; i+=8)
		{
		j = 0;
		for(; j<i+4; j+=4)
			{
			kernel_dgemm_pp_nt_8x4_avx_lib4(m, &pA[0+i*sda], &pA[0+(i+4)*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, 1);
/*			kernel_dgemm_pp_nt_8x4_avx_lib8(m, &pA[0+i*sda], &pA[4+j*sda], &pC[0+(j+4)*bs+i*sdc], bs, 1);*/
			}
/*		for(; j<i+4; j+=8)*/
		if(j<i+8)
			{
/*			kernel_dgemm_pp_nt_8x4_avx_lib8(m, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);*/
			kernel_dgemm_pp_nt_4x4_avx_lib4(m, &pA[0+(i+4)*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+(i+4)*sdc], bs, 1);
			}
		}
	for(; i<n; i+=4)
		{
		j = 0;
		for(; j<i+4; j+=4)
			{
			kernel_dgemm_pp_nt_4x4_avx_lib4(m, &pA[0+i*sda], &pA[0+j*sda], &pC[0+(j+0)*bs+i*sdc], bs, 1);
			}
		}

	}

