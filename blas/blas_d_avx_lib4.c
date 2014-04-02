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



/* test routine */
void dgemm_pup_nn_lib(int m, int n, int k, double *pA, int sda, double *B, int ldb, double *pC, int sdc, int alg)
	{

	const int bs = 4;

	int i, ii, j, jj;
	
	i = 0;
	for(; i<m-4; i+=8)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pu_nn_8x4_avx_lib4(k, &pA[0+i*sda], &pA[0+(i+4)*sda], &B[(0+j)*ldb], ldb, &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, alg, 0);
			}
		}
	ii = 0;
	for(; ii<m-i; ii+=4)
		{
		j = 0;
		for(; j<n-3; j+=4)
			{
			kernel_dgemm_pu_nn_4x4_avx_lib4(k, &pA[ii+i*sda], &B[(0+j)*ldb], ldb, &pC[ii+(j+0)*bs+i*sdc], bs, alg, 0);
			}
		}
	}



/* preforms                                          */
/* C  = A * B                                        */
/* where A, C are packed with block size 4, and B is */
/* unpacked lower triangular                         */
void dtrmm_pup_nn_lib(int n, int m, double *pA, int sda, double *B, int ldb, double *pC, int sdc)
	{
	
	const int bs = 4;
	
	int i, j;
	
	__m256d
		a_0123, a_4567,
		b_0, b_1, b_2, b_3,
		ab_0, ab_1,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73;

	i = 0;
	for(; i<n-4; i+=8)
		{
		j = 0;
		for(; j<m-3; j+=4)
			{
			kernel_dgemm_pu_nn_8x4_avx_lib4(m-j-0, &pA[0+(j+0)*bs+i*sda], &pA[0+(j+0)*bs+(i+4)*sda], &B[j+j*ldb], ldb, &pC[0+(j+0)*bs+i*sdc], &pC[0+(j+0)*bs+(i+4)*sdc], bs, 0, 1);
			}

		// corner cases (no kernel)
		if(m-j!=0)
			{
			if(m-j==1)
				{
				c_00_10_20_30 = _mm256_setzero_pd();
				c_40_50_60_70 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );

				_mm256_store_pd( &pC[0+(j+0)*bs+(i+0)*sdc], c_00_10_20_30 );
				_mm256_store_pd( &pC[0+(j+0)*bs+(i+4)*sdc], c_40_50_60_70 );
				}
			else if(m-j==2)
				{
				c_00_10_20_30 = _mm256_setzero_pd();
				c_40_50_60_70 = _mm256_setzero_pd();
				c_01_11_21_31 = _mm256_setzero_pd();
				c_41_51_61_71 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );
	
				// k = 1
				a_0123        = _mm256_load_pd( &pA[0+(j+1)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+1)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+1+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+1+(j+1)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_1 );
				c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_1 );

				_mm256_store_pd( &pC[0+(j+0)*bs+(i+0)*sdc], c_00_10_20_30 );
				_mm256_store_pd( &pC[0+(j+0)*bs+(i+4)*sdc], c_40_50_60_70 );
				_mm256_store_pd( &pC[0+(j+1)*bs+(i+0)*sdc], c_01_11_21_31 );
				_mm256_store_pd( &pC[0+(j+1)*bs+(i+4)*sdc], c_41_51_61_71 );
				}
			else // m-j==3
				{
				c_00_10_20_30 = _mm256_setzero_pd();
				c_40_50_60_70 = _mm256_setzero_pd();
				c_01_11_21_31 = _mm256_setzero_pd();
				c_41_51_61_71 = _mm256_setzero_pd();
				c_02_12_22_32 = _mm256_setzero_pd();
				c_42_52_62_72 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+0)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );
	
				// k = 1
				a_0123        = _mm256_load_pd( &pA[0+(j+1)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+1)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+1+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+1+(j+1)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_1 );
				c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_1 );

				// k = 2	
				a_0123        = _mm256_load_pd( &pA[0+(j+2)*bs+(i+0)*sda] );
				a_4567        = _mm256_load_pd( &pA[0+(j+2)*bs+(i+4)*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+2+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+2+(j+1)*ldb] );
				b_2           = _mm256_broadcast_sd( &B[j+2+(j+2)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_0 );
				c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_1 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_1 );
				c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_1 );
				ab_0          = _mm256_mul_pd( a_0123, b_2 );
				c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_0 );
				ab_1          = _mm256_mul_pd( a_4567, b_2 );
				c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_1 );

				_mm256_store_pd( &pC[0+(j+0)*bs+(i+0)*sdc], c_00_10_20_30 );
				_mm256_store_pd( &pC[0+(j+0)*bs+(i+4)*sdc], c_40_50_60_70 );
				_mm256_store_pd( &pC[0+(j+1)*bs+(i+0)*sdc], c_01_11_21_31 );
				_mm256_store_pd( &pC[0+(j+1)*bs+(i+4)*sdc], c_41_51_61_71 );
				_mm256_store_pd( &pC[0+(j+2)*bs+(i+0)*sdc], c_02_12_22_32 );
				_mm256_store_pd( &pC[0+(j+2)*bs+(i+4)*sdc], c_42_52_62_72 );
				}
			}

		}
	for(; i<n; i+=4)
		{
		j = 0;
		for(; j<m-3; j+=4)
			{
			kernel_dgemm_pu_nn_4x4_avx_lib4(m-j-0, &pA[0+(j+0)*bs+i*sda], &B[j+j*ldb], ldb, &pC[0+(j+0)*bs+i*sdc], bs, 0, 1);
			}

		// corner cases (no kernel)
		if(m-j!=0)
			{
			if(m-j==1)
				{
				c_00_10_20_30 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );

				_mm256_store_pd( &pC[0+(j+0)*bs+i*sdc], c_00_10_20_30 );
				}
			else if(m-j==2)
				{
				c_00_10_20_30 = _mm256_setzero_pd();
				c_01_11_21_31 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
	
				// k = 1
				a_0123        = _mm256_load_pd( &pA[0+(j+1)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+1+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+1+(j+1)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );

				_mm256_store_pd( &pC[0+(j+0)*bs+i*sdc], c_00_10_20_30 );
				_mm256_store_pd( &pC[0+(j+1)*bs+i*sdc], c_01_11_21_31 );
				}
			else // m-j==3
				{
				c_00_10_20_30 = _mm256_setzero_pd();
				c_01_11_21_31 = _mm256_setzero_pd();
				c_02_12_22_32 = _mm256_setzero_pd();

				// k = 0
				a_0123        = _mm256_load_pd( &pA[0+(j+0)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+0+(j+0)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
	
				// k = 1
				a_0123        = _mm256_load_pd( &pA[0+(j+1)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+1+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+1+(j+1)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );

				// k = 2	
				a_0123        = _mm256_load_pd( &pA[0+(j+2)*bs+i*sda] );
				b_0           = _mm256_broadcast_sd( &B[j+2+(j+0)*ldb] );
				b_1           = _mm256_broadcast_sd( &B[j+2+(j+1)*ldb] );
				b_2           = _mm256_broadcast_sd( &B[j+2+(j+2)*ldb] );
				ab_0          = _mm256_mul_pd( a_0123, b_0 );
				c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_0 );
				ab_0          = _mm256_mul_pd( a_0123, b_1 );
				c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_0 );
				ab_0          = _mm256_mul_pd( a_0123, b_2 );
				c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_0 );

				_mm256_store_pd( &pC[0+(j+0)*bs+i*sdc], c_00_10_20_30 );
				_mm256_store_pd( &pC[0+(j+1)*bs+i*sdc], c_01_11_21_31 );
				_mm256_store_pd( &pC[0+(j+2)*bs+i*sdc], c_02_12_22_32 );
				}
			}
	

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

