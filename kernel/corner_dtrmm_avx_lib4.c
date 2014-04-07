#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX



/*inline void corner_dtrmm_pp_nt_8x3_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_8x3_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32, a_40_50_60_70, a_41_51_61_71, a_42_52_62_72,
		b_00, b_10, b_20, b_11, b_21, b_22,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_40_50_60_70, c_41_51_61_71, c_42_52_62_72;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A0[0+4*1] );
	a_41_51_61_71 = _mm256_load_pd( &A1[0+4*1] );
	a_02_12_22_32 = _mm256_load_pd( &A0[0+4*2] );
	a_42_52_62_72 = _mm256_load_pd( &A1[0+4*2] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	b_20 = _mm256_broadcast_sd( &B[0+4*2] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_41_51_61_71, b_10 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_20 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_42_52_62_72, b_20 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );

	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
	
	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );
	b_21 = _mm256_broadcast_sd( &B[1+4*2] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	c_41_51_61_71 = _mm256_mul_pd( a_41_51_61_71, b_11 );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_21 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	ab_temp = _mm256_mul_pd( a_42_52_62_72, b_21 );
	c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_temp );
	
	_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
	
	// third column 
	b_22 = _mm256_broadcast_sd( &B[2+4*2] );

	c_02_12_22_32 = _mm256_mul_pd( a_02_12_22_32, b_22 );
	c_42_52_62_72 = _mm256_mul_pd( a_42_52_62_72, b_22 );

	_mm256_store_pd( &C0[0+ldc*2], c_02_12_22_32 );
	_mm256_store_pd( &C1[0+ldc*2], c_42_52_62_72 );

	}
	


/*inline void corner_dtrmm_pp_nt_8x2_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_8x2_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_40_50_60_70, a_41_51_61_71,
		b_00, b_10, b_11,
		c_00_10_20_30, c_01_11_21_31, c_40_50_60_70, c_41_51_61_71;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A0[0+4*1] );
	a_41_51_61_71 = _mm256_load_pd( &A1[0+4*1] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	ab_temp = _mm256_mul_pd( a_41_51_61_71, b_10 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_temp );
	
	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );

	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	c_41_51_61_71 = _mm256_mul_pd( a_41_51_61_71, b_11 );
	
	_mm256_store_pd( &C0[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &C1[0+ldc*1], c_41_51_61_71 );
	
	}



/*inline void corner_dtrmm_pp_nt_8x1_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_8x1_avx_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30, a_40_50_60_70,
		b_00,
		c_00_10_20_30, c_40_50_60_70;
	
	a_00_10_20_30 = _mm256_load_pd( &A0[0+4*0] );
	a_40_50_60_70 = _mm256_load_pd( &A1[0+4*0] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );
	c_40_50_60_70 = _mm256_mul_pd( a_40_50_60_70, b_00 );

	_mm256_store_pd( &C0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &C1[0+ldc*0], c_40_50_60_70 );
	
	}


/*inline void corner_dtrmm_pp_nt_4x3_avx_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_pp_nt_4x3_avx_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31, a_02_12_22_32,
		b_00, b_10, b_20, b_11, b_21, b_22,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A[0+4*1] );
	a_02_12_22_32 = _mm256_load_pd( &A[0+4*2] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	b_20 = _mm256_broadcast_sd( &B[0+4*2] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_20 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );

	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );
	
	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );
	b_21 = _mm256_broadcast_sd( &B[1+4*2] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );

	ab_temp = _mm256_mul_pd( a_02_12_22_32, b_21 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	
	_mm256_store_pd( &C[0+ldc*1], c_01_11_21_31 );
	
	// third column 
	b_22 = _mm256_broadcast_sd( &B[2+4*2] );

	c_02_12_22_32 = _mm256_mul_pd( a_02_12_22_32, b_22 );

	_mm256_store_pd( &C[0+ldc*2], c_02_12_22_32 );

	}
	


/*inline void corner_dtrmm_pp_nt_4x2_avx_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_pp_nt_4x2_avx_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30, a_01_11_21_31,
		b_00, b_10, b_11,
		c_00_10_20_30, c_01_11_21_31;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	a_01_11_21_31 = _mm256_load_pd( &A[0+4*1] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	b_10 = _mm256_broadcast_sd( &B[0+4*1] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	ab_temp = _mm256_mul_pd( a_01_11_21_31, b_10 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	
	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );

	// second column 
	b_11 = _mm256_broadcast_sd( &B[1+4*1] );

	c_01_11_21_31 = _mm256_mul_pd( a_01_11_21_31, b_11 );
	
	_mm256_store_pd( &C[0+ldc*1], c_01_11_21_31 );
	
	}



/*inline void corner_dtrmm_pp_nt_4x1_avx_lib4(double *A, double *B, double *C, int ldc)*/
void corner_dtrmm_pp_nt_4x1_avx_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m256d
		ab_temp,
		a_00_10_20_30,
		b_00,
		c_00_10_20_30;
	
	a_00_10_20_30 = _mm256_load_pd( &A[0+4*0] );
	
	// first column 
	b_00 = _mm256_broadcast_sd( &B[0+4*0] );
	
	c_00_10_20_30 = _mm256_mul_pd( a_00_10_20_30, b_00 );

	_mm256_store_pd( &C[0+ldc*0], c_00_10_20_30 );
	
	}
