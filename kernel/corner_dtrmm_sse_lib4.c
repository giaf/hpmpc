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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
/*#include <immintrin.h>  // AVX*/



/*inline void corner_dtrmm_pp_nt_4x3_sse_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_4x3_sse_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m128d
		ab_temp,
		a_00_10, a_01_11, a_02_12, a_20_30, a_21_31, a_22_32,
		b_00, b_10, b_20, b_11, b_21, b_22;
	
	a_00_10 = _mm_load_pd( &A[0+4*0] );
	a_20_30 = _mm_load_pd( &A[2+4*0] );
	a_01_11 = _mm_load_pd( &A[0+4*1] );
	a_21_31 = _mm_load_pd( &A[2+4*1] );
	a_02_12 = _mm_load_pd( &A[0+4*2] );
	a_22_32 = _mm_load_pd( &A[2+4*2] );
	
	// first column 
	b_00 = _mm_loaddup_pd( &B[0+4*0] );
	b_10 = _mm_loaddup_pd( &B[0+4*1] );
	b_20 = _mm_loaddup_pd( &B[0+4*2] );
	
	a_00_10 = _mm_mul_pd( a_00_10, b_00 );
	a_20_30 = _mm_mul_pd( a_20_30, b_00 );

	ab_temp = a_01_11;
	ab_temp = _mm_mul_pd( ab_temp, b_10 );
	a_00_10 = _mm_add_pd( a_00_10, ab_temp );
	ab_temp = a_21_31;
	ab_temp = _mm_mul_pd( ab_temp, b_10 );
	a_20_30 = _mm_add_pd( a_20_30, ab_temp );

	ab_temp = a_02_12;
	ab_temp = _mm_mul_pd( ab_temp, b_20 );
	a_00_10 = _mm_add_pd( a_00_10, ab_temp );
	ab_temp = a_22_32;
	ab_temp = _mm_mul_pd( ab_temp, b_20 );
	a_20_30 = _mm_add_pd( a_20_30, ab_temp );

	_mm_store_pd( &C[0+ldc*0], a_00_10 );
	_mm_store_pd( &C[2+ldc*0], a_20_30 );
	
	// second column 
	b_11 = _mm_loaddup_pd( &B[1+4*1] );
	b_21 = _mm_loaddup_pd( &B[1+4*2] );

	a_01_11 = _mm_mul_pd( a_01_11, b_11 );
	a_21_31 = _mm_mul_pd( a_21_31, b_11 );

	ab_temp = a_02_12;
	ab_temp = _mm_mul_pd( ab_temp, b_21 );
	a_01_11 = _mm_add_pd( a_01_11, ab_temp );
	ab_temp = a_22_32;
	ab_temp = _mm_mul_pd( ab_temp, b_21 );
	a_21_31 = _mm_add_pd( a_21_31, ab_temp );
	
	_mm_store_pd( &C[0+ldc*1], a_01_11 );
	_mm_store_pd( &C[2+ldc*1], a_21_31 );
	
	// third column 
	b_22 = _mm_loaddup_pd( &B[2+4*2] );

	a_02_12 = _mm_mul_pd( a_02_12, b_22 );
	a_22_32 = _mm_mul_pd( a_22_32, b_22 );

	_mm_store_pd( &C[0+ldc*2], a_02_12 );
	_mm_store_pd( &C[2+ldc*2], a_22_32 );

	}
	


/*inline void corner_dtrmm_pp_nt_4x2_sse_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_4x2_sse_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m128d
		ab_temp,
		a_00_10, a_01_11, a_20_30, a_21_31,
		b_00, b_10, b_11;
	
	a_00_10 = _mm_load_pd( &A[0+4*0] );
	a_20_30 = _mm_load_pd( &A[2+4*0] );
	a_01_11 = _mm_load_pd( &A[0+4*1] );
	a_21_31 = _mm_load_pd( &A[2+4*1] );
	
	// first column 
	b_00 = _mm_loaddup_pd( &B[0+4*0] );
	b_10 = _mm_loaddup_pd( &B[0+4*1] );
	
	a_00_10 = _mm_mul_pd( a_00_10, b_00 );
	a_20_30 = _mm_mul_pd( a_20_30, b_00 );

	ab_temp = a_01_11;
	ab_temp = _mm_mul_pd( ab_temp, b_10 );
	a_00_10 = _mm_add_pd( a_00_10, ab_temp );
	ab_temp = a_21_31;
	ab_temp = _mm_mul_pd( ab_temp, b_10 );
	a_20_30 = _mm_add_pd( a_20_30, ab_temp );

	_mm_store_pd( &C[0+ldc*0], a_00_10 );
	_mm_store_pd( &C[2+ldc*0], a_20_30 );
	
	// second column 
	b_11 = _mm_loaddup_pd( &B[1+4*1] );

	a_01_11 = _mm_mul_pd( a_01_11, b_11 );
	a_21_31 = _mm_mul_pd( a_21_31, b_11 );
	
	_mm_store_pd( &C[0+ldc*1], a_01_11 );
	_mm_store_pd( &C[2+ldc*1], a_21_31 );
	
	}



/*inline void corner_dtrmm_pp_nt_4x1_sse_lib4(double *A0, double *A1, double *B, double *C0, double *C1, int ldc)*/
void corner_dtrmm_pp_nt_4x1_sse_lib4(double *A, double *B, double *C, int ldc)
	{
	
	__m128d
/*		ab_temp,*/
		a_00_10, a_20_30,
		b_00;
	
	a_00_10 = _mm_load_pd( &A[0+4*0] );
	a_20_30 = _mm_load_pd( &A[2+4*0] );
	
	// first column 
	b_00 = _mm_loaddup_pd( &B[0+4*0] );
	
	a_00_10 = _mm_mul_pd( a_00_10, b_00 );
	a_20_30 = _mm_mul_pd( a_20_30, b_00 );

	_mm_store_pd( &C[0+ldc*0], a_00_10 );
	_mm_store_pd( &C[2+ldc*0], a_20_30 );
	
	}

