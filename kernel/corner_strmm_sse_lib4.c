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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
/*#include <emmintrin.h>  // SSE2*/
/*#include <pmmintrin.h>  // SSE3*/
/*#include <smmintrin.h>  // SSE4*/
/*#include <immintrin.h>  // AVX*/



/*inline void corner_strmm_pp_nt_8x3_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)*/
void corner_strmm_pp_nt_8x3_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)
	{

	__m128
		ab_temp,
		a_0_0, a_0_1, a_0_2, a_4_0, a_4_1, a_4_2,
		b_00, b_10, b_20, b_11, b_21, b_22;
	
	a_0_0 = _mm_load_ps( &A0[0+4*0] );
	a_4_0 = _mm_load_ps( &A1[0+4*0] );
	a_0_1 = _mm_load_ps( &A0[0+4*1] );
	a_4_1 = _mm_load_ps( &A1[0+4*1] );
	a_0_2 = _mm_load_ps( &A0[0+4*2] );
	a_4_2 = _mm_load_ps( &A1[0+4*2] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	b_10 = _mm_load1_ps( &B[0+4*1] );
	b_20 = _mm_load1_ps( &B[0+4*2] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );
	a_4_0 = _mm_mul_ps( a_4_0, b_00 );

	ab_temp = a_0_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );
	ab_temp = a_4_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_4_0 = _mm_add_ps( a_4_0, ab_temp );

	ab_temp = a_0_2;
	ab_temp = _mm_mul_ps( ab_temp, b_20 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );
	ab_temp = a_4_2;
	ab_temp = _mm_mul_ps( ab_temp, b_20 );
	a_4_0 = _mm_add_ps( a_4_0, ab_temp );

	_mm_store_ps( &C0[0+ldc*0], a_0_0 );
	_mm_store_ps( &C1[0+ldc*0], a_4_0 );
	
	// second column 
	b_11 = _mm_load1_ps( &B[1+4*1] );
	b_21 = _mm_load1_ps( &B[1+4*2] );

	a_0_1 = _mm_mul_ps( a_0_1, b_11 );
	a_4_1 = _mm_mul_ps( a_4_1, b_11 );

	ab_temp = a_0_2;
	ab_temp = _mm_mul_ps( ab_temp, b_21 );
	a_0_1 = _mm_add_ps( a_0_1, ab_temp );
	ab_temp = a_4_2;
	ab_temp = _mm_mul_ps( ab_temp, b_21 );
	a_4_1 = _mm_add_ps( a_4_1, ab_temp );
	
	_mm_store_ps( &C0[0+ldc*1], a_0_1 );
	_mm_store_ps( &C1[0+ldc*1], a_4_1 );
	
	// third column 
	b_22 = _mm_load1_ps( &B[2+4*2] );

	a_0_2 = _mm_mul_ps( a_0_2, b_22 );
	a_4_2 = _mm_mul_ps( a_4_2, b_22 );

	_mm_store_ps( &C0[0+ldc*2], a_0_2 );
	_mm_store_ps( &C1[0+ldc*2], a_4_2 );

	}

	

/*inline void corner_strmm_pp_nt_8x2_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)*/
void corner_strmm_pp_nt_8x2_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)
	{

	__m128
		ab_temp,
		a_0_0, a_0_1, a_4_0, a_4_1, 
		b_00, b_10, b_11;
	
	a_0_0 = _mm_load_ps( &A0[0+4*0] );
	a_4_0 = _mm_load_ps( &A1[0+4*0] );
	a_0_1 = _mm_load_ps( &A0[0+4*1] );
	a_4_1 = _mm_load_ps( &A1[0+4*1] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	b_10 = _mm_load1_ps( &B[0+4*1] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );
	a_4_0 = _mm_mul_ps( a_4_0, b_00 );

	ab_temp = a_0_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );
	ab_temp = a_4_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_4_0 = _mm_add_ps( a_4_0, ab_temp );

	_mm_store_ps( &C0[0+ldc*0], a_0_0 );
	_mm_store_ps( &C1[0+ldc*0], a_4_0 );
	
	// second column 
	b_11 = _mm_load1_ps( &B[1+4*1] );

	a_0_1 = _mm_mul_ps( a_0_1, b_11 );
	a_4_1 = _mm_mul_ps( a_4_1, b_11 );

	_mm_store_ps( &C0[0+ldc*1], a_0_1 );
	_mm_store_ps( &C1[0+ldc*1], a_4_1 );

	}



/*inline void corner_strmm_pp_nt_8x1_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)*/
void corner_strmm_pp_nt_8x1_lib4(float *A0, float *A1, float *B, float *C0, float *C1, int ldc)
	{

	__m128
		ab_temp,
		a_0_0, a_4_0,
		b_00;
	
	a_0_0 = _mm_load_ps( &A0[0+4*0] );
	a_4_0 = _mm_load_ps( &A1[0+4*0] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );
	a_4_0 = _mm_mul_ps( a_4_0, b_00 );

	_mm_store_ps( &C0[0+ldc*0], a_0_0 );
	_mm_store_ps( &C1[0+ldc*0], a_4_0 );

	}



/*inline void corner_strmm_pp_nt_4x3_lib4(float *A, float *B, float *C, int ldc)*/
void corner_strmm_pp_nt_4x3_lib4(float *A, float *B, float *C, int ldc)
	{

	__m128
		ab_temp,
		a_0_0, a_0_1, a_0_2,
		b_00, b_10, b_20, b_11, b_21, b_22;
	
	a_0_0 = _mm_load_ps( &A[0+4*0] );
	a_0_1 = _mm_load_ps( &A[0+4*1] );
	a_0_2 = _mm_load_ps( &A[0+4*2] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	b_10 = _mm_load1_ps( &B[0+4*1] );
	b_20 = _mm_load1_ps( &B[0+4*2] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );

	ab_temp = a_0_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );

	ab_temp = a_0_2;
	ab_temp = _mm_mul_ps( ab_temp, b_20 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );

	_mm_store_ps( &C[0+ldc*0], a_0_0 );
	
	// second column 
	b_11 = _mm_load1_ps( &B[1+4*1] );
	b_21 = _mm_load1_ps( &B[1+4*2] );

	a_0_1 = _mm_mul_ps( a_0_1, b_11 );

	ab_temp = a_0_2;
	ab_temp = _mm_mul_ps( ab_temp, b_21 );
	a_0_1 = _mm_add_ps( a_0_1, ab_temp );
	
	_mm_store_ps( &C[0+ldc*1], a_0_1 );
	
	// third column 
	b_22 = _mm_load1_ps( &B[2+4*2] );

	a_0_2 = _mm_mul_ps( a_0_2, b_22 );

	_mm_store_ps( &C[0+ldc*2], a_0_2 );

	}



/*inline void corner_strmm_pp_nt_4x2_lib4(float *A, float *B, float *C, int ldc)*/
void corner_strmm_pp_nt_4x2_lib4(float *A, float *B, float *C, int ldc)
	{

	__m128
		ab_temp,
		a_0_0, a_0_1,
		b_00, b_10, b_11;
	
	a_0_0 = _mm_load_ps( &A[0+4*0] );
	a_0_1 = _mm_load_ps( &A[0+4*1] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	b_10 = _mm_load1_ps( &B[0+4*1] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );

	ab_temp = a_0_1;
	ab_temp = _mm_mul_ps( ab_temp, b_10 );
	a_0_0 = _mm_add_ps( a_0_0, ab_temp );

	_mm_store_ps( &C[0+ldc*0], a_0_0 );
	
	// second column 
	b_11 = _mm_load1_ps( &B[1+4*1] );

	a_0_1 = _mm_mul_ps( a_0_1, b_11 );

	_mm_store_ps( &C[0+ldc*1], a_0_1 );
	
	}



/*inline void corner_strmm_pp_nt_4x1_lib4(float *A, float *B, float *C, int ldc)*/
void corner_strmm_pp_nt_4x1_lib4(float *A, float *B, float *C, int ldc)
	{

	__m128
		ab_temp,
		a_0_0,
		b_00;
	
	a_0_0 = _mm_load_ps( &A[0+4*0] );
	
	// first column 
	b_00 = _mm_load1_ps( &B[0+4*0] );
	
	a_0_0 = _mm_mul_ps( a_0_0, b_00 );

	_mm_store_ps( &C[0+ldc*0], a_0_0 );
	
	}

