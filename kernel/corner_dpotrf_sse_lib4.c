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
//#include <immintrin.h>  // AVX



void corner_dpotrf_dtrsv_dcopy_3x3_sse_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);
	const int shfi2 = ((shfi+2)/lda)*lda*(sdl-1);

	__m128d
		ones, ab_temp,
		a_00, a_10, a_20, a_11, a_21, a_22;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	_mm_store_sd( &L[0+0*lda+shfi0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_20 = _mm_load_sd( &A[2+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	a_20 = _mm_mul_sd( a_20, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	_mm_store_sd( &A[2+lda*0], a_20 );
	_mm_store_sd( &L[0+1*lda+shfi0], a_10 );
	_mm_store_sd( &L[0+2*lda+shfi0], a_20 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );
	_mm_store_sd( &L[1+1*lda+shfi1], a_11 );
	a_11 = _mm_div_sd( ones, a_11 );
	a_21 = _mm_load_sd( &A[2+lda*1] );
	ab_temp = _mm_mul_sd( a_20, a_10 );
	a_21 = _mm_sub_sd( a_21, ab_temp );
	a_21 = _mm_mul_sd( a_21, a_11 );
	_mm_store_sd( &A[2+lda*1], a_21 );
	_mm_store_sd( &L[1+2*lda+shfi1], a_21 );
	
	a_22 = _mm_load_sd( &A[2+lda*2] );
	ab_temp = _mm_mul_sd( a_20, a_20 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	ab_temp = _mm_mul_sd( a_21, a_21 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	a_22 = _mm_sqrt_sd( a_22, a_22 );
	_mm_store_sd( &A[2+lda*2], a_22 );
	_mm_store_sd( &L[2+2*lda+shfi2], a_22 );

	}



void corner_dpotrf_dtrsv_dcopy_2x2_sse_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);

	__m128d
		ones, ab_temp,
		a_00, a_10, a_11;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	_mm_store_sd( &L[0+0*lda+shfi0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	_mm_store_sd( &L[0+1*lda+shfi0], a_10 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );
	_mm_store_sd( &L[1+1*lda+shfi1], a_11 );
	
	}


void corner_dpotrf_dtrsv_dcopy_1x1_sse_lib4(double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);

	__m128d
		ones, ab_temp,
		a_00;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	_mm_store_sd( &L[0+0*lda+shfi0], a_00 );
	
	}

