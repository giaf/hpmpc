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
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
//#include <immintrin.h>  // AVX



void kernel_dpotrf_dtrsv_dcopy_4x4_lib4(int kmax, double *A, int sda, int shf, double *L, int sdl, int *info)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);
	const int shfi2 = ((shfi+2)/lda)*lda*(sdl-1);
	const int shfi3 = ((shfi+3)/lda)*lda*(sdl-1);

	__m128d
		zeros, ones, ab_temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33,
		b_00_10, b_01_11, b_02_12, b_03_13;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	if( _mm_comile_sd ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	_mm_store_sd( &L[0+0*lda+shfi0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_20 = _mm_load_sd( &A[2+lda*0] );
	a_30 = _mm_load_sd( &A[3+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	a_20 = _mm_mul_sd( a_20, a_00 );
	a_30 = _mm_mul_sd( a_30, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	_mm_store_sd( &A[2+lda*0], a_20 );
	_mm_store_sd( &A[3+lda*0], a_30 );
	_mm_store_sd( &L[0+1*lda+shfi0], a_10 );
	_mm_store_sd( &L[0+2*lda+shfi0], a_20 );
	_mm_store_sd( &L[0+3*lda+shfi0], a_30 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	if( _mm_comile_sd ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );
	_mm_store_sd( &L[1+1*lda+shfi1], a_11 );
	a_11 = _mm_div_sd( ones, a_11 );
	a_21 = _mm_load_sd( &A[2+lda*1] );
	a_31 = _mm_load_sd( &A[3+lda*1] );
	ab_temp = _mm_mul_sd( a_20, a_10 );
	a_21 = _mm_sub_sd( a_21, ab_temp );
	ab_temp = _mm_mul_sd( a_30, a_10 );
	a_31 = _mm_sub_sd( a_31, ab_temp );
	a_21 = _mm_mul_sd( a_21, a_11 );
	a_31 = _mm_mul_sd( a_31, a_11 );
	_mm_store_sd( &A[2+lda*1], a_21 );
	_mm_store_sd( &A[3+lda*1], a_31 );
	_mm_store_sd( &L[1+2*lda+shfi1], a_21 );
	_mm_store_sd( &L[1+3*lda+shfi1], a_31 );
	
	a_22 = _mm_load_sd( &A[2+lda*2] );
	ab_temp = _mm_mul_sd( a_20, a_20 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	ab_temp = _mm_mul_sd( a_21, a_21 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	if( _mm_comile_sd ( a_22, zeros ) ) { *info = 1; return; }
	a_22 = _mm_sqrt_sd( a_22, a_22 );
	_mm_store_sd( &A[2+lda*2], a_22 );
	_mm_store_sd( &L[2+2*lda+shfi2], a_22 );
	a_22 = _mm_div_sd( ones, a_22 );
	a_32 = _mm_load_sd( &A[3+lda*2] );
	ab_temp = _mm_mul_sd( a_30, a_20 );
	a_32 = _mm_sub_sd( a_32, ab_temp );
	ab_temp = _mm_mul_sd( a_31, a_21 );
	a_32 = _mm_sub_sd( a_32, ab_temp );
	a_32 = _mm_mul_sd( a_32, a_22 );
	_mm_store_sd( &A[3+lda*2], a_32 );
	_mm_store_sd( &L[2+3*lda+shfi2], a_32 );

	a_33 = _mm_load_sd( &A[3+lda*3] );
	ab_temp = _mm_mul_sd( a_30, a_30 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	ab_temp = _mm_mul_sd( a_31, a_31 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	ab_temp = _mm_mul_sd( a_32, a_32 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	if( _mm_comile_sd ( a_33, zeros ) ) { *info = 1; return; }
	a_33 = _mm_sqrt_sd( a_33, a_33 );
	_mm_store_sd( &A[3+lda*3], a_33 );
	_mm_store_sd( &L[3+3*lda+shfi3], a_33 );


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_33 = _mm_div_sd( ones, a_33 );

	a_00 = _mm_movedup_pd( a_00 );
	a_10 = _mm_movedup_pd( a_10 );
	a_20 = _mm_movedup_pd( a_20 );
	a_30 = _mm_movedup_pd( a_30 );
	a_11 = _mm_movedup_pd( a_11 );
	a_21 = _mm_movedup_pd( a_21 );
	a_31 = _mm_movedup_pd( a_31 );
	a_22 = _mm_movedup_pd( a_22 );
	a_32 = _mm_movedup_pd( a_32 );
	a_33 = _mm_movedup_pd( a_33 );
	
	int k, kk, kend;
	
	double
		*AA, *LL;
	
	AA = A+4;
	LL = L+4*lda;
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_pd( &AA[0+lda*0] );
		b_01_11 = _mm_load_pd( &AA[0+lda*1] );
		b_02_12 = _mm_load_pd( &AA[0+lda*2] );
		b_03_13 = _mm_load_pd( &AA[0+lda*3] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[0+lda*0], b_00_10 );
		_mm_storel_pd( &LL[0+shfi0+0*lda], b_00_10 );
		_mm_storeh_pd( &LL[0+shfi0+1*lda], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[0+lda*1], b_01_11 );
		_mm_storel_pd( &LL[1+shfi1+0*lda], b_01_11 );
		_mm_storeh_pd( &LL[1+shfi1+1*lda], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[0+lda*2], b_02_12 );
		_mm_storel_pd( &LL[2+shfi2+0*lda], b_02_12 );
		_mm_storeh_pd( &LL[2+shfi2+1*lda], b_02_12 );

		ab_temp = _mm_mul_pd( b_00_10, a_30 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_31 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_02_12, a_32 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_pd( b_03_13, a_33 );
		_mm_store_pd( &AA[0+lda*3], b_03_13 );
		_mm_storel_pd( &LL[3+shfi3+0*lda], b_03_13 );
		_mm_storeh_pd( &LL[3+shfi3+1*lda], b_03_13 );



		b_00_10 = _mm_load_pd( &AA[2+lda*0] );
		b_01_11 = _mm_load_pd( &AA[2+lda*1] );
		b_02_12 = _mm_load_pd( &AA[2+lda*2] );
		b_03_13 = _mm_load_pd( &AA[2+lda*3] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[2+lda*0], b_00_10 );
		_mm_storel_pd( &LL[0+shfi0+2*lda], b_00_10 );
		_mm_storeh_pd( &LL[0+shfi0+3*lda], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[2+lda*1], b_01_11 );
		_mm_storel_pd( &LL[1+shfi1+2*lda], b_01_11 );
		_mm_storeh_pd( &LL[1+shfi1+3*lda], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[2+lda*2], b_02_12 );
		_mm_storel_pd( &LL[2+shfi2+2*lda], b_02_12 );
		_mm_storeh_pd( &LL[2+shfi2+3*lda], b_02_12 );

		ab_temp = _mm_mul_pd( b_00_10, a_30 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_31 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_02_12, a_32 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_pd( b_03_13, a_33 );
		_mm_store_pd( &AA[2+lda*3], b_03_13 );
		_mm_storel_pd( &LL[3+shfi3+2*lda], b_03_13 );
		_mm_storeh_pd( &LL[3+shfi3+3*lda], b_03_13 );

		AA += 4;
		LL += 4*lda;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );
		b_02_12 = _mm_load_sd( &AA[lda*2] );
		b_03_13 = _mm_load_sd( &AA[lda*3] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
		_mm_store_sd( &LL[0+shfi0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );
		_mm_store_sd( &LL[1+shfi1], b_01_11 );

		ab_temp = _mm_mul_sd( b_00_10, a_20 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_21 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_sd( b_02_12, a_22 );
		_mm_store_sd( &AA[lda*2], b_02_12 );
		_mm_store_sd( &LL[2+shfi2], b_02_12 );

		ab_temp = _mm_mul_sd( b_00_10, a_30 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_31 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		ab_temp = _mm_mul_sd( b_02_12, a_32 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_sd( b_03_13, a_33 );
		_mm_store_sd( &AA[lda*3], b_03_13 );
		_mm_store_sd( &LL[3+shfi3], b_03_13 );

		AA += 1;
		LL += lda;
		}
	
	}



void kernel_dpotrf_dtrsv_4x4_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128d
		zeros, ones, ab_temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33,
		b_00_10, b_01_11, b_02_12, b_03_13;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	if( _mm_comile_sd ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_20 = _mm_load_sd( &A[2+lda*0] );
	a_30 = _mm_load_sd( &A[3+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	a_20 = _mm_mul_sd( a_20, a_00 );
	a_30 = _mm_mul_sd( a_30, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	_mm_store_sd( &A[2+lda*0], a_20 );
	_mm_store_sd( &A[3+lda*0], a_30 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	if( _mm_comile_sd ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );
	a_11 = _mm_div_sd( ones, a_11 );
	a_21 = _mm_load_sd( &A[2+lda*1] );
	a_31 = _mm_load_sd( &A[3+lda*1] );
	ab_temp = _mm_mul_sd( a_20, a_10 );
	a_21 = _mm_sub_sd( a_21, ab_temp );
	ab_temp = _mm_mul_sd( a_30, a_10 );
	a_31 = _mm_sub_sd( a_31, ab_temp );
	a_21 = _mm_mul_sd( a_21, a_11 );
	a_31 = _mm_mul_sd( a_31, a_11 );
	_mm_store_sd( &A[2+lda*1], a_21 );
	_mm_store_sd( &A[3+lda*1], a_31 );
	
	a_22 = _mm_load_sd( &A[2+lda*2] );
	ab_temp = _mm_mul_sd( a_20, a_20 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	ab_temp = _mm_mul_sd( a_21, a_21 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	if( _mm_comile_sd ( a_22, zeros ) ) { *info = 1; return; }
	a_22 = _mm_sqrt_sd( a_22, a_22 );
	_mm_store_sd( &A[2+lda*2], a_22 );
	a_22 = _mm_div_sd( ones, a_22 );
	a_32 = _mm_load_sd( &A[3+lda*2] );
	ab_temp = _mm_mul_sd( a_30, a_20 );
	a_32 = _mm_sub_sd( a_32, ab_temp );
	ab_temp = _mm_mul_sd( a_31, a_21 );
	a_32 = _mm_sub_sd( a_32, ab_temp );
	a_32 = _mm_mul_sd( a_32, a_22 );
	_mm_store_sd( &A[3+lda*2], a_32 );

	a_33 = _mm_load_sd( &A[3+lda*3] );
	ab_temp = _mm_mul_sd( a_30, a_30 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	ab_temp = _mm_mul_sd( a_31, a_31 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	ab_temp = _mm_mul_sd( a_32, a_32 );
	a_33 = _mm_sub_sd( a_33, ab_temp );
	if( _mm_comile_sd ( a_33, zeros ) ) { *info = 1; return; }
	a_33 = _mm_sqrt_sd( a_33, a_33 );
	_mm_store_sd( &A[3+lda*3], a_33 );


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_33 = _mm_div_sd( ones, a_33 );

	a_00 = _mm_movedup_pd( a_00 );
	a_10 = _mm_movedup_pd( a_10 );
	a_20 = _mm_movedup_pd( a_20 );
	a_30 = _mm_movedup_pd( a_30 );
	a_11 = _mm_movedup_pd( a_11 );
	a_21 = _mm_movedup_pd( a_21 );
	a_31 = _mm_movedup_pd( a_31 );
	a_22 = _mm_movedup_pd( a_22 );
	a_32 = _mm_movedup_pd( a_32 );
	a_33 = _mm_movedup_pd( a_33 );
	
	int k, kk, kend;
	
	double
		*AA;
	
	AA = A+4;
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_pd( &AA[0+lda*0] );
		b_01_11 = _mm_load_pd( &AA[0+lda*1] );
		b_02_12 = _mm_load_pd( &AA[0+lda*2] );
		b_03_13 = _mm_load_pd( &AA[0+lda*3] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[0+lda*1], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[0+lda*2], b_02_12 );

		ab_temp = _mm_mul_pd( b_00_10, a_30 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_31 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_02_12, a_32 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_pd( b_03_13, a_33 );
		_mm_store_pd( &AA[0+lda*3], b_03_13 );



		b_00_10 = _mm_load_pd( &AA[2+lda*0] );
		b_01_11 = _mm_load_pd( &AA[2+lda*1] );
		b_02_12 = _mm_load_pd( &AA[2+lda*2] );
		b_03_13 = _mm_load_pd( &AA[2+lda*3] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[2+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[2+lda*1], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[2+lda*2], b_02_12 );

		ab_temp = _mm_mul_pd( b_00_10, a_30 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_31 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		ab_temp = _mm_mul_pd( b_02_12, a_32 );
		b_03_13 = _mm_sub_pd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_pd( b_03_13, a_33 );
		_mm_store_pd( &AA[2+lda*3], b_03_13 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );
		b_02_12 = _mm_load_sd( &AA[lda*2] );
		b_03_13 = _mm_load_sd( &AA[lda*3] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_sd( b_00_10, a_20 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_21 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_sd( b_02_12, a_22 );
		_mm_store_sd( &AA[lda*2], b_02_12 );

		ab_temp = _mm_mul_sd( b_00_10, a_30 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_31 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		ab_temp = _mm_mul_sd( b_02_12, a_32 );
		b_03_13 = _mm_sub_sd( b_03_13, ab_temp );
		b_03_13 = _mm_mul_sd( b_03_13, a_33 );
		_mm_store_sd( &AA[lda*3], b_03_13 );

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_3x3_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128d
		zeros, ones, ab_temp,
		a_00, a_10, a_20, a_11, a_21, a_22,
		b_00_10, b_01_11, b_02_12;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	if( _mm_comile_sd ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_20 = _mm_load_sd( &A[2+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	a_20 = _mm_mul_sd( a_20, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	_mm_store_sd( &A[2+lda*0], a_20 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	if( _mm_comile_sd ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );
	a_11 = _mm_div_sd( ones, a_11 );
	a_21 = _mm_load_sd( &A[2+lda*1] );
	ab_temp = _mm_mul_sd( a_20, a_10 );
	a_21 = _mm_sub_sd( a_21, ab_temp );
	a_21 = _mm_mul_sd( a_21, a_11 );
	_mm_store_sd( &A[2+lda*1], a_21 );
	
	a_22 = _mm_load_sd( &A[2+lda*2] );
	ab_temp = _mm_mul_sd( a_20, a_20 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	ab_temp = _mm_mul_sd( a_21, a_21 );
	a_22 = _mm_sub_sd( a_22, ab_temp );
	if( _mm_comile_sd ( a_22, zeros ) ) { *info = 1; return; }
	a_22 = _mm_sqrt_sd( a_22, a_22 );
	_mm_store_sd( &A[2+lda*2], a_22 );

	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_22 = _mm_div_sd( ones, a_22 );

	a_00 = _mm_movedup_pd( a_00 );
	a_10 = _mm_movedup_pd( a_10 );
	a_20 = _mm_movedup_pd( a_20 );
	a_11 = _mm_movedup_pd( a_11 );
	a_21 = _mm_movedup_pd( a_21 );
	a_22 = _mm_movedup_pd( a_22 );
	
	int k, kna;
	
	double
		*AA;
	
	AA = A + 3;
	k = 0;

	// clean up unaligned stuff at the beginning
	kna = 1;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );
		b_02_12 = _mm_load_sd( &AA[lda*2] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_sd( b_00_10, a_20 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_21 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_sd( b_02_12, a_22 );
		_mm_store_sd( &AA[lda*2], b_02_12 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_pd( &AA[0+lda*0] );
		b_01_11 = _mm_load_pd( &AA[0+lda*1] );
		b_02_12 = _mm_load_pd( &AA[0+lda*2] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[0+lda*1], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[0+lda*2], b_02_12 );



		b_00_10 = _mm_load_pd( &AA[2+lda*0] );
		b_01_11 = _mm_load_pd( &AA[2+lda*1] );
		b_02_12 = _mm_load_pd( &AA[2+lda*2] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[2+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[2+lda*1], b_01_11 );

		ab_temp = _mm_mul_pd( b_00_10, a_20 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		ab_temp = _mm_mul_pd( b_01_11, a_21 );
		b_02_12 = _mm_sub_pd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_pd( b_02_12, a_22 );
		_mm_store_pd( &AA[2+lda*2], b_02_12 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );
		b_02_12 = _mm_load_sd( &AA[lda*2] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_sd( b_00_10, a_20 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		ab_temp = _mm_mul_sd( b_01_11, a_21 );
		b_02_12 = _mm_sub_sd( b_02_12, ab_temp );
		b_02_12 = _mm_mul_sd( b_02_12, a_22 );
		_mm_store_sd( &AA[lda*2], b_02_12 );

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_2x2_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128d
		zeros, ones, ab_temp,
		a_00, a_10, a_11,
		b_00_10, b_01_11;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	if( _mm_comile_sd ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	a_00 = _mm_div_sd( ones, a_00 );
	a_10 = _mm_load_sd( &A[1+lda*0] );
	a_10 = _mm_mul_sd( a_10, a_00 );
	_mm_store_sd( &A[1+lda*0], a_10 );
	
	a_11 = _mm_load_sd( &A[1+lda*1] );
	ab_temp = _mm_mul_sd( a_10, a_10 );
	a_11 = _mm_sub_sd( a_11, ab_temp );
	if( _mm_comile_sd ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_sd( a_11, a_11 );
	_mm_store_sd( &A[1+lda*1], a_11 );

	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_11 = _mm_div_sd( ones, a_11 );

	a_00 = _mm_movedup_pd( a_00 );
	a_10 = _mm_movedup_pd( a_10 );
	a_11 = _mm_movedup_pd( a_11 );
	
	int k, kna;
	
	double
		*AA;
	
	AA = A + 2;

	k = 0;
	
	// clean up unaligned stuff at the beginning
	kna = 2;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_pd( &AA[0+lda*0] );
		b_01_11 = _mm_load_pd( &AA[0+lda*1] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[0+lda*1], b_01_11 );



		b_00_10 = _mm_load_pd( &AA[2+lda*0] );
		b_01_11 = _mm_load_pd( &AA[2+lda*1] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[2+lda*0], b_00_10 );

		ab_temp = _mm_mul_pd( b_00_10, a_10 );
		b_01_11 = _mm_sub_pd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_pd( b_01_11, a_11 );
		_mm_store_pd( &AA[2+lda*1], b_01_11 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );
		b_01_11 = _mm_load_sd( &AA[lda*1] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_sd( b_00_10, a_10 );
		b_01_11 = _mm_sub_sd( b_01_11, ab_temp );
		b_01_11 = _mm_mul_sd( b_01_11, a_11 );
		_mm_store_sd( &AA[lda*1], b_01_11 );

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_1x1_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128d
		zeros, ones, ab_temp,
		a_00,
		b_00_10;
	
	a_00 = _mm_load_sd( &A[0+lda*0] );
	if( _mm_comile_sd ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_sd( a_00, a_00 );
	ones = _mm_set_sd( 1.0 );
	_mm_store_sd( &A[0+lda*0], a_00 );
	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_00 = _mm_div_sd( ones, a_00 );

	a_00 = _mm_movedup_pd( a_00 );
	
	int k, kna;
	
	double
		*AA;
	
	AA = A + 1;
	k = 0;

	// clean up unaligned stuff at the beginning
	kna = 3;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_pd( &AA[0+lda*0] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[0+lda*0], b_00_10 );



		b_00_10 = _mm_load_pd( &AA[2+lda*0] );

		b_00_10 = _mm_mul_pd( b_00_10, a_00 );
		_mm_store_pd( &AA[2+lda*0], b_00_10 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_sd( &AA[lda*0] );

		b_00_10 = _mm_mul_sd( b_00_10, a_00 );
		_mm_store_sd( &AA[lda*0], b_00_10 );

		AA += 1;
		}
	
	}

