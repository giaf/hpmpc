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
//#include <immintrin.h>  // AVX



void kernel_spotrf_strsv_4x4_lib4(int kmax, int kinv, float *A, int sda, int *info)
	{
	
	const int lda = 4;
	

	__m128
		zeros, ones, ab_temp,
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33,
		b_00_10, b_01_11, b_02_12, b_03_13;
	
	zeros = _mm_set_ss( 0.0 );

	if(kinv==0)
		{
		
		a_00 = _mm_load_ss( &A[0+lda*0] );
		if( _mm_comile_ss ( a_00, zeros ) ) { *info = 1; return; }
		a_00 = _mm_sqrt_ss( a_00 );
		ones = _mm_set_ss( 1.0 );
		_mm_store_ss( &A[0+lda*0], a_00 );
		a_00 = _mm_div_ss( ones, a_00 );
		a_10 = _mm_load_ss( &A[1+lda*0] );
		a_20 = _mm_load_ss( &A[2+lda*0] );
		a_30 = _mm_load_ss( &A[3+lda*0] );
		a_10 = _mm_mul_ss( a_10, a_00 );
		a_20 = _mm_mul_ss( a_20, a_00 );
		a_30 = _mm_mul_ss( a_30, a_00 );
		_mm_store_ss( &A[1+lda*0], a_10 );
		_mm_store_ss( &A[2+lda*0], a_20 );
		_mm_store_ss( &A[3+lda*0], a_30 );
	
		a_11 = _mm_load_ss( &A[1+lda*1] );
		ab_temp = _mm_mul_ss( a_10, a_10 );
		a_11 = _mm_sub_ss( a_11, ab_temp );
		if( _mm_comile_ss ( a_11, zeros ) ) { *info = 1; return; }
		a_11 = _mm_sqrt_ss( a_11 );
		_mm_store_ss( &A[1+lda*1], a_11 );
		a_11 = _mm_div_ss( ones, a_11 );
		a_21 = _mm_load_ss( &A[2+lda*1] );
		a_31 = _mm_load_ss( &A[3+lda*1] );
		ab_temp = _mm_mul_ss( a_20, a_10 );
		a_21 = _mm_sub_ss( a_21, ab_temp );
		ab_temp = _mm_mul_ss( a_30, a_10 );
		a_31 = _mm_sub_ss( a_31, ab_temp );
		a_21 = _mm_mul_ss( a_21, a_11 );
		a_31 = _mm_mul_ss( a_31, a_11 );
		_mm_store_ss( &A[2+lda*1], a_21 );
		_mm_store_ss( &A[3+lda*1], a_31 );
	
		a_22 = _mm_load_ss( &A[2+lda*2] );
		ab_temp = _mm_mul_ss( a_20, a_20 );
		a_22 = _mm_sub_ss( a_22, ab_temp );
		ab_temp = _mm_mul_ss( a_21, a_21 );
		a_22 = _mm_sub_ss( a_22, ab_temp );
		if( _mm_comile_ss ( a_22, zeros ) ) { *info = 1; return; }
		a_22 = _mm_sqrt_ss( a_22 );
		_mm_store_ss( &A[2+lda*2], a_22 );
		a_22 = _mm_div_ss( ones, a_22 );
		a_32 = _mm_load_ss( &A[3+lda*2] );
		ab_temp = _mm_mul_ss( a_30, a_20 );
		a_32 = _mm_sub_ss( a_32, ab_temp );
		ab_temp = _mm_mul_ss( a_31, a_21 );
		a_32 = _mm_sub_ss( a_32, ab_temp );
		a_32 = _mm_mul_ss( a_32, a_22 );
		_mm_store_ss( &A[3+lda*2], a_32 );
			
		a_33 = _mm_load_ss( &A[3+lda*3] );
		ab_temp = _mm_mul_ss( a_30, a_30 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		ab_temp = _mm_mul_ss( a_31, a_31 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		ab_temp = _mm_mul_ss( a_32, a_32 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		if( _mm_comile_ss ( a_33, zeros ) ) { *info = 1; return; }
		a_33 = _mm_sqrt_ss( a_33 );
		_mm_store_ss( &A[3+lda*3], a_33 );
		if(kmax>0)
			a_33 = _mm_div_ss( ones, a_33 );

		}
	else // kinv == {1, 2, 3}
		{		

		a_00 = _mm_load_ss( &A[0+lda*0] );
		if( _mm_comile_ss ( a_00, zeros ) ) { *info = 1; return; }
		a_00 = _mm_sqrt_ss( a_00 );
		ones = _mm_set_ss( 1.0 );
		a_00 = _mm_div_ss( ones, a_00 );
		_mm_store_ss( &A[0+lda*0], a_00 );
		a_10 = _mm_load_ss( &A[1+lda*0] );
		a_20 = _mm_load_ss( &A[2+lda*0] );
		a_30 = _mm_load_ss( &A[3+lda*0] );
		a_10 = _mm_mul_ss( a_10, a_00 );
		a_20 = _mm_mul_ss( a_20, a_00 );
		a_30 = _mm_mul_ss( a_30, a_00 );
		_mm_store_ss( &A[1+lda*0], a_10 );
		_mm_store_ss( &A[2+lda*0], a_20 );
		_mm_store_ss( &A[3+lda*0], a_30 );
	
		a_11 = _mm_load_ss( &A[1+lda*1] );
		ab_temp = _mm_mul_ss( a_10, a_10 );
		a_11 = _mm_sub_ss( a_11, ab_temp );
		if( _mm_comile_ss ( a_11, zeros ) ) { *info = 1; return; }
		a_11 = _mm_sqrt_ss( a_11 );
		if(kinv<=1)
			{
			_mm_store_ss( &A[1+lda*1], a_11 );
			}
		a_11 = _mm_div_ss( ones, a_11 );
		if(kinv>1)
			_mm_store_ss( &A[1+lda*1], a_11 );
		a_21 = _mm_load_ss( &A[2+lda*1] );
		a_31 = _mm_load_ss( &A[3+lda*1] );
		ab_temp = _mm_mul_ss( a_20, a_10 );
		a_21 = _mm_sub_ss( a_21, ab_temp );
		ab_temp = _mm_mul_ss( a_30, a_10 );
		a_31 = _mm_sub_ss( a_31, ab_temp );
		a_21 = _mm_mul_ss( a_21, a_11 );
		a_31 = _mm_mul_ss( a_31, a_11 );
		_mm_store_ss( &A[2+lda*1], a_21 );
		_mm_store_ss( &A[3+lda*1], a_31 );
	
		a_22 = _mm_load_ss( &A[2+lda*2] );
		ab_temp = _mm_mul_ss( a_20, a_20 );
		a_22 = _mm_sub_ss( a_22, ab_temp );
		ab_temp = _mm_mul_ss( a_21, a_21 );
		a_22 = _mm_sub_ss( a_22, ab_temp );
		if( _mm_comile_ss ( a_22, zeros ) ) { *info = 1; return; }
		a_22 = _mm_sqrt_ss( a_22 );
		if(kinv<=2)
			{
			_mm_store_ss( &A[2+lda*2], a_22 );
			}
		a_22 = _mm_div_ss( ones, a_22 );
		if(kinv>2)
			_mm_store_ss( &A[2+lda*2], a_22 );
		a_32 = _mm_load_ss( &A[3+lda*2] );
		ab_temp = _mm_mul_ss( a_30, a_20 );
		a_32 = _mm_sub_ss( a_32, ab_temp );
		ab_temp = _mm_mul_ss( a_31, a_21 );
		a_32 = _mm_sub_ss( a_32, ab_temp );
		a_32 = _mm_mul_ss( a_32, a_22 );
		_mm_store_ss( &A[3+lda*2], a_32 );
		
		a_33 = _mm_load_ss( &A[3+lda*3] );
		ab_temp = _mm_mul_ss( a_30, a_30 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		ab_temp = _mm_mul_ss( a_31, a_31 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		ab_temp = _mm_mul_ss( a_32, a_32 );
		a_33 = _mm_sub_ss( a_33, ab_temp );
		if( _mm_comile_ss ( a_33, zeros ) ) { *info = 1; return; }
		a_33 = _mm_sqrt_ss( a_33 );
		_mm_store_ss( &A[3+lda*3], a_33 );
		if(kinv<=3)
			{
			_mm_store_ss( &A[3+lda*3], a_33 );
			}
		a_33 = _mm_div_ss( ones, a_33 );
		if(kinv>3)
			_mm_store_ss( &A[3+lda*3], a_33 );

		}

	
	if(kmax<=0)
		return;
	
	// strsv

/*	a_33 = _mm_div_ss( ones, a_33 );*/

	a_00 = _mm_shuffle_ps( a_00, a_00, 0 );
	a_10 = _mm_shuffle_ps( a_10, a_10, 0 );
	a_20 = _mm_shuffle_ps( a_20, a_20, 0 );
	a_30 = _mm_shuffle_ps( a_30, a_30, 0 );
	a_11 = _mm_shuffle_ps( a_11, a_11, 0 );
	a_21 = _mm_shuffle_ps( a_21, a_21, 0 );
	a_31 = _mm_shuffle_ps( a_31, a_31, 0 );
	a_22 = _mm_shuffle_ps( a_22, a_22, 0 );
	a_32 = _mm_shuffle_ps( a_32, a_32, 0 );
	a_33 = _mm_shuffle_ps( a_33, a_33, 0 );
	
	int k;
	
	float
		*AA;
	
	AA = A+4;
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_ps( &AA[0+lda*0] );
		b_01_11 = _mm_load_ps( &AA[0+lda*1] );
		b_02_12 = _mm_load_ps( &AA[0+lda*2] );
		b_03_13 = _mm_load_ps( &AA[0+lda*3] );

		b_00_10 = _mm_mul_ps( b_00_10, a_00 );
		_mm_store_ps( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_ps( b_00_10, a_10 );
		b_01_11 = _mm_sub_ps( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ps( b_01_11, a_11 );
		_mm_store_ps( &AA[0+lda*1], b_01_11 );

		ab_temp = _mm_mul_ps( b_00_10, a_20 );
		b_02_12 = _mm_sub_ps( b_02_12, ab_temp );
		ab_temp = _mm_mul_ps( b_01_11, a_21 );
		b_02_12 = _mm_sub_ps( b_02_12, ab_temp );
		b_02_12 = _mm_mul_ps( b_02_12, a_22 );
		_mm_store_ps( &AA[0+lda*2], b_02_12 );

		ab_temp = _mm_mul_ps( b_00_10, a_30 );
		b_03_13 = _mm_sub_ps( b_03_13, ab_temp );
		ab_temp = _mm_mul_ps( b_01_11, a_31 );
		b_03_13 = _mm_sub_ps( b_03_13, ab_temp );
		ab_temp = _mm_mul_ps( b_02_12, a_32 );
		b_03_13 = _mm_sub_ps( b_03_13, ab_temp );
		b_03_13 = _mm_mul_ps( b_03_13, a_33 );
		_mm_store_ps( &AA[0+lda*3], b_03_13 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );
		b_01_11 = _mm_load_ss( &AA[lda*1] );
		b_02_12 = _mm_load_ss( &AA[lda*2] );
		b_03_13 = _mm_load_ss( &AA[lda*3] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_ss( b_00_10, a_10 );
		b_01_11 = _mm_sub_ss( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ss( b_01_11, a_11 );
		_mm_store_ss( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_ss( b_00_10, a_20 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		ab_temp = _mm_mul_ss( b_01_11, a_21 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		b_02_12 = _mm_mul_ss( b_02_12, a_22 );
		_mm_store_ss( &AA[lda*2], b_02_12 );

		ab_temp = _mm_mul_ss( b_00_10, a_30 );
		b_03_13 = _mm_sub_ss( b_03_13, ab_temp );
		ab_temp = _mm_mul_ss( b_01_11, a_31 );
		b_03_13 = _mm_sub_ss( b_03_13, ab_temp );
		ab_temp = _mm_mul_ss( b_02_12, a_32 );
		b_03_13 = _mm_sub_ss( b_03_13, ab_temp );
		b_03_13 = _mm_mul_ss( b_03_13, a_33 );
		_mm_store_ss( &AA[lda*3], b_03_13 );

		AA += 1;
		}
	
	}



// inverted diagonal !!!
void kernel_spotrf_strsv_3x3_lib4(int kmax, float *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128
		zeros, ones, ab_temp,
		a_00, a_10, a_20, a_11, a_21, a_22,
		b_00_10, b_01_11, b_02_12;
	
	zeros = _mm_set_ss( 0.0 );

	a_00 = _mm_load_ss( &A[0+lda*0] );
	if( _mm_comile_ss ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_ss( a_00 );
	ones = _mm_set_ss( 1.0 );
	a_00 = _mm_div_ss( ones, a_00 );
	_mm_store_ss( &A[0+lda*0], a_00 );
	a_10 = _mm_load_ss( &A[1+lda*0] );
	a_20 = _mm_load_ss( &A[2+lda*0] );
	a_10 = _mm_mul_ss( a_10, a_00 );
	a_20 = _mm_mul_ss( a_20, a_00 );
	_mm_store_ss( &A[1+lda*0], a_10 );
	_mm_store_ss( &A[2+lda*0], a_20 );
	
	a_11 = _mm_load_ss( &A[1+lda*1] );
	ab_temp = _mm_mul_ss( a_10, a_10 );
	a_11 = _mm_sub_ss( a_11, ab_temp );
	if( _mm_comile_ss ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_ss( a_11 );
	a_11 = _mm_div_ss( ones, a_11 );
	_mm_store_ss( &A[1+lda*1], a_11 );
	a_21 = _mm_load_ss( &A[2+lda*1] );
	ab_temp = _mm_mul_ss( a_20, a_10 );
	a_21 = _mm_sub_ss( a_21, ab_temp );
	a_21 = _mm_mul_ss( a_21, a_11 );
	_mm_store_ss( &A[2+lda*1], a_21 );
	
	a_22 = _mm_load_ss( &A[2+lda*2] );
	ab_temp = _mm_mul_ss( a_20, a_20 );
	a_22 = _mm_sub_ss( a_22, ab_temp );
	ab_temp = _mm_mul_ss( a_21, a_21 );
	a_22 = _mm_sub_ss( a_22, ab_temp );
	if( _mm_comile_ss ( a_22, zeros ) ) { *info = 1; return; }
	a_22 = _mm_sqrt_ss( a_22 );
	a_22 = _mm_div_ss( ones, a_22 );
	_mm_store_ss( &A[2+lda*2], a_22 );

	
	if(kmax<=0)
		return;
	
	// strsv


	a_00 = _mm_shuffle_ps( a_00, a_00, 0 );
	a_10 = _mm_shuffle_ps( a_10, a_10, 0 );
	a_20 = _mm_shuffle_ps( a_20, a_20, 0 );
	a_11 = _mm_shuffle_ps( a_11, a_11, 0 );
	a_21 = _mm_shuffle_ps( a_21, a_21, 0 );
	a_22 = _mm_shuffle_ps( a_22, a_22, 0 );
	
	int k, kna;
	
	float
		*AA;
	
	AA = A + 3;
	k = 0;

	// clean up unaligned stuff at the beginning
	kna = 1;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );
		b_01_11 = _mm_load_ss( &AA[lda*1] );
		b_02_12 = _mm_load_ss( &AA[lda*2] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_ss( b_00_10, a_10 );
		b_01_11 = _mm_sub_ss( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ss( b_01_11, a_11 );
		_mm_store_ss( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_ss( b_00_10, a_20 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		ab_temp = _mm_mul_ss( b_01_11, a_21 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		b_02_12 = _mm_mul_ss( b_02_12, a_22 );
		_mm_store_ss( &AA[lda*2], b_02_12 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_ps( &AA[0+lda*0] );
		b_01_11 = _mm_load_ps( &AA[0+lda*1] );
		b_02_12 = _mm_load_ps( &AA[0+lda*2] );

		b_00_10 = _mm_mul_ps( b_00_10, a_00 );
		_mm_store_ps( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_ps( b_00_10, a_10 );
		b_01_11 = _mm_sub_ps( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ps( b_01_11, a_11 );
		_mm_store_ps( &AA[0+lda*1], b_01_11 );

		ab_temp = _mm_mul_ps( b_00_10, a_20 );
		b_02_12 = _mm_sub_ps( b_02_12, ab_temp );
		ab_temp = _mm_mul_ps( b_01_11, a_21 );
		b_02_12 = _mm_sub_ps( b_02_12, ab_temp );
		b_02_12 = _mm_mul_ps( b_02_12, a_22 );
		_mm_store_ps( &AA[0+lda*2], b_02_12 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );
		b_01_11 = _mm_load_ss( &AA[lda*1] );
		b_02_12 = _mm_load_ss( &AA[lda*2] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_ss( b_00_10, a_10 );
		b_01_11 = _mm_sub_ss( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ss( b_01_11, a_11 );
		_mm_store_ss( &AA[lda*1], b_01_11 );

		ab_temp = _mm_mul_ss( b_00_10, a_20 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		ab_temp = _mm_mul_ss( b_01_11, a_21 );
		b_02_12 = _mm_sub_ss( b_02_12, ab_temp );
		b_02_12 = _mm_mul_ss( b_02_12, a_22 );
		_mm_store_ss( &AA[lda*2], b_02_12 );

		AA += 1;
		}
	
	}



// inverted diagonal !!!
void kernel_spotrf_strsv_2x2_lib4(int kmax, float *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128
		zeros, ones, ab_temp,
		a_00, a_10, a_11,
		b_00_10, b_01_11;
	
	zeros = _mm_set_ss( 0.0 );

	a_00 = _mm_load_ss( &A[0+lda*0] );
	if( _mm_comile_ss ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_ss( a_00 );
	ones = _mm_set_ss( 1.0 );
	a_00 = _mm_div_ss( ones, a_00 );
	_mm_store_ss( &A[0+lda*0], a_00 );
	a_10 = _mm_load_ss( &A[1+lda*0] );
	a_10 = _mm_mul_ss( a_10, a_00 );
	_mm_store_ss( &A[1+lda*0], a_10 );
	
	a_11 = _mm_load_ss( &A[1+lda*1] );
	ab_temp = _mm_mul_ss( a_10, a_10 );
	a_11 = _mm_sub_ss( a_11, ab_temp );
	if( _mm_comile_ss ( a_11, zeros ) ) { *info = 1; return; }
	a_11 = _mm_sqrt_ss( a_11 );
	a_11 = _mm_div_ss( ones, a_11 );
	_mm_store_ss( &A[1+lda*1], a_11 );

	
	if(kmax<=0)
		return;
	
	// strsv


	a_00 = _mm_shuffle_ps( a_00, a_00, 0 );
	a_10 = _mm_shuffle_ps( a_10, a_10, 0 );
	a_11 = _mm_shuffle_ps( a_11, a_11, 0 );
	
	int k, kna;
	
	float
		*AA;
	
	AA = A + 2;

	k = 0;
	
	// clean up unaligned stuff at the beginning
	kna = 2;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );
		b_01_11 = _mm_load_ss( &AA[lda*1] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_ss( b_00_10, a_10 );
		b_01_11 = _mm_sub_ss( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ss( b_01_11, a_11 );
		_mm_store_ss( &AA[lda*1], b_01_11 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_ps( &AA[0+lda*0] );
		b_01_11 = _mm_load_ps( &AA[0+lda*1] );

		b_00_10 = _mm_mul_ps( b_00_10, a_00 );
		_mm_store_ps( &AA[0+lda*0], b_00_10 );

		ab_temp = _mm_mul_ps( b_00_10, a_10 );
		b_01_11 = _mm_sub_ps( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ps( b_01_11, a_11 );
		_mm_store_ps( &AA[0+lda*1], b_01_11 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );
		b_01_11 = _mm_load_ss( &AA[lda*1] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );
	
		ab_temp = _mm_mul_ss( b_00_10, a_10 );
		b_01_11 = _mm_sub_ss( b_01_11, ab_temp );
		b_01_11 = _mm_mul_ss( b_01_11, a_11 );
		_mm_store_ss( &AA[lda*1], b_01_11 );

		AA += 1;
		}
	
	}



// inverted diagonal !!!
void kernel_spotrf_strsv_1x1_lib4(int kmax, float *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	__m128
		zeros, ones,
		a_00,
		b_00_10;
	
	zeros = _mm_set_ss( 0.0 );

	a_00 = _mm_load_ss( &A[0+lda*0] );
	if( _mm_comile_ss ( a_00, zeros ) ) { *info = 1; return; }
	a_00 = _mm_sqrt_ss( a_00 );
	ones = _mm_set_ss( 1.0 );
	a_00 = _mm_div_ss( ones, a_00 );
	_mm_store_ss( &A[0+lda*0], a_00 );
	
	if(kmax<=0)
		return;
	
	// strsv


	a_00 = _mm_shuffle_ps( a_00, a_00, 0 );
	
	int k, kna;
	
	float
		*AA;
	
	AA = A + 1;
	k = 0;

	// clean up unaligned stuff at the beginning
	kna = 3;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		
		b_00_10 = _mm_load_ps( &AA[0+lda*0] );

		b_00_10 = _mm_mul_ps( b_00_10, a_00 );
		_mm_store_ps( &AA[0+lda*0], b_00_10 );

		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00_10 = _mm_load_ss( &AA[lda*0] );

		b_00_10 = _mm_mul_ss( b_00_10, a_00 );
		_mm_store_ss( &AA[lda*0], b_00_10 );

		AA += 1;
		}
	
	}

