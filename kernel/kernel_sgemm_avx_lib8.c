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
#include <immintrin.h>  // AVX



// normal-transposed, 16x4 with data packed in 4
void kernel_sgemm_pp_nt_16x4_lib8(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256
		temp,
		a_07, a_8f,
		b_03, b_0, b_1, b_2, b_3,
		c_00, c_01, c_02, c_03,
		c_80, c_81, c_82, c_83;
	
	b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] ); // prefetch

	a_07 = _mm256_load_ps( &A0[0] );
	a_8f = _mm256_load_ps( &A1[0] );

	c_00 = _mm256_setzero_ps();
	c_01 = _mm256_setzero_ps();
	c_02 = _mm256_setzero_ps();
	c_03 = _mm256_setzero_ps();
	c_80 = _mm256_setzero_ps();
	c_81 = _mm256_setzero_ps();
	c_82 = _mm256_setzero_ps();
	c_83 = _mm256_setzero_ps();

	k = 0;
	for(; k<kmax-3; k+=4)
		{
		
/*		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );*/
		
/*		a_07 = _mm256_load_ps( &A0[0] );*/
/*		a_8f = _mm256_load_ps( &A1[0] );*/
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[8] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[8] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[8] );
		c_83 = _mm256_add_ps( c_83, temp );


		
		
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[16] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[16] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[16] );
		c_83 = _mm256_add_ps( c_83, temp );



		
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[24] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[24] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[24] );
		c_83 = _mm256_add_ps( c_83, temp );


		
		
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[32] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[32] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[32] );
		c_83 = _mm256_add_ps( c_83, temp );


		A0 += 32;
		A1 += 32;
		B  += 32;

		}
	if(kmax%4>=2)
		{
		
/*		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );*/
		
/*		a_07 = _mm256_load_ps( &A0[0] );*/
/*		a_8f = _mm256_load_ps( &A1[0] );*/
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[8] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[8] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[8] );
		c_83 = _mm256_add_ps( c_83, temp );


		
		
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[16] );
		
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		a_07 = _mm256_load_ps( &A0[16] );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		a_8f = _mm256_load_ps( &A1[16] );
		c_83 = _mm256_add_ps( c_83, temp );


		A0 += 16;
		A1 += 16;
		B  += 16;
		
		}
	if(kmax%2==1)
		{
		
/*		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );*/
		
/*		a_07 = _mm256_load_ps( &A0[0] );*/
/*		a_8f = _mm256_load_ps( &A1[0] );*/
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_8f, b_0 );
		c_80 = _mm256_add_ps( c_80, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_8f, b_1 );
		c_81 = _mm256_add_ps( c_81, temp );
		
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_8f, b_2 );
		c_82 = _mm256_add_ps( c_82, temp );
	
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_8f, b_3 );
		c_83 = _mm256_add_ps( c_83, temp );

/*		A0 += 8;*/
/*		A1 += 8;*/
/*		B  += 8;*/
		
		}

	__m256
		d_00, d_01, d_02, d_03,
		d_80, d_81, d_82, d_83;

	if(alg==0)
		{
		_mm256_store_ps( &C0[0+ldc*0], c_00 );
		_mm256_store_ps( &C0[0+ldc*1], c_01 );
		_mm256_store_ps( &C0[0+ldc*2], c_02 );
		_mm256_store_ps( &C0[0+ldc*3], c_03 );
		_mm256_store_ps( &C1[0+ldc*0], c_80 );
		_mm256_store_ps( &C1[0+ldc*1], c_81 );
		_mm256_store_ps( &C1[0+ldc*2], c_82 );
		_mm256_store_ps( &C1[0+ldc*3], c_83 );
		}
	else
		{
		d_00 = _mm256_load_ps( &C0[0+ldc*0] );
		d_01 = _mm256_load_ps( &C0[0+ldc*1] );
		d_02 = _mm256_load_ps( &C0[0+ldc*2] );
		d_03 = _mm256_load_ps( &C0[0+ldc*3] );
		d_80 = _mm256_load_ps( &C1[0+ldc*0] );
		d_81 = _mm256_load_ps( &C1[0+ldc*1] );
		d_82 = _mm256_load_ps( &C1[0+ldc*2] );
		d_83 = _mm256_load_ps( &C1[0+ldc*3] );
		
		if(alg==1)
			{
			d_00 = _mm256_add_ps( d_00, c_00 );
			d_01 = _mm256_add_ps( d_01, c_01 );
			d_02 = _mm256_add_ps( d_02, c_02 );
			d_03 = _mm256_add_ps( d_03, c_03 );
			d_80 = _mm256_add_ps( d_80, c_80 );
			d_81 = _mm256_add_ps( d_81, c_81 );
			d_82 = _mm256_add_ps( d_82, c_82 );
			d_83 = _mm256_add_ps( d_83, c_83 );
			}
		else // alg == -1
			{
			d_00 = _mm256_sub_ps( d_00, c_00 );
			d_01 = _mm256_sub_ps( d_01, c_01 );
			d_02 = _mm256_sub_ps( d_02, c_02 );
			d_03 = _mm256_sub_ps( d_03, c_03 );
			d_80 = _mm256_sub_ps( d_80, c_80 );
			d_81 = _mm256_sub_ps( d_81, c_81 );
			d_82 = _mm256_sub_ps( d_82, c_82 );
			d_83 = _mm256_sub_ps( d_83, c_83 );
			}

		_mm256_store_ps( &C0[0+ldc*0], d_00 );
		_mm256_store_ps( &C0[0+ldc*1], d_01 );
		_mm256_store_ps( &C0[0+ldc*2], d_02 );
		_mm256_store_ps( &C0[0+ldc*3], d_03 );
		_mm256_store_ps( &C1[0+ldc*0], d_80 );
		_mm256_store_ps( &C1[0+ldc*1], d_81 );
		_mm256_store_ps( &C1[0+ldc*2], d_82 );
		_mm256_store_ps( &C1[0+ldc*3], d_83 );
		}

	}



// normal-transposed, 8x4 with data packed in 4
void kernel_sgemm_pp_nt_8x4_lib8(int kmax, float *A0, float *B, float *C0, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256
		temp,
		a_07,
		b_03, b_0, b_1, b_2, b_3,
		c_00, c_01, c_02, c_03;
	
	c_00 = _mm256_setzero_ps();
	c_01 = _mm256_setzero_ps();
	c_02 = _mm256_setzero_ps();
	c_03 = _mm256_setzero_ps();

	k = 0;
	for(; k<kmax-3; k+=4)
		{
		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );
		
		a_07 = _mm256_load_ps( &A0[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[8] );
		
		a_07 = _mm256_load_ps( &A0[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );



		b_03 = _mm256_broadcast_ps( (__m128 *) &B[16] );
		
		a_07 = _mm256_load_ps( &A0[16] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[24] );
		
		a_07 = _mm256_load_ps( &A0[24] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		A0 += 32;
		B  += 32;

		}
	if(kmax%4>=2)
		{
		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );
		
		a_07 = _mm256_load_ps( &A0[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[8] );
		
		a_07 = _mm256_load_ps( &A0[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );

		A0 += 16;
		B  += 16;
		
		}
	if(kmax%2==1)
		{
		
		b_03 = _mm256_broadcast_ps( (__m128 *) &B[0] );
		
		a_07 = _mm256_load_ps( &A0[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_07, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_07, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_07, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_07, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );

/*		A0 += 8;*/
/*		B  += 8;*/
		
		}

	__m256
		d_00, d_01, d_02, d_03;

	if(alg==0)
		{
		_mm256_store_ps( &C0[0+ldc*0], c_00 );
		_mm256_store_ps( &C0[0+ldc*1], c_01 );
		_mm256_store_ps( &C0[0+ldc*2], c_02 );
		_mm256_store_ps( &C0[0+ldc*3], c_03 );
		}
	else
		{
		d_00 = _mm256_load_ps( &C0[0+ldc*0] );
		d_01 = _mm256_load_ps( &C0[0+ldc*1] );
		d_02 = _mm256_load_ps( &C0[0+ldc*2] );
		d_03 = _mm256_load_ps( &C0[0+ldc*3] );
		
		if(alg==1)
			{
			d_00 = _mm256_add_ps( d_00, c_00 );
			d_01 = _mm256_add_ps( d_01, c_01 );
			d_02 = _mm256_add_ps( d_02, c_02 );
			d_03 = _mm256_add_ps( d_03, c_03 );
			}
		else // alg == -1
			{
			d_00 = _mm256_sub_ps( d_00, c_00 );
			d_01 = _mm256_sub_ps( d_01, c_01 );
			d_02 = _mm256_sub_ps( d_02, c_02 );
			d_03 = _mm256_sub_ps( d_03, c_03 );
			}

		_mm256_store_ps( &C0[0+ldc*0], d_00 );
		_mm256_store_ps( &C0[0+ldc*1], d_01 );
		_mm256_store_ps( &C0[0+ldc*2], d_02 );
		_mm256_store_ps( &C0[0+ldc*3], d_03 );
		}

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_sgemm_pp_nt_4x4_lib8(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1, c_03_2, c_03_3,
		a_03,
		b_0, b_1; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();
	c_03_2 = _mm_setzero_ps();
	c_03_3 = _mm_setzero_ps();

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );



		b_0 = _mm_load_ps( &B[8] );
		
		a_03 = _mm_load_ps( &A[8] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );


		
		b_0 = _mm_load_ps( &B[16] );
		
		a_03 = _mm_load_ps( &A[16] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );



		b_0 = _mm_load_ps( &B[24] );
		
		a_03 = _mm_load_ps( &A[24] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );


		A += 32;
		B += 32;

		}
	
	for(; k<kmax; k++)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );
		
		A += 8;
		B += 8;
		
		}

	__m128
		d_03_0, d_03_1, d_03_2, d_03_3;

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], c_03_0 );
		_mm_store_ps( &C[0+ldc*1], c_03_1 );
		_mm_store_ps( &C[0+ldc*2], c_03_2 );
		_mm_store_ps( &C[0+ldc*3], c_03_3 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		d_03_3 = _mm_load_ps( &C[0+ldc*3] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_add_ps( d_03_2, c_03_2 );
		d_03_3 = _mm_add_ps( d_03_3, c_03_3 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		_mm_store_ps( &C[0+ldc*3], d_03_3 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		d_03_3 = _mm_load_ps( &C[0+ldc*3] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_sub_ps( d_03_2, c_03_2 );
		d_03_3 = _mm_sub_ps( d_03_3, c_03_3 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		_mm_store_ps( &C[0+ldc*3], d_03_3 );
		}

	}

