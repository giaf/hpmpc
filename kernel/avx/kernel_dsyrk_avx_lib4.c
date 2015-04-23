/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
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

#include "../../include/block_size.h"



// normal-transposed, 8x4 with data packed in 4
void kernel_dsyrk_nt_8x4_lib4(int kadd, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg)
	{
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31,
		c_40_51_62_73, c_41_50_63_72, c_43_52_61_70, c_42_53_60_71;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A0[0] );
	a_4567        = _mm256_load_pd( &A1[0] );
	b_0123        = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();
	c_40_51_62_73 = _mm256_setzero_pd();
	c_41_50_63_72 = _mm256_setzero_pd();
	c_43_52_61_70 = _mm256_setzero_pd();
	c_42_53_60_71 = _mm256_setzero_pd();


	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kadd%2==1)
		{
		
/*printf("\nk0 = ciao %d\n", (d_ncl-kadd%d_ncl)%d_ncl);*/

		ab_tmp0       = _mm256_mul_pd( a_0123, b_0123 );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0123 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch*/
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_tmp0 );
		c_40_51_62_73 = _mm256_add_pd( c_40_51_62_73, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1032 );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_tmp0 );
		c_41_50_63_72 = _mm256_add_pd( c_41_50_63_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3210 );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_tmp0 );
		c_43_52_61_70 = _mm256_add_pd( c_43_52_61_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2301 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_tmp0 );
		c_42_53_60_71 = _mm256_add_pd( c_42_53_60_71, ab_tmp1 );
		
		A0 += 4; // keep it !!!
		A1 += 4; // keep it !!!
		B  += 4; // keep it !!!

		}

	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_40_50_62_72, c_41_51_63_73, c_42_52_60_70, c_43_53_61_71,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33,
		d_40_50_60_70, d_41_51_61_71, d_42_52_62_72, d_43_53_63_73;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	c_40_50_62_72 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0xa );
	c_41_51_63_73 = _mm256_blend_pd( c_40_51_62_73, c_41_50_63_72, 0x5 );
	c_42_52_60_70 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0xa );
	c_43_53_61_71 = _mm256_blend_pd( c_42_53_60_71, c_43_52_61_70, 0x5 );
	
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
	c_40_50_60_70 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0xc );
	c_42_52_62_72 = _mm256_blend_pd( c_40_50_62_72, c_42_52_60_70, 0x3 );
	c_41_51_61_71 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0xc );
	c_43_53_63_73 = _mm256_blend_pd( c_41_51_63_73, c_43_53_61_71, 0x3 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &D0[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &D0[0+ldc*3] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );
		c_02_12_22_32 = _mm256_blend_pd( c_02_12_22_32, d_02_12_22_32, 0x3 );
		c_03_13_23_33 = _mm256_blend_pd( c_03_13_23_33, d_03_13_23_33, 0x7 );

		_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
		_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );
		_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );
		_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
		_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
		_mm256_store_pd( &D1[0+ldc*2], c_42_52_62_72 );
		_mm256_store_pd( &D1[0+ldc*3], c_43_53_63_73 );
		}
	else 
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &C0[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &C0[0+ldc*3] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		d_42_52_62_72 = _mm256_load_pd( &C1[0+ldc*2] );
		d_43_53_63_73 = _mm256_load_pd( &C1[0+ldc*3] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
			d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
			d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
			d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
			d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
			d_42_52_62_72 = _mm256_add_pd( d_42_52_62_72, c_42_52_62_72 );
			d_43_53_63_73 = _mm256_add_pd( d_43_53_63_73, c_43_53_63_73 );
			}
		else // C -= A * B'
			{
			d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
			d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
			d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
			d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
			d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
			d_42_52_62_72 = _mm256_sub_pd( d_42_52_62_72, c_42_52_62_72 );
			d_43_53_63_73 = _mm256_sub_pd( d_43_53_63_73, c_43_53_63_73 );
			}

		c_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );
		c_02_12_22_32 = _mm256_load_pd( &D0[0+ldc*2] );
		c_03_13_23_33 = _mm256_load_pd( &D0[0+ldc*3] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );
		d_02_12_22_32 = _mm256_blend_pd( d_02_12_22_32, c_02_12_22_32, 0x3 );
		d_03_13_23_33 = _mm256_blend_pd( d_03_13_23_33, c_03_13_23_33, 0x7 );

		_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &D0[0+ldc*2], d_02_12_22_32 );
		_mm256_store_pd( &D0[0+ldc*3], d_03_13_23_33 );
		_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
		_mm256_store_pd( &D1[0+ldc*2], d_42_52_62_72 );
		_mm256_store_pd( &D1[0+ldc*3], d_43_53_63_73 );
		}

	}



// normal-transposed, 8x2 with data packed in 4
void kernel_dsyrk_nt_8x2_lib4(int kmax, double *A0, int sda, double *B, double *C0, int sdc, double *D0, int sdd, int alg)
	{
	
	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0101, b_1010,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_11_20_31, c_01_10_21_30,
		c_40_51_60_71, c_41_50_61_70;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A0[0] );
	a_4567 = _mm256_load_pd( &A1[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	c_40_51_60_71 = _mm256_setzero_pd();
	c_41_50_61_70 = _mm256_setzero_pd();


	for(k=0; k<kmax-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );


/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kmax%4>=2)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kmax%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0101 );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0101 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_tmp0 );
		c_40_51_60_71 = _mm256_add_pd( c_40_51_60_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1010 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_tmp0 );
		c_41_50_61_70 = _mm256_add_pd( c_41_50_61_70, ab_tmp1 );
		
		}

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		c_40_50_60_70, c_41_51_61_71,
		d_00_10_20_30, d_01_11_21_31,
		d_40_50_60_70, d_41_51_61_71;

	c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
	c_40_50_60_70 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0xa );
	c_41_51_61_71 = _mm256_blend_pd( c_40_51_60_71, c_41_50_61_70, 0x5 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );

		_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
		_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
		_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
		}
	else 
		{
		d_00_10_20_30 = _mm256_load_pd( &C0[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C0[0+ldc*1] );
		d_40_50_60_70 = _mm256_load_pd( &C1[0+ldc*0] );
		d_41_51_61_71 = _mm256_load_pd( &C1[0+ldc*1] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
			d_40_50_60_70 = _mm256_add_pd( d_40_50_60_70, c_40_50_60_70 );
			d_41_51_61_71 = _mm256_add_pd( d_41_51_61_71, c_41_51_61_71 );
			}
		else // C -= A * B'
			{
			d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
			d_40_50_60_70 = _mm256_sub_pd( d_40_50_60_70, c_40_50_60_70 );
			d_41_51_61_71 = _mm256_sub_pd( d_41_51_61_71, c_41_51_61_71 );
			}

		c_01_11_21_31 = _mm256_load_pd( &D0[0+ldc*1] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );

		_mm256_store_pd( &D0[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D0[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &D1[0+ldc*0], d_40_50_60_70 );
		_mm256_store_pd( &D1[0+ldc*1], d_41_51_61_71 );
		}

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dsyrk_nt_4x4_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0123, //A_0123,
		b_0123, b_1032, b_3210, b_2301,
		ab_temp, // temporary results
		c_00_11_22_33, c_01_10_23_32, c_03_12_21_30, c_02_13_20_31;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A[0] );
	b_0123        = _mm256_load_pd( &B[0] );

	// zero registers
	c_00_11_22_33 = _mm256_setzero_pd();
	c_01_10_23_32 = _mm256_setzero_pd();
	c_03_12_21_30 = _mm256_setzero_pd();
	c_02_13_20_31 = _mm256_setzero_pd();


	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[16] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
		b_0123        = _mm256_load_pd( &B[8] ); // prefetch 
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0123 );
		c_00_11_22_33 = _mm256_add_pd( c_00_11_22_33, ab_temp );
		b_1032        = _mm256_shuffle_pd( b_0123, b_0123, 0x5 );
/*		b_0123        = _mm256_load_pd( &B[4] ); // prefetch */
		ab_temp       = _mm256_mul_pd( a_0123, b_1032 );
		c_01_10_23_32 = _mm256_add_pd( c_01_10_23_32, ab_temp );
		b_3210        = _mm256_permute2f128_pd( b_1032, b_1032, 0x1 );
		ab_temp       = _mm256_mul_pd( a_0123, b_3210 );
		c_03_12_21_30 = _mm256_add_pd( c_03_12_21_30, ab_temp );
		b_2301        = _mm256_shuffle_pd( b_3210, b_3210, 0x5 );
		ab_temp       = _mm256_mul_pd( a_0123, b_2301 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch */
		c_02_13_20_31 = _mm256_add_pd( c_02_13_20_31, ab_temp );
		
		A += 4; // keep it !!!
		B += 4; // keep it !!!
		
		}



	__m256d
		c_00_10_22_32, c_01_11_23_33, c_02_12_20_30, c_03_13_21_31,
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		d_00_10_20_30, d_01_11_21_31, d_02_12_22_32, d_03_13_23_33;

	c_00_10_22_32 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0xa );
	c_01_11_23_33 = _mm256_blend_pd( c_00_11_22_33, c_01_10_23_32, 0x5 );
	c_02_12_20_30 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0xa );
	c_03_13_21_31 = _mm256_blend_pd( c_02_13_20_31, c_03_12_21_30, 0x5 );
	
	c_00_10_20_30 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0xc );
	c_02_12_22_32 = _mm256_blend_pd( c_00_10_22_32, c_02_12_20_30, 0x3 );
	c_01_11_21_31 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0xc );
	c_03_13_23_33 = _mm256_blend_pd( c_01_11_23_33, c_03_13_21_31, 0x3 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &D[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &D[0+ldc*3] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );
		c_02_12_22_32 = _mm256_blend_pd( c_02_12_22_32, d_02_12_22_32, 0x3 );
		c_03_13_23_33 = _mm256_blend_pd( c_03_13_23_33, d_03_13_23_33, 0x7 );

		_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
		_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );
		_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );
		}
	else 
		{
		d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
		d_02_12_22_32 = _mm256_load_pd( &C[0+ldc*2] );
		d_03_13_23_33 = _mm256_load_pd( &C[0+ldc*3] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
			d_02_12_22_32 = _mm256_add_pd( d_02_12_22_32, c_02_12_22_32 );
			d_03_13_23_33 = _mm256_add_pd( d_03_13_23_33, c_03_13_23_33 );
			}
		else // C -= A * B'
			{
			d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
			d_02_12_22_32 = _mm256_sub_pd( d_02_12_22_32, c_02_12_22_32 );
			d_03_13_23_33 = _mm256_sub_pd( d_03_13_23_33, c_03_13_23_33 );
			}

		c_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );
		c_02_12_22_32 = _mm256_load_pd( &D[0+ldc*2] );
		c_03_13_23_33 = _mm256_load_pd( &D[0+ldc*3] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );
		d_02_12_22_32 = _mm256_blend_pd( d_02_12_22_32, c_02_12_22_32, 0x3 );
		d_03_13_23_33 = _mm256_blend_pd( d_03_13_23_33, c_03_13_23_33, 0x7 );

		_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
		_mm256_store_pd( &D[0+ldc*2], d_02_12_22_32 );
		_mm256_store_pd( &D[0+ldc*3], d_03_13_23_33 );
		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_dsyrk_nt_4x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m256d
		a_0123,
		b_0101, b_1010,
		ab_temp, // temporary results
		c_00_11_20_31, c_01_10_21_30, C_00_11_20_31, C_01_10_21_30;
	
	// prefetch
	a_0123 = _mm256_load_pd( &A[0] );
	b_0101 = _mm256_broadcast_pd( (__m128d *) &B[0] );

	// zero registers
	c_00_11_20_31 = _mm256_setzero_pd();
	c_01_10_21_30 = _mm256_setzero_pd();
	C_00_11_20_31 = _mm256_setzero_pd();
	C_01_10_21_30 = _mm256_setzero_pd();

	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		C_00_11_20_31 = _mm256_add_pd( C_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		C_01_10_21_30 = _mm256_add_pd( C_01_10_21_30, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0101 );
		c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, ab_temp );
		b_1010        = _mm256_shuffle_pd( b_0101, b_0101, 0x5 );
/*		b_0101        = _mm256_broadcast_pd( (__m128d *) &B[4] ); // prefetch*/
		ab_temp       = _mm256_mul_pd( a_0123, b_1010 );
/*		a_0123        = _mm256_load_pd( &A[4] ); // prefetch*/
		c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, ab_temp );
		
		A += 4; // keep it !!!
		B += 4; // keep it !!!

		}
		
	c_00_11_20_31 = _mm256_add_pd( c_00_11_20_31, C_00_11_20_31 );
	c_01_10_21_30 = _mm256_add_pd( c_01_10_21_30, C_01_10_21_30 );

	__m256d
		c_00_10_20_30, c_01_11_21_31,
		d_00_10_20_30, d_01_11_21_31;

	c_00_10_20_30 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0xa );
	c_01_11_21_31 = _mm256_blend_pd( c_00_11_20_31, c_01_10_21_30, 0x5 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );

		c_01_11_21_31 = _mm256_blend_pd( c_01_11_21_31, d_01_11_21_31, 0x1 );

		_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
		}
	else 
		{
		d_00_10_20_30 = _mm256_load_pd( &C[0+ldc*0] );
		d_01_11_21_31 = _mm256_load_pd( &C[0+ldc*1] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10_20_30 = _mm256_add_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_add_pd( d_01_11_21_31, c_01_11_21_31 );
			}
		else // C -= A * B'
			{
			d_00_10_20_30 = _mm256_sub_pd( d_00_10_20_30, c_00_10_20_30 );
			d_01_11_21_31 = _mm256_sub_pd( d_01_11_21_31, c_01_11_21_31 );
			}

		c_01_11_21_31 = _mm256_load_pd( &D[0+ldc*1] );

		d_01_11_21_31 = _mm256_blend_pd( d_01_11_21_31, c_01_11_21_31, 0x1 );

		_mm256_store_pd( &D[0+ldc*0], d_00_10_20_30 );
		_mm256_store_pd( &D[0+ldc*1], d_01_11_21_31 );
		}


	}



// normal-transposed, 2x2 with data packed in 4
void kernel_dsyrk_nt_2x2_lib4(int kadd, double *A, double *B, double *C, double *D, int alg)
	{
	
	const int bs = 4;
	const int ldc = bs;
	
	int k;
	
	__m128d
		a_01,
		b_01, b_10,
		ab_temp, // temporary results
		c_00_11, c_01_10, C_00_11, C_01_10;
	
	// prefetch
	a_01 = _mm_load_pd( &A[0] );
	b_01 = _mm_load_pd( &B[0] );

	// zero registers
	c_00_11 = _mm_setzero_pd();
	c_01_10 = _mm_setzero_pd();
	C_00_11 = _mm_setzero_pd();
	C_01_10 = _mm_setzero_pd();

	for(k=0; k<kadd-3; k+=4)
		{
		
/*	__builtin_prefetch( A+32 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[4] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[4] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		
/*	__builtin_prefetch( A+40 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[8] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[8] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );


/*	__builtin_prefetch( A+48 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[12] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[12] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );


/*	__builtin_prefetch( A+56 );*/
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[16] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[16] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[4] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[4] ); // prefetch
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		C_00_11 = _mm_add_pd( C_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
		b_01    = _mm_load_pd( &B[8] ); // prefetch
		ab_temp = _mm_mul_pd( a_01, b_10 );
		a_01    = _mm_load_pd( &A[8] ); // prefetch
		C_01_10 = _mm_add_pd( C_01_10, ab_temp );
		
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp = _mm_mul_pd( a_01, b_01 );
		c_00_11 = _mm_add_pd( c_00_11, ab_temp );
		b_10    = _mm_shuffle_pd( b_01, b_01, 0x5 );
/*		b_01    = _mm_load_pd( &B[4] ); // prefetch*/
		ab_temp = _mm_mul_pd( a_01, b_10 );
/*		a_01    = _mm_load_pd( &A[4] ); // prefetch*/
		c_01_10 = _mm_add_pd( c_01_10, ab_temp );
		
		A += 4; // keep it !!!
		B += 4; // keep it !!!

		}
		


	c_00_11 = _mm_add_pd( c_00_11, C_00_11 );
	c_01_10 = _mm_add_pd( c_01_10, C_01_10 );

	__m128d
		c_00_10, c_01_11,
		d_00_10, d_01_11;

	c_00_10 = _mm_blend_pd( c_00_11, c_01_10, 0x2 );
	c_01_11 = _mm_blend_pd( c_00_11, c_01_10, 0x1 );
		
	if(alg==0) // C = A * B'
		{
		d_01_11 = _mm_load_pd( &D[0+ldc*1] );

		c_01_11 = _mm_blend_pd( c_01_11, d_01_11, 0x1 );

		_mm_store_pd( &D[0+ldc*0], c_00_10 );
		_mm_store_pd( &D[0+ldc*1], c_01_11 );
		}
	else 
		{
		d_00_10 = _mm_load_pd( &C[0+ldc*0] );
		d_01_11 = _mm_load_pd( &C[0+ldc*1] );
		
		if(alg==1) // C += A * B'
			{
			d_00_10 = _mm_add_pd( d_00_10, c_00_10 );
			d_01_11 = _mm_add_pd( d_01_11, c_01_11 );
			}
		else // C -= A * B'
			{
			d_00_10 = _mm_sub_pd( d_00_10, c_00_10 );
			d_01_11 = _mm_sub_pd( d_01_11, c_01_11 );
			}

		c_01_11 = _mm_load_pd( &D[0+ldc*1] );

		d_01_11 = _mm_blend_pd( d_01_11, c_01_11, 0x1 );

		_mm_store_pd( &D[0+ldc*0], d_00_10 );
		_mm_store_pd( &D[0+ldc*1], d_01_11 );
		}

	}



void kernel_dsyrk_nn_4x4_lib4(int kadd, double *A, double *B, int sdb, double *C, double *D, int alg)
	{

	if(kadd<=0)
		return;

	const int bs = 4;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, 
		c_10=0, c_11=0, 
		c_20=0, c_21=0, c_22=0, 
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		b_2 = B[1+bs*2];
		b_3 = B[1+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		b_2 = B[2+bs*2];
		b_3 = B[2+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		b_2 = B[3+bs*2];
		b_3 = B[3+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		b_2 = B[0+bs*2];
		b_3 = B[0+bs*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		A += 4;
		B += 1;

		}

	double
		d_00,
		d_10, d_11,
		d_20, d_21, d_22,
		d_30, d_31, d_32, d_33;
	
	if(alg==0) // C = A * B'
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;

		D[2+bs*2] = c_22;
		D[3+bs*2] = c_32;

		D[3+bs*3] = c_33;
		}
	else 
		{
		d_00 = C[0+bs*0];
		d_10 = C[1+bs*0];
		d_20 = C[2+bs*0];
		d_30 = C[3+bs*0];
		
		d_11 = C[1+bs*1];
		d_21 = C[2+bs*1];
		d_31 = C[3+bs*1];
		
		d_22 = C[2+bs*2];
		d_32 = C[3+bs*2];
		
		d_33 = C[3+bs*3];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;

			d_22 += c_22;
			d_32 += c_32;

			d_33 += c_33;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;

			d_22 -= c_22;
			d_32 -= c_32;

			d_33 -= c_33;
			}

		D[0+bs*0] = d_00;
		D[1+bs*0] = d_10;
		D[2+bs*0] = d_20;
		D[3+bs*0] = d_30;

		D[1+bs*1] = d_11;
		D[2+bs*1] = d_21;
		D[3+bs*1] = d_31;

		D[2+bs*2] = d_22;
		D[3+bs*2] = d_32;

		D[3+bs*3] = d_33;
		}
	
	}



void kernel_dsyrk_nn_4x2_lib4(int kadd, double *A, double *B, int sdb, double *C, double *D, int alg)
	{

	if(kadd<=0)
		return;

	const int bs = 4;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, 
		c_10=0, c_11=0, 
		c_20=0, c_21=0,  
		c_30=0, c_31=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		a_2 = A[2+bs*1];
		a_3 = A[3+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		a_2 = A[2+bs*2];
		a_3 = A[3+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		a_2 = A[2+bs*3];
		a_3 = A[3+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		a_2 = A[2+bs*0];
		a_3 = A[3+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		A += 4;
		B += 1;

		}
	double
		d_00,
		d_10, d_11,
		d_20, d_21,
		d_30, d_31;
	
	if(alg==0) // C = A * B'
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;
		D[2+bs*0] = c_20;
		D[3+bs*0] = c_30;

		D[1+bs*1] = c_11;
		D[2+bs*1] = c_21;
		D[3+bs*1] = c_31;
		}
	else 
		{
		d_00 = C[0+bs*0];
		d_10 = C[1+bs*0];
		d_20 = C[2+bs*0];
		d_30 = C[3+bs*0];
		
		d_11 = C[1+bs*1];
		d_21 = C[2+bs*1];
		d_31 = C[3+bs*1];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;
			}

		D[0+bs*0] = d_00;
		D[1+bs*0] = d_10;
		D[2+bs*0] = d_20;
		D[3+bs*0] = d_30;

		D[1+bs*1] = d_11;
		D[2+bs*1] = d_21;
		D[3+bs*1] = d_31;
		}

	}



void kernel_dsyrk_nn_2x2_lib4(int kadd, double *A, double *B, int sdb, double *C, double *D, int alg)
	{

	if(kadd<=0)
		return;

	const int bs = 4;

	int k;

	double
		a_0, a_1,
		b_0, b_1,
		c_00=0, 
		c_10=0, c_11=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+bs*1];
		a_1 = A[1+bs*1];
		
		b_0 = B[1+bs*0];
		b_1 = B[1+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+bs*2];
		a_1 = A[1+bs*2];
		
		b_0 = B[2+bs*0];
		b_1 = B[2+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+bs*3];
		a_1 = A[1+bs*3];
		
		b_0 = B[3+bs*0];
		b_1 = B[3+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;
		
		
		A += 16;
		B += 4*sdb;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+bs*0];
		a_1 = A[1+bs*0];
		
		b_0 = B[0+bs*0];
		b_1 = B[0+bs*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		A += 4;
		B += 1;

		}
	
	double
		d_00,
		d_10, d_11;
	
	if(alg==0) // C = A * B'
		{
		D[0+bs*0] = c_00;
		D[1+bs*0] = c_10;

		D[1+bs*1] = c_11;
		}
	else 
		{
		d_00 = C[0+bs*0];
		d_10 = C[1+bs*0];
		
		d_11 = C[1+bs*1];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;

			d_11 += c_11;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;

			d_11 -= c_11;
			}

		D[0+bs*0] = d_00;
		D[1+bs*0] = d_10;

		D[1+bs*1] = d_11;
		}

	}



// Al and Ar are the diagonal of two matrices
void kernel_dsyrk_diag_left_right_4_lib4(int kmax, double *Al, double *Ar, double *B, double *C, double *D, int alg)
	{

	if(kmax<=0)
		{
		return;
		}
	
	// assume kmax to be multiple of 4
	
	const int bs = 4;

	int k;

	double
		a_r,
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_0, c_1, c_2, c_3;
		
	if(alg==-1)
		{
		a_0 = - Al[0];
		a_1 = - Al[1];
		a_2 = - Al[2];
		a_3 = - Al[3];
		}
	else
		{
		a_0 = Al[0];
		a_1 = Al[1];
		a_2 = Al[2];
		a_3 = Al[3];
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-4; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;
			c_3 = a_3 * b_3 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;
			c_3 = a_3 * b_3 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
			D[3+bs*1] = c_3;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;
			c_3 = a_3 * b_3 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
			D[3+bs*2] = c_3;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			b_3 = B[3+bs*3];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;
			c_3 = a_3 * b_3 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
			D[3+bs*3] = c_3;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = a_0 * b_0 * a_r;
		c_1 = a_1 * b_1 * a_r;
		c_2 = a_2 * b_2 * a_r;
		c_3 = a_3 * b_3 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_1 = a_1 * b_1 * a_r;
		c_2 = a_2 * b_2 * a_r;
		c_3 = a_3 * b_3 * a_r;

		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;


		a_r = Ar[2];
		
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_2 = a_2 * b_2 * a_r;
		c_3 = a_3 * b_3 * a_r;

		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;


		a_r = Ar[3];
		
		b_3 = B[3+bs*3];
		
		c_3 = a_3 * b_3 * a_r;

		D[3+bs*3] = c_3;

		}
	else
		{

		for(k=0; k<kmax-4; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			b_3 = B[3+bs*0];
			
			c_0 = C[0+bs*0] + a_0 * b_0 * a_r;
			c_1 = C[1+bs*0] + a_1 * b_1 * a_r;
			c_2 = C[2+bs*0] + a_2 * b_2 * a_r;
			c_3 = C[3+bs*0] + a_3 * b_3 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
			D[3+bs*0] = c_3;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			b_3 = B[3+bs*1];
			
			c_0 = C[0+bs*1] + a_0 * b_0 * a_r;
			c_1 = C[1+bs*1] + a_1 * b_1 * a_r;
			c_2 = C[2+bs*1] + a_2 * b_2 * a_r;
			c_3 = C[3+bs*1] + a_3 * b_3 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
			D[3+bs*1] = c_3;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			b_3 = B[3+bs*2];
			
			c_0 = C[0+bs*2] + a_0 * b_0 * a_r;
			c_1 = C[1+bs*2] + a_1 * b_1 * a_r;
			c_2 = C[2+bs*2] + a_2 * b_2 * a_r;
			c_3 = C[3+bs*2] + a_3 * b_3 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
			D[3+bs*2] = c_3;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			b_3 = B[3+bs*3];
			
			c_0 = C[0+bs*3] + a_0 * b_0 * a_r;
			c_1 = C[1+bs*3] + a_1 * b_1 * a_r;
			c_2 = C[2+bs*3] + a_2 * b_2 * a_r;
			c_3 = C[3+bs*3] + a_3 * b_3 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
			D[3+bs*3] = c_3;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];

		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		b_3 = B[3+bs*0];
		
		c_0 = C[0+bs*0] + a_0 * b_0 * a_r;
		c_1 = C[1+bs*0] + a_1 * b_1 * a_r;
		c_2 = C[2+bs*0] + a_2 * b_2 * a_r;
		c_3 = C[3+bs*0] + a_3 * b_3 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;
		D[3+bs*0] = c_3;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		b_3 = B[3+bs*1];
		
		c_1 = C[1+bs*1] + a_1 * b_1 * a_r;
		c_2 = C[2+bs*1] + a_2 * b_2 * a_r;
		c_3 = C[3+bs*1] + a_3 * b_3 * a_r;

		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;
		D[3+bs*1] = c_3;


		a_r = Ar[2];
		
		b_2 = B[2+bs*2];
		b_3 = B[3+bs*2];
		
		c_2 = C[2+bs*2] + a_2 * b_2 * a_r;
		c_3 = C[3+bs*2] + a_3 * b_3 * a_r;

		D[2+bs*2] = c_2;
		D[3+bs*2] = c_3;


		a_r = Ar[3];
		
		b_3 = B[3+bs*3];
		
		c_3 = C[3+bs*3] + a_3 * b_3 * a_r;

		D[3+bs*3] = c_3;

		}
	
	}



// Al and Ar are the diagonal of two matrices
void kernel_dsyrk_diag_left_right_3_lib4(int kmax, double *Al, double *Ar, double *B, double *C, double *D, int alg)
	{

	if(kmax<=0)
		{
		return;
		}
	
	// assume kmax to be multiple of 4
	
	const int bs = 4;

	int k;

	double
		a_r,
		a_0, a_1, a_2,
		b_0, b_1, b_2,
		c_0, c_1, c_2;
		
	if(alg==-1)
		{
		a_0 = - Al[0];
		a_1 = - Al[1];
		a_2 = - Al[2];
		}
	else
		{
		a_0 = Al[0];
		a_1 = Al[1];
		a_2 = Al[2];
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-3; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;
			c_2 = a_2 * b_2 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		
		c_0 = a_0 * b_0 * a_r;
		c_1 = a_1 * b_1 * a_r;
		c_2 = a_2 * b_2 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		
		c_1 = a_1 * b_1 * a_r;
		c_2 = a_2 * b_2 * a_r;

		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;


		a_r = Ar[2];
		
		b_2 = B[2+bs*2];
		
		c_2 = a_2 * b_2 * a_r;

		D[2+bs*2] = c_2;

		}
	else
		{

		for(k=0; k<kmax-3; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			b_2 = B[2+bs*0];
			
			c_0 = C[0*bs*0] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*0] + a_1 * b_1 * a_r;
			c_2 = C[2*bs*0] + a_2 * b_2 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
			D[2+bs*0] = c_2;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			b_2 = B[2+bs*1];
			
			c_0 = C[0*bs*1] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*1] + a_1 * b_1 * a_r;
			c_2 = C[2*bs*1] + a_2 * b_2 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
			D[2+bs*1] = c_2;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			b_2 = B[2+bs*2];
			
			c_0 = C[0*bs*2] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*2] + a_1 * b_1 * a_r;
			c_2 = C[2*bs*2] + a_2 * b_2 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
			D[2+bs*2] = c_2;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			b_2 = B[2+bs*3];
			
			c_0 = C[0*bs*3] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*3] + a_1 * b_1 * a_r;
			c_2 = C[2*bs*3] + a_2 * b_2 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
			D[2+bs*3] = c_2;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		b_2 = B[2+bs*0];
		
		c_0 = C[0+bs*0] + a_0 * b_0 * a_r;
		c_1 = C[1+bs*0] + a_1 * b_1 * a_r;
		c_2 = C[2+bs*0] + a_2 * b_2 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;
		D[2+bs*0] = c_2;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		b_2 = B[2+bs*1];
		
		c_1 = C[1+bs*1] + a_1 * b_1 * a_r;
		c_2 = C[2+bs*1] + a_2 * b_2 * a_r;

		D[1+bs*1] = c_1;
		D[2+bs*1] = c_2;


		a_r = Ar[2];
		
		b_2 = B[2+bs*2];
		
		c_2 = C[2+bs*2] + a_2 * b_2 * a_r;

		D[2+bs*2] = c_2;

		}
	
	}


// Al and Ar are the diagonal of two matrices
void kernel_dsyrk_diag_left_right_2_lib4(int kmax, double *Al, double *Ar, double *B, double *C, double *D, int alg)
	{

	if(kmax<=0)
		{
		return;
		}
	
	// assume kmax to be multiple of 4
	
	const int bs = 4;

	int k;

	double
		a_r,
		a_0, a_1,
		b_0, b_1,
		c_0, c_1;
		
	if(alg==-1)
		{
		a_0 = - Al[0];
		a_1 = - Al[1];
		}
	else
		{
		a_0 = Al[0];
		a_1 = Al[1];
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-2; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			
			c_0 = a_0 * b_0 * a_r;
			c_1 = a_1 * b_1 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_0 = a_0 * b_0 * a_r;
		c_1 = a_1 * b_1 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		
		c_1 = a_1 * b_1 * a_r;

		D[1+bs*1] = c_1;

		}
	else
		{

		for(k=0; k<kmax-2; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			b_1 = B[1+bs*0];
			
			c_0 = C[0*bs*0] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*0] + a_1 * b_1 * a_r;

			D[0+bs*0] = c_0;
			D[1+bs*0] = c_1;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			b_1 = B[1+bs*1];
			
			c_0 = C[0*bs*1] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*1] + a_1 * b_1 * a_r;

			D[0+bs*1] = c_0;
			D[1+bs*1] = c_1;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			b_1 = B[1+bs*2];
			
			c_0 = C[0*bs*2] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*2] + a_1 * b_1 * a_r;

			D[0+bs*2] = c_0;
			D[1+bs*2] = c_1;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			b_1 = B[1+bs*3];
			
			c_0 = C[0*bs*3] + a_0 * b_0 * a_r;
			c_1 = C[1*bs*3] + a_1 * b_1 * a_r;

			D[0+bs*3] = c_0;
			D[1+bs*3] = c_1;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		b_1 = B[1+bs*0];
		
		c_0 = C[0+bs*0] + a_0 * b_0 * a_r;
		c_1 = C[1+bs*0] + a_1 * b_1 * a_r;

		D[0+bs*0] = c_0;
		D[1+bs*0] = c_1;


		a_r = Ar[1];
		
		b_1 = B[1+bs*1];
		
		c_1 = C[1+bs*1] + a_1 * b_1 * a_r;

		D[1+bs*1] = c_1;

		}
	
	}


// Al and Ar are the diagonal of two matrices
void kernel_dsyrk_diag_left_right_1_lib4(int kmax, double *Al, double *Ar, double *B, double *C, double *D, int alg)
	{

	if(kmax<=0)
		{
		return;
		}
	
	// assume kmax to be multiple of 4
	
	const int bs = 4;

	int k;

	double
		a_r,
		a_0,
		b_0,
		c_0;
		
	if(alg==-1)
		{
		a_0 = - Al[0];
		}
	else
		{
		a_0 = Al[0];
		}
	
	if(alg==0)
		{
		
		for(k=0; k<kmax-1; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			
			c_0 = a_0 * b_0 * a_r;

			D[0+bs*0] = c_0;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			
			c_0 = a_0 * b_0 * a_r;

			D[0+bs*1] = c_0;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			
			c_0 = a_0 * b_0 * a_r;

			D[0+bs*2] = c_0;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			
			c_0 = a_0 * b_0 * a_r;

			D[0+bs*3] = c_0;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		
		c_0 = a_0 * b_0 * a_r;

		D[0+bs*0] = c_0;

		}
	else
		{

		for(k=0; k<kmax-2; k+=4)
			{

			a_r = Ar[0];
			
			b_0 = B[0+bs*0];
			
			c_0 = C[0*bs*0] + a_0 * b_0 * a_r;

			D[0+bs*0] = c_0;
	

			a_r = Ar[1];
			
			b_0 = B[0+bs*1];
			
			c_0 = C[0*bs*1] + a_0 * b_0 * a_r;

			D[0+bs*1] = c_0;
	

			a_r = Ar[2];
			
			b_0 = B[0+bs*2];
			
			c_0 = C[0*bs*2] + a_0 * b_0 * a_r;

			D[0+bs*2] = c_0;
	

			a_r = Ar[3];
			
			b_0 = B[0+bs*3];
			
			c_0 = C[0*bs*3] + a_0 * b_0 * a_r;

			D[0+bs*3] = c_0;
	
			Ar += 4;
			B  += 16;
			C  += 16;
			D  += 16;
			
			}

		a_r = Ar[0];
		
		b_0 = B[0+bs*0];
		
		c_0 = C[0+bs*0] + a_0 * b_0 * a_r;

		D[0+bs*0] = c_0;

		}
	
	}



