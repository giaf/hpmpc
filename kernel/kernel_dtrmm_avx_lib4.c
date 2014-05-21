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



// normal-transposed, 8x4 with data packed in 4
void kernel_dtrmm_pp_nt_8x4_lib4(int kadd, double *A0, double *A1, double *B, double *D0, double *D1, int ldc)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	int k;
	
	__m256d
		a_0123, a_4567, //A_0123,
		b_0, b_1, b_2, b_3,
		ab_tmp0, ab_tmp1, // temporary results
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33,
		c_40_50_60_70, c_41_51_61_71, c_42_52_62_72, c_43_53_63_73;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A0[0] );
	a_4567        = _mm256_load_pd( &A1[0] );
	b_0           = _mm256_broadcast_sd( &B[0] );
	b_1           = _mm256_broadcast_sd( &B[5] );
	b_2           = _mm256_broadcast_sd( &B[10] );
	b_3           = _mm256_broadcast_sd( &B[15] );

/*	__builtin_prefetch( A+32 );*/
	c_00_10_20_30 = _mm256_mul_pd( a_0123, b_0 );
	a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
	c_40_50_60_70 = _mm256_mul_pd( a_4567, b_0 );
	b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
	a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
	
	
/*	__builtin_prefetch( A+40 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
	b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
	c_01_11_21_31 = _mm256_mul_pd( a_0123, b_1 );
	a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
	c_41_51_61_71 = _mm256_mul_pd( a_4567, b_1 );
	b_1           = _mm256_broadcast_sd( &B[9] );
	a_4567        = _mm256_load_pd( &A1[8] ); // prefetch


/*	__builtin_prefetch( A+48 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
	b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
	b_1           = _mm256_broadcast_sd( &B[13] );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
	c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
	c_02_12_22_32 = _mm256_mul_pd( a_0123, b_2 );
	a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
	c_42_52_62_72 = _mm256_mul_pd( a_4567, b_2 );
	b_2           = _mm256_broadcast_sd( &B[14] );
	a_4567        = _mm256_load_pd( &A1[12] ); // prefetch


/*	__builtin_prefetch( A+56 );*/
	ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
	b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
	c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
	b_1           = _mm256_broadcast_sd( &B[17] );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
	c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
	ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
	ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
	b_2           = _mm256_broadcast_sd( &B[18] );
	c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
	c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
	c_03_13_23_33 = _mm256_mul_pd( a_0123, b_3 );
	a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
	c_43_53_63_73 = _mm256_mul_pd( a_4567, b_3 );
	a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
	b_3           = _mm256_broadcast_sd( &B[19] );
	
	A0 += 16;
	A1 += 16;
	B  += 16;

	for(k=4; k<kadd-3; k+=4)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[5] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[6] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[7] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[9] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[10] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[11] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );


	/*	__builtin_prefetch( A+48 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[13] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[14] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[12] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[12] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[15] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );


	/*	__builtin_prefetch( A+56 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[17] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[18] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[16] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[16] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[19] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );
		
		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	if(kadd%4>=2)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[5] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[6] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[7] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
		b_1           = _mm256_broadcast_sd( &B[9] );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
		b_2           = _mm256_broadcast_sd( &B[10] );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A0[8] ); // prefetch
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
		a_4567        = _mm256_load_pd( &A1[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
		b_3           = _mm256_broadcast_sd( &B[11] );
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );
		
		
		A0 += 8;
		A1 += 8;
		B  += 8;

		}

	if(kadd%2==1)
		{
		
		ab_tmp0       = _mm256_mul_pd( a_0123, b_0 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_0 );
/*		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch*/
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_tmp0 );
		c_40_50_60_70 = _mm256_add_pd( c_40_50_60_70, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_1 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_1 );
/*		b_1           = _mm256_broadcast_sd( &B[5] );*/
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_tmp0 );
		c_41_51_61_71 = _mm256_add_pd( c_41_51_61_71, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_2 );
		ab_tmp1       = _mm256_mul_pd( a_4567, b_2 );
/*		b_2           = _mm256_broadcast_sd( &B[6] );*/
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_tmp0 );
		c_42_52_62_72 = _mm256_add_pd( c_42_52_62_72, ab_tmp1 );
		ab_tmp0       = _mm256_mul_pd( a_0123, b_3 );
/*		a_0123        = _mm256_load_pd( &A0[4] ); // prefetch*/
		ab_tmp1       = _mm256_mul_pd( a_4567, b_3 );
/*		a_4567        = _mm256_load_pd( &A1[4] ); // prefetch*/
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_tmp0 );
/*		b_3           = _mm256_broadcast_sd( &B[7] );*/
		c_43_53_63_73 = _mm256_add_pd( c_43_53_63_73, ab_tmp1 );
		
		}

	_mm256_store_pd( &D0[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &D0[0+ldc*2], c_02_12_22_32 );
	_mm256_store_pd( &D0[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &D0[0+ldc*3], c_03_13_23_33 );
	_mm256_store_pd( &D1[0+ldc*0], c_40_50_60_70 );
	_mm256_store_pd( &D1[0+ldc*2], c_42_52_62_72 );
	_mm256_store_pd( &D1[0+ldc*1], c_41_51_61_71 );
	_mm256_store_pd( &D1[0+ldc*3], c_43_53_63_73 );

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_dtrmm_pp_nt_4x4_lib4(int kadd, double *A, double *B, double *D, int ldc)
	{
	
/*	if(kmax<=0)*/
/*		return;*/
	
	const int d_ncl = 2;
	
	int k;
	
	__m256d
		a_0123, //A_0123,
		b_0, b_1, b_2, b_3,
		ab_temp, // temporary results
		c_00_10_20_30, c_01_11_21_31, c_02_12_22_32, c_03_13_23_33;
	
	// prefetch
	a_0123        = _mm256_load_pd( &A[0] );
	b_0           = _mm256_broadcast_sd( &B[0] );
	b_1           = _mm256_broadcast_sd( &B[5] );
	b_2           = _mm256_broadcast_sd( &B[10] );
	b_3           = _mm256_broadcast_sd( &B[15] );

	// zero registers
	c_00_10_20_30 = _mm256_mul_pd( a_0123, b_0 );
	a_0123        = _mm256_load_pd( &A[4] ); // prefetch
	b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
	
	
/*	__builtin_prefetch( A+40 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
	c_01_11_21_31 = _mm256_mul_pd( a_0123, b_1 );
	a_0123        = _mm256_load_pd( &A[8] ); // prefetch
	b_1           = _mm256_broadcast_sd( &B[9] );


/*	__builtin_prefetch( A+48 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
	ab_temp       = _mm256_mul_pd( a_0123, b_1 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	b_1           = _mm256_broadcast_sd( &B[13] );
	c_02_12_22_32 = _mm256_mul_pd( a_0123, b_2 );
	a_0123        = _mm256_load_pd( &A[12] ); // prefetch
	b_2           = _mm256_broadcast_sd( &B[14] );


/*	__builtin_prefetch( A+56 );*/
	ab_temp       = _mm256_mul_pd( a_0123, b_0 );
	c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
	b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
	ab_temp       = _mm256_mul_pd( a_0123, b_1 );
	c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
	b_1           = _mm256_broadcast_sd( &B[17] );
	ab_temp       = _mm256_mul_pd( a_0123, b_2 );
	c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
	b_2           = _mm256_broadcast_sd( &B[18] );
	c_03_13_23_33 = _mm256_mul_pd( a_0123, b_3 );
	a_0123        = _mm256_load_pd( &A[16] ); // prefetch
	b_3           = _mm256_broadcast_sd( &B[19] );
	
	A += 16;
	B += 16;

	for(k=4; k<kadd-3; k+=4)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[7] );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[11] );


	/*	__builtin_prefetch( A+48 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[12] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[13] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[14] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[12] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[15] );


	/*	__builtin_prefetch( A+56 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[16] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[17] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[18] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[16] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[19] );
		
		A += 16;
		B += 16;

		}
	
	if(kadd%4>=2)
		{
		
	/*	__builtin_prefetch( A+32 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[4] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[5] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[6] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[7] );
	
	
	/*	__builtin_prefetch( A+40 );*/
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		b_0           = _mm256_broadcast_sd( &B[8] ); // prefetch
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		b_1           = _mm256_broadcast_sd( &B[9] );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		b_2           = _mm256_broadcast_sd( &B[10] );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[8] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		b_3           = _mm256_broadcast_sd( &B[11] );
		
		A += 8;
		B += 8;

		}

	if(kadd%2==1)
		{
		
		ab_temp       = _mm256_mul_pd( a_0123, b_0 );
		c_00_10_20_30 = _mm256_add_pd( c_00_10_20_30, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_1 );
		c_01_11_21_31 = _mm256_add_pd( c_01_11_21_31, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_2 );
		c_02_12_22_32 = _mm256_add_pd( c_02_12_22_32, ab_temp );
		ab_temp       = _mm256_mul_pd( a_0123, b_3 );
		a_0123        = _mm256_load_pd( &A[4] ); // prefetch
		c_03_13_23_33 = _mm256_add_pd( c_03_13_23_33, ab_temp );
		
		}

	_mm256_store_pd( &D[0+ldc*0], c_00_10_20_30 );
	_mm256_store_pd( &D[0+ldc*2], c_02_12_22_32 );
	_mm256_store_pd( &D[0+ldc*1], c_01_11_21_31 );
	_mm256_store_pd( &D[0+ldc*3], c_03_13_23_33 );

	}

