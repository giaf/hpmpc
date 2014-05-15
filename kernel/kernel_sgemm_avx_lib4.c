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



/*// normal-transposed, 16x4 with data packed in 4*/
/*void kernel_sgemm_pp_nt_16x4_lib4(int kmax, float *A0, float *A1, float *A2, float *A3, float *B, float *C0, float *C1, float *C2, float *C3, int ldc, int alg)*/
/*	{*/
/*	*/
/*	if(kmax<=0)*/
/*		return;*/
/*	*/
/*	int k;*/
/*	*/
/*	__m128*/
/*		la_07, ha_07, la_8f, ha_8f,*/
/*		lb_03; */

/*	__m256*/
/*		temp,*/
/*		c_00, c_01, c_02, c_03,*/
/*		c_80, c_81, c_82, c_83,*/
/*		a_07, a_8f,*/
/*		b_03, b_0, b_1, b_2, b_3; */
/*	*/
/*	c_00 = _mm256_setzero_ps();*/
/*	c_01 = _mm256_setzero_ps();*/
/*	c_02 = _mm256_setzero_ps();*/
/*	c_03 = _mm256_setzero_ps();*/
/*	c_80 = _mm256_setzero_ps();*/
/*	c_81 = _mm256_setzero_ps();*/
/*	c_82 = _mm256_setzero_ps();*/
/*	c_83 = _mm256_setzero_ps();*/

/*	for(k=0; k<kmax-3; k+=4)*/
/*		{*/

/*//		_mm256_loadu2_m128 (float const* hiaddr, float const* loaddr)*/
/*		*/
/*		la_07 = _mm_load_ps( &A0[0] );*/
/*		ha_07 = _mm_load_ps( &A1[0] );*/
/*		a_07 = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_07), _mm256_castps128_ps256(ha_07), 0 );*/

/*		la_8f = _mm_load_ps( &A2[0] );*/
/*		ha_8f = _mm_load_ps( &A3[0] );*/
/*		a_8f = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_8f), _mm256_castps128_ps256(ha_8f), 0 );*/

/*		lb_03 = _mm_load_ps( &B[0] );*/
/*		b_03 = _mm256_castps128_ps256( lb_03 );*/
/*		b_03 = _mm256_permute2f128_ps(b_03, b_03, 0);*/
/*		*/
/*		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm256_mul_ps( a_07, b_0 );*/
/*		c_00 = _mm256_add_ps( c_00, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_0 );*/
/*		c_80 = _mm256_add_ps( c_80, temp );*/
/*		*/
/*		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm256_mul_ps( a_07, b_1 );*/
/*		c_01 = _mm256_add_ps( c_01, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_1 );*/
/*		c_81 = _mm256_add_ps( c_81, temp );*/
/*		*/
/*		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm256_mul_ps( a_07, b_2 );*/
/*		c_02 = _mm256_add_ps( c_02, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_2 );*/
/*		c_82 = _mm256_add_ps( c_82, temp );*/
/*	*/
/*		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm256_mul_ps( a_07, b_3 );*/
/*		c_03 = _mm256_add_ps( c_03, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_3 );*/
/*		c_83 = _mm256_add_ps( c_83, temp );*/



/*		la_07 = _mm_load_ps( &A0[4] );*/
/*		ha_07 = _mm_load_ps( &A1[4] );*/
/*		a_07 = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_07), _mm256_castps128_ps256(ha_07), 0 );*/

/*		la_8f = _mm_load_ps( &A2[4] );*/
/*		ha_8f = _mm_load_ps( &A3[4] );*/
/*		a_8f = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_8f), _mm256_castps128_ps256(ha_8f), 0 );*/

/*		lb_03 = _mm_load_ps( &B[4] );*/
/*		b_03 = _mm256_castps128_ps256( lb_03 );*/
/*		b_03 = _mm256_permute2f128_ps(b_03, b_03, 0);*/
/*		*/
/*		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm256_mul_ps( a_07, b_0 );*/
/*		c_00 = _mm256_add_ps( c_00, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_0 );*/
/*		c_80 = _mm256_add_ps( c_80, temp );*/
/*		*/
/*		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm256_mul_ps( a_07, b_1 );*/
/*		c_01 = _mm256_add_ps( c_01, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_1 );*/
/*		c_81 = _mm256_add_ps( c_81, temp );*/
/*		*/
/*		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm256_mul_ps( a_07, b_2 );*/
/*		c_02 = _mm256_add_ps( c_02, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_2 );*/
/*		c_82 = _mm256_add_ps( c_82, temp );*/
/*	*/
/*		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm256_mul_ps( a_07, b_3 );*/
/*		c_03 = _mm256_add_ps( c_03, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_3 );*/
/*		c_83 = _mm256_add_ps( c_83, temp );*/


/*		*/
/*		la_07 = _mm_load_ps( &A0[8] );*/
/*		ha_07 = _mm_load_ps( &A1[8] );*/
/*		a_07 = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_07), _mm256_castps128_ps256(ha_07), 0 );*/

/*		la_8f = _mm_load_ps( &A2[8] );*/
/*		ha_8f = _mm_load_ps( &A3[8] );*/
/*		a_8f = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_8f), _mm256_castps128_ps256(ha_8f), 0 );*/

/*		lb_03 = _mm_load_ps( &B[8] );*/
/*		b_03 = _mm256_castps128_ps256( lb_03 );*/
/*		b_03 = _mm256_permute2f128_ps(b_03, b_03, 0);*/
/*		*/
/*		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm256_mul_ps( a_07, b_0 );*/
/*		c_00 = _mm256_add_ps( c_00, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_0 );*/
/*		c_80 = _mm256_add_ps( c_80, temp );*/
/*		*/
/*		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm256_mul_ps( a_07, b_1 );*/
/*		c_01 = _mm256_add_ps( c_01, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_1 );*/
/*		c_81 = _mm256_add_ps( c_81, temp );*/
/*		*/
/*		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm256_mul_ps( a_07, b_2 );*/
/*		c_02 = _mm256_add_ps( c_02, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_2 );*/
/*		c_82 = _mm256_add_ps( c_82, temp );*/
/*	*/
/*		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm256_mul_ps( a_07, b_3 );*/
/*		c_03 = _mm256_add_ps( c_03, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_3 );*/
/*		c_83 = _mm256_add_ps( c_83, temp );*/



/*		la_07 = _mm_load_ps( &A0[12] );*/
/*		ha_07 = _mm_load_ps( &A1[12] );*/
/*		a_07 = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_07), _mm256_castps128_ps256(ha_07), 0 );*/

/*		la_8f = _mm_load_ps( &A2[12] );*/
/*		ha_8f = _mm_load_ps( &A3[12] );*/
/*		a_8f = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_8f), _mm256_castps128_ps256(ha_8f), 0 );*/

/*		lb_03 = _mm_load_ps( &B[12] );*/
/*		b_03 = _mm256_castps128_ps256( lb_03 );*/
/*		b_03 = _mm256_permute2f128_ps(b_03, b_03, 0);*/
/*		*/
/*		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm256_mul_ps( a_07, b_0 );*/
/*		c_00 = _mm256_add_ps( c_00, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_0 );*/
/*		c_80 = _mm256_add_ps( c_80, temp );*/
/*		*/
/*		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm256_mul_ps( a_07, b_1 );*/
/*		c_01 = _mm256_add_ps( c_01, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_1 );*/
/*		c_81 = _mm256_add_ps( c_81, temp );*/
/*		*/
/*		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm256_mul_ps( a_07, b_2 );*/
/*		c_02 = _mm256_add_ps( c_02, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_2 );*/
/*		c_82 = _mm256_add_ps( c_82, temp );*/
/*	*/
/*		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm256_mul_ps( a_07, b_3 );*/
/*		c_03 = _mm256_add_ps( c_03, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_3 );*/
/*		c_83 = _mm256_add_ps( c_83, temp );*/


/*		A0 += 16;*/
/*		A1 += 16;*/
/*		A2 += 16;*/
/*		A3 += 16;*/
/*		B  += 16;*/

/*		}*/
/*	*/
/*	for(; k<kmax; k++)*/
/*		{*/
/*		*/
/*		la_07 = _mm_load_ps( &A0[0] );*/
/*		ha_07 = _mm_load_ps( &A1[0] );*/
/*		a_07 = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_07), _mm256_castps128_ps256(ha_07), 0 );*/

/*		la_8f = _mm_load_ps( &A2[0] );*/
/*		ha_8f = _mm_load_ps( &A3[0] );*/
/*		a_8f = _mm256_permute2f128_ps( _mm256_castps128_ps256(la_8f), _mm256_castps128_ps256(ha_8f), 0 );*/

/*		lb_03 = _mm_load_ps( &B[0] );*/
/*		b_03 = _mm256_castps128_ps256( lb_03 );*/
/*		b_03 = _mm256_permute2f128_ps(b_03, b_03, 0);*/
/*		*/
/*		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm256_mul_ps( a_07, b_0 );*/
/*		c_00 = _mm256_add_ps( c_00, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_0 );*/
/*		c_80 = _mm256_add_ps( c_80, temp );*/
/*		*/
/*		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm256_mul_ps( a_07, b_1 );*/
/*		c_01 = _mm256_add_ps( c_01, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_1 );*/
/*		c_81 = _mm256_add_ps( c_81, temp );*/
/*		*/
/*		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm256_mul_ps( a_07, b_2 );*/
/*		c_02 = _mm256_add_ps( c_02, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_2 );*/
/*		c_82 = _mm256_add_ps( c_82, temp );*/
/*	*/
/*		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm256_mul_ps( a_07, b_3 );*/
/*		c_03 = _mm256_add_ps( c_03, temp );*/
/*		temp = _mm256_mul_ps( a_8f, b_3 );*/
/*		c_83 = _mm256_add_ps( c_83, temp );*/
/*		*/
/*		A0 += 4;*/
/*		A1 += 4;*/
/*		A2 += 4;*/
/*		A3 += 4;*/
/*		B  += 4;*/
/*		*/
/*		}*/

/*	__m128*/
/*		lc_00, lc_01, lc_02, lc_03,*/
/*		hc_00, hc_01, hc_02, hc_03,*/
/*		lc_80, lc_81, lc_82, lc_83,*/
/*		hc_80, hc_81, hc_82, hc_83,*/
/*		d_00, d_01, d_02, d_03,*/
/*		d_40, d_41, d_42, d_43;*/

/*	hc_00 = _mm256_extractf128_ps ( c_00, 1 );*/
/*	lc_00 = _mm256_castps256_ps128( c_00 );*/
/*	hc_01 = _mm256_extractf128_ps ( c_01, 1 );*/
/*	lc_01 = _mm256_castps256_ps128( c_01 );*/
/*	hc_02 = _mm256_extractf128_ps ( c_02, 1 );*/
/*	lc_02 = _mm256_castps256_ps128( c_02 );*/
/*	hc_03 = _mm256_extractf128_ps ( c_03, 1 );*/
/*	lc_03 = _mm256_castps256_ps128( c_03 );*/
/*	hc_80 = _mm256_extractf128_ps ( c_80, 1 );*/
/*	lc_80 = _mm256_castps256_ps128( c_80 );*/
/*	hc_81 = _mm256_extractf128_ps ( c_81, 1 );*/
/*	lc_81 = _mm256_castps256_ps128( c_81 );*/
/*	hc_82 = _mm256_extractf128_ps ( c_82, 1 );*/
/*	lc_82 = _mm256_castps256_ps128( c_82 );*/
/*	hc_83 = _mm256_extractf128_ps ( c_83, 1 );*/
/*	lc_83 = _mm256_castps256_ps128( c_83 );*/

/*		_mm_store_ps( &C0[0+ldc*0], lc_00 );*/
/*		_mm_store_ps( &C0[0+ldc*1], lc_01 );*/
/*		_mm_store_ps( &C0[0+ldc*2], lc_02 );*/
/*		_mm_store_ps( &C0[0+ldc*3], lc_03 );*/
/*		_mm_store_ps( &C1[0+ldc*0], hc_00 );*/
/*		_mm_store_ps( &C1[0+ldc*1], hc_01 );*/
/*		_mm_store_ps( &C1[0+ldc*2], hc_02 );*/
/*		_mm_store_ps( &C1[0+ldc*3], hc_03 );*/
/*		_mm_store_ps( &C2[0+ldc*0], lc_80 );*/
/*		_mm_store_ps( &C2[0+ldc*1], lc_81 );*/
/*		_mm_store_ps( &C2[0+ldc*2], lc_82 );*/
/*		_mm_store_ps( &C2[0+ldc*3], lc_83 );*/
/*		_mm_store_ps( &C3[0+ldc*0], hc_80 );*/
/*		_mm_store_ps( &C3[0+ldc*1], hc_81 );*/
/*		_mm_store_ps( &C3[0+ldc*2], hc_82 );*/
/*		_mm_store_ps( &C3[0+ldc*3], hc_83 );*/

/*	}*/



/*// normal-transposed, 8x4 with data packed in 4*/
/*void kernel_sgemm_pp_nt_8x4_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)*/
/*	{*/
/*	*/
/*	if(kmax<=0)*/
/*		return;*/
/*	*/
/*	int k;*/
/*	*/
/*	__m128*/
/*		temp,*/
/*		c_00, c_01, c_02, c_03,*/
/*		c_40, c_41, c_42, c_43,*/
/*		a_03, a_47,*/
/*		b_03, b_0, b_1, b_2, b_3; */
/*	*/
/*	c_00 = _mm_setzero_ps();*/
/*	c_01 = _mm_setzero_ps();*/
/*	c_02 = _mm_setzero_ps();*/
/*	c_03 = _mm_setzero_ps();*/
/*	c_40 = _mm_setzero_ps();*/
/*	c_41 = _mm_setzero_ps();*/
/*	c_42 = _mm_setzero_ps();*/
/*	c_43 = _mm_setzero_ps();*/

/*	for(k=0; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		b_03 = _mm_load_ps( &B[0] );*/
/*		*/
/*		a_03 = _mm_load_ps( &A0[0] );*/
/*		a_47 = _mm_load_ps( &A1[0] );*/
/*		*/
/*		b_0 = _mm_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm_mul_ps( a_03, b_0 );*/
/*		c_00 = _mm_add_ps( c_00, temp );*/
/*		temp = _mm_mul_ps( a_47, b_0 );*/
/*		c_40 = _mm_add_ps( c_40, temp );*/
/*		*/
/*		b_1 = _mm_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm_mul_ps( a_03, b_1 );*/
/*		c_01 = _mm_add_ps( c_01, temp );*/
/*		temp = _mm_mul_ps( a_47, b_1 );*/
/*		c_41 = _mm_add_ps( c_41, temp );*/
/*		*/
/*		b_2 = _mm_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm_mul_ps( a_03, b_2 );*/
/*		c_02 = _mm_add_ps( c_02, temp );*/
/*		temp = _mm_mul_ps( a_47, b_2 );*/
/*		c_42 = _mm_add_ps( c_42, temp );*/
/*	*/
/*		b_3 = _mm_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm_mul_ps( a_03, b_3 );*/
/*		c_03 = _mm_add_ps( c_03, temp );*/
/*		temp = _mm_mul_ps( a_47, b_3 );*/
/*		c_43 = _mm_add_ps( c_43, temp );*/



/*		b_03 = _mm_load_ps( &B[4] );*/
/*		*/
/*		a_03 = _mm_load_ps( &A0[4] );*/
/*		a_47 = _mm_load_ps( &A1[4] );*/
/*		*/
/*		b_0 = _mm_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm_mul_ps( a_03, b_0 );*/
/*		c_00 = _mm_add_ps( c_00, temp );*/
/*		temp = _mm_mul_ps( a_47, b_0 );*/
/*		c_40 = _mm_add_ps( c_40, temp );*/
/*		*/
/*		b_1 = _mm_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm_mul_ps( a_03, b_1 );*/
/*		c_01 = _mm_add_ps( c_01, temp );*/
/*		temp = _mm_mul_ps( a_47, b_1 );*/
/*		c_41 = _mm_add_ps( c_41, temp );*/
/*		*/
/*		b_2 = _mm_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm_mul_ps( a_03, b_2 );*/
/*		c_02 = _mm_add_ps( c_02, temp );*/
/*		temp = _mm_mul_ps( a_47, b_2 );*/
/*		c_42 = _mm_add_ps( c_42, temp );*/
/*	*/
/*		b_3 = _mm_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm_mul_ps( a_03, b_3 );*/
/*		c_03 = _mm_add_ps( c_03, temp );*/
/*		temp = _mm_mul_ps( a_47, b_3 );*/
/*		c_43 = _mm_add_ps( c_43, temp );*/


/*		*/
/*		b_03 = _mm_load_ps( &B[8] );*/
/*		*/
/*		a_03 = _mm_load_ps( &A0[8] );*/
/*		a_47 = _mm_load_ps( &A1[8] );*/
/*		*/
/*		b_0 = _mm_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm_mul_ps( a_03, b_0 );*/
/*		c_00 = _mm_add_ps( c_00, temp );*/
/*		temp = _mm_mul_ps( a_47, b_0 );*/
/*		c_40 = _mm_add_ps( c_40, temp );*/
/*		*/
/*		b_1 = _mm_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm_mul_ps( a_03, b_1 );*/
/*		c_01 = _mm_add_ps( c_01, temp );*/
/*		temp = _mm_mul_ps( a_47, b_1 );*/
/*		c_41 = _mm_add_ps( c_41, temp );*/
/*		*/
/*		b_2 = _mm_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm_mul_ps( a_03, b_2 );*/
/*		c_02 = _mm_add_ps( c_02, temp );*/
/*		temp = _mm_mul_ps( a_47, b_2 );*/
/*		c_42 = _mm_add_ps( c_42, temp );*/
/*	*/
/*		b_3 = _mm_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm_mul_ps( a_03, b_3 );*/
/*		c_03 = _mm_add_ps( c_03, temp );*/
/*		temp = _mm_mul_ps( a_47, b_3 );*/
/*		c_43 = _mm_add_ps( c_43, temp );*/



/*		b_03 = _mm_load_ps( &B[12] );*/
/*		*/
/*		a_03 = _mm_load_ps( &A0[12] );*/
/*		a_47 = _mm_load_ps( &A1[12] );*/
/*		*/
/*		b_0 = _mm_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm_mul_ps( a_03, b_0 );*/
/*		c_00 = _mm_add_ps( c_00, temp );*/
/*		temp = _mm_mul_ps( a_47, b_0 );*/
/*		c_40 = _mm_add_ps( c_40, temp );*/
/*		*/
/*		b_1 = _mm_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm_mul_ps( a_03, b_1 );*/
/*		c_01 = _mm_add_ps( c_01, temp );*/
/*		temp = _mm_mul_ps( a_47, b_1 );*/
/*		c_41 = _mm_add_ps( c_41, temp );*/
/*		*/
/*		b_2 = _mm_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm_mul_ps( a_03, b_2 );*/
/*		c_02 = _mm_add_ps( c_02, temp );*/
/*		temp = _mm_mul_ps( a_47, b_2 );*/
/*		c_42 = _mm_add_ps( c_42, temp );*/
/*	*/
/*		b_3 = _mm_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm_mul_ps( a_03, b_3 );*/
/*		c_03 = _mm_add_ps( c_03, temp );*/
/*		temp = _mm_mul_ps( a_47, b_3 );*/
/*		c_43 = _mm_add_ps( c_43, temp );*/


/*		A0 += 16;*/
/*		A1 += 16;*/
/*		B  += 16;*/

/*		}*/
/*	*/
/*	for(; k<kmax; k++)*/
/*		{*/
/*		*/
/*		b_03 = _mm_load_ps( &B[0] );*/
/*		*/
/*		a_03 = _mm_load_ps( &A0[0] );*/
/*		a_47 = _mm_load_ps( &A1[0] );*/
/*		*/
/*		b_0 = _mm_shuffle_ps( b_03, b_03, 0 );*/
/*		temp = _mm_mul_ps( a_03, b_0 );*/
/*		c_00 = _mm_add_ps( c_00, temp );*/
/*		temp = _mm_mul_ps( a_47, b_0 );*/
/*		c_40 = _mm_add_ps( c_40, temp );*/
/*		*/
/*		b_1 = _mm_shuffle_ps( b_03, b_03, 85 );*/
/*		temp = _mm_mul_ps( a_03, b_1 );*/
/*		c_01 = _mm_add_ps( c_01, temp );*/
/*		temp = _mm_mul_ps( a_47, b_1 );*/
/*		c_41 = _mm_add_ps( c_41, temp );*/
/*		*/
/*		b_2 = _mm_shuffle_ps( b_03, b_03, 170 );*/
/*		temp = _mm_mul_ps( a_03, b_2 );*/
/*		c_02 = _mm_add_ps( c_02, temp );*/
/*		temp = _mm_mul_ps( a_47, b_2 );*/
/*		c_42 = _mm_add_ps( c_42, temp );*/
/*	*/
/*		b_3 = _mm_shuffle_ps( b_03, b_03, 255 );*/
/*		temp = _mm_mul_ps( a_03, b_3 );*/
/*		c_03 = _mm_add_ps( c_03, temp );*/
/*		temp = _mm_mul_ps( a_47, b_3 );*/
/*		c_43 = _mm_add_ps( c_43, temp );*/
/*		*/
/*		A0 += 4;*/
/*		A1 += 4;*/
/*		B  += 4;*/
/*		*/
/*		}*/

/*	__m128*/
/*		d_00, d_01, d_02, d_03,*/
/*		d_40, d_41, d_42, d_43;*/

/*	if(alg==0)*/
/*		{*/
/*		_mm_store_ps( &C0[0+ldc*0], c_00 );*/
/*		_mm_store_ps( &C0[0+ldc*1], c_01 );*/
/*		_mm_store_ps( &C0[0+ldc*2], c_02 );*/
/*		_mm_store_ps( &C0[0+ldc*3], c_03 );*/
/*		_mm_store_ps( &C1[0+ldc*0], c_40 );*/
/*		_mm_store_ps( &C1[0+ldc*1], c_41 );*/
/*		_mm_store_ps( &C1[0+ldc*2], c_42 );*/
/*		_mm_store_ps( &C1[0+ldc*3], c_43 );*/
/*		}*/
/*	else*/
/*		{*/
/*		d_00 = _mm_load_ps( &C0[0+ldc*0] );*/
/*		d_01 = _mm_load_ps( &C0[0+ldc*1] );*/
/*		d_02 = _mm_load_ps( &C0[0+ldc*2] );*/
/*		d_03 = _mm_load_ps( &C0[0+ldc*3] );*/
/*		d_40 = _mm_load_ps( &C1[0+ldc*0] );*/
/*		d_41 = _mm_load_ps( &C1[0+ldc*1] );*/
/*		d_42 = _mm_load_ps( &C1[0+ldc*2] );*/
/*		d_43 = _mm_load_ps( &C1[0+ldc*3] );*/
/*		*/
/*		if(alg==1)*/
/*			{*/
/*			d_00 = _mm_add_ps( d_00, c_00 );*/
/*			d_01 = _mm_add_ps( d_01, c_01 );*/
/*			d_02 = _mm_add_ps( d_02, c_02 );*/
/*			d_03 = _mm_add_ps( d_03, c_03 );*/
/*			d_40 = _mm_add_ps( d_40, c_40 );*/
/*			d_41 = _mm_add_ps( d_41, c_41 );*/
/*			d_42 = _mm_add_ps( d_42, c_42 );*/
/*			d_43 = _mm_add_ps( d_43, c_43 );*/
/*			}*/
/*		else // alg == -1*/
/*			{*/
/*			d_00 = _mm_sub_ps( d_00, c_00 );*/
/*			d_01 = _mm_sub_ps( d_01, c_01 );*/
/*			d_02 = _mm_sub_ps( d_02, c_02 );*/
/*			d_03 = _mm_sub_ps( d_03, c_03 );*/
/*			d_40 = _mm_sub_ps( d_40, c_40 );*/
/*			d_41 = _mm_sub_ps( d_41, c_41 );*/
/*			d_42 = _mm_sub_ps( d_42, c_42 );*/
/*			d_43 = _mm_sub_ps( d_43, c_43 );*/
/*			}*/

/*		_mm_store_ps( &C0[0+ldc*0], d_00 );*/
/*		_mm_store_ps( &C0[0+ldc*1], d_01 );*/
/*		_mm_store_ps( &C0[0+ldc*2], d_02 );*/
/*		_mm_store_ps( &C0[0+ldc*3], d_03 );*/
/*		_mm_store_ps( &C1[0+ldc*0], d_40 );*/
/*		_mm_store_ps( &C1[0+ldc*1], d_41 );*/
/*		_mm_store_ps( &C1[0+ldc*2], d_42 );*/
/*		_mm_store_ps( &C1[0+ldc*3], d_43 );*/
/*		}*/

/*	}*/



// normal-transposed, 8x4 with data packed in 4
void kernel_sgemm_pp_nt_8x4_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256
		temp,
		c_00, c_01, c_02, c_03,
		c_40, c_41, c_42, c_43,
		a_03, a_47,
		b_03, b_0, b_1, b_2, b_3; 
	
	c_00 = _mm256_setzero_ps();
	c_01 = _mm256_setzero_ps();
	c_02 = _mm256_setzero_ps();
	c_03 = _mm256_setzero_ps();
	c_40 = _mm256_setzero_ps();
	c_41 = _mm256_setzero_ps();
	c_42 = _mm256_setzero_ps();
	c_43 = _mm256_setzero_ps();

	k = 0;
	for(; k<kmax-7; k+=8)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A0[0] );
		a_47 = _mm256_load_ps( &A1[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		
		b_03 = _mm256_load_ps( &B[8] );
		
		a_03 = _mm256_load_ps( &A0[8] );
		a_47 = _mm256_load_ps( &A1[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		b_03 = _mm256_load_ps( &B[16] );
		
		a_03 = _mm256_load_ps( &A0[16] );
		a_47 = _mm256_load_ps( &A1[16] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		
		b_03 = _mm256_load_ps( &B[24] );
		
		a_03 = _mm256_load_ps( &A0[24] );
		a_47 = _mm256_load_ps( &A1[24] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		A0 += 32;
		A1 += 32;
		B  += 32;

		}
	for(; k<kmax-3; k+=4)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A0[0] );
		a_47 = _mm256_load_ps( &A1[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		
		b_03 = _mm256_load_ps( &B[8] );
		
		a_03 = _mm256_load_ps( &A0[8] );
		a_47 = _mm256_load_ps( &A1[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );


		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax-1; k+=2)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A0[0] );
		a_47 = _mm256_load_ps( &A1[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		temp = _mm256_mul_ps( a_47, b_0 );
		c_40 = _mm256_add_ps( c_40, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		temp = _mm256_mul_ps( a_47, b_1 );
		c_41 = _mm256_add_ps( c_41, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
		temp = _mm256_mul_ps( a_47, b_2 );
		c_42 = _mm256_add_ps( c_42, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );
		temp = _mm256_mul_ps( a_47, b_3 );
		c_43 = _mm256_add_ps( c_43, temp );

		A0 += 8;
		A1 += 8;
		B  += 8;
		
		}

	__m128
		ltemp,
		la_03, la_47,
		lb_03, lb_0, lb_1, lb_2, lb_3,
		lc_00, lc_01, lc_02, lc_03,
		lc_40, lc_41, lc_42, lc_43,
		hc_00, hc_01, hc_02, hc_03,
		hc_40, hc_41, hc_42, hc_43,
		d_00, d_01, d_02, d_03,
		d_40, d_41, d_42, d_43;
	
	hc_00 = _mm256_extractf128_ps ( c_00, 1 );
	lc_00 = _mm256_castps256_ps128( c_00 );
	lc_00 = _mm_add_ps( lc_00, hc_00 );
	hc_01 = _mm256_extractf128_ps ( c_01, 1 );
	lc_01 = _mm256_castps256_ps128( c_01 );
	lc_01 = _mm_add_ps( lc_01, hc_01 );
	hc_02 = _mm256_extractf128_ps ( c_02, 1 );
	lc_02 = _mm256_castps256_ps128( c_02 );
	lc_02 = _mm_add_ps( lc_02, hc_02 );
	hc_03 = _mm256_extractf128_ps ( c_03, 1 );
	lc_03 = _mm256_castps256_ps128( c_03 );
	lc_03 = _mm_add_ps( lc_03, hc_03 );
	hc_40 = _mm256_extractf128_ps ( c_40, 1 );
	lc_40 = _mm256_castps256_ps128( c_40 );
	lc_40 = _mm_add_ps( lc_40, hc_40 );
	hc_41 = _mm256_extractf128_ps ( c_41, 1 );
	lc_41 = _mm256_castps256_ps128( c_41 );
	lc_41 = _mm_add_ps( lc_41, hc_41 );
	hc_42 = _mm256_extractf128_ps ( c_42, 1 );
	lc_42 = _mm256_castps256_ps128( c_42 );
	lc_42 = _mm_add_ps( lc_42, hc_42 );
	hc_43 = _mm256_extractf128_ps ( c_43, 1 );
	lc_43 = _mm256_castps256_ps128( c_43 );
	lc_43 = _mm_add_ps( lc_43, hc_43 );

	for(; k<kmax; k++)
		{
		
		lb_03 = _mm_load_ps( &B[0] );
		
		la_03 = _mm_load_ps( &A0[0] );
		la_47 = _mm_load_ps( &A1[0] );
		
		lb_0 = _mm_shuffle_ps( lb_03, lb_03, 0 );
		ltemp = _mm_mul_ps( la_03, lb_0 );
		lc_00 = _mm_add_ps( lc_00, ltemp );
		ltemp = _mm_mul_ps( la_47, lb_0 );
		lc_40 = _mm_add_ps( lc_40, ltemp );
		
		lb_1 = _mm_shuffle_ps( lb_03, lb_03, 85 );
		ltemp = _mm_mul_ps( la_03, lb_1 );
		lc_01 = _mm_add_ps( lc_01, ltemp );
		ltemp = _mm_mul_ps( la_47, lb_1 );
		lc_41 = _mm_add_ps( lc_41, ltemp );
		
		lb_2 = _mm_shuffle_ps( lb_03, lb_03, 170 );
		ltemp = _mm_mul_ps( la_03, lb_2 );
		lc_02 = _mm_add_ps( lc_02, ltemp );
		ltemp = _mm_mul_ps( la_47, lb_2 );
		lc_42 = _mm_add_ps( lc_42, ltemp );
	
		lb_3 = _mm_shuffle_ps( lb_03, lb_03, 255 );
		ltemp = _mm_mul_ps( la_03, lb_3 );
		lc_03 = _mm_add_ps( lc_03, ltemp );
		ltemp = _mm_mul_ps( la_47, lb_3 );
		lc_43 = _mm_add_ps( lc_43, ltemp );

		A0 += 8;
		A1 += 8;
		B  += 8;
		
		}

	if(alg==0)
		{
		_mm_store_ps( &C0[0+ldc*0], lc_00 );
		_mm_store_ps( &C0[0+ldc*1], lc_01 );
		_mm_store_ps( &C0[0+ldc*2], lc_02 );
		_mm_store_ps( &C0[0+ldc*3], lc_03 );
		_mm_store_ps( &C1[0+ldc*0], lc_40 );
		_mm_store_ps( &C1[0+ldc*1], lc_41 );
		_mm_store_ps( &C1[0+ldc*2], lc_42 );
		_mm_store_ps( &C1[0+ldc*3], lc_43 );
		}
	else
		{
		d_00 = _mm_load_ps( &C0[0+ldc*0] );
		d_01 = _mm_load_ps( &C0[0+ldc*1] );
		d_02 = _mm_load_ps( &C0[0+ldc*2] );
		d_03 = _mm_load_ps( &C0[0+ldc*3] );
		d_40 = _mm_load_ps( &C1[0+ldc*0] );
		d_41 = _mm_load_ps( &C1[0+ldc*1] );
		d_42 = _mm_load_ps( &C1[0+ldc*2] );
		d_43 = _mm_load_ps( &C1[0+ldc*3] );
		
		if(alg==1)
			{
			d_00 = _mm_add_ps( d_00, lc_00 );
			d_01 = _mm_add_ps( d_01, lc_01 );
			d_02 = _mm_add_ps( d_02, lc_02 );
			d_03 = _mm_add_ps( d_03, lc_03 );
			d_40 = _mm_add_ps( d_40, lc_40 );
			d_41 = _mm_add_ps( d_41, lc_41 );
			d_42 = _mm_add_ps( d_42, lc_42 );
			d_43 = _mm_add_ps( d_43, lc_43 );
			}
		else // alg == -1
			{
			d_00 = _mm_sub_ps( d_00, lc_00 );
			d_01 = _mm_sub_ps( d_01, lc_01 );
			d_02 = _mm_sub_ps( d_02, lc_02 );
			d_03 = _mm_sub_ps( d_03, lc_03 );
			d_40 = _mm_sub_ps( d_40, lc_40 );
			d_41 = _mm_sub_ps( d_41, lc_41 );
			d_42 = _mm_sub_ps( d_42, lc_42 );
			d_43 = _mm_sub_ps( d_43, lc_43 );
			}

		_mm_store_ps( &C0[0+ldc*0], d_00 );
		_mm_store_ps( &C0[0+ldc*1], d_01 );
		_mm_store_ps( &C0[0+ldc*2], d_02 );
		_mm_store_ps( &C0[0+ldc*3], d_03 );
		_mm_store_ps( &C1[0+ldc*0], d_40 );
		_mm_store_ps( &C1[0+ldc*1], d_41 );
		_mm_store_ps( &C1[0+ldc*2], d_42 );
		_mm_store_ps( &C1[0+ldc*3], d_43 );
		}

	}



// normal-transposed, 8x3 with data packed in 4
void kernel_sgemm_pp_nt_8x3_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1, c_03_2,
		c_47_0, c_47_1, c_47_2,
		a_03, a_47,
		b_0, b_1, b_2; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();
	c_03_2 = _mm_setzero_ps();
	c_47_0 = _mm_setzero_ps();
	c_47_1 = _mm_setzero_ps();
	c_47_2 = _mm_setzero_ps();

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_2 = _mm_add_ps( c_47_2, b_2 );



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A0[4] );
		a_47 = _mm_load_ps( &A1[4] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_2 = _mm_add_ps( c_47_2, b_2 );


		
		b_0 = _mm_load_ps( &B[8] );
		
		a_03 = _mm_load_ps( &A0[8] );
		a_47 = _mm_load_ps( &A1[8] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_2 = _mm_add_ps( c_47_2, b_2 );



		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A0[12] );
		a_47 = _mm_load_ps( &A1[12] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_2 = _mm_add_ps( c_47_2, b_2 );


		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	for(; k<kmax; k++)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 170 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_2 = _mm_add_ps( c_03_2, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_2 = _mm_add_ps( c_47_2, b_2 );
		
		A0 += 4;
		A1 += 4;
		B  += 4;
		
		}

	__m128
		d_03_0, d_03_1, d_03_2,
		d_47_0, d_47_1, d_47_2;

	if(alg==0)
		{
		_mm_store_ps( &C0[0+ldc*0], c_03_0 );
		_mm_store_ps( &C1[0+ldc*0], c_47_0 );
		_mm_store_ps( &C0[0+ldc*1], c_03_1 );
		_mm_store_ps( &C1[0+ldc*1], c_47_1 );
		_mm_store_ps( &C0[0+ldc*2], c_03_2 );
		_mm_store_ps( &C1[0+ldc*2], c_47_2 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C0[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C0[0+ldc*2] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		d_47_1 = _mm_load_ps( &C1[0+ldc*1] );
		d_47_2 = _mm_load_ps( &C1[0+ldc*2] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_add_ps( d_03_2, c_03_2 );
		d_47_0 = _mm_add_ps( d_47_0, c_47_0 );
		d_47_1 = _mm_add_ps( d_47_1, c_47_1 );
		d_47_2 = _mm_add_ps( d_47_2, c_47_2 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		_mm_store_ps( &C0[0+ldc*1], d_03_1 );
		_mm_store_ps( &C1[0+ldc*1], d_47_1 );
		_mm_store_ps( &C0[0+ldc*2], d_03_2 );
		_mm_store_ps( &C1[0+ldc*2], d_47_2 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C0[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C0[0+ldc*2] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		d_47_1 = _mm_load_ps( &C1[0+ldc*1] );
		d_47_2 = _mm_load_ps( &C1[0+ldc*2] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_sub_ps( d_03_2, c_03_2 );
		d_47_0 = _mm_sub_ps( d_47_0, c_47_0 );
		d_47_1 = _mm_sub_ps( d_47_1, c_47_1 );
		d_47_2 = _mm_sub_ps( d_47_2, c_47_2 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		_mm_store_ps( &C0[0+ldc*1], d_03_1 );
		_mm_store_ps( &C1[0+ldc*1], d_47_1 );
		_mm_store_ps( &C0[0+ldc*2], d_03_2 );
		_mm_store_ps( &C1[0+ldc*2], d_47_2 );
		}

	}



// normal-transposed, 8x2 with data packed in 4
void kernel_sgemm_pp_nt_8x2_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1,
		c_47_0, c_47_1,
		a_03, a_47,
		b_0, b_1, b_2; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();
	c_47_0 = _mm_setzero_ps();
	c_47_1 = _mm_setzero_ps();

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A0[4] );
		a_47 = _mm_load_ps( &A1[4] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );


		
		b_0 = _mm_load_ps( &B[8] );
		
		a_03 = _mm_load_ps( &A0[8] );
		a_47 = _mm_load_ps( &A1[8] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );



		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A0[12] );
		a_47 = _mm_load_ps( &A1[12] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );


		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	for(; k<kmax; k++)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_1 = _mm_add_ps( c_47_1, b_2 );
		
		A0 += 4;
		A1 += 4;
		B  += 4;
		
		}

	__m128
		d_03_0, d_03_1,
		d_47_0, d_47_1;

	if(alg==0)
		{
		_mm_store_ps( &C0[0+ldc*0], c_03_0 );
		_mm_store_ps( &C1[0+ldc*0], c_47_0 );
		_mm_store_ps( &C0[0+ldc*1], c_03_1 );
		_mm_store_ps( &C1[0+ldc*1], c_47_1 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C0[0+ldc*1] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		d_47_1 = _mm_load_ps( &C1[0+ldc*1] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );
		d_47_0 = _mm_add_ps( d_47_0, c_47_0 );
		d_47_1 = _mm_add_ps( d_47_1, c_47_1 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		_mm_store_ps( &C0[0+ldc*1], d_03_1 );
		_mm_store_ps( &C1[0+ldc*1], d_47_1 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C0[0+ldc*1] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		d_47_1 = _mm_load_ps( &C1[0+ldc*1] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );
		d_47_0 = _mm_sub_ps( d_47_0, c_47_0 );
		d_47_1 = _mm_sub_ps( d_47_1, c_47_1 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		_mm_store_ps( &C0[0+ldc*1], d_03_1 );
		_mm_store_ps( &C1[0+ldc*1], d_47_1 );
		}

	}



// normal-transposed, 8x1 with data packed in 4
void kernel_sgemm_pp_nt_8x1_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0,
		c_47_0,
		a_03, a_47,
		b_0, b_1, b_2; 
	
	c_03_0 = _mm_setzero_ps();
	c_47_0 = _mm_setzero_ps();

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A0[4] );
		a_47 = _mm_load_ps( &A1[4] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );


		
		b_0 = _mm_load_ps( &B[8] );
		
		a_03 = _mm_load_ps( &A0[8] );
		a_47 = _mm_load_ps( &A1[8] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		


		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A0[12] );
		a_47 = _mm_load_ps( &A1[12] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );


		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	
	for(; k<kmax; k++)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A0[0] );
		a_47 = _mm_load_ps( &A1[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_2 = b_1;
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		b_2 = _mm_mul_ps( a_47, b_2 );
		c_47_0 = _mm_add_ps( c_47_0, b_2 );
		
		A0 += 4;
		A1 += 4;
		B  += 4;
		
		}

	__m128
		d_03_0,
		d_47_0;

	if(alg==0)
		{
		_mm_store_ps( &C0[0+ldc*0], c_03_0 );
		_mm_store_ps( &C1[0+ldc*0], c_47_0 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_47_0 = _mm_add_ps( d_47_0, c_47_0 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C0[0+ldc*0] );
		d_47_0 = _mm_load_ps( &C1[0+ldc*0] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_47_0 = _mm_sub_ps( d_47_0, c_47_0 );

		_mm_store_ps( &C0[0+ldc*0], d_03_0 );
		_mm_store_ps( &C1[0+ldc*0], d_47_0 );
		}

	}



// normal-transposed, 4x4 with data packed in 4
void kernel_sgemm_pp_nt_4x4_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k;
	
	__m256
		temp,
		c_00, c_01, c_02, c_03,
		a_03,
		b_03, b_0, b_1, b_2, b_3; 
	
	c_00 = _mm256_setzero_ps();
	c_01 = _mm256_setzero_ps();
	c_02 = _mm256_setzero_ps();
	c_03 = _mm256_setzero_ps();

	k = 0;
	for(; k<kmax-7; k+=8)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_load_ps( &B[8] );
		
		a_03 = _mm256_load_ps( &A[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		b_03 = _mm256_load_ps( &B[16] );
		
		a_03 = _mm256_load_ps( &A[16] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_load_ps( &B[24] );
		
		a_03 = _mm256_load_ps( &A[24] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		A += 32;
		B += 32;

		}
	for(; k<kmax-3; k+=4)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		
		b_03 = _mm256_load_ps( &B[8] );
		
		a_03 = _mm256_load_ps( &A[8] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );


		A += 16;
		B += 16;

		}
	for(; k<kmax-1; k+=2)
		{
		
		b_03 = _mm256_load_ps( &B[0] );
		
		a_03 = _mm256_load_ps( &A[0] );
		
		b_0 = _mm256_shuffle_ps( b_03, b_03, 0 );
		temp = _mm256_mul_ps( a_03, b_0 );
		c_00 = _mm256_add_ps( c_00, temp );
		
		b_1 = _mm256_shuffle_ps( b_03, b_03, 85 );
		temp = _mm256_mul_ps( a_03, b_1 );
		c_01 = _mm256_add_ps( c_01, temp );
		
		b_2 = _mm256_shuffle_ps( b_03, b_03, 170 );
		temp = _mm256_mul_ps( a_03, b_2 );
		c_02 = _mm256_add_ps( c_02, temp );
	
		b_3 = _mm256_shuffle_ps( b_03, b_03, 255 );
		temp = _mm256_mul_ps( a_03, b_3 );
		c_03 = _mm256_add_ps( c_03, temp );

		A += 8;
		B += 8;
		
		}

	__m128
		ltemp,
		la_03,
		lb_03, lb_0, lb_1, lb_2, lb_3,
		lc_00, lc_01, lc_02, lc_03,
		hc_00, hc_01, hc_02, hc_03,
		d_00, d_01, d_02, d_03;
	
	hc_00 = _mm256_extractf128_ps ( c_00, 1 );
	lc_00 = _mm256_castps256_ps128( c_00 );
	lc_00 = _mm_add_ps( lc_00, hc_00 );
	hc_01 = _mm256_extractf128_ps ( c_01, 1 );
	lc_01 = _mm256_castps256_ps128( c_01 );
	lc_01 = _mm_add_ps( lc_01, hc_01 );
	hc_02 = _mm256_extractf128_ps ( c_02, 1 );
	lc_02 = _mm256_castps256_ps128( c_02 );
	lc_02 = _mm_add_ps( lc_02, hc_02 );
	hc_03 = _mm256_extractf128_ps ( c_03, 1 );
	lc_03 = _mm256_castps256_ps128( c_03 );
	lc_03 = _mm_add_ps( lc_03, hc_03 );

	for(; k<kmax; k++)
		{
		
		lb_03 = _mm_load_ps( &B[0] );
		
		la_03 = _mm_load_ps( &A[0] );
		
		lb_0 = _mm_shuffle_ps( lb_03, lb_03, 0 );
		ltemp = _mm_mul_ps( la_03, lb_0 );
		lc_00 = _mm_add_ps( lc_00, ltemp );
		
		lb_1 = _mm_shuffle_ps( lb_03, lb_03, 85 );
		ltemp = _mm_mul_ps( la_03, lb_1 );
		lc_01 = _mm_add_ps( lc_01, ltemp );
		
		lb_2 = _mm_shuffle_ps( lb_03, lb_03, 170 );
		ltemp = _mm_mul_ps( la_03, lb_2 );
		lc_02 = _mm_add_ps( lc_02, ltemp );
	
		lb_3 = _mm_shuffle_ps( lb_03, lb_03, 255 );
		ltemp = _mm_mul_ps( la_03, lb_3 );
		lc_03 = _mm_add_ps( lc_03, ltemp );

		A += 8;
		B += 8;
		
		}

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], lc_00 );
		_mm_store_ps( &C[0+ldc*1], lc_01 );
		_mm_store_ps( &C[0+ldc*2], lc_02 );
		_mm_store_ps( &C[0+ldc*3], lc_03 );
		}
	else
		{
		d_00 = _mm_load_ps( &C[0+ldc*0] );
		d_01 = _mm_load_ps( &C[0+ldc*1] );
		d_02 = _mm_load_ps( &C[0+ldc*2] );
		d_03 = _mm_load_ps( &C[0+ldc*3] );
		
		if(alg==1)
			{
			d_00 = _mm_add_ps( d_00, lc_00 );
			d_01 = _mm_add_ps( d_01, lc_01 );
			d_02 = _mm_add_ps( d_02, lc_02 );
			d_03 = _mm_add_ps( d_03, lc_03 );
			}
		else // alg == -1
			{
			d_00 = _mm_sub_ps( d_00, lc_00 );
			d_01 = _mm_sub_ps( d_01, lc_01 );
			d_02 = _mm_sub_ps( d_02, lc_02 );
			d_03 = _mm_sub_ps( d_03, lc_03 );
			}

		_mm_store_ps( &C[0+ldc*0], d_00 );
		_mm_store_ps( &C[0+ldc*1], d_01 );
		_mm_store_ps( &C[0+ldc*2], d_02 );
		_mm_store_ps( &C[0+ldc*3], d_03 );
		}

	}



// normal-transposed, 4x3 with data packed in 4
void kernel_sgemm_pp_nt_4x3_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1, c_03_2,
		a_03,
		b_0, b_1; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();
	c_03_2 = _mm_setzero_ps();

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



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A[4] );
		
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



		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A[12] );
		
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


		A += 16;
		B += 16;

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
		
		A += 4;
		B += 4;
		
		}

	__m128
		d_03_0, d_03_1, d_03_2;

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], c_03_0 );
		_mm_store_ps( &C[0+ldc*1], c_03_1 );
		_mm_store_ps( &C[0+ldc*2], c_03_2 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_add_ps( d_03_2, c_03_2 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_sub_ps( d_03_2, c_03_2 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_sgemm_pp_nt_4x2_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1,
		a_03,
		b_0, b_1; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();

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



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A[4] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );


		
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



		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A[12] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 85 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_1 = _mm_add_ps( c_03_1, b_1 );


		A += 16;
		B += 16;

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
		
		A += 4;
		B += 4;
		
		}

	__m128
		d_03_0, d_03_1;

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], c_03_0 );
		_mm_store_ps( &C[0+ldc*1], c_03_1 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		}

	}



// normal-transposed, 4x1 with data packed in 4
void kernel_sgemm_pp_nt_4x1_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0,
		a_03,
		b_0, b_1; 
	
	c_03_0 = _mm_setzero_ps();

	for(k=0; k<kmax-3; k+=4)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );



		b_0 = _mm_load_ps( &B[4] );
		
		a_03 = _mm_load_ps( &A[4] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );


		
		b_0 = _mm_load_ps( &B[8] );
		
		a_03 = _mm_load_ps( &A[8] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );



		b_0 = _mm_load_ps( &B[12] );
		
		a_03 = _mm_load_ps( &A[12] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );


		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{
		
		b_0 = _mm_load_ps( &B[0] );
		
		a_03 = _mm_load_ps( &A[0] );
		
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 0 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_0 = _mm_add_ps( c_03_0, b_1 );
		
		A += 4;
		B += 4;
		
		}

	__m128
		d_03_0;

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], c_03_0 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		}

	}

