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

#include <math.h>

#include "../../include/blas_d.h"



void kernel_dgetrf_r_nn_8x4_lib4(int kmax, double *A0, int sda, double *B, int sdb, int alg, double *C0, int sdc, double *D0, int sdd, double *inv_diag_D, double *E0, int sde)
	{

	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;
	double *E1 = E0 + 4*sde;

	__builtin_prefetch( B+0 );
	__builtin_prefetch( B+8 );

	const int bs = 4;

	const int B_next = bs*sdb;

	__builtin_prefetch( B+B_next+0 );
	__builtin_prefetch( B+B_next+8 );

	int k;

	__m128d
		ones,
		v_0;

	__m256d
		zeros,
		a_0, a_4, A_0, A_4,
		b_0,
		tmp,
		d_0, d_1, d_2, d_3,
		d_4, d_5, d_6, d_7,
		c_0, 
		e_0, t_0,
		u_0, u_1;
	

	// correction phase

	d_0 = _mm256_setzero_pd();
	d_1 = _mm256_setzero_pd();
	d_2 = _mm256_setzero_pd();
	d_3 = _mm256_setzero_pd();
	d_4 = _mm256_setzero_pd();
	d_5 = _mm256_setzero_pd();
	d_6 = _mm256_setzero_pd();
	d_7 = _mm256_setzero_pd();

	if(kmax<=0)
		goto add;

	// prefetch
	a_0 = _mm256_load_pd( &A0[0+bs*0] );
	a_4 = _mm256_load_pd( &A1[0+bs*0] );

	for(k=0; k<kmax-3; k+=4)
		{

		__builtin_prefetch( B+2*B_next+0 );

		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		A_0 = _mm256_load_pd( &A0[0+bs*1] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		A_4 = _mm256_load_pd( &A1[0+bs*1] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		b_0 = _mm256_broadcast_sd( &B[1+bs*0] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*1] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		a_4 = _mm256_load_pd( &A1[0+bs*2] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*2] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*3] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );

		__builtin_prefetch( B+2*B_next+8 );

		b_0 = _mm256_broadcast_sd( &B[2+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		A_0 = _mm256_load_pd( &A0[0+bs*3] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		A_4 = _mm256_load_pd( &A1[0+bs*3] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		b_0 = _mm256_broadcast_sd( &B[3+bs*0] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		a_0 = _mm256_load_pd( &A0[0+bs*4] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*1] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		a_4 = _mm256_load_pd( &A1[0+bs*4] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*2] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*3] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		A0 += 4*bs;
		A1 += 4*bs;
		B  += B_next;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_4 = _mm256_load_pd( &A1[0+bs*0] );
		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		A0 += 1*bs;
		A1 += 1*bs;
		B  += 1;

		}
	
	add:

	if(alg!=0)
		{
		c_0 = _mm256_load_pd( &C0[0+bs*0] );
		d_0 = _mm256_add_pd( c_0, d_0 );
		c_0 = _mm256_load_pd( &C0[0+bs*1] );
		d_1 = _mm256_add_pd( c_0, d_1 );
		c_0 = _mm256_load_pd( &C0[0+bs*2] );
		d_2 = _mm256_add_pd( c_0, d_2 );
		c_0 = _mm256_load_pd( &C0[0+bs*3] );
		d_3 = _mm256_add_pd( c_0, d_3 );
		c_0 = _mm256_load_pd( &C1[0+bs*0] );
		d_4 = _mm256_add_pd( c_0, d_4 );
		c_0 = _mm256_load_pd( &C1[0+bs*1] );
		d_5 = _mm256_add_pd( c_0, d_5 );
		c_0 = _mm256_load_pd( &C1[0+bs*2] );
		d_6 = _mm256_add_pd( c_0, d_6 );
		c_0 = _mm256_load_pd( &C1[0+bs*3] );
		d_7 = _mm256_add_pd( c_0, d_7 );
		}
	


	// solution of top & correction of bottom

	zeros = _mm256_setzero_pd();

	e_0 = _mm256_load_pd( &E0[0+bs*0] );
	a_0 = _mm256_load_pd( &E1[0+bs*0] );
	e_0 = _mm256_blend_pd( e_0, zeros, 0x1 );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_0 ) );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_0 = _mm256_sub_pd( d_0, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_4 = _mm256_sub_pd( d_4, tmp );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_1 ) );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_1 = _mm256_sub_pd( d_1, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_2 ) );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_3 ) );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );

	e_0 = _mm256_load_pd( &E0[0+bs*1] );
	a_0 = _mm256_load_pd( &E1[0+bs*1] );
	e_0 = _mm256_blend_pd( e_0, zeros, 0x3 );
	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_0 ), 0x3 );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_0 = _mm256_sub_pd( d_0, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_4 = _mm256_sub_pd( d_4, tmp );
	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_1 ), 0x3 );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_1 = _mm256_sub_pd( d_1, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_2 ), 0x3 );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_3 ), 0x3 );
	t_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x00 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );

	e_0 = _mm256_load_pd( &E0[0+bs*2] );
	a_0 = _mm256_load_pd( &E1[0+bs*2] );
	e_0 = _mm256_blend_pd( e_0, zeros, 0x7 );
	t_0 = _mm256_permute2f128_pd( d_0, d_0, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0x0 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_0 = _mm256_sub_pd( d_0, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_4 = _mm256_sub_pd( d_4, tmp );
	t_0 = _mm256_permute2f128_pd( d_1, d_1, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0x0 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_1 = _mm256_sub_pd( d_1, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	t_0 = _mm256_permute2f128_pd( d_2, d_2, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0x0 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	t_0 = _mm256_permute2f128_pd( d_3, d_3, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0x0 );
	tmp = _mm256_mul_pd( e_0, t_0 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );

	a_0 = _mm256_load_pd( &E1[0+bs*3] );
	t_0 = _mm256_permute2f128_pd( d_0, d_0, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0xf );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_4 = _mm256_sub_pd( d_4, tmp );
	t_0 = _mm256_permute2f128_pd( d_1, d_1, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0xf );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	t_0 = _mm256_permute2f128_pd( d_2, d_2, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0xf );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	t_0 = _mm256_permute2f128_pd( d_3, d_3, 0x11 );
	t_0 = _mm256_permute_pd( t_0, 0xf );
	tmp = _mm256_mul_pd( a_0, t_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );

	// store upper
	_mm256_store_pd( &D0[0+bs*0], d_0 );
	_mm256_store_pd( &D0[0+bs*1], d_1 );
	_mm256_store_pd( &D0[0+bs*2], d_2 );
	_mm256_store_pd( &D0[0+bs*3], d_3 );


	// factorization of bottom

	ones = _mm_set_pd( 1.0, 1.0 );


	// first column
	u_0 = _mm256_blend_pd( u_0, d_4, 0x1 );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_4 ) );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_4 = _mm256_mul_pd( d_4, u_1 );
	_mm_store_sd( &inv_diag_D[0], v_0 );
	d_4 = _mm256_blend_pd( d_4, u_0, 0x1 );
	_mm256_store_pd( &D1[0+bs*0], d_4 );


	// second column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_5 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	u_0 = _mm256_blend_pd( u_0, d_5, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_5 ), 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_5 = _mm256_mul_pd( d_5, u_1 );
	_mm_store_sd( &inv_diag_D[1], v_0 );
	d_5 = _mm256_blend_pd( d_5, u_0, 0x3 );
	_mm256_store_pd( &D1[0+bs*1], d_5 );


	// third column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_6 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	u_0 = _mm256_blend_pd( u_0, d_6, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_6 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_5, u_1 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	u_0 = _mm256_blend_pd( u_0, d_6, 0x4 );

	v_0 = _mm256_extractf128_pd( d_6, 0x1 );
	v_0 = _mm_movedup_pd( v_0 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_6 = _mm256_mul_pd( d_6, u_1 );
	_mm_store_sd( &inv_diag_D[2], v_0 );
	d_6 = _mm256_blend_pd( d_6, u_0, 0x7 );
	_mm256_store_pd( &D1[0+bs*2], d_6 );


	// fourth column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_7 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_7, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_7 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_5, u_1 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_7, 0x4 );

	u_1 = _mm256_permute2f128_pd( d_7, d_7, 0x11 );
	u_1 = _mm256_permute_pd( u_1, 0x0 );
	tmp = _mm256_mul_pd( d_6, u_1 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_7, 0x8 );
	_mm256_store_pd( &D1[0+bs*3], u_0 );

	v_0 = _mm256_extractf128_pd( d_7, 0x1 );
	v_0 = _mm_permute_pd( v_0, 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	_mm_store_sd( &inv_diag_D[3], v_0 );

	return;

	}



void kernel_dgetrf_l_nn_8x4_lib4(int kmax, double *A0, int sda, double *B, int sdb, int alg, double *C0, int sdc, double *D0, int sdd, double *inv_diag_D)
	{

	double *A1 = A0 + 4*sda;
	double *C1 = C0 + 4*sdc;
	double *D1 = D0 + 4*sdd;

	__builtin_prefetch( B+0 );
	__builtin_prefetch( B+8 );

	const int bs = 4;

	const int B_next = bs*sdb;

	__builtin_prefetch( B+B_next+0 );
	__builtin_prefetch( B+B_next+8 );

	int k;

	__m128d
		ones,
		v_0;

	__m256d
		a_0, a_4, A_0, A_4,
		b_0,
		tmp,
		d_0, d_1, d_2, d_3,
		d_4, d_5, d_6, d_7,
		c_0, 
		u_0, u_1;
	

	// correction phase

	d_0 = _mm256_setzero_pd();
	d_1 = _mm256_setzero_pd();
	d_2 = _mm256_setzero_pd();
	d_3 = _mm256_setzero_pd();
	d_4 = _mm256_setzero_pd();
	d_5 = _mm256_setzero_pd();
	d_6 = _mm256_setzero_pd();
	d_7 = _mm256_setzero_pd();

	if(kmax<=0)
		goto add;

	// prefetch
	a_0 = _mm256_load_pd( &A0[0+bs*0] );
	a_4 = _mm256_load_pd( &A1[0+bs*0] );

	for(k=0; k<kmax-3; k+=4)
		{

		__builtin_prefetch( B+2*B_next+0 );

		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		A_0 = _mm256_load_pd( &A0[0+bs*1] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		A_4 = _mm256_load_pd( &A1[0+bs*1] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		b_0 = _mm256_broadcast_sd( &B[1+bs*0] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*1] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		a_4 = _mm256_load_pd( &A1[0+bs*2] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*2] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*3] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );

		__builtin_prefetch( B+2*B_next+8 );

		b_0 = _mm256_broadcast_sd( &B[2+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		A_0 = _mm256_load_pd( &A0[0+bs*3] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		A_4 = _mm256_load_pd( &A1[0+bs*3] );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		b_0 = _mm256_broadcast_sd( &B[3+bs*0] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		a_0 = _mm256_load_pd( &A0[0+bs*4] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*1] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		a_4 = _mm256_load_pd( &A1[0+bs*4] );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*2] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*3] );
		tmp = _mm256_mul_pd( A_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( A_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		A0 += 4*bs;
		A1 += 4*bs;
		B  += B_next;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_4 = _mm256_load_pd( &A1[0+bs*0] );
		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_4 = _mm256_sub_pd( d_4, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_5 = _mm256_sub_pd( d_5, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_6 = _mm256_sub_pd( d_6, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );
		tmp = _mm256_mul_pd( a_4, b_0 );
		d_7 = _mm256_sub_pd( d_7, tmp );


		A0 += 1*bs;
		A1 += 1*bs;
		B  += 1;

		}
	
	add:

	if(alg!=0)
		{
		c_0 = _mm256_load_pd( &C0[0+bs*0] );
		d_0 = _mm256_add_pd( c_0, d_0 );
		c_0 = _mm256_load_pd( &C0[0+bs*1] );
		d_1 = _mm256_add_pd( c_0, d_1 );
		c_0 = _mm256_load_pd( &C0[0+bs*2] );
		d_2 = _mm256_add_pd( c_0, d_2 );
		c_0 = _mm256_load_pd( &C0[0+bs*3] );
		d_3 = _mm256_add_pd( c_0, d_3 );
		c_0 = _mm256_load_pd( &C1[0+bs*0] );
		d_4 = _mm256_add_pd( c_0, d_4 );
		c_0 = _mm256_load_pd( &C1[0+bs*1] );
		d_5 = _mm256_add_pd( c_0, d_5 );
		c_0 = _mm256_load_pd( &C1[0+bs*2] );
		d_6 = _mm256_add_pd( c_0, d_6 );
		c_0 = _mm256_load_pd( &C1[0+bs*3] );
		d_7 = _mm256_add_pd( c_0, d_7 );
		}
	


	// factorization

	ones = _mm_set_pd( 1.0, 1.0 );


	// first column
	u_0 = _mm256_blend_pd( u_0, d_0, 0x1 );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_0 ) );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_0 = _mm256_mul_pd( d_0, u_1 );
	d_4 = _mm256_mul_pd( d_4, u_1 );
	_mm_store_sd( &inv_diag_D[0], v_0 );
	d_0 = _mm256_blend_pd( d_0, u_0, 0x1 );
	_mm256_store_pd( &D0[0+bs*0], d_0 );
	_mm256_store_pd( &D1[0+bs*0], d_4 );


	// second column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_1 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_0, u_0 );
	d_1 = _mm256_sub_pd( d_1, tmp );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_5 = _mm256_sub_pd( d_5, tmp );
	u_0 = _mm256_blend_pd( u_0, d_1, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_1 ), 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_1 = _mm256_mul_pd( d_1, u_1 );
	d_5 = _mm256_mul_pd( d_5, u_1 );
	_mm_store_sd( &inv_diag_D[1], v_0 );
	d_1 = _mm256_blend_pd( d_1, u_0, 0x3 );
	_mm256_store_pd( &D0[0+bs*1], d_1 );
	_mm256_store_pd( &D1[0+bs*1], d_5 );


	// third column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_2 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_0, u_0 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	u_0 = _mm256_blend_pd( u_0, d_2, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_2 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_1, u_1 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	tmp = _mm256_mul_pd( d_5, u_1 );
	d_6 = _mm256_sub_pd( d_6, tmp );
	u_0 = _mm256_blend_pd( u_0, d_2, 0x4 );

	v_0 = _mm256_extractf128_pd( d_2, 0x1 );
	v_0 = _mm_movedup_pd( v_0 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_2 = _mm256_mul_pd( d_2, u_1 );
	d_6 = _mm256_mul_pd( d_6, u_1 );
	_mm_store_sd( &inv_diag_D[2], v_0 );
	d_2 = _mm256_blend_pd( d_2, u_0, 0x7 );
	_mm256_store_pd( &D0[0+bs*2], d_2 );
	_mm256_store_pd( &D1[0+bs*2], d_6 );


	// fourth column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_3 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_0, u_0 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( d_4, u_0 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_3 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( d_1, u_1 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( d_5, u_1 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x4 );

	u_1 = _mm256_permute2f128_pd( d_3, d_3, 0x11 );
	u_1 = _mm256_permute_pd( u_1, 0x0 );
	tmp = _mm256_mul_pd( d_2, u_1 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	tmp = _mm256_mul_pd( d_6, u_1 );
	d_7 = _mm256_sub_pd( d_7, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x8 );
	_mm256_store_pd( &D0[0+bs*3], u_0 );

	v_0 = _mm256_extractf128_pd( d_3, 0x1 );
	v_0 = _mm_permute_pd( v_0, 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	_mm_store_sd( &inv_diag_D[3], v_0 );
	d_7 = _mm256_mul_pd( d_7, u_1 );
	_mm256_store_pd( &D1[0+bs*3], d_7 );

	return;

	}



void kernel_dgetrf_nn_4x4_lib4(int kmax, double *A, double *B, int sdb, int alg, double *C, double *D, double *inv_diag_D)
	{

	const int bs = 4;

	int k;

	__m128d
		ones,
		v_0;

	__m256d
		a_0,
		b_0,
		tmp,
		d_0, d_1, d_2, d_3,
		c_0, 
		u_0, u_1;
	

	// correction phase

	d_0 = _mm256_setzero_pd();
	d_1 = _mm256_setzero_pd();
	d_2 = _mm256_setzero_pd();
	d_3 = _mm256_setzero_pd();

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A[0+bs*0] );
		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );


		a_0 = _mm256_load_pd( &A[0+bs*1] );
		b_0 = _mm256_broadcast_sd( &B[1+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		b_0 = _mm256_broadcast_sd( &B[1+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );


		a_0 = _mm256_load_pd( &A[0+bs*2] );
		b_0 = _mm256_broadcast_sd( &B[2+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		b_0 = _mm256_broadcast_sd( &B[2+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );


		a_0 = _mm256_load_pd( &A[0+bs*3] );
		b_0 = _mm256_broadcast_sd( &B[3+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		b_0 = _mm256_broadcast_sd( &B[3+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );


		A += 4*bs;
		B += bs*sdb;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A[0+bs*0] );
		b_0 = _mm256_broadcast_sd( &B[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_0 = _mm256_sub_pd( d_0, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_1 = _mm256_sub_pd( d_1, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_2 = _mm256_sub_pd( d_2, tmp );
		b_0 = _mm256_broadcast_sd( &B[0+bs*3] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		d_3 = _mm256_sub_pd( d_3, tmp );


		A += 1*bs;
		B += 1;

		}
	
	if(alg!=0)
		{
		c_0 = _mm256_load_pd( &C[0+bs*0] );
		d_0 = _mm256_add_pd( c_0, d_0 );
		c_0 = _mm256_load_pd( &C[0+bs*1] );
		d_1 = _mm256_add_pd( c_0, d_1 );
		c_0 = _mm256_load_pd( &C[0+bs*2] );
		d_2 = _mm256_add_pd( c_0, d_2 );
		c_0 = _mm256_load_pd( &C[0+bs*3] );
		d_3 = _mm256_add_pd( c_0, d_3 );
		}
	


	// factorization

	ones = _mm_set_pd( 1.0, 1.0 );


	// first column
	u_0 = _mm256_blend_pd( u_0, d_0, 0x1 );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_0 ) );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_0 = _mm256_mul_pd( d_0, u_1 );
	_mm_store_sd( &inv_diag_D[0], v_0 );
	d_0 = _mm256_blend_pd( d_0, u_0, 0x1 );
	_mm256_store_pd( &D[0+bs*0], d_0 );


	// second column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_1 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, d_0 );
	d_1 = _mm256_sub_pd( d_1, tmp );
	u_0 = _mm256_blend_pd( u_0, d_1, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_1 ), 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_1 = _mm256_mul_pd( d_1, u_1 );
	_mm_store_sd( &inv_diag_D[1], v_0 );
	d_1 = _mm256_blend_pd( d_1, u_0, 0x3 );
	_mm256_store_pd( &D[0+bs*1], d_1 );


	// third column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_2 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, d_0 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	u_0 = _mm256_blend_pd( u_0, d_2, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_2 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_1, d_1 );
	d_2 = _mm256_sub_pd( d_2, tmp );
	u_0 = _mm256_blend_pd( u_0, d_2, 0x4 );

	v_0 = _mm256_extractf128_pd( d_2, 0x1 );
	v_0 = _mm_movedup_pd( v_0 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	d_2 = _mm256_mul_pd( d_2, u_1 );
	_mm_store_sd( &inv_diag_D[2], v_0 );
	d_2 = _mm256_blend_pd( d_2, u_0, 0x7 );
	_mm256_store_pd( &D[0+bs*2], d_2 );


	// fourth column
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( d_3 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, d_0 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( d_3 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_1, d_1 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x4 );

	u_1 = _mm256_permute2f128_pd( d_3, d_3, 0x11 );
	u_1 = _mm256_permute_pd( u_1, 0x0 );
	tmp = _mm256_mul_pd( u_1, d_2 );
	d_3 = _mm256_sub_pd( d_3, tmp );
	u_0 = _mm256_blend_pd( u_0, d_3, 0x8 );
	_mm256_store_pd( &D[0+bs*3], u_0 );

	v_0 = _mm256_extractf128_pd( d_3, 0x1 );
	v_0 = _mm_permute_pd( v_0, 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	_mm_store_sd( &inv_diag_D[3], v_0 );

	return;

	}



void corner_dgetrf_nn_4x4_lib4(double *C, double *LU, double *inv_diag_U)
	{

	const int bs = 4;

	__m128d
		ones,
		v_0;

	__m256d
		tmp,
		c_0, c_1, c_2, c_3,
		u_0, u_1;
	
	ones = _mm_set_pd( 1.0, 1.0 );



	// first column
	c_0 = _mm256_load_pd( &C[0+bs*0] );

	u_0 = _mm256_blend_pd( u_0, c_0, 0x1 );
	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( c_0 ) );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	c_0 = _mm256_mul_pd( c_0, u_1 );
	_mm_store_sd( &inv_diag_U[0], v_0 );
	c_0 = _mm256_blend_pd( c_0, u_0, 0x1 );
	_mm256_store_pd( &LU[0+bs*0], c_0 );



	// second column
	c_1 = _mm256_load_pd( &C[0+bs*1] );

	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( c_1 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, c_0 );
	c_1 = _mm256_sub_pd( c_1, tmp );
	u_0 = _mm256_blend_pd( u_0, c_1, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( c_1 ), 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	c_1 = _mm256_mul_pd( c_1, u_1 );
	_mm_store_sd( &inv_diag_U[1], v_0 );
	c_1 = _mm256_blend_pd( c_1, u_0, 0x3 );
	_mm256_store_pd( &LU[0+bs*1], c_1 );



	// third column
	c_2 = _mm256_load_pd( &C[0+bs*2] );

	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( c_2 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, c_0 );
	c_2 = _mm256_sub_pd( c_2, tmp );
	u_0 = _mm256_blend_pd( u_0, c_2, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( c_2 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_1, c_1 );
	c_2 = _mm256_sub_pd( c_2, tmp );
	u_0 = _mm256_blend_pd( u_0, c_2, 0x4 );

	v_0 = _mm256_extractf128_pd( c_2, 0x1 );
	v_0 = _mm_movedup_pd( v_0 );
	v_0 = _mm_div_pd( ones, v_0 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	c_2 = _mm256_mul_pd( c_2, u_1 );
	_mm_store_sd( &inv_diag_U[2], v_0 );
	c_2 = _mm256_blend_pd( c_2, u_0, 0x7 );
	_mm256_store_pd( &LU[0+bs*2], c_2 );



	// fourth column
	c_3 = _mm256_load_pd( &C[0+bs*3] );

	v_0 = _mm_movedup_pd( _mm256_castpd256_pd128( c_3 ) );
	u_0 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_0, c_0 );
	c_3 = _mm256_sub_pd( c_3, tmp );
	u_0 = _mm256_blend_pd( u_0, c_3, 0x2 );

	v_0 = _mm_permute_pd( _mm256_castpd256_pd128( c_3 ), 0x3 );
	u_1 = _mm256_permute2f128_pd( _mm256_castpd128_pd256( v_0 ), _mm256_castpd128_pd256( v_0 ), 0x0 );
	tmp = _mm256_mul_pd( u_1, c_1 );
	c_3 = _mm256_sub_pd( c_3, tmp );
	u_0 = _mm256_blend_pd( u_0, c_3, 0x4 );

	u_1 = _mm256_permute2f128_pd( c_3, c_3, 0x11 );
	u_1 = _mm256_permute_pd( u_1, 0x0 );
	tmp = _mm256_mul_pd( u_1, c_2 );
	c_3 = _mm256_sub_pd( c_3, tmp );
	u_0 = _mm256_blend_pd( u_0, c_3, 0x8 );
	_mm256_store_pd( &LU[0+bs*3], u_0 );

	v_0 = _mm256_extractf128_pd( c_3, 0x1 );
	v_0 = _mm_permute_pd( v_0, 0x3 );
	v_0 = _mm_div_pd( ones, v_0 );
	_mm_store_sd( &inv_diag_U[3], v_0 );


	}



// C numbering, starting from 0
void idamax_lib4(int n, int offset, double *pA, int sda, int *p_idamax, double *p_amax)
	{

	const int bs = 4;

#if 0
pA -= offset%4;
pA[0] = 1.3;
pA[1] = 1.9;
pA[2] = 1.2;
pA[3] = 1.3;
pA[0+1*sda*bs] = -1.4;
pA[1+1*sda*bs] = 1.4;
pA[2+1*sda*bs] = 0.4;
pA[3+1*sda*bs] = 0.4;
pA[0+2*sda*bs] = 0.4;
pA[1+2*sda*bs] = 0.4;
pA[2+2*sda*bs] = 0.4;
pA[3+2*sda*bs] = 1.4;
pA += offset%4;
d_print_pmat( n+offset%4, 1, bs, pA-offset%4, sda);
#endif

	int idamax, ii;
	double tmp, amax;
		
	p_idamax[0] = -1;
	if(n<1)
		return;

	int na = (bs - offset%bs)%bs;
	na = n<na ? n : na;

	double dna, didamax, dmx;

	__m256d
		vna, idx, imx, max, msk, a_0, sng;
	
	__m128d
		max0, max1, msk0, imx0, imx1;

	amax = -1.0;
	ii = 0;
	if(na>0)
		{
		for( ; ii<na; ii++)
			{
			tmp = fabs(pA[0]);
			if(tmp>amax)
				{
				idamax = ii+0;
				amax = tmp;
				}
			pA += 1;
			}
		pA += bs*(sda-1);
		}
#if 1
	// XXX implemented using doubles since in AVX there is no support for SIMD integer
	dna = (double) na;
	didamax = (double) idamax;
	vna = _mm256_broadcast_sd( &dna );
	idx = _mm256_set_pd( 3.0, 2.0, 1.0, 0.0 );
	idx = _mm256_add_pd( vna, idx );
	vna = _mm256_set_pd( 4.0, 4.0, 4.0, 4.0 );
	imx = _mm256_broadcast_sd( &didamax );
	max = _mm256_broadcast_sd( &amax );
	sng = _mm256_set_pd( -0.0, -0.0, -0.0, -0.0 );
	for( ; ii<n-7; ii+=8)
		{
		a_0 = _mm256_load_pd( &pA[0] );
		a_0 = _mm256_andnot_pd( sng, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 );
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pA += bs*sda;
		a_0 = _mm256_load_pd( &pA[0] );
		a_0 = _mm256_andnot_pd( sng, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 );
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pA += bs*sda;
		}
	for( ; ii<n-3; ii+=4)
		{
		a_0 = _mm256_load_pd( &pA[0] );
		a_0 = _mm256_andnot_pd( sng, a_0 ); // abs
		msk = _mm256_cmp_pd( a_0, max, 14 );
		max = _mm256_blendv_pd( max, a_0, msk );
		imx = _mm256_blendv_pd( imx, idx, msk );
		idx = _mm256_add_pd( idx, vna );
		pA += bs*sda;
		}
	// reduction 2
	max0 = _mm256_extractf128_pd( max, 0x0 );
	max1 = _mm256_extractf128_pd( max, 0x1 );
	imx0 = _mm256_extractf128_pd( imx, 0x0 ); // lower indexes in case of identical max value
	imx1 = _mm256_extractf128_pd( imx, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );

	// reduction 1
	max1 = _mm_permute_pd( max0, 0x1 );
	imx1 = _mm_permute_pd( imx0, 0x1 );
	msk0 = _mm_cmp_pd( max1, max0, 14 );
	max0 = _mm_blendv_pd( max0, max1, msk0 );
	imx0 = _mm_blendv_pd( imx0, imx1, msk0 );

	_mm_store_sd( &dmx, max0 );
	_mm_store_sd( &didamax, imx0 );

//	printf("\n%f %f\n", dmx, didamax);

//	dmx = amax + 1.0; //
	if(dmx>amax)
		{
		amax = dmx;
		idamax = round(didamax);
		}
#else
	for( ; ii<n-3; ii+=4)
		{
		tmp = fabs(pA[0]);
		if(tmp>amax)
			{
			idamax = ii+0;
			amax = tmp;
			}
		tmp = fabs(pA[1]);
		if(tmp>amax)
			{
			idamax = ii+1;
			amax = tmp;
			}
		tmp = fabs(pA[2]);
		if(tmp>amax)
			{
			idamax = ii+2;
			amax = tmp;
			}
		tmp = fabs(pA[3]);
		if(tmp>amax)
			{
			idamax = ii+3;
			amax = tmp;
			}
		pA += bs*sda;
		}
#endif
	for( ; ii<n; ii++)
		{
		tmp = fabs(pA[0]);
		if(tmp>amax)
			{
			idamax = ii+1;
			}
		pA += 1;
		}
	
	p_amax[0] = amax;
	p_idamax[0] = idamax;

	return;

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_4_lib4(int m, double *pA, int sda, double *inv_diag_A, int* ipiv)
	{

	const int bs = 4;

	// assume m>=4
	int ma = m-4;

	__m256d
		tmp,
		a_0,
		b_0, b_1, b_2,
		c_0;

	double
		tmp0, tmp1, tmp2, tmp3,
		u_00, u_01, u_02, u_03,
		      u_11, u_12, u_13,
		            u_22, u_23,
		                  u_33;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	// first column
	idamax_lib4(m-0, 0, &pA[0+bs*0], sda, &idamax, &tmp0);
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			drowsw_lib(4, pA+0, pA+ipiv[0]/bs*bs*sda+ipiv[0]%bs);

		tmp0 = 1.0 / pA[0+bs*0];
		inv_diag_A[0] = tmp0;
		pA[1+bs*0] *= tmp0;
		pA[2+bs*0] *= tmp0;
		pA[3+bs*0] *= tmp0;
		pB = pA + bs*sda;
#if 1
		b_0 = _mm256_broadcast_sd( &tmp0 );
		k = 0;
		for(; k<ma-7; k+=8)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*0] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*0], c_0 );
			pB += bs*sda;
			c_0 = _mm256_load_pd( &pB[0+bs*0] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*0], c_0 );
			pB += bs*sda;
			}
		for(; k<ma-3; k+=4)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*0] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*0], c_0 );
			pB += bs*sda;
			}
#else
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*0] *= tmp0;
			pB[1+bs*0] *= tmp0;
			pB[2+bs*0] *= tmp0;
			pB[3+bs*0] *= tmp0;
			pB += bs*sda;
			}
#endif
		for( ; k<ma; k++)
			{
			pB[0+bs*0] *= tmp0;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[0] = 0.0;
		}

	// second column
	u_01  = pA[0+bs*1];
	tmp1  = pA[1+bs*1];
	tmp2  = pA[2+bs*1];
	tmp3  = pA[3+bs*1];
	tmp1 -= pA[1+bs*0] * u_01;
	tmp2 -= pA[2+bs*0] * u_01;
	tmp3 -= pA[3+bs*0] * u_01;
	pA[1+bs*1] = tmp1;
	pA[2+bs*1] = tmp2;
	pA[3+bs*1] = tmp3;
	pB = pA + bs*sda;
#if 1
	b_0 = _mm256_broadcast_sd( &u_01 );
	k = 0;
	for(; k<ma-7; k+=8)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		pB += bs*sda;
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		pB += bs*sda;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*1] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*1], c_0 );
		pB += bs*sda;
		}
#else
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*1];
		tmp1  = pB[1+bs*1];
		tmp2  = pB[2+bs*1];
		tmp3  = pB[3+bs*1];
		tmp0 -= pB[0+bs*0] * u_01;
		tmp1 -= pB[1+bs*0] * u_01;
		tmp2 -= pB[2+bs*0] * u_01;
		tmp3 -= pB[3+bs*0] * u_01;
		pB[0+bs*1] = tmp0;
		pB[1+bs*1] = tmp1;
		pB[2+bs*1] = tmp2;
		pB[3+bs*1] = tmp3;
		pB += bs*sda;
		}
#endif
	for( ; k<ma; k++)
		{
		tmp0 = pB[0+bs*1];
		tmp0 -= pB[0+bs*0] * u_01;
		pB[0+bs*1] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-1, 1, &pA[1+bs*1], sda, &idamax, &tmp1);
	ipiv[1] = idamax+1;
	if(tmp1!=0)
		{
		if(ipiv[1]!=1)
			drowsw_lib(4, pA+1, pA+ipiv[1]/bs*bs*sda+ipiv[1]%bs);

		tmp1 = 1.0 / pA[1+bs*1];
		inv_diag_A[1] = tmp1;
		pA[2+bs*1] *= tmp1;
		pA[3+bs*1] *= tmp1;
		pB = pA + bs*sda;
#if 1
		b_0 = _mm256_broadcast_sd( &tmp1 );
		k = 0;
		for(; k<ma-7; k+=8)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*1] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*1], c_0 );
			pB += bs*sda;
			c_0 = _mm256_load_pd( &pB[0+bs*1] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*1], c_0 );
			pB += bs*sda;
			}
		for(; k<ma-3; k+=4)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*1] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*1], c_0 );
			pB += bs*sda;
			}
#else
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*1] *= tmp1;
			pB[1+bs*1] *= tmp1;
			pB[2+bs*1] *= tmp1;
			pB[3+bs*1] *= tmp1;
			pB += bs*sda;
			}
#endif
		for( ; k<ma; k++)
			{
			pB[0+bs*1] *= tmp1;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[1] = 0.0;
		}

	// third column
	u_02  = pA[0+bs*2];
	u_12  = pA[1+bs*2];
	u_12 -= pA[1+bs*0] * u_02;
	pA[1+bs*2] = u_12;
	tmp2  = pA[2+bs*2];
	tmp3  = pA[3+bs*2];
	tmp2 -= pA[2+bs*0] * u_02;
	tmp3 -= pA[3+bs*0] * u_02;
	tmp2 -= pA[2+bs*1] * u_12;
	tmp3 -= pA[3+bs*1] * u_12;
	pA[2+bs*2] = tmp2;
	pA[3+bs*2] = tmp3;
	pB = pA + bs*sda;
#if 1
	b_0 = _mm256_broadcast_sd( &u_02 );
	b_1 = _mm256_broadcast_sd( &u_12 );
	k = 0;
	for(; k<ma-7; k+=8)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		pB += bs*sda;
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		pB += bs*sda;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*2] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*2], c_0 );
		pB += bs*sda;
		}
#else
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*2];
		tmp1  = pB[1+bs*2];
		tmp2  = pB[2+bs*2];
		tmp3  = pB[3+bs*2];
		tmp0 -= pB[0+bs*0] * u_02;
		tmp1 -= pB[1+bs*0] * u_02;
		tmp2 -= pB[2+bs*0] * u_02;
		tmp3 -= pB[3+bs*0] * u_02;
		tmp0 -= pB[0+bs*1] * u_12;
		tmp1 -= pB[1+bs*1] * u_12;
		tmp2 -= pB[2+bs*1] * u_12;
		tmp3 -= pB[3+bs*1] * u_12;
		pB[0+bs*2] = tmp0;
		pB[1+bs*2] = tmp1;
		pB[2+bs*2] = tmp2;
		pB[3+bs*2] = tmp3;
		pB += bs*sda;
		}
#endif
	for( ; k<ma; k++)
		{
		tmp0  = pB[0+bs*2];
		tmp0 -= pB[0+bs*0] * u_02;
		tmp0 -= pB[0+bs*1] * u_12;
		pB[0+bs*2] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-2, 2, &pA[2+bs*2], sda, &idamax, &tmp2);
	ipiv[2] = idamax+2;
	if(tmp2!=0)
		{
		if(ipiv[2]!=2)
			drowsw_lib(4, pA+2, pA+ipiv[2]/bs*bs*sda+ipiv[2]%bs);

		tmp2 = 1.0 / pA[2+bs*2];
		inv_diag_A[2] = tmp2;
		pA[3+bs*2] *= tmp2;
		pB = pA + bs*sda;
#if 1
		b_0 = _mm256_broadcast_sd( &tmp2 );
		k = 0;
		for(; k<ma-7; k+=8)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*2] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*2], c_0 );
			pB += bs*sda;
			c_0 = _mm256_load_pd( &pB[0+bs*2] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*2], c_0 );
			pB += bs*sda;
			}
		for(; k<ma-3; k+=4)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*2] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*2], c_0 );
			pB += bs*sda;
			}
#else
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*2] *= tmp2;
			pB[1+bs*2] *= tmp2;
			pB[2+bs*2] *= tmp2;
			pB[3+bs*2] *= tmp2;
			pB += bs*sda;
			}
#endif
		for( ; k<ma; k++)
			{
			pB[0+bs*2] *= tmp2;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[2] = 0.0;
		}

	// fourth column
	u_03  = pA[0+bs*3];
	u_13  = pA[1+bs*3];
	u_13 -= pA[1+bs*0] * u_03;
	pA[1+bs*3] = u_13;
	u_23  = pA[2+bs*3];
	u_23 -= pA[2+bs*0] * u_03;
	u_23 -= pA[2+bs*1] * u_13;
	pA[2+bs*3] = u_23;
	tmp3  = pA[3+bs*3];
	tmp3 -= pA[3+bs*0] * u_03;
	tmp3 -= pA[3+bs*1] * u_13;
	tmp3 -= pA[3+bs*2] * u_23;
	pA[3+bs*3] = tmp3;
	pB = pA + bs*sda;
#if 1
	b_0 = _mm256_broadcast_sd( &u_03 );
	b_1 = _mm256_broadcast_sd( &u_13 );
	b_2 = _mm256_broadcast_sd( &u_23 );
	k = 0;
	for(; k<ma-7; k+=8)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += bs*sda;
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += bs*sda;
		}
	for(; k<ma-3; k+=4)
		{
		c_0 = _mm256_load_pd( &pB[0+bs*3] );
		a_0 = _mm256_load_pd( &pB[0+bs*0] );
		tmp = _mm256_mul_pd( a_0, b_0 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*1] );
		tmp = _mm256_mul_pd( a_0, b_1 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		a_0 = _mm256_load_pd( &pB[0+bs*2] );
		tmp = _mm256_mul_pd( a_0, b_2 );
		c_0 = _mm256_sub_pd( c_0, tmp );
		_mm256_store_pd( &pB[0+bs*3], c_0 );
		pB += bs*sda;
		}
#else
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*3];
		tmp1  = pB[1+bs*3];
		tmp2  = pB[2+bs*3];
		tmp3  = pB[3+bs*3];
		tmp0 -= pB[0+bs*0] * u_03;
		tmp1 -= pB[1+bs*0] * u_03;
		tmp2 -= pB[2+bs*0] * u_03;
		tmp3 -= pB[3+bs*0] * u_03;
		tmp0 -= pB[0+bs*1] * u_13;
		tmp1 -= pB[1+bs*1] * u_13;
		tmp2 -= pB[2+bs*1] * u_13;
		tmp3 -= pB[3+bs*1] * u_13;
		tmp0 -= pB[0+bs*2] * u_23;
		tmp1 -= pB[1+bs*2] * u_23;
		tmp2 -= pB[2+bs*2] * u_23;
		tmp3 -= pB[3+bs*2] * u_23;
		pB[0+bs*3] = tmp0;
		pB[1+bs*3] = tmp1;
		pB[2+bs*3] = tmp2;
		pB[3+bs*3] = tmp3;
		pB += bs*sda;
		}
#endif
	for( ; k<ma; k++)
		{
		tmp0  = pB[0+bs*3];
		tmp0 -= pB[0+bs*0] * u_03;
		tmp0 -= pB[0+bs*1] * u_13;
		tmp0 -= pB[0+bs*2] * u_23;
		pB[0+bs*3] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-3, 3, &pA[3+bs*3], sda, &idamax, &tmp3);
	ipiv[3] = idamax+3;
	if(tmp3!=0)
		{
		if(ipiv[3]!=3)
			drowsw_lib(4, pA+3, pA+ipiv[3]/bs*bs*sda+ipiv[3]%bs);

		tmp3 = 1.0 / pA[3+bs*3];
		inv_diag_A[3] = tmp3;
		pB = pA + bs*sda;
#if 1
		b_0 = _mm256_broadcast_sd( &tmp3 );
		k = 0;
		for(; k<ma-7; k+=8)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*3] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*3], c_0 );
			pB += bs*sda;
			c_0 = _mm256_load_pd( &pB[0+bs*3] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*3], c_0 );
			pB += bs*sda;
			}
		for(; k<ma-3; k+=4)
			{
			c_0 = _mm256_load_pd( &pB[0+bs*3] );
			c_0 = _mm256_mul_pd( c_0, b_0 );
			_mm256_store_pd( &pB[0+bs*3], c_0 );
			pB += bs*sda;
			}
#else
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*3] *= tmp3;
			pB[1+bs*3] *= tmp3;
			pB[2+bs*3] *= tmp3;
			pB[3+bs*3] *= tmp3;
			pB += bs*sda;
			}
#endif
		for( ; k<ma; k++)
			{
			pB[0+bs*3] *= tmp3;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[3] = 0.0;
		}
	
	return;

	}


	


