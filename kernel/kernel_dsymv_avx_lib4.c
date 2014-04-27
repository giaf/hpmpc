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

// it moves vertically across blocks
void kernel_dsymv_4_lib4(int kmax, int kna, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;

/*printf("\nciao %d\n", kmax);	*/
	const int bs = 4;
	
	__builtin_prefetch( A + bs*0 );
	__builtin_prefetch( A + bs*2 );

	int k, ka;
	ka = kmax-kna; // number from aligned positon
	
	double *sA, *sy_n, *sx_t;

	__m256d
		zeros, temp,
		a_00, a_01, a_02, a_03,
		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,
		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;

	__m128d
		stemp,
		sa_00, sa_01, sa_02, sa_03,
		sx_n_0, sx_n_1, sx_n_2, sx_n_3, sy_n_0,
		sx_t_0, sy_t_0, sy_t_1, sy_t_2, sy_t_3;
	
	zeros = _mm256_setzero_pd();

	x_n_0 = _mm256_broadcast_sd( &x_n[0] );
	x_n_1 = _mm256_broadcast_sd( &x_n[1] );
	x_n_2 = _mm256_broadcast_sd( &x_n[2] );
	x_n_3 = _mm256_broadcast_sd( &x_n[3] );

	if(alg==-1)
		{
		x_n_0 = _mm256_sub_pd( zeros, x_n_0 );
		x_n_1 = _mm256_sub_pd( zeros, x_n_1 );
		x_n_2 = _mm256_sub_pd( zeros, x_n_2 );
		x_n_3 = _mm256_sub_pd( zeros, x_n_3 );
		}

	y_t_0 = _mm256_setzero_pd();
	y_t_1 = _mm256_setzero_pd();
	y_t_2 = _mm256_setzero_pd();
	y_t_3 = _mm256_setzero_pd();
	
	sx_n_0 = _mm256_castpd256_pd128( x_n_0 );
	sx_n_1 = _mm256_castpd256_pd128( x_n_1 );
	sx_n_2 = _mm256_castpd256_pd128( x_n_2 );
	sx_n_3 = _mm256_castpd256_pd128( x_n_3 );

	sy_t_0 = _mm256_castpd256_pd128( y_t_0 );
	sy_t_1 = _mm256_castpd256_pd128( y_t_1 );
	sy_t_2 = _mm256_castpd256_pd128( y_t_2 );
	sy_t_3 = _mm256_castpd256_pd128( y_t_3 );

	if(kna>0) // it can be only kna = {1, 2, 3}
		{
		k=0;

		for(; k<ka; k++)
			{
		
			sy_n_0 = _mm_load_sd( &y_n[0] );
			sx_t_0 = _mm_load_sd( &x_t[0] );
		
			sa_00 = _mm_load_sd( &A[0+bs*0] );
			sa_01 = _mm_load_sd( &A[0+bs*1] );
			sa_02 = _mm_load_sd( &A[0+bs*2] );
			sa_03 = _mm_load_sd( &A[0+bs*3] );
		
			stemp  = _mm_mul_sd( sa_00, sx_n_0 );
			sy_n_0 = _mm_add_sd( sy_n_0, stemp );
			stemp  = _mm_mul_sd( sa_00, sx_t_0 );
			sy_t_0 = _mm_add_sd( sy_t_0, stemp );
			stemp  = _mm_mul_sd( sa_01, sx_n_1 );
			sy_n_0 = _mm_add_sd( sy_n_0, stemp );
			stemp  = _mm_mul_sd( sa_01, sx_t_0 );
			sy_t_1 = _mm_add_sd( sy_t_1, stemp );
			stemp  = _mm_mul_sd( sa_02, sx_n_2 );
			sy_n_0 = _mm_add_sd( sy_n_0, stemp );
			stemp  = _mm_mul_sd( sa_02, sx_t_0 );
			sy_t_2 = _mm_add_sd( sy_t_2, stemp );
			stemp  = _mm_mul_sd( sa_03, sx_n_3 );
			sy_n_0 = _mm_add_sd( sy_n_0, stemp );
			stemp  = _mm_mul_sd( sa_03, sx_t_0 );
			sy_t_3 = _mm_add_sd( sy_t_3, stemp );
		
			_mm_store_sd( &sy_n[0], sy_n_0 );

		
			A += 1;
			y_n += 1;
			x_t += 1;

			}

		A += (sda-1)*bs;
		}

	k = bs*(ka/bs);
	sA = A + (ka/bs)*sda*bs;
	sy_n = y_n + (ka/bs)*bs;
	sx_t = x_t + (ka/bs)*bs;

	for(; k<ka; k++)
		{
		
		sy_n_0 = _mm_load_sd( &sy_n[0] );
		sx_t_0 = _mm_load_sd( &sx_t[0] );
		
		sa_00 = _mm_load_sd( &sA[0+bs*0] );
		sa_01 = _mm_load_sd( &sA[0+bs*1] );
		sa_02 = _mm_load_sd( &sA[0+bs*2] );
		sa_03 = _mm_load_sd( &sA[0+bs*3] );
		
		stemp  = _mm_mul_sd( sa_00, sx_n_0 );
		sy_n_0 = _mm_add_sd( sy_n_0, stemp );
		stemp  = _mm_mul_sd( sa_00, sx_t_0 );
		sy_t_0 = _mm_add_sd( sy_t_0, stemp );
		stemp  = _mm_mul_sd( sa_01, sx_n_1 );
		sy_n_0 = _mm_add_sd( sy_n_0, stemp );
		stemp  = _mm_mul_sd( sa_01, sx_t_0 );
		sy_t_1 = _mm_add_sd( sy_t_1, stemp );
		stemp  = _mm_mul_sd( sa_02, sx_n_2 );
		sy_n_0 = _mm_add_sd( sy_n_0, stemp );
		stemp  = _mm_mul_sd( sa_02, sx_t_0 );
		sy_t_2 = _mm_add_sd( sy_t_2, stemp );
		stemp  = _mm_mul_sd( sa_03, sx_n_3 );
		sy_n_0 = _mm_add_sd( sy_n_0, stemp );
		stemp  = _mm_mul_sd( sa_03, sx_t_0 );
		sy_t_3 = _mm_add_sd( sy_t_3, stemp );
		
		_mm_store_sd( &sy_n[0], sy_n_0 );

		
		sA += 1;
		sy_n += 1;
		sx_t += 1;

		}

	y_t_0 = _mm256_castpd128_pd256( sy_t_0 );
	y_t_1 = _mm256_castpd128_pd256( sy_t_1 );
	y_t_2 = _mm256_castpd128_pd256( sy_t_2 );
	y_t_3 = _mm256_castpd128_pd256( sy_t_3 );

	k=0;

	// corner
	if(tri==1)
		{
		
		__builtin_prefetch( A + sda*bs +bs*0 );
		__builtin_prefetch( A + sda*bs +bs*2 );

		y_n_0 = _mm256_loadu_pd( &y_n[0] );
		x_t_0 = _mm256_loadu_pd( &x_t[0] );
		
		a_00 = _mm256_load_pd( &A[0+bs*0] );
		a_01 = _mm256_load_pd( &A[0+bs*1] );
		a_02 = _mm256_load_pd( &A[0+bs*2] );
		a_03 = _mm256_load_pd( &A[0+bs*3] );
		
		temp  = _mm256_mul_pd( a_00, x_n_0 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_00, x_t_0 );
		temp  = _mm256_blend_pd( zeros, temp, 14 );
		y_t_0 = _mm256_add_pd( y_t_0, temp );
		temp  = _mm256_mul_pd( a_01, x_n_1 );
		temp  = _mm256_blend_pd( zeros, temp, 14 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_01, x_t_0 );
		temp  = _mm256_blend_pd( zeros, temp, 12 );
		y_t_1 = _mm256_add_pd( y_t_1, temp );
		temp  = _mm256_mul_pd( a_02, x_n_2 );
		temp  = _mm256_blend_pd( zeros, temp, 12 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_02, x_t_0 );
		temp  = _mm256_blend_pd( zeros, temp, 8 );
		y_t_2 = _mm256_add_pd( y_t_2, temp );
		temp  = _mm256_mul_pd( a_03, x_n_3 );
		temp  = _mm256_blend_pd( zeros, temp, 8 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		
		_mm256_storeu_pd( &y_n[0], y_n_0 );
		

		A += sda*bs;
		y_n += 4;
		x_t += 4;

		k += 4;

		}

	for(; k<ka-3; k+=bs)
		{
		
		__builtin_prefetch( A + sda*bs +bs*0 );
		__builtin_prefetch( A + sda*bs +bs*2 );

		y_n_0 = _mm256_loadu_pd( &y_n[0] );
		x_t_0 = _mm256_loadu_pd( &x_t[0] );
		
		a_00 = _mm256_load_pd( &A[0+bs*0] );
		a_01 = _mm256_load_pd( &A[0+bs*1] );
		a_02 = _mm256_load_pd( &A[0+bs*2] );
		a_03 = _mm256_load_pd( &A[0+bs*3] );
		
		temp  = _mm256_mul_pd( a_00, x_n_0 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_00, x_t_0 );
		y_t_0 = _mm256_add_pd( y_t_0, temp );
		temp  = _mm256_mul_pd( a_01, x_n_1 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_01, x_t_0 );
		y_t_1 = _mm256_add_pd( y_t_1, temp );
		temp  = _mm256_mul_pd( a_02, x_n_2 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_02, x_t_0 );
		y_t_2 = _mm256_add_pd( y_t_2, temp );
		temp  = _mm256_mul_pd( a_03, x_n_3 );
		y_n_0 = _mm256_add_pd( y_n_0, temp );
		temp  = _mm256_mul_pd( a_03, x_t_0 );
		y_t_3 = _mm256_add_pd( y_t_3, temp );
		
		_mm256_storeu_pd( &y_n[0], y_n_0 );

		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		}
	
	__m256d
		y_0_1_2_3;

	y_t_0 = _mm256_hadd_pd( y_t_0, y_t_1 );
	y_t_2 = _mm256_hadd_pd( y_t_2, y_t_3 );

	y_t_1 = _mm256_permute2f128_pd( y_t_2, y_t_0, 2  );	
	y_t_0 = _mm256_permute2f128_pd( y_t_2, y_t_0, 19 );	

	y_t_0 = _mm256_add_pd( y_t_0, y_t_1 );

	if(alg==1)
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y_t[0] );
		y_0_1_2_3 = _mm256_add_pd( y_0_1_2_3, y_t_0 );
		_mm256_storeu_pd( &y_t[0], y_0_1_2_3 );
		}
	else // alg==-1
		{
		y_0_1_2_3 = _mm256_loadu_pd( &y_t[0] );
		y_0_1_2_3 = _mm256_sub_pd( y_0_1_2_3, y_t_0 );
		_mm256_storeu_pd( &y_t[0], y_0_1_2_3 );
		}
	
	}



// it moves vertically across blocks
void kernel_dsymv_2_lib4(int kmax, int kna, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int bs = 4;
	
	int k;
	
	double
		a_00, a_01,
		x_n_0, x_n_1, y_n_0,
		x_t_0, y_t_0, y_t_1;
	
	if(alg==1)
		{
		x_n_0 = x_n[0];
		x_n_1 = x_n[1];
		}
	else // alg==-1
		{
		x_n_0 = - x_n[0];
		x_n_1 = - x_n[1];
		}

	y_t_0 = 0;
	y_t_1 = 0;
	
	k=0;

	// corner
	if(tri==1)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
/*		y_n_0 += a_00 * x_n_0;*/
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;


/*		y_n_0 = y_n[1];*/
		x_t_0 = x_t[1];

		a_01 = A[1+bs*1];

		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[1] = y_n_0;

		
		A += 2 + (sda-1)*bs;
		y_n += 2;
		x_t += 2;

		k += 2;

		}

	for(; k<kmax-3; k+=bs)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;


		y_n_0 = y_n[1];
		x_t_0 = x_t[1];
		
		a_00 = A[1+bs*0];
		a_01 = A[1+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[1] = y_n_0;

		
		y_n_0 = y_n[2];
		x_t_0 = x_t[2];
		
		a_00 = A[2+bs*0];
		a_01 = A[2+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[2] = y_n_0;


		y_n_0 = y_n[3];
		x_t_0 = x_t[3];
		
		a_00 = A[3+bs*0];
		a_01 = A[3+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[3] = y_n_0;

		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		}
	
	for(; k<kmax; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		a_01 = A[0+bs*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}

	if(alg==1)
		{
		y_t[0] += y_t_0;
		y_t[1] += y_t_1;
		}
	else // alg==-1
		{
		y_t[0] -= y_t_0;
		y_t[1] -= y_t_1;
		}
	
	}



// it moves vertically across blocks
void kernel_dsymv_1_lib4(int kmax, int kna, double *A, int sda, double *x_n, double *y_n, double *x_t, double *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int bs = 4;
	
	int k;
	
	double
		a_00,
		x_n_0, y_n_0,
		x_t_0, y_t_0;
	
	if(alg==1)
		{
		x_n_0 = x_n[0];
		}
	else // alg==-1
		{
		x_n_0 = - x_n[0];
		}

	y_t_0 = 0;
	
	k=0;

	// corner
	if(tri==1)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		
/*		y_n_0 += a_00 * x_n_0;*/
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;


		A += 1 + (sda-1)*bs;
		y_n += 1;
		x_t += 1;

		k += 1;

		}

	for(; k<kmax-3; k+=bs)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;


		y_n_0 = y_n[1];
		x_t_0 = x_t[1];
		
		a_00 = A[1+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[1] = y_n_0;

		
		y_n_0 = y_n[2];
		x_t_0 = x_t[2];
		
		a_00 = A[2+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[2] = y_n_0;


		y_n_0 = y_n[3];
		x_t_0 = x_t[3];
		
		a_00 = A[3+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[3] = y_n_0;

		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		}
	
	for(; k<kmax; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}

	if(alg==1)
		{
		y_t[0] += y_t_0;
		}
	else // alg==-1
		{
		y_t[0] -= y_t_0;
		}
	
	}

