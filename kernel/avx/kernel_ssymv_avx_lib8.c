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



void kernel_ssymv_4_lib8(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 8;
	
	int k;
	
	__m128
		zeros, temp,
		a_00, a_01, a_02, a_03,
		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,
		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;
	
	zeros = _mm_setzero_ps();

	x_n_0 = _mm_broadcast_ss( &x_n[0] );
	x_n_1 = _mm_broadcast_ss( &x_n[1] );
	x_n_2 = _mm_broadcast_ss( &x_n[2] );
	x_n_3 = _mm_broadcast_ss( &x_n[3] );

	if(alg==-1)
		{
		x_n_0 = _mm_sub_ps( zeros, x_n_0 );
		x_n_1 = _mm_sub_ps( zeros, x_n_1 );
		x_n_2 = _mm_sub_ps( zeros, x_n_2 );
		x_n_3 = _mm_sub_ps( zeros, x_n_3 );
		}

	y_t_0 = _mm_setzero_ps();
	y_t_1 = _mm_setzero_ps();
	y_t_2 = _mm_setzero_ps();
	y_t_3 = _mm_setzero_ps();
	
	k=0;

	// corner
	if(tri==1)
		{
		
		y_n_0 = _mm_load_ss( &y_n[0] );
		x_t_0 = _mm_load_ss( &x_t[0] );
		
		a_00  = _mm_load_ss( &A[0+lda*0] );
		a_01  = _mm_load_ss( &A[0+lda*1] );
		a_02  = _mm_load_ss( &A[0+lda*2] );
		a_03  = _mm_load_ss( &A[0+lda*3] );
		
/*		temp  = _mm_mul_ss( a_00, x_n_0 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
		temp  = _mm_mul_ss( a_00, x_t_0 );
		y_t_0 = _mm_add_ss( y_t_0, temp );
		temp  = _mm_mul_ss( a_01, x_n_1 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_01, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, temp );
		temp  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_02, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, temp );
		temp  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[0], y_n_0 );


		y_n_0 = _mm_load_ss( &y_n[1] );
		x_t_0 = _mm_load_ss( &x_t[1] );
		
/*		a_00  = _mm_load_ss( &A[1+lda*0] );*/
		a_01  = _mm_load_ss( &A[1+lda*1] );
		a_02  = _mm_load_ss( &A[1+lda*2] );
		a_03  = _mm_load_ss( &A[1+lda*3] );
		
/*		temp  = _mm_mul_ss( a_00, x_n_0 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_00, x_t_0 );*/
/*		y_t_0 = _mm_add_ss( y_t_0, temp );*/
/*		temp  = _mm_mul_ss( a_01, x_n_1 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
		temp  = _mm_mul_ss( a_01, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, temp );
		temp  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_02, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, temp );
		temp  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[1], y_n_0 );


		y_n_0 = _mm_load_ss( &y_n[2] );
		x_t_0 = _mm_load_ss( &x_t[2] );
		
/*		a_00  = _mm_load_ss( &A[2+lda*0] );*/
/*		a_01  = _mm_load_ss( &A[2+lda*1] );*/
		a_02  = _mm_load_ss( &A[2+lda*2] );
		a_03  = _mm_load_ss( &A[2+lda*3] );
		
/*		temp  = _mm_mul_ss( a_00, x_n_0 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_00, x_t_0 );*/
/*		y_t_0 = _mm_add_ss( y_t_0, temp );*/
/*		temp  = _mm_mul_ss( a_01, x_n_1 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_01, x_t_0 );*/
/*		y_t_1 = _mm_add_ss( y_t_1, temp );*/
/*		temp  = _mm_mul_ss( a_02, x_n_2 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
		temp  = _mm_mul_ss( a_02, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, temp );
		temp  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[2], y_n_0 );

		
		y_n_0 = _mm_load_ss( &y_n[3] );
		x_t_0 = _mm_load_ss( &x_t[3] );
		
/*		a_00  = _mm_load_ss( &A[3+lda*0] );*/
/*		a_01  = _mm_load_ss( &A[3+lda*1] );*/
/*		a_02  = _mm_load_ss( &A[3+lda*2] );*/
		a_03  = _mm_load_ss( &A[3+lda*3] );
		
/*		temp  = _mm_mul_ss( a_00, x_n_0 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_00, x_t_0 );*/
/*		y_t_0 = _mm_add_ss( y_t_0, temp );*/
/*		temp  = _mm_mul_ss( a_01, x_n_1 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_01, x_t_0 );*/
/*		y_t_1 = _mm_add_ss( y_t_1, temp );*/
/*		temp  = _mm_mul_ss( a_02, x_n_2 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
/*		temp  = _mm_mul_ss( a_02, x_t_0 );*/
/*		y_t_2 = _mm_add_ss( y_t_2, temp );*/
/*		temp  = _mm_mul_ss( a_03, x_n_3 );*/
/*		y_n_0 = _mm_add_ss( y_n_0, temp );*/
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[3], y_n_0 );
		

		A   += 4;
		y_n += 4;
		x_t += 4;

		k += 4;

		}
	for(; k<kna; k++)
		{
		
		y_n_0 = _mm_load_ss( &y_n[0] );
		x_t_0 = _mm_load_ss( &x_t[0] );
		
		a_00  = _mm_load_ss( &A[0+lda*0] );
		a_01  = _mm_load_ss( &A[0+lda*1] );
		a_02  = _mm_load_ss( &A[0+lda*2] );
		a_03  = _mm_load_ss( &A[0+lda*3] );
		
		temp  = _mm_mul_ss( a_00, x_n_0 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_00, x_t_0 );
		y_t_0 = _mm_add_ss( y_t_0, temp );
		temp  = _mm_mul_ss( a_01, x_n_1 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_01, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, temp );
		temp  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_02, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, temp );
		temp  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[0], y_n_0 );

	
		A   += 1;
		y_n += 1;
		x_t += 1;
		
		}
	if(kna>0 || tri==1)
		{
		A += (sda-1)*lda;
		}
	for(; k<kmax-7; k+=8)
		{
		
		y_n_0 = _mm_loadu_ps( &y_n[0] );
		x_t_0 = _mm_loadu_ps( &x_t[0] );
		
		a_00  = _mm_load_ps( &A[0+lda*0] );
		a_01  = _mm_load_ps( &A[0+lda*1] );
		a_02  = _mm_load_ps( &A[0+lda*2] );
		a_03  = _mm_load_ps( &A[0+lda*3] );
		
		temp  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_00, x_t_0 );
		y_t_0 = _mm_add_ps( y_t_0, temp );
		temp  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_01, x_t_0 );
		y_t_1 = _mm_add_ps( y_t_1, temp );
		temp  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_02, x_t_0 );
		y_t_2 = _mm_add_ps( y_t_2, temp );
		temp  = _mm_mul_ps( a_03, x_n_3 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_03, x_t_0 );
		y_t_3 = _mm_add_ps( y_t_3, temp );
		
		_mm_storeu_ps( &y_n[0], y_n_0 );
		

		y_n_0 = _mm_loadu_ps( &y_n[4] );
		x_t_0 = _mm_loadu_ps( &x_t[4] );
		
		a_00  = _mm_load_ps( &A[4+lda*0] );
		a_01  = _mm_load_ps( &A[4+lda*1] );
		a_02  = _mm_load_ps( &A[4+lda*2] );
		a_03  = _mm_load_ps( &A[4+lda*3] );
		
		temp  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_00, x_t_0 );
		y_t_0 = _mm_add_ps( y_t_0, temp );
		temp  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_01, x_t_0 );
		y_t_1 = _mm_add_ps( y_t_1, temp );
		temp  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_02, x_t_0 );
		y_t_2 = _mm_add_ps( y_t_2, temp );
		temp  = _mm_mul_ps( a_03, x_n_3 );
		y_n_0 = _mm_add_ps( y_n_0, temp );
		temp  = _mm_mul_ps( a_03, x_t_0 );
		y_t_3 = _mm_add_ps( y_t_3, temp );
		
		_mm_storeu_ps( &y_n[4], y_n_0 );
		

		A   += sda*lda;
		y_n += 8;
		x_t += 8;

		}
	
	for(; k<kmax; k++)
		{
		
		y_n_0 = _mm_load_ss( &y_n[0] );
		x_t_0 = _mm_load_ss( &x_t[0] );
		
		a_00  = _mm_load_ss( &A[0+lda*0] );
		a_01  = _mm_load_ss( &A[0+lda*1] );
		a_02  = _mm_load_ss( &A[0+lda*2] );
		a_03  = _mm_load_ss( &A[0+lda*3] );
		
		temp  = _mm_mul_ss( a_00, x_n_0 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_00, x_t_0 );
		y_t_0 = _mm_add_ss( y_t_0, temp );
		temp  = _mm_mul_ss( a_01, x_n_1 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_01, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, temp );
		temp  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_02, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, temp );
		temp  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, temp );
		temp  = _mm_mul_ss( a_03, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, temp );
		
		_mm_store_ss( &y_n[0], y_n_0 );

	
		A   += 1;
		y_n += 1;
		x_t += 1;
		
		}

	// reduction
	y_t_0 = _mm_hadd_ps(y_t_0, y_t_1);
	y_t_2 = _mm_hadd_ps(y_t_2, y_t_3);

	y_t_0 = _mm_hadd_ps(y_t_0, y_t_2);

	if(alg==1)
		{
		y_t_1 = _mm_loadu_ps( &y_t[0] );

		y_t_1 = _mm_add_ps(y_t_1, y_t_0);

		_mm_storeu_ps(&y_t[0], y_t_1);
		}
	else // alg==-1
		{
		y_t_1 = _mm_loadu_ps( &y_t[0] );

		y_t_1 = _mm_sub_ps(y_t_1, y_t_0);

		_mm_storeu_ps(&y_t[0], y_t_1);
		}
	
	}
	
	
	
void kernel_ssymv_2_lib8(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 8;
	
	int k;
	
	float
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
		
		a_00 = A[0+lda*0];
		a_01 = A[0+lda*1];
		
/*		y_n_0 += a_00 * x_n_0;*/
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;


/*		y_n_0 = y_n[1];*/
		x_t_0 = x_t[1];

		a_01 = A[1+lda*1];

/*		y_n_0 += a_01 * x_n_1;*/
		y_t_1 += a_01 * x_t_0;
		
/*		y_n[1] = y_n_0;*/

		
		A += 2;
		y_n += 2;
		x_t += 2;

		k += 2;

		}
	for(; k<kna; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		a_01 = A[0+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}
	if(kna>0 || tri==1)
		{
		A += (sda-1)*lda;
		}
	for(; k<kmax-lda+1; k+=lda)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		a_01 = A[0+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[0] = y_n_0;


		y_n_0 = y_n[1];
		x_t_0 = x_t[1];
		
		a_00 = A[1+lda*0];
		a_01 = A[1+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[1] = y_n_0;

		
		y_n_0 = y_n[2];
		x_t_0 = x_t[2];
		
		a_00 = A[2+lda*0];
		a_01 = A[2+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[2] = y_n_0;


		y_n_0 = y_n[3];
		x_t_0 = x_t[3];
		
		a_00 = A[3+lda*0];
		a_01 = A[3+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[3] = y_n_0;

		
		y_n_0 = y_n[4];
		x_t_0 = x_t[4];
		
		a_00 = A[4+lda*0];
		a_01 = A[4+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[4] = y_n_0;


		y_n_0 = y_n[5];
		x_t_0 = x_t[5];
		
		a_00 = A[5+lda*0];
		a_01 = A[5+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[5] = y_n_0;

		
		y_n_0 = y_n[6];
		x_t_0 = x_t[6];
		
		a_00 = A[6+lda*0];
		a_01 = A[6+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[6] = y_n_0;


		y_n_0 = y_n[7];
		x_t_0 = x_t[7];
		
		a_00 = A[7+lda*0];
		a_01 = A[7+lda*1];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		y_n_0 += a_01 * x_n_1;
		y_t_1 += a_01 * x_t_0;
		
		y_n[7] = y_n_0;

		
		A += sda*lda;
		y_n += 8;
		x_t += 8;

		}
	
	for(; k<kmax; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		a_01 = A[0+lda*1];
		
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
	
	
	
void kernel_ssymv_1_lib8(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 8;
	
	int k;
	
	float
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
		
/*		y_n_0 = y_n[0];*/
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		
/*		y_n_0 += a_00 * x_n_0;*/
		y_t_0 += a_00 * x_t_0;
		
/*		y_n[0] = y_n_0;*/

		A += 1;
		y_n += 1;
		x_t += 1;

		k += 1;

		}
	for(; k<kna; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}
	if(kna>0 || tri==1)
		{
		A += (sda-1)*lda;
		}
	for(; k<kmax-lda+1; k+=lda)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;


		y_n_0 = y_n[1];
		x_t_0 = x_t[1];
		
		a_00 = A[1+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[1] = y_n_0;

		
		y_n_0 = y_n[2];
		x_t_0 = x_t[2];
		
		a_00 = A[2+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[2] = y_n_0;


		y_n_0 = y_n[3];
		x_t_0 = x_t[3];
		
		a_00 = A[3+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[3] = y_n_0;

		
		y_n_0 = y_n[4];
		x_t_0 = x_t[4];
		
		a_00 = A[4+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[4] = y_n_0;


		y_n_0 = y_n[5];
		x_t_0 = x_t[5];
		
		a_00 = A[5+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[5] = y_n_0;

		
		y_n_0 = y_n[6];
		x_t_0 = x_t[6];
		
		a_00 = A[6+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[6] = y_n_0;


		y_n_0 = y_n[7];
		x_t_0 = x_t[7];
		
		a_00 = A[7+lda*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[7] = y_n_0;

		
		A += sda*lda;
		y_n += 8;
		x_t += 8;

		}
	
	for(; k<kmax; k++)
		{
		
		y_n_0 = y_n[0];
		x_t_0 = x_t[0];
		
		a_00 = A[0+lda*0];
		
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
	
	
	

