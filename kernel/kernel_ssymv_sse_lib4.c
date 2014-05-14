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
#include <pmmintrin.h>  // SSE3
/*#include <smmintrin.h>  // SSE4*/
/*#include <immintrin.h>  // AVX*/



// it moves vertically across blocks
void kernel_ssymv_4_lib4(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int bs = 4;
	
	int k;
	
/*	float*/
/*		a_00, a_01, a_02, a_03,*/
/*		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,*/
/*		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;*/
	
	__m128
		zeros,
		a_00, a_01, a_02, a_03, a_tmp,
		x_n_0, x_n_1, x_n_2, x_n_3, y_n_0,
		x_t_0, y_t_0, y_t_1, y_t_2, y_t_3;

	zeros = _mm_setzero_ps();

	x_n_0 = _mm_load1_ps( &x_n[0] );
	x_n_1 = _mm_load1_ps( &x_n[1] );
	x_n_2 = _mm_load1_ps( &x_n[2] );
	x_n_3 = _mm_load1_ps( &x_n[3] );
	
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
		
		y_n_0 = _mm_loadu_ps( &y_n[0] );
		x_t_0 = _mm_loadu_ps( &x_t[0] );
		
		a_00 = _mm_load_ps( &A[0+bs*0] );
		a_01 = _mm_load_ps( &A[0+bs*1] );
/*		a_01 = _mm_blend_ps( zeros, a_01, 14 );*/
		a_01 = _mm_move_ss( a_01, zeros );
		a_02 = _mm_load_ps( &A[0+bs*2] );
/*		a_02 = _mm_blend_ps( zeros, a_02, 12 );*/
		a_02 = _mm_move_ss( a_02, zeros );
		a_02 = _mm_shuffle_ps( a_02, a_02, 0xe0 );
		a_03 = _mm_load_ps( &A[0+bs*3] );
/*		a_03 = _mm_blend_ps( zeros, a_03, 8 );*/
		a_03 = _mm_move_ss( a_03, zeros );
		a_03 = _mm_shuffle_ps( a_03, a_03, 0xc0 );
		
		a_tmp = a_00;
/*		a_tmp = _mm_blend_ps( zeros, a_tmp, 14 );*/
		a_tmp = _mm_move_ss( a_tmp, zeros );
		a_00  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, a_00 );
		y_t_0 = _mm_mul_ps( a_tmp, x_t_0 );
/*		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );*/
/*		y_t_0 = _mm_add_ps( y_t_0, a_tmp );*/

		a_tmp = a_01;
/*		a_tmp = _mm_blend_ps( zeros, a_tmp, 12 );*/
		a_tmp = _mm_shuffle_ps( a_tmp, a_tmp, 0xe0 );
		a_01  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, a_01 );
		y_t_1 = _mm_mul_ps( a_tmp, x_t_0 );
/*		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );*/
/*		y_t_1 = _mm_add_ps( y_t_1, a_tmp );*/

		a_tmp = a_02;
/*		a_tmp = _mm_blend_ps( zeros, a_tmp, 8 );*/
		a_tmp = _mm_shuffle_ps( a_tmp, a_tmp, 0xc0 );
		a_02  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, a_02 );
		y_t_2 = _mm_mul_ps( a_tmp, x_t_0 );
/*		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );*/
/*		y_t_2 = _mm_add_ps( y_t_2, a_tmp );*/

		_mm_storeu_ps( &y_n[0], y_n_0 );

/*		a_03  = _mm_mul_ps( a_03, x_n_3 );*/
/*		y_n_0 = _mm_add_ps( y_n_0, a_03 ); // TODO update y_t_3 instead*/
		y_t_3 = _mm_mul_ps( a_03, x_t_0 );
/*		a_03  = _mm_mul_ps( a_03, x_t_0 );*/
/*		y_t_3 = _mm_add_ps( y_t_3, a_03 ); // TODO avoid useless add with zero the first time*/
		
		

		A += 4;
		y_n += 4;
		x_t += 4;

		k += 4;

		}
	for(; k<kna; k++)
		{
		
		y_n_0 = _mm_load_ss( &y_n[0] );
		x_t_0 = _mm_load_ss( &x_t[0] );
			
		a_00 = _mm_load_ss( &A[0+bs*0] );
		a_01 = _mm_load_ss( &A[0+bs*1] );
		a_02 = _mm_load_ss( &A[0+bs*2] );
		a_03 = _mm_load_ss( &A[0+bs*3] );
		
		a_tmp = a_00;
		a_00  = _mm_mul_ss( a_00, x_n_0 );
		y_n_0 = _mm_add_ss( y_n_0, a_00 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_0 = _mm_add_ss( y_t_0, a_tmp );
		a_tmp = a_01;
		a_01  = _mm_mul_ss( a_01, x_n_1 );
		y_n_0 = _mm_add_ss( y_n_0, a_01 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, a_tmp );
		a_tmp = a_02;
		a_02  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, a_02 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, a_tmp );
		a_tmp = a_03;
		a_03  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, a_03 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, a_tmp );
		
		_mm_store_ss( &y_n[0], y_n_0 );

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}
	if(kna>0 || tri==1)
		{
		A += (sda-1)*bs;
		}
	for(; k<kmax-7; k+=8)
		{
		
		y_n_0 = _mm_loadu_ps( &y_n[0] );
		x_t_0 = _mm_loadu_ps( &x_t[0] );
			
		a_00 = _mm_load_ps( &A[0+bs*0] );
		a_01 = _mm_load_ps( &A[0+bs*1] );
		a_02 = _mm_load_ps( &A[0+bs*2] );
		a_03 = _mm_load_ps( &A[0+bs*3] );
		
		a_tmp = a_00;
		a_00  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, a_00 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_0 = _mm_add_ps( y_t_0, a_tmp );
		a_tmp = a_01;
		a_01  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, a_01 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_1 = _mm_add_ps( y_t_1, a_tmp );
		a_tmp = a_02;
		a_02  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, a_02 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_2 = _mm_add_ps( y_t_2, a_tmp );
		a_tmp = a_03;
		a_03  = _mm_mul_ps( a_03, x_n_3 );
		y_n_0 = _mm_add_ps( y_n_0, a_03 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_3 = _mm_add_ps( y_t_3, a_tmp );
		
		_mm_storeu_ps( &y_n[0], y_n_0 );
		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		y_n_0 = _mm_loadu_ps( &y_n[0] );
		x_t_0 = _mm_loadu_ps( &x_t[0] );
			
		a_00 = _mm_load_ps( &A[0+bs*0] );
		a_01 = _mm_load_ps( &A[0+bs*1] );
		a_02 = _mm_load_ps( &A[0+bs*2] );
		a_03 = _mm_load_ps( &A[0+bs*3] );
		
		a_tmp = a_00;
		a_00  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, a_00 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_0 = _mm_add_ps( y_t_0, a_tmp );
		a_tmp = a_01;
		a_01  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, a_01 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_1 = _mm_add_ps( y_t_1, a_tmp );
		a_tmp = a_02;
		a_02  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, a_02 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_2 = _mm_add_ps( y_t_2, a_tmp );
		a_tmp = a_03;
		a_03  = _mm_mul_ps( a_03, x_n_3 );
		y_n_0 = _mm_add_ps( y_n_0, a_03 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_3 = _mm_add_ps( y_t_3, a_tmp );
		
		_mm_storeu_ps( &y_n[0], y_n_0 );
		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		}
	for(; k<kmax-3; k+=4)
		{
		
		y_n_0 = _mm_loadu_ps( &y_n[0] );
		x_t_0 = _mm_loadu_ps( &x_t[0] );
			
		a_00 = _mm_load_ps( &A[0+bs*0] );
		a_01 = _mm_load_ps( &A[0+bs*1] );
		a_02 = _mm_load_ps( &A[0+bs*2] );
		a_03 = _mm_load_ps( &A[0+bs*3] );
		
		a_tmp = a_00;
		a_00  = _mm_mul_ps( a_00, x_n_0 );
		y_n_0 = _mm_add_ps( y_n_0, a_00 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_0 = _mm_add_ps( y_t_0, a_tmp );
		a_tmp = a_01;
		a_01  = _mm_mul_ps( a_01, x_n_1 );
		y_n_0 = _mm_add_ps( y_n_0, a_01 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_1 = _mm_add_ps( y_t_1, a_tmp );
		a_tmp = a_02;
		a_02  = _mm_mul_ps( a_02, x_n_2 );
		y_n_0 = _mm_add_ps( y_n_0, a_02 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_2 = _mm_add_ps( y_t_2, a_tmp );
		a_tmp = a_03;
		a_03  = _mm_mul_ps( a_03, x_n_3 );
		y_n_0 = _mm_add_ps( y_n_0, a_03 );
		a_tmp = _mm_mul_ps( a_tmp, x_t_0 );
		y_t_3 = _mm_add_ps( y_t_3, a_tmp );
		
		_mm_storeu_ps( &y_n[0], y_n_0 );
		
		A += sda*bs;
		y_n += 4;
		x_t += 4;

		}
	for(; k<kmax; k++)
		{
		
		y_n_0 = _mm_load_ss( &y_n[0] );
		x_t_0 = _mm_load_ss( &x_t[0] );
			
		a_00 = _mm_load_ss( &A[0+bs*0] );
		a_01 = _mm_load_ss( &A[0+bs*1] );
		a_02 = _mm_load_ss( &A[0+bs*2] );
		a_03 = _mm_load_ss( &A[0+bs*3] );
		
		a_tmp = a_00;
		a_00  = _mm_mul_ss( a_00, x_n_0 );
		y_n_0 = _mm_add_ss( y_n_0, a_00 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_0 = _mm_add_ss( y_t_0, a_tmp );
		a_tmp = a_01;
		a_01  = _mm_mul_ss( a_01, x_n_1 );
		y_n_0 = _mm_add_ss( y_n_0, a_01 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_1 = _mm_add_ss( y_t_1, a_tmp );
		a_tmp = a_02;
		a_02  = _mm_mul_ss( a_02, x_n_2 );
		y_n_0 = _mm_add_ss( y_n_0, a_02 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_2 = _mm_add_ss( y_t_2, a_tmp );
		a_tmp = a_03;
		a_03  = _mm_mul_ss( a_03, x_n_3 );
		y_n_0 = _mm_add_ss( y_n_0, a_03 );
		a_tmp = _mm_mul_ss( a_tmp, x_t_0 );
		y_t_3 = _mm_add_ss( y_t_3, a_tmp );
		
		_mm_store_ss( &y_n[0], y_n_0 );
	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}

	__m128
		y_03;

	y_t_0 = _mm_hadd_ps(y_t_0, y_t_1);
	y_t_2 = _mm_hadd_ps(y_t_2, y_t_3);
	y_t_0 = _mm_hadd_ps(y_t_0, y_t_2);

	if(alg==1)
		{
		y_03 = _mm_loadu_ps( &y_t[0] );

		y_03 = _mm_add_ps(y_03, y_t_0);
	
		_mm_storeu_ps(&y_t[0], y_03);
		}
	else // alg==-1
		{
		y_03 = _mm_loadu_ps( &y_t[0] );

		y_03 = _mm_sub_ps(y_03, y_t_0);
	
		_mm_storeu_ps(&y_t[0], y_03);
		}
	
	}



// it moves vertically across blocks
void kernel_ssymv_2_lib4(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int bs = 4;
	
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
	if(kna>0 || tri==1)
		{
		A += (sda-1)*bs;
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
void kernel_ssymv_1_lib4(int kmax, int kna, float *A, int sda, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int bs = 4;
	
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
		
		a_00 = A[0+bs*0];
		
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
		
		a_00 = A[0+bs*0];
		
		y_n_0 += a_00 * x_n_0;
		y_t_0 += a_00 * x_t_0;
		
		y_n[0] = y_n_0;

	
		A += 1;
		y_n += 1;
		x_t += 1;
		
		}
	if(kna>0 || tri==1)
		{
		A += (sda-1)*bs;
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



/*// it moves horizontally inside a block*/
/*void kernel_ssymv_4_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)*/
/*	{*/
/*	*/
/*	*/
/*	const int bs  = 4;*/
/*	*/
/*	int	k;*/
/*	*/
/*	float*/
/*		a_00, a_10, a_20, a_30,*/
/*		x_n_0, y_n_0, y_n_1, y_n_2, y_n_3,*/
/*		x_t_0, x_t_1, x_t_2, x_t_3, y_t_0;*/
/*	*/
/*	y_n_0 = 0;*/
/*	y_n_1 = 0;*/
/*	y_n_2 = 0;*/
/*	y_n_3 = 0;*/
/*	*/
/*	x_t_0 = x_t[0];*/
/*	x_t_1 = x_t[1];*/
/*	x_t_2 = x_t[2];*/
/*	x_t_3 = x_t[3];*/

/*	if(alg==1)*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 += a_20 * x_n_0;*/
/*			y_t_0 += a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*1];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*1];*/
/*			y_n_2 += a_20 * x_n_0;*/
/*			y_t_0 += a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*1];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/

/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 += a_20 * x_n_0;*/
/*			y_t_0 += a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/

/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/

/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 += a_20 * x_n_0;*/
/*			y_t_0 += a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_10 = A[1+bs*1];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*		*/
/*			a_20 = A[2+bs*1];*/
/*			y_n_2 += a_20 * x_n_0;*/
/*			y_t_0 += a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*1];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/

/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			// unroll 3*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 += a_20 * x_n_0;*/

/*			a_30 = A[3+bs*0];*/
/*			y_n_3 += a_30 * x_n_0;*/
/*			y_t_0 += a_30 * x_t_3;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/

/*			// unroll 4*/
/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 += a_30 * x_n_0;*/

/*			}*/

/*		}*/
/*	else // alg==-1*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 -= a_20 * x_n_0;*/
/*			y_t_0 -= a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/

/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*1];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*1];*/
/*			y_n_2 -= a_20 * x_n_0;*/
/*			y_t_0 -= a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*1];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/

/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 -= a_20 * x_n_0;*/
/*			y_t_0 -= a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/

/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/

/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 -= a_20 * x_n_0;*/
/*			y_t_0 -= a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_10 = A[1+bs*1];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*		*/
/*			a_20 = A[2+bs*1];*/
/*			y_n_2 -= a_20 * x_n_0;*/
/*			y_t_0 -= a_20 * x_t_2;*/
/*		*/
/*			a_30 = A[3+bs*1];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/

/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			// unroll 3*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_20 = A[2+bs*0];*/
/*			y_n_2 -= a_20 * x_n_0;*/

/*			a_30 = A[3+bs*0];*/
/*			y_n_3 -= a_30 * x_n_0;*/
/*			y_t_0 -= a_30 * x_t_3;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			// unroll 4*/
/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_30 = A[3+bs*0];*/
/*			y_n_3 -= a_30 * x_n_0;*/

/*			}*/

/*		}		*/

/*	y_n[0] += y_n_0;*/
/*	y_n[1] += y_n_1;*/
/*	y_n[2] += y_n_2;*/
/*	y_n[3] += y_n_3;*/

/*	}*/



/*// it moves horizontally inside a block*/
/*void kernel_ssymv_2_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)*/
/*	{*/
/*	*/
/*	*/
/*	const int bs  = 4;*/
/*	*/
/*	int	k;*/
/*	*/
/*	float*/
/*		a_00, a_10,*/
/*		x_t_0, x_t_1, y_t_0,*/
/*		x_n_0, y_n_0, y_n_1;*/
/*	*/
/*	y_n_0 = 0;*/
/*	y_n_1 = 0;*/
/*	*/
/*	x_t_0 = x_t[0];*/
/*	x_t_1 = x_t[1];*/

/*	if(alg==1)*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*1];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/
/*			y_t_0 += a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/

/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_10 = A[1+bs*0];*/
/*			y_n_1 += a_10 * x_n_0;*/

/*			}*/

/*		}*/
/*	else // alg==-1*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*1];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/

/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/
/*			y_t_0 -= a_10 * x_t_1;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/

/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_10 = A[1+bs*0];*/
/*			y_n_1 -= a_10 * x_n_0;*/

/*			}*/

/*		}		*/

/*	y_n[0] += y_n_0;*/
/*	y_n[1] += y_n_1;*/

/*	}*/



/*// it moves horizontally inside a block*/
/*void kernel_ssymv_1_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)*/
/*	{*/
/*	*/
/*	*/
/*	const int bs  = 4;*/
/*	*/
/*	int	k;*/
/*	*/
/*	float*/
/*		a_00,*/
/*		x_t_0, y_t_0,*/
/*		x_n_0, y_n_0;*/
/*	*/
/*	y_n_0 = 0;*/
/*	*/
/*	x_t_0 = x_t[0];*/

/*	if(alg==1)*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/
/*		*/
/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/
/*			y_t_0 += a_00 * x_t_0;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/
/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 += a_00 * x_n_0;*/

/*			}*/
/*		*/
/*		}*/
/*	else // alg==-1*/
/*		{*/
/*		k=0;*/
/*		for(; k<kmax-1; k+=2)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			// unroll 2*/
/*			x_n_0 = x_n[1];*/
/*			y_t_0 = y_t[1];*/
/*		*/
/*			a_00 = A[0+bs*1];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/
/*		*/
/*			y_t[1] = y_t_0;*/
/*		*/
/*			A   += 2*bs;*/
/*			x_n += 2;*/
/*			y_t += 2;*/

/*			}*/
/*		for(; k<kmax; k++)*/
/*			{*/
/*		*/
/*			// unroll 1*/
/*			x_n_0 = x_n[0];*/
/*			y_t_0 = y_t[0];*/
/*		*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/
/*			y_t_0 -= a_00 * x_t_0;*/
/*		*/
/*			y_t[0] = y_t_0;*/
/*		*/
/*			A   += 1*bs;*/
/*			x_n += 1;*/
/*			y_t += 1;*/

/*			}*/
/*		if(tri==1)*/
/*			{*/

/*			// corner*/
/*			x_n_0 = x_n[0];*/
/*	*/
/*			a_00 = A[0+bs*0];*/
/*			y_n_0 -= a_00 * x_n_0;*/

/*			}*/
/*		*/
/*		}*/

/*	y_n[0] += y_n_0;*/

/*	}*/
