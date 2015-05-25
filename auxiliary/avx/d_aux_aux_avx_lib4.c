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



// both A and B are aligned to 256-bit boundaries
void kernel_align_panel_8_0_lib4(int kmax, double *A0, int sda,  double *B0, int sdb)
	{

	if(kmax<=0)
		return;
	
	const int bs = 4;

	double *A1 = A0 + bs*sda;
	double *B1 = B0 + bs*sdb;

	__m256d
		a_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		_mm256_store_pd( &B0[0+bs*0], a_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		_mm256_store_pd( &B0[0+bs*1], a_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		_mm256_store_pd( &B0[0+bs*2], a_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		_mm256_store_pd( &B0[0+bs*3], a_0 );

		A0 += 16;
		B0 += 16;

		a_0 = _mm256_load_pd( &A1[0+bs*0] );
		_mm256_store_pd( &B1[0+bs*0], a_0 );

		a_0 = _mm256_load_pd( &A1[0+bs*1] );
		_mm256_store_pd( &B1[0+bs*1], a_0 );

		a_0 = _mm256_load_pd( &A1[0+bs*2] );
		_mm256_store_pd( &B1[0+bs*2], a_0 );

		a_0 = _mm256_load_pd( &A1[0+bs*3] );
		_mm256_store_pd( &B1[0+bs*3], a_0 );

		A1 += 16;
		B1 += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		_mm256_store_pd( &B0[0+bs*0], a_0 );

		A0 += 4;
		B0 += 4;

		a_0 = _mm256_load_pd( &A1[0+bs*0] );
		_mm256_store_pd( &B1[0+bs*0], a_0 );

		A1 += 4;
		B1 += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 1 element of A must be skipped
void kernel_align_panel_8_1_lib4(int kmax, double *A0, int sda, double *B0, int sdb)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;
	double *A2 = A1 + bs*sda;
	double *B1 = B0 + bs*sdb;

	__m256d
		a_0, a_1, a_2,
		b_0, b_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_2 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_shuffle_pd( a_1, a_2, 0x5 );
		b_0 = _mm256_shuffle_pd( a_0, b_0, 0x5 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );

		a_2 = _mm256_load_pd( &A2[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_2 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_shuffle_pd( a_1, a_2, 0x5 );
		b_0 = _mm256_shuffle_pd( a_0, b_0, 0x5 );
		_mm256_store_pd( &B1[0+bs*1], b_1 );
		_mm256_store_pd( &B0[0+bs*1], b_0 );

		a_2 = _mm256_load_pd( &A2[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_2 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_shuffle_pd( a_1, a_2, 0x5 );
		b_0 = _mm256_shuffle_pd( a_0, b_0, 0x5 );
		_mm256_store_pd( &B1[0+bs*2], b_1 );
		_mm256_store_pd( &B0[0+bs*2], b_0 );

		a_2 = _mm256_load_pd( &A2[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_2 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_shuffle_pd( a_1, a_2, 0x5 );
		b_0 = _mm256_shuffle_pd( a_0, b_0, 0x5 );
		_mm256_store_pd( &B1[0+bs*3], b_1 );
		_mm256_store_pd( &B0[0+bs*3], b_0 );

		A0 += 16;
		A1 += 16;
		A2 += 16;
		B0 += 16;
		B1 += 16;

		}
	for(; k<kmax; k++)
		{

		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_2 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_shuffle_pd( a_1, a_2, 0x5 );
		b_0 = _mm256_shuffle_pd( a_0, b_0, 0x5 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );

		A0 += 4;
		A1 += 4;
		A2 += 4;
		B0 += 4;
		B1 += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_8_2_lib4(int kmax, double *A0, int sda, double *B0, int sdb)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;
	double *A2 = A1 + bs*sda;
	double *B1 = B0 + bs*sdb;

	__m256d
		a_0, a_1, a_2,
		b_0, b_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		a_2 = _mm256_load_pd( &A2[0+bs*1] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		_mm256_store_pd( &B0[0+bs*1], b_0 );
		_mm256_store_pd( &B1[0+bs*1], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		a_2 = _mm256_load_pd( &A2[0+bs*2] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		_mm256_store_pd( &B0[0+bs*2], b_0 );
		_mm256_store_pd( &B1[0+bs*2], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		a_2 = _mm256_load_pd( &A2[0+bs*3] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		_mm256_store_pd( &B0[0+bs*3], b_0 );
		_mm256_store_pd( &B1[0+bs*3], b_1 );

		A0 += 16;
		A1 += 16;
		A2 += 16;
		B0 += 16;
		B1 += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );

		A0 += 4;
		A1 += 4;
		A2 += 4;
		B0 += 4;
		B1 += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_8_3_lib4(int kmax, double *A0, int sda, double *B0, int sdb)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;
	double *A2 = A1 + bs*sda;
	double *B1 = B0 + bs*sdb;

	__m256d
		a_0, a_1, a_2,
		b_0, b_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		b_1 = _mm256_shuffle_pd( b_1, a_2, 0x5 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		a_2 = _mm256_load_pd( &A2[0+bs*1] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		b_1 = _mm256_shuffle_pd( b_1, a_2, 0x5 );
		_mm256_store_pd( &B0[0+bs*1], b_0 );
		_mm256_store_pd( &B1[0+bs*1], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		a_2 = _mm256_load_pd( &A2[0+bs*2] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		b_1 = _mm256_shuffle_pd( b_1, a_2, 0x5 );
		_mm256_store_pd( &B0[0+bs*2], b_0 );
		_mm256_store_pd( &B1[0+bs*2], b_1 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		a_2 = _mm256_load_pd( &A2[0+bs*3] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		b_1 = _mm256_shuffle_pd( b_1, a_2, 0x5 );
		_mm256_store_pd( &B0[0+bs*3], b_0 );
		_mm256_store_pd( &B1[0+bs*3], b_1 );

		A0 += 16;
		A1 += 16;
		A2 += 16;
		B0 += 16;
		B1 += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_2 = _mm256_load_pd( &A2[0+bs*0] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_1 = _mm256_permute2f128_pd( a_1, a_2, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		b_1 = _mm256_shuffle_pd( b_1, a_2, 0x5 );
		_mm256_store_pd( &B0[0+bs*0], b_0 );
		_mm256_store_pd( &B1[0+bs*0], b_1 );

		A0 += 4;
		A1 += 4;
		A2 += 4;
		B0 += 4;
		B1 += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries
void kernel_align_panel_4_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	__m256d
		a_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A[0+bs*0] );
		_mm256_store_pd( &B[0+bs*0], a_0 );

		a_0 = _mm256_load_pd( &A[0+bs*1] );
		_mm256_store_pd( &B[0+bs*1], a_0 );

		a_0 = _mm256_load_pd( &A[0+bs*2] );
		_mm256_store_pd( &B[0+bs*2], a_0 );

		a_0 = _mm256_load_pd( &A[0+bs*3] );
		_mm256_store_pd( &B[0+bs*3], a_0 );

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A[0+bs*0] );
		_mm256_store_pd( &B[0+bs*0], a_0 );

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 1 element of A must be skipped
void kernel_align_panel_4_1_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m256d
		a_0, a_1,
		b_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_1 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		a_1 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*1], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		a_1 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*2], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		a_1 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*3], b_0 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_1 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_4_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m256d
		a_0, a_1,
		b_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		_mm256_store_pd( &B[0+bs*1], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		_mm256_store_pd( &B[0+bs*2], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		_mm256_store_pd( &B[0+bs*3], b_0 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		b_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_4_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m256d
		a_0, a_1,
		b_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*1] );
		a_1 = _mm256_load_pd( &A1[0+bs*1] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*1], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*2] );
		a_1 = _mm256_load_pd( &A1[0+bs*2] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*2], b_0 );

		a_0 = _mm256_load_pd( &A0[0+bs*3] );
		a_1 = _mm256_load_pd( &A1[0+bs*3] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*3], b_0 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm256_load_pd( &A0[0+bs*0] );
		a_1 = _mm256_load_pd( &A1[0+bs*0] );
		a_0 = _mm256_permute2f128_pd( a_0, a_1, 0x21 );
		b_0 = _mm256_shuffle_pd( a_0, a_1, 0x5 );
		_mm256_store_pd( &B[0+bs*0], b_0 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_3_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	__m128d
		a_0, a_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_loadu_pd( &A[0+bs*0] );
		a_1 = _mm_load_sd( &A[2+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );
		_mm_store_sd( &B[2+bs*0], a_1 );

		a_0 = _mm_loadu_pd( &A[0+bs*1] );
		a_1 = _mm_load_sd( &A[2+bs*1] );
		_mm_storeu_pd( &B[0+bs*1], a_0 );
		_mm_store_sd( &B[2+bs*1], a_1 );

		a_0 = _mm_loadu_pd( &A[0+bs*2] );
		a_1 = _mm_load_sd( &A[2+bs*2] );
		_mm_storeu_pd( &B[0+bs*2], a_0 );
		_mm_store_sd( &B[2+bs*2], a_1 );

		a_0 = _mm_loadu_pd( &A[0+bs*3] );
		a_1 = _mm_load_sd( &A[2+bs*3] );
		_mm_storeu_pd( &B[0+bs*3], a_0 );
		_mm_store_sd( &B[2+bs*3], a_1 );

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_loadu_pd( &A[0+bs*0] );
		a_1 = _mm_load_sd( &A[2+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );
		_mm_store_sd( &B[2+bs*0], a_1 );

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_3_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m128d
		a_0, a_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_loadu_pd( &A0[2+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );
		a_1 = _mm_load_sd( &A1[0+bs*0] );
		_mm_store_sd( &B[2+bs*0], a_1 );

		a_0 = _mm_loadu_pd( &A0[2+bs*1] );
		_mm_storeu_pd( &B[0+bs*1], a_0 );
		a_1 = _mm_load_sd( &A1[0+bs*1] );
		_mm_store_sd( &B[2+bs*1], a_1 );

		a_0 = _mm_loadu_pd( &A0[2+bs*2] );
		_mm_storeu_pd( &B[0+bs*2], a_0 );
		a_1 = _mm_load_sd( &A1[0+bs*2] );
		_mm_store_sd( &B[2+bs*2], a_1 );

		a_0 = _mm_loadu_pd( &A0[2+bs*3] );
		_mm_storeu_pd( &B[0+bs*3], a_0 );
		a_1 = _mm_load_sd( &A1[0+bs*3] );
		_mm_store_sd( &B[2+bs*3], a_1 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_loadu_pd( &A0[2+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );
		a_1 = _mm_load_sd( &A1[0+bs*0] );
		_mm_store_sd( &B[2+bs*0], a_1 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_3_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m128d
		a_0, a_1;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_load_sd( &A0[3+bs*0] );
		_mm_store_sd( &B[0+bs*0], a_0 );
		a_1 = _mm_loadu_pd( &A1[0+bs*0] );
		_mm_storeu_pd( &B[1+bs*0], a_1 );

		a_0 = _mm_load_sd( &A0[3+bs*1] );
		_mm_store_sd( &B[0+bs*1], a_0 );
		a_1 = _mm_loadu_pd( &A1[0+bs*1] );
		_mm_storeu_pd( &B[1+bs*1], a_1 );

		a_0 = _mm_load_sd( &A0[3+bs*2] );
		_mm_store_sd( &B[0+bs*2], a_0 );
		a_1 = _mm_loadu_pd( &A1[0+bs*2] );
		_mm_storeu_pd( &B[1+bs*2], a_1 );

		a_0 = _mm_load_sd( &A0[3+bs*3] );
		_mm_store_sd( &B[0+bs*3], a_0 );
		a_1 = _mm_loadu_pd( &A1[0+bs*3] );
		_mm_storeu_pd( &B[1+bs*3], a_1 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_load_sd( &A0[3+bs*0] );
		_mm_store_sd( &B[0+bs*0], a_0 );
		a_1 = _mm_loadu_pd( &A1[0+bs*0] );
		_mm_storeu_pd( &B[1+bs*0], a_1 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_2_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	__m128d
		a_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_loadu_pd( &A[0+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );

		a_0 = _mm_loadu_pd( &A[0+bs*1] );
		_mm_storeu_pd( &B[0+bs*1], a_0 );

		a_0 = _mm_loadu_pd( &A[0+bs*2] );
		_mm_storeu_pd( &B[0+bs*2], a_0 );

		a_0 = _mm_loadu_pd( &A[0+bs*3] );
		_mm_storeu_pd( &B[0+bs*3], a_0 );

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_loadu_pd( &A[0+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 128-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_2_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	__m128d
		a_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_load_sd( &A0[3+bs*0] );
		a_0 = _mm_loadh_pd( a_0, &A1[0+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );

		a_0 = _mm_load_sd( &A0[3+bs*1] );
		a_0 = _mm_loadh_pd( a_0, &A1[0+bs*1] );
		_mm_storeu_pd( &B[0+bs*1], a_0 );

		a_0 = _mm_load_sd( &A0[3+bs*2] );
		a_0 = _mm_loadh_pd( a_0, &A1[0+bs*2] );
		_mm_storeu_pd( &B[0+bs*2], a_0 );

		a_0 = _mm_load_sd( &A0[3+bs*3] );
		a_0 = _mm_loadh_pd( a_0, &A1[0+bs*3] );
		_mm_storeu_pd( &B[0+bs*3], a_0 );

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_load_sd( &A0[3+bs*0] );
		a_0 = _mm_loadh_pd( a_0, &A1[0+bs*0] );
		_mm_storeu_pd( &B[0+bs*0], a_0 );

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned 64-bit boundaries
void kernel_align_panel_1_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	__m128d
		a_0;
	
	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		a_0 = _mm_load_sd( &A[0+bs*0] );
		_mm_store_sd( &B[0+bs*0], a_0 );

		a_0 = _mm_load_sd( &A[0+bs*1] );
		_mm_store_sd( &B[0+bs*1], a_0 );

		a_0 = _mm_load_sd( &A[0+bs*2] );
		_mm_store_sd( &B[0+bs*2], a_0 );

		a_0 = _mm_load_sd( &A[0+bs*3] );
		_mm_store_sd( &B[0+bs*3], a_0 );

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		a_0 = _mm_load_sd( &A[0+bs*0] );
		_mm_store_sd( &B[0+bs*0], a_0 );

		A += 4;
		B += 4;

		}

	}




