/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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

// 32-bit : can not use the %ebx register with the flag -fPIC !!!                                


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
//#include <smmintrin.h>  // SSE4
//#include <immintrin.h>  // AVX



void kernel_sgemv_t_4_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{

	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;
	int ka = kmax-kna; // number from aligned positon
	int rka = ka%4;
	int qka = ka/4;
	
	int offset = (sda-1)*lda*sizeof(float);
	
/*printf("\nciao %d %d %d\n", kna, qka, rka);*/

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movl   %0, %%esi                \n\t" // load address of A
		"movl   %1, %%edi                \n\t" // load address of x
		"                                \n\t"
		"                                \n\t"
		"xorps  %%xmm0, %%xmm0           \n\t" // y_0
		"movaps %%xmm0, %%xmm1           \n\t" // y_1
		"movaps %%xmm0, %%xmm2           \n\t" // y_2
		"movaps %%xmm0, %%xmm3           \n\t" // y_3
		"                                \n\t"
		"                                \n\t"
		"movl   %6, %%eax                \n\t" // load offset
		"                                \n\t"
		"                                \n\t"
		"movl   %3, %%ecx                \n\t" // kna
		"testl  %%ecx, %%ecx             \n\t" // check kna
		"je     .DCONSMAINLOOP           \n\t" // if kna==0, jump to main loop
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DKNALOOP:                      \n\t"
		"                                \n\t"
		"movss	(%%edi), %%xmm7          \n\t"
		"decl   %%ecx                    \n\t" // kna -= 1;
		"movss	%%xmm7, %%xmm4           \n\t"
		"mulss	0(%%esi), %%xmm4         \n\t"
		"movss	%%xmm7, %%xmm5           \n\t"
		"mulss	16(%%esi), %%xmm5        \n\t"
		"movss	%%xmm7, %%xmm6           \n\t"
		"mulss	32(%%esi), %%xmm6        \n\t"
		"addss	%%xmm4, %%xmm0           \n\t"
		"mulss	48(%%esi), %%xmm7        \n\t"
		"addss	%%xmm5, %%xmm1           \n\t"
		"leal	4(%%edi), %%edi          \n\t" // x += 1
		"addss	%%xmm6, %%xmm2           \n\t"
		"leal	4(%%esi), %%esi          \n\t" // A += 1
		"addss	%%xmm7, %%xmm3           \n\t"
		"                                \n\t"
		"jne    .DKNALOOP                \n\t" // iterate again if kna != 0.
		"                                \n\t"
		"                                \n\t"
		"addl	%%eax, %%esi             \n\t" // A += offset
		"                                \n\t"
		"                                \n\t"
		".DCONSMAINLOOP:                 \n\t"
		"                                \n\t"
		"movl   %4, %%ecx                \n\t" // qka
		"testl  %%ecx, %%ecx             \n\t" // check qka
		"je     .DCONSCLEANLOOP          \n\t" // if qka==0, jump to clean-up loop
		"                                \n\t"
		"                                \n\t"
		".DMAINLOOP:                     \n\t"
		"                                \n\t"
		"movups	(%%edi), %%xmm4          \n\t"
		"decl   %%ecx                    \n\t" // qka -= 1;
		"movaps	%%xmm4, %%xmm5           \n\t"
//		"shufps	$0, %%xmm4, %%xmm4       \n\t" // unroll #1
		"mulps	0(%%esi), %%xmm4         \n\t"
		"movaps	%%xmm5, %%xmm6           \n\t"
//		"shufps	$85, %%xmm5, %%xmm5      \n\t"
		"mulps	16(%%esi), %%xmm5        \n\t"
		"movaps	%%xmm6, %%xmm7           \n\t"
//		"shufps	$170, %%xmm6, %%xmm6     \n\t"
		"mulps	32(%%esi), %%xmm6        \n\t"
		"addps	%%xmm4, %%xmm0           \n\t"
//		"shufps	$255, %%xmm7, %%xmm7     \n\t"
		"mulps	48(%%esi), %%xmm7        \n\t"
		"addps	%%xmm5, %%xmm1           \n\t"
		"leal	16(%%edi), %%edi         \n\t" // x += 4
		"addps	%%xmm6, %%xmm2           \n\t"
		"leal	16(%%esi, %%eax), %%esi         \n\t" // A += 4
		"addps	%%xmm7, %%xmm3           \n\t"
		"                                \n\t"
		"jne    .DMAINLOOP               \n\t" // iterate again if kna != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSCLEANLOOP:                \n\t"
		"                                \n\t"
		"movl   %5, %%ecx                \n\t" // rka
		"testl  %%ecx, %%ecx             \n\t" // check rka
		"je     .DPOSTLOOP               \n\t" // if rka==0, jump to post loop
		"                                \n\t"
		".DCLEANLOOP:                    \n\t"
		"                                \n\t"
		"movss	(%%edi), %%xmm7          \n\t"
		"decl   %%ecx                    \n\t" // kna -= 1;
		"movss	%%xmm7, %%xmm4           \n\t"
		"mulss	0(%%esi), %%xmm4         \n\t"
		"movss	%%xmm7, %%xmm5           \n\t"
		"mulss	16(%%esi), %%xmm5        \n\t"
		"movss	%%xmm7, %%xmm6           \n\t"
		"mulss	32(%%esi), %%xmm6        \n\t"
		"addss	%%xmm4, %%xmm0           \n\t"
		"mulss	48(%%esi), %%xmm7        \n\t"
		"addss	%%xmm5, %%xmm1           \n\t"
		"leal	4(%%edi), %%edi          \n\t" // x += 1
		"addss	%%xmm6, %%xmm2           \n\t"
		"leal	4(%%esi), %%esi          \n\t" // A += 1
		"addss	%%xmm7, %%xmm3           \n\t"
		"                                \n\t"
		"jne    .DCLEANLOOP              \n\t" // iterate again if kna != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTLOOP:                     \n\t"
		"                                \n\t"
		"haddps %%xmm1, %%xmm0           \n\t"
		"haddps %%xmm3, %%xmm2           \n\t"
		"haddps %%xmm2, %%xmm0           \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %2, %%eax                \n\t" // load address of y
		"                                \n\t"
		"                                \n\t"
		"movl   %7, %%ecx                \n\t" // alg
		"testl  %%ecx, %%ecx             \n\t" // check alg
		"je     .DZERO                   \n\t" // if alg==0, jump
		"                                \n\t"
		"cmpl	$1, %%ecx                \n\t"
		"                                \n\t"
		"movups  (%%eax), %%xmm4         \n\t"
		"                                \n\t"
		"je     .DONE                    \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"subps  %%xmm0, %%xmm4           \n\t"
		"                                \n\t"
		"movups  %%xmm4, (%%eax)         \n\t"
		"                                \n\t"
		"jmp    .DEND                    \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DONE:                          \n\t" // alg==1
		"                                \n\t"
		"addps  %%xmm4, %%xmm0           \n\t"
		"                                \n\t"
		".DZERO:                         \n\t" // alg==0
		"                                \n\t"
		"movups	%%xmm0, (%%eax)          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DEND:                          \n\t" // end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "m" (A),			// %0
		  "m" (x),			// %1
		  "m" (y),			// %2
		  "m" (kna),		// %3
		  "m" (qka),		// %4
		  "m" (rka),		// %5
		  "m" (offset),		// %6
		  "m" (alg)			// %7
		: // register clobber list
		  "eax", "ecx", "esi", "edi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);
}



void kernel_sgemv_t_2_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m128
		a_00_10_20_30, a_01_11_21_31,
		x_0_1_2_3,
		y_0, y_1;
	
	y_0 = _mm_setzero_ps();	
	y_1 = _mm_setzero_ps();	

	k = 0;
	if(kna>0)
		{
		for(; k<kna; k++)
			{
		
			x_0_1_2_3 = _mm_load_ss( &x[0] );

			a_00_10_20_30 = _mm_load_ss( &A[0+lda*0] );
			a_01_11_21_31 = _mm_load_ss( &A[0+lda*1] );
		
/*			y_0 += a_00_10_20_30 * x_0_1_2_3;*/
/*			y_1 += a_01_11_21_31 * x_0_1_2_3;*/
			a_00_10_20_30 = _mm_mul_ss( a_00_10_20_30, x_0_1_2_3 );
			a_01_11_21_31 = _mm_mul_ss( a_01_11_21_31, x_0_1_2_3 );
			y_0 = _mm_add_ss( y_0, a_00_10_20_30 );
			y_1 = _mm_add_ss( y_1, a_01_11_21_31 );
		
			x += 1;
			A += 1;

			}

		A += (sda-1)*lda;
		}

	k = 0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1_2_3 = _mm_loadu_ps( &x[0] );

		a_00_10_20_30 = _mm_load_ps( &A[0+lda*0] );
		a_01_11_21_31 = _mm_load_ps( &A[0+lda*1] );
		
/*		y_0 += a_00_10_20_30 * x_0_1_2_3;*/
/*		y_1 += a_01_11_21_31 * x_0_1_2_3;*/
		a_00_10_20_30 = _mm_mul_ps( a_00_10_20_30, x_0_1_2_3 );
		a_01_11_21_31 = _mm_mul_ps( a_01_11_21_31, x_0_1_2_3 );
		y_0 = _mm_add_ps( y_0, a_00_10_20_30 );
		y_1 = _mm_add_ps( y_1, a_01_11_21_31 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka; k++)
		{
		
		x_0_1_2_3 = _mm_load_ss( &x[0] );

		a_00_10_20_30 = _mm_load_ss( &A[0+lda*0] );
		a_01_11_21_31 = _mm_load_ss( &A[0+lda*1] );
	
/*		y_0 += a_00_10_20_30 * x_0_1_2_3;*/
/*		y_1 += a_01_11_21_31 * x_0_1_2_3;*/
		a_00_10_20_30 = _mm_mul_ss( a_00_10_20_30, x_0_1_2_3 );
		a_01_11_21_31 = _mm_mul_ss( a_01_11_21_31, x_0_1_2_3 );
		y_0 = _mm_add_ss( y_0, a_00_10_20_30 );
		y_1 = _mm_add_ss( y_1, a_01_11_21_31 );
	
		x += 1;
		A += 1;
		
		}

	__m128
		y_0_1_2_3;

	y_0 = _mm_hadd_ps(y_0, y_1);
	y_1 = _mm_setzero_ps();
	y_0 = _mm_hadd_ps(y_0, y_1);

	if(alg==0)
		{
		y_1 = _mm_shuffle_ps( y_0, y_0, 0x1 );
		_mm_store_ss( &y[0], y_0 );
		_mm_store_ss( &y[1], y_1 );
		}
	else if(alg==1)
		{
		y_0_1_2_3 = _mm_loadu_ps( &y[0] );

		y_0_1_2_3 = _mm_add_ps(y_0_1_2_3, y_0);
	
		y_1 = _mm_shuffle_ps( y_0_1_2_3, y_0_1_2_3, 0x1 );

	//	_mm_storeu_ps(&y[0], y_0_1_2_3); // seg fault if free y_2 or y_3 !!!
		_mm_store_ss( &y[0], y_0_1_2_3 );
		_mm_store_ss( &y[1], y_1 );
		}
	else // alg==-1
		{
		y_0_1_2_3 = _mm_loadu_ps( &y[0] );

		y_0_1_2_3 = _mm_sub_ps(y_0_1_2_3, y_0);
	
		y_1 = _mm_shuffle_ps( y_0_1_2_3, y_0_1_2_3, 0x1 );

	//	_mm_storeu_ps(&y[0], y_0_1_2_3); // seg fault if free y_2 or y_3 !!!
		_mm_store_ss( &y[0], y_0_1_2_3 );
		_mm_store_ss( &y[1], y_1 );
		}

	}



void kernel_sgemv_t_1_lib4(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	if(kmax<=0) return;
	
	const int lda = 4;
	
	int k;
	int ka = kmax-kna; // number from aligned positon
	
	__m128
		a_00_10_20_30,
		x_0_1_2_3,
		y_0, y_1;
	
	y_0 = _mm_setzero_ps();	

	k = 0;
	if(kna>0)
		{
		for(; k<kna; k++)
			{
		
			x_0_1_2_3 = _mm_load_ss( &x[0] );

			a_00_10_20_30 = _mm_load_ss( &A[0+lda*0] );
		
/*			y_0 += a_00_10_20_30 * x_0_1_2_3;*/
			a_00_10_20_30 = _mm_mul_ss( a_00_10_20_30, x_0_1_2_3 );
			y_0 = _mm_add_ss( y_0, a_00_10_20_30 );
		
			x += 1;
			A += 1;

			}

		A += (sda-1)*lda;
		}

	k = 0;
	for(; k<ka-3; k+=4)
		{
		
		x_0_1_2_3 = _mm_loadu_ps( &x[0] );

		a_00_10_20_30 = _mm_load_ps( &A[0+lda*0] );
		
/*		y_0 += a_00_10_20_30 * x_0_1_2_3;*/
		a_00_10_20_30 = _mm_mul_ps( a_00_10_20_30, x_0_1_2_3 );
		y_0 = _mm_add_ps( y_0, a_00_10_20_30 );
		
		x += 4;
		A += 4;

		A += (sda-1)*lda;

		}
	for(; k<ka; k++)
		{
		
		x_0_1_2_3 = _mm_load_ss( &x[0] );

		a_00_10_20_30 = _mm_load_ss( &A[0+lda*0] );
	
/*		y_0 += a_00_10_20_30 * x_0_1_2_3;*/
		a_00_10_20_30 = _mm_mul_ss( a_00_10_20_30, x_0_1_2_3 );
		y_0 = _mm_add_ss( y_0, a_00_10_20_30 );
	
		x += 1;
		A += 1;
		
		}

	__m128
		y_0_1_2_3;

	y_1 = _mm_setzero_ps();
	y_0 = _mm_hadd_ps(y_0, y_1);
	y_0 = _mm_hadd_ps(y_0, y_1);

	if(alg==0)
		{
		_mm_store_ss(&y[0], y_0);
		}
	else if(alg==1)
		{
		y_0_1_2_3 = _mm_load_ss( &y[0] );

		y_0_1_2_3 = _mm_add_ss(y_0_1_2_3, y_0);
	
		_mm_store_ss(&y[0], y_0_1_2_3);
		}
	else // alg==-1
		{
		y_0_1_2_3 = _mm_load_ss( &y[0] );

		y_0_1_2_3 = _mm_sub_ss(y_0_1_2_3, y_0);
	
		_mm_store_ss(&y[0], y_0_1_2_3);
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_4_lib4(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128
		a_00_10_20_30, a_01_11_21_31,
		x_0, x_1,
		y_0_1_2_3, y_0_1_2_3_b, z_0_1_2_3;
	
	y_0_1_2_3 = _mm_setzero_ps();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load1_ps( &x[0] );
		x_1 = _mm_load1_ps( &x[1] );

		a_00_10_20_30 = _mm_load_ps( &A[0+lda*0] );
		a_01_11_21_31 = _mm_load_ps( &A[0+lda*1] );

		y_0_1_2_3   += a_00_10_20_30 * x_0;
		y_0_1_2_3_b += a_01_11_21_31 * x_1;

		x_0 = _mm_load1_ps( &x[2] );
		x_1 = _mm_load1_ps( &x[3] );

		a_00_10_20_30 = _mm_load_ps( &A[0+lda*2] );
		a_01_11_21_31 = _mm_load_ps( &A[0+lda*3] );

		y_0_1_2_3   += a_00_10_20_30 * x_0;
		y_0_1_2_3_b += a_01_11_21_31 * x_1;
		
		A += 4*lda;
		x += 4;

		}
	
	y_0_1_2_3 += y_0_1_2_3_b;

	for(; k<kmax; k++)
		{

		x_0 = _mm_load1_ps( &x[0] );

		a_00_10_20_30 = _mm_load_ps( &A[0+lda*0] );

		y_0_1_2_3 += a_00_10_20_30 * x_0;
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		_mm_storeu_ps(&y[0], y_0_1_2_3);
		}
	else if(alg==1)
		{
		z_0_1_2_3 = _mm_loadu_ps( &y[0] );

		z_0_1_2_3 = _mm_add_ps( z_0_1_2_3, y_0_1_2_3 );

		_mm_storeu_ps(&y[0], z_0_1_2_3);
		}
	else // alg==-1
		{
		z_0_1_2_3 = _mm_loadu_ps( &y[0] );

		z_0_1_2_3 = _mm_sub_ps( z_0_1_2_3, y_0_1_2_3 );

		_mm_storeu_ps(&y[0], z_0_1_2_3);
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_2_lib4(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128
		a_00, a_01,
		a_10, a_11,
		x_0, x_1,
		y_0, y_1, z_0, z_1;
	
	y_0 = _mm_setzero_ps();	
	y_1 = _mm_setzero_ps();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load_ss( &x[0] );
		x_1 = _mm_load_ss( &x[1] );

		a_00 = _mm_load_ss( &A[0+lda*0] );
		a_10 = _mm_load_ss( &A[1+lda*0] );
		a_01 = _mm_load_ss( &A[0+lda*1] );
		a_11 = _mm_load_ss( &A[1+lda*1] );

		y_0 += a_00 * x_0;
		y_1 += a_10 * x_0;
		y_0 += a_01 * x_1;
		y_1 += a_11 * x_1;

		x_0 = _mm_load_ss( &x[2] );
		x_1 = _mm_load_ss( &x[3] );

		a_00 = _mm_load_ss( &A[0+lda*2] );
		a_10 = _mm_load_ss( &A[1+lda*2] );
		a_01 = _mm_load_ss( &A[0+lda*3] );
		a_11 = _mm_load_ss( &A[1+lda*3] );

		y_0 += a_00 * x_0;
		y_1 += a_10 * x_0;
		y_0 += a_01 * x_1;
		y_1 += a_11 * x_1;
		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = _mm_load_ss( &x[0] );

		a_00 = _mm_load_ss( &A[0+lda*0] );
		a_10 = _mm_load_ss( &A[1+lda*0] );

		y_0 += a_00 * x_0;
		y_1 += a_10 * x_0;
		
		A += 1*lda;
		x += 1;

		}


	if(alg==0)
		{
		_mm_store_ss(&y[0], y_0);
		_mm_store_ss(&y[1], y_1);
		}
	else if(alg==1)
		{
		z_0 = _mm_load_ss( &y[0] );
		z_1 = _mm_load_ss( &y[1] );

		z_0 += y_0;
		z_1 += y_1;

		_mm_store_ss(&y[0], z_0);
		_mm_store_ss(&y[1], z_1);
		}
	else // alg==-1
		{
		z_0 = _mm_load_ss( &y[0] );
		z_1 = _mm_load_ss( &y[1] );

		z_0 -= y_0;
		z_1 -= y_1;

		_mm_store_ss(&y[0], z_0);
		_mm_store_ss(&y[1], z_1);
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_1_lib4(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 4;
	
	int k;

	__m128
		a_00, a_01,
		x_0, x_1,
		y_0, y_0_b, z_0;
	
	y_0 = _mm_setzero_ps();	

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = _mm_load_ss( &x[0] );
		x_1 = _mm_load_ss( &x[1] );

		a_00 = _mm_load_ss( &A[0+lda*0] );
		a_01 = _mm_load_ss( &A[0+lda*1] );

		y_0   += a_00 * x_0;
		y_0_b += a_01 * x_1;

		x_0 = _mm_load_ss( &x[2] );
		x_1 = _mm_load_ss( &x[3] );

		a_00 = _mm_load_ss( &A[0+lda*2] );
		a_01 = _mm_load_ss( &A[0+lda*3] );

		y_0   += a_00 * x_0;
		y_0_b += a_01 * x_1;
		
		A += 4*lda;
		x += 4;

		}

	y_0 += y_0_b;

	for(; k<kmax; k++)
		{

		x_0 = _mm_load_ss( &x[0] );

		a_00 = _mm_load_ss( &A[0+lda*0] );

		y_0 += a_00 * x_0;
		
		A += 1*lda;
		x += 1;

		}


	if(alg==0)
		{
		_mm_store_ss(&y[0], y_0);
		}
	else if(alg==1)
		{
		z_0 = _mm_load_ss( &y[0] );

		z_0 += y_0;

		_mm_store_ss(&y[0], z_0);
		}
	else // alg==-1
		{
		z_0 = _mm_load_ss( &y[0] );

		z_0 -= y_0;

		_mm_store_ss(&y[0], z_0);
		}

	}

