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



// 4x4 with data packed in 4
void kernel_sgemm_pp_nt_4x4_atom_lib4(int kmax, float *A, float *B, float *C, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movl   %2, %%eax                \n\t" // load address of A
		"movl   %3, %%edx                \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"prefetcht0 0(%%eax)             \n\t"
		"prefetcht0 0(%%edx)             \n\t"
		"prefetcht0 64(%%eax)            \n\t"
		"prefetcht0 64(%%edx)            \n\t"
		"                                \n\t"
		"                                \n\t" // zero registers
		"movaps (%%edx), %%xmm0          \n\t" // B[0]
		"                                \n\t"
		"                                \n\t" // zero registers
		"xorps  %%xmm2, %%xmm2           \n\t" //
		"movaps %%xmm2, %%xmm1           \n\t" //
		"movaps %%xmm2, %%xmm3           \n\t" //
		"movaps %%xmm2, %%xmm4           \n\t" //
		"movaps %%xmm2, %%xmm5           \n\t"
		"movaps %%xmm2, %%xmm6           \n\t"
		"movaps %%xmm2, %%xmm7           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %0, %%ecx                \n\t" // i = k_iter;
		"testl  %%ecx, %%ecx             \n\t" // check i
		"je     .DCONSIDKLEFT3            \n\t" // if i==0, jump to k_left loop
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER3:                   \n\t" // MAIN LOOP
		"                                \n\t"
		"                                \n\t"
		"prefetcht0 128(%%eax)           \n\t"
		"prefetcht0 128(%%edx)           \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // unroll #1
		"mulps	0(%%eax), %%xmm0         \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	%%xmm1, %%xmm2           \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t"
		"mulps	0(%%eax), %%xmm1         \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$170, %%xmm2, %%xmm2     \n\t"
		"mulps	0(%%eax), %%xmm2         \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	16(%%edx), %%xmm0        \n\t"
		"shufps	$255, %%xmm3, %%xmm3     \n\t"
		"mulps	0(%%eax), %%xmm3         \n\t"
		"decl   %%ecx                    \n\t" // i -= 1;
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // unroll #1
		"mulps	16(%%eax), %%xmm0        \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	%%xmm1, %%xmm2           \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t"
		"mulps	16(%%eax), %%xmm1        \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$170, %%xmm2, %%xmm2     \n\t"
		"mulps	16(%%eax), %%xmm2        \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	32(%%edx), %%xmm0        \n\t"
		"shufps	$255, %%xmm3, %%xmm3     \n\t"
		"mulps	16(%%eax), %%xmm3        \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // unroll #1
		"mulps	32(%%eax), %%xmm0        \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	%%xmm1, %%xmm2           \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t"
		"mulps	32(%%eax), %%xmm1        \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$170, %%xmm2, %%xmm2     \n\t"
		"mulps	32(%%eax), %%xmm2        \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	48(%%edx), %%xmm0        \n\t"
		"shufps	$255, %%xmm3, %%xmm3     \n\t"
		"mulps	32(%%eax), %%xmm3        \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // unroll #1
		"mulps	48(%%eax), %%xmm0        \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	%%xmm1, %%xmm2           \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t"
		"mulps	48(%%eax), %%xmm1        \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$170, %%xmm2, %%xmm2     \n\t"
		"mulps	48(%%eax), %%xmm2        \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	64(%%edx), %%xmm0        \n\t"
		"leal	64(%%edx), %%edx         \n\t"
		"shufps	$255, %%xmm3, %%xmm3     \n\t"
		"mulps	48(%%eax), %%xmm3        \n\t"
		"leal	64(%%eax), %%eax         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKITER3             \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDKLEFT3:                 \n\t"
		"                                \n\t"
		"movl   %1, %%ecx                \n\t" // i = k_left;
		"testl  %%ecx, %%ecx             \n\t" // check i via logical AND.
		"je     .DPOSTACCUM3             \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT3:                   \n\t" // EDGE LOOP
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // unroll #1
		"mulps	0(%%eax), %%xmm0         \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	%%xmm1, %%xmm2           \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t"
		"mulps	0(%%eax), %%xmm1         \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$170, %%xmm2, %%xmm2     \n\t"
		"mulps	0(%%eax), %%xmm2         \n\t"
		"shufps	$255, %%xmm3, %%xmm3     \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	16(%%edx), %%xmm0        \n\t"
		"leal	16(%%edx), %%edx         \n\t"
		"mulps	0(%%eax), %%xmm3         \n\t"
		"decl   %%ecx                    \n\t" // i -= 1;
		"leal	16(%%eax), %%eax         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKLEFT3             \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM3:                   \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %4, %%eax                \n\t" // load address of C
		"                                \n\t"
		"                                \n\t"
		"movl   %5, %%ecx                \n\t" // alg
		"testl  %%ecx, %%ecx             \n\t" // check alg
		"je     .D03                     \n\t" // if alg==0, jump
		"                                \n\t"
		"cmpl	$1, %%ecx                \n\t"
		"movaps  (%%eax), %%xmm0         \n\t"
		"movaps  16(%%eax), %%xmm1       \n\t"
		"movaps  32(%%eax), %%xmm2       \n\t"
		"movaps  48(%%eax), %%xmm3       \n\t"
		"je     .D13                     \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"subps  %%xmm4, %%xmm0           \n\t"
		"subps  %%xmm5, %%xmm1           \n\t"
		"subps  %%xmm6, %%xmm2           \n\t"
		"subps  %%xmm7, %%xmm3           \n\t"
		"                                \n\t"
		"movaps  %%xmm0, (%%eax)         \n\t"
		"movaps  %%xmm1, 16(%%eax)       \n\t"
		"movaps  %%xmm2, 32(%%eax)       \n\t"
		"movaps  %%xmm3, 48(%%eax)       \n\t"
		"                                \n\t"
		"jmp    .SDONE3                  \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".D13:                           \n\t" // alg==1
		"                                \n\t"
		"addps  %%xmm0, %%xmm4           \n\t"
		"addps  %%xmm1, %%xmm5           \n\t"
		"addps  %%xmm2, %%xmm6           \n\t"
		"addps  %%xmm3, %%xmm7           \n\t"
		"                                \n\t"
		".D03:                           \n\t" // alg==0
		"                                \n\t"
		"movaps	%%xmm4, (%%eax)          \n\t"
		"movaps	%%xmm5, 16(%%eax)        \n\t"
		"movaps	%%xmm6, 32(%%eax)        \n\t"
		"movaps	%%xmm7, 48(%%eax)        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".SDONE3:                        \n\t" // end
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "m" (k_iter),		// %0
		  "m" (k_left),		// %1
		  "m" (A),			// %2
		  "m" (B),			// %3
		  "m" (C),			// %4
		  "m" (alg)			// %5
		: // register clobber list
		  "eax", "edx", "ecx",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);
}



// normal-transposed, 4x3 with data packed in 4
void kernel_sgemm_pp_nt_4x3_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int bs = 4;*/

	int k;
	
	float
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2,
		c_00=0, c_01=0, c_02=0,
		c_10=0, c_11=0, c_12=0,
		c_20=0, c_21=0, c_22=0,
		c_30=0, c_31=0, c_32=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;
		
		
		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];
		
		b_0 = B[4];
		b_1 = B[5];
		b_2 = B[6];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;
		
		
		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];
		
		b_0 = B[8];
		b_1 = B[9];
		b_2 = B[10];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;
		
		
		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];
		
		b_0 = B[12];
		b_1 = B[13];
		b_2 = B[14];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;
		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		b_1 = B[1];
		b_2 = B[2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;
		
		A += 4;
		B += 4;
		
		}

	if(alg==0)
		{
		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;
		C[2+ldc*0] = c_20;
		C[3+ldc*0] = c_30;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;
		C[2+ldc*1] = c_21;
		C[3+ldc*1] = c_31;

		C[0+ldc*2] = c_02;
		C[1+ldc*2] = c_12;
		C[2+ldc*2] = c_22;
		C[3+ldc*2] = c_32;
		}
	else if(alg==1)
		{
		C[0+ldc*0] += c_00;
		C[1+ldc*0] += c_10;
		C[2+ldc*0] += c_20;
		C[3+ldc*0] += c_30;

		C[0+ldc*1] += c_01;
		C[1+ldc*1] += c_11;
		C[2+ldc*1] += c_21;
		C[3+ldc*1] += c_31;

		C[0+ldc*2] += c_02;
		C[1+ldc*2] += c_12;
		C[2+ldc*2] += c_22;
		C[3+ldc*2] += c_32;
		}
	else
		{
		C[0+ldc*0] -= c_00;
		C[1+ldc*0] -= c_10;
		C[2+ldc*0] -= c_20;
		C[3+ldc*0] -= c_30;

		C[0+ldc*1] -= c_01;
		C[1+ldc*1] -= c_11;
		C[2+ldc*1] -= c_21;
		C[3+ldc*1] -= c_31;

		C[0+ldc*2] -= c_02;
		C[1+ldc*2] -= c_12;
		C[2+ldc*2] -= c_22;
		C[3+ldc*2] -= c_32;
		}

	}



// normal-transposed, 4x2 with data packed in 4
void kernel_sgemm_pp_nt_4x2_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	float
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0,
		c_20=0, c_21=0,
		c_30=0, c_31=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		b_1 = B[1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];
		
		b_0 = B[4];
		b_1 = B[5];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];
		
		b_0 = B[8];
		b_1 = B[9];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];
		
		b_0 = B[12];
		b_1 = B[13];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		b_1 = B[1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		A += 4;
		B += 4;
		
		}

	if(alg==0)
		{
		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;
		C[2+ldc*0] = c_20;
		C[3+ldc*0] = c_30;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;
		C[2+ldc*1] = c_21;
		C[3+ldc*1] = c_31;
		}
	else if(alg==1)
		{
		C[0+ldc*0] += c_00;
		C[1+ldc*0] += c_10;
		C[2+ldc*0] += c_20;
		C[3+ldc*0] += c_30;

		C[0+ldc*1] += c_01;
		C[1+ldc*1] += c_11;
		C[2+ldc*1] += c_21;
		C[3+ldc*1] += c_31;
		}
	else
		{
		C[0+ldc*0] -= c_00;
		C[1+ldc*0] -= c_10;
		C[2+ldc*0] -= c_20;
		C[3+ldc*0] -= c_30;

		C[0+ldc*1] -= c_01;
		C[1+ldc*1] -= c_11;
		C[2+ldc*1] -= c_21;
		C[3+ldc*1] -= c_31;
		}

	}



// 4x4 with data packed in 4
void kernel_sgemm_pp_nt_4x2_atom_lib4(int kmax, float *A, float *B, float *C, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movl   %2, %%eax                \n\t" // load address of A
		"movl   %3, %%edx                \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
/*		"prefetcht0 0(%%eax)             \n\t"*/
/*		"prefetcht0 0(%%edx)             \n\t"*/
/*		"prefetcht0 64(%%eax)            \n\t"*/
/*		"prefetcht0 64(%%edx)            \n\t"*/
		"                                \n\t"
		"                                \n\t" // zero registers
		"movaps (%%edx), %%xmm0          \n\t" // B[0]
		"                                \n\t"
		"                                \n\t" // zero registers
		"xorps  %%xmm2, %%xmm2           \n\t" //
		"movaps %%xmm2, %%xmm1           \n\t" //
		"movaps %%xmm2, %%xmm3           \n\t" //
		"movaps %%xmm2, %%xmm4           \n\t" // c_03_0_a
		"movaps %%xmm2, %%xmm5           \n\t" // c_03_1_a
		"movaps %%xmm2, %%xmm6           \n\t" // c_03_0_b
		"movaps %%xmm2, %%xmm7           \n\t" // c_03_1_b
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %0, %%ecx                \n\t" // i = k_iter;
		"testl  %%ecx, %%ecx             \n\t" // check i
		"je     .DCONSIDKLEFT2            \n\t" // if i==0, jump to k_left loop
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER2:                   \n\t" // MAIN LOOP
		"                                \n\t"
		"                                \n\t"
/*		"prefetcht0 128(%%eax)           \n\t"*/
/*		"prefetcht0 128(%%edx)           \n\t"*/
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // b_0
		"mulps	0(%%eax), %%xmm0         \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	16(%%edx), %%xmm2        \n\t" // b_k+1
		"shufps	$85, %%xmm1, %%xmm1      \n\t" // b_1
		"mulps	0(%%eax), %%xmm1         \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$0, %%xmm2, %%xmm2       \n\t" // b_0
		"mulps	16(%%eax), %%xmm2        \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	32(%%edx), %%xmm0        \n\t" // b_k+2
		"shufps	$85, %%xmm3, %%xmm3      \n\t" // b_1
		"mulps	16(%%eax), %%xmm3        \n\t"
		"                                \n\t"
		"decl   %%ecx                    \n\t" // i -= 1;
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // b_0
		"mulps	32(%%eax), %%xmm0        \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"movaps	48(%%edx), %%xmm2        \n\t" // b_k+1
		"shufps	$85, %%xmm1, %%xmm1      \n\t" // b_1
		"leal	64(%%edx), %%edx         \n\t"
		"mulps	32(%%eax), %%xmm1        \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"movaps	%%xmm2, %%xmm3           \n\t"
		"shufps	$0, %%xmm2, %%xmm2       \n\t" // b_0
		"mulps	48(%%eax), %%xmm2        \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	0(%%edx), %%xmm0        \n\t" // b_k+2
		"shufps	$85, %%xmm3, %%xmm3      \n\t" // b_1
		"mulps	48(%%eax), %%xmm3        \n\t"
		"leal	64(%%eax), %%eax         \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKITER2             \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"addps	%%xmm2, %%xmm6           \n\t"
		"addps	%%xmm3, %%xmm7           \n\t"
		"addps	%%xmm6, %%xmm4           \n\t"
		"addps	%%xmm7, %%xmm5           \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDKLEFT2:                 \n\t"
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"movl   %1, %%ecx                \n\t" // i = k_left;
		"testl  %%ecx, %%ecx             \n\t" // check i via logical AND.
		"je     .DPOSTACCUM2             \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT2:                   \n\t" // EDGE LOOP
		"                                \n\t"
		"                                \n\t"
		"movaps	%%xmm0, %%xmm1           \n\t"
		"shufps	$0, %%xmm0, %%xmm0       \n\t" // b_0
		"mulps	0(%%eax), %%xmm0         \n\t"
		"shufps	$85, %%xmm1, %%xmm1      \n\t" // b_1
		"mulps	0(%%eax), %%xmm1         \n\t"
		"decl   %%ecx                    \n\t" // i -= 1;
		"leal	16(%%edx), %%edx         \n\t"
		"addps	%%xmm0, %%xmm4           \n\t"
		"movaps	0(%%edx), %%xmm0        \n\t" // b_k+1
		"leal	16(%%eax), %%eax         \n\t"
		"addps	%%xmm1, %%xmm5           \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKLEFT2             \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM2:                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %4, %%eax                \n\t" // load address of C
		"                                \n\t"
		"                                \n\t"
		"movl   %5, %%ecx                \n\t" // alg
		"testl  %%ecx, %%ecx             \n\t" // check alg
		"je     .DZERO2                  \n\t" // if alg==0, jump
		"                                \n\t"
		"cmpl	$1, %%ecx                \n\t"
		"movaps  (%%eax), %%xmm0         \n\t"
		"movaps  16(%%eax), %%xmm1       \n\t"
		"je     .DONE2                   \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"subps  %%xmm4, %%xmm0           \n\t"
		"subps  %%xmm5, %%xmm1           \n\t"
		"                                \n\t"
		"movaps  %%xmm0, (%%eax)         \n\t"
		"movaps  %%xmm1, 16(%%eax)       \n\t"
		"                                \n\t"
		"jmp    .DEND2                   \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DONE2:                         \n\t" // alg==1
		"                                \n\t"
		"addps  %%xmm0, %%xmm4           \n\t"
		"addps  %%xmm1, %%xmm5           \n\t"
		"                                \n\t"
		".DZERO2:                        \n\t" // alg==0
		"                                \n\t"
		"movaps	%%xmm4, (%%eax)          \n\t"
		"movaps	%%xmm5, 16(%%eax)        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DEND2 :                        \n\t" // end
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "m" (k_iter),		// %0
		  "m" (k_left),		// %1
		  "m" (A),			// %2
		  "m" (B),			// %3
		  "m" (C),			// %4
		  "m" (alg)			// %5
		: // register clobber list
		  "eax", "edx", "ecx",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);
}



// normal-transposed, 4x1 with data packed in 4
void kernel_sgemm_pp_nt_4x1_c99_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	float
		a_0, a_1, a_2, a_3,
		b_0,
		c_00=0,
		c_10=0,
		c_20=0,
		c_30=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		
		a_0 = A[4];
		a_1 = A[5];
		a_2 = A[6];
		a_3 = A[7];
		
		b_0 = B[4];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		
		a_0 = A[8];
		a_1 = A[9];
		a_2 = A[10];
		a_3 = A[11];
		
		b_0 = B[8];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		
		a_0 = A[12];
		a_1 = A[13];
		a_2 = A[14];
		a_3 = A[15];
		
		b_0 = B[12];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		A += 16;
		B += 16;

		}
	
	for(; k<kmax; k++)
		{

		a_0 = A[0];
		a_1 = A[1];
		a_2 = A[2];
		a_3 = A[3];
		
		b_0 = B[0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;
		
		A += 4;
		B += 4;
		
		}

	if(alg==0)
		{
		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;
		C[2+ldc*0] = c_20;
		C[3+ldc*0] = c_30;
		}
	else if(alg==1)
		{
		C[0+ldc*0] += c_00;
		C[1+ldc*0] += c_10;
		C[2+ldc*0] += c_20;
		C[3+ldc*0] += c_30;
		}
	else
		{
		C[0+ldc*0] -= c_00;
		C[1+ldc*0] -= c_10;
		C[2+ldc*0] -= c_20;
		C[3+ldc*0] -= c_30;
		}

	}


