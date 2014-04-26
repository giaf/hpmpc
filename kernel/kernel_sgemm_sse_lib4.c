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
/*#include <pmmintrin.h>  // SSE3*/
//#include <smmintrin.h>  // SSE4
//#include <immintrin.h>  // AVX



// normal-transposed, 8x4 with data packed in 4
void kernel_sgemm_pp_nt_8x4_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
	
	int k_iter = kmax / 4;
	int k_left = kmax % 4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movq          %2, %%rax         \n\t" // load address of A0
		"movq          %3, %%rcx         \n\t" // load address of A1
		"movq          %4, %%rbx         \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"movaps        0(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps        0(%%rcx), %%xmm1  \n\t" // of a and b.
		"movaps        0(%%rbx), %%xmm2  \n\t"
		"                                \n\t"
		"                                \n\t"
		"xorpd     %%xmm3,  %%xmm3       \n\t"
		"movaps    %%xmm3,  %%xmm4       \n\t"
		"movaps    %%xmm3,  %%xmm5       \n\t"
		"movaps    %%xmm3,  %%xmm6       \n\t"
		"movaps    %%xmm3,  %%xmm7       \n\t"
		"movaps    %%xmm3,  %%xmm8       \n\t"
		"movaps    %%xmm3,  %%xmm9       \n\t"
		"movaps    %%xmm3, %%xmm10       \n\t"
		"movaps    %%xmm3, %%xmm11       \n\t"
		"movaps    %%xmm3, %%xmm12       \n\t"
		"movaps    %%xmm3, %%xmm13       \n\t"
		"movaps    %%xmm3, %%xmm14       \n\t"
		"movaps    %%xmm3, %%xmm15       \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl      %0, %%esi             \n\t" // i = k_iter;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .SCONSIDKLEFT            \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".SLOOPKITER:                    \n\t" // MAIN LOOP
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t" // iteration 0
		"addps   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm2, %%xmm7  \n\t"
		"mulps   %%xmm0, %%xmm2          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"pshufd   $0x39, %%xmm7, %%xmm6  \n\t"
		"mulps   %%xmm0, %%xmm7          \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addps   %%xmm2, %%xmm8          \n\t"
		"movaps       16(%%rbx), %%xmm2  \n\t"
		"addps   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm6, %%xmm4  \n\t"
		"mulps   %%xmm0, %%xmm6          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm7, %%xmm9          \n\t"
		"addps   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
		"mulps   %%xmm0, %%xmm4          \n\t"
		"movaps       16(%%rax), %%xmm0  \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"movaps       16(%%rcx), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t" // iteration 1
		"addps   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm2, %%xmm7  \n\t"
		"mulps   %%xmm0, %%xmm2          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"pshufd   $0x39, %%xmm7, %%xmm6  \n\t"
		"mulps   %%xmm0, %%xmm7          \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addps   %%xmm2, %%xmm8          \n\t"
		"movaps       32(%%rbx), %%xmm2  \n\t"
		"addps   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm6, %%xmm4  \n\t"
		"mulps   %%xmm0, %%xmm6          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm7, %%xmm9          \n\t"
		"addps   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
		"mulps   %%xmm0, %%xmm4          \n\t"
		"movaps       32(%%rax), %%xmm0  \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"movaps       32(%%rcx), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t" // iteration 2
		"addps   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm2, %%xmm7  \n\t"
		"mulps   %%xmm0, %%xmm2          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"pshufd   $0x39, %%xmm7, %%xmm6  \n\t"
		"mulps   %%xmm0, %%xmm7          \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addps   %%xmm2, %%xmm8          \n\t"
		"movaps       48(%%rbx), %%xmm2  \n\t"
		"addps   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm6, %%xmm4  \n\t"
		"mulps   %%xmm0, %%xmm6          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm7, %%xmm9          \n\t"
		"addps   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
		"mulps   %%xmm0, %%xmm4          \n\t"
		"movaps       48(%%rax), %%xmm0  \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"movaps       48(%%rcx), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t" // iteration 3
		"addps   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm2, %%xmm7  \n\t"
		"mulps   %%xmm0, %%xmm2          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addq       $64, %%rax           \n\t" // A0 += 16
		"                                \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"pshufd   $0x39, %%xmm7, %%xmm6  \n\t"
		"mulps   %%xmm0, %%xmm7          \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addq       $64, %%rbx           \n\t" // B += 16
		"                                \n\t"
		"addps   %%xmm2, %%xmm8          \n\t"
		"movaps         (%%rbx), %%xmm2  \n\t"
		"addps   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm6, %%xmm4  \n\t"
		"mulps   %%xmm0, %%xmm6          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addq       $64, %%rcx           \n\t" // A1 += 16
		"                                \n\t"
		"addps   %%xmm7, %%xmm9          \n\t"
		"decl    %%esi                   \n\t" // i -= 1;
		"addps   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
		"mulps   %%xmm0, %%xmm4          \n\t"
		"movaps         (%%rax), %%xmm0  \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"movaps         (%%rcx), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .SLOOPKITER              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".SCONSIDKLEFT:                  \n\t"
		"                                \n\t"
		"movl      %1, %%esi             \n\t" // i = k_left;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .SPOSTACCUM              \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".SLOOPKLEFT:                    \n\t" // EDGE LOOP
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t" // iteration 0
		"addps   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm2, %%xmm7  \n\t"
		"mulps   %%xmm0, %%xmm2          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"pshufd   $0x39, %%xmm7, %%xmm6  \n\t"
		"mulps   %%xmm0, %%xmm7          \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addps   %%xmm2, %%xmm8          \n\t"
		"movaps       16(%%rbx), %%xmm2  \n\t"
		"addps   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x39, %%xmm6, %%xmm4  \n\t"
		"mulps   %%xmm0, %%xmm6          \n\t"
		"mulps   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addps   %%xmm7, %%xmm9          \n\t"
		"addps   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
		"mulps   %%xmm0, %%xmm4          \n\t"
		"movaps       16(%%rax), %%xmm0  \n\t"
		"mulps   %%xmm1, %%xmm5          \n\t"
		"movaps       16(%%rcx), %%xmm1  \n\t"
		"                                \n\t"
		"addq          $16, %%rax        \n\t" // A0 += 4
		"addq          $16, %%rcx        \n\t" // A1 += 4
		"addq          $16, %%rbx        \n\t" // B += 4
		"                                \n\t"
		"                                \n\t"
		"decl    %%esi                   \n\t" // i -= 1;
		"jne    .SLOOPKLEFT              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".SPOSTACCUM:                    \n\t"
		"                                \n\t"
		"addps   %%xmm6, %%xmm10         \n\t"
		"addps   %%xmm3, %%xmm14         \n\t"
		"addps   %%xmm4, %%xmm11         \n\t"
		"addps   %%xmm5, %%xmm15         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movaps  %%xmm9, %%xmm4          \n\t"
		"shufps   $0xd8, %%xmm8,  %%xmm9 \n\t"
		"shufps   $0xd8, %%xmm11, %%xmm8 \n\t"
		"shufps   $0xd8, %%xmm10, %%xmm11\n\t"
		"shufps   $0xd8, %%xmm4,  %%xmm10\n\t"
		"                                \n\t"
		"movaps  %%xmm8, %%xmm4          \n\t"
		"shufps   $0xd8, %%xmm10, %%xmm8 \n\t"
		"shufps   $0xd8, %%xmm4, %%xmm10 \n\t"
		"movaps  %%xmm9, %%xmm5          \n\t"
		"shufps   $0xd8, %%xmm11, %%xmm9 \n\t"
		"shufps   $0xd8, %%xmm5, %%xmm11 \n\t"
		"                                \n\t"
		"movaps  %%xmm13, %%xmm4         \n\t"
		"shufps   $0xd8, %%xmm12, %%xmm13\n\t"
		"shufps   $0xd8, %%xmm15, %%xmm12\n\t"
		"shufps   $0xd8, %%xmm14, %%xmm15\n\t"
		"shufps   $0xd8, %%xmm4,  %%xmm14\n\t"
		"                                \n\t"
		"movaps  %%xmm12, %%xmm4         \n\t"
		"shufps   $0xd8, %%xmm14, %%xmm12\n\t"
		"shufps   $0xd8, %%xmm4, %%xmm14 \n\t"
		"movaps  %%xmm13, %%xmm5         \n\t"
		"shufps   $0xd8, %%xmm15, %%xmm13\n\t"
		"shufps   $0xd8, %%xmm5, %%xmm15 \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq   %5, %%rax                \n\t" // load address of C0
		"movq   %6, %%rbx                \n\t" // load address of C1
		"                                \n\t"
		"                                \n\t"
		"movl   %7, %%ecx                \n\t" // alg
		"testl  %%ecx, %%ecx             \n\t" // check alg
		"je     .S0                      \n\t" // if alg==0, jump
		"                                \n\t"
		"cmpl	$1, %%ecx                \n\t"
		"                                \n\t"
		"movaps  (%%rax),   %%xmm0       \n\t" // load C0
		"movaps  16(%%rax), %%xmm1       \n\t"
		"movaps  32(%%rax), %%xmm2       \n\t"
		"movaps  48(%%rax), %%xmm3       \n\t"
		"movaps  (%%rbx),   %%xmm4       \n\t" // load C0
		"movaps  16(%%rbx), %%xmm5       \n\t"
		"movaps  32(%%rbx), %%xmm6       \n\t"
		"movaps  48(%%rbx), %%xmm7       \n\t"
		"                                \n\t"
		"je     .S1                      \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"subps  %%xmm8,  %%xmm0           \n\t"
		"subps  %%xmm9,  %%xmm1           \n\t"
		"subps  %%xmm10, %%xmm2           \n\t"
		"subps  %%xmm11, %%xmm3           \n\t"
		"subps  %%xmm12, %%xmm4           \n\t"
		"subps  %%xmm13, %%xmm5           \n\t"
		"subps  %%xmm14, %%xmm6           \n\t"
		"subps  %%xmm15, %%xmm7           \n\t"
		"                                \n\t"
		"movaps  %%xmm0, (%%rax)         \n\t"
		"movaps  %%xmm1, 16(%%rax)       \n\t"
		"movaps  %%xmm2, 32(%%rax)       \n\t"
		"movaps  %%xmm3, 48(%%rax)       \n\t"
		"movaps  %%xmm4, (%%rbx)         \n\t"
		"movaps  %%xmm5, 16(%%rbx)       \n\t"
		"movaps  %%xmm6, 32(%%rbx)       \n\t"
		"movaps  %%xmm7, 48(%%rbx)       \n\t"
		"                                \n\t"
		"jmp    .SDONE                   \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".S1:                            \n\t" // alg==1
		"                                \n\t"
		"addps  %%xmm0, %%xmm8           \n\t"
		"addps  %%xmm1, %%xmm9           \n\t"
		"addps  %%xmm2, %%xmm10          \n\t"
		"addps  %%xmm3, %%xmm11          \n\t"
		"addps  %%xmm4, %%xmm12          \n\t"
		"addps  %%xmm5, %%xmm13          \n\t"
		"addps  %%xmm6, %%xmm14          \n\t"
		"addps  %%xmm7, %%xmm15          \n\t"
		"                                \n\t"
		".S0:                            \n\t" // alg==0
		"                                \n\t"
		"movaps	%%xmm8,  (%%rax)          \n\t"
		"movaps	%%xmm9,  16(%%rax)        \n\t"
		"movaps	%%xmm10, 32(%%rax)        \n\t"
		"movaps	%%xmm11, 48(%%rax)        \n\t"
		"movaps	%%xmm12, (%%rbx)          \n\t"
		"movaps	%%xmm13, 16(%%rbx)        \n\t"
		"movaps	%%xmm14, 32(%%rbx)        \n\t"
		"movaps	%%xmm15, 48(%%rbx)        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".SDONE:                         \n\t"
		"                                \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter),		// %0
		  "m" (k_left),		// %1
		  "m" (A0),			// %2
		  "m" (A1),			// %3
		  "m" (B),			// %4
		  "m" (C0),			// %5
		  "m" (C1),			// %6
		  "m" (alg)			// %7
		: // register clobber list
		  "rax", "rbx", "rcx", "rsi", //"rdx", //"rdi", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);
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
	
/*	const int ldc = 8;*/

	int k;
	
	__m128
		c_03_0, c_03_1, c_03_2, c_03_3,
		a_03,
		b_0, b_1, b_2; 
	
	c_03_0 = _mm_setzero_ps();
	c_03_1 = _mm_setzero_ps();
	c_03_2 = _mm_setzero_ps();
	c_03_3 = _mm_setzero_ps();

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
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );



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
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );


		
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
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );



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
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );


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
	
		b_1 = b_0;
		b_1 = _mm_shuffle_ps( b_1, b_1, 255 );
		b_1 = _mm_mul_ps( a_03, b_1 );
		c_03_3 = _mm_add_ps( c_03_3, b_1 );
		
		A += 4;
		B += 4;
		
		}

	__m128
		d_03_0, d_03_1, d_03_2, d_03_3;

	if(alg==0)
		{
		_mm_store_ps( &C[0+ldc*0], c_03_0 );
		_mm_store_ps( &C[0+ldc*1], c_03_1 );
		_mm_store_ps( &C[0+ldc*2], c_03_2 );
		_mm_store_ps( &C[0+ldc*3], c_03_3 );
		}
	else if(alg==1)
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		d_03_3 = _mm_load_ps( &C[0+ldc*3] );
		
		d_03_0 = _mm_add_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_add_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_add_ps( d_03_2, c_03_2 );
		d_03_3 = _mm_add_ps( d_03_3, c_03_3 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		_mm_store_ps( &C[0+ldc*3], d_03_3 );
		}
	else
		{
		d_03_0 = _mm_load_ps( &C[0+ldc*0] );
		d_03_1 = _mm_load_ps( &C[0+ldc*1] );
		d_03_2 = _mm_load_ps( &C[0+ldc*2] );
		d_03_3 = _mm_load_ps( &C[0+ldc*3] );
		
		d_03_0 = _mm_sub_ps( d_03_0, c_03_0 );
		d_03_1 = _mm_sub_ps( d_03_1, c_03_1 );
		d_03_2 = _mm_sub_ps( d_03_2, c_03_2 );
		d_03_3 = _mm_sub_ps( d_03_3, c_03_3 );

		_mm_store_ps( &C[0+ldc*0], d_03_0 );
		_mm_store_ps( &C[0+ldc*1], d_03_1 );
		_mm_store_ps( &C[0+ldc*2], d_03_2 );
		_mm_store_ps( &C[0+ldc*3], d_03_3 );
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
		b_0, b_1, b_2; 
	
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
		b_0, b_1, b_2; 
	
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
		b_0, b_1, b_2; 
	
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

