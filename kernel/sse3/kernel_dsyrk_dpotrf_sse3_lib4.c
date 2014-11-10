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

#include <math.h>

#include "../../include/block_size.h"



#include "../../include/block_size.h"



#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
/*#include <smmintrin.h>  // SSE4*/
//#include <immintrin.h>  // AVX



// normal-transposed, 4x4 with data packed in 4
void kernel_dsyrk_dpotrf_nt_4x4_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc_dummy, double *fact)
	{
	
	int ki_add = kadd/4;
	int kl_add = kadd%4;
	int ki_sub = ksub/4;

	const int bs = D_MR;//4;
	const int d_ncl = D_NCL;//2;

	long long dA = bs*((d_ncl-kadd%d_ncl)%d_ncl)*sizeof(double);

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movq          %3, %%rax         \n\t" // load address of A
		"movq          %4, %%rbx         \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"movaps        0(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps       16(%%rax), %%xmm1  \n\t" // of a and b.
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
/*		"movaps    %%xmm3, %%xmm10       \n\t"*/
/*		"movaps    %%xmm3, %%xmm11       \n\t"*/
		"movaps    %%xmm3, %%xmm12       \n\t"
		"movaps    %%xmm3, %%xmm13       \n\t"
		"movaps    %%xmm3, %%xmm14       \n\t"
		"movaps    %%xmm3, %%xmm15       \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl      %0, %%esi             \n\t" // i = k_iter;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .DCONSIDERADD            \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPADD:                      \n\t" // MAIN LOOP add
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t" // iteration 0*/
		"movaps       16(%%rbx), %%xmm6  \n\t"
		"addpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm8          \n\t"
		"movaps       32(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm9          \n\t"
		"addpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       32(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       48(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t" // iteration 1*/
		"movaps       48(%%rbx), %%xmm6  \n\t"
		"addpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm8          \n\t"
		"movaps       64(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm9          \n\t"
		"addpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       64(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       80(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t" // iteration 2*/
		"movaps       80(%%rbx), %%xmm6  \n\t"
		"addpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm8          \n\t"
		"movaps       96(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm9          \n\t"
		"addpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       96(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps      112(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t" // iteration 3*/
		"movaps      112(%%rbx), %%xmm6  \n\t"
		"addpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addq      $128, %%rax           \n\t" // A0 += 16
		"                                \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addq      $128, %%rbx           \n\t" // B += 16
		"                                \n\t"
		"addpd   %%xmm2, %%xmm8          \n\t"
		"movaps         (%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm9          \n\t"
		"decl    %%esi                   \n\t" // i -= 1;
		"addpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps         (%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPADD                \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERADD:                  \n\t"
		"                                \n\t"
		"movl      %1, %%esi             \n\t" // i = k_left;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .DCONSIDERSUB            \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPLEFT:                     \n\t" // EDGE LOOP
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t" // iteration 0*/
		"movaps       16(%%rbx), %%xmm6  \n\t"
		"addpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm8          \n\t"
		"movaps       32(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm9          \n\t"
		"addpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       32(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       48(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"addq          $32, %%rax        \n\t" // A += 4
		"addq          $32, %%rbx        \n\t" // B += 4
		"                                \n\t"
		"                                \n\t"
		"decl    %%esi                   \n\t" // i -= 1;
		"jne    .DLOOPLEFT               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERSUB:                  \n\t" // padd
		"                                \n\t"
		"                                \n\t"
/*		"addpd   %%xmm6, %%xmm10         \n\t"*/
		"addpd   %%xmm3, %%xmm14         \n\t"
/*		"addpd   %%xmm4, %%xmm11         \n\t"*/
		"addpd   %%xmm5, %%xmm15         \n\t"
		"                                \n\t"
		"xorpd     %%xmm3,  %%xmm3       \n\t"
		"movaps    %%xmm3,  %%xmm4       \n\t"
		"movaps    %%xmm3,  %%xmm5       \n\t"
		"movaps    %%xmm3,  %%xmm6       \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %2, %%ecx                \n\t"
		"cmpl	$0, %%ecx                \n\t"
		"jle    .DPOSTACC                \n\t"
		"                                \n\t"
		"movq   %8, %%rcx                \n\t"
		"cmpq	$0, %%rcx                \n\t"
		"jle    .DPRELOOPSUB             \n\t"
		"                                \n\t"
		"movl   %0, %%ecx                \n\t"
		"movl   %1, %%edx                \n\t"
		"addl   %%edx, %%ecx              \n\t"
		"cmpl	$0, %%ecx                \n\t"
		"jle    .DPRELOOPSUB             \n\t"
		"                                \n\t"
		"movq   %8, %%rcx                \n\t"
		"addq   %%rcx, %%rax             \n\t" 
		"addq   %%rcx, %%rbx             \n\t" 
		"                                \n\t"
		"movaps        0(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps       16(%%rax), %%xmm1  \n\t" // of a and b.
		"movaps        0(%%rbx), %%xmm2  \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPRELOOPSUB:                   \n\t" // 
		"                                \n\t"
		"movl   %2, %%esi                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPSUB:                      \n\t" // main loop 2
		"                                \n\t"
/*		"subpd   %%xmm6, %%xmm10         \n\t" // iteration 0*/
		"movaps       16(%%rbx), %%xmm6  \n\t"
		"subpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"subpd   %%xmm4, %%xmm11         \n\t"*/
		"subpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"subpd   %%xmm2, %%xmm8          \n\t"
		"movaps       32(%%rbx), %%xmm2  \n\t"
		"subpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"subpd   %%xmm7, %%xmm9          \n\t"
		"subpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       32(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       48(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"subpd   %%xmm6, %%xmm10         \n\t" // iteration 1*/
		"movaps       48(%%rbx), %%xmm6  \n\t"
		"subpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"subpd   %%xmm4, %%xmm11         \n\t"*/
		"subpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"subpd   %%xmm2, %%xmm8          \n\t"
		"movaps       64(%%rbx), %%xmm2  \n\t"
		"subpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"subpd   %%xmm7, %%xmm9          \n\t"
		"subpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       64(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       80(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"subpd   %%xmm6, %%xmm10         \n\t" // iteration 2*/
		"movaps       80(%%rbx), %%xmm6  \n\t"
		"subpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
/*		"subpd   %%xmm4, %%xmm11         \n\t"*/
		"subpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"subpd   %%xmm2, %%xmm8          \n\t"
		"movaps       96(%%rbx), %%xmm2  \n\t"
		"subpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"subpd   %%xmm7, %%xmm9          \n\t"
		"subpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps       96(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps      112(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"subpd   %%xmm6, %%xmm10         \n\t" // iteration 3*/
		"movaps      112(%%rbx), %%xmm6  \n\t"
		"subpd   %%xmm3, %%xmm14         \n\t"
		"movaps  %%xmm2, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"addq      $128, %%rax           \n\t" // A0 += 16
		"                                \n\t"
/*		"subpd   %%xmm4, %%xmm11         \n\t"*/
		"subpd   %%xmm5, %%xmm15         \n\t"
		"movaps  %%xmm7, %%xmm5          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"                                \n\t"
		"addq      $128, %%rbx           \n\t" // B += 16
		"                                \n\t"
		"subpd   %%xmm2, %%xmm8          \n\t"
		"movaps         (%%rbx), %%xmm2  \n\t"
		"subpd   %%xmm3, %%xmm12         \n\t"
		"movaps  %%xmm6, %%xmm3          \n\t"
		"pshufd   $0x4e, %%xmm6, %%xmm4  \n\t"
/*		"mulpd   %%xmm0, %%xmm6          \n\t"*/
		"mulpd   %%xmm1, %%xmm3          \n\t"
		"                                \n\t"
		"                                \n\t"
		"subpd   %%xmm7, %%xmm9          \n\t"
		"decl    %%esi                   \n\t" // i -= 1;
		"subpd   %%xmm5, %%xmm13         \n\t"
		"movaps  %%xmm4, %%xmm5          \n\t"
/*		"mulpd   %%xmm0, %%xmm4          \n\t"*/
		"movaps         (%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm5          \n\t"
		"movaps       16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPSUB                \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
/*		"subpd   %%xmm6, %%xmm10         \n\t"*/
		"subpd   %%xmm3, %%xmm14         \n\t"
/*		"subpd   %%xmm4, %%xmm11         \n\t"*/
		"subpd   %%xmm5, %%xmm15         \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACC:                      \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movaps   %%xmm8,  %%xmm0        \n\t"
		"movsd    %%xmm9,  %%xmm8        \n\t"
		"movsd    %%xmm0,  %%xmm9        \n\t"
		"                                \n\t"
/*		"movaps  %%xmm10,  %%xmm0        \n\t"*/
/*		"movsd   %%xmm11, %%xmm10        \n\t"*/
/*		"movsd    %%xmm0, %%xmm11        \n\t"*/
		"                                \n\t"
		"movaps  %%xmm12,  %%xmm0        \n\t"
		"movsd   %%xmm13, %%xmm12        \n\t"
		"movsd    %%xmm0, %%xmm13        \n\t"
		"                                \n\t"
		"movaps  %%xmm14,  %%xmm0        \n\t"
		"movsd   %%xmm15, %%xmm14        \n\t"
		"movsd    %%xmm0, %%xmm15        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq   %5, %%rax                \n\t" // load address of C
		"                                \n\t"
		"                                \n\t"
		"movaps  (%%rax),   %%xmm0       \n\t" // load C0
		"movaps  32(%%rax), %%xmm1       \n\t"
/*		"movaps  64(%%rax), %%xmm2       \n\t"*/
/*		"movaps  96(%%rax), %%xmm3       \n\t"*/
		"movaps  16(%%rax), %%xmm4       \n\t" // load C0
		"movaps  48(%%rax), %%xmm5       \n\t"
		"movaps  80(%%rax), %%xmm6       \n\t"
		"movaps 112(%%rax), %%xmm7       \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd  %%xmm0, %%xmm9           \n\t"
		"addpd  %%xmm1, %%xmm8           \n\t"
/*		"addpd  %%xmm2, %%xmm11          \n\t"*/
/*		"addpd  %%xmm3, %%xmm10          \n\t"*/
		"addpd  %%xmm4, %%xmm13          \n\t"
		"addpd  %%xmm5, %%xmm12          \n\t"
		"addpd  %%xmm6, %%xmm15          \n\t"
		"addpd  %%xmm7, %%xmm14          \n\t"
		"                                \n\t"
		"                                \n\t" //  9  8 11 10
		"                                \n\t" // 13 12 15 14
		"                                \n\t"
		"movq   %7, %%rax                \n\t" // load address of fact
		"movq   %6, %%rbx                \n\t" // load address of D
		"                                \n\t"
		"                                \n\t"
		"movsd   .DTHR(%%rip), %%xmm7    \n\t" // 1e-15
		"movsd   .DONE(%%rip), %%xmm6    \n\t" // 1.0
		"                                \n\t"
		"                                \n\t"
		".DA00:                          \n\t"
		"movsd   %%xmm9, %%xmm0          \n\t" // a_00
		"ucomisd %%xmm7, %%xmm0          \n\t"
		"jbe     .DZERO0                 \n\t"
		"sqrtsd  %%xmm0, %%xmm0          \n\t"
		"movsd   %%xmm6, %%xmm5          \n\t" // 1.0
		"divsd   %%xmm0, %%xmm5          \n\t" // 1.0/a_00
		"movsd   %%xmm5, (%%rax)         \n\t" // fact[0]
		"movddup %%xmm5, %%xmm5          \n\t"
		"mulpd   %%xmm5, %%xmm9          \n\t"
		"mulpd   %%xmm5, %%xmm13         \n\t"
		"movaps	 %%xmm9,  (%%rbx)        \n\t"
		"movaps	 %%xmm13, 16(%%rbx)      \n\t"
		"                                \n\t"
		".DA11:                          \n\t"
		"movaps  %%xmm9, %%xmm0          \n\t"
		"shufpd  $3, %%xmm0, %%xmm0      \n\t" // a_10
		"movsd   %%xmm0, 8(%%rax)        \n\t" // fact[1]
		"movaps  %%xmm0,  %%xmm1         \n\t"
		"mulpd   %%xmm9, %%xmm0          \n\t"
		"mulpd   %%xmm13, %%xmm1         \n\t"
		"subpd   %%xmm0, %%xmm8          \n\t"
		"subpd   %%xmm1, %%xmm12         \n\t"
		"movaps  %%xmm8, %%xmm0          \n\t" // [a_01 a_11]
		"shufpd  $1, %%xmm0, %%xmm0      \n\t" // a_11
		"ucomisd %%xmm7, %%xmm0          \n\t"
		"jbe     .DZERO1                 \n\t"
		"sqrtsd  %%xmm0, %%xmm0          \n\t"
		"movsd   %%xmm6, %%xmm5          \n\t" // 1.0
		"divsd   %%xmm0, %%xmm5          \n\t" // 1.0/a_11
		"movsd   %%xmm5, 16(%%rax)       \n\t" // fact[2]
		"movddup %%xmm5, %%xmm5          \n\t"
		"mulpd   %%xmm5, %%xmm8          \n\t"
		"mulpd   %%xmm5, %%xmm12         \n\t"
		"movhpd	 %%xmm8,  40(%%rbx)      \n\t"
		"movaps	 %%xmm12, 48(%%rbx)      \n\t"
		"                                \n\t"
		".DA22:                          \n\t"
		"movddup %%xmm13, %%xmm0         \n\t" // a_20
		"movddup %%xmm12, %%xmm1         \n\t" // a_21
		"movsd   %%xmm0, 24(%%rax)       \n\t" // fact[3]
		"movsd   %%xmm1, 32(%%rax)       \n\t" // fact[4]
		"mulpd   %%xmm13, %%xmm0         \n\t"
		"mulpd   %%xmm12, %%xmm1         \n\t"
		"subpd   %%xmm0, %%xmm15         \n\t"
		"subpd   %%xmm1, %%xmm15         \n\t"
		"movsd   %%xmm15, %%xmm0         \n\t" // a_22
		"ucomisd %%xmm7, %%xmm0          \n\t"
		"jbe     .DZERO2                 \n\t"
		"sqrtsd  %%xmm0, %%xmm0          \n\t"
		"movsd   %%xmm6, %%xmm5          \n\t" // 1.0
		"divsd   %%xmm0, %%xmm5          \n\t" // 1.0/a_11
		"movsd   %%xmm5, 40(%%rax)       \n\t" // fact[5]
		"movddup %%xmm5, %%xmm5          \n\t"
		"mulpd   %%xmm5, %%xmm15         \n\t"
		"movaps	 %%xmm15, 80(%%rbx)      \n\t"
		"                                \n\t"
		".DA33:                          \n\t"
		"movaps  %%xmm13, %%xmm0         \n\t"
		"movaps  %%xmm12, %%xmm1         \n\t"
		"movaps  %%xmm15, %%xmm2         \n\t"
		"shufpd  $3, %%xmm0, %%xmm0      \n\t" // a_30
		"shufpd  $3, %%xmm1, %%xmm1      \n\t" // a_31
		"shufpd  $3, %%xmm2, %%xmm2      \n\t" // a_32
		"movsd   %%xmm0, 48(%%rax)       \n\t" // fact[6]
		"movsd   %%xmm1, 56(%%rax)       \n\t" // fact[7]
		"movsd   %%xmm2, 64(%%rax)       \n\t" // fact[8]
		"mulpd   %%xmm13, %%xmm0         \n\t"
		"mulpd   %%xmm12, %%xmm1         \n\t"
		"mulpd   %%xmm15, %%xmm2         \n\t"
		"subpd   %%xmm0, %%xmm14         \n\t"
		"subpd   %%xmm1, %%xmm14         \n\t"
		"subpd   %%xmm2, %%xmm14         \n\t"
		"movaps  %%xmm14, %%xmm0         \n\t" // [a_23 a_33]
		"shufpd  $1, %%xmm0, %%xmm0      \n\t" // a_33
		"ucomisd %%xmm7, %%xmm0          \n\t"
		"jbe     .DZERO3                 \n\t"
		"sqrtsd  %%xmm0, %%xmm0          \n\t"
		"movsd   %%xmm0, 120(%%rbx)      \n\t"
		"divsd   %%xmm0, %%xmm6          \n\t" // 1.0/a_11
		"movsd   %%xmm6, 72(%%rax)       \n\t" // fact[9]
		"                                \n\t"
		"jmp    .DEND                    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DZERO0:                        \n\t"
		"xorpd  %%xmm9,  %%xmm9          \n\t"
		"movaps %%xmm9,  %%xmm13         \n\t"
		"movaps %%xmm9,  (%%rbx)         \n\t"
		"movaps %%xmm13, 16(%%rbx)       \n\t"
		"movsd  %%xmm9,  (%%rax)         \n\t" // fact[0]
		"jmp    .DA11                    \n\t"
		"                                \n\t"
		".DZERO1:                        \n\t"
		"xorpd  %%xmm8,  %%xmm8          \n\t"
		"movaps %%xmm8,  %%xmm12         \n\t"
		"movsd  %%xmm8,  40(%%rbx)       \n\t"
		"movaps %%xmm12, 48(%%rbx)       \n\t"
		"movsd  %%xmm8,  16(%%rax)       \n\t" // fact[2]
		"jmp    .DA22                    \n\t"
		"                                \n\t"
		".DZERO2:                        \n\t"
		"xorpd  %%xmm15, %%xmm15         \n\t"
		"movaps %%xmm15, 80(%%rbx)       \n\t"
		"movsd  %%xmm15, 40(%%rax)       \n\t" // fact[5]
		"jmp    .DA33                    \n\t"
		"                                \n\t"
		".DZERO3:                        \n\t"
		"xorpd  %%xmm14, %%xmm14         \n\t"
		"movsd  %%xmm14, 120(%%rbx)      \n\t"
		"movsd  %%xmm14, 72(%%rax)       \n\t" // fact[2]
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"jmp    .DEND                    \n\t"
		".align 16                       \n\t"
		".DONE:                          \n\t"
		".long  0                        \n\t"
		".long  1072693248               \n\t"
		".long  0                        \n\t"
		".long  0                        \n\t"
		".align 16                       \n\t"
		".DTHR:                          \n\t"
		".long  2665960982               \n\t"
		".long  1020396463               \n\t"
		".long  0                        \n\t"
		".long  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		".DEND:                          \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "m" (ki_add),		// %0
		  "m" (kl_add),		// %1
		  "m" (ki_sub),		// %2
		  "m" (A),			// %3
		  "m" (B),			// %4
		  "m" (C),			// %5
		  "m" (D),			// %6
		  "m" (fact),		// %7
		  "m" (dA)			// %8
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", //"rdx", //"rdi", "r8", "r9", "r10", "r11",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);
}



void kernel_dsyrk_dpotrf_nt_4x4_lib4_old(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc, double *fact)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;
	const int lda = bs;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, 
		c_10=0, c_11=0, 
		c_20=0, c_21=0, c_22=0, 
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		b_2 = B[2+lda*0];
		b_3 = B[3+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		a_2 = A[2+lda*1];
		a_3 = A[3+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		b_2 = B[2+lda*1];
		b_3 = B[3+lda*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		a_2 = A[2+lda*2];
		a_3 = A[3+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		b_2 = B[2+lda*2];
		b_3 = B[3+lda*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		a_2 = A[2+lda*3];
		a_3 = A[3+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		b_2 = B[2+lda*3];
		b_3 = B[3+lda*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		b_2 = B[2+lda*0];
		b_3 = B[3+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_33 += a_3 * b_3;


		A += 4;
		B += 4;

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		b_2 = B[2+lda*0];
		b_3 = B[3+lda*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_33 -= a_3 * b_3;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		a_2 = A[2+lda*1];
		a_3 = A[3+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		b_2 = B[2+lda*1];
		b_3 = B[3+lda*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_33 -= a_3 * b_3;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		a_2 = A[2+lda*2];
		a_3 = A[3+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		b_2 = B[2+lda*2];
		b_3 = B[3+lda*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_33 -= a_3 * b_3;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		a_2 = A[2+lda*3];
		a_3 = A[3+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		b_2 = B[2+lda*3];
		b_3 = B[3+lda*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;

		c_22 -= a_2 * b_2;
		c_32 -= a_3 * b_2;

		c_33 -= a_3 * b_3;
		
		
		A += 16;
		B += 16;

		}

	c_00 += C[0+ldc*0];
	c_10 += C[1+ldc*0];
	c_20 += C[2+ldc*0];
	c_30 += C[3+ldc*0];

	c_11 += C[1+ldc*1];
	c_21 += C[2+ldc*1];
	c_31 += C[3+ldc*1];

	c_22 += C[2+ldc*2];
	c_32 += C[3+ldc*2];

	c_33 += C[3+ldc*3];
	
	// dpotrf
	
	// first column
	if(c_00 > 1e-15)
		{
		c_00 = sqrt(c_00);
		D[0+ldc*0] = c_00;
		c_00 = 1.0/c_00;
		c_10 *= c_00;
		c_20 *= c_00;
		c_30 *= c_00;
		}
	else
		{
		c_00 = 0.0;
		D[0+ldc*0] = c_00;
		c_10 = 0.0;
		c_20 = 0.0;
		c_30 = 0.0;
		}
	D[1+ldc*0] = c_10;
	D[2+ldc*0] = c_20;
	D[3+ldc*0] = c_30;
	
	// second column
	c_11 -= c_10*c_10;
	if(c_11 > 1e-15)
		{
		c_11 = sqrt(c_11);
		D[1+ldc*1] = c_11;
		c_11 = 1.0/c_11;
		c_21 -= c_20*c_10;
		c_31 -= c_30*c_10;
		c_21 *= c_11;
		c_31 *= c_11;
		}
	else
		{
		c_11 = 0.0;
		D[1+ldc*1] = c_11;
		c_21 = 0.0;
		c_31 = 0.0;
		}
	D[2+ldc*1] = c_21;
	D[3+ldc*1] = c_31;

	// third column
	c_22 -= c_20*c_20;
	c_22 -= c_21*c_21;
	if(c_22 > 1e-15)
		{
		c_22 = sqrt(c_22);
		D[2+ldc*2] = c_22;
		c_22 = 1.0/c_22;
		c_32 -= c_30*c_20;
		c_32 -= c_31*c_21;
		c_32 *= c_22;
		}
	else
		{
		c_22 = 0.0;
		D[2+ldc*2] = c_22;
		c_32 = 0.0;
		}
	D[3+ldc*2] = c_32;

	// fourth column
	c_33 -= c_30*c_30;
	c_33 -= c_31*c_31;
	c_33 -= c_32*c_32;
	if(c_33 > 1e-15)
		{
		c_33 = sqrt(c_33);
		D[3+ldc*3] = c_33;
		c_33 = 1.0/c_33;
		}
	else
		{
		c_33 = 0.0;
		D[3+ldc*3] = c_33;
		}
	
	// save factorized matrix with reciprocal of diagonal
	fact[0] = c_00;
	fact[1] = c_10;
	fact[3] = c_20;
	fact[6] = c_30;
	fact[2] = c_11;
	fact[4] = c_21;
	fact[7] = c_31;
	fact[5] = c_22;
	fact[8] = c_32;
	fact[9] = c_33;

	}



void kernel_dsyrk_dpotrf_nt_4x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc, double *fact)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;
	const int lda = bs;

	int k;

	double
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, 
		c_10=0, c_11=0, 
		c_20=0, c_21=0,  
		c_30=0, c_31=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		a_2 = A[2+lda*1];
		a_3 = A[3+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		a_2 = A[2+lda*2];
		a_3 = A[3+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		a_2 = A[2+lda*3];
		a_3 = A[3+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;
		c_20 += a_2 * b_0;
		c_30 += a_3 * b_0;

		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		A += 4;
		B += 4;

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		a_2 = A[2+lda*0];
		a_3 = A[3+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		a_2 = A[2+lda*1];
		a_3 = A[3+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		a_2 = A[2+lda*2];
		a_3 = A[3+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		a_2 = A[2+lda*3];
		a_3 = A[3+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;
		c_20 -= a_2 * b_0;
		c_30 -= a_3 * b_0;

		c_11 -= a_1 * b_1;
		c_21 -= a_2 * b_1;
		c_31 -= a_3 * b_1;
		
		
		A += 16;
		B += 16;

		}

	c_00 += C[0+ldc*0];
	c_10 += C[1+ldc*0];
	c_20 += C[2+ldc*0];
	c_30 += C[3+ldc*0];

	c_11 += C[1+ldc*1];
	c_21 += C[2+ldc*1];
	c_31 += C[3+ldc*1];
	
	// dpotrf
	
	// first column
	if(c_00 > 1e-15)
		{
		c_00 = sqrt(c_00);
		D[0+ldc*0] = c_00;
		c_00 = 1.0/c_00;
		c_10 *= c_00;
		c_20 *= c_00;
		c_30 *= c_00;
		}
	else
		{
		c_00 = 0.0;
		D[0+ldc*0] = c_00;
		c_10 = 0.0;
		c_20 = 0.0;
		c_30 = 0.0;
		}
	D[1+ldc*0] = c_10;
	D[2+ldc*0] = c_20;
	D[3+ldc*0] = c_30;
	
	// second column
	c_11 -= c_10*c_10;
	if(c_11 > 1e-15)
		{
		c_11 = sqrt(c_11);
		D[1+ldc*1] = c_11;
		c_11 = 1.0/c_11;
		c_21 -= c_20*c_10;
		c_31 -= c_30*c_10;
		c_21 *= c_11;
		c_31 *= c_11;
		}
	else
		{
		c_11 = 0.0;
		D[1+ldc*1] = c_11;
		c_21 = 0.0;
		c_31 = 0.0;
		}
	D[2+ldc*1] = c_21;
	D[3+ldc*1] = c_31;

	// save factorized matrix with reciprocal of diagonal
	fact[0] = c_00;
	fact[1] = c_10;
	fact[3] = c_20;
	fact[6] = c_30;
	fact[2] = c_11;
	fact[4] = c_21;
	fact[7] = c_31;
/*	fact[5] = c_22;*/
/*	fact[8] = c_32;*/
/*	fact[9] = c_33;*/

	}



void kernel_dsyrk_dpotrf_nt_2x2_lib4(int kadd, int ksub, double *A, double *B, double *C, double *D, int ldc, double *fact)
	{

	const int bs = 4;
	const int d_ncl = D_NCL;
	const int lda = bs;

	int k;

	double
		a_0, a_1,
		b_0, b_1,
		c_00=0, 
		c_10=0, c_11=0;
		
	for(k=0; k<kadd-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kadd; k++)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_11 += a_1 * b_1;


		A += 4;
		B += 4;

		}

	if(ksub>0)
		{
		if(kadd>0)
			{
			A += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			B += bs*((d_ncl-kadd%d_ncl)%d_ncl);
			}
		}

	for(k=0; k<ksub-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_11 -= a_1 * b_1;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_11 -= a_1 * b_1;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_11 -= a_1 * b_1;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		
		c_00 -= a_0 * b_0;
		c_10 -= a_1 * b_0;

		c_11 -= a_1 * b_1;
		
		
		A += 16;
		B += 16;

		}

	c_00 += C[0+ldc*0];
	c_10 += C[1+ldc*0];

	c_11 += C[1+ldc*1];
	
	// dpotrf
	
	// first column
	if(c_00 > 1e-15)
		{
		c_00 = sqrt(c_00);
		D[0+ldc*0] = c_00;
		c_00 = 1.0/c_00;
		c_10 *= c_00;
		}
	else
		{
		c_00 = 0.0;
		D[0+ldc*0] = c_00;
		c_10 = 0.0;
		}
	D[1+ldc*0] = c_10;
	
	// second column
	c_11 -= c_10*c_10;
	if(c_11 > 1e-15)
		{
		c_11 = sqrt(c_11);
		D[1+ldc*1] = c_11;
		}
	else
		{
		c_11 = 0.0;
		D[1+ldc*1] = c_11;
		}

	// save factorized matrix with reciprocal of diagonal
	fact[0] = c_00;
	fact[1] = c_10;
/*	fact[3] = c_20;*/
/*	fact[6] = c_30;*/
	fact[2] = c_11;
/*	fact[4] = c_21;*/
/*	fact[7] = c_31;*/
/*	fact[5] = c_22;*/
/*	fact[8] = c_32;*/
/*	fact[9] = c_33;*/

	}




