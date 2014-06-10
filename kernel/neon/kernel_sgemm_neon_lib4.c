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



// normal-transposed, 8x4 with data packed in 4
void kernel_sgemm_pp_nt_8x4_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, float *D0, float *D1, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
		
	__builtin_prefetch( A0 );
	__builtin_prefetch( A1 );
	__builtin_prefetch( B  );
#if defined(TARGET_CORTEX_A9)
	__builtin_prefetch( A0+8 );
	__builtin_prefetch( A1+8 );
	__builtin_prefetch( B +8 );
#endif

	int k_iter = kmax/4;
	int k_left = kmax%4;
	
//	printf("\n%d %d %d\n", kmax, k_iter, k_left);

	__asm__ volatile
	(
		"                                \n\t"
		"mov    r1, %2                   \n\t" // load address of A0
		"mov    r2, %3                   \n\t" // load address of A1
		"mov    r3, %4                   \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #64]                \n\t" // prefetch A0 to L1
		"pld    [r2, #64]                \n\t" // prefetch A1 to L1
		"pld    [r3, #64]                \n\t" // prefetch B to L1
#if defined(TARGET_CORTEX_A9)
		"pld    [r1, #96]                \n\t" // prefetch A0 to L1
		"pld    [r2, #96]                \n\t" // prefetch A1 to L1
		"pld    [r3, #96]                \n\t" // prefetch B to L1
#endif
//		"pld    [r1, #128]                \n\t"
//		"pld    [r2, #128]                \n\t"
		"                                \n\t"
		"                                \n\t"
//		"ldr    r0, %0                   \n\t" // k_iter
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"                                \n\t"
#if defined(TARGET_CORTEX_A15) //|| defined(TARGET_CORTEX_A9)
		"vld1.64   {d12, d13, d14, d15}, [r3:128] \n\t" // load B to registers
		"vld1.64   {d8, d9, d10, d11},   [r1:128] \n\t" // load A0 to registers
		"vld1.64   {d24, d25}, [r2:128] \n\t" // load A1
#endif
#if defined(TARGET_CORTEX_A9)
		"vld1.64   {d12, d13, d14, d15}, [r3:128]! \n\t" // load B to registers
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0 to registers
		"vld1.64   {d24, d25, d26, d27}, [r2:128]! \n\t" // load A1
#endif
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"ldr    r4, %5                   \n\t" // load address of C
		"ldr    r5, %6                   \n\t" // load address of C
		"ldr    r6, %7                   \n\t" // alg
		"                                \n\t"
		"                                \n\t"
		"vldr   d0, .DZERO4               \n\t" // load zero double
		"vldr   d1, .DZERO4+8             \n\t" // load zero double
		"vmov   q1, q0                   \n\t"
		"vmov   q2, q0                   \n\t"
		"vmov   q3, q0                   \n\t"
		"vmov   q8, q0                   \n\t"
		"vmov   q9, q0                   \n\t"
		"vmov   q10, q0                  \n\t"
		"vmov   q11, q0                  \n\t"
		"                                \n\t"
		"                                \n\t"
		"ble    .DCONSIDERLEFT24           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER4:                    \n\t" // main loop
		"                                \n\t"
#if defined(TARGET_CORTEX_A15) //|| defined(TARGET_CORTEX_A9)
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vldr   d26, [r2, #16]             \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vldr   d27, [r2, #24]             \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"pld    [r1, #128]                \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"vldr   d8, [r1, #32]             \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vldr   d9, [r1, #40]             \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vldr   d12, [r3, #32]             \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"pld    [r2, #128]                \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"vldr   d24, [r2, #32]             \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vldr   d25, [r2, #40]             \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vldr   d13, [r3, #40]             \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"pld    [r3, #128]                \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"vldr   d10, [r1, #48]             \n\t"
		"                                \n\t"
		"vmla.f32  q8, q13, d14[0]        \n\t"
		"vldr   d11, [r1, #56]             \n\t"
		"vmla.f32  q9, q13, d14[1]        \n\t"
		"vldr   d14, [r3, #48]             \n\t"
		"vmla.f32  q10, q13, d15[0]        \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vmla.f32  q11, q13, d15[1]        \n\t"
		"vldr   d15, [r3, #56]             \n\t"
		"                                \n\t"
#if defined(TARGET_CORTEX_A9)
		"pld    [r1, #160]                \n\t" // prefetch A0 to L1
		"pld    [r2, #160]                \n\t" // prefetch A1 to L1
		"pld    [r3, #160]                \n\t" // prefetch B to L1
#endif
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vldr   d26, [r2, #48]             \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vldr   d27, [r2, #56]             \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"vldr   d8, [r1, #64]             \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vldr   d9, [r1, #72]             \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vldr   d12, [r3, #64]             \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"vldr   d24, [r2, #72]             \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vldr   d25, [r2, #72]             \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vldr   d13, [r3, #72]             \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"add    r2, r2, #64              \n\t" // increase A
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"vldr   d10, [r1, #80]             \n\t"
		"                                \n\t"
		"vmla.f32  q8, q13, d14[0]        \n\t"
		"vldr   d11, [r1, #88]             \n\t"
		"vmla.f32  q9, q13, d14[1]        \n\t"
		"vldr   d14, [r3, #80]             \n\t"
		"vmla.f32  q10, q13, d15[0]        \n\t"
		"add    r1, r1, #64              \n\t" // increase A
		"vmla.f32  q11, q13, d15[1]        \n\t"
		"vldr   d15, [r3, #88]             \n\t"
		"add    r3, r3, #64              \n\t" // increase A
		"                                \n\t"
#endif
#if defined(TARGET_CORTEX_A9)
		"                                \n\t"
		"pld    [r1, #96]                \n\t"
		"pld    [r2, #96]                \n\t"
		"pld    [r3, #96]                \n\t"
		"                                \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q13, d14[0]        \n\t"
		"vmla.f32  q9, q13, d14[1]        \n\t"
		"vmla.f32  q10, q13, d15[0]        \n\t"
		"vmla.f32  q11, q13, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r3:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0
		"vld1.64   {d24, d25, d26, d27}, [r2:128]! \n\t" // load A0
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #96]                \n\t"
		"pld    [r2, #96]                \n\t"
		"pld    [r3, #96]                \n\t"
		"                                \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q13, d14[0]        \n\t"
		"vmla.f32  q9, q13, d14[1]        \n\t"
		"vmla.f32  q10, q13, d15[0]        \n\t"
		"vmla.f32  q11, q13, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r3:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0
		"vld1.64   {d24, d25, d26, d27}, [r2:128]! \n\t" // load A0
		"                                \n\t"
#endif
		"                                \n\t"
		"bgt    .DLOOPKITER4              \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT24:                 \n\t" // consider left k+=2
		"                                \n\t"
		"mov    r0, %1                   \n\t" // k_left
		"cmp    r0, #1                   \n\t"
		"ble    .DCONSIDERLEFT14          \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q13, d14[0]        \n\t"
		"vmla.f32  q9, q13, d14[1]        \n\t"
		"vmla.f32  q10, q13, d15[0]        \n\t"
		"vmla.f32  q11, q13, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r3:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0
		"vld1.64   {d24, d25, d26, d27}, [r2:128]! \n\t" // load A0
		"                                \n\t"
		"sub    r0, r0, #2               \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT14:                 \n\t" // consider left k++
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"ble    .DPOSTACCUM4              \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q8, q12, d12[0]        \n\t"
		"vmla.f32  q9, q12, d12[1]        \n\t"
		"vmla.f32  q10, q12, d13[0]        \n\t"
		"vmla.f32  q11, q12, d13[1]        \n\t"
		"                                \n\t"
/*		"vld1.64   {d12, d13}, [r2:128]  \n\t" // no need to increment pointer*/
/*		"vld1.64   {d8, d9}, [r1:128]    \n\t" // no need to increment pointer*/
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM4:                    \n\t"
		"                                \n\t"
/*		"mov    r2, %4                   \n\t" // load address of C*/
/*		"                                \n\t"*/
/*		"mov    r5, %5                   \n\t" // alg*/
		"cmp    r6, #0                   \n\t"
		"beq    .D04                      \n\t" // if alg==0, jump
		"                                \n\t"
		"cmp    r6, #1                   \n\t"
		"                                \n\t"
		"vld1.64   {d8, d9, d10, d11},   [r4:128]! \n\t" // load C0
		"vld1.64   {d12, d13, d14, d15}, [r4:128]  \n\t" // load C0
		"vld1.64   {d24, d25, d26, d27}, [r5:128]! \n\t" // load C1
		"vld1.64   {d28, d29, d30, d31}, [r5:128]  \n\t" // load C1
		"                                \n\t"
		"ldr    r4, %8                   \n\t" // load address of D
		"ldr    r5, %9                   \n\t" // load address of D
		"                                \n\t"
/*		"mov    r2, %4                   \n\t" // load address of C*/
		"                                \n\t"
		"beq    .D14                      \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"vsub.f32  q0, q4, q0            \n\t"
		"vsub.f32  q1, q5, q1            \n\t"
		"vsub.f32  q2, q6, q2            \n\t"
		"vsub.f32  q3, q7, q3            \n\t"
		"                                \n\t"
		"vsub.f32  q8,  q12, q8            \n\t"
		"vsub.f32  q9,  q13, q9            \n\t"
		"vsub.f32  q10, q14, q10           \n\t"
		"vsub.f32  q11, q15, q11           \n\t"
		"                                \n\t"
		"b      .D04                      \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		".D14:                            \n\t" // alg==1
		"                                \n\t"
		"vadd.f32  q0, q0, q4            \n\t"
		"vadd.f32  q1, q1, q5            \n\t"
		"vadd.f32  q2, q2, q6            \n\t"
		"vadd.f32  q3, q3, q7            \n\t"
		"                                \n\t"
		"vadd.f32  q8,  q8,  q12            \n\t"
		"vadd.f32  q9,  q9,  q13           \n\t"
		"vadd.f32  q10, q10, q14           \n\t"
		"vadd.f32  q11, q11, q15           \n\t"
		"                                \n\t"
		".D04:                            \n\t" // alg==0
		"                                \n\t"
		"vst1.64   {d0, d1, d2, d3},     [r4:128]!  \n\t" // store C
		"vst1.64   {d4, d5, d6, d7},     [r4:128]   \n\t" // store C
		"vst1.64   {d16, d17, d18, d19}, [r5:128]!  \n\t" // store C
		"vst1.64   {d20, d21, d22, d23}, [r5:128]   \n\t" // store C
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO4:                         \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (k_iter),		// %0
		  "r" (k_left),		// %1
		  "r" (A0),			// %2
		  "r" (A1),			// %3
		  "r" (B),			// %4
		  "m" (C0),			// %5
		  "m" (C1),			// %6
		  "m" (alg),		// %7
		  "m" (D0),			// %8
		  "m" (D1)			// %9
		: // register clobber list
		  "r0", "r1", "r2", "r3", "r4", "r5", "r6",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);
}



// normal-transposed, 4x4 with data packed in 4
void kernel_sgemm_pp_nt_4x4_lib4(int kmax, float *A, float *B, float *C, float *D, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
		
	__builtin_prefetch( A );
	__builtin_prefetch( B );
#if defined(TARGET_CORTEX_A9)
	__builtin_prefetch( A+8 );
	__builtin_prefetch( B+8 );
#endif

	int k_iter = kmax/4;
	int k_left = kmax%4;
	
//	printf("\n%d %d %d\n", kmax, k_iter, k_left);

	__asm__ volatile
	(
		"                                \n\t"
		"mov    r1, %2                   \n\t" // load address of A
		"mov    r2, %3                   \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #64]                \n\t" // prefetch A to L1
		"pld    [r2, #64]                \n\t" // prefetch B to L1
#if defined(TARGET_CORTEX_A9)
		"pld    [r1, #96]                \n\t" // prefetch A to L1
		"pld    [r2, #96]                \n\t" // prefetch B to L1
#endif
//		"pld    [r1, #128]                \n\t"
//		"pld    [r2, #128]                \n\t"
		"                                \n\t"
		"                                \n\t"
//		"ldr    r0, %0                   \n\t" // k_iter
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"                                \n\t"
#if defined(TARGET_CORTEX_A15)
		"vld1.64   {d12, d13, d14, d15}, [r2:128] \n\t" // load B to registers
		"vld1.64   {d8, d9}, [r1:128]   \n\t" // load A to registers
#endif
#if defined(TARGET_CORTEX_A9)
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B to registers
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0 to registers
#endif
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"vldr   d0, .DZERO2               \n\t" // load zero double
		"vldr   d1, .DZERO2+8             \n\t" // load zero double
		"vmov   q1, q0                   \n\t"
		"vmov   q2, q0                   \n\t"
		"vmov   q3, q0                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"ble    .DCONSIDERLEFT2           \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER2:                    \n\t" // main loop
		"                                \n\t"
#if defined(TARGET_CORTEX_A15)
		"                                \n\t"
		"pld    [r1, #128]                \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"pld    [r2, #128]                \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vldr   d10, [r1, #16]             \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vldr   d12, [r2, #32]             \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vldr   d11, [r1, #24]             \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"vldr   d13, [r2, #40]             \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vldr   d8, [r1, #32]             \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vldr   d14, [r2, #48]             \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vldr   d9, [r1, #40]             \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"vldr   d15, [r2, #56]             \n\t"
		"                                \n\t"
/*		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B*/
/*		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A*/
		"                                \n\t"
		"                                \n\t"
#if defined(TARGET_CORTEX_A9)
		"pld    [r1, #160]                \n\t"
		"pld    [r2, #160]                \n\t"
#endif
		"                                \n\t"
		"                                \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vldr   d10, [r1, #48]             \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vldr   d12, [r2, #64]             \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vldr   d11, [r1, #56]             \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"vldr   d13, [r2, #72]             \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vldr   d8, [r1, #64]             \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vldr   d14, [r2, #80]             \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vldr   d9, [r1, #72]             \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"vldr   d15, [r2, #88]             \n\t"
		"                                \n\t"
		"add    r1, r1, #64              \n\t" // increase A
		"add    r2, r2, #64              \n\t" // increase A
		"                                \n\t"
		"                                \n\t"
/*		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B*/
/*		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A*/
#endif
#if defined(TARGET_CORTEX_A9)
		"                                \n\t"
		"pld    [r1, #96]                \n\t"
		"pld    [r2, #96]                \n\t"
		"                                \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #96]                \n\t"
		"pld    [r2, #96]                \n\t"
		"                                \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0
		"                                \n\t"
#endif
		"                                \n\t"
		"bgt    .DLOOPKITER2              \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT2:                 \n\t" // consider left k+=2
		"                                \n\t"
		"mov    r0, %1                   \n\t" // k_left
		"cmp    r0, #1                   \n\t"
		"ble    .DCONSIDERLEFT1          \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A
		"                                \n\t"
		"sub    r0, r0, #2               \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT1:                 \n\t" // consider left k++
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"ble    .DPOSTACCUM2              \n\t"
		"                                \n\t"
		"vmla.f32  q0, q4, d12[0]        \n\t"
		"vmla.f32  q1, q4, d12[1]        \n\t"
		"vmla.f32  q2, q4, d13[0]        \n\t"
		"vmla.f32  q3, q4, d13[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13}, [r2:128]  \n\t" // no need to increment pointer
		"vld1.64   {d8, d9}, [r1:128]    \n\t" // no need to increment pointer
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM2:                    \n\t"
		"                                \n\t"
		"mov    r2, %4                   \n\t" // load address of C
		"                                \n\t"
		"mov    r5, %5                   \n\t" // alg
		"cmp    r5, #0                   \n\t"
		"beq    .D02                      \n\t" // if alg==0, jump
		"                                \n\t"
		"cmp    r5, #1                   \n\t"
		"                                \n\t"
		"vld1.64   {d8, d9, d10, d11}, [r2:128]!  \n\t" // load C
		"vld1.64   {d12, d13, d14, d15}, [r2:128] \n\t" // load C
		"                                \n\t"
		"mov    r2, %6                   \n\t" // load address of D
		"                                \n\t"
		"beq    .D12                      \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"vsub.f32  q0, q4, q0            \n\t"
		"vsub.f32  q1, q5, q1            \n\t"
		"vsub.f32  q2, q6, q2            \n\t"
		"vsub.f32  q3, q7, q3            \n\t"
		"                                \n\t"
		"b      .D02                      \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		".D12:                            \n\t" // alg==1
		"                                \n\t"
		"vadd.f32  q0, q0, q4            \n\t"
		"vadd.f32  q1, q1, q5            \n\t"
		"vadd.f32  q2, q2, q6            \n\t"
		"vadd.f32  q3, q3, q7            \n\t"
		"                                \n\t"
		".D02:                            \n\t" // alg==0
		"                                \n\t"
		"vst1.64   {d0, d1, d2, d3}, [r2:128]!  \n\t" // store D
		"vst1.64   {d4, d5, d6, d7}, [r2:128]   \n\t" // store D
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO2:                         \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (k_iter),		// %0
		  "r" (k_left),		// %1
		  "r" (A),			// %2
		  "r" (B),			// %3
		  "r" (C),			// %4
		  "r" (alg),		// %5
		  "r" (D)			// %6
		: // register clobber list
		  "r0", "r1", "r2", "r3", "r4", "r5",
		  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
		  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
		  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
		  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
		  "memory"
	);
}



void kernel_sgemm_pp_nt_4x4_lib4_old(int kmax, float *A, float *B, float *C, float *D, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int lda = 4;

	int k;

	float
		a_0, a_1, a_2, a_3,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0,
		c_20=0, c_21=0, c_22=0, c_23=0,
		c_30=0, c_31=0, c_32=0, c_33=0;
		
	for(k=0; k<kmax-3; k+=4)
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;
		c_22 += a_2 * b_2;
		c_32 += a_3 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		c_23 += a_2 * b_3;
		c_33 += a_3 * b_3;


		A += 4;
		B += 4;

		}
		
	float
		d_00, d_01, d_02, d_03,
		d_10, d_11, d_12, d_13,
		d_20, d_21, d_22, d_23,
		d_30, d_31, d_32, d_33;
	
	if(alg==0) // C = A * B'
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

		C[0+ldc*3] = c_03;
		C[1+ldc*3] = c_13;
		C[2+ldc*3] = c_23;
		C[3+ldc*3] = c_33;
		}
	else 
		{
		d_00 = D[0+ldc*0];
		d_10 = D[1+ldc*0];
		d_20 = D[2+ldc*0];
		d_30 = D[3+ldc*0];
		
		d_01 = D[0+ldc*1];
		d_11 = D[1+ldc*1];
		d_21 = D[2+ldc*1];
		d_31 = D[3+ldc*1];
		
		d_02 = D[0+ldc*2];
		d_12 = D[1+ldc*2];
		d_22 = D[2+ldc*2];
		d_32 = D[3+ldc*2];
		
		d_03 = D[0+ldc*3];
		d_13 = D[1+ldc*3];
		d_23 = D[2+ldc*3];
		d_33 = D[3+ldc*3];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_01 += c_01;
			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;

			d_02 += c_02;
			d_12 += c_12;
			d_22 += c_22;
			d_32 += c_32;

			d_03 += c_03;
			d_13 += c_13;
			d_23 += c_23;
			d_33 += c_33;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_01 -= c_01;
			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;

			d_02 -= c_02;
			d_12 -= c_12;
			d_22 -= c_22;
			d_32 -= c_32;

			d_03 -= c_03;
			d_13 -= c_13;
			d_23 -= c_23;
			d_33 -= c_33;
			}

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

		C[0+ldc*3] = c_03;
		C[1+ldc*3] = c_13;
		C[2+ldc*3] = c_23;
		C[3+ldc*3] = c_33;
		}
	
	}



// normal-transposed, 4x2 with data packed in 4
void kernel_sgemm_pp_nt_4x2_lib4(int kmax, float *A, float *B, float *C, float *D, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int lda = 4;

	int k;

	float
		a_0, a_1, a_2, a_3,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0,
		c_20=0, c_21=0,
		c_30=0, c_31=0;
		
	for(k=0; k<kmax-3; k+=4)
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

		c_01 += a_0 * b_1;
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

		c_01 += a_0 * b_1;
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

		c_01 += a_0 * b_1;
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
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

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		c_21 += a_2 * b_1;
		c_31 += a_3 * b_1;


		A += 4;
		B += 4;

		}
		
	float
		d_00, d_01,
		d_10, d_11,
		d_20, d_21,
		d_30, d_31;
	
	if(alg==0) // C = A * B'
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
	else 
		{
		d_00 = D[0+ldc*0];
		d_10 = D[1+ldc*0];
		d_20 = D[2+ldc*0];
		d_30 = D[3+ldc*0];
		
		d_01 = D[0+ldc*1];
		d_11 = D[1+ldc*1];
		d_21 = D[2+ldc*1];
		d_31 = D[3+ldc*1];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;
			d_20 += c_20;
			d_30 += c_30;

			d_01 += c_01;
			d_11 += c_11;
			d_21 += c_21;
			d_31 += c_31;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;
			d_20 -= c_20;
			d_30 -= c_30;

			d_01 -= c_01;
			d_11 -= c_11;
			d_21 -= c_21;
			d_31 -= c_31;
			}

		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;
		C[2+ldc*0] = c_20;
		C[3+ldc*0] = c_30;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;
		C[2+ldc*1] = c_21;
		C[3+ldc*1] = c_31;
		}
	
	}



// normal-transposed, 2x4 with data packed in 4
void kernel_sgemm_pp_nt_2x4_lib4(int kmax, float *A, float *B, float *C, float *D, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int lda = 4;

	int k;

	float
		a_0, a_1,
		b_0, b_1, b_2, b_3,
		c_00=0, c_01=0, c_02=0, c_03=0,
		c_10=0, c_11=0, c_12=0, c_13=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		b_2 = B[2+lda*0];
		b_3 = B[3+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		b_2 = B[2+lda*1];
		b_3 = B[3+lda*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		b_2 = B[2+lda*2];
		b_3 = B[3+lda*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		b_2 = B[2+lda*3];
		b_3 = B[3+lda*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		b_2 = B[2+lda*0];
		b_3 = B[3+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;

		c_02 += a_0 * b_2;
		c_12 += a_1 * b_2;

		c_03 += a_0 * b_3;
		c_13 += a_1 * b_3;


		A += 4;
		B += 4;

		}
		
	float
		d_00, d_01, d_02, d_03,
		d_10, d_11, d_12, d_13;
	
	if(alg==0) // C = A * B'
		{
		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;

		C[0+ldc*2] = c_02;
		C[1+ldc*2] = c_12;

		C[0+ldc*3] = c_03;
		C[1+ldc*3] = c_13;
		}
	else 
		{
		d_00 = D[0+ldc*0];
		d_10 = D[1+ldc*0];
		
		d_01 = D[0+ldc*1];
		d_11 = D[1+ldc*1];
		
		d_02 = D[0+ldc*2];
		d_12 = D[1+ldc*2];
		
		d_03 = D[0+ldc*3];
		d_13 = D[1+ldc*3];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;

			d_01 += c_01;
			d_11 += c_11;

			d_02 += c_02;
			d_12 += c_12;

			d_03 += c_03;
			d_13 += c_13;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;

			d_01 -= c_01;
			d_11 -= c_11;

			d_02 -= c_02;
			d_12 -= c_12;

			d_03 -= c_03;
			d_13 -= c_13;
			}

		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;

		C[0+ldc*2] = c_02;
		C[1+ldc*2] = c_12;

		C[0+ldc*3] = c_03;
		C[1+ldc*3] = c_13;
		}
	
	}



// normal-transposed, 2x2 with data packed in 4
void kernel_sgemm_pp_nt_2x2_lib4(int kmax, float *A, float *B, float *C, float *D, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;
	
	const int lda = 4;

	int k;

	float
		a_0, a_1,
		b_0, b_1,
		c_00=0, c_01=0,
		c_10=0, c_11=0;
		
	for(k=0; k<kmax-3; k+=4)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+lda*1];
		a_1 = A[1+lda*1];
		
		b_0 = B[0+lda*1];
		b_1 = B[1+lda*1];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+lda*2];
		a_1 = A[1+lda*2];
		
		b_0 = B[0+lda*2];
		b_1 = B[1+lda*2];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		a_0 = A[0+lda*3];
		a_1 = A[1+lda*3];
		
		b_0 = B[0+lda*3];
		b_1 = B[1+lda*3];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;
		
		
		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{
		
		a_0 = A[0+lda*0];
		a_1 = A[1+lda*0];
		
		b_0 = B[0+lda*0];
		b_1 = B[1+lda*0];
		
		c_00 += a_0 * b_0;
		c_10 += a_1 * b_0;

		c_01 += a_0 * b_1;
		c_11 += a_1 * b_1;


		A += 4;
		B += 4;

		}
		
	float
		d_00, d_01,
		d_10, d_11;
	
	if(alg==0) // C = A * B'
		{
		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;
		}
	else 
		{
		d_00 = D[0+ldc*0];
		d_10 = D[1+ldc*0];
		
		d_01 = D[0+ldc*1];
		d_11 = D[1+ldc*1];
		
		if(alg==1) // C += A * B'
			{
			d_00 += c_00;
			d_10 += c_10;

			d_01 += c_01;
			d_11 += c_11;
			}
		else // C -= A * B'
			{
			d_00 -= c_00;
			d_10 -= c_10;

			d_01 -= c_01;
			d_11 -= c_11;
			}

		C[0+ldc*0] = c_00;
		C[1+ldc*0] = c_10;

		C[0+ldc*1] = c_01;
		C[1+ldc*1] = c_11;
		}
	
	}

