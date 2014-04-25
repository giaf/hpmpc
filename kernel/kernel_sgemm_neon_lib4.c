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

void kernel_sgemm_pp_nt_8x4_lib4(int kmax, float *A0, float *A1, float *B, float *C0, float *C1, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
		
	__builtin_prefetch( A0 );
	__builtin_prefetch( A1 );
	__builtin_prefetch( B  );
	__builtin_prefetch( A0+8 );
	__builtin_prefetch( A1+8 );
	__builtin_prefetch( B +8 );

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
		"pld    [r1, #96]                \n\t" // prefetch A0 to L1
		"pld    [r2, #96]                \n\t" // prefetch A1 to L1
		"pld    [r3, #96]                \n\t" // prefetch B to L1
//		"pld    [r1, #128]                \n\t"
//		"pld    [r2, #128]                \n\t"
		"                                \n\t"
		"                                \n\t"
//		"ldr    r0, %0                   \n\t" // k_iter
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r3:128]! \n\t" // load B to registers
		"vld1.64   {d8, d9, d10, d11},   [r1:128]! \n\t" // load A0 to registers
		"vld1.64   {d24, d25, d26, d27}, [r2:128]! \n\t" // load A1
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
		".DLOOPKITER4:                    \n\t" // main loop
		"                                \n\t"
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
		"                                \n\t"
		"bgt    .DLOOPKITER4              \n\t"
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
		"ldr    r4, %5                   \n\t" // load address of C
		"ldr    r5, %6                   \n\t" // load address of C
		"                                \n\t"
		"mov    r2, %4                   \n\t" // load address of C
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
		  "m" (alg)			// %7
		: // register clobber list
		  "r0", "r1", "r2", "r3", "r4", "r5", "r6",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);
}



void kernel_sgemm_pp_nt_4x4_vfpv3_lib4(int kmax, float *A, float *B, float *C, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
		
	__builtin_prefetch( A );
	__builtin_prefetch( B );
	__builtin_prefetch( A+8 );
	__builtin_prefetch( B+8 );

	int k_iter = kmax/4;
	int k_left = kmax%4;
	
//	printf("\n%d %d %d\n", kmax, k_iter, k_left);

	__asm__ volatile
	(
		"                                \n\t"
//		"ldr    r1, %2                   \n\t" // load address of A
///		"ldr    r2, %3                   \n\t" // load address of B
		"mov    r1, %2                   \n\t" // load address of A
		"mov    r2, %3                   \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #64]                \n\t"
		"pld    [r2, #64]                \n\t"
		"pld    [r1, #96]                \n\t"
		"pld    [r2, #96]                \n\t"
		"                                \n\t"
		"                                \n\t"
//		"ldr    r0, %0                   \n\t" // k_iter
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"                                \n\t"
		"vldr   d8, [r1, #0]             \n\t" // prefetch A_even
		"vldr   d9, [r1, #8]             \n\t"
		"                                \n\t"
		"vldr   d10, [r2, #0]            \n\t" // prefetch B_even
		"vldr   d11, [r2, #8]            \n\t"
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"vldr   d12, [r1, #16]           \n\t" // prefetch A_odd
		"vldr   d13, [r1, #24]           \n\t"
		"                                \n\t"
		"vldr   d14, [r2, #16]           \n\t" // prefetch B_odd
		"vldr   d15, [r2, #24]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"flds   s0, .DOUBLEZERO          \n\t" // load zero double
		"fcpys  s1, s0                   \n\t"
		"fcpys  s2, s0                   \n\t"
		"fcpys  s3, s0                   \n\t"
		"fcpys  s4, s0                   \n\t"
		"fcpys  s5, s0                   \n\t"
		"fcpys  s6, s0                   \n\t"
		"fcpys  s7, s0                   \n\t"
		"fcpys  s8, s0                   \n\t"
		"fcpys  s9, s0                   \n\t"
		"fcpys  s10, s0                  \n\t"
		"fcpys  s11, s0                  \n\t"
		"fcpys  s12, s0                  \n\t"
		"fcpys  s13, s0                  \n\t"
		"fcpys  s14, s0                  \n\t"
		"fcpys  s15, s0                  \n\t"
		"                                \n\t"
		"                                \n\t"
		"ble    .DCONSIDERLEFT           \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER:                    \n\t" // main loop
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #128]               \n\t"
		"pld    [r2, #128]               \n\t"
		"                                \n\t"
		"                                \n\t"
		"fmacs  s0, s16, s20             \n\t"
		"fmacs  s1, s17, s20             \n\t"
		"fmacs  s2, s18, s20             \n\t"
		"fmacs  s3, s19, s20             \n\t"
		"flds   s20, [r2, #32]           \n\t" // prefetch B_even
		"                                \n\t"
		"fmacs  s4, s16, s21             \n\t"
		"fmacs  s5, s17, s21             \n\t"
		"fmacs  s6, s18, s21             \n\t"
		"fmacs  s7, s19, s21             \n\t"
		"flds   s21, [r2, #36]           \n\t"
		"                                \n\t"
		"fmacs  s8, s16, s22             \n\t"
		"fmacs  s9, s17, s22             \n\t"
		"fmacs  s10, s18, s22            \n\t"
		"fmacs  s11, s19, s22            \n\t"
		"flds   s22, [r2, #40]           \n\t"
		"                                \n\t"
		"fmacs  s12, s16, s23            \n\t"
		"flds   s16, [r1, #32]           \n\t" // prefetch A_even
		"fmacs  s13, s17, s23            \n\t"
		"flds   s17, [r1, #36]           \n\t"
		"fmacs  s14, s18, s23            \n\t"
		"flds   s18, [r1, #40]           \n\t"
		"fmacs  s15, s19, s23            \n\t"
		"flds   s23, [r2, #44]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"fmacs  s0, s24, s28             \n\t"
		"flds   s19, [r1, #44]           \n\t"
		"fmacs  s1, s25, s28             \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"fmacs  s2, s26, s28             \n\t"
		"fmacs  s3, s27, s28             \n\t"
		"flds   s28, [r2, #48]           \n\t" // prefetch B_odd
		"                                \n\t"
		"fmacs  s4, s24, s29             \n\t"
		"fmacs  s5, s25, s29             \n\t"
		"fmacs  s6, s26, s29             \n\t"
		"fmacs  s7, s27, s29             \n\t"
		"flds   s29, [r2, #52]           \n\t"
		"                                \n\t"
		"fmacs  s8, s24, s30             \n\t"
		"fmacs  s9, s25, s30             \n\t"
		"fmacs  s10, s26, s30            \n\t"
		"fmacs  s11, s27, s30            \n\t"
		"flds   s30, [r2, #56]           \n\t"
		"                                \n\t"
		"fmacs  s12, s24, s31            \n\t"
		"flds   s24, [r1, #48]           \n\t" // prefetch A_odd
		"fmacs  s13, s25, s31            \n\t"
		"flds   s25, [r1, #52]           \n\t"
		"fmacs  s14, s26, s31            \n\t"
		"flds   s26, [r1, #56]           \n\t"
		"fmacs  s15, s27, s31            \n\t"
		"flds   s31, [r2, #60]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"pld    [r1, #192]               \n\t"
		"pld    [r2, #192]               \n\t"
		"                                \n\t"
		"                                \n\t"
		"fmacs  s0, s16, s20             \n\t"
		"flds   s27, [r1, #60]           \n\t"
		"fmacs  s1, s17, s20             \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"fmacs  s2, s18, s20             \n\t"
		"fmacs  s3, s19, s20             \n\t"
		"flds   s20, [r2, #64]           \n\t" // prefetch B_even
		"                                \n\t"
		"fmacs  s4, s16, s21             \n\t"
		"fmacs  s5, s17, s21             \n\t"
		"fmacs  s6, s18, s21             \n\t"
		"fmacs  s7, s19, s21             \n\t"
		"flds   s21, [r2, #68]           \n\t"
		"                                \n\t"
		"fmacs  s8, s16, s22             \n\t"
		"fmacs  s9, s17, s22             \n\t"
		"fmacs  s10, s18, s22            \n\t"
		"fmacs  s11, s19, s22            \n\t"
		"flds   s22, [r2, #72]           \n\t"
		"                                \n\t"
		"fmacs  s12, s16, s23            \n\t"
		"flds   s16, [r1, #64]           \n\t" // prefetch A_even
		"fmacs  s13, s17, s23            \n\t"
		"flds   s17, [r1, #68]           \n\t"
		"fmacs  s14, s18, s23            \n\t"
		"flds   s18, [r1, #72]           \n\t"
		"fmacs  s15, s19, s23            \n\t"
		"flds   s19, [r1, #76]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"fmacs  s0, s24, s28             \n\t"
		"add    r1, r1, #64              \n\t" // increase A
		"fmacs  s1, s25, s28             \n\t"
		"flds   s23, [r2, #76]           \n\t"
		"fmacs  s2, s26, s28             \n\t"
		"add    r2, r2, #64              \n\t" // increase B
		"fmacs  s3, s27, s28             \n\t"
		"flds   s28, [r2, #16]           \n\t" // prefetch B_odd
		"                                \n\t"
		"fmacs  s4, s24, s29             \n\t"
		"fmacs  s5, s25, s29             \n\t"
		"fmacs  s6, s26, s29             \n\t"
		"fmacs  s7, s27, s29             \n\t"
		"flds   s29, [r2, #20]           \n\t"
		"                                \n\t"
		"fmacs  s8, s24, s30             \n\t"
		"fmacs  s9, s25, s30             \n\t"
		"fmacs  s10, s26, s30            \n\t"
		"fmacs  s11, s27, s30            \n\t"
		"flds   s30, [r2, #24]           \n\t"
		"                                \n\t"
		"fmacs  s12, s24, s31            \n\t"
		"flds   s24, [r1, #16]           \n\t" // prefetch A_odd
		"fmacs  s13, s25, s31            \n\t"
		"flds   s25, [r1, #20]           \n\t"
		"fmacs  s14, s26, s31            \n\t"
		"flds   s26, [r1, #24]           \n\t"
		"fmacs  s15, s27, s31            \n\t"
		"flds   s31, [r2, #28]           \n\t"
		"flds   s27, [r1, #28]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"bgt    .DLOOPKITER              \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT:                 \n\t" // consider left
		"                                \n\t"
//		"ldr    r0, %1                   \n\t" // k_left
		"mov    r0, %1                   \n\t" // k_left
		"cmp    r0, #0                   \n\t"
		"ble    .DPOSTACCUM              \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT:                    \n\t" // clean up loop
		"                                \n\t"
		"sub    r0, r0, #1               \n\t"
		"                                \n\t"
		"fmacs  s0, s16, s20             \n\t"
		"fmacs  s1, s17, s20             \n\t"
		"fmacs  s2, s18, s20             \n\t"
		"fmacs  s3, s19, s20             \n\t"
		"                                \n\t"
		"fmacs  s4, s16, s21             \n\t"
		"fmacs  s5, s17, s21             \n\t"
		"fmacs  s6, s18, s21             \n\t"
		"fmacs  s7, s19, s21             \n\t"
		"                                \n\t"
		"fmacs  s8, s16, s22             \n\t"
		"fmacs  s9, s17, s22             \n\t"
		"fmacs  s10, s18, s22            \n\t"
		"fmacs  s11, s19, s22            \n\t"
		"                                \n\t"
		"fmacs  s12, s16, s23            \n\t"
		"fmacs  s13, s17, s23            \n\t"
		"fmacs  s14, s18, s23            \n\t"
		"fmacs  s15, s19, s23            \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"flds   s16, [r1, #16]           \n\t" // prefetch A_even
		"flds   s17, [r1, #20]           \n\t"
		"flds   s18, [r1, #24]           \n\t"
		"flds   s19, [r1, #28]           \n\t"
		"                                \n\t"
		"flds   s20, [r2, #16]           \n\t" // prefetch B_even
		"flds   s21, [r2, #20]           \n\t"
		"flds   s22, [r2, #24]           \n\t"
		"flds   s23, [r2, #28]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"add    r1, r1, #16              \n\t"
		"add    r2, r2, #16              \n\t"
		"                                \n\t"
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"bgt    .DLOOPKLEFT              \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM:                    \n\t"
		"                                \n\t"
//		"ldr    r2, %4                   \n\t" // load address of C
		"mov    r2, %4                   \n\t" // load address of C
		"                                \n\t"
		"                                \n\t"
//		"ldr    r5, %5                   \n\t" // alg
		"mov    r5, %5                   \n\t" // alg
		"cmp    r5, #0                   \n\t"
		"beq    .D0                      \n\t" // if alg==0, jump
		"                                \n\t"
		"cmp    r5, #1                   \n\t"
		"                                \n\t"
		"flds   s16, [r2, #0]            \n\t" // load C elements
		"flds   s17, [r2, #4]            \n\t"
		"flds   s18, [r2, #8]            \n\t"
		"flds   s19, [r2, #12]           \n\t"
		"                                \n\t"
		"flds   s20, [r2, #16]           \n\t"
		"flds   s21, [r2, #20]           \n\t"
		"flds   s22, [r2, #24]           \n\t"
		"flds   s23, [r2, #28]           \n\t"
		"                                \n\t"
		"flds   s24, [r2, #32]           \n\t"
		"flds   s25, [r2, #36]           \n\t"
		"flds   s26, [r2, #40]           \n\t"
		"flds   s27, [r2, #44]           \n\t"
		"                                \n\t"
		"flds   s28, [r2, #48]           \n\t"
		"flds   s29, [r2, #52]           \n\t"
		"flds   s30, [r2, #56]           \n\t"
		"flds   s31, [r2, #60]           \n\t"
		"                                \n\t"
		"beq    .D1                      \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"fsubs  s0, s16, s0              \n\t"
		"fsubs  s1, s17, s1              \n\t"
		"fsubs  s2, s18, s2              \n\t"
		"fsubs  s3, s19, s3              \n\t"
		"                                \n\t"
		"fsubs  s4, s20, s4              \n\t"
		"fsubs  s5, s21, s5              \n\t"
		"fsubs  s6, s22, s6              \n\t"
		"fsubs  s7, s23, s7              \n\t"
		"                                \n\t"
		"fsubs  s8, s24, s8              \n\t"
		"fsubs  s9, s25, s9              \n\t"
		"fsubs  s10, s26, s10            \n\t"
		"fsubs  s11, s27, s11            \n\t"
		"                                \n\t"
		"fsubs  s12, s28, s12            \n\t"
		"fsubs  s13, s29, s13            \n\t"
		"fsubs  s14, s30, s14            \n\t"
		"fsubs  s15, s31, s15            \n\t"
		"                                \n\t"
		"b      .D0                      \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		".D1:                            \n\t" // alg==1
		"                                \n\t"
		"fadds  s0, s0, s16              \n\t"
		"fadds  s1, s1, s17              \n\t"
		"fadds  s2, s2, s18              \n\t"
		"fadds  s3, s3, s19              \n\t"
		"                                \n\t"
		"fadds  s4, s4, s20              \n\t"
		"fadds  s5, s5, s21              \n\t"
		"fadds  s6, s6, s22              \n\t"
		"fadds  s7, s7, s23              \n\t"
		"                                \n\t"
		"fadds  s8, s8, s24              \n\t"
		"fadds  s9, s9, s25              \n\t"
		"fadds  s10, s10, s26            \n\t"
		"fadds  s11, s11, s27            \n\t"
		"                                \n\t"
		"fadds  s12, s12, s28            \n\t"
		"fadds  s13, s13, s29            \n\t"
		"fadds  s14, s14, s30            \n\t"
		"fadds  s15, s15, s31            \n\t"
		"                                \n\t"
		".D0:                            \n\t" // alg==0
		"                                \n\t"
		"fsts   s0, [r2, #0]             \n\t" // store result
		"fsts   s1, [r2, #4]             \n\t"
		"fsts   s2, [r2, #8]             \n\t"
		"fsts   s3, [r2, #12]            \n\t"
		"                                \n\t"
		"fsts   s4, [r2, #16]            \n\t"
		"fsts   s5, [r2, #20]            \n\t"
		"fsts   s6, [r2, #24]            \n\t"
		"fsts   s7, [r2, #28]            \n\t"
		"                                \n\t"
		"fsts   s8, [r2, #32]            \n\t"
		"fsts   s9, [r2, #36]            \n\t"
		"fsts   s10, [r2, #40]           \n\t"
		"fsts   s11, [r2, #44]           \n\t"
		"                                \n\t"
		"fsts   s12, [r2, #48]           \n\t"
		"fsts   s13, [r2, #52]           \n\t"
		"fsts   s14, [r2, #56]           \n\t"
		"fsts   s15, [r2, #60]           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 2                        \n\t"
		".DOUBLEZERO:                    \n\t" // zero double word
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
		  "r" (alg)			// %5
		: // register clobber list
		  "r0", "r1", "r2", "r3", "r4", "r5",
		  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
		  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
		  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
		  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
		  "memory"
	);
}



void kernel_sgemm_pp_nt_4x4_lib4(int kmax, float *A, float *B, float *C, int ldc_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
		
	__builtin_prefetch( A );
	__builtin_prefetch( B );
	__builtin_prefetch( A+8 );
	__builtin_prefetch( B+8 );

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
		"pld    [r1, #96]                \n\t" // prefetch A to L1
		"pld    [r2, #96]                \n\t" // prefetch B to L1
//		"pld    [r1, #128]                \n\t"
//		"pld    [r2, #128]                \n\t"
		"                                \n\t"
		"                                \n\t"
//		"ldr    r0, %0                   \n\t" // k_iter
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B to registers
		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A to registers
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
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A
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
		"                                \n\t"
		"vmla.f32  q0, q5, d14[0]        \n\t"
		"vmla.f32  q1, q5, d14[1]        \n\t"
		"vmla.f32  q2, q5, d15[0]        \n\t"
		"vmla.f32  q3, q5, d15[1]        \n\t"
		"                                \n\t"
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load B
		"vld1.64   {d8, d9, d10, d11}, [r1:128]!   \n\t" // load A
		"                                \n\t"
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
		"mov    r2, %4                   \n\t" // load address of C
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
		"vst1.64   {d0, d1, d2, d3}, [r2:128]!  \n\t" // store C
		"vst1.64   {d4, d5, d6, d7}, [r2:128]   \n\t" // store C
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
		  "r" (alg)			// %5
		: // register clobber list
		  "r0", "r1", "r2", "r3", "r4", "r5",
		  "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
		  "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
		  "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
		  "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
		  "memory"
	);
}



// normal-transposed, 4x3 with data packed in 4
void kernel_sgemm_pp_nt_4x3_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
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
void kernel_sgemm_pp_nt_4x2_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
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



// normal-transposed, 4x1 with data packed in 4
void kernel_sgemm_pp_nt_4x1_lib4(int kmax, float *A, float *B, float *C, int ldc, int alg)
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

