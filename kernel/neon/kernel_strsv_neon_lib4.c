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



void kernel_strsv_n_8_lib4(int kmax, float *A0, float *A1, float *x, float *y)
	{

/*	if(kmax<=0) */
/*		return;*/
	
	__builtin_prefetch( A0 );
	__builtin_prefetch( A1 );

	int k_iter = kmax/8;
/*	int k_left = kmax%8;*/

/*printf("\n%d\n", k_iter);*/
/*exit(1);*/

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"pld    [%1, #64]                \n\t" // prefetch A0 to L1
		"pld    [%2, #64]                \n\t" // prefetch A1 to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vldr   d24, .DZERO_N_8          \n\t" // load zero double
		"vldr   d25, .DZERO_N_8+8        \n\t" // load zero double
		"vmov   q13, q12                 \n\t"
		"vmov   q14, q12                 \n\t"
		"vmov   q15, q12                 \n\t"
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d0, d1, d2, d3}, [%3:128]!   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%1:128]!   \n\t" // load A0 to registers
		"vld1.64   {d16, d17, d18, d19}, [%2:128]!   \n\t" // load A1 to registers
		"vld1.64   {d12, d13, d14, d15}, [%1:128]!   \n\t" // load A0 to registers
		"vld1.64   {d20, d21, d22, d23}, [%2:128]!   \n\t" // load A1 to registers
		"                                \n\t"
		"                                \n\t"
		"ble    .DPOSTACCUM_N_8          \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER_N_8:                \n\t" // main loop
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d4, d5, d6, d7}, [%3:128]!   \n\t" // load x to registers
		"pld    [%1, #64]                \n\t" // prefetch A0 to L1
		"vmla.f32  q12, q4, d0[0]         \n\t"
		"vmla.f32  q13, q5, d0[1]         \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%1:128]!   \n\t" // load A0 to registers
		"pld    [%2, #64]                \n\t" // prefetch A1 to L1
		"vmla.f32  q14, q8, d0[0]         \n\t"
		"vmla.f32  q15, q9, d0[1]         \n\t"
		"vld1.64   {d16, d17, d18, d19}, [%2:128]!   \n\t" // load A1 to registers
		"vmla.f32  q12, q6, d1[0]         \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vmla.f32  q13, q7, d1[1]         \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%1:128]!   \n\t" // load A0 to registers
		"vmla.f32  q14, q10, d1[0]         \n\t"
		"vmla.f32  q15, q11, d1[1]         \n\t"
		"vld1.64   {d20, d21, d22, d23}, [%2:128]!   \n\t" // load A1 to registers
		"vmov   q0, q2                   \n\t"
		"                                \n\t"
		"pld    [%1, #64]                \n\t" // prefetch A1 to L1
		"vmla.f32  q12, q4, d2[0]         \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"vmla.f32  q13, q5, d2[1]         \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%1:128]!   \n\t" // load A0 to registers
		"pld    [%2, #64]                \n\t" // prefetch A1 to L1
		"vmla.f32  q14, q8, d2[0]         \n\t"
		"vmla.f32  q15, q9, d2[1]         \n\t"
		"vld1.64   {d16, d17, d18, d19}, [%2:128]!   \n\t" // load A1 to registers
		"vmla.f32  q12, q6, d3[0]         \n\t"
		"vmla.f32  q13, q7, d3[1]         \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%1:128]!   \n\t" // load A0 to registers
		"vmla.f32  q14, q10, d3[0]         \n\t"
		"vmla.f32  q15, q11, d3[1]         \n\t"
		"vld1.64   {d20, d21, d22, d23}, [%2:128]!   \n\t" // load A1 to registers
		"vmov   q1, q3                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"bgt    .DLOOPKITER_N_8          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"vadd.f32  q12, q12, q13         \n\t"
		"vadd.f32  q14, q14, q15         \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM_N_8:                \n\t"
		"                                \n\t"
		"vmov   q2, q4                   \n\t"
		"vmov   q3, q5                   \n\t"
		"vmov   q4, q8                   \n\t"
		"vmov   q5, q9                   \n\t"
		"vmov   q8, q6                   \n\t"
		"vmov   q9, q7                   \n\t"
		"                                \n\t"// alg==-1
		"mov    r0, %4                   \n\t" // load address of y
		"                                \n\t"
		"vld1.64   {d0, d1, d2, d3}, [r0]  \n\t" // load y
		"vsub.f32  q0, q0, q12           \n\t"
		"vsub.f32  q1, q1, q14           \n\t"
		"                                \n\t"
/*		"vst1.64   {d0, d1}, [%4]!  \n\t" // store y*/
/*		"vst1.64   {d2, d3}, [%4]!  \n\t" // store y*/
/*		"b .DEND                         \n\t"// alg==-1*/
		"                                \n\t"
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%1:128]!  \n\t" // load A0*/
/*		"vld1.64   {d8, d9, d10, d11}, [%2:128]!  \n\t" // load A1*/
		"                                \n\t"
		"vmul.f32  s24, s0, s8           \n\t"
		"vmls.f32  q0, q2, d12[0]        \n\t"
		"vmls.f32  q1, q4, d12[0]        \n\t"
		"                                \n\t"
		"vmul.f32  s25, s1, s13          \n\t"
/*		"vmls.f32  q0, q3, d12[1]        \n\t"*/
		"vmls.f32  d1, d7, d12[1]        \n\t"
		"vmls.f32  q1, q5, d12[1]        \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%1:128]!  \n\t" // load A0*/
/*		"vld1.64   {d8, d9, d10, d11}, [%2:128]!  \n\t" // load A1*/
		"vmov   q2, q8                   \n\t"
		"vmov   q3, q9                   \n\t"
		"vmov   q4, q10                  \n\t"
		"vmov   q5, q11                  \n\t"
		"                                \n\t"
		"vmul.f32  s26, s2, s10          \n\t"
/*		"vmls.f32  q0, q2, d13[0]        \n\t"*/
/*		"vmls.f32  d1, d5, d13[0]        \n\t"*/
		"vmls.f32  s3, s11, s26          \n\t"
		"vmls.f32  q1, q4, d13[0]        \n\t"
		"                                \n\t"
		"vmul.f32  s27, s3, s15          \n\t"
/*		"vmls.f32  q0, q3, d13[1]        \n\t"*/
/*		"vmls.f32  s3, s15, s27          \n\t"*/
		"vmls.f32  q1, q5, d13[1]        \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%1:128]!  \n\t" // load A0*/
		"vld1.64   {d8, d9, d10, d11}, [%2:128]!  \n\t" // load A1
		"                                \n\t"
		"vmul.f32  s28, s4, s16          \n\t"
/*		"vmls.f32  q0, q2, d12[0]        \n\t"*/
		"vmls.f32  q1, q4, d14[0]        \n\t"
		"                                \n\t"
		"vmul.f32  s29, s5, s21          \n\t"
/*		"vmls.f32  q0, q3, d12[1]        \n\t"*/
/*		"vmls.f32  q1, q5, d14[1]        \n\t"*/
		"vmls.f32  d3, d11, d14[1]       \n\t"
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%1:128]!  \n\t" // load A0*/
		"vld1.64   {d8, d9, d10, d11}, [%2:128]!  \n\t" // load A1
		"                                \n\t"
		"vmul.f32  s30, s6, s18          \n\t"
/*		"vmls.f32  q0, q2, d12[0]        \n\t"*/
/*		"vmls.f32  q1, q4, d15[0]        \n\t"*/
		"vmls.f32  s7, s19, s30           \n\t"
		"                                \n\t"
		"vmul.f32  s31, s7, s23          \n\t"
/*		"vmls.f32  q0, q3, d12[1]        \n\t"*/
/*		"vmls.f32  q1, q5, d15[1]        \n\t"*/
		"                                \n\t"
		"                                \n\t"
		"vst1.64   {d12, d13, d14, d15}, [r0]  \n\t" // store y
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO_N_8:                     \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
/*		".DEND:                          \n\t"*/
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (k_iter),		// %0
		  "r" (A0),			// %1
		  "r" (A1),			// %2
		  "r" (x),			// %3
		  "r" (y)			// %4
		: // register clobber list
		  "r0", "r1", "r2",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);
	
	}



/*void kernel_strsv_n_8_lib4_old(int kmax, float *A0, float *A1, float *x, float *y)*/
/*	{*/

/*//	if(kmax<=0) */
/*//		return;*/
/*	*/
/*	const int lda = 4;*/
/*	*/
/*	int k;*/

/*	float*/
/*		x_0, x_1, x_2, x_3,*/
/*		y_0=0, y_1=0, y_2=0, y_3=0,*/
/*		y_4=0, y_5=0, y_6=0, y_7=0;*/
/*	*/
/*	k=0;*/
/*	for(; k<kmax-7; k+=8)*/
/*		{*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/

/*		y_0 += A0[0+lda*0] * x_0;*/
/*		y_1 += A0[1+lda*0] * x_0;*/
/*		y_2 += A0[2+lda*0] * x_0;*/
/*		y_3 += A0[3+lda*0] * x_0;*/
/*		y_4 += A1[0+lda*0] * x_0;*/
/*		y_5 += A1[1+lda*0] * x_0;*/
/*		y_6 += A1[2+lda*0] * x_0;*/
/*		y_7 += A1[3+lda*0] * x_0;*/

/*		y_0 += A0[0+lda*1] * x_1;*/
/*		y_1 += A0[1+lda*1] * x_1;*/
/*		y_2 += A0[2+lda*1] * x_1;*/
/*		y_3 += A0[3+lda*1] * x_1;*/
/*		y_4 += A1[0+lda*1] * x_1;*/
/*		y_5 += A1[1+lda*1] * x_1;*/
/*		y_6 += A1[2+lda*1] * x_1;*/
/*		y_7 += A1[3+lda*1] * x_1;*/

/*		y_0 += A0[0+lda*2] * x_2;*/
/*		y_1 += A0[1+lda*2] * x_2;*/
/*		y_2 += A0[2+lda*2] * x_2;*/
/*		y_3 += A0[3+lda*2] * x_2;*/
/*		y_4 += A1[0+lda*2] * x_2;*/
/*		y_5 += A1[1+lda*2] * x_2;*/
/*		y_6 += A1[2+lda*2] * x_2;*/
/*		y_7 += A1[3+lda*2] * x_2;*/

/*		y_0 += A0[0+lda*3] * x_3;*/
/*		y_1 += A0[1+lda*3] * x_3;*/
/*		y_2 += A0[2+lda*3] * x_3;*/
/*		y_3 += A0[3+lda*3] * x_3;*/
/*		y_4 += A1[0+lda*3] * x_3;*/
/*		y_5 += A1[1+lda*3] * x_3;*/
/*		y_6 += A1[2+lda*3] * x_3;*/
/*		y_7 += A1[3+lda*3] * x_3;*/
/*		*/
/*		A0 += 4*lda;*/
/*		A1 += 4*lda;*/
/*		x += 4;*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/

/*		y_0 += A0[0+lda*0] * x_0;*/
/*		y_1 += A0[1+lda*0] * x_0;*/
/*		y_2 += A0[2+lda*0] * x_0;*/
/*		y_3 += A0[3+lda*0] * x_0;*/
/*		y_4 += A1[0+lda*0] * x_0;*/
/*		y_5 += A1[1+lda*0] * x_0;*/
/*		y_6 += A1[2+lda*0] * x_0;*/
/*		y_7 += A1[3+lda*0] * x_0;*/

/*		y_0 += A0[0+lda*1] * x_1;*/
/*		y_1 += A0[1+lda*1] * x_1;*/
/*		y_2 += A0[2+lda*1] * x_1;*/
/*		y_3 += A0[3+lda*1] * x_1;*/
/*		y_4 += A1[0+lda*1] * x_1;*/
/*		y_5 += A1[1+lda*1] * x_1;*/
/*		y_6 += A1[2+lda*1] * x_1;*/
/*		y_7 += A1[3+lda*1] * x_1;*/

/*		y_0 += A0[0+lda*2] * x_2;*/
/*		y_1 += A0[1+lda*2] * x_2;*/
/*		y_2 += A0[2+lda*2] * x_2;*/
/*		y_3 += A0[3+lda*2] * x_2;*/
/*		y_4 += A1[0+lda*2] * x_2;*/
/*		y_5 += A1[1+lda*2] * x_2;*/
/*		y_6 += A1[2+lda*2] * x_2;*/
/*		y_7 += A1[3+lda*2] * x_2;*/

/*		y_0 += A0[0+lda*3] * x_3;*/
/*		y_1 += A0[1+lda*3] * x_3;*/
/*		y_2 += A0[2+lda*3] * x_3;*/
/*		y_3 += A0[3+lda*3] * x_3;*/
/*		y_4 += A1[0+lda*3] * x_3;*/
/*		y_5 += A1[1+lda*3] * x_3;*/
/*		y_6 += A1[2+lda*3] * x_3;*/
/*		y_7 += A1[3+lda*3] * x_3;*/
/*		*/
/*		A0 += 4*lda;*/
/*		A1 += 4*lda;*/
/*		x += 4;*/

/*		}*/
/*	*/
/*	y_0 = y[0] - y_0;*/
/*	y_1 = y[1] - y_1;*/
/*	y_2 = y[2] - y_2;*/
/*	y_3 = y[3] - y_3;*/
/*	y_4 = y[4] - y_4;*/
/*	y_5 = y[5] - y_5;*/
/*	y_6 = y[6] - y_6;*/
/*	y_7 = y[7] - y_7;*/

/*	float*/
/*		a_00, a_10, a_20, a_30, a_40, a_50, a_60, a_70,*/
/*		a_11, a_21, a_31, a_41, a_51, a_61, a_71;*/
/*	*/
/*	// A_00*/
/*	a_00 = A0[0+lda*0];*/
/*	a_10 = A0[1+lda*0];*/
/*	a_11 = A0[1+lda*1];*/
/*	y_0 *= a_00;*/
/*	y_1 -= a_10 * y_0;*/
/*	y_1 *= a_11;	*/
/*	y[0] = y_0;*/
/*	y[1] = y_1;*/

/*	a_20 = A0[2+lda*0];*/
/*	a_30 = A0[3+lda*0];*/
/*	a_21 = A0[2+lda*1];*/
/*	a_31 = A0[3+lda*1];*/
/*	y_2 -= a_20 * y_0;*/
/*	y_3 -= a_30 * y_0;*/
/*	y_2 -= a_21 * y_1;*/
/*	y_3 -= a_31 * y_1;*/
/*	*/
/*	a_40 = A1[0+lda*0];*/
/*	a_50 = A1[1+lda*0];*/
/*	a_60 = A1[2+lda*0];*/
/*	a_70 = A1[3+lda*0];*/
/*	a_41 = A1[0+lda*1];*/
/*	a_51 = A1[1+lda*1];*/
/*	a_61 = A1[2+lda*1];*/
/*	a_71 = A1[3+lda*1];*/
/*	y_4 -= a_40 * y_0;*/
/*	y_5 -= a_50 * y_0;*/
/*	y_6 -= a_60 * y_0;*/
/*	y_7 -= a_70 * y_0;*/
/*	y_4 -= a_41 * y_1;*/
/*	y_5 -= a_51 * y_1;*/
/*	y_6 -= a_61 * y_1;*/
/*	y_7 -= a_71 * y_1;*/

/*	// A_11*/
/*	a_00 = A0[2+lda*2];*/
/*	a_10 = A0[3+lda*2];*/
/*	a_11 = A0[3+lda*3];*/
/*	y_2 *= a_00;*/
/*	y_3 -= a_10 * y_2;*/
/*	y_3 *= a_11;	*/
/*	y[2] = y_2;*/
/*	y[3] = y_3;*/

/*	a_40 = A1[0+lda*2];*/
/*	a_50 = A1[1+lda*2];*/
/*	a_60 = A1[2+lda*2];*/
/*	a_70 = A1[3+lda*2];*/
/*	a_41 = A1[0+lda*3];*/
/*	a_51 = A1[1+lda*3];*/
/*	a_61 = A1[2+lda*3];*/
/*	a_71 = A1[3+lda*3];*/
/*	y_4 -= a_40 * y_2;*/
/*	y_5 -= a_50 * y_2;*/
/*	y_6 -= a_60 * y_2;*/
/*	y_7 -= a_70 * y_2;*/
/*	y_4 -= a_41 * y_3;*/
/*	y_5 -= a_51 * y_3;*/
/*	y_6 -= a_61 * y_3;*/
/*	y_7 -= a_71 * y_3;*/

/*	// A_22*/
/*	a_00 = A1[0+lda*4];*/
/*	a_10 = A1[1+lda*4];*/
/*	a_11 = A1[1+lda*5];*/
/*	y_4 *= a_00;*/
/*	y_5 -= a_10 * y_4;*/
/*	y_5 *= a_11;	*/
/*	y[4] = y_4;*/
/*	y[5] = y_5;*/

/*	a_20 = A1[2+lda*4];*/
/*	a_30 = A1[3+lda*4];*/
/*	a_21 = A1[2+lda*5];*/
/*	a_31 = A1[3+lda*5];*/
/*	y_6 -= a_20 * y_4;*/
/*	y_7 -= a_30 * y_4;*/
/*	y_6 -= a_21 * y_5;*/
/*	y_7 -= a_31 * y_5;*/

/*	// A_33*/
/*	a_00 = A1[2+lda*6];*/
/*	a_10 = A1[3+lda*6];*/
/*	a_11 = A1[3+lda*7];*/
/*	y_6 *= a_00;*/
/*	y_7 -= a_10 * y_6;*/
/*	y_7 *= a_11;	*/
/*	y[6] = y_6;*/
/*	y[7] = y_7;*/

/*	}*/



void kernel_strsv_n_4_lib4(int kmax, int ksv, float *A, float *x, float *y)
	{

/*	if(kmax<=0) */
/*		return;*/
	
	__builtin_prefetch( A );
/*	__builtin_prefetch( A1 );*/

	int k_iter = kmax/4;
/*	int k_left = kmax%8;*/

/*printf("\n%d %d\n", k_iter, ksv);*/
/*exit(1);*/

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
/*		"pld    [%1, #64]                \n\t" // prefetch A0 to L1*/
		"pld    [%2, #64]                \n\t" // prefetch A1 to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r0, %0                   \n\t" // k_iter
		"                                \n\t"
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vldr   d24, .DZERO_N_4          \n\t" // load zero double
		"vldr   d25, .DZERO_N_4+8        \n\t" // load zero double
		"vmov   q13, q12                 \n\t"
		"vmov   q14, q12                 \n\t"
		"vmov   q15, q12                 \n\t"
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d0, d1}, [%3:128]!   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%2:128]!   \n\t" // load A0 to registers
/*		"vld1.64   {d16, d17, d18, d19}, [%2:128]!   \n\t" // load A1 to registers*/
		"vld1.64   {d12, d13, d14, d15}, [%2:128]!   \n\t" // load A0 to registers
/*		"vld1.64   {d20, d21, d22, d23}, [%2:128]!   \n\t" // load A1 to registers*/
		"                                \n\t"
		"                                \n\t"
		"ble    .DPOSTACCUM_N_4          \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER_N_4:                \n\t" // main loop
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d2, d3}, [%3:128]!   \n\t" // load x to registers
		"pld    [%2, #64]                \n\t" // prefetch A0 to L1
		"vmla.f32  q12, q4, d0[0]         \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vmla.f32  q13, q5, d0[1]         \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%2:128]!   \n\t" // load A0 to registers
/*		"pld    [%2, #64]                \n\t" // prefetch A1 to L1*/
/*		"vmla.f32  q14, q8, d0[0]         \n\t"*/
/*		"vmla.f32  q15, q9, d0[1]         \n\t"*/
/*		"vld1.64   {d16, d17, d18, d19}, [%2:128]!   \n\t" // load A1 to registers*/
		"vmla.f32  q14, q6, d1[0]         \n\t"
		"cmp    r0, #0                   \n\t" // next iter?
		"vmla.f32  q15, q7, d1[1]         \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%2:128]!   \n\t" // load A0 to registers
/*		"vmla.f32  q14, q10, d1[0]         \n\t"*/
/*		"vmla.f32  q15, q11, d1[1]         \n\t"*/
/*		"vld1.64   {d20, d21, d22, d23}, [%2:128]!   \n\t" // load A1 to registers*/
		"vmov   q0, q1                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"bgt    .DLOOPKITER_N_4          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"vadd.f32  q12, q12, q13         \n\t"
		"vadd.f32  q14, q14, q15         \n\t"
		"vadd.f32  q12, q12, q14         \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM_N_4:                \n\t"
		"                                \n\t"
		"vmov   q2, q4                   \n\t"
		"vmov   q3, q5                   \n\t"
/*		"vmov   q4, q8                   \n\t"*/
/*		"vmov   q5, q9                   \n\t"*/
		"vmov   q8, q6                   \n\t"
		"vmov   q9, q7                   \n\t"
		"                                \n\t"// alg==-1
		"mov    r0, %4                   \n\t" // load address of y
		"                                \n\t"
		"vld1.64   {d0, d1}, [r0]        \n\t" // load y
		"vsub.f32  q0, q0, q12           \n\t"
		"                                \n\t"
		"                                \n\t"
		"mov    r1, %1                   \n\t" // load ksv
		"                                \n\t"
		"cmp    r1, #1                   \n\t" // next iter?
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%2:128]!  \n\t" // load A0*/
		"                                \n\t"
		"vmul.f32  s24, s0, s8           \n\t"
		"vmls.f32  q0, q2, d12[0]        \n\t"
		"                                \n\t"
		"beq    .DKSV1_N_4               \n\t"
		"                                \n\t"
		"cmp    r1, #2                   \n\t" // next iter?
		"                                \n\t"
		"vmul.f32  s25, s1, s13          \n\t"
		"vmls.f32  d1, d7, d12[1]        \n\t"
		"                                \n\t"
		"beq    .DKSV2_N_4               \n\t"
		"                                \n\t"
		"cmp    r1, #3                   \n\t" // next iter?
		"                                \n\t"
/*		"vld1.64   {d4, d5, d6, d7}, [%2:128]!  \n\t" // load A0*/
		"vmov   q2, q8                   \n\t"
		"vmov   q3, q9                   \n\t"
/*		"vmov   q4, q10                  \n\t"*/
/*		"vmov   q5, q11                  \n\t"*/
		"                                \n\t"
		"vmul.f32  s26, s2, s10          \n\t"
		"vmls.f32  s3, s11, s26          \n\t"
		"                                \n\t"
		"beq    .DKSV3_N_4               \n\t"
		"                                \n\t"
		"vmul.f32  s27, s3, s15          \n\t"
		"                                \n\t"
		"                                \n\t"
		"b      .DSTORE_N_4              \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DKSV1_N_4:                     \n\t"
		"                                \n\t"
		"vmov   s25, s1                  \n\t"
		"                                \n\t"
		"                                \n\t"
		".DKSV2_N_4:                     \n\t"
		"                                \n\t"
		"vmov   s26, s2                  \n\t"
		"                                \n\t"
		"                                \n\t"
		".DKSV3_N_4:                     \n\t"
		"                                \n\t"
		"vmov   s27, s3                  \n\t"
		"                                \n\t"
		"                                \n\t"
		".DSTORE_N_4:                    \n\t"
		"                                \n\t"
		"vst1.64   {d12, d13}, [r0]      \n\t" // store y
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO_N_4:                     \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
/*		".DEND:                          \n\t"*/
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (k_iter),		// %0
		  "r" (ksv),		// %1
		  "r" (A),			// %2
		  "r" (x),			// %3
		  "r" (y)			// %4
		: // register clobber list
		  "r0", "r1", "r2",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);
	
	}
	
	
	
/*void kernel_strsv_n_4_lib4_old(int kmax, int ksv, float *A, float *x, float *y)*/
/*	{*/

/*//	if(kmax<=0) */
/*//		return;*/
/*	*/
/*	const int lda = 4;*/
/*	*/
/*	int k;*/

/*	float*/
/*		x_0, x_1, x_2, x_3,*/
/*		y_0=0, y_1=0, y_2=0, y_3=0;*/
/*	*/
/*	k=0;*/
/*	for(; k<kmax-3; k+=4)*/
/*		{*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/

/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[1+lda*0] * x_0;*/
/*		y_2 += A[2+lda*0] * x_0;*/
/*		y_3 += A[3+lda*0] * x_0;*/

/*		y_0 += A[0+lda*1] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[2+lda*1] * x_1;*/
/*		y_3 += A[3+lda*1] * x_1;*/

/*		y_0 += A[0+lda*2] * x_2;*/
/*		y_1 += A[1+lda*2] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/
/*		y_3 += A[3+lda*2] * x_2;*/

/*		y_0 += A[0+lda*3] * x_3;*/
/*		y_1 += A[1+lda*3] * x_3;*/
/*		y_2 += A[2+lda*3] * x_3;*/
/*		y_3 += A[3+lda*3] * x_3;*/
/*		*/
/*		A += 4*lda;*/
/*		x += 4;*/

/*		}*/

/*	y_0 = y[0] - y_0;*/
/*	y_1 = y[1] - y_1;*/
/*	y_2 = y[2] - y_2;*/
/*	y_3 = y[3] - y_3;*/

/*	float*/
/*		a_00, a_10, a_20, a_30,*/
/*		a_11, a_21, a_31;*/
/*	*/
/*	// a_00*/
/*	a_00 = A[0+lda*0];*/
/*	a_10 = A[1+lda*0];*/
/*	a_20 = A[2+lda*0];*/
/*	a_30 = A[3+lda*0];*/
/*	y_0 *= a_00;*/
/*	y[0] = y_0;*/
/*	y_1 -= a_10 * y_0;*/
/*	y_2 -= a_20 * y_0;*/
/*	y_3 -= a_30 * y_0;*/

/*	if(ksv==1)*/
/*		{*/
/*		y[1] = y_1;*/
/*		y[2] = y_2;*/
/*		y[3] = y_3;*/
/*		return;*/
/*		}*/

/*	// a_11*/
/*	a_11 = A[1+lda*1];*/
/*	a_21 = A[2+lda*1];*/
/*	a_31 = A[3+lda*1];*/
/*	y_1 *= a_11;	*/
/*	y[1] = y_1;*/
/*	y_2 -= a_21 * y_1;*/
/*	y_3 -= a_31 * y_1;*/

/*	if(ksv==2)*/
/*		{*/
/*		y[2] = y_2;*/
/*		y[3] = y_3;*/
/*		return;*/
/*		}*/

/*	// a_22*/
/*	a_00 = A[2+lda*2];*/
/*	a_10 = A[3+lda*2];*/
/*	y_2 *= a_00;*/
/*	y[2] = y_2;*/
/*	y_3 -= a_10 * y_2;*/

/*	if(ksv==3)*/
/*		{*/
/*		y[3] = y_3;*/
/*		return;*/
/*		}*/

/*	// a_33*/
/*	a_11 = A[3+lda*3];*/
/*	y_3 *= a_11;	*/
/*	y[3] = y_3;*/

/*	}*/
	
	
	
void kernel_strsv_t_4_lib4(int kmax, float *A, int sda, float *x)
	{

	if(kmax<=0) 
		return;
	
	const int bs = 4;

/*	__builtin_prefetch( A );*/

	int incA = bs*(sda-4)*sizeof(float);

	kmax -= 4;
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"mov    r1, %0                   \n\t" // prefetch offset
		"add    r1, r1, #64              \n\t"
		"                                \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r2, %3                   \n\t" // backup A
		"mov    r3, %4                   \n\t" // backup x
		"                                \n\t"
		"                                \n\t"
		"add    %3, %3, r1               \n\t" // to next block
		"add    %4, %4, #16              \n\t"
		"                                \n\t"
		"                                \n\t"
		"mov    r0, %1                   \n\t" // k_loop
		"cmp    r0, #1                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vldr   d16, .DZERO_T_4          \n\t" // load zero double
		"vldr   d17, .DZERO_T_4+8        \n\t" // load zero double
		"vmov   q9, q8                   \n\t"
		"vmov   q10, q8                  \n\t"
		"vmov   q11, q8                  \n\t"
		"vmov   q2, q8                   \n\t" // zero vector
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d0, d1, d2, d3}, [%4:128]!   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vld1.64   {d12, d13, d14, d15}, [%3:128]!   \n\t" // load A0 to registers
		"                                \n\t"
		"                                \n\t"
		"beq    .DMAIN_LOOP_T_4          \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_4    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP2_T_4:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"vmla.f32  q11, q7, q0           \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%3:128]!   \n\t" // load A0 to registers
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"sub    r0, r0, #2               \n\t" // iter++
		"vmla.f32  q11, q7, q0           \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #1                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP2_T_4          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_4    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP_T_4:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vmla.f32  q11, q7, q0           \n\t"
		"vld1.64   {d12, d13, d14, d15}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #0                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP_T_4          \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONS_CLEAN_LOOP_T_4:          \n\t" // main loop
		"                                \n\t"
		"mov    r0, %2                   \n\t" // k_left
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"ble    .DPOSTACCUM_T_4          \n\t"
		"                                \n\t"
		"vmov   s3, s8                   \n\t"
		"cmp    r0, #2                   \n\t"
		"vmoveq s2, s8                   \n\t"
		"vmovlt s1, s8                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q8, q4, q0            \n\t"
		"vmla.f32  q9, q5, q0            \n\t"
		"vmla.f32  q10, q6, q0           \n\t"
		"vmla.f32  q11, q7, q0           \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM_T_4:                \n\t"
		"                                \n\t"
		"                                \n\t"
		"pld    [r2, #0]                 \n\t" // prefetch A to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r0, r3                   \n\t" // load address of x[0]
		"vld1.64   {d2, d3}, [r0:128]    \n\t" // load x[0]
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d8, d9, d10, d11}, [r2:128]!   \n\t" // load A to registers
		"vld1.64   {d12, d13, d14, d15}, [r2:128]! \n\t" // load A to registers
		"                                \n\t"
		"                                \n\t"
		// bottom trinagle
		"vpadd.f32 d6, d20, d21          \n\t"
		"vpadd.f32 d7, d22, d23          \n\t"
		"                                \n\t"
		"vpadd.f32 d1, d6, d7            \n\t"
		"                                \n\t"
		"vsub.f32  d1, d3, d1            \n\t"
		"                                \n\t"
		"vmul.f32  s3, s31, s3           \n\t"
		"                                \n\t"
		"vmls.f32  s2, s27, s3           \n\t"
		"vmul.f32  s2, s26, s2           \n\t"
		"                                \n\t"
		"                                \n\t"
		// square
		"vmla.f32  d16, d9, d1           \n\t"
		"vmla.f32  d18, d11, d1          \n\t"
		"                                \n\t"
		"                                \n\t"
		// top trinagle
		"vpadd.f32 d4, d16, d17          \n\t"
		"vpadd.f32 d5, d18, d19          \n\t"
		"                                \n\t"
		"vpadd.f32 d0, d4, d5            \n\t"
		"                                \n\t"
		"vsub.f32  d0, d2, d0            \n\t"
		"                                \n\t"
		"vmul.f32  s1, s21, s1           \n\t"
		"                                \n\t"
		"vmls.f32  s0, s17, s1           \n\t"
		"vmul.f32  s0, s16, s0           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"mov    r0, r3                   \n\t" // load address of x[0] */
		"vst1.64   {d0, d1}, [r0:128]    \n\t" // store x[0]
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO_T_4:                     \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (incA),		// %0
		  "r" (k_iter),		// %1
		  "r" (k_left),		// %2
		  "r" (A),			// %3
		  "r" (x)			// %4
		: // register clobber list
		  "r0", "r1", "r2", "r3",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);

	}



/*void kernel_strsv_t_4_lib4_old(int kmax, float *A, int sda, float *x)*/
/*	{*/

/*	if(kmax<=0) */
/*		return;*/
/*	*/
/*	const int lda = 4;*/
/*	const int bs  = 4;*/
/*	*/
/*	int*/
/*		k;*/
/*	*/
/*	float *tA, *tx;*/
/*	tA = A;*/
/*	tx = x;*/

/*	float*/
/*		x_0, x_1, x_2, x_3,*/
/*		y_0=0, y_1=0, y_2=0, y_3=0;*/
/*	*/
/*	k=4;*/
/*	A += 4 + (sda-1)*lda;*/
/*	x += 4;*/
/*	for(; k<kmax-7; k+=8)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		y_3 += A[0+lda*3] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		y_3 += A[1+lda*3] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/
/*		y_3 += A[2+lda*3] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		y_3 += A[3+lda*3] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		y_3 += A[0+lda*3] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		y_3 += A[1+lda*3] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/
/*		y_3 += A[2+lda*3] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		y_3 += A[3+lda*3] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		y_3 += A[0+lda*3] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		y_3 += A[1+lda*3] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/
/*		y_3 += A[2+lda*3] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		y_3 += A[3+lda*3] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax; k++)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		y_3 += A[0+lda*3] * x_0;*/
/*		*/
/*		A += 1;//sda*bs;*/
/*		x += 1;*/

/*		}*/
/*	*/
/*	A = tA;*/
/*	x = tx;*/

/*	// bottom trinagle*/
/*	y_3  = x[3] - y_3;*/
/*	y_3 *= A[3+lda*3];*/
/*	x[3] = y_3;*/

/*	y_2  = x[2] - A[3+lda*2] * y_3 - y_2;*/
/*	y_2 *= A[2+lda*2];*/
/*	x[2] = y_2;*/

/*	// square*/
/*	y_0 += A[2+lda*0]*y_2 + A[3+lda*0]*y_3;*/
/*	y_1 += A[2+lda*1]*y_2 + A[3+lda*1]*y_3;*/
/*		*/
/*	// top trinagle*/
/*	y_1  = x[1] - y_1;*/
/*	y_1 *= A[1+lda*1];*/
/*	x[1] = y_1;*/

/*	y_0  = x[0] - A[1+lda*0] * y_1 - y_0;*/
/*	y_0 *= A[0+lda*0];*/
/*	x[0] = y_0;*/

/*	}*/
	
	
	
void kernel_strsv_t_3_lib4(int kmax, float *A, int sda, float *x)
	{

	if(kmax<=0) 
		return;
	
	const int bs = 4;

/*	__builtin_prefetch( A );*/

	int incA = bs*(sda-3)*sizeof(float);

	kmax -= 4;
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"mov    r1, %0                   \n\t" // prefetch offset
		"add    r1, r1, #48              \n\t"
		"                                \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"                                \n\t"
		"                                \n\t"
		"vldr   d16, .DZERO_T_3          \n\t" // load zero double
		"vldr   d17, .DZERO_T_3+8        \n\t" // load zero double
		"vmov   q9, q8                   \n\t"
		"vmov   q10, q8                  \n\t"
		"vmov   q2, q8                   \n\t" // zero vector
		"                                \n\t"
		"                                \n\t"
		"mov    r2, %3                   \n\t" // backup A
		"mov    r3, %4                   \n\t" // backup x
		"                                \n\t"
		"                                \n\t"
		// cleanup at the beginning
		"vld1.64   {d0, d1}, [%4:128]    \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vld1.64   {d12, d13}, [%3:128]! \n\t" // load A0 to registers
		"vmov      s2, s8                \n\t" // zeros x[2]
		"vmla.f32  d16, d9, d1           \n\t"
		"vmla.f32  d18, d11, d1          \n\t"
		"vmla.f32  d20, d13, d1          \n\t"
		"                                \n\t"
		"mov    %3, r2                   \n\t" // restore A
		"                                \n\t"
		"                                \n\t"
		"add    %3, %3, r1               \n\t" // to next block
		"add    %4, %4, #16              \n\t"
		"                                \n\t"
		"                                \n\t"
		"mov    r0, %1                   \n\t" // k_loop
		"cmp    r0, #1                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d0, d1, d2, d3}, [%4:128]!   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vld1.64   {d12, d13}, [%3:128]!   \n\t" // load A0 to registers
		"                                \n\t"
		"                                \n\t"
		"beq    .DMAIN_LOOP_T_3          \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_3    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP2_T_3:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"vld1.64   {d12, d13}, [%3:128]!   \n\t" // load A0 to registers
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"sub    r0, r0, #2               \n\t" // iter++
		"vld1.64   {d12, d13}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #1                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP2_T_3          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_3    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP_T_3:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmla.f32  q10, q6, q0           \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vld1.64   {d12, d13}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #0                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP_T_3          \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONS_CLEAN_LOOP_T_3:          \n\t" // main loop
		"                                \n\t"
		"mov    r0, %2                   \n\t" // k_left
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"ble    .DPOSTACCUM_T_3          \n\t"
		"                                \n\t"
		"vmov   s3, s8                   \n\t"
		"cmp    r0, #2                   \n\t"
		"vmoveq s2, s8                   \n\t"
		"vmovlt s1, s8                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q8, q4, q0            \n\t"
		"vmla.f32  q9, q5, q0            \n\t"
		"vmla.f32  q10, q6, q0           \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM_T_3:                \n\t"
		"                                \n\t"
		"                                \n\t"
		"pld    [r2, #0]                 \n\t" // prefetch A to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r0, r3                   \n\t" // load address of x[0]
		"vld1.64   {d2, d3}, [r0:128]    \n\t" // load x[0]
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d8, d9, d10, d11}, [r2:128]!   \n\t" // load A to registers
		"vld1.64   {d12, d13}, [r2:128]! \n\t" // load A to registers
		"                                \n\t"
		"                                \n\t"
		// bottom trinagle
		"vpadd.f32 d6, d20, d21          \n\t"
		"                                \n\t"
		"vadd.f32  s2, s12, s13          \n\t"
		"                                \n\t"
		"vsub.f32  s2, s6, s2            \n\t"
		"                                \n\t"
		"vmul.f32  s2, s26, s2           \n\t"
		"                                \n\t"
		"                                \n\t"
		// top trinagle & square
		"vpadd.f32 d4, d16, d17          \n\t"
		"vpadd.f32 d5, d18, d19          \n\t"
		"                                \n\t"
		"vpadd.f32 d0, d4, d5            \n\t"
		"                                \n\t"
		"vmla.f32  s0, s18, s2           \n\t"
		"vmla.f32  s1, s22, s2          \n\t"
		"                                \n\t"
		"vsub.f32  d0, d2, d0            \n\t"
		"                                \n\t"
		"vmul.f32  s1, s21, s1           \n\t"
		"                                \n\t"
		"vmls.f32  s0, s17, s1           \n\t"
		"vmul.f32  s0, s16, s0           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"mov    r0, r3                   \n\t" // load address of x[0] */
		"vmov      s3, s7                \n\t" // restore x[3]
		"vst1.64   {d0, d1}, [r0:128]    \n\t" // store x[0]
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO_T_3:                     \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (incA),		// %0
		  "r" (k_iter),		// %1
		  "r" (k_left),		// %2
		  "r" (A),			// %3
		  "r" (x)			// %4
		: // register clobber list
		  "r0", "r1", "r2", "r3",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);

	}



/*void kernel_strsv_t_3_lib4_old(int kmax, float *A, int sda, float *x)*/
/*	{*/

/*	if(kmax<=0) */
/*		return;*/
/*	*/
/*	const int lda = 4;*/
/*	const int bs  = 4;*/
/*	*/
/*	int*/
/*		k;*/
/*	*/
/*	float *tA, *tx;*/
/*	tA = A;*/
/*	tx = x;*/

/*	float*/
/*		x_0, x_1, x_2, x_3,*/
/*		y_0=0, y_1=0, y_2=0;*/
/*	*/
/*	// clean up at the beginning*/
/*	x_3 = x[3];*/

/*	y_0 += A[3+lda*0] * x_3;*/
/*	y_1 += A[3+lda*1] * x_3;*/
/*	y_2 += A[3+lda*2] * x_3;*/

/*	k=4;*/
/*	A += 4 + (sda-1)*lda;*/
/*	x += 4;*/
/*	for(; k<kmax-7; k+=8)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		y_2 += A[1+lda*2] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/
/*		y_2 += A[2+lda*2] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		y_2 += A[3+lda*2] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax; k++)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		*/
/*		A += 1;//sda*bs;*/
/*		x += 1;*/

/*		}*/

/*	A = tA;*/
/*	x = tx;*/

/*	// bottom trinagle*/
/*	y_2  = x[2] - y_2;*/
/*	y_2 *= A[2+lda*2];*/
/*	x[2] = y_2;*/

/*	// square*/
/*	y_0 += A[2+lda*0]*y_2;*/
/*	y_1 += A[2+lda*1]*y_2;*/
/*		*/
/*	// top trinagle*/
/*	y_1  = x[1] - y_1;*/
/*	y_1 *= A[1+lda*1];*/
/*	x[1] = y_1;*/

/*	y_0  = x[0] - A[1+lda*0] * y_1 - y_0;*/
/*	y_0 *= A[0+lda*0];*/
/*	x[0] = y_0;*/

/*	}*/
	
	
	
void kernel_strsv_t_2_lib4(int kmax, float *A, int sda, float *x)
	{

	if(kmax<=0) 
		return;
	
	const int bs = 4;

	__builtin_prefetch( A );

	int incA = bs*(sda-2)*sizeof(float);

	kmax -= 4;
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"mov    r1, %0                   \n\t" // prefetch offset
		"add    r1, r1, #32              \n\t"
		"                                \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"                                \n\t"
		"                                \n\t"
		"vldr   d16, .DZERO_T_2          \n\t" // load zero double
		"vldr   d17, .DZERO_T_2+8        \n\t" // load zero double
		"vmov   q9, q8                   \n\t"
		"vmov   q2, q8                   \n\t" // zero vector
		"                                \n\t"
		"                                \n\t"
		"mov    r2, %3                   \n\t" // backup A
		"mov    r3, %4                   \n\t" // backup x
		"                                \n\t"
		"                                \n\t"
		// clean up at the beginning
		"vld1.64   {d0, d1}, [%4:128]   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%3:128]   \n\t" // load A0 to registers
		"                                \n\t"
		"vmla.f32  d16, d9, d1           \n\t"
		"vmla.f32  d18, d11, d1           \n\t"
		"                                \n\t"
		"                                \n\t"
		"add    %3, %3, r1               \n\t" // to next block
		"add    %4, %4, #16              \n\t"
		"                                \n\t"
		"                                \n\t"
		"mov    r0, %1                   \n\t" // k_loop
		"cmp    r0, #1                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d0, d1, d2, d3}, [%4:128]!   \n\t" // load x to registers
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"                                \n\t"
		"                                \n\t"
		"beq    .DMAIN_LOOP_T_2          \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_2    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP2_T_2:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"sub    r0, r0, #2               \n\t" // iter++
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #1                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP2_T_2          \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"blt    .DCONS_CLEAN_LOOP_T_2    \n\t"
		"                                \n\t"
		"                                \n\t"
		".DMAIN_LOOP_T_2:                \n\t" // main loop
		"                                \n\t"
		"add    %3, %3, %0               \n\t" // next band
		"vmla.f32  q8, q4, q0           \n\t"
		"pld    [%3, r1]                 \n\t" // prefetch A1 to L1
		"vmla.f32  q9, q5, q0           \n\t"
		"sub    r0, r0, #1               \n\t" // iter++
		"vld1.64   {d8, d9, d10, d11}, [%3:128]!   \n\t" // load A0 to registers
		"cmp    r0, #0                   \n\t" // next iter?
		"vmov   q0, q1                   \n\t"
		"vld1.64   {d2, d3}, [%4:128]!   \n\t" // load x to registers
		"                                \n\t"
		"bgt    .DMAIN_LOOP_T_2          \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONS_CLEAN_LOOP_T_2:          \n\t" // main loop
		"                                \n\t"
		"mov    r0, %2                   \n\t" // k_left
		"cmp    r0, #0                   \n\t"
		"                                \n\t"
		"ble    .DPOSTACCUM_T_2          \n\t"
		"                                \n\t"
		"vmov   s3, s8                   \n\t"
		"cmp    r0, #2                   \n\t"
		"vmoveq s2, s8                   \n\t"
		"vmovlt s1, s8                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"vmla.f32  q8, q4, q0            \n\t"
		"vmla.f32  q9, q5, q0            \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM_T_2:                \n\t"
		"                                \n\t"
		"                                \n\t"
		"pld    [r2, #0]                 \n\t" // prefetch A to L1
		"                                \n\t"
		"                                \n\t"
		"mov    r0, r3                   \n\t" // load address of x[0]
		"vld1.64   {d2}, [r0:64]         \n\t" // load x[0]
		"                                \n\t"
		"                                \n\t"
		"vld1.64   {d8, d9, d10, d11}, [r2:128]!   \n\t" // load A to registers
		"                                \n\t"
		"                                \n\t"
		// top trinagle
		"vpadd.f32 d4, d16, d17          \n\t"
		"vpadd.f32 d5, d18, d19          \n\t"
		"                                \n\t"
		"vpadd.f32 d0, d4, d5            \n\t"
		"                                \n\t"
		"vsub.f32  d0, d2, d0            \n\t"
		"                                \n\t"
		"vmul.f32  s1, s21, s1           \n\t"
		"                                \n\t"
		"vmls.f32  s0, s17, s1           \n\t"
		"vmul.f32  s0, s16, s0           \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
/*		"mov    r0, r3                   \n\t" // load address of x[0] */
		"vst1.64   {d0}, [r0:64]         \n\t" // store x[0]
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".align 3                        \n\t"
		".DZERO_T_2:                     \n\t" // zero quad word
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		".word  0                        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (incA),		// %0
		  "r" (k_iter),		// %1
		  "r" (k_left),		// %2
		  "r" (A),			// %3
		  "r" (x)			// %4
		: // register clobber list
		  "r0", "r1", "r2", "r3",
		  "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
		  "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
		  "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
		  "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
		  "memory"
	);

	}



/*void kernel_strsv_t_2_lib4_old(int kmax, float *A, int sda, float *x)*/
/*	{*/

/*	if(kmax<=0) */
/*		return;*/
/*	*/
/*	const int lda = 4;*/
/*	const int bs  = 4;*/
/*	*/
/*	int*/
/*		k;*/
/*	*/
/*	float *tA, *tx;*/
/*	tA = A;*/
/*	tx = x;*/

/*	float*/
/*		x_0, x_1, x_2, x_3,*/
/*		y_0=0, y_1=0;*/
/*	*/
/*	// clean up at the beginning*/
/*	x_2 = x[2];*/
/*	x_3 = x[3];*/

/*	y_0 += A[2+lda*0] * x_2;*/
/*	y_1 += A[2+lda*1] * x_2;*/

/*	y_0 += A[3+lda*0] * x_3;*/
/*	y_1 += A[3+lda*1] * x_3;*/

/*	k=4;*/
/*	A += 4 + (sda-1)*lda;*/
/*	x += 4;*/
/*	for(; k<kmax-7; k+=8)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax-3; k+=4)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
/*		x_2 = x[2];*/
/*		x_3 = x[3];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/

/*		y_0 += A[1+lda*0] * x_1;*/
/*		y_1 += A[1+lda*1] * x_1;*/
/*		*/
/*		y_0 += A[2+lda*0] * x_2;*/
/*		y_1 += A[2+lda*1] * x_2;*/

/*		y_0 += A[3+lda*0] * x_3;*/
/*		y_1 += A[3+lda*1] * x_3;*/
/*		*/
/*		A += sda*bs;*/
/*		x += 4;*/

/*		}*/
/*	for(; k<kmax; k++)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		*/
/*		A += 1;//sda*bs;*/
/*		x += 1;*/

/*		}*/

/*	A = tA;*/
/*	x = tx;*/

/*	// top trinagle*/
/*	y_1  = x[1] - y_1;*/
/*	y_1 *= A[1+lda*1];*/
/*	x[1] = y_1;*/

/*	y_0  = x[0] - A[1+lda*0] * y_1 - y_0;*/
/*	y_0 *= A[0+lda*0];*/
/*	x[0] = y_0;*/

/*	}*/
	
	
	
void kernel_strsv_t_1_lib4(int kmax, float *A, int sda, float *x)
	{

	if(kmax<=0) 
		return;
	
	const int lda = 4;
	const int bs  = 4;
	
	int
		k;
	
	float *tA, *tx;
	tA = A;
	tx = x;

	float
		x_0, x_1, x_2, x_3,
		y_0=0;
	
	// clean up at the beginning
	x_1 = x[1];
	x_2 = x[2];
	x_3 = x[3];

	y_0 += A[1+lda*0] * x_1;
	y_0 += A[2+lda*0] * x_2;
	y_0 += A[3+lda*0] * x_3;

	k=4;
	A += 4 + (sda-1)*lda;
	x += 4;
	for(; k<kmax-7; k+=8)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+lda*0] * x_0;
		y_0 += A[1+lda*0] * x_1;
		y_0 += A[2+lda*0] * x_2;
		y_0 += A[3+lda*0] * x_3;
		
		A += sda*bs;
		x += 4;

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+lda*0] * x_0;
		y_0 += A[1+lda*0] * x_1;
		y_0 += A[2+lda*0] * x_2;
		y_0 += A[3+lda*0] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	for(; k<kmax-3; k+=4)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];
		
		y_0 += A[0+lda*0] * x_0;
		y_0 += A[1+lda*0] * x_1;
		y_0 += A[2+lda*0] * x_2;
		y_0 += A[3+lda*0] * x_3;
		
		A += sda*bs;
		x += 4;

		}
	for(; k<kmax; k++)
		{
		
		x_0 = x[0];
		
		y_0 += A[0+lda*0] * x_0;
		
		A += 1;//sda*bs;
		x += 1;

		}

	A = tA;
	x = tx;

	// top trinagle
	y_0  = x[0] - y_0;
	y_0 *= A[0+lda*0];
	x[0] = y_0;

	}
	
	
	

