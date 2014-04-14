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

void kernel_dgemm_pp_nt_4x4_ppc_lib4(int kmax, double *A, double *B, double *C, int bs_dummy, int alg)
	{
	
	if(kmax<=0)
		return;
	
/*	__builtin_prefetch( A );*/
/*	__builtin_prefetch( B );*/
/*	__builtin_prefetch( A+4 );*/
/*	__builtin_prefetch( B+4 );*/

	__asm__ volatile
	(
		"                                \n\t"
		"li     5,0                    \n\t"
		"dcbt   %0,5                    \n\t" // prefetch A
		"dcbt   %1,5                    \n\t" // prefetch A
		"                                \n\t"
		"li     5,32                    \n\t"
		"dcbt   %0,5                    \n\t"
		"dcbt   %1,5                    \n\t"
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "r" (A),			// %0
		  "r" (B)			// %1
		: // register clobber list
		  "r5",
		  "memory"
	);


	int k_iter = kmax/4;
	int k_left = kmax%4;
	
	double zero[] = {0.0};
	
//	printf("\n%d %d %d\n", kmax, k_iter, k_left);

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"lfd    16,0(%2)                  \n\t" // load A_even
		"lfd    17,8(%2)                  \n\t"
		"lfd    18,16(%2)                 \n\t"
		"lfd    19,24(%2)                 \n\t"
		"                                \n\t"
		"lfd    20,0(%3)                  \n\t" // load B_even
		"lfd    21,8(%3)                  \n\t"
		"lfd    22,16(%3)                 \n\t"
		"lfd    23,24(%3)                 \n\t"
		"                                \n\t"
		"lfd    24,32(%2)                 \n\t" // load A_odd
		"lfd    25,40(%2)                 \n\t"
		"lfd    26,48(%2)                 \n\t"
		"lfd    27,56(%2)                 \n\t"
		"                                \n\t"
		"lfd    28,32(%3)                 \n\t" // load B_odd
		"lfd    29,40(%3)                 \n\t"
		"lfd    30,48(%3)                 \n\t"
		"lfd    31,56(%3)                 \n\t"
		"                                \n\t"
		"                                \n\t"
		"li     5,64                    \n\t"
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"                                \n\t"
		"li     5,96                    \n\t"
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"lfd    0,0(%6)                  \n\t" // load zero in c_00
		"fmr    1,0                      \n\t" //c_10
		"fmr    2,0                      \n\t" //c_20
		"fmr    3,0                      \n\t" //c_30
		"                                \n\t"
		"fmr    4,0                      \n\t" //c_01
		"fmr    5,0                      \n\t" //c_11
		"fmr    6,0                      \n\t" //c_21
		"fmr    7,0                      \n\t" //c_31
		"                                \n\t"
		"fmr    8,0                      \n\t" //c_02
		"fmr    9,0                      \n\t" //c_12
		"fmr    10,0                     \n\t" //c_22
		"fmr    11,0                     \n\t" //c_32
		"                                \n\t"
		"fmr    12,0                     \n\t" //c_03
		"fmr    13,0                     \n\t" //c_13
		"fmr    14,0                     \n\t" //c_23
		"fmr    15,0                     \n\t" //c_33
		"                                \n\t"
		"                                \n\t"
		"li     5,128                   \n\t"
		"                                \n\t"
		"                                \n\t"
		"cmpwi  0,%0,0                   \n\t" // (%0) < 0 ???
		"ble-   0,.DCONSIDERLEFT         \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKITER:                    \n\t" // main loop
		"                                \n\t"
		"                                \n\t" // k = 0
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"li     5,160                   \n\t"
		"                                \n\t" // k = 0
		"                                \n\t" // k = 0
		"fmadd  0,16,20,0                \n\t" // (0) = (16) * (20) + (0)
		"fmadd  1,17,20,1                \n\t"
		"fmadd  2,18,20,2                \n\t"
		"fmadd  3,19,20,3                \n\t"
		"lfd    20,64(%3)                \n\t" // load B_0_even
		"                                \n\t"
		"addic.   %0,%0,-1               \n\t"
		"                                \n\t"
		"fmadd  4,16,21,4                \n\t"
		"fmadd  5,17,21,5                \n\t"
		"fmadd  6,18,21,6                \n\t"
		"fmadd  7,19,21,7                \n\t"
		"lfd    21,72(%3)                 \n\t" // load B_1_even
		"                                \n\t"
		"cmpwi  0,%0,0                   \n\t" // (%0) < 0 ???
		"                                \n\t"
		"fmadd  8,16,22,8                \n\t"
		"fmadd  9,17,22,9                \n\t"
		"fmadd  10,18,22,10              \n\t"
		"fmadd  11,19,22,11              \n\t"
		"lfd    22,80(%3)                 \n\t" // load B_2_even
		"                                \n\t"
		"fmadd  12,16,23,12              \n\t"
		"lfd    16,64(%2)                 \n\t" // load A_0_even
		"fmadd  13,17,23,13              \n\t"
		"lfd    17,72(%2)                 \n\t" // load A_1_even
		"fmadd  14,18,23,14              \n\t"
		"lfd    18,80(%2)                 \n\t" // load A_2_even
		"fmadd  15,19,23,15              \n\t"
		"lfd    23,88(%3)                 \n\t" // load B_3_even
		"                                \n\t"
		"                                \n\t"
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"li     5,192                   \n\t"
		"                                \n\t" // k = 1
		"                                \n\t" // k = 1
		"fmadd  0,24,28,0                \n\t" // (0) = (16) * (20) + (0)
		"lfd    19,88(%2)                 \n\t" // load A_3_even
		"fmadd  1,25,28,1                \n\t"
		"fmadd  2,26,28,2                \n\t"
		"fmadd  3,27,28,3                \n\t"
		"lfd    28,96(%3)                 \n\t" // load B_0_odd
		"                                \n\t"
		"fmadd  4,24,29,4                \n\t"
		"fmadd  5,25,29,5                \n\t"
		"fmadd  6,26,29,6                \n\t"
		"fmadd  7,27,29,7                \n\t"
		"lfd    29,104(%3)                \n\t" // load B_1_odd
		"                                \n\t"
		"fmadd  8,24,30,8                \n\t"
		"fmadd  9,25,30,9                \n\t"
		"fmadd  10,26,30,10              \n\t"
		"fmadd  11,27,30,11              \n\t"
		"lfd    30,112(%3)                \n\t" // load B_2_odd
		"                                \n\t"
		"fmadd  12,24,31,12              \n\t"
		"lfd    24,96(%2)                 \n\t" // load A_0_odd
		"fmadd  13,25,31,13              \n\t"
		"lfd    25,104(%2)                 \n\t" // load A_1_odd
		"fmadd  14,26,31,14              \n\t"
		"lfd    26,112(%2)                 \n\t" // load A_2_odd
		"fmadd  15,27,31,15              \n\t"
		"lfd    31,120(%3)                \n\t" // load B_3_odd
		"                                \n\t"
		"                                \n\t"
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"li     5,224                   \n\t"
		"                                \n\t" // k = 1
		"                                \n\t" // k = 2
		"fmadd  0,16,20,0                \n\t" // (0) = (16) * (20) + (0)
		"lfd    27,120(%2)                \n\t" // load A_3_odd
		"fmadd  1,17,20,1                \n\t"
		"fmadd  2,18,20,2                \n\t"
		"fmadd  3,19,20,3                \n\t"
		"lfd    20,128(%3)                \n\t" // load B_0_even
		"                                \n\t"
		"fmadd  4,16,21,4                \n\t"
		"fmadd  5,17,21,5                \n\t"
		"fmadd  6,18,21,6                \n\t"
		"fmadd  7,19,21,7                \n\t"
		"lfd    21,136(%3)                \n\t" // load B_1_even
		"                                \n\t"
		"fmadd  8,16,22,8                \n\t"
		"fmadd  9,17,22,9                \n\t"
		"fmadd  10,18,22,10              \n\t"
		"fmadd  11,19,22,11              \n\t"
		"lfd    22,144(%3)                \n\t" // load B_2_even
		"                                \n\t"
		"fmadd  12,16,23,12              \n\t"
		"lfd    16,128(%2)                \n\t" // load A_0_even
		"fmadd  13,17,23,13              \n\t"
		"lfd    17,136(%2)                \n\t" // load A_1_even
		"fmadd  14,18,23,14              \n\t"
		"lfd    18,144(%2)                \n\t" // load A_2_even
		"fmadd  15,19,23,15              \n\t"
		"lfd    23,152(%3)                \n\t" // load B_3_even
		"                                \n\t"
		"                                \n\t"
		"dcbt   %2,5                    \n\t"
		"dcbt   %3,5                    \n\t"
		"li     5,128                   \n\t"
		"                                \n\t" // k = 1
		"                                \n\t" // k = 3
		"fmadd  0,24,28,0                \n\t" // (0) = (16) * (20) + (0)
		"addi   %3,%3,128                  \n\t" // B += 16
		"fmadd  1,25,28,1                \n\t"
		"lfd    19,152(%2)                \n\t" // load A_3_even
		"fmadd  2,26,28,2                \n\t"
		"addi   %2,%2,128                  \n\t" // A += 16
		"fmadd  3,27,28,3                \n\t"
		"lfd    28,32(%3)                 \n\t" // load B_0_odd
		"                                \n\t"
		"fmadd  4,24,29,4                \n\t"
		"fmadd  5,25,29,5                \n\t"
		"fmadd  6,26,29,6                \n\t"
		"fmadd  7,27,29,7                \n\t"
		"lfd    29,40(%3)                 \n\t" // load B_1_odd
		"                                \n\t"
		"fmadd  8,24,30,8                \n\t"
		"fmadd  9,25,30,9                \n\t"
		"fmadd  10,26,30,10              \n\t"
		"fmadd  11,27,30,11              \n\t"
		"lfd    30,48(%3)                 \n\t" // load B_2_odd
		"                                \n\t"
		"fmadd  12,24,31,12              \n\t"
		"lfd    24,32(%2)                 \n\t" // load A_0_odd
		"fmadd  13,25,31,13              \n\t"
		"lfd    25,40(%2)                 \n\t" // load A_1_odd
		"fmadd  14,26,31,14              \n\t"
		"lfd    26,48(%2)                 \n\t" // load A_2_odd
		"fmadd  15,27,31,15              \n\t"
		"lfd    31,56(%3)                 \n\t" // load B_3_odd
		"lfd    27,56(%2)                 \n\t" // load A_3_odd
		"                                \n\t"
		"bgt+   0,.DLOOPKITER            \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDERLEFT:                 \n\t" // consider left
		"                                \n\t"
		"                                \n\t"
		"cmpwi  0,%1,0                   \n\t" // (%0) < 0 ???
		"ble+   0,.DPOSTACCUM            \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT:                    \n\t" // clean up loop
		"                                \n\t"
		"                                \n\t"
		"fmadd  0,16,20,0                \n\t" // (0) = (16) * (20) + (0)
		"fmadd  1,17,20,1                \n\t"
		"fmadd  2,18,20,2                \n\t"
		"fmadd  3,19,20,3                \n\t"
		"lfd    20,32(%3)                 \n\t" // load B_0_even
		"                                \n\t"
		"addic.   %1,%1,-1               \n\t"
		"                                \n\t"
		"fmadd  4,16,21,4                \n\t"
		"fmadd  5,17,21,5                \n\t"
		"fmadd  6,18,21,6                \n\t"
		"fmadd  7,19,21,7                \n\t"
		"lfd    21,40(%3)                 \n\t" // load B_1_even
		"                                \n\t"
		"cmpwi  0,%1,0                   \n\t" // (%0) < 0 ???
		"                                \n\t"
		"fmadd  8,16,22,8                \n\t"
		"fmadd  9,17,22,9                \n\t"
		"fmadd  10,18,22,10              \n\t"
		"fmadd  11,19,22,11              \n\t"
		"lfd    22,48(%3)                 \n\t" // load B_2_even
		"                                \n\t"
		"fmadd  12,16,23,12              \n\t"
		"lfd    16,32(%2)                 \n\t" // load A_0_even
		"fmadd  13,17,23,13              \n\t"
		"lfd    17,40(%2)                 \n\t" // load A_1_even
		"fmadd  14,18,23,14              \n\t"
		"lfd    18,48(%2)                 \n\t" // load A_2_even
		"fmadd  15,19,23,15              \n\t"
		"lfd    23,56(%3)                 \n\t" // load B_3_even
		"addi   %3,%3,32                   \n\t" // B += 4
		"lfd    19,56(%2)                 \n\t" // load A_3_even
		"addi   %2,%2,32                   \n\t" // A += 4
		"                                \n\t"
		"bgt+   0,.DLOOPKLEFT            \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM:                    \n\t"
		"                                \n\t"
		"cmpwi  0,%5,0                   \n\t" // alg==0 ?
		"beq    0,.DSTORE                \n\t"
		"                                \n\t"
		"cmpwi  0,%5,1                   \n\t" // alg==1 ?
		"                                \n\t"
		"mr     6,%4                     \n\t" // load address of C
		"                                \n\t"
		"lfd    16,0(6)                  \n\t" // load elements of C
		"lfd    17,8(6)                  \n\t"
		"lfd    18,16(6)                 \n\t"
		"lfd    19,24(6)                 \n\t"
		"                                \n\t"
		"lfd    20,32(6)                 \n\t"
		"lfd    21,40(6)                 \n\t"
		"lfd    22,48(6)                 \n\t"
		"lfd    23,56(6)                 \n\t"
		"                                \n\t"
		"lfd    24,64(6)                 \n\t"
		"lfd    25,72(6)                 \n\t"
		"lfd    26,80(6)                 \n\t"
		"lfd    27,88(6)                 \n\t"
		"                                \n\t"
		"lfd    28,96(6)                 \n\t"
		"lfd    29,104(6)                \n\t"
		"lfd    30,112(6)                \n\t"
		"lfd    31,120(6)                \n\t"
		"                                \n\t"
		"beq    0,.DADD                  \n\t"
		"                                \n\t"
		"fsub   0,16,0                   \n\t"
		"fsub   1,17,1                   \n\t"
		"fsub   2,18,2                   \n\t"
		"fsub   3,19,3                   \n\t"
		"                                \n\t"
		"fsub   4,20,4                   \n\t"
		"fsub   5,21,5                   \n\t"
		"fsub   6,22,6                   \n\t"
		"fsub   7,23,7                   \n\t"
		"                                \n\t"
		"fsub   8,24,8                   \n\t"
		"fsub   9,25,9                   \n\t"
		"fsub   10,26,10                 \n\t"
		"fsub   11,27,11                 \n\t"
		"                                \n\t"
		"fsub   12,28,12                 \n\t"
		"fsub   13,29,13                 \n\t"
		"fsub   14,30,14                 \n\t"
		"fsub   15,31,15                 \n\t"
		"                                \n\t"
		"b      .DSTORE                  \n\t"
		"                                \n\t"
		"                                \n\t"
		".DADD:                          \n\t"
		"                                \n\t"
		"fadd   0,16,0                   \n\t"
		"fadd   1,17,1                   \n\t"
		"fadd   2,18,2                   \n\t"
		"fadd   3,19,3                   \n\t"
		"                                \n\t"
		"fadd   4,20,4                   \n\t"
		"fadd   5,21,5                   \n\t"
		"fadd   6,22,6                   \n\t"
		"fadd   7,23,7                   \n\t"
		"                                \n\t"
		"fadd   8,24,8                   \n\t"
		"fadd   9,25,9                   \n\t"
		"fadd   10,26,10                 \n\t"
		"fadd   11,27,11                 \n\t"
		"                                \n\t"
		"fadd   12,28,12                 \n\t"
		"fadd   13,29,13                 \n\t"
		"fadd   14,30,14                 \n\t"
		"fadd   15,31,15                 \n\t"
		"                                \n\t"
		"                                \n\t"
		".DSTORE:                        \n\t"
		"                                \n\t"
		"stfd   0,0(%4)                   \n\t" // store result
		"stfd   1,8(%4)                   \n\t"
		"stfd   2,16(%4)                  \n\t"
		"stfd   3,24(%4)                  \n\t"
		"                                \n\t"
		"stfd   4,32(%4)                  \n\t"
		"stfd   5,40(%4)                  \n\t"
		"stfd   6,48(%4)                  \n\t"
		"stfd   7,56(%4)                  \n\t"
		"                                \n\t"
		"stfd   8,64(%4)                  \n\t"
		"stfd   9,72(%4)                  \n\t"
		"stfd   10,80(%4)                 \n\t"
		"stfd   11,88(%4)                 \n\t"
		"                                \n\t"
		"stfd   12,96(%4)                 \n\t"
		"stfd   13,104(%4)                \n\t"
		"stfd   14,112(%4)                \n\t"
		"stfd   15,120(%4)                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
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
		  "r" (zero)        // %6
		: // register clobber list
		  "r5", "r6",
		  "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
		  "fr8", "fr9", "fr10", "fr11", "fr12", "fr13", "fr14", "fr15",
		  "fr16", "fr17", "fr18", "fr19", "fr20", "fr21", "fr22", "fr23",
		  "fr24", "fr25", "fr26", "fr27", "fr28", "fr29", "fr30", "fr31",
		  "memory"
	);
}



// normal-transposed, 4x3 with data packed in 4
void kernel_dgemm_pp_nt_4x3_c99_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int bs = 4;*/

	int k;
	
	double
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
void kernel_dgemm_pp_nt_4x2_c99_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	double
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
void kernel_dgemm_pp_nt_4x1_c99_lib4(int kmax, double *A, double *B, double *C, int ldc, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int ldc = 4;*/

	int k;
	
	double
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

