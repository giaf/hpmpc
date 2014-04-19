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

#include <math.h>



void kernel_dpotrf_dtrsv_dcopy_4x4_lib4(int kmax, double *A, int sda, int shf, double *L, int sdl)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);
	const int shfi2 = ((shfi+2)/lda)*lda*(sdl-1);
	const int shfi3 = ((shfi+3)/lda)*lda*(sdl-1);

	double
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	L[0+0*lda+shfi0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	a_20 = A[2+lda*0] * a_00;
	a_30 = A[3+lda*0] * a_00;
	A[1+lda*0] = a_10;
	A[2+lda*0] = a_20;
	A[3+lda*0] = a_30;
	L[0+1*lda+shfi0] = a_10;
	L[0+2*lda+shfi0] = a_20;
	L[0+3*lda+shfi0] = a_30;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	L[1+1*lda+shfi1] = a_11;
	a_11 = 1.0/a_11;
	a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
	a_31 = (A[3+lda*1] - a_30*a_10) * a_11;
	A[2+lda*1] = a_21;
	A[3+lda*1] = a_31;
	L[1+2*lda+shfi1] = a_21;
	L[1+3*lda+shfi1] = a_31;
	
	a_22 = sqrt(A[2+lda*2] - a_20*a_20 - a_21*a_21);
	A[2+lda*2] = a_22;
	L[2+2*lda+shfi2] = a_22;
	a_22 = 1.0/a_22;
	a_32 = (A[3+lda*2] - a_30*a_20 - a_31*a_21) * a_22;
	A[3+lda*2] = a_32;
	L[2+3*lda+shfi2] = a_32;
	
	a_33 = sqrt(A[3+lda*3] - a_30*a_30 - a_31*a_31 - a_32*a_32);
	A[3+lda*3] = a_33;
	L[3+3*lda+shfi3] = a_33;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_33 = 1.0/a_33;

	int k;
	
	double
		b_00, b_01, b_02, b_03,
		b_10, b_11, b_12, b_13,
		*AA, *LL;
	
	AA = A + 4;
	LL = L + 4*lda;
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_01 = AA[0+lda*1];
		b_11 = AA[1+lda*1];

		b_02 = AA[0+lda*2];
		b_12 = AA[1+lda*2];

		b_03 = AA[0+lda*3];
		b_13 = AA[1+lda*3];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;
		LL[0+shfi0+0*lda] = b_00;
		LL[0+shfi0+1*lda] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[0+lda*1] = b_01;
		AA[1+lda*1] = b_11;
		LL[1+shfi1+0*lda] = b_01;
		LL[1+shfi1+1*lda] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[0+lda*2] = b_02;
		AA[1+lda*2] = b_12;
		LL[2+shfi2+0*lda] = b_02;
		LL[2+shfi2+1*lda] = b_12;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		b_13 = (b_13 - b_10 * a_30 - b_11 * a_31 - b_12 * a_32) * a_33;
		AA[0+lda*3] = b_03;
		AA[1+lda*3] = b_13;
		LL[3+shfi3+0*lda] = b_03;
		LL[3+shfi3+1*lda] = b_13;


		b_00 = AA[2+lda*0];
		b_10 = AA[3+lda*0];

		b_01 = AA[2+lda*1];
		b_11 = AA[3+lda*1];

		b_02 = AA[2+lda*2];
		b_12 = AA[3+lda*2];

		b_03 = AA[2+lda*3];
		b_13 = AA[3+lda*3];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[2+lda*0] = b_00;
		AA[3+lda*0] = b_10;
		LL[0+shfi0+2*lda] = b_00;
		LL[0+shfi0+3*lda] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[2+lda*1] = b_01;
		AA[3+lda*1] = b_11;
		LL[1+shfi1+2*lda] = b_01;
		LL[1+shfi1+3*lda] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[2+lda*2] = b_02;
		AA[3+lda*2] = b_12;
		LL[2+shfi2+2*lda] = b_02;
		LL[2+shfi2+3*lda] = b_12;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		b_13 = (b_13 - b_10 * a_30 - b_11 * a_31 - b_12 * a_32) * a_33;
		AA[2+lda*3] = b_03;
		AA[3+lda*3] = b_13;
		LL[3+shfi3+2*lda] = b_03;
		LL[3+shfi3+3*lda] = b_13;


		AA += 4;
		LL += 4*lda;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_02 = AA[0+lda*2];

		b_03 = AA[0+lda*3];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;
		LL[0+shfi0+0*lda] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;
		LL[1+shfi1+0*lda] = b_01;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		AA[0+lda*2] = b_02;
		LL[2+shfi2+0*lda] = b_02;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		AA[0+lda*3] = b_03;
		LL[3+shfi3+0*lda] = b_03;

		AA += 1;
		LL += lda;
		}
	
	}



void kernel_dpotrf_dtrsv_4x4_lib4(int kmax, double *A, int sda)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	a_20 = A[2+lda*0] * a_00;
	a_30 = A[3+lda*0] * a_00;
	A[1+lda*0] = a_10;
	A[2+lda*0] = a_20;
	A[3+lda*0] = a_30;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	a_11 = 1.0/a_11;
	a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
	a_31 = (A[3+lda*1] - a_30*a_10) * a_11;
	A[2+lda*1] = a_21;
	A[3+lda*1] = a_31;
	
	a_22 = sqrt(A[2+lda*2] - a_20*a_20 - a_21*a_21);
	A[2+lda*2] = a_22;
	a_22 = 1.0/a_22;
	a_32 = (A[3+lda*2] - a_30*a_20 - a_31*a_21) * a_22;
	A[3+lda*2] = a_32;
	
	a_33 = sqrt(A[3+lda*3] - a_30*a_30 - a_31*a_31 - a_32*a_32);
	A[3+lda*3] = a_33;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_33 = 1.0/a_33;

	int k;
	
	double
		b_00, b_01, b_02, b_03,
		b_10, b_11, b_12, b_13,
		*AA;
	
	AA = A + 4;
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_01 = AA[0+lda*1];
		b_11 = AA[1+lda*1];

		b_02 = AA[0+lda*2];
		b_12 = AA[1+lda*2];

		b_03 = AA[0+lda*3];
		b_13 = AA[1+lda*3];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[0+lda*1] = b_01;
		AA[1+lda*1] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[0+lda*2] = b_02;
		AA[1+lda*2] = b_12;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		b_13 = (b_13 - b_10 * a_30 - b_11 * a_31 - b_12 * a_32) * a_33;
		AA[0+lda*3] = b_03;
		AA[1+lda*3] = b_13;


		b_00 = AA[2+lda*0];
		b_10 = AA[3+lda*0];

		b_01 = AA[2+lda*1];
		b_11 = AA[3+lda*1];

		b_02 = AA[2+lda*2];
		b_12 = AA[3+lda*2];

		b_03 = AA[2+lda*3];
		b_13 = AA[3+lda*3];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[2+lda*0] = b_00;
		AA[3+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[2+lda*1] = b_01;
		AA[3+lda*1] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[2+lda*2] = b_02;
		AA[3+lda*2] = b_12;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		b_13 = (b_13 - b_10 * a_30 - b_11 * a_31 - b_12 * a_32) * a_33;
		AA[2+lda*3] = b_03;
		AA[3+lda*3] = b_13;


		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_02 = AA[0+lda*2];

		b_03 = AA[0+lda*3];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		AA[0+lda*2] = b_02;

		b_03 = (b_03 - b_00 * a_30 - b_01 * a_31 - b_02 * a_32) * a_33;
		AA[0+lda*3] = b_03;

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_3x3_lib4(int kmax, double *A, int sda)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_20, a_11, a_21, a_22;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	a_20 = A[2+lda*0] * a_00;
	A[1+lda*0] = a_10;
	A[2+lda*0] = a_20;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;
	a_11 = 1.0/a_11;
	a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
	A[2+lda*1] = a_21;
	
	a_22 = sqrt(A[2+lda*2] - a_20*a_20 - a_21*a_21);
	A[2+lda*2] = a_22;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_22 = 1.0/a_22;

	int k, kna;
	
	double
		b_00, b_01, b_02,
		b_10, b_11, b_12,
		*AA;
	
	AA = A + 3;
	k = 0;
	
	kna = 1;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_02 = AA[0+lda*2];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		AA[0+lda*2] = b_02;

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_01 = AA[0+lda*1];
		b_11 = AA[1+lda*1];

		b_02 = AA[0+lda*2];
		b_12 = AA[1+lda*2];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[0+lda*1] = b_01;
		AA[1+lda*1] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[0+lda*2] = b_02;
		AA[1+lda*2] = b_12;


		b_00 = AA[2+lda*0];
		b_10 = AA[3+lda*0];

		b_01 = AA[2+lda*1];
		b_11 = AA[3+lda*1];

		b_02 = AA[2+lda*2];
		b_12 = AA[3+lda*2];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[2+lda*0] = b_00;
		AA[3+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[2+lda*1] = b_01;
		AA[3+lda*1] = b_11;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		b_12 = (b_12 - b_10 * a_20 - b_11 * a_21) * a_22;
		AA[2+lda*2] = b_02;
		AA[3+lda*2] = b_12;


		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_02 = AA[0+lda*2];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;

		b_02 = (b_02 - b_00 * a_20 - b_01 * a_21) * a_22;
		AA[0+lda*2] = b_02;

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_2x2_lib4(int kmax, double *A, int sda)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_11;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;
	a_00 = 1.0/a_00;
	a_10 = A[1+lda*0] * a_00;
	A[1+lda*0] = a_10;

	a_11 = sqrt(A[1+lda*1] - a_10*a_10);
	A[1+lda*1] = a_11;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_11 = 1.0/a_11;

	int k, kna;
	
	double
		b_00, b_01,
		b_10, b_11,
		*AA;
	
	AA = A + 2;
	k = 0;
	
	kna = 2;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_01 = AA[0+lda*1];
		b_11 = AA[1+lda*1];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[0+lda*1] = b_01;
		AA[1+lda*1] = b_11;


		b_00 = AA[2+lda*0];
		b_10 = AA[3+lda*0];

		b_01 = AA[2+lda*1];
		b_11 = AA[3+lda*1];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[2+lda*0] = b_00;
		AA[3+lda*0] = b_10;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		b_11 = (b_11 - b_10 * a_10) * a_11;
		AA[2+lda*1] = b_01;
		AA[3+lda*1] = b_11;


		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_01 = AA[0+lda*1];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		b_01 = (b_01 - b_00 * a_10) * a_11;
		AA[0+lda*1] = b_01;

		AA += 1;
		}
	
	}



void kernel_dpotrf_dtrsv_1x1_lib4(int kmax, double *A, int sda)
	{
	
	const int lda = 4;
	
	double
		a_00;

	// dpotrf
		
	a_00 = sqrt(A[0+lda*0]);
	A[0+lda*0] = a_00;


	
	if(kmax<=0)
		return;
	
	// dtrsv

	a_00 = 1.0/a_00;

	int k, kna;
	
	double
		b_00,
		b_10,
		*AA;
	
	AA = A + 1;
	k = 0;
	
	kna = 3;
	if(kmax<kna)
		kna = kmax;

	for(; k<kna; k++)
		{
		b_00 = AA[0+lda*0];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		AA += 1;
		}

	for(; k<kmax-3; k+=4)
		{

		AA += lda*(sda-1);
		

		b_00 = AA[0+lda*0];
		b_10 = AA[1+lda*0];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[0+lda*0] = b_00;
		AA[1+lda*0] = b_10;


		b_00 = AA[2+lda*0];
		b_10 = AA[3+lda*0];

		b_00 *= a_00;
		b_10 *= a_00;
		AA[2+lda*0] = b_00;
		AA[3+lda*0] = b_10;


		AA += 4;
		
		}

	AA += lda*(sda-1);

	for(; k<kmax; k++)
		{
		b_00 = AA[0+lda*0];

		b_00 *= a_00;
		AA[0+lda*0] = b_00;

		AA += 1;
		}
	
	}

