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



void kernel_dpotrf_dtrsv_4x4_lib4(int kmax, int kinv, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_20, a_30, a_11, a_21, a_31, a_22, a_32, a_33;

	// dpotrf

	if(kinv==0)
		{
		
		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
		A[0+lda*0] = a_00;
		a_00 = 1.0/a_00;
		a_10 = A[1+lda*0] * a_00;
		a_20 = A[2+lda*0] * a_00;
		a_30 = A[3+lda*0] * a_00;
		A[1+lda*0] = a_10;
		A[2+lda*0] = a_20;
		A[3+lda*0] = a_30;

		a_11 = A[1+lda*1] - a_10*a_10;
		if( a_11 <= 0.0 ) { *info = 1; return; }
		a_11 = sqrt( a_11 );
		A[1+lda*1] = a_11;
		a_11 = 1.0/a_11;
		a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
		a_31 = (A[3+lda*1] - a_30*a_10) * a_11;
		A[2+lda*1] = a_21;
		A[3+lda*1] = a_31;
	
		a_22 = A[2+lda*2] - a_20*a_20 - a_21*a_21;
		if( a_22 <= 0.0 ) { *info = 1; return; }
		a_22 = sqrt( a_22 );
		A[2+lda*2] = a_22;
		a_22 = 1.0/a_22;
		a_32 = (A[3+lda*2] - a_30*a_20 - a_31*a_21) * a_22;
		A[3+lda*2] = a_32;
			
		a_33 = A[3+lda*3] - a_30*a_30 - a_31*a_31 - a_32*a_32;
		if( a_33 <= 0.0 ) { *info = 1; return; }
		a_33 = sqrt( a_33 );
		A[3+lda*3] = a_33;
		if(kmax>0)
			a_33 = 1.0/a_33;

		}
	else // kinv == {1, 2, 3, 4}
		{		

		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
		a_00 = 1.0/a_00;
		A[0+lda*0] = a_00;
		a_10 = A[1+lda*0] * a_00;
		a_20 = A[2+lda*0] * a_00;
		a_30 = A[3+lda*0] * a_00;
		A[1+lda*0] = a_10;
		A[2+lda*0] = a_20;
		A[3+lda*0] = a_30;

		a_11 = A[1+lda*1] - a_10*a_10;
		if( a_11 <= 0.0 ) { *info = 1; return; }
		a_11 = sqrt( a_11 );
		if(kinv<=1)
			{
			A[1+lda*1] = a_11;
			}
		a_11 = 1.0/a_11;
		if(kinv>1)
			A[1+lda*1] = a_11;
		a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
		a_31 = (A[3+lda*1] - a_30*a_10) * a_11;
		A[2+lda*1] = a_21;
		A[3+lda*1] = a_31;
	
		a_22 = A[2+lda*2] - a_20*a_20 - a_21*a_21;
		if( a_22 <= 0.0 ) { *info = 1; return; }
		a_22 = sqrt( a_22 );
		if(kinv<=2)
			{
			A[2+lda*2] = a_22;
			}
		a_22 = 1.0/a_22;
		if(kinv>2)
			A[2+lda*2] = a_22;
		a_32 = (A[3+lda*2] - a_30*a_20 - a_31*a_21) * a_22;
		A[3+lda*2] = a_32;
	
		a_33 = A[3+lda*3] - a_30*a_30 - a_31*a_31 - a_32*a_32;
		if( a_33 <= 0.0 ) { *info = 1; return; }
		a_33 = sqrt( a_33 );
		if(kinv<=3)
			{
			A[3+lda*3] = a_33;
			}
		a_33 = 1.0/a_33;
		if(kinv>3)
			A[3+lda*3] = a_33;

		}
	

	if(kmax<=0)
		return;
	
	// dtrsv

/*	a_33 = 1.0/a_33;*/

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



// inverted diagonal !!!
void kernel_dpotrf_dtrsv_3x3_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_20, a_11, a_21, a_22;

	// dpotrf
		
	a_00 = A[0+lda*0];
	if( a_00 <= 0.0 ) { *info = 1; return; }
	a_00 = sqrt( a_00 );
	a_00 = 1.0/a_00;
	A[0+lda*0] = a_00;
	a_10 = A[1+lda*0] * a_00;
	a_20 = A[2+lda*0] * a_00;
	A[1+lda*0] = a_10;
	A[2+lda*0] = a_20;

	a_11 = A[1+lda*1] - a_10*a_10;
	if( a_11 <= 0.0 ) { *info = 1; return; }
	a_11 = sqrt( a_11 );
	a_11 = 1.0/a_11;
	A[1+lda*1] = a_11;
	a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
	A[2+lda*1] = a_21;
	
	a_22 = A[2+lda*2] - a_20*a_20 - a_21*a_21;
	if( a_22 <= 0.0 ) { *info = 1; return; }
	a_22 = sqrt( a_22 );
	a_22 = 1.0/a_22;
	A[2+lda*2] = a_22;


	
	if(kmax<=0)
		return;
	
	// dtrsv


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



// inverted diagonal !!!
void kernel_dpotrf_dtrsv_2x2_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	double
		a_00, a_10, a_11;

	// dpotrf
		
	a_00 = A[0+lda*0];
	if( a_00 <= 0.0 ) { *info = 1; return; }
	a_00 = sqrt( a_00 );
	a_00 = 1.0/a_00;
	A[0+lda*0] = a_00;
	a_10 = A[1+lda*0] * a_00;
	A[1+lda*0] = a_10;

	a_11 = A[1+lda*1] - a_10*a_10;
	if( a_11 <= 0.0 ) { *info = 1; return; }
	a_11 = sqrt( a_11 );
	a_11 = 1.0/a_11;
	A[1+lda*1] = a_11;


	
	if(kmax<=0)
		return;
	
	// dtrsv


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



// inverted diagonal !!!
void kernel_dpotrf_dtrsv_1x1_lib4(int kmax, double *A, int sda, int *info)
	{
	
	const int lda = 4;
	
	double
		a_00;

	// dpotrf
		
	a_00 = A[0+lda*0];
	if( a_00 <= 0.0 ) { *info = 1; return; }
	a_00 = sqrt( a_00 );
	a_00 = 1.0/a_00;
	A[0+lda*0] = a_00;


	
	if(kmax<=0)
		return;
	
	// dtrsv


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

