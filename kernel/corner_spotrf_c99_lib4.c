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



void corner_spotrf_strsv_scopy_3x3_lib4(int kinv, float *A, int sda, int shf, float *L, int sdl, int *info)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);
	const int shfi2 = ((shfi+2)/lda)*lda*(sdl-1);

	float
		a_00, a_10, a_20, a_11, a_21, a_22;

	// spotrf
		
	if(kinv==0)
		{

		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
		A[0+lda*0] = a_00;
		L[0+0*lda+shfi0] = a_00;
		a_00 = 1.0/a_00;
		a_10 = A[1+lda*0] * a_00;
		a_20 = A[2+lda*0] * a_00;
		A[1+lda*0] = a_10;
		A[2+lda*0] = a_20;
		L[0+1*lda+shfi0] = a_10;
		L[0+2*lda+shfi0] = a_20;

		a_11 = A[1+lda*1] - a_10*a_10;
		if( a_11 <= 0.0 ) { *info = 1; return; }
		a_11 = sqrt( a_11 );
		A[1+lda*1] = a_11;
		L[1+1*lda+shfi1] = a_11;
		a_11 = 1.0/a_11;
		a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
		A[2+lda*1] = a_21;
		L[1+2*lda+shfi1] = a_21;
		
		}
	else // kinv == {1,2}
		{

		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
/*		A[0+lda*0] = a_00;*/
/*		L[0+0*lda+shfi0] = a_00;*/
		a_00 = 1.0/a_00;
		A[0+lda*0] = a_00;
		a_10 = A[1+lda*0] * a_00;
		a_20 = A[2+lda*0] * a_00;
		A[1+lda*0] = a_10;
		A[2+lda*0] = a_20;
		L[0+1*lda+shfi0] = a_10;
		L[0+2*lda+shfi0] = a_20;

		a_11 = A[1+lda*1] - a_10*a_10;
		if( a_11 <= 0.0 ) { *info = 1; return; }
		a_11 = sqrt( a_11 );
		if(kinv<=1)
			{
			A[1+lda*1] = a_11;
			L[1+1*lda+shfi1] = a_11;
			}
		a_11 = 1.0/a_11;
		if(kinv>1)
			A[1+lda*1] = a_11;
		a_21 = (A[2+lda*1] - a_20*a_10) * a_11;
		A[2+lda*1] = a_21;
		L[1+2*lda+shfi1] = a_21;

		}
	
	a_22 = A[2+lda*2] - a_20*a_20 - a_21*a_21;
	if( a_22 <= 0.0 ) { *info = 1; return; }
	a_22 = sqrt( a_22 );
	A[2+lda*2] = a_22;
	L[2+2*lda+shfi2] = a_22;

	}



void corner_spotrf_strsv_scopy_2x2_lib4(int kinv, float *A, int sda, int shf, float *L, int sdl, int *info)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);
	const int shfi1 = ((shfi+1)/lda)*lda*(sdl-1);

	float
		a_00, a_10, a_11;

	// spotrf
		
	if(kinv==0)
		{

		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
		A[0+lda*0] = a_00;
		L[0+0*lda+shfi0] = a_00;
		a_00 = 1.0/a_00;
		a_10 = A[1+lda*0] * a_00;
		A[1+lda*0] = a_10;
		L[0+1*lda+shfi0] = a_10;
		
		}
	else // kinv == 1
		{

		a_00 = A[0+lda*0];
		if( a_00 <= 0.0 ) { *info = 1; return; }
		a_00 = sqrt( a_00 );
/*		L[0+0*lda+shfi0] = a_00;*/
		a_00 = 1.0/a_00;
		A[0+lda*0] = a_00;
		a_10 = A[1+lda*0] * a_00;
		A[1+lda*0] = a_10;
		L[0+1*lda+shfi0] = a_10;
		
		}

	a_11 = A[1+lda*1] - a_10*a_10;
	if( a_11 <= 0.0 ) { *info = 1; return; }
	a_11 = sqrt( a_11 );
	A[1+lda*1] = a_11;
	L[1+1*lda+shfi1] = a_11;

	}


void corner_spotrf_strsv_scopy_1x1_lib4(float *A, int sda, int shf, float *L, int sdl, int *info)
	{
	
	const int lda = 4;
	
	L += shf*(lda+1);
	const int shfi = shf + lda - 4;
	const int shfi0 = ((shfi+0)/lda)*lda*(sdl-1);

	float
		a_00;

	// spotrf
		
	a_00 = A[0+lda*0];
	if( a_00 <= 0.0 ) { *info = 1; return; }
	a_00 = sqrt( a_00 );
	A[0+lda*0] = a_00;
	L[0+0*lda+shfi0] = a_00;

	}

