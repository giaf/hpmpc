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

void corner_dtrmm_pp_nt_4x3_c99_lib4(double *A, double *B, double *C, int ldc)
	{
	
	const int bs = 4;
	
	double
		a_00, a_01, a_02, a_10, a_11, a_12,
		b_00, b_10, b_20, b_11, b_21, b_22;
	
	b_00 = B[0+bs*0];
	b_10 = B[0+bs*1];
	b_20 = B[0+bs*2];
	b_11 = B[1+bs*1];
	b_21 = B[1+bs*2];
	b_22 = B[2+bs*2];
	
	a_00 = A[0+bs*0];
	a_10 = A[1+bs*0];
	a_01 = A[0+bs*1];
	a_11 = A[1+bs*1];
	a_02 = A[0+bs*2];
	a_12 = A[1+bs*2];
	
	C[0+ldc*0] = a_00*b_00 + a_01*b_10 + a_02*b_20;
	C[1+ldc*0] = a_10*b_00 + a_11*b_10 + a_12*b_20;
	C[0+ldc*1] = a_01*b_11 + a_02*b_21;
	C[1+ldc*1] = a_11*b_11 + a_12*b_21;
	C[0+ldc*2] = a_02*b_22;
	C[1+ldc*2] = a_12*b_22;
	
	a_00 = A[2+bs*0];
	a_10 = A[3+bs*0];
	a_01 = A[2+bs*1];
	a_11 = A[3+bs*1];
	a_02 = A[2+bs*2];
	a_12 = A[3+bs*2];

	C[2+ldc*0] = a_00*b_00 + a_01*b_10 + a_02*b_20;
	C[3+ldc*0] = a_10*b_00 + a_11*b_10 + a_12*b_20;
	C[2+ldc*1] = a_01*b_11 + a_02*b_21;
	C[3+ldc*1] = a_11*b_11 + a_12*b_21;
	C[2+ldc*2] = a_02*b_22;
	C[3+ldc*2] = a_12*b_22;

	}



void corner_dtrmm_pp_nt_4x2_c99_lib4(double *A, double *B, double *C, int ldc)
	{
	
	const int bs = 4;
	
	double
		a_00, a_01, a_10, a_11,
		b_00, b_10, b_11;
	
	b_00 = B[0+bs*0];
	b_10 = B[0+bs*1];
	b_11 = B[1+bs*1];
	
	a_00 = A[0+bs*0];
	a_10 = A[1+bs*0];
	a_01 = A[0+bs*1];
	a_11 = A[1+bs*1];
	
	C[0+ldc*0] = a_00*b_00 + a_01*b_10;
	C[1+ldc*0] = a_10*b_00 + a_11*b_10;
	C[0+ldc*1] = a_01*b_11;
	C[1+ldc*1] = a_11*b_11;
	
	a_00 = A[2+bs*0];
	a_10 = A[3+bs*0];
	a_01 = A[2+bs*1];
	a_11 = A[3+bs*1];

	C[2+ldc*0] = a_00*b_00 + a_01*b_10;
	C[3+ldc*0] = a_10*b_00 + a_11*b_10;
	C[2+ldc*1] = a_01*b_11;
	C[3+ldc*1] = a_11*b_11;

	}



void corner_dtrmm_pp_nt_4x1_c99_lib4(double *A, double *B, double *C, int ldc)
	{
	
	const int bs = 4;
	
	double
		a_00, a_10,
		b_00;
	
	b_00 = B[0+bs*0];
	
	a_00 = A[0+bs*0];
	a_10 = A[1+bs*0];
	
	C[0+ldc*0] = a_00*b_00;
	C[1+ldc*0] = a_10*b_00;
	
	a_00 = A[2+bs*0];
	a_10 = A[3+bs*0];

	C[2+ldc*0] = a_00*b_00;
	C[3+ldc*0] = a_10*b_00;

	}

