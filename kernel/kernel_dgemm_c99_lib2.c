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

// normal-transposed, 2x2 with data packed in 2
void kernel_dgemm_pp_nt_2x2_lib2(int kmax, double *A, double *B, double *C, int bs, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int bs = 2;	*/

	int k;
	
	double
		a_0k, a_1k, b_0k, b_1k,
		c_00=0, c_01=0, c_10=0, c_11=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		b_1k = B[1];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		c_01 += a_0k * b_1k;
		c_11 += a_1k * b_1k;
		
		
		a_0k = A[2];
		a_1k = A[3];
		
		b_0k = B[2];
		b_1k = B[3];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		c_01 += a_0k * b_1k;
		c_11 += a_1k * b_1k;
		
		
		a_0k = A[4];
		a_1k = A[5];
		
		b_0k = B[4];
		b_1k = B[5];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		c_01 += a_0k * b_1k;
		c_11 += a_1k * b_1k;
		
		
		a_0k = A[6];
		a_1k = A[7];
		
		b_0k = B[6];
		b_1k = B[7];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		c_01 += a_0k * b_1k;
		c_11 += a_1k * b_1k;
		
		A += 8;
		B += 8;

		}
	
	for(; k<kmax; k++)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		b_1k = B[1];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		c_01 += a_0k * b_1k;
		c_11 += a_1k * b_1k;
		
		A += 2;
		B += 2;
		
		}

	if(alg==0)
		{
		C[0+bs*0] = c_00;
		C[1+bs*0] = c_10;
		C[0+bs*1] = c_01;
		C[1+bs*1] = c_11;
		}
	else if(alg==1)
		{
		C[0+bs*0] += c_00;
		C[1+bs*0] += c_10;
		C[0+bs*1] += c_01;
		C[1+bs*1] += c_11;
		}
	else
		{
		C[0+bs*0] -= c_00;
		C[1+bs*0] -= c_10;
		C[0+bs*1] -= c_01;
		C[1+bs*1] -= c_11;
		}

	}



// normal-transposed, 2x1 with data packed in 2
void kernel_dgemm_pp_nt_2x1_lib2(int kmax, double *A, double *B, double *C, int bs, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int bs = 2;	*/

	int k;
	
	double
		a_0k, a_1k, b_0k,
		c_00=0, c_10=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[2];
		a_1k = A[3];
		
		b_0k = B[2];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[4];
		a_1k = A[5];
		
		b_0k = B[4];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[6];
		a_1k = A[7];
		
		b_0k = B[6];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		A += 8;
		B += 8;

		}
	
	for(; k<kmax; k++)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		A += 2;
		B += 2;
		
		}

	if(alg==0)
		{
		C[0+bs*0] = c_00;
		C[1+bs*0] = c_10;
		}
	else if(alg==1)
		{
		C[0+bs*0] += c_00;
		C[1+bs*0] += c_10;
		}
	else
		{
		C[0+bs*0] -= c_00;
		C[1+bs*0] -= c_10;
		}

	}




