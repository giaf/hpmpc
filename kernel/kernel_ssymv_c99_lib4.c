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

// it moves horizontally inside a block
void kernel_ssymv_4_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
/*	if(kmax<=0) */
/*		return;*/
	
	const int bs  = 4;
	
	int	k;
	
	float
		a_00, a_10, a_20, a_30,
		x_n_0, y_n_0, y_n_1, y_n_2, y_n_3,
		x_t_0, x_t_1, x_t_2, x_t_3, y_t_0;
	
	y_n_0 = 0;
	y_n_1 = 0;
	y_n_2 = 0;
	y_n_3 = 0;
	
	x_t_0 = x_t[0];
	x_t_1 = x_t[1];
	x_t_2 = x_t[2];
	x_t_3 = x_t[3];

	if(alg==1)
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 += a_20 * x_n_0;
			y_t_0 += a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*1];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			a_20 = A[2+bs*1];
			y_n_2 += a_20 * x_n_0;
			y_t_0 += a_20 * x_t_2;
		
			a_30 = A[3+bs*1];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;

			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 += a_20 * x_n_0;
			y_t_0 += a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;

			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner

			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
/*			y_t_0 += a_00 * x_t_0;*/

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 += a_20 * x_n_0;
			y_t_0 += a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_10 = A[1+bs*1];
			y_n_1 += a_10 * x_n_0;
/*			y_t_0 += a_10 * x_t_1;*/
		
			a_20 = A[2+bs*1];
			y_n_2 += a_20 * x_n_0;
			y_t_0 += a_20 * x_t_2;
		
			a_30 = A[3+bs*1];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;

			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			// unroll 3
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_20 = A[2+bs*0];
			y_n_2 += a_20 * x_n_0;
/*			y_t_0 += a_20 * x_t_2;*/

			a_30 = A[3+bs*0];
			y_n_3 += a_30 * x_n_0;
			y_t_0 += a_30 * x_t_3;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
/*			y_t += 1;*/

			// unroll 4
			x_n_0 = x_n[0];
	
			a_30 = A[3+bs*0];
			y_n_3 += a_30 * x_n_0;
/*			y_t_0 += a_00 * x_t_3;*/

/*			y_t[0] = y_t_0;*/

			}

		}
	else // alg==-1
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 -= a_20 * x_n_0;
			y_t_0 -= a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;

			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*1];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			a_20 = A[2+bs*1];
			y_n_2 -= a_20 * x_n_0;
			y_t_0 -= a_20 * x_t_2;
		
			a_30 = A[3+bs*1];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;

			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 -= a_20 * x_n_0;
			y_t_0 -= a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;

			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner

			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
/*			y_t_0 -= a_00 * x_t_0;*/

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			a_20 = A[2+bs*0];
			y_n_2 -= a_20 * x_n_0;
			y_t_0 -= a_20 * x_t_2;
		
			a_30 = A[3+bs*0];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_10 = A[1+bs*1];
			y_n_1 -= a_10 * x_n_0;
/*			y_t_0 -= a_10 * x_t_1;*/
		
			a_20 = A[2+bs*1];
			y_n_2 -= a_20 * x_n_0;
			y_t_0 -= a_20 * x_t_2;
		
			a_30 = A[3+bs*1];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;

			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			// unroll 3
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_20 = A[2+bs*0];
			y_n_2 -= a_20 * x_n_0;
/*			y_t_0 -= a_20 * x_t_2;*/

			a_30 = A[3+bs*0];
			y_n_3 -= a_30 * x_n_0;
			y_t_0 -= a_30 * x_t_3;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			// unroll 4
			x_n_0 = x_n[0];
	
			a_30 = A[3+bs*0];
			y_n_3 -= a_30 * x_n_0;
/*			y_t_0 -= a_00 * x_t_3;*/

/*			y_t[0] = y_t_0;*/

			}

		}		

	y_n[0] += y_n_0;
	y_n[1] += y_n_1;
	y_n[2] += y_n_2;
	y_n[3] += y_n_3;

	}



// it moves horizontally inside a block
void kernel_ssymv_2_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
/*	if(kmax<=0) */
/*		return;*/
	
	const int bs  = 4;
	
	int	k;
	
	float
		a_00, a_10,
		x_t_0, x_t_1, y_t_0,
		x_n_0, y_n_0, y_n_1;
	
	y_n_0 = 0;
	y_n_1 = 0;
	
	x_t_0 = x_t[0];
	x_t_1 = x_t[1];

	if(alg==1)
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*1];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;

			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;
			y_t_0 += a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
/*			y_t += 1;*/

			x_n_0 = x_n[0];
	
			a_10 = A[1+bs*0];
			y_n_1 += a_10 * x_n_0;

			}

		}
	else // alg==-1
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*1];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;

			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;
			y_t_0 -= a_10 * x_t_1;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
/*			y_t += 1;*/

			x_n_0 = x_n[0];
	
			a_10 = A[1+bs*0];
			y_n_1 -= a_10 * x_n_0;

			}

		}		

	y_n[0] += y_n_0;
	y_n[1] += y_n_1;

	}



// it moves horizontally inside a block
void kernel_ssymv_1_lib4(int kmax, float *A, float *x_n, float *y_n, float *x_t, float *y_t, int tri, int alg)
	{
	
/*	if(kmax<=0) */
/*		return;*/
	
	const int bs  = 4;
	
	int	k;
	
	float
		a_00,
		x_t_0, y_t_0,
		x_n_0, y_n_0;
	
	y_n_0 = 0;
	
	x_t_0 = x_t[0];

	if(alg==1)
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
		
			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;
			y_t_0 += a_00 * x_t_0;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner
			x_n_0 = x_n[0];
	
			a_00 = A[0+bs*0];
			y_n_0 += a_00 * x_n_0;

			}
		
		}
	else // alg==-1
		{
		k=0;
		for(; k<kmax-1; k+=2)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;
		
			y_t[0] = y_t_0;
		
			// unroll 2
			x_n_0 = x_n[1];
			y_t_0 = y_t[1];
		
			a_00 = A[0+bs*1];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;
		
			y_t[1] = y_t_0;
		
			A   += 2*bs;
			x_n += 2;
			y_t += 2;

			}
		for(; k<kmax; k++)
			{
		
			// unroll 1
			x_n_0 = x_n[0];
			y_t_0 = y_t[0];
		
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;
			y_t_0 -= a_00 * x_t_0;
		
			y_t[0] = y_t_0;
		
			A   += 1*bs;
			x_n += 1;
			y_t += 1;

			}
		if(tri==1)
			{

			// corner
			x_n_0 = x_n[0];
	
			a_00 = A[0+bs*0];
			y_n_0 -= a_00 * x_n_0;

			}
		
		}

	y_n[0] += y_n_0;

	}
