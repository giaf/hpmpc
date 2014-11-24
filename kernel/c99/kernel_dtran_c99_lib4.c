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



// transpose an aligned upper triangular matrix in an aligned lower triangular matrix
void kernel_dtrtr_u_4_lib4(int kmax, double *A, double *C, int sdc)
	{

	const int bs = 4;

	int k;

	C[0+bs*0] = A[0+bs*0];

	C[1+bs*0] = A[0+bs*1];
	C[1+bs*1] = A[1+bs*1];

	C[2+bs*0] = A[0+bs*2];
	C[2+bs*1] = A[1+bs*2];
	C[2+bs*2] = A[2+bs*2];

	C[3+bs*0] = A[0+bs*3];
	C[3+bs*1] = A[1+bs*3];
	C[3+bs*2] = A[2+bs*3];
	C[3+bs*3] = A[3+bs*3];

	C += bs*sdc;
	A += bs*bs;

	k = 4;
	for( ; k<kmax-3; k+=4)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];
		C[0+bs*3] = A[3+bs*0];

		C[1+bs*0] = A[0+bs*1];
		C[1+bs*1] = A[1+bs*1];
		C[1+bs*2] = A[2+bs*1];
		C[1+bs*3] = A[3+bs*1];

		C[2+bs*0] = A[0+bs*2];
		C[2+bs*1] = A[1+bs*2];
		C[2+bs*2] = A[2+bs*2];
		C[2+bs*3] = A[3+bs*2];

		C[3+bs*0] = A[0+bs*3];
		C[3+bs*1] = A[1+bs*3];
		C[3+bs*2] = A[2+bs*3];
		C[3+bs*3] = A[3+bs*3];

		C += bs*sdc;
		A += bs*bs;
		}
	
	for( ; k<kmax; k++)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];
		C[0+bs*3] = A[3+bs*0];

		C += 1;
		A += bs;
		}

	}



// transposed of general matrices, read along panels, write across panels TODO test if it is the best way
void kernel_dgetr_4_lib4(int kmax, int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;
	
	int k;

	k = 0;
	if(kna>0)
		{
		for( ; k<kna; k++)
			{
			C[0+bs*0] = A[0+bs*0];
			C[0+bs*1] = A[1+bs*0];
			C[0+bs*2] = A[2+bs*0];
			C[0+bs*3] = A[3+bs*0];

			C += 1;
			A += bs;
			}
		C += bs*(sdc-1);
		}
	
	for( ; k<kmax-3; k+=4)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];
		C[0+bs*3] = A[3+bs*0];

		C[1+bs*0] = A[0+bs*1];
		C[1+bs*1] = A[1+bs*1];
		C[1+bs*2] = A[2+bs*1];
		C[1+bs*3] = A[3+bs*1];

		C[2+bs*0] = A[0+bs*2];
		C[2+bs*1] = A[1+bs*2];
		C[2+bs*2] = A[2+bs*2];
		C[2+bs*3] = A[3+bs*2];

		C[3+bs*0] = A[0+bs*3];
		C[3+bs*1] = A[1+bs*3];
		C[3+bs*2] = A[2+bs*3];
		C[3+bs*3] = A[3+bs*3];

		C += bs*sdc;
		A += bs*bs;
		}
	for( ; k<kmax; k++)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];
		C[0+bs*3] = A[3+bs*0];

		C += 1;
		A += bs;
		}

	}



// transposed of general matrices, read along panels, write across panels TODO test if it is the best way
void kernel_dgetr_3_lib4(int kmax, int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;
	
	int k;

	k = 0;
	if(kna>0)
		{
		for( ; k<kna; k++)
			{
			C[0+bs*0] = A[0+bs*0];
			C[0+bs*1] = A[1+bs*0];
			C[0+bs*2] = A[2+bs*0];

			C += 1;
			A += bs;
			}
		C += bs*(sdc-1);
		}
	
	for( ; k<kmax-3; k+=4)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];

		C[1+bs*0] = A[0+bs*1];
		C[1+bs*1] = A[1+bs*1];
		C[1+bs*2] = A[2+bs*1];

		C[2+bs*0] = A[0+bs*2];
		C[2+bs*1] = A[1+bs*2];
		C[2+bs*2] = A[2+bs*2];

		C[3+bs*0] = A[0+bs*3];
		C[3+bs*1] = A[1+bs*3];
		C[3+bs*2] = A[2+bs*3];

		C += bs*sdc;
		A += bs*bs;
		}
	for( ; k<kmax; k++)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];

		C += 1;
		A += bs;
		}

	}



// transposed of general matrices, read along panels, write across panels TODO test if it is the best way
void kernel_dgetr_2_lib4(int kmax, int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;
	
	int k;

	k = 0;
	if(kna>0)
		{
		for( ; k<kna; k++)
			{
			C[0+bs*0] = A[0+bs*0];
			C[0+bs*1] = A[1+bs*0];

			C += 1;
			A += bs;
			}
		C += bs*(sdc-1);
		}
	
	for( ; k<kmax-3; k+=4)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];

		C[1+bs*0] = A[0+bs*1];
		C[1+bs*1] = A[1+bs*1];

		C[2+bs*0] = A[0+bs*2];
		C[2+bs*1] = A[1+bs*2];

		C[3+bs*0] = A[0+bs*3];
		C[3+bs*1] = A[1+bs*3];

		C += bs*sdc;
		A += bs*bs;
		}
	for( ; k<kmax; k++)
		{
		C[0+bs*0] = A[0+bs*0];
		C[0+bs*1] = A[1+bs*0];

		C += 1;
		A += bs;
		}

	}



// transposed of general matrices, read along panels, write across panels TODO test if it is the best way
void kernel_dgetr_1_lib4(int kmax, int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;
	
	int k;

	k = 0;
	if(kna>0)
		{
		for( ; k<kna; k++)
			{
			C[0+bs*0] = A[0+bs*0];

			C += 1;
			A += bs;
			}
		C += bs*(sdc-1);
		}
	
	for( ; k<kmax-3; k+=4)
		{
		C[0+bs*0] = A[0+bs*0];

		C[1+bs*0] = A[0+bs*1];

		C[2+bs*0] = A[0+bs*2];

		C[3+bs*0] = A[0+bs*3];

		C += bs*sdc;
		A += bs*bs;
		}
	for( ; k<kmax; k++)
		{
		C[0+bs*0] = A[0+bs*0];

		C += 1;
		A += bs;
		}

	}



// TODO change name of routine TODO is this the best way???
void kernel_dtran_4_lib4(int kmax, int kna, double *A, int sda, double *C)
	{
	
	// kmax is at least 4 !!!
	
	int k;

	const int bs = 4;
	
	k=0;

	if(kna==0)
		{

		C[0+bs*0] = A[0+bs*0];

		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];

		C[0+bs*3] = A[3+bs*0];
		C[1+bs*3] = A[3+bs*1];
		C[2+bs*3] = A[3+bs*2];
		C[3+bs*3] = A[3+bs*3];
		
		A += 4*sda;
		C += 4*bs;
		k += 4;
		
		}
	else if(kna==1)
		{
		
		// top 1x1 triangle
		C[0+bs*0] = A[0+bs*0];
		
		A += 1 + bs*(sda-1);
		C += bs;
		k += 1;

		// 4x4
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];
		C[3+bs*2] = A[2+bs*3];
		
		if(kmax==4)
			return;

		C[0+bs*3] = A[3+bs*0];
		C[1+bs*3] = A[3+bs*1];
		C[2+bs*3] = A[3+bs*2];
		C[3+bs*3] = A[3+bs*3];

		A += bs*sda;
		C += 4*bs;
		k += 4;

		}
	else if(kna==2)
		{

		// top 2x2 triangle
		C[0+bs*0] = A[0+bs*0];

		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];

		A += 2 + bs*(sda-1);
		C += 2*bs;
		k += 2;

		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];
		C[3+bs*1] = A[1+bs*3];
		
		if(kmax==4)
			return;

		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];
		C[3+bs*2] = A[2+bs*3];
		
		if(kmax==5)
			return;

		C[0+bs*3] = A[3+bs*0];
		C[1+bs*3] = A[3+bs*1];
		C[2+bs*3] = A[3+bs*2];
		C[3+bs*3] = A[3+bs*3];

		A += bs*sda;
		C += 4*bs;
		k += 4;

		}
	else // if(kna==3)
		{

		// top 1x1 triangle
		C[0+bs*0] = A[0+bs*0];

		// 2x2 square
		C[0+bs*1] = A[1+bs*0];
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[1+bs*2] = A[2+bs*1];

		// low 1x1 triangle
		C[2+bs*2] = A[2+bs*2];

		A += 3 + bs*(sda-1);
		C += 3*bs;
		k += 3;

		}

	for(; k<kmax-3; k+=4)
		{
		
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];
		C[3+bs*0] = A[0+bs*3];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];
		C[3+bs*1] = A[1+bs*3];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];
		C[3+bs*2] = A[2+bs*3];
		
		C[0+bs*3] = A[3+bs*0];
		C[1+bs*3] = A[3+bs*1];
		C[2+bs*3] = A[3+bs*2];
		C[3+bs*3] = A[3+bs*3];
	
		A += bs*sda;
		C += 4*bs;

		}

	if(k==kmax)
		return;

	if(kmax-k==1)
		{
		
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];
		C[3+bs*0] = A[0+bs*3];

		}
	else if(kmax-k==2)
		{
		
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];
		C[3+bs*0] = A[0+bs*3];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];
		C[3+bs*1] = A[1+bs*3];

		}
	else // if(kmax-k==3)
		{

		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];
		C[3+bs*0] = A[0+bs*3];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];
		C[3+bs*1] = A[1+bs*3];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];
		C[3+bs*2] = A[2+bs*3];

		}

	return;
	
	}



void corner_dtran_3_lib4(int kna, double *A, int sda, double *C)
	{

	const int bs = 4;
	
	if(kna==0)
		{
		
		C[0+bs*0] = A[0+bs*0];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];

		}
	else if(kna==1)
		{
		
		C[0+bs*0] = A[0+bs*0];
		
		A += 1 + bs*(sda-1);
		C += bs;

		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[1+bs*2];

		}
	else if(kna==2)
		{

		C[0+bs*0] = A[0+bs*0];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];

		A += 2 + bs*(sda-1);
		C += 2*bs;

		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];
		C[2+bs*0] = A[0+bs*2];

		}
	else // if(kna==3)
		{

		C[0+bs*0] = A[0+bs*0];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];
		
		C[0+bs*2] = A[2+bs*0];
		C[1+bs*2] = A[2+bs*1];
		C[2+bs*2] = A[2+bs*2];

		}

	}



void corner_dtran_2_lib4(int kna, double *A, int sda, double *C)
	{

	const int bs = 4;
	
	if(kna==1)
		{

		C[0+bs*0] = A[0+bs*0];
		
		A += 1 + bs*(sda-1);
		C += bs;

		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[0+bs*1];

		}
	else // if(kna==3)
		{
		
		C[0+bs*0] = A[0+bs*0];
		
		C[0+bs*1] = A[1+bs*0];
		C[1+bs*1] = A[1+bs*1];

		}

	}



// mis-align a triangolar matrix; it moves across panels; read aligned, write mis-aligned
void kernel_dtrma_4_lib4(int kmax, int kna, double *A, int sda, double *C, int sdc)
	{

	// assume kmax >= 4 !!!

	const int bs = 4;

	int k;

	if(kna==0) // same alignment
		{

		// triangle at the beginning
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];
		C[2+bs*0] = A[2+bs*0];
		C[3+bs*0] = A[3+bs*0];

		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[2+bs*1];
		C[3+bs*1] = A[3+bs*1];

		C[2+bs*2] = A[2+bs*2];
		C[3+bs*2] = A[3+bs*2];

		C[3+bs*3] = A[3+bs*3];

		A += bs*sda;
		C += bs*sdc;
		k = 4;

		// main loop
		for( ; k<kmax-3; k+=4)
			{
			C[0+bs*0] = A[0+bs*0];
			C[1+bs*0] = A[1+bs*0];
			C[2+bs*0] = A[2+bs*0];
			C[3+bs*0] = A[3+bs*0];

			C[0+bs*1] = A[0+bs*1];
			C[1+bs*1] = A[1+bs*1];
			C[2+bs*1] = A[2+bs*1];
			C[3+bs*1] = A[3+bs*1];

			C[0+bs*2] = A[0+bs*2];
			C[1+bs*2] = A[1+bs*2];
			C[2+bs*2] = A[2+bs*2];
			C[3+bs*2] = A[3+bs*2];

			C[0+bs*3] = A[0+bs*3];
			C[1+bs*3] = A[1+bs*3];
			C[2+bs*3] = A[2+bs*3];
			C[3+bs*3] = A[3+bs*3];

			A += bs*sda;
			C += bs*sdc;
			}

		// clean-up loop
		for( ; k<kmax; k++)
			{
			C[0+bs*0] = A[0+bs*0];

			C[0+bs*1] = A[0+bs*1];

			C[0+bs*2] = A[0+bs*2];

			C[0+bs*3] = A[0+bs*3];

			A += 1;
			C += 1;
			}

		return;

		}
	else if(kna==1)
		{

		// one row in the first C panel
		C[0+bs*0] = A[0+bs*0];

		C += 1 + bs*(sdc-1);

		C[0+bs*0] = A[1+bs*0];
		C[1+bs*0] = A[2+bs*0];
		C[2+bs*0] = A[3+bs*0];

		C[0+bs*1] = A[1+bs*1];
		C[1+bs*1] = A[2+bs*1];
		C[2+bs*1] = A[3+bs*1];

		C[1+bs*2] = A[2+bs*2];
		C[2+bs*2] = A[3+bs*2];

		C[2+bs*3] = A[3+bs*3];

		A += bs*sda;
		k = 4;
	
		// main loop
		for( ; k<kmax-3; k+=4)
			{
			C[3+bs*0] = A[0+bs*0];

			C[3+bs*1] = A[0+bs*1];

			C[3+bs*2] = A[0+bs*2];

			C[3+bs*3] = A[0+bs*3];

			C += bs*sdc;

			C[0+bs*0] = A[1+bs*0];
			C[1+bs*0] = A[2+bs*0];
			C[2+bs*0] = A[3+bs*0];

			C[0+bs*1] = A[1+bs*1];
			C[1+bs*1] = A[2+bs*1];
			C[2+bs*1] = A[3+bs*1];

			C[0+bs*2] = A[1+bs*2];
			C[1+bs*2] = A[2+bs*2];
			C[2+bs*2] = A[3+bs*2];

			C[0+bs*3] = A[1+bs*3];
			C[1+bs*3] = A[2+bs*3];
			C[2+bs*3] = A[3+bs*3];

			A += bs*sda;
			}

		if(k>=kmax)
			{
			return;
			}

		C[3+bs*0] = A[0+bs*0];

		C[3+bs*1] = A[0+bs*1];

		C[3+bs*2] = A[0+bs*2];

		C[3+bs*3] = A[0+bs*3];

		k++;

		if(k==kmax)
			{
			return;
			}

		C += bs*sdc;

		C[0+bs*0] = A[1+bs*0];

		C[0+bs*1] = A[1+bs*1];

		C[0+bs*2] = A[1+bs*2];

		C[0+bs*3] = A[1+bs*3];

		k++;

		if(k==kmax)
			{
			return;
			}

		C[1+bs*0] = A[2+bs*0];

		C[1+bs*1] = A[2+bs*1];

		C[1+bs*2] = A[2+bs*2];

		C[1+bs*3] = A[2+bs*3];

		return;

		}
	else if(kna==2)
		{

		// two rows in the first C panel
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];

		C[1+bs*1] = A[1+bs*1];

		C += 2 + bs*(sdc-1);

		C[0+bs*0] = A[2+bs*0];
		C[1+bs*0] = A[3+bs*0];

		C[0+bs*1] = A[2+bs*1];
		C[1+bs*1] = A[3+bs*1];

		C[0+bs*2] = A[2+bs*2];
		C[1+bs*2] = A[3+bs*2];

		C[1+bs*3] = A[3+bs*3];

		A += bs*sda;
		k = 4;
	
		// main loop
		for( ; k<kmax-3; k+=4)
			{
			C[2+bs*0] = A[0+bs*0];
			C[3+bs*0] = A[1+bs*0];

			C[2+bs*1] = A[0+bs*1];
			C[3+bs*1] = A[1+bs*1];

			C[2+bs*2] = A[0+bs*2];
			C[3+bs*2] = A[1+bs*2];

			C[2+bs*3] = A[0+bs*3];
			C[3+bs*3] = A[1+bs*3];

			C += bs*sdc;

			C[0+bs*0] = A[2+bs*0];
			C[1+bs*0] = A[3+bs*0];

			C[0+bs*1] = A[2+bs*1];
			C[1+bs*1] = A[3+bs*1];

			C[0+bs*2] = A[2+bs*2];
			C[1+bs*2] = A[3+bs*2];

			C[0+bs*3] = A[2+bs*3];
			C[1+bs*3] = A[3+bs*3];

			A += bs*sda;
			}

		if(k>=kmax)
			{
			return;
			}

		C[2+bs*0] = A[0+bs*0];

		C[2+bs*1] = A[0+bs*1];

		C[2+bs*2] = A[0+bs*2];

		C[2+bs*3] = A[0+bs*3];

		k++;

		if(k==kmax)
			{
			return;
			}

		C[3+bs*0] = A[1+bs*0];

		C[3+bs*1] = A[1+bs*1];

		C[3+bs*2] = A[1+bs*2];

		C[3+bs*3] = A[1+bs*3];

		k++;

		if(k==kmax)
			{
			return;
			}

		C += bs*sdc;

		C[0+bs*0] = A[3+bs*0];

		C[0+bs*1] = A[3+bs*1];

		C[0+bs*2] = A[3+bs*2];

		C[0+bs*3] = A[3+bs*3];

		return;

		}
	else  //if(kna==3)
		{

		// three rows in the first C panel
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];
		C[2+bs*0] = A[2+bs*0];

		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[2+bs*1];

		C[2+bs*2] = A[2+bs*2];

		C += 3 + bs*(sdc-1);

		C[0+bs*0] = A[3+bs*0];

		C[0+bs*1] = A[3+bs*1];

		C[0+bs*2] = A[3+bs*2];

		C[0+bs*3] = A[3+bs*3];

		A += bs*sda;
		k = 4;

		// main loop
		for( ; k<kmax-3; k+=4)
			{
			C[1+bs*0] = A[0+bs*0];
			C[2+bs*0] = A[1+bs*0];
			C[3+bs*0] = A[2+bs*0];

			C[1+bs*1] = A[0+bs*1];
			C[2+bs*1] = A[1+bs*1];
			C[3+bs*1] = A[2+bs*1];

			C[1+bs*2] = A[0+bs*2];
			C[2+bs*2] = A[1+bs*2];
			C[3+bs*2] = A[2+bs*2];

			C[1+bs*3] = A[0+bs*3];
			C[2+bs*3] = A[1+bs*3];
			C[3+bs*3] = A[2+bs*3];

			C += bs*sdc;

			C[0+bs*0] = A[3+bs*0];

			C[0+bs*1] = A[3+bs*1];

			C[0+bs*2] = A[3+bs*2];

			C[0+bs*3] = A[3+bs*3];

			A += bs*sda;
			}

		if(k>=kmax)
			{
			return;
			}

		C[1+bs*0] = A[0+bs*0];

		C[1+bs*1] = A[0+bs*1];

		C[1+bs*2] = A[0+bs*2];

		C[1+bs*3] = A[0+bs*3];

		k++;


		if(k==kmax)
			{
			return;
			}

		C[2+bs*0] = A[1+bs*0];

		C[2+bs*1] = A[1+bs*1];

		C[2+bs*2] = A[1+bs*2];

		C[2+bs*3] = A[1+bs*3];

		k++;


		if(k==kmax)
			{
			return;
			}

		C[3+bs*0] = A[2+bs*0];

		C[3+bs*1] = A[2+bs*1];

		C[3+bs*2] = A[2+bs*2];

		C[3+bs*3] = A[2+bs*3];

		return;

		}

	}



// mis-align a triangolar matrix; it moves across panels; read aligned, write mis-aligned
void corner_dtrma_3_lib4(int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;

	if(kna==0 || kna==3)
		{
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];
		C[2+bs*0] = A[2+bs*0];

		C[1+bs*1] = A[1+bs*1];
		C[2+bs*1] = A[2+bs*1];

		C[2+bs*2] = A[2+bs*2];

		return;
		}
	else if(kna==1)
		{
		C[0+bs*0] = A[0+bs*0];

		C += 1 + bs*(sdc-1);

		C[0+bs*0] = A[1+bs*0];
		C[1+bs*0] = A[2+bs*0];

		C[0+bs*1] = A[1+bs*1];
		C[1+bs*1] = A[2+bs*1];

		C[1+bs*2] = A[2+bs*2];

		return;
		}
	else //if(kna==2)
		{
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];

		C[1+bs*1] = A[1+bs*1];

		C += 2 + bs*(sdc-1);

		C[0+bs*0] = A[2+bs*0];

		C[0+bs*1] = A[2+bs*1];

		C[0+bs*2] = A[2+bs*2];

		return;
		}

	}




// mis-align a triangolar matrix; it moves across panels; read aligned, write mis-aligned
void corner_dtrma_2_lib4(int kna, double *A, double *C, int sdc)
	{

	const int bs = 4;

	if(kna==0 || kna==2 || kna==3)
		{
		C[0+bs*0] = A[0+bs*0];
		C[1+bs*0] = A[1+bs*0];

		C[1+bs*1] = A[1+bs*1];

		return;
		}
	else //if(kna==1)
		{
		C[0+bs*0] = A[0+bs*0];

		C += 1 + bs*(sdc-1);

		C[0+bs*0] = A[1+bs*0];

		C[0+bs*1] = A[1+bs*1];

		return;
		}
	
	}

