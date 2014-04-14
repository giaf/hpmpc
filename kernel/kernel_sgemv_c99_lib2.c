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

// it moves vertically across blocks
/*void kernel_sgemv_t_4_c99_lib2(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)*/
/*	{*/
/*	*/
/*	if(kmax<=0) */
/*		return;*/
/*	*/
/*	const int lda = 2;*/
/*	const int bs  = 2;*/
/*	*/
/*	int*/
/*		k, ka=kmax-kna;*/
/*	*/
/*	float*/
/*		x_0, x_1,*/
/*		y_0=0, y_1=0, y_2=0, y_3=0;*/
/*	*/
/*	if(kna>0)*/
/*		{*/
/*		k=0;*/
/*		for(; k<kna; k++)*/
/*			{*/
/*		*/
/*			x_0 = x[0];*/
/*		*/
/*			y_0 += A[0+lda*0] * x_0;*/
/*			y_1 += A[0+lda*1] * x_0;*/
/*			y_2 += A[0+lda*2] * x_0;*/
/*			y_3 += A[0+lda*3] * x_0;*/
/*		*/
/*			A += 1;*/
/*			x += 1;*/
/*		*/
/*			}*/
/*	*/
/*		A += (sda-1)*lda;*/
/*		}*/

/*	k=0;*/
/*	for(; k<ka-bs+1; k+=bs)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*		x_1 = x[1];*/
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
/*		A += 2 + (sda-1)*lda;*/
/*		x += 2;*/

/*		}*/
/*	*/
/*	for(; k<ka; k++)*/
/*		{*/
/*		*/
/*		x_0 = x[0];*/
/*	*/
/*		y_0 += A[0+lda*0] * x_0;*/
/*		y_1 += A[0+lda*1] * x_0;*/
/*		y_2 += A[0+lda*2] * x_0;*/
/*		y_3 += A[0+lda*3] * x_0;*/
/*	*/
/*		A += 1;*/
/*		x += 1;*/
/*		*/
/*		}*/

/*	if(alg==0)*/
/*		{*/
/*		y[0] = y_0;*/
/*		y[1] = y_1;*/
/*		y[2] = y_2;*/
/*		y[3] = y_3;*/
/*		}*/
/*	else if(alg==1)*/
/*		{*/
/*		y[0] += y_0;*/
/*		y[1] += y_1;*/
/*		y[2] += y_2;*/
/*		y[3] += y_3;*/
/*		}*/
/*	else // alg==-1*/
/*		{*/
/*		y[0] -= y_0;*/
/*		y[1] -= y_1;*/
/*		y[2] -= y_2;*/
/*		y[3] -= y_3;*/
/*		}*/
/*	*/
/*	}*/



// it moves vertically across blocks
void kernel_sgemv_t_2_c99_lib2(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 2;
	const int bs  = 2;
	
	int
		k, ka=kmax-kna;
	
	float
		x_0, x_1,
		y_0=0, y_1=0;
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0 = x[0];
		
			y_0 += A[0+lda*0] * x_0;
			y_1 += A[0+lda*1] * x_0;
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-bs+1; k+=bs)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		
		y_0 += A[0+lda*0] * x_0;
		y_1 += A[0+lda*1] * x_0;

		y_0 += A[1+lda*0] * x_1;
		y_1 += A[1+lda*1] * x_1;
		
		A += 2 + (sda-1)*lda;
		x += 2;

		}
	
	for(; k<ka; k++)
		{
		
		x_0 = x[0];
	
		y_0 += A[0+lda*0] * x_0;
		y_1 += A[0+lda*1] * x_0;
	
		A += 1;
		x += 1;
		
		}

	if(alg==0)
		{
		y[0] = y_0;
		y[1] = y_1;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		y[1] += y_1;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		y[1] -= y_1;
		}
	
	}



// it moves vertically across blocks
void kernel_sgemv_t_1_c99_lib2(int kmax, int kna, float *A, int sda, float *x, float *y, int alg)
	{
	
	if(kmax<=0) 
		return;
	
	const int lda = 2;
	const int bs  = 2;
	
	int
		k, ka=kmax-kna;
	
	float
		x_0, x_1,
		y_0=0;
	
	if(kna>0)
		{
		k=0;
		for(; k<kna; k++)
			{
		
			x_0 = x[0];
		
			y_0 += A[0+lda*0] * x_0;
		
			A += 1;
			x += 1;
		
			}
	
		A += (sda-1)*lda;
		}

	k=0;
	for(; k<ka-bs+1; k+=bs)
		{
		
		x_0 = x[0];
		x_1 = x[1];
		
		y_0 += A[0+lda*0] * x_0;
		y_0 += A[1+lda*0] * x_1;
		
		A += 2 + (sda-1)*lda;
		x += 2;

		}
	
	for(; k<ka; k++)
		{
		
		x_0 = x[0];
	
		y_0 += A[0+lda*0] * x_0;
	
		A += 1;
		x += 1;
		
		}

	if(alg==0)
		{
		y[0] = y_0;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		}
	
	}



// it moves horizontally inside a block
void kernel_sgemv_n_2_c99_lib2(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 2;
	
	int k;

	float
		x_0, x_1, x_2, x_3,
		y_0=0, y_1=0;

	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 += A[0+lda*0] * x_0;
		y_1 += A[1+lda*0] * x_0;
		y_0 += A[0+lda*1] * x_1;
		y_1 += A[1+lda*1] * x_1;
		y_0 += A[0+lda*2] * x_2;
		y_1 += A[1+lda*2] * x_2;
		y_0 += A[0+lda*3] * x_3;
		y_1 += A[1+lda*3] * x_3;
		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+lda*0] * x_0;
		y_1 += A[1+lda*0] * x_0;
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		y[0] = y_0;
		y[1] = y_1;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		y[1] += y_1;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		y[1] -= y_1;
		}

	}



// it moves horizontally inside a block
void kernel_sgemv_n_1_c99_lib2(int kmax, float *A, float *x, float *y, int alg)
	{
	if(kmax<=0) 
		return;
	
	const int lda = 2;
	
	int k;

	float
		x_0, x_1, x_2, x_3,
		y_0=0;
	
	k=0;
	for(; k<kmax-3; k+=4)
		{

		x_0 = x[0];
		x_1 = x[1];
		x_2 = x[2];
		x_3 = x[3];

		y_0 += A[0+lda*0] * x_0;
		y_0 += A[0+lda*1] * x_1;
		y_0 += A[0+lda*2] * x_2;
		y_0 += A[0+lda*3] * x_3;
		
		A += 4*lda;
		x += 4;

		}

	for(; k<kmax; k++)
		{

		x_0 = x[0];

		y_0 += A[0+lda*0] * x_0;
		
		A += 1*lda;
		x += 1;

		}

	if(alg==0)
		{
		y[0] = y_0;
		}
	else if(alg==1)
		{
		y[0] += y_0;
		}
	else // alg==-1
		{
		y[0] -= y_0;
		}

	}

