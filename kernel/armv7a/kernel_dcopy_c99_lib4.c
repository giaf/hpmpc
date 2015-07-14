/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014-2015 by Technical University of Denmark. All rights reserved.                *
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



// both A and B are aligned to 256-bit boundaries
void kernel_align_panel_4_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];
		B[2+bs*0] = A[2+bs*0];
		B[3+bs*0] = A[3+bs*0];

		B[0+bs*1] = A[0+bs*1];
		B[1+bs*1] = A[1+bs*1];
		B[2+bs*1] = A[2+bs*1];
		B[3+bs*1] = A[3+bs*1];

		B[0+bs*2] = A[0+bs*2];
		B[1+bs*2] = A[1+bs*2];
		B[2+bs*2] = A[2+bs*2];
		B[3+bs*2] = A[3+bs*2];

		B[0+bs*3] = A[0+bs*3];
		B[1+bs*3] = A[1+bs*3];
		B[2+bs*3] = A[2+bs*3];
		B[3+bs*3] = A[3+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];
		B[2+bs*0] = A[2+bs*0];
		B[3+bs*0] = A[3+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 1 element of A must be skipped
void kernel_align_panel_4_1_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[1+bs*0];
		B[1+bs*0] = A0[2+bs*0];
		B[2+bs*0] = A0[3+bs*0];
		B[3+bs*0] = A1[0+bs*0];

		B[0+bs*1] = A0[1+bs*1];
		B[1+bs*1] = A0[2+bs*1];
		B[2+bs*1] = A0[3+bs*1];
		B[3+bs*1] = A1[0+bs*1];

		B[0+bs*2] = A0[1+bs*2];
		B[1+bs*2] = A0[2+bs*2];
		B[2+bs*2] = A0[3+bs*2];
		B[3+bs*2] = A1[0+bs*2];

		B[0+bs*3] = A0[1+bs*3];
		B[1+bs*3] = A0[2+bs*3];
		B[2+bs*3] = A0[3+bs*3];
		B[3+bs*3] = A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[1+bs*0];
		B[1+bs*0] = A0[2+bs*0];
		B[2+bs*0] = A0[3+bs*0];
		B[3+bs*0] = A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_4_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[2+bs*0];
		B[1+bs*0] = A0[3+bs*0];
		B[2+bs*0] = A1[0+bs*0];
		B[3+bs*0] = A1[1+bs*0];

		B[0+bs*1] = A0[2+bs*1];
		B[1+bs*1] = A0[3+bs*1];
		B[2+bs*1] = A1[0+bs*1];
		B[3+bs*1] = A1[1+bs*1];

		B[0+bs*2] = A0[2+bs*2];
		B[1+bs*2] = A0[3+bs*2];
		B[2+bs*2] = A1[0+bs*2];
		B[3+bs*2] = A1[1+bs*2];

		B[0+bs*3] = A0[2+bs*3];
		B[1+bs*3] = A0[3+bs*3];
		B[2+bs*3] = A1[0+bs*3];
		B[3+bs*3] = A1[1+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[2+bs*0];
		B[1+bs*0] = A0[3+bs*0];
		B[2+bs*0] = A1[0+bs*0];
		B[3+bs*0] = A1[1+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_4_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];
		B[2+bs*0] = A1[1+bs*0];
		B[3+bs*0] = A1[2+bs*0];

		B[0+bs*1] = A0[3+bs*1];
		B[1+bs*1] = A1[0+bs*1];
		B[2+bs*1] = A1[1+bs*1];
		B[3+bs*1] = A1[2+bs*1];

		B[0+bs*2] = A0[3+bs*2];
		B[1+bs*2] = A1[0+bs*2];
		B[2+bs*2] = A1[1+bs*2];
		B[3+bs*2] = A1[2+bs*2];

		B[0+bs*3] = A0[3+bs*3];
		B[1+bs*3] = A1[0+bs*3];
		B[2+bs*3] = A1[1+bs*3];
		B[3+bs*3] = A1[2+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];
		B[2+bs*0] = A1[1+bs*0];
		B[3+bs*0] = A1[2+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_3_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];
		B[2+bs*0] = A[2+bs*0];

		B[0+bs*1] = A[0+bs*1];
		B[1+bs*1] = A[1+bs*1];
		B[2+bs*1] = A[2+bs*1];

		B[0+bs*2] = A[0+bs*2];
		B[1+bs*2] = A[1+bs*2];
		B[2+bs*2] = A[2+bs*2];

		B[0+bs*3] = A[0+bs*3];
		B[1+bs*3] = A[1+bs*3];
		B[2+bs*3] = A[2+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];
		B[2+bs*0] = A[2+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_3_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[2+bs*0];
		B[1+bs*0] = A0[3+bs*0];
		B[2+bs*0] = A1[0+bs*0];

		B[0+bs*1] = A0[2+bs*1];
		B[1+bs*1] = A0[3+bs*1];
		B[2+bs*1] = A1[0+bs*1];

		B[0+bs*2] = A0[2+bs*2];
		B[1+bs*2] = A0[3+bs*2];
		B[2+bs*2] = A1[0+bs*2];

		B[0+bs*3] = A0[2+bs*3];
		B[1+bs*3] = A0[3+bs*3];
		B[2+bs*3] = A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[2+bs*0];
		B[1+bs*0] = A0[3+bs*0];
		B[2+bs*0] = A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_3_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];
		B[2+bs*0] = A1[1+bs*0];

		B[0+bs*1] = A0[3+bs*1];
		B[1+bs*1] = A1[0+bs*1];
		B[2+bs*1] = A1[1+bs*1];

		B[0+bs*2] = A0[3+bs*2];
		B[1+bs*2] = A1[0+bs*2];
		B[2+bs*2] = A1[1+bs*2];

		B[0+bs*3] = A0[3+bs*3];
		B[1+bs*3] = A1[0+bs*3];
		B[2+bs*3] = A1[1+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];
		B[2+bs*0] = A1[1+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_2_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];

		B[0+bs*1] = A[0+bs*1];
		B[1+bs*1] = A[1+bs*1];

		B[0+bs*2] = A[0+bs*2];
		B[1+bs*2] = A[1+bs*2];

		B[0+bs*3] = A[0+bs*3];
		B[1+bs*3] = A[1+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A[0+bs*0];
		B[1+bs*0] = A[1+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 128-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_2_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];

		B[0+bs*1] = A0[3+bs*1];
		B[1+bs*1] = A1[0+bs*1];

		B[0+bs*2] = A0[3+bs*2];
		B[1+bs*2] = A1[0+bs*2];

		B[0+bs*3] = A0[3+bs*3];
		B[1+bs*3] = A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A0[3+bs*0];
		B[1+bs*0] = A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned 64-bit boundaries
void kernel_align_panel_1_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] = A[0+bs*0];

		B[0+bs*1] = A[0+bs*1];

		B[0+bs*2] = A[0+bs*2];

		B[0+bs*3] = A[0+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] = A[0+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries
void kernel_add_align_panel_4_0_lib4(int kmax, double alpha, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];
		B[2+bs*0] += alpha * A[2+bs*0];
		B[3+bs*0] += alpha * A[3+bs*0];

		B[0+bs*1] += alpha * A[0+bs*1];
		B[1+bs*1] += alpha * A[1+bs*1];
		B[2+bs*1] += alpha * A[2+bs*1];
		B[3+bs*1] += alpha * A[3+bs*1];

		B[0+bs*2] += alpha * A[0+bs*2];
		B[1+bs*2] += alpha * A[1+bs*2];
		B[2+bs*2] += alpha * A[2+bs*2];
		B[3+bs*2] += alpha * A[3+bs*2];

		B[0+bs*3] += alpha * A[0+bs*3];
		B[1+bs*3] += alpha * A[1+bs*3];
		B[2+bs*3] += alpha * A[2+bs*3];
		B[3+bs*3] += alpha * A[3+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];
		B[2+bs*0] += alpha * A[2+bs*0];
		B[3+bs*0] += alpha * A[3+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 1 element of A must be skipped
void kernel_add_align_panel_4_1_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[1+bs*0];
		B[1+bs*0] += alpha * A0[2+bs*0];
		B[2+bs*0] += alpha * A0[3+bs*0];
		B[3+bs*0] += alpha * A1[0+bs*0];

		B[0+bs*1] += alpha * A0[1+bs*1];
		B[1+bs*1] += alpha * A0[2+bs*1];
		B[2+bs*1] += alpha * A0[3+bs*1];
		B[3+bs*1] += alpha * A1[0+bs*1];

		B[0+bs*2] += alpha * A0[1+bs*2];
		B[1+bs*2] += alpha * A0[2+bs*2];
		B[2+bs*2] += alpha * A0[3+bs*2];
		B[3+bs*2] += alpha * A1[0+bs*2];

		B[0+bs*3] += alpha * A0[1+bs*3];
		B[1+bs*3] += alpha * A0[2+bs*3];
		B[2+bs*3] += alpha * A0[3+bs*3];
		B[3+bs*3] += alpha * A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[1+bs*0];
		B[1+bs*0] += alpha * A0[2+bs*0];
		B[2+bs*0] += alpha * A0[3+bs*0];
		B[3+bs*0] += alpha * A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_add_align_panel_4_2_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[2+bs*0];
		B[1+bs*0] += alpha * A0[3+bs*0];
		B[2+bs*0] += alpha * A1[0+bs*0];
		B[3+bs*0] += alpha * A1[1+bs*0];

		B[0+bs*1] += alpha * A0[2+bs*1];
		B[1+bs*1] += alpha * A0[3+bs*1];
		B[2+bs*1] += alpha * A1[0+bs*1];
		B[3+bs*1] += alpha * A1[1+bs*1];

		B[0+bs*2] += alpha * A0[2+bs*2];
		B[1+bs*2] += alpha * A0[3+bs*2];
		B[2+bs*2] += alpha * A1[0+bs*2];
		B[3+bs*2] += alpha * A1[1+bs*2];

		B[0+bs*3] += alpha * A0[2+bs*3];
		B[1+bs*3] += alpha * A0[3+bs*3];
		B[2+bs*3] += alpha * A1[0+bs*3];
		B[3+bs*3] += alpha * A1[1+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[2+bs*0];
		B[1+bs*0] += alpha * A0[3+bs*0];
		B[2+bs*0] += alpha * A1[0+bs*0];
		B[3+bs*0] += alpha * A1[1+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_add_align_panel_4_3_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];
		B[2+bs*0] += alpha * A1[1+bs*0];
		B[3+bs*0] += alpha * A1[2+bs*0];

		B[0+bs*1] += alpha * A0[3+bs*1];
		B[1+bs*1] += alpha * A1[0+bs*1];
		B[2+bs*1] += alpha * A1[1+bs*1];
		B[3+bs*1] += alpha * A1[2+bs*1];

		B[0+bs*2] += alpha * A0[3+bs*2];
		B[1+bs*2] += alpha * A1[0+bs*2];
		B[2+bs*2] += alpha * A1[1+bs*2];
		B[3+bs*2] += alpha * A1[2+bs*2];

		B[0+bs*3] += alpha * A0[3+bs*3];
		B[1+bs*3] += alpha * A1[0+bs*3];
		B[2+bs*3] += alpha * A1[1+bs*3];
		B[3+bs*3] += alpha * A1[2+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];
		B[2+bs*0] += alpha * A1[1+bs*0];
		B[3+bs*0] += alpha * A1[2+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_add_align_panel_3_0_lib4(int kmax, double alpha, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];
		B[2+bs*0] += alpha * A[2+bs*0];

		B[0+bs*1] += alpha * A[0+bs*1];
		B[1+bs*1] += alpha * A[1+bs*1];
		B[2+bs*1] += alpha * A[2+bs*1];

		B[0+bs*2] += alpha * A[0+bs*2];
		B[1+bs*2] += alpha * A[1+bs*2];
		B[2+bs*2] += alpha * A[2+bs*2];

		B[0+bs*3] += alpha * A[0+bs*3];
		B[1+bs*3] += alpha * A[1+bs*3];
		B[2+bs*3] += alpha * A[2+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];
		B[2+bs*0] += alpha * A[2+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_add_align_panel_3_2_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[2+bs*0];
		B[1+bs*0] += alpha * A0[3+bs*0];
		B[2+bs*0] += alpha * A1[0+bs*0];

		B[0+bs*1] += alpha * A0[2+bs*1];
		B[1+bs*1] += alpha * A0[3+bs*1];
		B[2+bs*1] += alpha * A1[0+bs*1];

		B[0+bs*2] += alpha * A0[2+bs*2];
		B[1+bs*2] += alpha * A0[3+bs*2];
		B[2+bs*2] += alpha * A1[0+bs*2];

		B[0+bs*3] += alpha * A0[2+bs*3];
		B[1+bs*3] += alpha * A0[3+bs*3];
		B[2+bs*3] += alpha * A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[2+bs*0];
		B[1+bs*0] += alpha * A0[3+bs*0];
		B[2+bs*0] += alpha * A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_add_align_panel_3_3_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];
		B[2+bs*0] += alpha * A1[1+bs*0];

		B[0+bs*1] += alpha * A0[3+bs*1];
		B[1+bs*1] += alpha * A1[0+bs*1];
		B[2+bs*1] += alpha * A1[1+bs*1];

		B[0+bs*2] += alpha * A0[3+bs*2];
		B[1+bs*2] += alpha * A1[0+bs*2];
		B[2+bs*2] += alpha * A1[1+bs*2];

		B[0+bs*3] += alpha * A0[3+bs*3];
		B[1+bs*3] += alpha * A1[0+bs*3];
		B[2+bs*3] += alpha * A1[1+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];
		B[2+bs*0] += alpha * A1[1+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned to 64-bit boundaries
void kernel_add_align_panel_2_0_lib4(int kmax, double alpha, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];

		B[0+bs*1] += alpha * A[0+bs*1];
		B[1+bs*1] += alpha * A[1+bs*1];

		B[0+bs*2] += alpha * A[0+bs*2];
		B[1+bs*2] += alpha * A[1+bs*2];

		B[0+bs*3] += alpha * A[0+bs*3];
		B[1+bs*3] += alpha * A[1+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A[0+bs*0];
		B[1+bs*0] += alpha * A[1+bs*0];

		A += 4;
		B += 4;

		}

	}



// both A and B are aligned to 128-bit boundaries, 3 elements of A must be skipped
void kernel_add_align_panel_2_3_lib4(int kmax, double alpha, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];

		B[0+bs*1] += alpha * A0[3+bs*1];
		B[1+bs*1] += alpha * A1[0+bs*1];

		B[0+bs*2] += alpha * A0[3+bs*2];
		B[1+bs*2] += alpha * A1[0+bs*2];

		B[0+bs*3] += alpha * A0[3+bs*3];
		B[1+bs*3] += alpha * A1[0+bs*3];

		A0 += 16;
		A1 += 16;
		B  += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A0[3+bs*0];
		B[1+bs*0] += alpha * A1[0+bs*0];

		A0 += 4;
		A1 += 4;
		B  += 4;

		}

	}



// both A and B are aligned 64-bit boundaries
void kernel_add_align_panel_1_0_lib4(int kmax, double alpha, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	int k;

	for(k=0; k<kmax-3; k+=4)
		{
		B[0+bs*0] += alpha * A[0+bs*0];

		B[0+bs*1] += alpha * A[0+bs*1];

		B[0+bs*2] += alpha * A[0+bs*2];

		B[0+bs*3] += alpha * A[0+bs*3];

		A += 16;
		B += 16;

		}
	for(; k<kmax; k++)
		{

		B[0+bs*0] += alpha * A[0+bs*0];

		A += 4;
		B += 4;

		}

	}





