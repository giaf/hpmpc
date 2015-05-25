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

	}



// both A and B are aligned to 256-bit boundaries, 1 element of A must be skipped
void kernel_align_panel_4_1_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_4_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_4_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_3_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	}



// both A and B are aligned to 256-bit boundaries, 2 elements of A must be skipped
void kernel_align_panel_3_2_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned to 256-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_3_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned to 64-bit boundaries
void kernel_align_panel_2_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	}



// both A and B are aligned to 128-bit boundaries, 3 elements of A must be skipped
void kernel_align_panel_2_3_lib4(int kmax, double *A0, int sda, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	double *A1 = A0 + bs*sda;

	}



// both A and B are aligned 64-bit boundaries
void kernel_align_panel_1_0_lib4(int kmax, double *A, double *B)
	{

	if(kmax<=0)
		return;

	const int bs = 4;

	}




