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

#include <stdlib.h>
#include <stdio.h>

int posix_memalign(void **memptr, size_t alignment, size_t size);



/* creates a zero matrix aligned */
void d_zeros(double **pA, int row, int col)
	{
	void *temp = malloc((row*col)*sizeof(double));
	*pA = temp;
	double *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* creates a zero matrix aligned to a cache line */
void d_zeros_align(double **pA, int row, int col)
	{
	void *temp;
	int err = posix_memalign(&temp, 64, (row*col)*sizeof(double));
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*pA = temp;
	double *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 0.0;
	}



/* creates a zero matrix aligned */
void d_ones(double **pA, int row, int col)
	{
	void *temp = malloc((row*col)*sizeof(double));
	*pA = temp;
	double *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 1.0;
	}



/* creates a zero matrix aligned to a cache line */
void d_ones_align(double **pA, int row, int col)
	{
	void *temp;
	int err = posix_memalign(&temp, 64, (row*col)*sizeof(double));
	if(err!=0)
		{
		printf("Memory allocation error");
		exit(1);
		}
	*pA = temp;
	double *A = *pA;
	int i;
	for(i=0; i<row*col; i++) A[i] = 1.0;
	}



/* creates a zero matrix aligned */
void d_eye(double **pA, int row)
	{
	void *temp = malloc((row*row)*sizeof(double));
	*pA = temp;
	double *A = *pA;
	int i;
	for(i=0; i<row*row; i++) A[i] = 0.0;
	for(i=0; i<row; i++) A[i*(row+1)] = 1.0;
	}



/* copies a matrix */
void d_rep_mat(int reps, int row, int col, double *A, int lda, double *B, int ldb)
	{
	
	int i, j, l;
	
	for(l=0; l<reps; l++)
		{
		for(j=0; j<col; j++)
			{
			for(i=0; i<row; i++)
				{
				B[i+j*ldb+l*row*col] = A[i+j*lda];
				}
			}
		}
		
	}



/* copies a matrix */
void d_copy_mat(int row, int col, double *A, int lda, double *B, int ldb)
	{
	
	int i, j;
	
	for(j=0; j<col; j++)
		{
		for(i=0; i<row; i++)
			{
			B[i+j*ldb] = A[i+j*lda];
			}
		}
	
	}



/* copies a packed matrix */
void d_copy_pmat(int row, int col, int bs_dummy, double *A, int sda, double *B, int sdb)
	{
	
	const int bs = 4;
	
	int i, ii, j, row2;
	
	ii = 0;
	for(; ii<row-3; ii+=bs)
		{
		for(j=0; j<col; j++)
			{
			B[0+j*bs+ii*sdb] = A[0+j*bs+ii*sda];
			B[1+j*bs+ii*sdb] = A[1+j*bs+ii*sda];
			B[2+j*bs+ii*sdb] = A[2+j*bs+ii*sda];
			B[3+j*bs+ii*sdb] = A[3+j*bs+ii*sda];
			}
		}
	if(ii<row)
		{
		row2 = row-ii;
		for(j=0; j<col; j++)
			{
			for(i=0; i<row2; i++)
				{
				B[i+j*bs+ii*sdb] = A[i+j*bs+ii*sda];
				}
			}
		}
	
	}



// copies a lower triangular packed matrix 
void d_copy_pmat_l(int row, int bs_dummy, double *A, int sda, double *B, int sdb)
	{
	
	const int bs = 4;

	int i, ii, j, row2, row0;
	
	ii = 0;
	for(; ii<row-3; ii+=bs)
		{
		j = 0;
		for(; j<ii; j++)
			{
			B[0+j*bs+ii*sdb] = A[0+j*bs+ii*sda];
			B[1+j*bs+ii*sdb] = A[1+j*bs+ii*sda];
			B[2+j*bs+ii*sdb] = A[2+j*bs+ii*sda];
			B[3+j*bs+ii*sdb] = A[3+j*bs+ii*sda];
			}
		for(; j<ii+bs; j++)
			{
			row0 = j-ii;
			if(row0<0) row0=0;
			for(i=row0; i<bs; i++)
				{
				B[i+j*bs+ii*sdb] = A[i+j*bs+ii*sda];
				}
			}
		}
	if(ii<row)
		{
		row2 = row-ii;
		if(bs<row2) row2 = bs;
		for(j=0; j<ii+row2; j++)
			{
			row0 = j-ii;
			if(row0<0) row0=0;
			for(i=row0; i<row2; i++)
				{
				B[i+j*bs+ii*sdb] = A[i+j*bs+ii*sda];
				}
			}
		}
	
	}



/* transposes a lower triangular packed matrix */
/*void d_transpose_pmat_lo(int row, int offset, double *A, int sda, double *B, int sdb)*/
/*	{*/
/*	*/
/*	const int bs = 4;*/

/*	int i, j, jj;*/
/*	*/
/*	int row0, row1, row2, row3;*/
/*	row0 = (bs-offset%bs)%bs; // row2 < bs !!!*/
/*	*/
/*	double *pA, *pB;*/

/*	jj = 0;*/
/*	for(; jj<row-3; jj+=4)*/
/*		{*/
/*		row1 = row - jj;*/
/*		pA = A + jj*bs + jj*sda;*/
/*		pB = B + jj*bs + jj*sdb;*/
/*		row2 = row0; // row2 < bs !!!*/
/*		if(row1<row2)*/
/*			row2 = row1;*/
/*		i = 0;*/
/*		if(row2>0)*/
/*			{*/
/*			for(; i<row2; i++)*/
/*				{*/
/*				for(j=0; j<=i; j++)*/
/*					{*/
/*					pB[j] = pA[j*bs];*/
/*					}*/
/*				pA += 1;*/
/*				pB += bs;*/
/*				}*/
/*			pA += (sda-1)*bs;*/
/*			}*/
/*		row3 = row2 + 4;*/
/*		if(row1<row3)*/
/*			row3 = row1;*/
/*		row2 = 4;*/
/*		if(row1<row2)*/
/*			row2 = row1;*/
/*		for(; i<row2; i++)*/
/*			{*/
/*			for(j=0; j<=i; j++)*/
/*				{*/
/*				pB[j] = pA[j*bs];*/
/*				}*/
/*			pA += 1;*/
/*			pB += bs;*/
/*			}*/
/*		for(; i<row3; i++)*/
/*			{*/
/*			pB[0] = pA[0*bs];*/
/*			pB[1] = pA[1*bs];*/
/*			pB[2] = pA[2*bs];*/
/*			pB[3] = pA[3*bs];*/
/*			pA += 1;*/
/*			pB += bs;*/
/*			}*/
/*		pA += (sda-1)*bs;*/
/*		for(; i<row1-3; i+=4)*/
/*			{*/
/*			// buildin_prefetch*/
/*			// unroll 0*/
/*			pB[0+0*bs] = pA[0+0*bs];*/
/*			pB[1+0*bs] = pA[0+1*bs];*/
/*			pB[2+0*bs] = pA[0+2*bs];*/
/*			pB[3+0*bs] = pA[0+3*bs];*/
/*			// unroll 1*/
/*			pB[0+1*bs] = pA[1+0*bs];*/
/*			pB[1+1*bs] = pA[1+1*bs];*/
/*			pB[2+1*bs] = pA[1+2*bs];*/
/*			pB[3+1*bs] = pA[1+3*bs];*/
/*			// unroll 2*/
/*			pB[0+2*bs] = pA[2+0*bs];*/
/*			pB[1+2*bs] = pA[2+1*bs];*/
/*			pB[2+2*bs] = pA[2+2*bs];*/
/*			pB[3+2*bs] = pA[2+3*bs];*/
/*			// unroll 3*/
/*			pB[0+3*bs] = pA[3+0*bs];*/
/*			pB[1+3*bs] = pA[3+1*bs];*/
/*			pB[2+3*bs] = pA[3+2*bs];*/
/*			pB[3+3*bs] = pA[3+3*bs];*/
/*			pA += sda*bs;*/
/*			pB += 4*bs;*/
/*			}*/
/*		for(; i<row1; i++)*/
/*			{*/
/*			pB[0] = pA[0*bs];*/
/*			pB[1] = pA[1*bs];*/
/*			pB[2] = pA[2*bs];*/
/*			pB[3] = pA[3*bs];*/
/*			pA += 1;*/
/*			pB += bs;*/
/*			}*/
/*		}*/
/*	if(jj<row)*/
/*		{*/
/*		row1 = row - jj;*/
/*		pA = A + jj*bs + jj*sda;*/
/*		pB = B + jj*bs + jj*sdb;*/
/*		row2 = row0; // row2 < bs !!!*/
/*		if(row1<row2)*/
/*			row2 = row1;*/
/*		i = 0;*/
/*		if(row2>0)*/
/*			{*/
/*			for(; i<row2; i++)*/
/*				{*/
/*				for(j=0; j<=i; j++)*/
/*					{*/
/*					pB[j] = pA[j*bs];*/
/*					}*/
/*				pA += 1;*/
/*				pB += bs;*/
/*				}*/
/*			pA += (sda-1)*bs;*/
/*			}*/
/*		row2 = 4;*/
/*		if(row1<row2)*/
/*			row2 = row1;*/
/*		for(; i<row2; i++)*/
/*			{*/
/*			for(j=0; j<=i; j++)*/
/*				{*/
/*				pB[j] = pA[j*bs];*/
/*				}*/
/*			pA += 1;*/
/*			pB += bs;*/
/*			}*/
/*		}*/
/*	*/
/*	}*/



// copies a packed matrix into an aligned packed matrix ; A has to be aligned at the beginning of the current block : the offset takes care of the row to be copied TODO IMPROVE IMPLEMENTATION !!!
void d_align_pmat(int row, int col, int offset, int bs_dummy, double *A, int sda, double *B, int sdb)
	{
	
	const int bs = 4;

	int i, j;
	
	double *ptrA, *ptrB;

	int miss_align = offset%bs;

	i = 0;
	if(miss_align==0) // B is aligned
		{
		for(; i<row-3; i+=4)
			{
			ptrA = A + (i+offset)*sda;
			ptrB = B + i*sdb;
			j = 0;
			for(; j<col-3; j+=4)
				{
				ptrB[0+(j+0)*bs] = ptrA[0+(j+0)*bs];
				ptrB[1+(j+0)*bs] = ptrA[1+(j+0)*bs];
				ptrB[2+(j+0)*bs] = ptrA[2+(j+0)*bs];
				ptrB[3+(j+0)*bs] = ptrA[3+(j+0)*bs];

				ptrB[0+(j+1)*bs] = ptrA[0+(j+1)*bs];
				ptrB[1+(j+1)*bs] = ptrA[1+(j+1)*bs];
				ptrB[2+(j+1)*bs] = ptrA[2+(j+1)*bs];
				ptrB[3+(j+1)*bs] = ptrA[3+(j+1)*bs];

				ptrB[0+(j+2)*bs] = ptrA[0+(j+2)*bs];
				ptrB[1+(j+2)*bs] = ptrA[1+(j+2)*bs];
				ptrB[2+(j+2)*bs] = ptrA[2+(j+2)*bs];
				ptrB[3+(j+2)*bs] = ptrA[3+(j+2)*bs];

				ptrB[0+(j+3)*bs] = ptrA[0+(j+3)*bs];
				ptrB[1+(j+3)*bs] = ptrA[1+(j+3)*bs];
				ptrB[2+(j+3)*bs] = ptrA[2+(j+3)*bs];
				ptrB[3+(j+3)*bs] = ptrA[3+(j+3)*bs];
				}
			for(; j<col; j++)
				{
				ptrB[0+j*bs] = ptrA[0+j*bs];
				ptrB[1+j*bs] = ptrA[1+j*bs];
				ptrB[2+j*bs] = ptrA[2+j*bs];
				ptrB[3+j*bs] = ptrA[3+j*bs];
				}
			}
		}
	for(; i<row; i++)
		{
		ptrA = A + ((offset+i)/bs)*bs*sda + ((offset+i)%bs);
		ptrB = B + (i/bs)*bs*sdb + (i%bs);
		j = 0;
		for(; j<col-3; j+=4)
			{
			ptrB[(j+0)*bs] = ptrA[(j+0)*bs];
			ptrB[(j+1)*bs] = ptrA[(j+1)*bs];
			ptrB[(j+2)*bs] = ptrA[(j+2)*bs];
			ptrB[(j+3)*bs] = ptrA[(j+3)*bs];
			}
		for(; j<col; j++)
			{
			ptrB[j*bs] = ptrA[j*bs];
			}
		}
	
	}



/* converts a matrix into a packed matrix */
void d_cvt_mat2pmat(int row, int col, int offset, int bs_dummy, double *A, int lda, double *pA, int sda)
	{
	
	const int bs = 4;

	int i, ii, j, row0, row1, row2;
	
	row0 = (bs-offset%bs)%bs;
	if(row0>row)
		row0 = row;
	row1 = row - row0;
	
	ii = 0;
	if(row0>0)
		{
		for(j=0; j<col; j++)
			{
			for(i=0; i<row0; i++)
				{
				pA[i+j*bs+ii*sda] = A[i+ii+j*lda];
				}
			}
	
		A  += row0;
		pA += row0 + bs*(sda-1);
		}
	
	ii = 0;
	for(; ii<row1-3; ii+=bs)
		{
		j=0;
		for(; j<col-3; j+=4)
			{
			// unroll 0
			pA[0+(j+0)*bs+ii*sda] = A[ii+0+(j+0)*lda];
			pA[1+(j+0)*bs+ii*sda] = A[ii+1+(j+0)*lda];
			pA[2+(j+0)*bs+ii*sda] = A[ii+2+(j+0)*lda];
			pA[3+(j+0)*bs+ii*sda] = A[ii+3+(j+0)*lda];
			// unroll 1
			pA[0+(j+1)*bs+ii*sda] = A[ii+0+(j+1)*lda];
			pA[1+(j+1)*bs+ii*sda] = A[ii+1+(j+1)*lda];
			pA[2+(j+1)*bs+ii*sda] = A[ii+2+(j+1)*lda];
			pA[3+(j+1)*bs+ii*sda] = A[ii+3+(j+1)*lda];
			// unroll 2
			pA[0+(j+2)*bs+ii*sda] = A[ii+0+(j+2)*lda];
			pA[1+(j+2)*bs+ii*sda] = A[ii+1+(j+2)*lda];
			pA[2+(j+2)*bs+ii*sda] = A[ii+2+(j+2)*lda];
			pA[3+(j+2)*bs+ii*sda] = A[ii+3+(j+2)*lda];
			// unroll 3
			pA[0+(j+3)*bs+ii*sda] = A[ii+0+(j+3)*lda];
			pA[1+(j+3)*bs+ii*sda] = A[ii+1+(j+3)*lda];
			pA[2+(j+3)*bs+ii*sda] = A[ii+2+(j+3)*lda];
			pA[3+(j+3)*bs+ii*sda] = A[ii+3+(j+3)*lda];
			}
		for(; j<col; j++)
			{
			pA[0+(j+0)*bs+ii*sda] = A[ii+0+(j+0)*lda];
			pA[1+(j+0)*bs+ii*sda] = A[ii+1+(j+0)*lda];
			pA[2+(j+0)*bs+ii*sda] = A[ii+2+(j+0)*lda];
			pA[3+(j+0)*bs+ii*sda] = A[ii+3+(j+0)*lda];
			}
		}
	if(ii<row1)
		{
		row2 = row1-ii;
		if(bs<row2) row2 = bs;
		for(j=0; j<col; j++)
			{
			for(i=0; i<row2; i++)
				{
				pA[i+j*bs+ii*sda] = A[i+ii+j*lda];
				}
			}
		}
	
	}



/* converts a matrix into a packed matrix */
// row and col of the source matrix, offsett in the destination matrix
void d_cvt_tran_mat2pmat(int row, int col, int offset, int bs_dummy, double *A, int lda, double *pA, int sda)
	{
	
	const int bs = 4;

	int i, ii, j, row0, row1, row2;
	
	row0 = (bs-offset%bs)%bs;
/*	if(row0>row)*/
/*		row0 = row;*/
/*	row1 = row - row0;*/
	if(row0>col)
		row0 = col;
	row1 = col - row0;
	
	ii = 0;
	if(row0>0)
		{
/*		for(j=0; j<col; j++)*/
		for(j=0; j<row; j++)
			{
			for(i=0; i<row0; i++)
				{
/*				pA[i+j*bs+ii*sda] = A[i+ii+j*lda];*/
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
	
/*		A  += row0;*/
		A  += row0*lda;
		pA += row0 + bs*(sda-1);
		}
	
	ii = 0;
	for(; ii<row1-3; ii+=bs)
		{
		j=0;
/*		for(; j<col-3; j+=4)*/
		for(; j<row-3; j+=4)
			{
			// unroll 0
			pA[0+(j+0)*bs+ii*sda] = A[j+0+(ii+0)*lda];
			pA[1+(j+0)*bs+ii*sda] = A[j+0+(ii+1)*lda];
			pA[2+(j+0)*bs+ii*sda] = A[j+0+(ii+2)*lda];
			pA[3+(j+0)*bs+ii*sda] = A[j+0+(ii+3)*lda];
			// unroll 1
			pA[0+(j+1)*bs+ii*sda] = A[j+1+(ii+0)*lda];
			pA[1+(j+1)*bs+ii*sda] = A[j+1+(ii+1)*lda];
			pA[2+(j+1)*bs+ii*sda] = A[j+1+(ii+2)*lda];
			pA[3+(j+1)*bs+ii*sda] = A[j+1+(ii+3)*lda];
			// unroll 2
			pA[0+(j+2)*bs+ii*sda] = A[j+2+(ii+0)*lda];
			pA[1+(j+2)*bs+ii*sda] = A[j+2+(ii+1)*lda];
			pA[2+(j+2)*bs+ii*sda] = A[j+2+(ii+2)*lda];
			pA[3+(j+2)*bs+ii*sda] = A[j+2+(ii+3)*lda];
			// unroll 3
			pA[0+(j+3)*bs+ii*sda] = A[j+3+(ii+0)*lda];
			pA[1+(j+3)*bs+ii*sda] = A[j+3+(ii+1)*lda];
			pA[2+(j+3)*bs+ii*sda] = A[j+3+(ii+2)*lda];
			pA[3+(j+3)*bs+ii*sda] = A[j+3+(ii+3)*lda];
			}
/*		for(; j<col; j++)*/
		for(; j<row; j++)
			{
			pA[0+(j+0)*bs+ii*sda] = A[j+0+(ii+0)*lda];
			pA[1+(j+0)*bs+ii*sda] = A[j+0+(ii+1)*lda];
			pA[2+(j+0)*bs+ii*sda] = A[j+0+(ii+2)*lda];
			pA[3+(j+0)*bs+ii*sda] = A[j+0+(ii+3)*lda];
			}
		}
	if(ii<row1)
		{
		row2 = row1-ii;
		if(bs<row2) row2 = bs;
/*		for(j=0; j<col; j++)*/
		for(j=0; j<row; j++)
			{
			for(i=0; i<row2; i++)
				{
/*				pA[i+j*bs+ii*sda] = A[i+ii+j*lda];*/
				pA[i+j*bs+ii*sda] = A[j+(i+ii)*lda];
				}
			}
		}
	
	}



/* converts a packed matrix into a matrix */
void d_cvt_pmat2mat(int row, int col, int offset, int bs_dummy, double *pA, int sda, double *A, int lda)
	{
	
	const int bs = 4;

	int i, ii, jj;
	
	int row0 = (bs-offset%bs)%bs;
	
	double *ptr_pA;
	

	jj=0;
	for(; jj<col; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(row0>0)
			{
			for(; ii<row0; ii++)
				{
				A[ii+lda*jj] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row-bs+1; ii+=bs)
			{
			i=0;
			for(; i<bs; i++)
				{
				A[i+ii+lda*jj] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row; ii++)
			{
			A[ii+lda*jj] = ptr_pA[0];
			ptr_pA++;
			}
		}

	}



/* converts a packed matrix into a matrix */
void d_cvt_tran_pmat2mat(int row, int col, int offset, int bs_dummy, double *pA, int sda, double *A, int lda)
	{
	
	const int bs = 4;

	int i, ii, jj;
	
	int row0 = (bs-offset%bs)%bs;
	
	double *ptr_pA;
	

	jj=0;
	for(; jj<col; jj++)
		{
		ptr_pA = pA + jj*bs;
		ii = 0;
		if(row0>0)
			{
			for(; ii<row0; ii++)
				{
				A[jj+lda*ii] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row-bs+1; ii+=bs)
			{
			i=0;
			for(; i<bs; i++)
				{
				A[jj+lda*(i+ii)] = ptr_pA[0];
				ptr_pA++;
				}
			ptr_pA += (sda-1)*bs;
			}
		for(; ii<row; ii++)
			{
			A[jj+lda*ii] = ptr_pA[0];
			ptr_pA++;
			}
		}

	}



/* prints a matrix */
void d_print_mat(int row, int col, double *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
//			printf("%5.2f ", *(A+i+j*lda));
//			printf("%7.3f ", *(A+i+j*lda));
			printf("%9.5f ", *(A+i+j*lda));
//			printf("%11.7f ", *(A+i+j*lda));
//			printf("%13.9f ", *(A+i+j*lda));
//			printf("%19.15f ", *(A+i+j*lda));
//			printf("%e\t", *(A+i+j*lda));
			}
		printf("\n");
		}
	printf("\n");
	}	

void d_print_mat_e(int row, int col, double *A, int lda)
	{
	int i, j;
	for(i=0; i<row; i++)
		{
		for(j=0; j<col; j++)
			{
			printf("%e\t", *(A+i+j*lda));
			}
		printf("\n");
		}
	printf("\n");
	}	



/* prints a packed matrix */
void d_print_pmat(int row, int col, int bs, double *A, int sda)
	{

	int ii, i, j, row2;

	for(ii=0; ii<row; ii+=bs)
		{
		row2 = row-ii; if(bs<row2) row2=bs;
		for(i=0; i<row2; i++)
			{
			for(j=0; j<col; j++)
				{
//				printf("%5.2f ", *(A+i+j*lda));
//				printf("%7.3f ", *(A+i+j*bs+ii*sda));
				printf("%9.5f ", *(A+i+j*bs+ii*sda));
//				printf("%11.7f ", *(A+i+j*bs+ii*sda));
//				printf("%13.9f ", *(A+i+j*bs+ii*sda));
//				printf("%19.15f ", *(A+i+j*lda));
//				printf("%e\t", *(A+i+j*lda));
				}
			printf("\n");
			}
//		printf("\n");
		}
	printf("\n");

	}	

void d_print_pmat_e(int row, int col, int bs, double *A, int sda)
	{

	int ii, i, j, row2;

	for(ii=0; ii<row; ii+=bs)
		{
		row2 = row-ii; if(bs<row2) row2=bs;
		for(i=0; i<row2; i++)
			{
			for(j=0; j<col; j++)
				{
				printf("%e ", *(A+i+j*bs+ii*sda));
				}
			printf("\n");
			}
//		printf("\n");
		}
	printf("\n");

	}	

