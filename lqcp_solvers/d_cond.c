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

#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/block_size.h"



void d_compute_Gamma_u(int N, int nx, int nu, double **pA, double **pBt, double **pGamma_u)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;

	int ii, offset;

	// first stage
	d_copy_pmat_general(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);

	// general stage
	for(ii=1; ii<N; ii++)
		{
		offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0, 0); // Gamma_u * A^T
#endif
		d_copy_pmat_general(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
		}
	
	}



void d_compute_Gamma_u_Q(int N, int nx, int nu, double **pGamma_u, double **pQ, double **pGamma_u_Q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;

	int ii;

	// general stage
	for(ii=0; ii<N; ii++)
		{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cnx, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0, 0); // Gamma_u * A^T
#endif
		}
	
	}
