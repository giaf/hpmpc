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



void d_cond_R(int N, int nx, int nu, double **pA, double **pAt, double **pBt, double **pQ, double **pS, double **pR, int compute_Gamma_u, double **pGamma_u, double **pGamma_u_Q, double *pH_R)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = ((N-1)*nu+cnu+ncl-1)/ncl*ncl;

	int ii, offset;

	// Gamma_u
	if(compute_Gamma_u)
		{
		d_copy_pmat_general(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
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
		
	// Gamma_u * Q
	for(ii=0; ii<N; ii++)
		{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cnx, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0, 0); // Gamma_u * A^T
#endif
		}
	
	for(ii=N-1; ii>0; ii--)
		{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx, ii*nu, nx, pAt[ii], cnx, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii-1], cnx, pGamma_u_Q[ii-1], cnx, 1, 1, 1);
#else
		dgemm_nt_lib(ii*nu, nx, nx, pGamma_u_Q[ii], cnx, pAt[ii], cnx, pGamma_u_Q[ii-1], cnx, pGamma_u_Q[ii-1], cnx, 1, 0, 0);
#endif
		}
	
	// Gamma_u * bar_S
	for(ii=1; ii<N; ii++)
		{
		dgemm_nt_lib(ii*nu, nu, nx, pGamma_u[ii-1], cnx, pS[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0, 0);
		}
	
	// R
	for(ii=0; ii<N; ii++)
		{
		d_copy_pmat_general(nu, nu, 0, pR[ii], cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
		}

	for(ii=0; ii<N; ii++)
		{
		dgemm_nt_lib((ii+1)*nu, nu, nx, pGamma_u_Q[ii], cnx, pBt[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 1, 0, 0);
		}
	
	// transpose H in the lower triangular
	dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

	}



void d_cond_Q(int N, int nx, int nu, double **pA, double **pQ, double **pL, int compute_Gamma_0, double **pGamma_0, double **pGamma_0_Q, double *pH_Q, double *work)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;

	int ii;

	if(compute_Gamma_0)
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_0[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii-1], cnx, pA[ii], cnx, pGamma_0[ii], cnx, pGamma_0[ii], cnx, 0, 0, 0);
			}
		}
	
#if 0

	dpotrf_lib(nx, nx, pQ[1], cnx, pL[1], cnx, work);
	dtrtr_l_lib(nx, 0, pL[1], cnx, pL[1], cnx);
	dtrmm_l_lib(nx, nx, pGamma_0[0], cnx, pL[1], cnx, pGamma_0_Q[0], cnx);
	dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, pQ[0], cnx, pH_Q, cnx, 1);
	for(ii=1; ii<N; ii++)
		{

		dpotrf_lib(nx, nx, pQ[ii+1], cnx, pL[ii+1], cnx, work);
		dtrtr_l_lib(nx, 0, pL[ii+1], cnx, pL[ii+1], cnx);
		dtrmm_l_lib(nx, nx, pGamma_0[ii], cnx, pL[ii+1], cnx, pGamma_0_Q[ii], cnx);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1);

		}
#else
	
	// Gamma_0 * bar_Q * Gamma_0'
	dgemm_nt_lib(nx, nx, nx, pGamma_0[0], cnx, pQ[1], cnx, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, 0, 0, 0);
	//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0[0], cnx, pQ[0], cnx, pH_Q, cnx, 1, 0, 0);
	dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0[0], cnx, pQ[0], cnx, pH_Q, cnx, 1);
	for(ii=1; ii<N; ii++)
		{
		dgemm_nt_lib(nx, nx, nx, pGamma_0[ii], cnx, pQ[ii+1], cnx, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, 0, 0, 0);
		//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1, 0, 0);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1);
		}
#endif

	}



