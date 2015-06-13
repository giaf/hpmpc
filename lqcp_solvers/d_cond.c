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

#include <math.h>

#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/lqcp_aux.h"
#include "../include/block_size.h"



void d_cond_Q(int N, int nx, int nu, double **pA, int diag_Q, double **pQ, double **pL, int compute_Gamma_0, double **pGamma_0, double **pGamma_0_Q, double *pH_Q, double *work)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj;

	if(compute_Gamma_0)
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_0[0], cNnx);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii-1], cNnx, pA[ii], cnx, pGamma_0[ii], cNnx, pGamma_0[ii], cNnx, 0, 0, 0);
			}
		}
	
	if(diag_Q)
		{

		for(jj=0; jj<nx; jj++) pL[1][jj] = sqrt(pQ[1][jj]);
		dgemm_diag_right_lib(nx, nx, pGamma_0[0], cNnx, pL[1], pGamma_0_Q[0], cNnx, pGamma_0_Q[0], cNnx, 0);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cNnx, pGamma_0_Q[0], cNnx, pH_Q, cnx, pH_Q, cnx, 0);
		for(ii=1; ii<N; ii++)
			{
			for(jj=0; jj<nx; jj++) pL[ii+1][jj] = sqrt(pQ[ii+1][jj]);
			dgemm_diag_right_lib(nx, nx, pGamma_0[ii], cNnx, pL[ii+1], pGamma_0_Q[ii], cNnx, pGamma_0_Q[ii], cNnx, 0);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cNnx, pGamma_0_Q[ii], cNnx, pH_Q, cnx, pH_Q, cnx, 1);
			}
		d_add_diag_pmat(nx, pH_Q, cnx, pQ[0]);

		}
	else
		{
#if 1

		dpotrf_lib(nx, nx, pQ[1], cnx, pL[1], cnx, work);
		dtrtr_l_lib(nx, 0, pL[1], cnx, pL[1], cnx);
		dtrmm_l_lib(nx, nx, pGamma_0[0], cNnx, pL[1], cnx, pGamma_0_Q[0], cNnx);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cNnx, pGamma_0_Q[0], cNnx, pQ[0], cnx, pH_Q, cnx, 1);
		for(ii=1; ii<N; ii++)
			{
			dpotrf_lib(nx, nx, pQ[ii+1], cnx, pL[ii+1], cnx, work);
			dtrtr_l_lib(nx, 0, pL[ii+1], cnx, pL[ii+1], cnx);
			dtrmm_l_lib(nx, nx, pGamma_0[ii], cNnx, pL[ii+1], cnx, pGamma_0_Q[ii], cNnx);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cNnx, pGamma_0_Q[ii], cNnx, pH_Q, cnx, pH_Q, cnx, 1);
			}
#else
	
		// Gamma_0 * bar_Q * Gamma_0'
		dgemm_nt_lib(nx, nx, nx, pGamma_0[0], cNnx, pQ[1], cnx, pGamma_0_Q[0], cNnx, pGamma_0_Q[0], cNnx, 0, 0, 0);
		//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[0], cNnx, pGamma_0[0], cNnx, pQ[0], cnx, pH_Q, cnx, 1, 0, 0);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cNnx, pGamma_0[0], cNnx, pQ[0], cnx, pH_Q, cnx, 1);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii], cNnx, pQ[ii+1], cnx, pGamma_0_Q[ii], cNnx, pGamma_0_Q[ii], cNnx, 0, 0, 0);
			//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cNnx, pGamma_0[ii], cNnx, pH_Q, cnx, pH_Q, cnx, 1, 0, 0);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cNnx, pGamma_0[ii], cNnx, pH_Q, cnx, pH_Q, cnx, 1);
			}
#endif
		}

	}



void d_cond_R(int N, int nx, int nu, int N2_cond, double **pA, double **pAt, double **pBt, int diag_Q, double **pQ, int nzero_S, double **pS, double **pR, int compute_Gamma_u, double **pGamma_u, double **pGamma_u_Q, double **pGamma_u_Q_A, double *pH_R)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, offset, i_temp;

	// Gamma_u
	if(compute_Gamma_u)
		{
		dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cNnx);
		for(ii=1; ii<N; ii++)
			{
			offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cNnx, pGamma_u[ii], cNnx, pGamma_u[ii], cNnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cNnx, pA[ii], cnx, pGamma_u[ii], cNnx, pGamma_u[ii], cNnx, 0, 0, 0); // Gamma_u * A^T
#endif
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cNnx+offset%bs, cNnx);
			}
		}
		
	// Gamma_u * Q
	if(diag_Q)
		{
		for(ii=0; ii<N; ii++)
			{
			dgemm_diag_right_lib((ii+1)*nu, nx, pGamma_u[ii], cNnx, pQ[ii+1], pGamma_u_Q[ii], cNnx, pGamma_u_Q[ii], cNnx, 0);
			}
		}
	else
		{
		for(ii=0; ii<N; ii++)
			{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cNnx, pGamma_u_Q[ii], cNnx, pGamma_u_Q[ii], cNnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cNnx, pQ[ii+1], cnx, pGamma_u_Q[ii], cNnx, pGamma_u_Q[ii], cNnx, 0, 0, 0); // Gamma_u * A^T
#endif
			}
		}
	
	if(N2_cond)
		{

		// Gamma_u_Q * bar_A
		dgecp_lib(N*nu, nx, 0, pGamma_u_Q[N-1], cNnx, 0, pGamma_u_Q_A[N-1], cNnx);
		for(ii=N-1; ii>0; ii--)
			{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pAt[ii], cnx, pGamma_u_Q_A[ii], cNnx, pGamma_u_Q[ii-1], cNnx, pGamma_u_Q_A[ii-1], cNnx, 1, 1, 1);
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u_Q_A[ii], cNnx, pAt[ii], cnx, pGamma_u_Q[ii-1], cNnx, pGamma_u_Q_A[ii-1], cNnx, 1, 0, 0);
#endif
			}

		if(nzero_S)
			{
			// Gamma_u * bar_S
			for(ii=1; ii<N; ii++)
				{
				dgemm_nt_lib(ii*nu, nu, nx, pGamma_u[ii-1], cNnx, pS[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0, 0);
				}
			}
		else
			{
			for(ii=0; ii<N*nu; ii+=4)
				{
				for(jj=ii; jj<N*nu; jj++)
					{
					pH_R[ii*cNnu+jj*bs+0] = 0.0;
					pH_R[ii*cNnu+jj*bs+1] = 0.0;
					pH_R[ii*cNnu+jj*bs+2] = 0.0;
					pH_R[ii*cNnu+jj*bs+3] = 0.0;
					}
				}
			}
		
		// R
		for(ii=0; ii<N; ii++)
			{
			dgecp_lib(nu, nu, 0, pR[ii], cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
			}

		// Gamma_u_Q_A * B
		for(ii=0; ii<N; ii++)
			{
			dgemm_nt_lib((ii+1)*nu, nu, nx, pGamma_u_Q_A[ii], cNnx, pBt[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 1, 0, 0);
			}

		// transpose H in the lower triangular
		dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

		}
	else // N3 cond
		{
		
		if(nzero_S)
			{
			// Gamma_u * bar_S
			for(ii=1; ii<N; ii++)
				{
				dgemm_nt_lib(ii*nu, nu, nx, pGamma_u[ii-1], cNnx, pS[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0, 0);
				}

			// transpose H in the lower triangular
			dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);
			}
		else
			{
			for(ii=0; ii<N*nu; ii+=4)
				{
				i_temp = ii+4<cNnu ? ii+4 : cNnu;
				for(jj=0; jj<i_temp; jj++)
					{
					pH_R[ii*cNnu+jj*bs+0] = 0.0;
					pH_R[ii*cNnu+jj*bs+1] = 0.0;
					pH_R[ii*cNnu+jj*bs+2] = 0.0;
					pH_R[ii*cNnu+jj*bs+3] = 0.0;
					}
				}
			}
			
		// R
		for(ii=0; ii<N; ii++)
			{
			dgecp_lib(nu, nu, 0, pR[ii], cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
			}

		for(ii=0; ii<N; ii++)
			dsyrk_nt_lib((N-ii)*nu, (N-ii)*nu, nx, pGamma_u_Q[N-1-ii], cNnx, pGamma_u[N-1-ii], cNnx, pH_R, cNnu, pH_R, cNnu, 1); // TODO make exact dsyrk !!!!!!!!!!!!!!!!!!!!!!

		}
	
	}



void d_cond_S(int N, int nx, int nu, int nzero_S, double **pS, double **pGamma_0, double **pGamma_u_Q, double *pH_S)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	if(nzero_S)
		{
		// Gamma_0 * bar_S
		dgetr_lib(nu, nx, 0, pS[0], cnx, 0, pH_S, cNnu);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nu, nx, pGamma_0[ii-1], cNnx, pS[ii], cnx, pH_S+ii*nu*bs, cNnu, pH_S+ii*nu*bs, cNnu, 0, 0, 0);
			}

		for(ii=0; ii<N; ii++)
			{
			dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0[ii], cNnx, pGamma_u_Q[ii], cNnx, pH_S, cNnu, pH_S, cNnu, 1, 0, 0);
			}
		}
	else
		{
		dgemm_nt_lib(nx, N*nu, nx, pGamma_0[N-1], cNnx, pGamma_u_Q[N-1], cNnx, pH_S, cNnu, pH_S, cNnu, 0, 0, 0);
		for(ii=0; ii<N-1; ii++)
			{
			dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0[ii], cNnx, pGamma_u_Q[ii], cNnx, pH_S, cNnu, pH_S, cNnu, 1, 0, 0);
			}
		}
	
	}



void d_cond_q(int N, int nx, int nu, double **pA, double **b, int diag_Q, double **pQ, double **q, double **pGamma_0, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], b[ii], Gamma_b[ii], 1);
			}
		}
	
	// Gamma_b * Q + q
	if(compute_Gamma_b_q)
		{
		if(diag_Q)
			{
			for(ii=0; ii<N; ii++)
				{
				dgemv_diag_lib(nx, pQ[ii+1], Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				}
			}
		else
			{
			for(ii=0; ii<N; ii++)
				{
				//dgemv_n_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				dsymv_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				}
			}
		}
		
	// Gamma_0' * Gamma_b_q
	d_copy_mat(nx, 1, q[0], 1, H_q, 1);
	for(ii=0; ii<N; ii++)
		{
		dgemv_t_lib(nx, nx, pGamma_0[ii], cNnx, Gamma_b_q[ii], H_q, H_q, 1);
		}
	
	}



void d_cond_r(int N, int nx, int nu, double **pA, double **b, int diag_Q, double **pQ, int nzero_S, double **pS, double **q, double **r, double **pGamma_u, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_r)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], b[ii], Gamma_b[ii], 1);
			}
		}

	// barS * Gamma_b
	if(nzero_S)
		{
		for(ii=0; ii<N; ii++)
			{
			dgemv_n_lib(nu, nx, pS[ii], cnx, Gamma_b[ii], r[ii], H_r+ii*nu, 1);
			}
		}
	else
		{
		for(ii=0; ii<N; ii++)
			{
			d_copy_mat(nu, 1, r[ii], 1, H_r+ii*nu, 1);
			}
		}
	
	// Gamma_b * Q + q
	if(compute_Gamma_b_q)
		{
		if(diag_Q)
			{
			for(ii=0; ii<N; ii++)
				{
				dgemv_diag_lib(nx, pQ[ii+1], Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				}
			}
		else
			{
			for(ii=0; ii<N; ii++)
				{
				//dgemv_n_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				dsymv_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				}
			}
		}

	// Gamma_u * Gamma_b_q
	for(ii=0; ii<N; ii++)
		{
		dgemv_n_lib((ii+1)*nu, nx, pGamma_u[ii], cNnx, Gamma_b_q[ii], H_r, H_r, 1);
		}
		
	}


void d_cond_A(int N, int nx, int nu, double **pA, int compute_Gamma_0, double **pGamma_0, double *pH_A)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj;

	if(compute_Gamma_0)
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_0[0], cNnx);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii-1], cNnx, pA[ii], cnx, pGamma_0[ii], cNnx, pGamma_0[ii], cNnx, 0, 0, 0);
			}
		}
	
	dgetr_lib(nx, nx, 0, pGamma_0[N-1], cNnx, 0, pH_A, cnx);

	}



void d_cond_B(int N, int nx, int nu, double **pA, double **pBt, int compute_Gamma_u, double **pGamma_u, double *pH_B)
	{
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, offset, i_temp;

	// Gamma_u
	if(compute_Gamma_u)
		{
		dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cNnx);
		for(ii=1; ii<N; ii++)
			{
			offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cNnx, pGamma_u[ii], cNnx, pGamma_u[ii], cNnx, 0, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cNnx, pA[ii], cnx, pGamma_u[ii], cNnx, pGamma_u[ii], cNnx, 0, 0, 0); // Gamma_u * A^T
#endif
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cNnx+offset%bs, cNnx);
			}
		}
	
	dgetr_lib(N*nu, nx, 0, pGamma_u[N-1], cNnx, 0, pH_B, cNnu);

	}



void d_cond_b(int N, int nx, int nu, double **pA, double **b, int compute_Gamma_b, double **Gamma_b, double *H_b)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], b[ii], Gamma_b[ii], 1);
			}
		}
	
	d_copy_mat(nx, 1, Gamma_b[N-1], 1, H_b, 1);
	
	}
	
