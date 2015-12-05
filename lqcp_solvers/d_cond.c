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



void d_cond_Gamma_u_T(int N, int nx, int nu, int free_x0, double **pA, double **pBt, double **pGamma_u)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;

	int ii, offset;

	if(free_x0) // MHE
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_u[0], cnx);
		offset = nx;
		dgecp_lib(nu, nx, 0, pBt[0], cnx, offset, pGamma_u[0]+offset/bs*bs*cnx+offset%bs, cnx);
		for(ii=1; ii<N; ii++)
			{
			offset = nx + ii*nu;
			dgemm_nt_lib(nx+ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
			}
		}
	else // MPC
		{
		dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
			}
		}

	}



void d_cond_R_N3_nx2(int N, int nx, int nu, int free_x0, double **pAt, double **pBt, int diag_hessian, int nzero_Q_N, double **pQ, double **pS, double **pR, double *pL, double *dL, double **pGamma_u, double **pGamma_u_Q, double *pH_R)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;

	int ii, jj, kk, offset, i_temp;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	
	if(free_x0) // MHE
		{

		int cnxNnu = (nx+N*nu+ncl-1)/ncl*ncl;

		if(diag_hessian)
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
				dgemm_diag_right_lib(nx+(ii+1)*nu, nx, pGamma_u[ii], cnx, pQ[ii+1], 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx);
				}
			if(nzero_Q_N==0)
				dgeset_lib(nx+N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			dgeset_lib(nx+N*nu, nx+N*nu, 0.0, 0, pH_R, cnxNnu);

			// dR
			ddiain_lib(nx, pQ[0], 0, pH_R, cnxNnu);
			for(ii=0; ii<N; ii++)
				{
				offset = nx+ii*nu;
				ddiain_lib(nu, pR[ii], offset, pH_R+offset/bs*bs*cnxNnu+offset%bs+offset*bs, cnxNnu);
				}

			for(ii=0; ii<N1; ii++)
				dsyrk_nt_lib(nx+(N1-ii)*nu, nx+(N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cnxNnu, pH_R, cnxNnu); 

			}
		else
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
#if 1
				dpotrf_lib(nx, nx, pQ[ii+1], cnx, pL, cnx, dL);
				dtrmm_nt_u_lib(nx+(ii+1)*nu, nx, pGamma_u[ii], cnx, pL, cnx, pGamma_u_Q[ii], cnx);
#else
				dgemm_nt_lib(nx+(ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
				}
			if(nzero_Q_N==0)
				dgeset_lib(nx+N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			// Gamma_u * bar_S
			dgetr_lib(nu, nx, 0, pS[0], cnx, 0, pH_R, cnxNnu);
			for(ii=1; ii<N; ii++)
				{
				dgemm_nt_lib(nx+ii*nu, nu, nx, pGamma_u[ii-1], cnx, pS[ii], cnx, 0, pH_R+ii*nu*bs, cnxNnu, pH_R+ii*nu*bs, cnxNnu, 0, 0);
				}

			// transpose H in the lower triangular
			dtrtr_u_lib(nx+N*nu, pH_R, cnxNnu, pH_R, cnxNnu);

			// R
			dgecp_lib(nx, nx, 0, pQ[0], cnx, 0, pH_R, cnxNnu);
			for(ii=0; ii<N; ii++)
				{
				offset = nx + ii*nu;
				dgecp_lib(nu, nu, 0, pR[ii], cnu, offset, pH_R+offset/bs*bs*cnxNnu+offset%bs+offset*bs, cnxNnu);
				}

#if 1
				for(ii=0; ii<N1; ii++)
					dsyrk_nt_lib(nx+(N1-ii)*nu, nx+(N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u_Q[N1-1-ii], cnx, 1, pH_R, cnxNnu, pH_R, cnxNnu); 
#else
				for(ii=0; ii<N1; ii++)
					dsyrk_nt_lib(nx+(N1-ii)*nu, nx+(N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cnxNnu, pH_R, cnxNnu); 
#endif

			}


		}
	else // MPC
		{

		int cNnu = (N*nu+ncl-1)/ncl*ncl;

		if(diag_hessian)
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
				dgemm_diag_right_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, pQ[ii+1], 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx);
				}

			if(nzero_Q_N==0)
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			dgeset_lib(N*nu, N*nu, 0.0, 0, pH_R, cNnu);

			// dR
			for(ii=0; ii<N; ii++)
				{
				ddiain_lib(nu, pR[ii], ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

			for(ii=0; ii<N1; ii++)
				dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 

			}
		else
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
	//			if(use_L)
	//				{
	//				dtrmm_nt_u_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, pL[ii+1], cnx, pGamma_u_Q[ii], cnx);
	//				}
	//			else
	//				{
	#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
					dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
	#else
					dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0); // Gamma_u * A^T
	#endif
	//				}
				}

			if(nzero_Q_N==0)
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			// Gamma_u * bar_S
			for(ii=1; ii<N; ii++)
				{
				dgemm_nt_lib(ii*nu, nu, nx, pGamma_u[ii-1], cnx, pS[ii], cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);
				}

			// transpose H in the lower triangular
			dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

			// R
			for(ii=0; ii<N; ii++)
				{
				dgecp_lib(nu, nu, 0, pR[ii], cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

	//		if(use_L)
	//			{
	//			for(ii=0; ii<N1; ii++)
	//				dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u_Q[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 
	//			}
	//		else
	//			{
				for(ii=0; ii<N1; ii++)
					dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 
	//			}

			}

		}

	return;

	}













void d_cond_Q(int N, int nx, int nu, double **pA, int diag_Q, int nzero_Q_N, double **pQ, double **pL, int compute_Gamma_0, double **pGamma_0, double **pGamma_0_Q, double *pH_Q, double *work)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	
	if(compute_Gamma_0)
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_0[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii-1], cnx, pA[ii], cnx, 0, pGamma_0[ii], cnx, pGamma_0[ii], cnx, 0, 0);
			}
		}
	
	if(diag_Q)
		{

#if 0
		for(jj=0; jj<nx; jj++) pL[1][jj] = sqrt(pQ[1][jj]);
		dgemm_diag_right_lib(nx, nx, pGamma_0[0], cnx, pL[1], pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, 0);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, pH_Q, cnx, pH_Q, cnx, 0);
		for(ii=1; ii<N1; ii++)
			{
			for(jj=0; jj<nx; jj++) pL[ii+1][jj] = sqrt(pQ[ii+1][jj]);
			dgemm_diag_right_lib(nx, nx, pGamma_0[ii], cnx, pL[ii+1], pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, 0);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1);
			}
		d_add_diag_pmat(nx, pH_Q, cnx, pQ[0]);
#else
		if(N1>0)
			{
			dgemm_diag_right_lib(nx, nx, pGamma_0[0], cnx, pQ[1], 0, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0[0], cnx, 0, pH_Q, cnx, pH_Q, cnx);
			}
		else
			{
			dgeset_lib(nx, nx, 0.0, 0, pH_Q, cnx);
			}
		for(ii=1; ii<N1; ii++)
			{
			dgemm_diag_right_lib(nx, nx, pGamma_0[ii], cnx, pQ[ii+1], 0, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0[ii], cnx, 1, pH_Q, cnx, pH_Q, cnx);
			}
		//d_add_diag_pmat(nx, pH_Q, cnx, pQ[0]);
		ddiaad_lib(nx, 1.0, pQ[0], 0, pH_Q, cnx);
#endif

		}
	else
		{
#if 1

		dpotrf_lib_old(nx, nx, pQ[1], cnx, pL[1], cnx, work);
		dtrtr_l_lib(nx, 0, pL[1], cnx, pL[1], cnx);
		dtrmm_nt_u_lib(nx, nx, pGamma_0[0], cnx, pL[1], cnx, pGamma_0_Q[0], cnx);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, 1, pQ[0], cnx, pH_Q, cnx);
		for(ii=1; ii<N1; ii++)
			{
			dpotrf_lib_old(nx, nx, pQ[ii+1], cnx, pL[ii+1], cnx, work);
			dtrtr_l_lib(nx, 0, pL[ii+1], cnx, pL[ii+1], cnx);
			dtrmm_nt_u_lib(nx, nx, pGamma_0[ii], cnx, pL[ii+1], cnx, pGamma_0_Q[ii], cnx);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, 1, pH_Q, cnx, pH_Q, cnx);
			}
#else
	
		// Gamma_0 * bar_Q * Gamma_0'
		dgemm_nt_lib(nx, nx, nx, pGamma_0[0], cnx, pQ[1], cnx, pGamma_0_Q[0], cnx, pGamma_0_Q[0], cnx, 0, 0, 0);
		//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0[0], cnx, pQ[0], cnx, pH_Q, cnx, 1, 0, 0);
		dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[0], cnx, pGamma_0[0], cnx, pQ[0], cnx, pH_Q, cnx, 1);
		for(ii=1; ii<N1; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii], cnx, pQ[ii+1], cnx, pGamma_0_Q[ii], cnx, pGamma_0_Q[ii], cnx, 0, 0, 0);
			//dgemm_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1, 0, 0);
			dsyrk_nt_lib(nx, nx, nx, pGamma_0_Q[ii], cnx, pGamma_0[ii], cnx, pH_Q, cnx, pH_Q, cnx, 1);
			}
#endif
		}

	}



// condensing algorithms:
// alg==0 : N^3 n_x^2 algorithm
// alg==1 : N^2 n_x^2 algorithm
// alg==2 : N^2 n_x^3 algorithm
void d_cond_R(int N, int nx, int nu, int alg, double **pA, double **pAt, double **pBt, double **pBAt, int diag_hessian, int nzero_Q_N, double **pQ, int use_L, double **pL, double **pS, double **pR, double **pRSQ, double *pD, double *pM, double *pP, double *pLam, double *diag, double *pBAtL, int compute_Gamma_u, double **pGamma_u, double **pGamma_u_Q, double **pGamma_u_Q_A, double *pH_R)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, kk, offset, i_temp;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	

	// Gamma_u^T
	if(compute_Gamma_u)
		{
		dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
			}
		}


	if(alg==0) // N^3 n_x^2
		{
		
		if(diag_hessian)
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
				dgemm_diag_right_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, pQ[ii+1], 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx);
				}

			if(nzero_Q_N==0)
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			dgeset_lib(N*nu, N*nu, 0.0, 0, pH_R, cNnu);

			// dR
			for(ii=0; ii<N; ii++)
				{
				ddiain_lib(nu, pR[ii], ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

			for(ii=0; ii<N1; ii++)
				dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 

			}
		else
			{

			// Gamma_u * Q
			for(ii=0; ii<N1; ii++)
				{
				if(use_L)
					{
					dtrmm_nt_u_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, pL[ii+1], cnx, pGamma_u_Q[ii], cnx);
					}
				else
					{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
					dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
					dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
					}
				}

			if(nzero_Q_N==0)
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);

			// Gamma_u * bar_S
			for(ii=1; ii<N; ii++)
				{
				dgemm_nt_lib(ii*nu, nu, nx, pGamma_u[ii-1], cnx, pS[ii], cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);
				}

			// transpose H in the lower triangular
			dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

			// R
			for(ii=0; ii<N; ii++)
				{
				dgecp_lib(nu, nu, 0, pR[ii], cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

			if(use_L)
				{
				for(ii=0; ii<N1; ii++)
					dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u_Q[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 
				}
			else
				{
				for(ii=0; ii<N1; ii++)
					dsyrk_nt_lib((N1-ii)*nu, (N1-ii)*nu, nx, pGamma_u_Q[N1-1-ii], cnx, pGamma_u[N1-1-ii], cnx, 1, pH_R, cNnu, pH_R, cNnu); 
				}

			}

		return;

		}
	if(alg==1) // N^2 n_x^2
		{

		// Gamma_u * Q
		if(diag_hessian)
			{
			for(ii=0; ii<N1; ii++)
				{
				dgemm_diag_right_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, pQ[ii+1], 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx);
				}

			if(nzero_Q_N==0)
				{
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q_A[N-1], cnx);
				}
			
			// zero S
			for(ii=1; ii<N; ii++)
				{
				//dgecp_lib(nu, nx, 0, pS[ii], cnx, (ii)*nu, pGamma_u_Q[ii-1]+(ii)*nu/bs*bs*cnx+(ii)*nu%bs, cnx);
				dgeset_lib(nu, nx, 0.0, (ii)*nu, pGamma_u_Q[ii-1]+(ii)*nu/bs*bs*cnx+(ii)*nu%bs, cnx);
				}

			// Gamma_u_Q * bar_A
			dgecp_lib(N1*nu, nx, 0, pGamma_u_Q[N1-1], cnx, 0, pGamma_u_Q_A[N1-1], cnx);
			for(ii=N1-1; ii>0; ii--)
				{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				//dgemm_nt_lib(nx, ii*nu, nx, pAt[ii], cnx, pGamma_u_Q_A[ii], cnx, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 1, 1, 1);
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pAt[ii], cnx, pGamma_u_Q_A[ii], cnx, 1, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 1, 1);
#else
				//dgemm_nt_lib(ii*nu, nx, nx, pGamma_u_Q_A[ii], cnx, pAt[ii], cnx, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 1, 0, 0);
				dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u_Q_A[ii], cnx, pAt[ii], cnx, 1, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 0, 0);
#endif
				}

#if 0
			dgeset_lib(N*nu, N*nu, 0.0, 0, pH_R, cNnu);
			
			// R
			for(ii=0; ii<N; ii++)
				{
				ddiain_lib(nu, pR[ii], ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

			// Gamma_u_Q_A * B
			for(ii=0; ii<N1; ii++)
				{
				dgemm_nt_lib((ii+1)*nu, nu, nx, pGamma_u_Q_A[ii], cnx, pBt[ii], cnx, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 1, 0, 0);
				}

#else

			// Gamma_u * M
			for(ii=1; ii<N1; ii++)
				{
				dgecp_lib(nu, nx, ii*nu, pGamma_u_Q_A[ii-1]+(ii*nu)/bs*bs*cnx+ii*nu%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pM, cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);
				}

			// R
			for(ii=0; ii<N; ii++)
				{
				dgecp_lib(nu, nx, ii*nu, pGamma_u_Q_A[ii]+(ii*nu)/bs*bs*cnx+(ii*nu)%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[ii], cnx, 0, pD, cnu, pD, cnu, 0, 0);
				ddiaad_lib(nu, 1.0, pR[ii], 0, pD, cnu);
				dgecp_lib(nu, nu, 0, pD, cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

#endif

			}
		else
			{
			for(ii=0; ii<N1; ii++)
				{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pQ[ii+1], cnx, pGamma_u[ii], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
				dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u[ii], cnx, pQ[ii+1], cnx, 0, pGamma_u_Q[ii], cnx, pGamma_u_Q[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
				}

			if(nzero_Q_N==0)
				{
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q[N-1], cnx);
				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q_A[N-1], cnx);
				}

			// copy S
			for(ii=1; ii<N; ii++)
				{
				dgecp_lib(nu, nx, 0, pS[ii], cnx, (ii)*nu, pGamma_u_Q[ii-1]+(ii)*nu/bs*bs*cnx+(ii)*nu%bs, cnx);
				}

//			if(nzero_Q_N==0)
//				dgeset_lib(N*nu, nx, 0.0, 0, pGamma_u_Q_A[N-1], cnx);
			
			// Gamma_u_Q * bar_A^{-1}
			dgecp_lib(N1*nu, nx, 0, pGamma_u_Q[N1-1], cnx, 0, pGamma_u_Q_A[N1-1], cnx);
			for(ii=N1-1; ii>0; ii--)
				{
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pAt[ii], cnx, pGamma_u_Q_A[ii], cnx, 1, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 1, 1);
#else
				dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_u_Q_A[ii], cnx, pAt[ii], cnx, 1, pGamma_u_Q[ii-1], cnx, pGamma_u_Q_A[ii-1], cnx, 0, 0);
#endif
				}

			// Gamma_u * M
			for(ii=1; ii<N1; ii++)
				{
				dgecp_lib(nu, nx, ii*nu, pGamma_u_Q_A[ii-1]+(ii*nu)/bs*bs*cnx+ii*nu%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pM, cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);
				}

			// R
			for(ii=0; ii<N; ii++)
				{
				dgecp_lib(nu, nx, ii*nu, pGamma_u_Q_A[ii]+(ii*nu)/bs*bs*cnx+(ii*nu)%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[ii], cnx, 1, pR[ii], cnu, pD, cnu, 0, 0);
				dgecp_lib(nu, nu, 0, pD, cnu, ii*nu, pH_R+(ii*nu)/bs*bs*cNnu+(ii*nu)%bs+ii*nu*bs, cNnu);
				}

			}

		// transpose H in the lower triangular
		dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

		return;

		}
	if(alg==2) // N^2 n_x^3 algorithm
		{

		// TODO nzero_Q_N !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		int nz = nx+nu;
		int cnz = (nz+ncl-1)/ncl*ncl;

		if(diag_hessian)
			{

			// final stage 
			dgeset_lib(nx, nx, 0.0, 0, pP, cnx);
			for(jj=0; jj<nx-3; jj+=4)
				{
				pP[jj*cnx+0+(jj+0)*bs] = sqrt(pRSQ[N][nu+jj+0]);
				pP[jj*cnx+1+(jj+1)*bs] = sqrt(pRSQ[N][nu+jj+1]);
				pP[jj*cnx+2+(jj+2)*bs] = sqrt(pRSQ[N][nu+jj+2]);
				pP[jj*cnx+3+(jj+3)*bs] = sqrt(pRSQ[N][nu+jj+3]);
				}
			for(kk=0; kk<nx-jj; kk++)
				{
				pP[jj*cnx+kk+(jj+kk)*bs] = sqrt(pRSQ[N][nu+jj+kk]);
				}

			// middle stages 
			for(ii=N-1; ii>0; ii--)
				{	
				dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pP, cnx, pBAtL, cnx);
				dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
				ddiaad_lib(nz, 1.0, pRSQ[ii], 0, pLam, cnz);
				dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

				dgecp_lib(nu, nu, 0, pLam, cnz, (ii)*nu, pH_R+((ii)*nu)/bs*bs*cNnu+((ii)*nu)%bs+(ii)*nu*bs, cNnu);
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemm_nt_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pM, cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);

				dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pP, cnx);
				dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
				dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

				}

			// first stage 
			dtrmm_nt_u_lib(nu, nx, pBAt[0], cnx, pP, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nu, nu, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nu, 1.0, pRSQ[0], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, 0, pH_R, cNnu);

			}
		else
			{

			// final stage 
			dgecp_lib(nx, nx, nu, pRSQ[N]+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pP, cnx);
			dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
			dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

			// middle stages 
			for(ii=N-1; ii>0; ii--)
				{	
				dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pP, cnx, pBAtL, cnx);
				dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[ii], cnz, pLam, cnz);
				dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

				dgecp_lib(nu, nu, 0, pLam, cnz, (ii)*nu, pH_R+((ii)*nu)/bs*bs*cNnu+((ii)*nu)%bs+(ii)*nu*bs, cNnu);
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemm_nt_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pM, cnx, 0, pH_R+ii*nu*bs, cNnu, pH_R+ii*nu*bs, cNnu, 0, 0);

				dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pP, cnx);
				dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
				dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

				}

			// first stage 
			dtrmm_nt_u_lib(nu, nx, pBAt[0], cnx, pP, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nu, nu, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[0], cnz, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, 0, pH_R, cNnu);

			}

		// transpose H in the lower triangular
		dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

		return;

		}


	
	}



void d_cond_St(int N, int nx, int nu, int nzero_S, double **pS, int nzero_Q_N, double **pGamma_0, int use_pGamma_0_Q, double **pGamma_0_Q, double **pGamma_u_Q, double *pH_St)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	
	if(nzero_S)
		{
		// Gamma_0 * bar_S
		dgetr_lib(nu, nx, 0, pS[0], cnx, 0, pH_St, cNnu);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nu, nx, pGamma_0[ii-1], cnx, pS[ii], cnx, 0, pH_St+ii*nu*bs, cNnu, pH_St+ii*nu*bs, cNnu, 0, 0);
			}

		if(use_pGamma_0_Q)
			{
			for(ii=0; ii<N1; ii++)
				{
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0_Q[ii], cnx, pGamma_u_Q[ii], 1, cnx, pH_St, cNnu, pH_St, cNnu, 0, 0);
				}
			}
		else
			{
			for(ii=0; ii<N1; ii++)
				{
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0[ii], cnx, pGamma_u_Q[ii], cnx, 1, pH_St, cNnu, pH_St, cNnu, 0, 0);
				}
			}
		}
	else
		{
		if(use_pGamma_0_Q)
			{
			dgemm_nt_lib(nx, N1*nu, nx, pGamma_0_Q[N1-1], cnx, pGamma_u_Q[N1-1], cnx, 0, pH_St, cNnu, pH_St, cNnu, 0, 0);
			if(nzero_Q_N==0)
				dgeset_lib(nx, nu, 0.0, 0, pH_St+N1*nu*bs, cNnu);
			for(ii=0; ii<N1-1; ii++)
				{
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0_Q[ii], cnx, pGamma_u_Q[ii], cnx, 1, pH_St, cNnu, pH_St, cNnu, 0, 0);
				}
			}
		else
			{
			dgemm_nt_lib(nx, N1*nu, nx, pGamma_0[N1-1], cnx, pGamma_u_Q[N1-1], cnx, 0, pH_St, cNnu, pH_St, cNnu, 0, 0);
			if(nzero_Q_N==0)
				dgeset_lib(nx, nu, 0.0, 0, pH_St+N1*nu*bs, cNnu);
			for(ii=0; ii<N1-1; ii++)
				{
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pGamma_0[ii], cnx, pGamma_u_Q[ii], cnx, 1, pH_St, cNnu, pH_St, cNnu, 0, 0);
				}
			}
		}
	
	}



void d_cond_q(int N, int nx, int nu, double **pA, double **b, int diag_hessian, int nzero_Q_N, double **pQ, double **q, double **pGamma_0, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	
	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], 1, b[ii], Gamma_b[ii]);
			}
		}
	
	// Gamma_b * Q + q
	if(compute_Gamma_b_q)
		{
		if(diag_hessian)
			{
			for(ii=0; ii<N1; ii++)
				{
				dgemv_diag_lib(nx, pQ[ii+1], Gamma_b[ii], 1, q[ii+1], Gamma_b_q[ii]);
				}
			}
		else
			{
			for(ii=0; ii<N1; ii++)
				{
				//dgemv_n_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				dsymv_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], 1, q[ii+1], Gamma_b_q[ii]);
				}
			}
		}
		
	// Gamma_0' * Gamma_b_q
	d_copy_mat(nx, 1, q[0], 1, H_q, 1);
	for(ii=0; ii<N1; ii++)
		{
		dgemv_n_lib(nx, nx, pGamma_0[ii], cnx, Gamma_b_q[ii], 1, H_q, H_q);
		}
	
	}



void d_cond_r(int N, int nx, int nu, double **pA, double **b, int diag_hessian, int nzero_Q_N, double **pQ, double **pS, double **q, double **r, double **pGamma_u, int compute_Gamma_b, double **Gamma_b, int compute_Gamma_b_q, double **Gamma_b_q, double *H_r)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	int N1 = N;
	if(nzero_Q_N==0)
		N1 = N-1;
	
	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], 1, b[ii], Gamma_b[ii]);
			}
		}

	// barS * Gamma_b
	d_copy_mat(nu, 1, r[0], 1, H_r, 1);
	if(diag_hessian)
		{
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nu, nx, pS[ii], cnx, Gamma_b[ii-1], 1, r[ii], H_r+ii*nu);
			}
		}
	else
		{
		for(ii=1; ii<N; ii++)
			{
			d_copy_mat(nu, 1, r[ii], 1, H_r+ii*nu, 1);
			}
		}
	
	// Gamma_b * Q + q
	if(compute_Gamma_b_q)
		{
		if(diag_hessian)
			{
			for(ii=0; ii<N1; ii++)
				{
				dgemv_diag_lib(nx, pQ[ii+1], Gamma_b[ii], 1, q[ii+1], Gamma_b_q[ii]);
				}
			}
		else
			{
			for(ii=0; ii<N1; ii++)
				{
				//dgemv_n_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], q[ii+1], Gamma_b_q[ii], 1);
				dsymv_lib(nx, nx, pQ[ii+1], cnx, Gamma_b[ii], 1, q[ii+1], Gamma_b_q[ii]);
				}
			}
		}

	// Gamma_u * Gamma_b_q
	for(ii=0; ii<N1; ii++)
		{
		dgemv_n_lib((ii+1)*nu, nx, pGamma_u[ii], cnx, Gamma_b_q[ii], 1, H_r, H_r);
		}
		
	}



// pGamma_x = [A_0' A_0'A_1' A_0'A_1'A_2' ... ], there is not I at the beginning !!!
void d_cond_A(int N, int nx, int nu, double **pA, int compute_Gamma_0, double **pGamma_0, double *pH_A)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj;

	if(compute_Gamma_0)
		{
		dgetr_lib(nx, nx, 0, pA[0], cnx, 0, pGamma_0[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			dgemm_nt_lib(nx, nx, nx, pGamma_0[ii-1], cnx, pA[ii], cnx, 0, pGamma_0[ii], cnx, pGamma_0[ii], cnx, 0, 0);
			}
		}
	
	dgetr_lib(nx, nx, 0, pGamma_0[N-1], cnx, 0, pH_A, cnx);

	}



void d_cond_B(int N, int nx, int nu, double **pA, double **pBt, int compute_Gamma_u, double **pGamma_u, double *pH_B)
	{
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, offset, i_temp;

	// Gamma_u
	if(compute_Gamma_u)
		{
		dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
		for(ii=1; ii<N; ii++)
			{
			offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
			dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
			dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
			}
		}
	
	dgetr_lib(N*nu, nx, 0, pGamma_u[N-1], cnx, 0, pH_B, cNnu);

	}



void d_cond_b(int N, int nx, int nu, double **pA, double **b, int compute_Gamma_b, double **Gamma_b, double *H_b)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii;

	// Gamma_b
	if(compute_Gamma_b)
		{
		d_copy_mat(nx, 1, b[0], 1, Gamma_b[0], 1);
		for(ii=1; ii<N; ii++)
			{
			dgemv_n_lib(nx, nx, pA[ii], cnx, Gamma_b[ii-1], 1, b[ii], Gamma_b[ii]);
			}
		}
	
	d_copy_mat(nx, 1, Gamma_b[N-1], 1, H_b, 1);
	
	}
	


void d_part_cond_RQrq(int N, int nx, int nu, double **pBAt, double **b, int diag_hessian, double **pRSQ, double **rq, double **pGamma_x, double **pGamma_u, double **pGamma_b, double *pM, double *pLam, double *diag, double *pBAtL, double *pH_R, double *pH_St, double *pH_Q, double *H_r, double *H_q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, kk, offset, i_temp;


	int nz = nx+nu;
	int cnz = (nz+ncl-1)/ncl*ncl;

	if(diag_hessian)
		{

		// final stage N1 = N-1
		ii = N;

		d_copy_mat(nx, 1, rq[ii-1]+nu, 1, H_q, 1);

		d_copy_mat(nu, 1, rq[ii-1], 1, H_r+(ii-1)*nu, 1);

		dgeset_lib(nu, nu, 0.0, 0, pLam, cnz);
		ddiain_lib(nu, pRSQ[ii-1], 0, pLam, cnz);

		dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);
		dgeset_lib((ii-1)*nu, nu, 0.0, 0, pH_R+(ii-1)*nu*bs, cNnu);

//		dgeset_lib(nx, nu, 0.0, 0, pH_St+(ii-1)*bs, cNnu);

		dgeset_lib(nx, nx, 0.0, 0, pH_Q, cnx);
		ddiain_lib(nx, pRSQ[ii-1]+nu, 0, pH_Q, cnx);

		// second final stage N1-1
		ii = N-1;
		if(ii>0) // at least another stage
			{

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			for(jj=0; jj<nx-3; jj+=4)
				{
				pH_Q[jj*cnx+0+(jj+0)*bs] = sqrt(pH_Q[jj*cnx+0+(jj+0)*bs]);
				pH_Q[jj*cnx+1+(jj+1)*bs] = sqrt(pH_Q[jj*cnx+1+(jj+1)*bs]);
				pH_Q[jj*cnx+2+(jj+2)*bs] = sqrt(pH_Q[jj*cnx+2+(jj+2)*bs]);
				pH_Q[jj*cnx+3+(jj+3)*bs] = sqrt(pH_Q[jj*cnx+3+(jj+3)*bs]);
				}
			for(kk=0; kk<nx-jj; kk++)
				{
				pH_Q[jj*cnx+kk+(jj+kk)*bs] = sqrt(pH_Q[jj*cnx+kk+(jj+kk)*bs]);
				}

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii-1], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
//				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0, 0);
				}
			else
				{
//				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			// decrease index
			ii--;

			}
		// first stages 
		for(; ii>0; ii--)
			{	

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii-1], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
//				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0, 0);
				}
			else
				{
//				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			}

		}
	else // dense hessian
		{

		// final stage N1 = N-1
		ii = N;

		d_copy_mat(nx, 1, rq[ii-1]+nu, 1, H_q, 1);

		dgecp_lib(nu, nu, 0, pRSQ[ii-1], cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

		if(ii>1)
			{
			dgetr_lib(nx, nu, nu, pRSQ[ii-1]+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
			dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, rq[ii-1], H_r+(ii-1)*nu);
			dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
//			dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0, 0);
			}
		else
			{
			d_copy_mat(nu, 1, rq[ii-1], 1, H_r+(ii-1)*nu, 1);
//			dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
			}

		dgecp_lib(nx, nx, nu, pRSQ[ii-1]+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

		// first stages 
		for(ii=N-1; ii>0; ii--)
			{	

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			drowad_lib(nx, 1.0, H_q, pBAtL+(nu+nx)/bs*bs*cnx+(nu+nx)%bs);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[ii-1], cnz, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
//				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0, 0);
				}
			else
				{
//				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			}

		}

	// transpose H in the lower triangular
	dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

	return;

	
	}



void d_part_cond_RSQrq(int N, int nx, int nu, double **pBAt, double **b, int diag_hessian, double **pRSQ, double **rq, double **pGamma_x, double **pGamma_u, double **pGamma_b, double *pM, double *pLam, double *diag, double *pBAtL, double *pH_R, double *pH_St, double *pH_Q, double *H_r, double *H_q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, kk, offset, i_temp;


	int nz = nx+nu;
	int cnz = (nz+ncl-1)/ncl*ncl;

	if(diag_hessian)
		{

		// final stage N1 = N-1
		ii = N;

		d_copy_mat(nx, 1, rq[ii-1]+nu, 1, H_q, 1);

		d_copy_mat(nu, 1, rq[ii-1], 1, H_r+(ii-1)*nu, 1);

		dgeset_lib(nu, nu, 0.0, 0, pLam, cnz);
		ddiain_lib(nu, pRSQ[ii-1], 0, pLam, cnz);

		dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);
		dgeset_lib((ii-1)*nu, nu, 0.0, 0, pH_R+(ii-1)*nu*bs, cNnu);

		dgeset_lib(nx, nu, 0.0, 0, pH_St+(ii-1)*bs, cNnu);

		dgeset_lib(nx, nx, 0.0, 0, pH_Q, cnx);
		ddiain_lib(nx, pRSQ[ii-1]+nu, 0, pH_Q, cnx);

		// second final stage N1-1
		ii = N-1;
		if(ii>0) // at least another stage
			{

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			for(jj=0; jj<nx-3; jj+=4)
				{
				pH_Q[jj*cnx+0+(jj+0)*bs] = sqrt(pH_Q[jj*cnx+0+(jj+0)*bs]);
				pH_Q[jj*cnx+1+(jj+1)*bs] = sqrt(pH_Q[jj*cnx+1+(jj+1)*bs]);
				pH_Q[jj*cnx+2+(jj+2)*bs] = sqrt(pH_Q[jj*cnx+2+(jj+2)*bs]);
				pH_Q[jj*cnx+3+(jj+3)*bs] = sqrt(pH_Q[jj*cnx+3+(jj+3)*bs]);
				}
			for(kk=0; kk<nx-jj; kk++)
				{
				pH_Q[jj*cnx+kk+(jj+kk)*bs] = sqrt(pH_Q[jj*cnx+kk+(jj+kk)*bs]);
				}

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii-1], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			// decrease index
			ii--;

			}
		// first stages 
		for(; ii>0; ii--)
			{	

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii-1], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			}

		}
	else // dense hessian
		{

		// final stage N1 = N-1
		ii = N;

		d_copy_mat(nx, 1, rq[ii-1]+nu, 1, H_q, 1);

		dgecp_lib(nu, nu, 0, pRSQ[ii-1], cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

		if(ii>1)
			{
			dgetr_lib(nx, nu, nu, pRSQ[ii-1]+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
			dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, rq[ii-1], H_r+(ii-1)*nu);
			dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
			dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
			}
		else
			{
			d_copy_mat(nu, 1, rq[ii-1], 1, H_r+(ii-1)*nu, 1);
			dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
			}

		dgecp_lib(nx, nx, nu, pRSQ[ii-1]+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

		// first stages 
		for(ii=N-1; ii>0; ii--)
			{	

			dsymv_lib(nx, nx, pH_Q, cnx, b[ii-1], 1, H_q, H_q);
			dgemv_n_lib(nz, nx, pBAt[ii-1], cnx, H_q, 1, rq[ii-1], pLam);
			d_copy_mat(nu, 1, pLam, 1, H_r+(ii-1)*nu, 1);
			d_copy_mat(nx, 1, pLam+nu, 1, H_q, 1);

			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii-1], cnx, pH_Q, cnx, pBAtL, cnx);
			drowad_lib(nx, 1.0, H_q, pBAtL+(nu+nx)/bs*bs*cnx+(nu+nx)%bs);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[ii-1], cnz, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemv_n_lib(nu, nx, pM, cnx, pGamma_b[ii-2], 1, H_r+(ii-1)*nu, H_r+(ii-1)*nu);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

			}

		}

	// transpose H in the lower triangular
	dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

	return;

	
	}



void d_part_cond_RSQ(int N, int nx, int nu, double **pBAt, int diag_hessian, double **pRSQ, double **pGamma_x, double **pGamma_u, double *pM, double *pLam, double *diag, double *pBAtL, double *pH_R, double *pH_St, double *pH_Q)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int ii, jj, kk, offset, i_temp;


	int nz = nx+nu;
	int cnz = (nz+ncl-1)/ncl*ncl;

	if(diag_hessian)
		{

		// final stage N1 = N-1
		ii = N;

		dgeset_lib(nu, nu, 0.0, 0, pLam, cnz);
		ddiain_lib(nu, pRSQ[N], 0, pLam, cnz);

		dgecp_lib(nu, nu, 0, pLam, cnz, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);
		dgeset_lib((N-1)*nu, nu, 0.0, 0, pH_R+(N-1)*nu*bs, cNnu);

		dgeset_lib(nx, nu, 0.0, 0, pH_St+(N-1)*bs, cNnu);

		dgeset_lib(nx, nx, 0.0, 0, pH_Q, cnx);
		ddiain_lib(nx, pRSQ[N-1]+nu, 0, pH_Q, cnx);

		// second final stage N1-1
		ii = N-1;
		if(ii>0) // at least another stage
			{

			for(jj=0; jj<nx-3; jj+=4)
				{
				pH_Q[jj*cnx+0+(jj+0)*bs] = sqrt(pH_Q[jj*cnx+0+(jj+0)*bs]);
				pH_Q[jj*cnx+1+(jj+1)*bs] = sqrt(pH_Q[jj*cnx+1+(jj+1)*bs]);
				pH_Q[jj*cnx+2+(jj+2)*bs] = sqrt(pH_Q[jj*cnx+2+(jj+2)*bs]);
				pH_Q[jj*cnx+3+(jj+3)*bs] = sqrt(pH_Q[jj*cnx+3+(jj+3)*bs]);
				}
			for(kk=0; kk<nx-jj; kk++)
				{
				pH_Q[jj*cnx+kk+(jj+kk)*bs] = sqrt(pH_Q[jj*cnx+kk+(jj+kk)*bs]);
				}

			dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);
//			dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
//			dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

			// decrease index
			ii--;

			}
		// first stages 
		for(; ii>0; ii--)
			{	
			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pLam, cnz, pLam, cnz);
			ddiaad_lib(nz, 1.0, pRSQ[ii], 0, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);
//			dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
//			dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

			}

		}
	else
		{

		// final stage N1 = N-1
		ii = N;

		dgecp_lib(nu, nu, 0, pRSQ[N], cnz, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);

		if(ii>1)
			{
			dgetr_lib(nx, nu, nu, pRSQ[N]+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
			dgemm_nt_lib((N-1)*nu, nu, nx, pGamma_u[N-2], cnx, pM, cnx, 0, pH_R+(N-1)*nu*bs, cNnu, pH_R+(N-1)*nu*bs, cNnu, 0, 0);
			dgemm_nt_lib(nx, nu, nx, pGamma_x[N-2], cnx, pM, cnx, 0, pH_St+(N-1)*nu*bs, cNnu, pH_St+(N-1)*nu*bs, cNnu, 0, 0);
			}
		else
			{
			dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
			}

		dgecp_lib(nx, nx, nu, pRSQ[N]+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);

		// first stages 
		for(ii=N-1; ii>0; ii--)
			{	
			dpotrf_lib_old(nx, nx, pH_Q, cnx, pH_Q, cnx, diag);
			dtrtr_l_lib(nx, 0, pH_Q, cnx, pH_Q, cnx);	

			dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pH_Q, cnx, pBAtL, cnx);
			dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[ii], cnz, pLam, cnz);
			dtrtr_l_lib(nu, 0, pLam, cnz, pLam, cnz);	

			dgecp_lib(nu, nu, 0, pLam, cnz, (ii-1)*nu, pH_R+((ii-1)*nu)/bs*bs*cNnu+((ii-1)*nu)%bs+(ii-1)*nu*bs, cNnu);

			if(ii>1)
				{
				dgetr_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pM, cnx);
				dgemm_nt_lib((ii-1)*nu, nu, nx, pGamma_u[ii-2], cnx, pM, cnx, 0, pH_R+(ii-1)*nu*bs, cNnu, pH_R+(ii-1)*nu*bs, cNnu, 0, 0);
				dgemm_nt_lib(nx, nu, nx, pGamma_x[ii-2], cnx, pM, cnx, 0, pH_St+(ii-1)*nu*bs, cNnu, pH_St+(ii-1)*nu*bs, cNnu, 0, 0);
				}
			else
				{
				dgecp_lib(nx, nu, nu, pLam+nu/bs*bs*cnz+nu%bs, cnz, 0, pH_St, cNnu);
				}

			dgecp_lib(nx, nx, nu, pLam+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pH_Q, cnx);
//			dpotrf_lib_old(nx, nx, pP, cnx, pP, cnx, diag);
//			dtrtr_l_lib(nx, 0, pP, cnx, pP, cnx);	

			}

		}

	// transpose H in the lower triangular
	dtrtr_u_lib(N*nu, pH_R, cNnu, pH_R, cNnu);

	return;

	
	}



int d_cond_lqcp_work_space(int N, int nx, int nu, int N2, int alg)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int nz = nu+nx;
	int pnx = (nx+bs-1)/bs*bs;
	int pnu = (nu+bs-1)/bs*bs;
	int pnz = (nz+bs-1)/bs*bs;
	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cnz = (nz+ncl-1)/ncl*ncl;

	// find problem size
	int N1 = N/N2; // floor
	int R1 = N - N2*N1; // the first r1 stages are of size N1+1
	int M1 = R1>0 ? N1+1 : N1;

	int work_space_size = 0;

	int jj;

	if(alg==0 || alg==1)
		{
		for(jj=0; jj<M1; jj++)
			{
			work_space_size += 3*pnx*cnx + 3*((jj+2)*nu+bs-1)/bs*bs*cnx + 2*pnx;
			}
		work_space_size += pnx*cnx + pnx + pnu*cnu + pnu*cnx;
		return work_space_size;
		}

	if(alg==2)
		{
		for(jj=0; jj<M1; jj++)
			{
			work_space_size += 1*pnx*cnx + 1*((jj+2)*nu+bs-1)/bs*bs*cnx + 2*pnx;
			}
		work_space_size += pnx + pnz*cnz + pnz*cnx + pnu*cnx;
		return work_space_size;
		}

	}



void d_cond_lqcp(int N, int nx, int nu, int alg, double **hpA, double **hpAt, double **hpBt, double **hb, double **hpBAt, int diag_hessian, double **hpQ, double **hpS, double **hpR, double **hr, double **hq, double **hpRSQ, double **hrq, int N2, int *nx2, int *nu2, double **hpA2, double **hpB2, double **hb2, double **hpR2, double **hpSt2, double **hpQ2, double **hr2, double **hq2, double *work_double)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int pnx = (nx+bs-1)/bs*bs;
	int pnu = (nu+bs-1)/bs*bs;
	int cnx = (nx+ncl-1)/ncl*ncl;
	int cnu = (nu+ncl-1)/ncl*ncl;

	// find problem size
	int N1 = N/N2; // floor
	int R1 = N - N2*N1; // the first r1 stages are of size N1+1
	int M1 = R1>0 ? N1+1 : N1;
	int T1;

	int ii, jj, nn;

	if(alg==0 || alg==1) // using N^3 n_x^2 or N^2 n_x^2 condensing algorithm
		{

		int use_Gamma_x_Q = 0;
		if(alg==0 && diag_hessian==0)
			use_Gamma_x_Q = 1;

		double *(hpGamma_x[M1]);
		double *(hpGamma_x_Q[M1]);
		double *(hpGamma_u[M1]);
		double *(hpGamma_u_Q[M1]);
		double *(hpGamma_u_Q_A[M1]);
		double *(hGamma_b[M1]);
		double *(hGamma_b_q[M1]);
		double *(hpL[M1+1]);
		double *pD;
		double *pM;
		double *work;

		double *ptr;
		ptr = work_double;

		for(jj=0; jj<M1; jj++)
			{
			hpGamma_x[jj] = ptr;
			ptr += pnx*cnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hpGamma_x_Q[jj] = ptr;
			ptr += pnx*cnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hpGamma_u[jj] = ptr;
			ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;
			}

		for(jj=0; jj<M1-1; jj++)
			{
			hpGamma_u_Q[jj] = ptr;
			ptr += ((jj+2)*nu+bs-1)/bs*bs*cnx;
			}
		jj = M1-1;
		hpGamma_u_Q[jj] = ptr;
		ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;

		for(jj=0; jj<M1-1; jj++)
			{
			hpGamma_u_Q_A[jj] = ptr;
			ptr += ((jj+2)*nu+bs-1)/bs*bs*cnx;
			}
		jj = M1-1;
		hpGamma_u_Q_A[jj] = ptr;
		ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;
		
		for(jj=0; jj<=M1; jj++)
			{
			hpL[jj] = ptr;
			if(diag_hessian)
				ptr += pnx;
			else
				ptr += pnx*cnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hGamma_b[jj] = ptr;
			ptr += pnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hGamma_b_q[jj] = ptr;
			ptr += pnx;
			}
		
		pD = ptr;
		ptr += pnu*cnu;

		pM = ptr;
		ptr += pnu*cnx;

		work = ptr;
		ptr += pnx;

		double *dummy;
		double **pdummy;


		// first stage
		nn = 0;
		jj = 0;

		T1 = jj<R1 ? M1 : N1;

		nx2[jj] = 0;
		nu2[jj] = T1*nu;


		// condense dynamic system
		//d_cond_A(T1, nx, nu, hpA+nn, 1, hpGamma_x, hpA2[jj]);

		d_cond_B(T1, nx, nu, hpA+nn, hpBt+nn, 1, hpGamma_u, hpB2[jj]);

		d_cond_b(T1, nx, nu, hpA+nn, hb+nn, 1, hGamma_b, hb2[jj]);


		// condense cost function
		//d_cond_Q(T1, nx, nu, hpA+nn, diag_Q, 0, hpQ+nn, hpL, 0, hpGamma_x, hpGamma_x_Q, hpQ2[jj], work);
		
		d_cond_R(T1, nx, nu, alg, hpA+nn, hpAt+nn, hpBt+nn, pdummy, diag_hessian, 0, hpQ+nn, 0, hpL, hpS+nn, hpR+nn, pdummy, pD, pM, dummy, dummy, dummy, dummy, 0, hpGamma_u, hpGamma_u_Q, hpGamma_u_Q_A, hpR2[jj]);

		//d_cond_St(T1, nx, nu, nzero_S, hpS+nn, 0, hpGamma_x, use_Gamma_x_Q, hpGamma_x_Q, hpGamma_u_Q, hpSt2[jj]);

		//d_cond_q(T1, nx, nu, hpA+nn, hb+nn, diag_Q, 0, hpQ+nn, hq+nn, hpGamma_x, 0, hGamma_b, 1, hGamma_b_q, hq2[jj]);

		d_cond_r(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hpS+nn, hq+nn, hr+nn, hpGamma_u, 0, hGamma_b, 1, hGamma_b_q, hr2[jj]);


		// increment stage counter
		nn += T1;
		jj++;


		// general stages
		for(; jj<N2; jj++)
			{

			T1 = jj<R1 ? M1 : N1;

			nx2[jj] = nx;
			nu2[jj] = T1*nu;


			// condense dynamic system
			d_cond_A(T1, nx, nu, hpA+nn, 1, hpGamma_x, hpA2[jj]);

			d_cond_B(T1, nx, nu, hpA+nn, hpBt+nn, 1, hpGamma_u, hpB2[jj]);

			d_cond_b(T1, nx, nu, hpA+nn, hb+nn, 1, hGamma_b, hb2[jj]);


			// condense cost function
			d_cond_Q(T1, nx, nu, hpA+nn, diag_hessian, 0, hpQ+nn, hpL, 0, hpGamma_x, hpGamma_x_Q, hpQ2[jj], work);
			
			d_cond_R(T1, nx, nu, alg, hpA+nn, hpAt+nn, hpBt+nn, pdummy, diag_hessian, 0, hpQ+nn, 1, hpL, hpS+nn, hpR+nn, pdummy, pD, pM, dummy, dummy, dummy, dummy, 0, hpGamma_u, hpGamma_u_Q, hpGamma_u_Q_A, hpR2[jj]);

			d_cond_St(T1, nx, nu, diag_hessian, hpS+nn, 0, hpGamma_x, use_Gamma_x_Q, hpGamma_x_Q, hpGamma_u_Q, hpSt2[jj]);

			d_cond_q(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hq+nn, hpGamma_x, 0, hGamma_b, 1, hGamma_b_q, hq2[jj]);

			d_cond_r(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hpS+nn, hq+nn, hr+nn, hpGamma_u, 0, hGamma_b, 0, hGamma_b_q, hr2[jj]);


			// increment stage counter
			nn += T1;

			}

			return;

		}
	if(alg==2) // using N^2 n_x^3 condensing algorithm
		{

#if 1

		int nz = nu+nx;
		int pnz = (nz+bs-1)/bs*bs;
		int cnz = (nz+ncl-1)/ncl*ncl;

		double *(hpGamma_x[M1]);
		double *(hpGamma_u[M1]);
		double *(hpGamma_u_Q[M1]); // TODO remove !!!!!
		double *(hpGamma_u_Q_A[M1]); // TODO remove !!!!!
		double *(hGamma_b[M1]);
		double *(hGamma_b_q[M1]);
		double *pBAtL;
		double *pLam;
		double *pM;
		double *diag;

		double *ptr;
		ptr = work_double;

		for(jj=0; jj<M1; jj++)
			{
			hpGamma_x[jj] = ptr;
			ptr += pnx*cnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hpGamma_u[jj] = ptr;
			ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;
			}

#if 0
		for(jj=0; jj<M1-1; jj++)
			{
			hpGamma_u_Q[jj] = ptr;
			ptr += ((jj+2)*nu+bs-1)/bs*bs*cnx;
			}
		jj = M1-1;
		hpGamma_u_Q[jj] = ptr;
		ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;

		for(jj=0; jj<M1-1; jj++)
			{
			hpGamma_u_Q_A[jj] = ptr;
			ptr += ((jj+2)*nu+bs-1)/bs*bs*cnx;
			}
		jj = M1-1;
		hpGamma_u_Q_A[jj] = ptr;
		ptr += ((jj+1)*nu+bs-1)/bs*bs*cnx;
#endif

		pBAtL = ptr;
		ptr += pnz*cnx;

		pLam = ptr;
		ptr += pnz*cnz;

		pM = ptr;
		ptr += pnu*cnx;

		for(jj=0; jj<M1; jj++)
			{
			hGamma_b[jj] = ptr;
			ptr += pnx;
			}

		for(jj=0; jj<M1; jj++)
			{
			hGamma_b_q[jj] = ptr;
			ptr += pnx;
			}
		
		diag = ptr;
		ptr += pnx;

		double *dummy;
		double **pdummy;


		// first stage
		nn = 0;
		jj = 0;

#if 1
		T1 = jj<R1 ? M1 : N1;

		nx2[jj] = 0;
		nu2[jj] = T1*nu;


		// condense dynamic system
		//d_cond_A(T1, nx, nu, hpA+nn, 1, hpGamma_x, hpA2[jj]);

		d_cond_B(T1, nx, nu, hpA+nn, hpBt+nn, 1, hpGamma_u, hpB2[jj]);

		d_cond_b(T1, nx, nu, hpA+nn, hb+nn, 1, hGamma_b, hb2[jj]);


		// condense cost function
		//d_cond_Q(T1, nx, nu, hpA+nn, diag_Q, 0, hpQ+nn, hpL, 0, hpGamma_x, hpGamma_x_Q, hpQ2[jj], diag);
		
		//d_cond_R(T1, nx, nu, 0, hpA+nn, hpAt+nn, hpBt+nn, pdummy, diag_hessian, 0, hpQ+nn, 0, pdummy, hpS+nn, hpR+nn, pdummy, pLam, pM, dummy, dummy, dummy, dummy, 0, hpGamma_u, hpGamma_u_Q, hpGamma_u_Q_A, hpR2[jj]);

		//d_cond_St(T1, nx, nu, nzero_S, hpS+nn, 0, hpGamma_x, use_Gamma_x_Q, hpGamma_x_Q, hpGamma_u_Q, hpSt2[jj]);

		//d_cond_q(T1, nx, nu, hpA+nn, hb+nn, diag_Q, 0, hpQ+nn, hq+nn, hpGamma_x, 0, hGamma_b, 1, hGamma_b_q, hq2[jj]);

		//d_cond_r(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hpS+nn, hq+nn, hr+nn, hpGamma_u, 0, hGamma_b, 1, hGamma_b_q, hr2[jj]);

		d_part_cond_RQrq(T1, nx, nu, hpBAt+nn, hb+nn, diag_hessian, hpRSQ+nn, hrq+nn, hpGamma_x, hpGamma_u, hGamma_b, pM, pLam, diag, pBAtL, hpR2[jj], hpSt2[jj], hpQ2[jj], hr2[jj], hq2[jj]); // XXX no S !!!!!

		// increment stage counter
		nn += T1;
		jj++;
#endif


		// general stages
		for(; jj<N2; jj++)
			{

			T1 = jj<R1 ? M1 : N1;

			nx2[jj] = nx;
			nu2[jj] = T1*nu;


			// condense dynamic system
			d_cond_A(T1, nx, nu, hpA+nn, 1, hpGamma_x, hpA2[jj]);

			d_cond_B(T1, nx, nu, hpA+nn, hpBt+nn, 1, hpGamma_u, hpB2[jj]);

			d_cond_b(T1, nx, nu, hpA+nn, hb+nn, 1, hGamma_b, hb2[jj]);


			// condense cost function
			//d_cond_Q(T1, nx, nu, hpA+nn, diag_hessian, 0, hpQ+nn, hpL, 0, hpGamma_x, hpGamma_x_Q, hpQ2[jj], diag);
			
			//d_cond_R(T1, nx, nu, alg, hpA+nn, hpAt+nn, hpBt+nn, pdummy, diag_hessian, 0, hpQ+nn, 1, hpL, hpS+nn, hpR+nn, pdummy, pD, pM, dummy, dummy, dummy, dummy, 0, hpGamma_u, hpGamma_u_Q, hpGamma_u_Q_A, hpR2[jj]);

			//d_cond_St(T1, nx, nu, diag_hessian, hpS+nn, 0, hpGamma_x, use_Gamma_x_Q, hpGamma_x_Q, hpGamma_u_Q, hpSt2[jj]);

			//d_cond_q(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hq+nn, hpGamma_x, 0, hGamma_b, 1, hGamma_b_q, hq2[jj]);

			//d_cond_r(T1, nx, nu, hpA+nn, hb+nn, diag_hessian, 0, hpQ+nn, hpS+nn, hq+nn, hr+nn, hpGamma_u, 0, hGamma_b, 0, hGamma_b_q, hr2[jj]);

			d_part_cond_RSQrq(T1, nx, nu, hpBAt+nn, hb+nn, diag_hessian, hpRSQ+nn, hrq+nn, hpGamma_x, hpGamma_u, hGamma_b, pM, pLam, diag, pBAtL, hpR2[jj], hpSt2[jj], hpQ2[jj], hr2[jj], hq2[jj]);


			// increment stage counter
			nn += T1;

			}

			return;


#endif

		}

	}



void d_cond_fact_R(int N, int nx, int nu, int nx2_fact, double **pA, double **pAt, double **pBt, int diag_hessian, double **pQ, double **pS, double **pR, double *pQs, double *pM, double *pD, int compute_Gamma_u, double **pGamma_u, double **pGamma_w, double *diag, double **pBAt, double **pRSQ, double *pL, double *pBAtL, double *pH_R)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int cnx = (nx+ncl-1)/ncl*ncl;
	int pnu = (nu+bs-1)/bs*bs;
	int pnx = (nx+bs-1)/bs*bs;
	int cnu = (nu+ncl-1)/ncl*ncl;
	int cNnu = (N*nu+ncl-1)/ncl*ncl;
	//int cNnx = (N*nx+ncl-1)/ncl*ncl;

	int nz = nx+nu;
	int cnz = (nz+ncl-1)/ncl*ncl;
	int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

	int ii, jj, offset, i_temp;


	if(nx2_fact) // N^2 n_x^2 algorithm
		{

		// Gamma_u
		if(compute_Gamma_u)
			{
			dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
			for(ii=1; ii<N; ii++)
				{
				offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
				dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
				dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
				}
			}

		if(diag_hessian)
			{

			// last stage
			dgecp_lib(nx, nx, 0, pQ[N], cnx, 0, pQs, cnx);
			dgemm_diag_right_lib(N*nu, nx, pGamma_u[N-1], cnx, pQs, 0, pGamma_w[N-1], cnx, pGamma_w[N-1], cnx);

			// middle stages
			for(ii=N-1; ii>0; ii--)
				{

#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pAt[ii], cnx, pGamma_w[ii], cnx, 0, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 1, 1);
#else
				dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_w[ii], cnx, pAt[ii], cnx, 0, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 0, 0);
#endif

				dgecp_lib(nu, nx, ii*nu, pGamma_w[ii]+(ii*nu)/bs*bs*cnx+(ii*nu)%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[ii], cnx, 0, pD, cnu, pD, cnu, 0, 0);
				ddiaad_lib(nu, 1.0, pR[ii], 0, pD, cnu);
				dgetr_lib(nu, nx, ii*nu, pGamma_w[ii-1]+(ii*nu)/bs*bs*cnx+ii*nu%bs, cnx, 0, pD+pnu*cnu, cnu); // tr?? cp??

				dpotrf_lib_old(pnu+nx, nu, pD, cnu, pD, cnu, diag); // TODO copy last part in the hole
				dgecp_lib(nu, nu, 0, pD, cnu, (N-1-ii)*nu, pH_R+((N-1-ii)*nu)/bs*bs*cNnu+((N-1-ii)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
			
				dgemm_nn_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pD+pnu*cnu, cnu, 0, pH_R+(N-1)*nu*bs, cNnu, pH_R+(N-1)*nu*bs, cNnu, 0, 0);
				for(jj=0; jj<ii; jj++)
					{
					dgecp_lib(nu, nu, jj*nu, pH_R+(jj*nu)/bs*bs*cNnu+(jj*nu)%bs+(N-1)*nu*bs, cNnu, (N-1-jj)*nu, pH_R+((N-1-jj)*nu)/bs*bs*cNnu+((N-1-jj)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
					}

				dgeset_lib(nx, nx, 0.0, 0, pQs, cnx);
				ddiain_lib(nx, pQ[ii], 0, pQs, cnx);
				dsyrk_nt_lib(nx, nx, nu, pD+pnu*cnu, cnu, pD+pnu*cnu, cnu, -1, pQs, cnx, pQs, cnx);
				dtrtr_l_lib(nx, 0, pQs, cnx, pQs, cnx);	
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, ii*nu, nx, pQs, cnx, pGamma_u[ii-1], cnx, 1, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 1, 1);
#else
				dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pQs, cnx, 1, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 0, 0);
#endif

				}

			dgecp_lib(nu, nx, ii*nu, pGamma_w[0], cnx, 0, pM, cnx);
			dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[0], cnx, 0, pD, cnu, pD, cnu, 0, 0);
			ddiaad_lib(nu, 1.0, pR[0], 0, pD, cnu);
			dpotrf_lib_old(nu, nu, pD, cnu, pD, cnu, diag);
			dgecp_lib(nu, nu, 0, pD, cnu, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);

			}
		else
			{

			// last stage
			dgecp_lib(nx, nx, 0, pQ[N], cnx, 0, pQs, cnx);
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
			dgemm_nt_lib(nx, N*nu, nx, pQs, cnx, pGamma_u[N-1], cnx, 0, pGamma_w[N-1], cnx, pGamma_w[N-1], cnx, 0, 1);
#else
			dgemm_nt_lib(N*nu, nx, nx, pGamma_u[N-1], cnx, pQs, cnx, 0, pGamma_w[N-1], cnx, pGamma_w[N-1], cnx, 0, 0);
#endif

			// middle stages
			for(ii=N-1; ii>0; ii--)
				{

#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, (ii+1)*nu, nx, pAt[ii], cnx, pGamma_w[ii], cnx, 0, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 1, 1);
#else
				dgemm_nt_lib((ii+1)*nu, nx, nx, pGamma_w[ii], cnx, pAt[ii], cnx, 0, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 0, 0);
#endif
				//dgecp_lib(nu, nx, 0, pS[ii], cnx, (ii)*nu, pGamma_w[ii-1]+(ii)*nu/bs*bs*cnx+(ii)*nu%bs, cnx);
				dgead_lib(nu, nx, 1.0, 0, pS[ii], cnx, (ii)*nu, pGamma_w[ii-1]+(ii)*nu/bs*bs*cnx+(ii)*nu%bs, cnx);

				dgecp_lib(nu, nx, ii*nu, pGamma_w[ii]+(ii*nu)/bs*bs*cnx+(ii*nu)%bs, cnx, 0, pM, cnx);
				dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[ii], cnx, 1, pR[ii], cnu, pD, cnu, 0, 0);
				dgetr_lib(nu, nx, ii*nu, pGamma_w[ii-1]+(ii*nu)/bs*bs*cnx+ii*nu%bs, cnx, 0, pD+pnu*cnu, cnu); // tr?? cp??

				dpotrf_lib_old(pnu+nx, nu, pD, cnu, pD, cnu, diag); // TODO copy last part in the hole
				dgecp_lib(nu, nu, 0, pD, cnu, (N-1-ii)*nu, pH_R+((N-1-ii)*nu)/bs*bs*cNnu+((N-1-ii)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
			
				dgemm_nn_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pD+pnu*cnu, cnu, 0, pH_R+(N-1)*nu*bs, cNnu, pH_R+(N-1)*nu*bs, cNnu, 0, 0);
				for(jj=0; jj<ii; jj++)
					{
					dgecp_lib(nu, nu, jj*nu, pH_R+(jj*nu)/bs*bs*cNnu+(jj*nu)%bs+(N-1)*nu*bs, cNnu, (N-1-jj)*nu, pH_R+((N-1-jj)*nu)/bs*bs*cNnu+((N-1-jj)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
					}

				dsyrk_nt_lib(nx, nx, nu, pD+pnu*cnu, cnu, pD+pnu*cnu, cnu, -1, pQ[ii], cnx, pQs, cnx);
				dtrtr_l_lib(nx, 0, pQs, cnx, pQs, cnx);	
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, ii*nu, nx, pQs, cnx, pGamma_u[ii-1], cnx, 1, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 1, 1);
#else
				dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pQs, cnx, 1, pGamma_w[ii-1], cnx, pGamma_w[ii-1], cnx, 0, 0);
#endif

				}

			dgecp_lib(nu, nx, ii*nu, pGamma_w[0], cnx, 0, pM, cnx);
			dgemm_nt_lib(nu, nu, nx, pM, cnx, pBt[0], cnx, 1, pR[0], cnu, pD, cnu, 0, 0);
			dpotrf_lib_old(nu, nu, pD, cnu, pD, cnu, diag);
			dgecp_lib(nu, nu, 0, pD, cnu, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);

			}

		}
	else // N^2 n_x^3 algorithm
		{

		// Gamma_u
		if(compute_Gamma_u)
			{
			dgecp_lib(nu, nx, 0, pBt[0], cnx, 0, pGamma_u[0], cnx);
			for(ii=1; ii<N-1; ii++)
				{
				offset = ii*nu;
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
				dgemm_nt_lib(nx, ii*nu, nx, pA[ii], cnx, pGamma_u[ii-1], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 1); // (A * Gamma_u^T)^T
#else
				dgemm_nt_lib(ii*nu, nx, nx, pGamma_u[ii-1], cnx, pA[ii], cnx, 0, pGamma_u[ii], cnx, pGamma_u[ii], cnx, 0, 0); // Gamma_u * A^T
#endif
				dgecp_lib(nu, nx, 0, pBt[ii], cnx, offset, pGamma_u[ii]+offset/bs*bs*cnx+offset%bs, cnx);
				}
			}

		if(diag_hessian)
			{

			// final stage 
			dgeset_lib(nx, nx, 0.0, 0, pL, cnl);
			ddiain_sqrt_lib(nx, pRSQ[N]+nu, 0, pL, cnl);

			dtrtr_l_lib(nx, 0, pL, cnl, pL+(ncl)*bs, cnl);	

			// middle stages 
			for(ii=N-1; ii>0; ii--)
				{	
				dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pL+(ncl)*bs, cnl, pBAtL, cnx);
				if(nz<128)
					{
					dgeset_lib(nz, nz, 0.0, 0, pL, cnl);
					ddiain_lib(nz, pRSQ[ii], 0, pL, cnl);
					dsyrk_dpotrf_lib_old(nz, nz, nx, pBAtL, cnx, 1, pL, cnl, pL, cnl, diag, 0);
					}
				else
					{
					dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 0, pL, cnl, pL, cnl);
					ddiaad_lib(nz, 1.0, pRSQ[ii], 0, pL, cnl);
					dpotrf_lib_old(nz, nz, pL, cnl, pL, cnl, diag);
					}

				dgecp_lib(nu, nu, 0, pL, cnl, (N-1-ii)*nu, pH_R+((N-1-ii)*nu)/bs*bs*cNnu+((N-1-ii)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
				dgecp_lib(nx, nu, nu, pL+nu/bs*bs*cnl+nu%bs, cnl, 0, pD+pnu*cnu, cnu);
				dgemm_nn_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pD+pnu*cnu, cnu, 0, pH_R+(N-1)*nu*bs, cNnu, pH_R+(N-1)*nu*bs, cNnu, 0, 0);
				for(jj=0; jj<ii; jj++)
					{
					dgecp_lib(nu, nu, jj*nu, pH_R+(jj*nu)/bs*bs*cNnu+(jj*nu)%bs+(N-1)*nu*bs, cNnu, (N-1-jj)*nu, pH_R+((N-1-jj)*nu)/bs*bs*cNnu+((N-1-jj)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
					}

				dtrtr_l_lib(nx, nu, pL+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, pL+(ncl)*bs, cnl);	
				}

			// first stage 
			dtrmm_nt_u_lib(nu, nx, pBAt[0], cnx, pL+(ncl)*bs, cnl, pBAtL, cnx);
			dgeset_lib(nu, nu, 0.0, 0, pL, cnl);
			ddiain_lib(nu, pRSQ[0], 0, pL, cnl);
			dsyrk_dpotrf_lib_old(nu, nu, nx, pBAtL, cnx, 1, pL, cnl, pL, cnl, diag, 0);

			dgecp_lib(nu, nu, 0, pL, cnl, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);

			}
		else
			{

			// final stage 
			dgecp_lib(nx, nx, nu, pRSQ[N]+nu/bs*bs*cnz+nu%bs+nu*bs, cnz, 0, pL, cnl);
			dpotrf_lib_old(nx, nx, pL, cnl, pL, cnl, diag);

			dtrtr_l_lib(nx, 0, pL, cnl, pL+(ncl)*bs, cnl);	

			// middle stages 
			for(ii=N-1; ii>0; ii--)
				{	
				dtrmm_nt_u_lib(nz, nx, pBAt[ii], cnx, pL+(ncl)*bs, cnl, pBAtL, cnx);
				if(nz<128)
					{
					dsyrk_dpotrf_lib_old(nz, nz, nx, pBAtL, cnx, 1, pRSQ[ii], cnz, pL, cnl, diag, 0);
					}
				else
					{
					dsyrk_nt_lib(nz, nz, nx, pBAtL, cnx, pBAtL, cnx, 1, pRSQ[ii], cnz, pL, cnl);
					dpotrf_lib_old(nz, nz, pL, cnl, pL, cnl, diag);
					}

				dgecp_lib(nu, nu, 0, pL, cnl, (N-1-ii)*nu, pH_R+((N-1-ii)*nu)/bs*bs*cNnu+((N-1-ii)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
				dgecp_lib(nx, nu, nu, pL+nu/bs*bs*cnl+nu%bs, cnl, 0, pD+pnu*cnu, cnu);
				dgemm_nn_lib((ii)*nu, nu, nx, pGamma_u[ii-1], cnx, pD+pnu*cnu, cnu, 0, pH_R+(N-1)*nu*bs, cNnu, pH_R+(N-1)*nu*bs, cNnu, 0, 0);
				for(jj=0; jj<ii; jj++)
					{
					dgecp_lib(nu, nu, jj*nu, pH_R+(jj*nu)/bs*bs*cNnu+(jj*nu)%bs+(N-1)*nu*bs, cNnu, (N-1-jj)*nu, pH_R+((N-1-jj)*nu)/bs*bs*cNnu+((N-1-jj)*nu)%bs+(N-1-ii)*nu*bs, cNnu);
					}

				dtrtr_l_lib(nx, nu, pL+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, pL+(ncl)*bs, cnl);	
				}

			// first stage 
			dtrmm_nt_u_lib(nz, nx, pBAt[0], cnx, pL+(ncl)*bs, cnl, pBAtL, cnx);
			dsyrk_dpotrf_lib_old(nz, nu, nx, pBAtL, cnx, 1, pRSQ[0], cnz, pL, cnl, diag, 0);

			dgecp_lib(nu, nu, 0, pL, cnl, (N-1)*nu, pH_R+((N-1)*nu)/bs*bs*cNnu+((N-1)*nu)%bs+(N-1)*nu*bs, cNnu);

			}

		}

	}



	
