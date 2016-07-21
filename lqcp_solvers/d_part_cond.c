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



// pGamma_x = [A_0' A_0'A_1' A_0'A_1'A_2' ..... A_0'...A_{N-1}'], there is not I at the beginning !!!
// pGamma[ii] has size nx[0] x nx[ii+1]
void d_cond_A(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_x0, double *pBAbt2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	

	dgecp_lib(nx[0], nx[1], nu[0], pBAbt[0]+nu[0]/bs*bs*cnx[1]+nu[0]%bs, cnx[1], 0, pGamma_x0[0], cnx[1]);

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		dgemm_nt_lib(nx[0], nx[ii+1], nx[ii], pGamma_x0[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_x0[ii], cnx[ii+1], pGamma_x0[ii], cnx[ii+1], 0, 0);
		}
	
	dgecp_lib(nx[0], nx[N], 0, pGamma_x0[N-1], cnx[N], nu2, pBAbt2+nu2/bs*bs*cnx[N]+nu2%bs, cnx[N]);

	}



//void d_cond_B(int N, int nx, int nu, double **pA, double **pBt, int compute_Gamma_u, double **pGamma_u, double *pH_B)
void d_cond_B(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_u, double *pBAbt2)
	{
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	int nu_tmp = 0;

	// Gamma_u
	dgecp_lib(nu[0], nx[1], 0, pBAbt[0], cnx[1], 0, pGamma_u[0], cnx[1]);
	nu_tmp += nu[0];

	for(ii=1; ii<N; ii++)
		{
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx[ii+1], nu_tmp, nx[ii], work, cnx[ii], pGamma_u[ii-1], cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib(nu_tmp, nx[ii+1], nx[ii], pGamma_u[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 0); // Gamma_u * A^T
#endif
		dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], nu_tmp, pGamma_u[ii]+nu_tmp/bs*bs*cnx[ii+1]+nu_tmp%bs, cnx[ii+1]);
		nu_tmp += nu[ii];
		}
	
	dgecp_lib(nu_tmp, nx[N], 0, pGamma_u[N-1], cnx[N], 0, pBAbt2, cnx[N]);

	}



void d_cond_b(int N, int *nx, int *nu, double **pBAbt, double *work, double **Gamma_b, double *pBAbt2)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	// Gamma_b
	ii = 0;
	dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	for(ii=1; ii<N; ii++)
		{
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		}
	
	dgetr_lib(nx[N], 1, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);
	
	}
	


void d_cond_BAb(int N, int *nx, int *nu, double **pBAbt, double *work, double **pGamma_u, double **pGamma_x0, double **Gamma_b, double *pBAbt2)
	{

	// XXX can merge Gamma matrices ???
	
	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;

	int cnx[N+1];
	int nu2 = 0;
	for(ii=0; ii<=N; ii++)
		{
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		nu2 += nu[ii];
		}
	
	int nu_tmp = 0;

	ii = 0;
	// B
	dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], 0, pGamma_u[ii], cnx[ii+1]);
	// A
	dgecp_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, pGamma_x0[ii], cnx[ii+1]);
	// b
	dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
	//
	nu_tmp += nu[0];

	for(ii=1; ii<N; ii++)
		{
		// TODO check for equal pointers and avoid copy
		dgetr_lib(nx[ii], nx[ii+1], nu[ii], pBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1], 0, work, cnx[ii]); // pA in work
		// B
#if defined(TARGET_X64_AVX2) || defined(TARGET_X64_AVX) || defined(TARGET_C99_4X4)
		dgemm_nt_lib(nx[ii+1], nu_tmp, nx[ii], work, cnx[ii], pGamma_u[ii-1], cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 1); // (A * Gamma_u^T)^T
#else
		dgemm_nt_lib(nu_tmp, nx[ii+1], nx[ii], pGamma_u[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_u[ii], cnx[ii+1], pGamma_u[ii], cnx[ii+1], 0, 0); // Gamma_u * A^T
#endif
		dgecp_lib(nu[ii], nx[ii+1], 0, pBAbt[ii], cnx[ii+1], nu_tmp, pGamma_u[ii]+nu_tmp/bs*bs*cnx[ii+1]+nu_tmp%bs, cnx[ii+1]);
		// A
		dgemm_nt_lib(nx[0], nx[ii+1], nx[ii], pGamma_x0[ii-1], cnx[ii], work, cnx[ii], 0, pGamma_x0[ii], cnx[ii+1], pGamma_x0[ii], cnx[ii+1], 0, 0);
		// b
		dgetr_lib(1, nx[ii+1], nu[ii]+nx[ii], pBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1], 0, Gamma_b[ii], 1);
		dgemv_n_lib(nx[ii], nx[ii+1], work, cnx[ii+1], Gamma_b[ii-1], 1, Gamma_b[ii], Gamma_b[ii]);
		//
		nu_tmp += nu[ii];
		}
	
	// B
	dgecp_lib(nu_tmp, nx[N], 0, pGamma_u[N-1], cnx[N], 0, pBAbt2, cnx[N]);
	// A
	dgecp_lib(nx[0], nx[N], 0, pGamma_x0[N-1], cnx[N], nu2, pBAbt2+nu2/bs*bs*cnx[N]+nu2%bs, cnx[N]);
	// b
	dgetr_lib(nx[N], 1, 0, Gamma_b[N-1], 1, nu2+nx[0], pBAbt2+(nu2+nx[0])/bs*bs*cnx[N]+(nu2+nx[0])%bs, cnx[N]);

	}




