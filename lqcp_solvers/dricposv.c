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

#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/block_size.h"



/* version tailored for mpc (x0 fixed) */
void dricposv_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	
	const int nz = nx+nu+1;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = nx+pad+cnz;

	int ii, jj;

	/* factorization and backward substitution */

	// final stage 
	dsyrk_dpotrf_pp_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag);

/*	d_transpose_pmat_lo(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);*/
	dtrtr_l_p_lib(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);	

/*d_print_pmat(pnz, cnl, bs, hpL[N], cnl);*/

	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		dtrmm_ppp_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl); // TODO allow 'rectanguar' B
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		dsyrk_dpotrf_pp_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
/*		d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);*/
		dtrtr_l_p_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
		}

	// first stage 
	dtrmm_ppp_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
	dsyrk_dpotrf_pp_lib(nz, nx, ((nu+2-1)/2)*2, hpL[0], cnl, hpQ[0], cnz, diag);
	for(jj=0; jj<nu; jj++) hpL[0][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal

d_print_pmat(pnz, cnl, bs, hpL[0], cnl);
exit(1);

	// forward substitution 
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
		dtrsv_dgemv_p_t_lib(nx+nu, nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hpL[ii+1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
			dtrmv_p_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1); // TODO remove nu
			dtrmv_p_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
	}



void dricpotrs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int nz = nx+nu+1;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = nx+pad+cnz;

	int ii, jj;
	
	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) work[jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b
		dtrmv_p_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);
		for(jj=0; jj<nx; jj++) work[jj] = hq[N-ii][nu+jj]; // copy p
		dtrmv_p_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+pnz, work, 1); // L*(L'*b) + p
		dgemv_p_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		dtrsv_dgemv_p_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);
		}

/*d_print_pmat(nz, nz, bs, hpL[0]+(nx+pad)*bs, cnl);*/
/*exit(3);*/

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		dtrsv_dgemv_p_t_lib(nx+nu, nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			dtrmv_p_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);
			dtrmv_p_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}

