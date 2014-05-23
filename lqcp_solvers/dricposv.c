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
	const int d_ncl = D_NCL;
	
	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int sda = pnz; // TODO remove

	int pnx = (d_ncl-nx%d_ncl)%d_ncl; // packing between BAbtL & P

	int ii, jj;
	
	int nz = nx+nu+1;

	/* factorization and backward substitution */

	// final stage 
/*	dsyrk_dpotrf_pp_lib(nz, 0, nz, hpL[N]+(nx+pnx)*bs, 2*sda, hpQ[N], sda, diag);*/
	dsyrk_dpotrf_pp_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+(nu/bs)*bs*bs, 2*sda, hpQ[N]+(nu/bs)*bs*sda+(nu/bs)*bs*bs, sda, diag);

/*d_print_pmat(nz, nz, bs, hpL[N]+(nx+pnx)*bs, 2*sda);*/
/*exit(2);*/

	d_transpose_pmat_lo(nx, nu, hpL[N]+(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, hpL[N]+(nx+pnx+d_ncl)*bs, 2*sda);

	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		dtrmm_ppp_lib(nz, nx, hpBAbt[N-ii-1], sda, hpL[N-ii]+(nx+pnx+d_ncl)*bs, 2*sda, hpL[N-ii-1], 2*sda); // TODO allow 'rectanguar' B
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*(2*sda)+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*2*sda+(nx+nu)%bs+(nx+pnx+nu+jj)*bs];
		dsyrk_dpotrf_pp_lib(nz, nx, nu+nx, hpL[N-ii-1], 2*sda, hpQ[N-ii-1], sda, diag);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pnx)*bs+(jj/bs)*bs*(2*sda)+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
		d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, hpL[N-ii-1]+(nx+pnx+d_ncl)*bs, 2*sda);
		}

	// first stage 
	dtrmm_ppp_lib(nz, nx, hpBAbt[0], sda, hpL[1]+(nx+pnx+d_ncl)*bs, 2*sda, hpL[0], 2*sda);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*(2*sda)+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*2*sda+(nx+nu)%bs+(nx+pnx+nu+jj)*bs];
	dsyrk_dpotrf_pp_lib(nz, nx, ((nu+2-1)/2)*2, hpL[0], 2*sda, hpQ[0], sda, diag);
	for(jj=0; jj<nu; jj++) hpL[0][(nx+pnx)*bs+(jj/bs)*bs*(2*sda)+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal



	// forward substitution 
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pnx)*bs+((nu+nx)/bs)*bs*(2*sda)+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx, nu, nu, &hpL[ii][(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+nu%bs], 2*sda, &hux[ii][nu], &hux[ii][0], -1);
		dtrsv_p_t_lib(nu, &hpL[ii][(nx+pnx)*bs], 2*sda, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[nu+jj] = hpL[ii+1][(nx+pnx)*bs+((nu+nx)/bs)*bs*(2*sda)+(nu+nx)%bs+bs*(nu+jj)]; // work space
			dtrmv_p_t_lib(nx, nu, hpL[ii+1]+(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, &hux[ii+1][nu], &work[nu], 1); // L'*pi
			dtrmv_p_n_lib(nx, nu, hpL[ii+1]+(nx+pnx)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, &work[nu], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
/*exit(3);*/

	}



void dricpotrs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int d_ncl = D_NCL;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int sda = pnz; // TODO remove
	int pnx = bs*((nx+bs-1)/bs);

	int pad = (d_ncl-nx%d_ncl)%d_ncl; // packing between BAbtL & P

	int ii, jj;
	
	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) work[nu+jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj]; // copy b
		dtrmv_p_t_lib(nx, nu, hpL[N-ii]+(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, work+nu, work+pnz+nu, 0); // L'*b
		for(jj=0; jj<nx; jj++) work[nu+jj] = hq[N-ii][nu+jj]; // copy p
		dtrmv_p_n_lib(nx, nu, hpL[N-ii]+(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, work+pnz+nu, work+nu, 1); // L*(L'*b) + p
/*		dgemv_p_n_lib(nx+nu, nx, 0, hpBAbt[N-ii-1], sda, work+nu, hq[N-ii-1], 1);*/
		dgemv_p_n_lib(nx+nu, nx, hpBAbt[N-ii-1], sda, work+nu, hq[N-ii-1], 1);
/*		dtrsv_p_n_lib(nu, hpL[N-ii-1]+(nx+pad)*bs, 2*sda, hq[N-ii-1]);*/
/*		dgemv_p_n_lib(nx, nu, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs, 2*sda, hq[N-ii-1], hq[N-ii-1]+nu, -1);*/
		dtrsv_dgemv_p_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, 2*sda, hq[N-ii-1]);
		}


	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		dgemv_p_t_lib(nx, nu, nu, &hpL[ii][(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs], 2*sda, &hux[ii][nu], &hux[ii][0], -1);
		dtrsv_p_t_lib(nu, &hpL[ii][(nx+pad)*bs], 2*sda, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		dgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			dtrmv_p_t_lib(nx, nu, hpL[ii+1]+(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, &hux[ii+1][nu], &work[nu], 0); // L'*pi
			dtrmv_p_n_lib(nx, nu, hpL[ii+1]+(nx+pad)*bs+(nu/bs)*bs*(2*sda)+nu%bs+nu*bs, 2*sda, &work[nu], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}

