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

#include <stdlib.h>

#include "../include/blas_s.h"
#include "../include/aux_s.h"
#include "../include/block_size.h"



/* version tailoerd for mpc (x0 fixed) */
void sricposv_mpc(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL, int compute_pi, float **hpi, int *info)
	{
	
	const int bs = S_MR; //d_get_mr();

	int ii, jj;
	
	int nz = nx+nu+1;

	/* factorization and backward substitution */

	/* final stage */
	int nu4 = (nu/bs)*bs;
	spotrf_p_lib(nz-nu4, nu%bs, hpQ[N]+nu4*(sda+bs), sda, info);
	if(*info!=0) return;
	s_transpose_pmat_lo(nx, nu, hpQ[N]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pL, sda);

	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{	
		strmm_ppp_lib(nz, nx, nu, hpBAbt[N-ii-1], sda, pL, sda, pBAbtL, sda);
		for(jj=0; jj<nx; jj++) pBAbtL[((nx+nu)/bs)*bs*sda+(nx+nu)%bs+jj*bs] += hpQ[N-ii][((nx+nu)/bs)*bs*sda+(nx+nu)%bs+(nu+jj)*bs];
		ssyrk_ppp_lib(nz, nz, nx, pBAbtL, sda, hpQ[N-ii-1], sda);
		spotrf_p_lib(nz, nu, hpQ[N-ii-1], sda, info);
		if(*info!=0) return;
		s_transpose_pmat_lo(nx, nu, hpQ[N-ii-1]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pL, sda);
		}

	/* initial stage */
	strmm_ppp_lib(nz, nx, nu, hpBAbt[0], sda, pL, sda, pBAbtL, sda);
	for(jj=0; jj<nx; jj++) pBAbtL[((nx+nu)/bs)*bs*sda+(nx+nu)%bs+jj*bs] += hpQ[1][((nx+nu)/bs)*bs*sda+(nx+nu)%bs+(nu+jj)*bs];
	ssyrk_ppp_lib(nz, nu, nx, pBAbtL, sda, hpQ[0], sda);
	spotrf_rec_p_lib(nz, nu, hpQ[0], sda, info);
	if(*info!=0) return;

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], -1);
		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) pBAbtL[nu+jj] = hpQ[ii+1][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*(nu+jj)];
			strmv_p_t_lib(nx, nu, hpQ[ii+1]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, &hux[ii+1][nu], &pBAbtL[nu], 1); // L'*pi
			strmv_p_n_lib(nx, nu, hpQ[ii+1]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, &pBAbtL[nu], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
	}



void sricpotrs_mpc(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hq, float **hux, float *pBAbtL, int compute_pi, float **hpi)
	{
	
	const int bs = S_MR; //d_get_mr();

	int ii, jj;
	
	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) pBAbtL[nu+jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj]; // copy b
		strmv_p_t_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+nu, pBAbtL+sda+nu, 0); // L'*b
		for(jj=0; jj<nx; jj++) pBAbtL[nu+jj] = hq[N-ii][nu+jj]; // copy p
		strmv_p_n_lib(nx, nu, hpQ[N-ii]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, pBAbtL+sda+nu, pBAbtL+nu, 1); // L*(L'*b) + p
		sgemv_p_n_lib(nx+nu, nx, 0, hpBAbt[N-ii-1], sda, pBAbtL+nu, hq[N-ii-1], 1);
		strsv_p_n_lib(nu, hpQ[N-ii-1], sda, hq[N-ii-1]);
		sgemv_p_n_lib(nx, nu, nu, hpQ[N-ii-1]+(nu/bs)*bs*sda+nu%bs, sda, hq[N-ii-1], hq[N-ii-1]+nu, -1);
		}


	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], -1);
		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];
		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			strmv_p_t_lib(nx, nu, hpQ[ii+1]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, &hux[ii+1][nu], &pBAbtL[nu], 0); // L'*pi
			strmv_p_n_lib(nx, nu, hpQ[ii+1]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, &pBAbtL[nu], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}
	
	}



/*void sricposv_orig(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL)*/
/*	{*/
/*	*/
/*	const int bs = S_MR; //d_get_mr();*/

/*	int ii, jj;*/
/*	*/
/*	int nz = nx+nu+1;*/

/*	spotrf_p_scopy_p_t_lib(nz, nu, hpQ[N], sda, pL, sda);*/


/*	for(ii=0; ii<N; ii++)*/
/*		{	*/
/*		strmm_ppp_lib(nz, nx, nu, hpBAbt[N-ii-1], sda, pL, sda, pBAbtL, sda);*/
/*		ssyrk_ppp_lib(nz, nz, nx, pBAbtL, sda, hpQ[N-ii-1], sda);*/
/*		spotrf_p_scopy_p_t_lib(nz, nu, hpQ[N-ii-1], sda, pL, sda);*/
/*		}*/


/*	for(ii=0; ii<N; ii++)*/
/*		{*/
/*		for(jj=0; jj<nu; jj++) hux[ii][jj] = hpQ[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];*/
/*		sgemv_p_t_lib(nx, nu, nu, &hpQ[ii][(nu/bs)*bs*sda+nu%bs], sda, &hux[ii][nu], &hux[ii][0], 1);*/
/*		strsv_p_t_lib(nu, hpQ[ii], sda, &hux[ii][0]);*/
/*		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hux[ii][jj];*/
/*		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*sda+(nu+nx)%bs+bs*jj];*/
/*		sgemv_p_t_lib(nx+nu, nx, 0, hpBAbt[ii], sda, &hux[ii][0], &hux[ii+1][nu], 1);*/
/*		}*/
/*	}*/




