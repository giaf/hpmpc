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


#include "../include/blas_d.h"
#include "../include/block_size.h"



void dres(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)
	{

	const int bs = D_MR; //d_get_mr();

	const int sda = bs*((nx+nu+1+bs-1)/bs); // TODO remove

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;

	// first block
	for(jj=0; jj<nu; jj++) hrq[0][jj] = - hq[0][jj];
	dgemv_p_t_lib(nx, nu, nu, hpQ[0]+(nu/bs)*bs*sda+nu%bs, sda, hux[0]+nu, hrq[0], -1);
	dsymv_p_lib(nu, 0, hpQ[0], sda, hux[0], hrq[0], -1);
	dgemv_p_n_lib(nu, nx, 0, hpBAbt[0], sda, hpi[1], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*sda+nxu%bs+bs*jj];
	dgemv_p_t_lib(nxu, nx, 0, hpBAbt[0], sda, hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj];
		dsymv_p_lib(nxu, 0, hpQ[ii], sda, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*sda+nxu%bs+bs*jj];
/*		dgemv_p_n_lib(nxu, nx, 0, hpBAbt[ii], sda, hpi[ii+1], hrq[ii], -1);*/
/*		dgemv_p_t_lib(nxu, nx, 0, hpBAbt[ii], sda, hux[ii], hrb[ii], -1);*/
		dmvmv_p_lib(nxu, nx, 0, hpBAbt[ii], sda, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}
	
/*exit(3);*/

	// last block
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj];
	dsymv_p_lib(nx, nu, hpQ[N]+(nu/bs)*bs*sda+nu%bs+nu*bs, sda, hux[N]+nu, hrq[N]+nu, -1);
	
	}

