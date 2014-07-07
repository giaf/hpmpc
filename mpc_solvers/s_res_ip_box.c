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


#include "../include/blas_s.h"
#include "../include/block_size.h"



void s_res_ip_box_mpc(int nx, int nu, int N, int nb, float **hpBAbt, float **hpQ, float **hq, float **hux, float **hdb, float **hpi, float **hlam, float **ht, float **hrq, float **hrb, float **hrd, float *mu)
	{

	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;
	
	int nbu = nu<nb ? nu : nb ;
	
	// first block
	mu[0] = 0;
	for(jj=0 ; jj<2*nbu; jj+=2) mu[0] += hlam[0][jj+0] * ht[0][jj+0] + hlam[0][jj+1] * ht[0][jj+1];
	for(jj=0; jj<2*nbu; jj+=2) { hrd[0][jj+0] = hux[0][jj/2] - hdb[0][jj+0] - ht[0][jj+0]; hrd[0][jj+1] = - hdb[0][jj+1] - hux[0][jj/2] - ht[0][jj+1]; }
	for(jj=0; jj<nu; jj++) hrq[0][jj] = - hq[0][jj] + hlam[0][2*jj+0] - hlam[0][2*jj+1];
	sgemv_t_lib(nx, nu, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%bs, cnz, hux[0]+nu, hrq[0], -1);
	ssymv_lib(nu, 0, hpQ[0], cnz, hux[0], hrq[0], -1);
	sgemv_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	sgemv_t_lib(nxu, nx, 0, hpBAbt[0], cnx, hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0 ; jj<2*nb; jj+=2) mu[0] += hlam[ii][jj+0] * ht[ii][jj+0] + hlam[ii][jj+1] * ht[ii][jj+1];
		for(jj=0; jj<2*nb; jj+=2) {	hrd[ii][jj+0] = hux[ii][jj/2] - hdb[ii][jj+0] - ht[ii][jj+0]; hrd[ii][jj+1] = - hdb[ii][jj+1] - hux[ii][jj/2] - ht[ii][jj+1]; }
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj] + hlam[ii][2*jj+0] - hlam[ii][2*jj+1];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj] + hlam[ii][2*nu+2*jj+0] - hlam[ii][2*nu+2*jj+1];
		ssymv_lib(nxu, 0, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		smvmv_lib(nxu, nx, 0, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}
	
/*exit(3);*/

	// last block
	for(jj=2*nu ; jj<2*nb; jj+=2) mu[0] += hlam[N][jj+0] * ht[N][jj+0] + hlam[N][jj+1] * ht[N][jj+1];
	mu[0] /= N*2*nb; // + 2*nbx;
	for(jj=2*nu; jj<2*nb; jj+=2) { hrd[N][jj+0] = hux[N][jj/2] - hdb[N][jj+0] - ht[N][jj+0]; hrd[N][jj+1] = - hdb[N][jj+1] - hux[N][jj/2] - ht[N][jj+1]; }
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][2*nu+2*jj+0] - hlam[N][2*nu+2*jj+1];
	ssymv_lib(nx, nu, hpQ[N]+(nu/bs)*bs*cnz+nu%bs+nu*bs, cnz, hux[N]+nu, hrq[N]+nu, -1);
	
	}



void s_res_ip_box_mhe(int nx, int nu, int N, int nb, float **hpBAbt, float **hpQ, float **hq, float **hux, float **hdb, float **hpi, float **hlam, float **ht, float **hrq, float **hrb, float **hrd, float *mu)
	{

	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;
	
	int nbu = nu<nb ? nu : nb ;
	
	// first block
	mu[0] = 0;
	for(jj=0 ; jj<2*nb; jj+=2) mu[0] += hlam[0][jj+0] * ht[0][jj+0] + hlam[0][jj+1] * ht[0][jj+1];
	for(jj=0; jj<2*nb; jj+=2) {	hrd[0][jj+0] = hux[0][jj/2] - hdb[0][jj+0] - ht[0][jj+0]; hrd[0][jj+1] = - hdb[0][jj+1] - hux[0][jj/2] - ht[0][jj+1]; }
	for(jj=0; jj<nu+nx; jj++) hrq[0][jj] = - hq[0][jj] + hlam[0][2*jj+0] - hlam[0][2*jj+1];
	ssymv_lib(nxu, 0, hpQ[0], cnz, hux[0], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	smvmv_lib(nxu, nx, 0, hpBAbt[0], cnx, hpi[1], hrq[0], hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0 ; jj<2*nb; jj+=2) mu[0] += hlam[ii][jj+0] * ht[ii][jj+0] + hlam[ii][jj+1] * ht[ii][jj+1];
		for(jj=0; jj<2*nb; jj+=2) {	hrd[ii][jj+0] = hux[ii][jj/2] - hdb[ii][jj+0] - ht[ii][jj+0]; hrd[ii][jj+1] = - hdb[ii][jj+1] - hux[ii][jj/2] - ht[ii][jj+1]; }
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj] + hlam[ii][2*jj+0] - hlam[ii][2*jj+1];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj] + hlam[ii][2*nu+2*jj+0] - hlam[ii][2*nu+2*jj+1];
		ssymv_lib(nxu, 0, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		smvmv_lib(nxu, nx, 0, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}
	
/*exit(3);*/

	// last block
	for(jj=2*nu ; jj<2*nb; jj+=2) mu[0] += hlam[N][jj+0] * ht[N][jj+0] + hlam[N][jj+1] * ht[N][jj+1];
	mu[0] /= N*2*nb; // + 2*nbx;
	for(jj=2*nu; jj<2*nb; jj+=2) { hrd[N][jj+0] = hux[N][jj/2] - hdb[N][jj+0] - ht[N][jj+0]; hrd[N][jj+1] = - hdb[N][jj+1] - hux[N][jj/2] - ht[N][jj+1]; }
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][2*nu+2*jj+0] - hlam[N][2*nu+2*jj+1];
	ssymv_lib(nx, nu, hpQ[N]+(nu/bs)*bs*cnz+nu%bs+nu*bs, cnz, hux[N]+nu, hrq[N]+nu, -1);
	
	}

