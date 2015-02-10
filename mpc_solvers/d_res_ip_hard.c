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


#include "../include/blas_d.h"
#include "../include/block_size.h"



void d_res_ip_hard_mpc(int nx, int nu, int N, int nb, int ng, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hdb, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	//const int anb = nal*((nb+nal-1)/nal); // cache aligned number of box constraints
	const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box constraints

	static double temp[D_MR] = {};

	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;
	
	int nbu = nu<nb ? nu : nb ;
	


	// first block
	mu[0] = 0;
	for(jj=0 ; jj<nbu; jj++) 
		mu[0] += hlam[0][jj] * ht[0][jj] + hlam[0][pnb+jj] * ht[0][pnb+jj];
	for(jj=0; jj<nbu; jj++) 
		{ 
		hrd[0][jj]     =   hux[0][jj] - hdb[0][jj]     - ht[0][jj]; 
		hrd[0][pnb+jj] = - hux[0][jj] - hdb[0][pnb+jj] - ht[0][pnb+jj]; 
		}
	for(jj=0; jj<nbu; jj++) 
		hrq[0][jj] = - hq[0][jj] + hlam[0][jj] - hlam[0][pnb+jj];
	for(; jj<nu; jj++) 
		hrq[0][jj] = - hq[0][jj];
	for(jj=0; jj<nu%bs; jj++) 
		{ 
		temp[jj] = hux[0][(nu/bs)*bs+jj]; 
		hux[0][(nu/bs)*bs+jj] = 0.0; 
		}
	dgemv_t_lib(nx, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%bs, cnz, hux[0]+nu, hrq[0], -1);
	for(jj=0; jj<nu%bs; jj++) 
		hux[0][(nu/bs)*bs+jj] = temp[jj];
	dsymv_lib(nu, nu, hpQ[0], cnz, hux[0], hrq[0], -1);
	dgemv_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], -1);
	for(jj=0; jj<nx; jj++) 
		hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dgemv_t_lib(nxu, nx, hpBAbt[0], cnx, hux[0], hrb[0], -1);



	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0 ; jj<nb; jj++) 
			mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];
		for(jj=0; jj<nb; jj++) 
			{	
			hrd[ii][jj]     =   hux[ii][jj] - hdb[ii][jj]     - ht[ii][jj]; 
			hrd[ii][pnb+jj] = - hux[ii][jj] - hdb[ii][pnb+jj] - ht[ii][pnb+jj]; 
			}
		for(jj=0; jj<nbu; jj++) 
			hrq[ii][jj] = - hq[ii][jj] + hlam[ii][jj] - hlam[ii][pnb+jj];
		for(; jj<nu; jj++) 
			hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nbx; jj++) 
			hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj] + hlam[ii][nu+jj] - hlam[ii][pnb+nu+jj];
		for(; jj<nx; jj++) 
			hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj];
		dsymv_lib(nxu, nxu, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) 
			hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		dmvmv_lib(nxu, nx, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}
	


	// last block
	for(jj=nu ; jj<nb; jj++) 
		mu[0] += hlam[N][jj] * ht[N][jj] + hlam[N][pnb+jj] * ht[N][pnb+jj];
	mu[0] /= N*2*nb; // + 2*nbx;
	for(jj=nu; jj<nb; jj++) 
		{ 
		hrd[N][jj]     =   hux[N][jj] - hdb[N][jj]     - ht[N][jj]; 
		hrd[N][pnb+jj] = - hux[N][jj] - hdb[N][pnb+jj] - ht[N][pnb+jj]; 
		}
	for(jj=0; jj<nbx; jj++) 
		hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][nu+jj] - hlam[N][pnb+nu+jj];
	for(; jj<nx; jj++) 
		hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs, -1);
	
	}



#if 0
void d_res_ip_box_mhe_old(int nx, int nu, int N, int nb, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hdb, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

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
	dsymv_lib(nxu, nxu, hpQ[0], cnz, hux[0], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dmvmv_lib(nxu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0 ; jj<2*nb; jj+=2) mu[0] += hlam[ii][jj+0] * ht[ii][jj+0] + hlam[ii][jj+1] * ht[ii][jj+1];
		for(jj=0; jj<2*nb; jj+=2) {	hrd[ii][jj+0] = hux[ii][jj/2] - hdb[ii][jj+0] - ht[ii][jj+0]; hrd[ii][jj+1] = - hdb[ii][jj+1] - hux[ii][jj/2] - ht[ii][jj+1]; }
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj] + hlam[ii][2*jj+0] - hlam[ii][2*jj+1];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj] + hlam[ii][2*nu+2*jj+0] - hlam[ii][2*nu+2*jj+1];
		dsymv_lib(nxu, nxu, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		dmvmv_lib(nxu, nx, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}
	
/*exit(3);*/

	// last block
	for(jj=2*nu ; jj<2*nb; jj+=2) mu[0] += hlam[N][jj+0] * ht[N][jj+0] + hlam[N][jj+1] * ht[N][jj+1];
	mu[0] /= N*2*nb; // + 2*nbx;
	for(jj=2*nu; jj<2*nb; jj+=2) { hrd[N][jj+0] = hux[N][jj/2] - hdb[N][jj+0] - ht[N][jj+0]; hrd[N][jj+1] = - hdb[N][jj+1] - hux[N][jj/2] - ht[N][jj+1]; }
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][2*nu+2*jj+0] - hlam[N][2*nu+2*jj+1];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs, -1);
	
	}
#endif
