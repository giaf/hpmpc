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



void d_res_ip_soft_mpc(int nx, int nu, int N, int nh, int ns, double **hpBAbt, double **hpQ, double **hq, double **hZ, double **hz, double **hux, double **hdb, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double **hrz, double *mu)
	{

	// number of either hard or soft (box) constraints
	int nb = nh + ns;

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int pnb = bs*((2*nb+bs-1)/bs);

	static double temp[D_MR] = {};

	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;
	
	int nbu = nu<nb ? nu : nb ;
	
	// first block : hard input constraints
	mu[0] = 0;
	for(jj=0 ; jj<2*nbu; jj+=2) mu[0] += hlam[0][jj+0] * ht[0][jj+0] + hlam[0][jj+1] * ht[0][jj+1];
	for(jj=0; jj<2*nbu; jj+=2) 
		{ 
		hrd[0][jj+0] = ht[0][pnb+jj+0] + hux[0][jj/2] - hdb[0][jj+0] - ht[0][jj+0]; 
		hrd[0][jj+1] = ht[0][pnb+jj+1] - hdb[0][jj+1] - hux[0][jj/2] - ht[0][jj+1]; 
		}
	for(jj=0; jj<nu; jj++) hrq[0][jj] = - hq[0][jj] + hlam[0][2*jj+0] - hlam[0][2*jj+1];
	for(jj=0; jj<nu%bs; jj++) { temp[jj] = hux[0][(nu/bs)*bs+jj]; hux[0][(nu/bs)*bs+jj] = 0.0; }
	dgemv_t_lib(nx, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%bs, cnz, hux[0]+nu, -1, hrq[0], hrq[0]);
	for(jj=0; jj<nu%bs; jj++) hux[0][(nu/bs)*bs+jj] = temp[jj];
	dsymv_lib(nu, nu, hpQ[0], cnz, hux[0], -1, hrq[0], hrq[0]);
	dgemv_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], -1, hrq[0], hrq[0]);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dgemv_t_lib(nxu, nx, hpBAbt[0], cnx, hux[0], -1, hrb[0], hrb[0]);

	// middle blocks : hard input constraints and hard & soft state constraints
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<2*nb; jj+=2) mu[0] += hlam[ii][jj+0] * ht[ii][jj+0] + hlam[ii][jj+1] * ht[ii][jj+1];
		for(jj=2*nh; jj<2*nb; jj+=2) mu[0] += hlam[ii][pnb+jj+0] * ht[ii][pnb+jj+0] + hlam[ii][pnb+jj+1] * ht[ii][pnb+jj+1];
		for(jj=0; jj<2*nh; jj+=2) 
			{	
			hrd[ii][jj+0] =   hux[ii][jj/2] - hdb[ii][jj+0] - ht[ii][jj+0]; 
			hrd[ii][jj+1] = - hux[ii][jj/2] - hdb[ii][jj+1] - ht[ii][jj+1]; 
			}
		for(; jj<2*nb; jj+=2) 
			{	
			hrd[ii][jj+0] = ht[ii][pnb+jj+0] + hux[ii][jj/2] - hdb[ii][jj+0] - ht[ii][jj+0]; 
			hrd[ii][jj+1] = ht[ii][pnb+jj+1] - hdb[ii][jj+1] - hux[ii][jj/2] - ht[ii][jj+1]; 
			}
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj] + hlam[ii][2*jj+0] - hlam[ii][2*jj+1];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj] + hlam[ii][2*nu+2*jj+0] - hlam[ii][2*nu+2*jj+1];
		dsymv_lib(nxu, nxu, hpQ[ii], cnz, hux[ii], -1, hrq[ii], hrq[ii]);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		dgemv_nt_lib(nxu, nx, hpBAbt[ii], cnx, hpi[ii+1], hux[ii], -1, hrq[ii], hrb[ii], hrq[ii], hrb[ii]);
		for(jj=2*nh; jj<2*nb; jj+=2) 
			{ 
			hrz[ii][jj+0] = hz[ii][jj+0] + hZ[ii][jj+0]*ht[ii][pnb+jj+0] - hlam[ii][jj+0] - hlam[ii][pnb+jj+0]; 
			hrz[ii][jj+1] = hz[ii][jj+1] + hZ[ii][jj+1]*ht[ii][pnb+jj+1] - hlam[ii][jj+1] - hlam[ii][pnb+jj+1];
			}
		}
	
/*exit(3);*/

	// last block
	for(jj=2*nu ; jj<2*nb; jj+=2) mu[0] += hlam[N][jj+0] * ht[N][jj+0] + hlam[N][jj+1] * ht[N][jj+1];
	for(jj=2*nh ; jj<2*nb; jj+=2) mu[0] += hlam[N][pnb+jj+0] * ht[N][pnb+jj+0] + hlam[N][pnb+jj+1] * ht[N][pnb+jj+1];
	mu[0] /= N*(2*nb + 2*nh);
	for(jj=2*nu; jj<2*nh; jj+=2) 
		{	
		hrd[N][jj+0] =   hux[N][jj/2] - hdb[N][jj+0] - ht[N][jj+0]; 
		hrd[N][jj+1] = - hux[N][jj/2] - hdb[N][jj+1] - ht[N][jj+1]; 
		}
	for(; jj<2*nb; jj+=2) 
		{ 
		hrd[N][jj+0] = ht[N][pnb+jj+0] + hux[N][jj/2] - hdb[N][jj+0] - ht[N][jj+0]; 
		hrd[N][jj+1] = ht[N][pnb+jj+1] - hdb[N][jj+1] - hux[N][jj/2] - ht[N][jj+1]; 
		}
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][2*nu+2*jj+0] - hlam[N][2*nu+2*jj+1];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, -1, hrq[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs);
	for(jj=2*nh; jj<2*nbx; jj+=2) 
		{ 
		hrz[N][jj+0] = hz[N][jj+0] + hZ[N][jj+0]*ht[N][pnb+jj+0] - hlam[N][jj+0] - hlam[N][pnb+jj+0]; 
		hrz[N][jj+1] = hz[N][jj+1] + hZ[N][jj+1]*ht[N][pnb+jj+1] - hlam[N][jj+1] - hlam[N][pnb+jj+1];
		}
	
	}


