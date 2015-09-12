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



void d_res_ip_soft_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **hpBAbt, double **hpQ, double **hq, double **hZ, double **hz, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double **hrz, double *mu)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	static double temp[D_MR] = {};

	int ii, jj;
	
	int nu0, nu1, cnz0, nx0, nx1, nxm, cnx0, cnx1, nb0, pnb, ng0, png, cng, ns0, pns, nb_tot;


	// initialize mu
	nb_tot = 0;
	mu[0] = 0;



#if 0
	// first block
	ii = 0;
	nu0 = nu[ii];
	nu1 = nu[ii+1];
	nx0 = nx[ii]; // nx1;
	nx1 = nx[ii+1];
	cnx1  = (nx1+ncl-1)/ncl*ncl;
	cnz0  = (nu0+nx0+1+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = (ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;
	nb_tot += nb0 + ng0;

	for(jj=0; jj<nb0; jj++)
		mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];
	for(jj=0; jj<ng0; jj++) 
		mu[0] += hlam[ii][2*pnb+jj] * ht[ii][2*pnb+jj] + hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];

	
	for(jj=0; jj<nb0; jj++)
		{
		hrd[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
		hrd[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
		}
	if(ng0>0)
		{
		dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], 0, hrd[ii]+2*pnb, hrd[ii]+2*pnb);
		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
			hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
			}
		}

	for(jj=0; jj<nu0; jj++) 
		hrq[ii][jj] = - hq[ii][jj];
	for(jj=0; jj<nb0; jj++) 
		hrq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
	for(jj=0; jj<nu0%bs; jj++) 
		{ 
		temp[jj] = hux[ii][nu0/bs*bs+jj]; 
		hux[ii][nu0/bs*bs+jj] = 0.0; 
		}
	dgemv_t_lib(nx0+nu0%bs, nu0, hpQ[ii]+nu0/bs*bs*cnz0, cnz0, hux[ii]+nu0/bs*bs, -1, hrq[ii], hrq[ii]);
	for(jj=0; jj<nu0%bs; jj++) 
		hux[ii][nu0/bs*bs+jj] = temp[jj];
	dsymv_lib(nu0, nu0, hpQ[ii], cnz0, hux[ii], -1, hrq[ii], hrq[ii]);
	dgemv_n_lib(nu0, nx1, hpBAbt[ii], cnx1, hpi[ii+1], -1, hrq[ii], hrq[ii]);
	if(ng0>0)
		{
		// TODO work space + one dgemv call
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, 1, hrq[ii], hrq[ii]);
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, -1, hrq[ii], hrq[ii]);
		}
	
	for(jj=0; jj<nx1; jj++) 
		hrb[ii][jj] = hux[ii+1][nu1+jj] - hpBAbt[ii][(nu0+nx0)/bs*bs*cnx1+(nu0+nx0)%bs+bs*jj];
	dgemv_t_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hux[ii], -1, hrb[ii], hrb[ii]);
#endif



	nu1 = nu[0];
	nx1 = nx[0];
	cnx1  = (nx1+ncl-1)/ncl*ncl;
	// first blocks
	for(ii=0; ii<N; ii++)
		{
		nu0 = nu1;
		nu1 = nu[ii+1];
		nx0 = nx1;
		nx1 = nx[ii+1];
		cnx0 = cnx1;
		cnx1  = (nx1+ncl-1)/ncl*ncl;
		cnz0  = (nu0+nx0+1+ncl-1)/ncl*ncl;
		nb0 = nb[ii];
		pnb = (nb0+bs-1)/bs*bs;
		ng0 = ng[ii];
		png = (ng0+bs-1)/bs*bs;
		cng = (ng0+ncl-1)/ncl*ncl;
		ns0 = ns[ii];
		pns = (ns0+bs-1)/bs*bs;
		nb_tot += nb0 + ng0 + ns0;

		for(jj=0; jj<nb0; jj++)
			mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];
		for(jj=0; jj<ng0; jj++) 
			mu[0] += hlam[ii][2*pnb+jj] * ht[ii][2*pnb+jj] + hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
		for(jj=0; jj<ns0; jj++) 
			mu[0] += hlam[ii][2*pnb+2*png+0*pns+jj] * ht[ii][2*pnb+2*png+0*pns+jj] + hlam[ii][2*pnb+2*png+1*pns+jj] * ht[ii][2*pnb+2*png+1*pns+jj] + hlam[ii][2*pnb+2*png+2*pns+jj] * ht[ii][2*pnb+2*png+2*pns+jj] + hlam[ii][2*pnb+2*png+3*pns+jj] * ht[ii][2*pnb+2*png+3*pns+jj];

		for(jj=0; jj<nb0; jj++)
			{
			hrd[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
			hrd[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
			}
		if(ng0>0)
			{
			dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], 0, hrd[ii]+2*pnb, hrd[ii]+2*pnb);
			for(jj=0; jj<ng0; jj++)
				{
				hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
				hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
				hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
				}
			}
		for(jj=0; jj<ns0; jj++)
			{
			hrd[ii][2*pnb+2*png+0*pns+jj] = ht[ii][2*pnb+2*png+2*pns+jj] + hux[ii][idxb[ii][nu0+jj]] - hd[ii][2*pnb+2*png+0*pns+jj] - ht[ii][2*pnb+2*png+0*pns+jj];
			hrd[ii][2*pnb+2*png+1*pns+jj] = ht[ii][2*pnb+2*png+3*pns+jj] - hux[ii][idxb[ii][nu0+jj]] - hd[ii][2*pnb+2*png+1*pns+jj] - ht[ii][2*pnb+2*png+1*pns+jj];
			}

		for(jj=0; jj<nu0; jj++) 
			hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nx0; jj++) 
			hrq[ii][nu0+jj] = - hq[ii][nu0+jj] + hpi[ii][jj];
		dsymv_lib(nu0+nx0, nu0+nx0, hpQ[ii], cnz0, hux[ii], -1, hrq[ii], hrq[ii]);
		for(jj=0; jj<nb0; jj++) 
			hrq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
		if(ng0>0)
			{
			// TODO work space + one dgemv call
			dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, 1, hrq[ii], hrq[ii]);
			dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, -1, hrq[ii], hrq[ii]);
			}
		for(jj=0; jj<ns0; jj++) 
			hrq[ii][idxb[ii][nu0+jj]] += hlam[ii][2*pnb+2*png+0*pns+jj] - hlam[ii][2*pnb+2*png+1*pns+jj];
		for(jj=0; jj<nx1; jj++) 
			hrb[ii][jj] = hux[ii+1][nu1+jj] - hpBAbt[ii][(nu0+nx0)/bs*bs*cnx1+(nu0+nx0)%bs+bs*jj];
		dgemv_nt_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hpi[ii+1], hux[ii], -1, hrq[ii], hrb[ii], hrq[ii], hrb[ii]);

		for(jj=0; jj<ns0; jj++) 
			{ 
			hrz[ii][0*pns+jj] = hz[ii][0*pns+jj] + hZ[ii][0*pns+jj]*ht[ii][2*pnb+2*png+2*pns+jj] - hlam[ii][2*pnb+2*png+0*pns+jj] - hlam[ii][2*pnb+2*png+2*pns+jj]; 
			hrz[ii][1*pns+jj] = hz[ii][1*pns+jj] + hZ[ii][1*pns+jj]*ht[ii][2*pnb+2*png+3*pns+jj] - hlam[ii][2*pnb+2*png+1*pns+jj] - hlam[ii][2*pnb+2*png+3*pns+jj]; 
			}

		}
	

	// last block
	ii = N;
	nu0 = nu1;
	nx0 = nx1;
	cnz0  = (nu0+nx0+1+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = (ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;
	ns0 = ns[ii];
	pns = (ns0+bs-1)/bs*bs;
	nb_tot += nb0 + ng0 + ns0;

	for(jj=0; jj<nb0; jj++)
		mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];
	for(jj=0; jj<ng0; jj++) 
		mu[0] += hlam[ii][2*pnb+jj] * ht[ii][2*pnb+jj] + hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
	for(jj=0; jj<ns0; jj++) 
		mu[0] += hlam[ii][2*pnb+2*png+0*pns+jj] * ht[ii][2*pnb+2*png+0*pns+jj] + hlam[ii][2*pnb+2*png+1*pns+jj] * ht[ii][2*pnb+2*png+1*pns+jj] + hlam[ii][2*pnb+2*png+2*pns+jj] * ht[ii][2*pnb+2*png+2*pns+jj] + hlam[ii][2*pnb+2*png+3*pns+jj] * ht[ii][2*pnb+2*png+3*pns+jj];

	for(jj=0; jj<nb0; jj++)
		{
		hrd[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
		hrd[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
		}
	if(ng0>0)
		{
		dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], 0, hrd[ii]+2*pnb, hrd[ii]+2*pnb);
		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
			hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
			}
		}
	for(jj=0; jj<ns0; jj++)
		{
		hrd[ii][2*pnb+2*png+0*pns+jj] = ht[ii][2*pnb+2*png+2*pns+jj] + hux[ii][idxb[ii][nu0+jj]] - hd[ii][2*pnb+2*png+0*pns+jj] - ht[ii][2*pnb+2*png+0*pns+jj];
		hrd[ii][2*pnb+2*png+1*pns+jj] = ht[ii][2*pnb+2*png+3*pns+jj] - hux[ii][idxb[ii][nu0+jj]] - hd[ii][2*pnb+2*png+1*pns+jj] - ht[ii][2*pnb+2*png+1*pns+jj];
		}


	for(jj=0; jj<nx0; jj++) 
		hrq[ii][nu0+jj] = hpi[ii][jj] - hq[ii][nu0+jj];
	for(jj=0; jj<nb0; jj++) 
		hrq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
	dsymv_lib(nx0+nu0%bs, nx0+nu0%bs, hpQ[ii]+nu0/bs*bs*cnz0+nu0/bs*bs*bs, cnz0, hux[ii]+nu0/bs*bs, -1, hrq[ii]+nu0/bs*bs, hrq[ii]+nu0/bs*bs);
	if(ng0>0)
		{
		// TODO work space + one dgemv call
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, 1, hrq[ii], hrq[ii]);
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, -1, hrq[ii], hrq[ii]);
		}
	for(jj=0; jj<ns0; jj++) 
		hrq[ii][idxb[ii][nu0+jj]] += - hlam[ii][2*pnb+2*png+2*pns+jj] + hlam[ii][2*pnb+2*png+3*pns+jj];
	
	for(jj=0; jj<ns0; jj++) 
		{ 
		hrz[ii][0*pns+jj] = hz[ii][0*pns+jj] + hZ[ii][0*pns+jj]*ht[ii][2*pnb+2*png+2*pns+jj] - hlam[ii][2*pnb+2*png+0*pns+jj] - hlam[ii][2*pnb+2*png+2*pns+jj]; 
		hrz[ii][1*pns+jj] = hz[ii][1*pns+jj] + hZ[ii][1*pns+jj]*ht[ii][2*pnb+2*png+3*pns+jj] - hlam[ii][2*pnb+2*png+1*pns+jj] - hlam[ii][2*pnb+2*png+3*pns+jj]; 
		}



	// normalize mu
	mu[0] /= 2.0*nb_tot;

	}



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


