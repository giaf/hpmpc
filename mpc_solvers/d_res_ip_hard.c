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



void d_res_ip_hard_mpc_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	static double temp[D_MR] = {};

	int ii, jj;
	
	int nu0, nu1, cnz0, nx0, nx1, nxm, cnx0, cnx1, nb0, pnb, ng0, png, cng, nb_tot;


	// initialize mu
	nb_tot = 0;
	mu[0] = 0;



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
		dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], hrd[ii]+2*pnb, hrd[ii]+2*pnb, 0);
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
	dgemv_t_lib(nx0+nu0%bs, nu0, hpQ[ii]+nu0/bs*bs*cnz0, cnz0, hux[ii]+nu0/bs*bs, hrq[ii], hrq[ii], -1);
	for(jj=0; jj<nu0%bs; jj++) 
		hux[ii][nu0/bs*bs+jj] = temp[jj];
	dsymv_lib(nu0, nu0, hpQ[ii], cnz0, hux[ii], hrq[ii], -1);
	dgemv_n_lib(nu0, nx1, hpBAbt[ii], cnx1, hpi[ii+1], hrq[ii], hrq[ii], -1);
	if(ng0>0)
		{
		// TODO work space + one dgemv call
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, hrq[ii], hrq[ii], 1);
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, hrq[ii], hrq[ii], -1);
		}
	
	for(jj=0; jj<nx1; jj++) 
		hrb[ii][jj] = hux[ii+1][nu1+jj] - hpBAbt[ii][(nu0+nx0)/bs*bs*cnx1+(nu0+nx0)%bs+bs*jj];
	dgemv_t_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hux[ii], hrb[ii], hrb[ii], -1);



	// middle blocks
	for(ii=1; ii<N; ii++)
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
			dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], hrd[ii]+2*pnb, hrd[ii]+2*pnb, 0);
			for(jj=0; jj<ng0; jj++)
				{
				hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
				hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
				hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
				}
			}

		for(jj=0; jj<nu0; jj++) 
			hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nx0; jj++) 
			hrq[ii][nu0+jj] = - hq[ii][nu0+jj] + hpi[ii][jj];
		for(jj=0; jj<nb0; jj++) 
			hrq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
		dsymv_lib(nu0+nx0, nu0+nx0, hpQ[ii], cnz0, hux[ii], hrq[ii], -1);
		if(ng0>0)
			{
			// TODO work space + one dgemv call
			dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, hrq[ii], hrq[ii], 1);
			dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, hrq[ii], hrq[ii], -1);
			}

		for(jj=0; jj<nx1; jj++) 
			hrb[ii][jj] = hux[ii+1][nu1+jj] - hpBAbt[ii][(nu0+nx0)/bs*bs*cnx1+(nu0+nx0)%bs+bs*jj];
		dmvmv_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);

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
		dgemv_t_lib(nu0+nx0, ng0, hpDCt[ii], cng, hux[ii], hrd[ii]+2*pnb, hrd[ii]+2*pnb, 0);
		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
			hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
			}
		}

	for(jj=0; jj<nx0; jj++) 
		hrq[ii][nu0+jj] = hpi[ii][jj] - hq[ii][nu0+jj];
	for(jj=0; jj<nb0; jj++) 
		hrq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
	dsymv_lib(nx0+nu0%bs, nx0+nu0%bs, hpQ[ii]+nu0/bs*bs*cnz0+nu0/bs*bs*bs, cnz0, hux[ii]+nu0/bs*bs, hrq[ii]+nu0/bs*bs, -1);
	if(ng0>0)
		{
		// TODO work space + one dgemv call
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb, hrq[ii], hrq[ii], 1);
		dgemv_n_lib(nu0+nx0, ng0, hpDCt[ii], cng, hlam[ii]+2*pnb+png, hrq[ii], hrq[ii], -1);
		}
	


	// normalize mu
	mu[0] /= 2.0*nb_tot;

	}



void d_res_ip_hard_mpc(int nx, int nu, int N, int nb, int ng, int ngN, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double **hrq, double **hrb, double **hrd, double *mu)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int pnz  = bs*((nx+nu+1+bs-1)/bs);
	const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	//const int anb = nal*((nb+nal-1)/nal); // cache aligned number of box constraints
	const int pnb  = bs*((nb+bs-1)/bs); // cache aligned number of box constraints
	const int png  = bs*((ng+bs-1)/bs); // cache aligned number of box constraints
	const int pngN = bs*((ngN+bs-1)/bs); // cache aligned number of box constraints

	static double temp[D_MR] = {};

	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	int ii, jj;
	
/*	int nz = nx+nu+1;*/
	int nxu = nx+nu;
	
	int nbu = nu<nb ? nu : nb ;
	


	// first block
	ii = 0;

	mu[0] = 0;
	for(jj=0 ; jj<nbu; jj++) 
		mu[0] += hlam[0][jj] * ht[0][jj] + hlam[0][pnb+jj] * ht[0][pnb+jj];

	for(jj=0; jj<nbu; jj++) 
		{ 
		hrd[0][jj]     =   hux[0][jj] - hd[0][jj]     - ht[0][jj]; 
		hrd[0][pnb+jj] = - hux[0][jj] - hd[0][pnb+jj] - ht[0][pnb+jj]; 
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
	//dgemv_t_lib(nx, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%bs, cnz, hux[0]+nu, hrq[0], -1); // TODO fix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	dgemv_t_lib(nx+nu%bs, nu, hpQ[0]+(nu/bs)*bs*cnz, cnz, hux[0]+nu/bs*bs, hrq[0], hrq[0], -1);
	for(jj=0; jj<nu%bs; jj++) 
		hux[0][(nu/bs)*bs+jj] = temp[jj];
	dsymv_lib(nu, nu, hpQ[0], cnz, hux[0], hrq[0], -1);
	dgemv_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], hrq[0], -1);

	for(jj=0; jj<nx; jj++) 
		hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dgemv_t_lib(nxu, nx, hpBAbt[0], cnx, hux[0], hrb[0], hrb[0], -1);



	// middle blocks
	for(ii=1; ii<N; ii++)
		{

		for(jj=0 ; jj<nb; jj++) 
			mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];

		for(jj=0; jj<nb; jj++) 
			{	
			hrd[ii][jj]     =   hux[ii][jj] - hd[ii][jj]     - ht[ii][jj]; 
			hrd[ii][pnb+jj] = - hux[ii][jj] - hd[ii][pnb+jj] - ht[ii][pnb+jj]; 
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

	for(jj=nu; jj<nb; jj++) 
		{ 
		hrd[N][jj]     =   hux[N][jj] - hd[N][jj]     - ht[N][jj]; 
		hrd[N][pnb+jj] = - hux[N][jj] - hd[N][pnb+jj] - ht[N][pnb+jj]; 
		}

	for(jj=0; jj<nbx; jj++) 
		hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj] + hlam[N][nu+jj] - hlam[N][pnb+nu+jj];
	for(; jj<nx; jj++) 
		hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs, -1);
	


	// general constraints
	if(ng>0)
		{
		for(ii=0; ii<N; ii++)
			{

			for(jj=0 ; jj<ng; jj++) 
				mu[0] += hlam[ii][2*pnb+jj] * ht[ii][2*pnb+jj] + hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];

			dgemv_t_lib(nx+nu, ng, hpDCt[ii], cng, hux[ii], hrd[ii]+2*pnb, hrd[ii]+2*pnb, 0);
			for(jj=0; jj<ng; jj++)
				{
				hrd[ii][2*pnb+png+jj] = - hrd[ii][2*pnb+jj];
				hrd[ii][2*pnb+jj] += - hd[ii][2*pnb+jj] - ht[ii][2*pnb+jj];
				hrd[ii][2*pnb+png+jj] += - hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];
				}

			// TODO work space + one dgemv call
			dgemv_n_lib(nx+nu, ng, hpDCt[ii], cng, hlam[ii]+2*pnb, hrq[ii], hrq[ii], 1);
			dgemv_n_lib(nx+nu, ng, hpDCt[ii], cng, hlam[ii]+2*pnb+png, hrq[ii], hrq[ii], -1);
				
			}
		}
	if(ngN>0)
		{
		for(jj=0 ; jj<ngN; jj++) 
			mu[0] += hlam[N][2*pnb+jj] * ht[N][2*pnb+jj] + hlam[N][2*pnb+pngN+jj] * ht[N][2*pnb+pngN+jj];

		dgemv_t_lib(nx+nu, ngN, hpDCt[N], cngN, hux[N], hrd[N]+2*pnb, hrd[N]+2*pnb, 0);
		for(jj=0; jj<ngN; jj++)
			{
			hrd[N][2*pnb+pngN+jj] = - hrd[N][2*pnb+jj];
			hrd[N][2*pnb+jj] += - hd[N][2*pnb+jj] - ht[N][2*pnb+jj];
			hrd[N][2*pnb+pngN+jj] += - hd[N][2*pnb+pngN+jj] - ht[N][2*pnb+pngN+jj];
			}

		// TODO work space + one dgemv call
		dgemv_n_lib(nx+nu, ngN, hpDCt[N], cngN, hlam[N]+2*pnb, hrq[N], hrq[N], 1);
		dgemv_n_lib(nx+nu, ngN, hpDCt[N], cngN, hlam[N]+2*pnb+pngN, hrq[N], hrq[N], -1);
			
		}

	// normalize mu
	mu[0] /= (N-1)*2*(nb+ng) + 2*(nb+ngN); // + 2*nbx;

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




void d_res_ip_diag_mpc(int N, int *nx, int *nu, int *nb, int **idxb, double **hdA, double **hpBt, double **hpR, double **hpSt, double **hpQ, double **hb, double **hrq, double **hd, double **hux, double **hpi, double **hlam, double **ht, double **hres_rq, double **hres_b, double **hres_d, double *mu, double *work)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	int ii, jj;

	int nu0, nu1, cnu0, nx0, nx1, nxm, cnx0, cnx1, nb0, pnb, nb_tot;


	// initialize mu
	nb_tot = 0;
	mu[0] = 0;


	// first stage
	ii = 0;
	nu0 = nu[ii];
	nu1 = nu[ii+1];
	nx0 = nx[ii]; // nx1;
	nx1 = nx[ii+1];
	cnu0  = (nu0+ncl-1)/ncl*ncl;
	cnx1  = (nx1+ncl-1)/ncl*ncl;
	nxm = (nx0<nx1) ? nx0 : nx1;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	nb_tot += nb[ii];

	for(jj=0; jj<nb0; jj++)
		mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];
	
	for(jj=0; jj<nb0; jj++)
		{
		hres_d[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
		hres_d[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
		}

	for(jj=0; jj<nu0; jj++) hres_rq[ii][jj] = - hrq[ii][jj];
	for(jj=0; jj<nb0; jj++) hres_rq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
	for(jj=0; jj<nx0; jj++) work[jj] = hux[ii][nu0+jj];
	dgemv_t_lib(nx0, nu0, hpSt[ii], cnu0, work, hres_rq[ii], hres_rq[ii], -1);
	dsymv_lib(nu0, nu0, hpR[ii], cnu0, hux[ii], hres_rq[ii], -1);
	dgemv_n_lib(nu0, nx1, hpBt[ii], cnx1, hpi[ii+1], hres_rq[ii], hres_rq[ii], -1);

	for(jj=0; jj<nx1; jj++) hres_b[ii][jj] = hux[ii+1][nu1+jj] - hb[ii][jj];
	for(jj=0; jj<nxm; jj++) hres_b[ii][jj] -= hdA[ii][jj] * work[jj];
	dgemv_t_lib(nu0, nx1, hpBt[ii], cnx1, hux[ii], hres_b[ii], hres_b[ii], -1);


	// middle stages
	for(ii=1; ii<N; ii++)
		{
		nu0 = nu1;
		nu1 = nu[ii+1];
		nx0 = nx1;
		nx1 = nx[ii+1];
		cnu0  = ncl*((nu0+ncl-1)/ncl);
		cnx0 = cnx1;
		cnx1  = ncl*((nx1+ncl-1)/ncl);
		nxm = (nx0<nx1) ? nx0 : nx1;
		nb0 = nb[ii];
		pnb = (nb0+bs-1)/bs*bs;
		nb_tot += nb[ii];

		for(jj=0; jj<nb0; jj++)
			mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];

		for(jj=0; jj<nb0; jj++)
			{
			hres_d[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
			hres_d[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
			}

		for(jj=0; jj<nu0+nx0; jj++) hres_rq[ii][jj] = - hrq[ii][jj];
		for(jj=0; jj<nb0; jj++) hres_rq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];

		for(jj=0; jj<nx0; jj++) work[jj] = hux[ii][nu0+jj];
		dgemv_t_lib(nx0, nu0, hpSt[ii], cnu0, work, hres_rq[ii], hres_rq[ii], -1);
		dsymv_lib(nu0, nu0, hpR[ii], cnu0, hux[ii], hres_rq[ii], -1);
		dgemv_n_lib(nu0, nx1, hpBt[ii], cnx1, hpi[ii+1], hres_rq[ii], hres_rq[ii], -1);

		for(jj=0; jj<nx0; jj++) hres_rq[ii][nu0+jj] += hpi[ii][jj];
		for(jj=0; jj<nxm; jj++) hres_rq[ii][nu0+jj] -= hdA[ii][jj] * hpi[ii+1][jj];
		dgemv_n_lib(nx0, nu0, hpSt[ii], cnu0, hux[ii], hres_rq[ii]+nu0, hres_rq[ii]+nu0, -1);
		dsymv_lib(nx0, nx0, hpQ[ii], cnx0, work, hres_rq[ii]+nu0, -1);

		for(jj=0; jj<nx1; jj++) hres_b[ii][jj] = hux[ii+1][nu1+jj] - hb[ii][jj];
		for(jj=0; jj<nxm; jj++) hres_b[ii][jj] -= hdA[ii][jj] * work[jj];
		dgemv_t_lib(nu0, nx1, hpBt[ii], cnx1, hux[ii], hres_b[ii], hres_b[ii], -1);

		}

	// last stage
	ii = N;
	nu0 = nu1;
	nx0 = nx1;
	cnx0 = cnx1;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	nb_tot += nb[ii];

	for(jj=0; jj<nb0; jj++)
		mu[0] += hlam[ii][jj] * ht[ii][jj] + hlam[ii][pnb+jj] * ht[ii][pnb+jj];

	for(jj=0; jj<nb0; jj++)
		{
		hres_d[ii][jj]     =   hux[ii][idxb[ii][jj]] - hd[ii][jj]     - ht[ii][jj];
		hres_d[ii][pnb+jj] = - hux[ii][idxb[ii][jj]] - hd[ii][pnb+jj] - ht[ii][pnb+jj];
		}

	for(jj=0; jj<nx0; jj++) hres_rq[ii][jj] = hpi[ii][jj] - hrq[ii][jj]; 
	for(jj=0; jj<nb0; jj++) hres_rq[ii][idxb[ii][jj]] += hlam[ii][jj] - hlam[ii][pnb+jj];
	for(jj=0; jj<nx0; jj++) work[jj] = hux[ii][nu0+jj];
	dsymv_lib(nx0, nx0, hpQ[ii], cnx0, work, hres_rq[ii]+nu0, -1);

	// normalize mu
	mu[0] /= 2.0*nb_tot;

	}




