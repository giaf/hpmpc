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


#include "../../include/blas_d.h"
#include "../../include/block_size.h"



/* supports the problem size to change stage-wise */
void d_res_res_mpc_hard_tv(int N, int const * nx, int const * nu, int const * nb, int **idxb, int const * ng, double **hpBAbt, double **hb, double **hpQ, double **hq, double **hux, double **hpDCt, double **hd, double **hpi, double **hlam, double **ht, double *work, double **hrq, double **hrb, double **hrd, double **hrm, double *mu)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii, jj;
	
	int nu0, nu1, cnux0, nx0, nx1, nxm, cnx0, cnx1, nb0, pnb, ng0, png, cng, nb_tot;

	double mu2;


	// initialize mu
	nb_tot = 0;
	mu2 = 0;



	// first stage
	ii = 0;
	nu0 = nu[ii];
	nu1 = nu[ii+1];
	nx0 = nx[ii]; // nx1;
	nx1 = nx[ii+1];
	cnx1  = (nx1+ncl-1)/ncl*ncl;
	cnux0 = (nu0+nx0+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = (ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;

//	for(jj=0; jj<nu0; jj++) 
//		hrq[ii][jj] = hq[ii][jj];

//	for(jj=0; jj<nx0; jj++) 
//		hrq[ii][nu0+jj] = hq[ii][nu0+jj]; // - hpi[ii-1][jj];

	for(jj=0; jj<nu0+nx0; jj++) 
		hrq[ii][jj] = hq[ii][jj];

	if(nb0>0)
		{

		nb_tot += nb0;

		for(jj=0; jj<nb0; jj++) 
			{
			hrq[ii][idxb[ii][jj]] += - hlam[ii][jj] + hlam[ii][pnb+jj];

			hrd[ii][jj]     = hd[ii][jj]     - hux[ii][idxb[ii][jj]] + ht[ii][jj];
			hrd[ii][pnb+jj] = hd[ii][pnb+jj] - hux[ii][idxb[ii][jj]] - ht[ii][pnb+jj];

			hrm[ii][jj]     = hlam[ii][jj]     * ht[ii][jj];
			hrm[ii][pnb+jj] = hlam[ii][pnb+jj] * ht[ii][pnb+jj];
			mu2 += hrm[ii][jj] + hrm[ii][pnb+jj];
			}
		}

	dsymv_lib(nu0+nx0, nu0+nx0, hpQ[ii], cnux0, hux[ii], 1, hrq[ii], hrq[ii]);

	for(jj=0; jj<nx1; jj++) 
		hrb[ii][jj] = hb[ii][jj] - hux[ii+1][nu1+jj];

	dgemv_nt_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hpi[ii], hux[ii], 1, 1, hrq[ii], hrb[ii], hrq[ii], hrb[ii]);

	if(ng0>0)
		{

		nb_tot += ng0;

		for(jj=0; jj<ng0; jj++)
			{
			work[jj] = hlam[ii][jj+2*pnb+png] - hlam[ii][jj+2*pnb+0];

			hrd[ii][2*pnb+jj]     = hd[ii][2*pnb+jj]     + ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] = hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];

			hrm[ii][2*pnb+jj]     = hlam[ii][2*pnb+jj]     * ht[ii][2*pnb+jj];
			hrm[ii][2*pnb+png+jj] = hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
			mu2 += hrm[ii][2*pnb+jj] + hrm[ii][2*pnb+png+jj];
			}

		dgemv_nt_lib(nu0+nx0, ng0, hpDCt[ii], cng, work, hux[ii], 1, 0, hrq[ii], work+png, hrq[ii], work+png);

		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+jj]     -= work[png+jj];
			hrd[ii][2*pnb+png+jj] -= work[png+jj];
			}

		}



	// middle stages
	for(ii=1; ii<N; ii++)
		{
		nu0 = nu1;
		nu1 = nu[ii+1];
		nx0 = nx1;
		nx1 = nx[ii+1];
		cnx0 = cnx1;
		cnx1  = (nx1+ncl-1)/ncl*ncl;
		cnux0  = (nu0+nx0+ncl-1)/ncl*ncl;
		nb0 = nb[ii];
		pnb = (nb0+bs-1)/bs*bs;
		ng0 = ng[ii];
		png = (ng0+bs-1)/bs*bs;
		cng = (ng0+ncl-1)/ncl*ncl;

		for(jj=0; jj<nu0; jj++) 
			hrq[ii][jj] = + hq[ii][jj];

		for(jj=0; jj<nx0; jj++) 
			hrq[ii][nu0+jj] = + hq[ii][nu0+jj] - hpi[ii-1][jj];

		if(nb0>0)
			{

			nb_tot += nb0;

			for(jj=0; jj<nb0; jj++) 
				{
				hrq[ii][idxb[ii][jj]] += - hlam[ii][jj] + hlam[ii][pnb+jj];

				hrd[ii][jj]     = hd[ii][jj]     - hux[ii][idxb[ii][jj]] + ht[ii][jj];
				hrd[ii][pnb+jj] = hd[ii][pnb+jj] - hux[ii][idxb[ii][jj]] - ht[ii][pnb+jj];

				hrm[ii][jj]     = hlam[ii][jj]     * ht[ii][jj];
				hrm[ii][pnb+jj] = hlam[ii][pnb+jj] * ht[ii][pnb+jj];
				mu2 += hrm[ii][jj] + hrm[ii][pnb+jj];
				}
			}

		dsymv_lib(nu0+nx0, nu0+nx0, hpQ[ii], cnux0, hux[ii], 1, hrq[ii], hrq[ii]);

		for(jj=0; jj<nx1; jj++) 
			hrb[ii][jj] = hb[ii][jj] - hux[ii+1][nu1+jj];

		dgemv_nt_lib(nu0+nx0, nx1, hpBAbt[ii], cnx1, hpi[ii], hux[ii], 1, 1, hrq[ii], hrb[ii], hrq[ii], hrb[ii]);

		if(ng0>0)
			{

			nb_tot += ng0;

			for(jj=0; jj<ng0; jj++)
				{
				work[jj] = hlam[ii][jj+2*pnb+png] - hlam[ii][jj+2*pnb+0];

				hrd[ii][2*pnb+jj]     = hd[ii][2*pnb+jj]     + ht[ii][2*pnb+jj];
				hrd[ii][2*pnb+png+jj] = hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];

				hrm[ii][2*pnb+jj]     = hlam[ii][2*pnb+jj]     * ht[ii][2*pnb+jj];
				hrm[ii][2*pnb+png+jj] = hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
				mu2 += hrm[ii][2*pnb+jj] + hrm[ii][2*pnb+png+jj];
				}

			dgemv_nt_lib(nu0+nx0, ng0, hpDCt[ii], cng, work, hux[ii], 1, 0, hrq[ii], work+png, hrq[ii], work+png);

			for(jj=0; jj<ng0; jj++)
				{
				hrd[ii][2*pnb+jj]     -= work[png+jj];
				hrd[ii][2*pnb+png+jj] -= work[png+jj];
				}

			}

		}
	


	// last stage
	ii = N;
	nu0 = nu1;
	nx0 = nx1;
	cnux0  = (nu0+nx0+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = (nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = (ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;

	// res_q
	for(jj=0; jj<nx0; jj++) 
		hrq[ii][nu0+jj] = - hpi[ii-1][jj] + hq[ii][nu0+jj];

	if(nb0>0)
		{

		nb_tot += nb0;

		for(jj=0; jj<nb0; jj++) 
			{
			hrq[ii][idxb[ii][jj]] += - hlam[ii][jj] + hlam[ii][pnb+jj];

			hrd[ii][jj]     = hd[ii][jj]     - hux[ii][idxb[ii][jj]] + ht[ii][jj];
			hrd[ii][pnb+jj] = hd[ii][pnb+jj] - hux[ii][idxb[ii][jj]] - ht[ii][pnb+jj];

			hrm[ii][jj]     = hlam[ii][jj]     * ht[ii][jj];
			hrm[ii][pnb+jj] = hlam[ii][pnb+jj] * ht[ii][pnb+jj];
			mu2 += hrm[ii][jj] + hrm[ii][pnb+jj];
			}
		}

	dsymv_lib(nx0+nu0%bs, nx0+nu0%bs, hpQ[ii]+nu0/bs*bs*cnux0+nu0/bs*bs*bs, cnux0, hux[ii]+nu0/bs*bs, 1, hrq[ii]+nu0/bs*bs, hrq[ii]+nu0/bs*bs);
	
	if(ng0>0)
		{

		nb_tot += ng0;

		for(jj=0; jj<ng0; jj++)
			{
			work[jj] = hlam[ii][jj+2*pnb+png] - hlam[ii][jj+2*pnb+0];

			hrd[ii][2*pnb+jj]     = hd[ii][2*pnb+jj]     + ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] = hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];

			hrm[ii][2*pnb+jj]     = hlam[ii][2*pnb+jj]     * ht[ii][2*pnb+jj];
			hrm[ii][2*pnb+png+jj] = hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
			mu2 += hrm[ii][2*pnb+jj] + hrm[ii][2*pnb+png+jj];
			}

		dgemv_nt_lib(nu0+nx0, ng0, hpDCt[ii], cng, work, hux[ii], 1, 0, hrq[ii], work+png, hrq[ii], work+png);

		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+jj]     -= work[png+jj];
			hrd[ii][2*pnb+png+jj] -= work[png+jj];
			}
		}

	

	// normalize mu
	if(nb_tot!=0)
		{
		mu2 /= 2.0*nb_tot;
		mu[0] = mu2;
		}



	return;

	}



