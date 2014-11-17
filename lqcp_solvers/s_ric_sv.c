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

/*#include "../include/aux_s.h"*/
#include "../include/blas_s.h"
#include "../include/block_size.h"



/* version tailored for mpc (x0 fixed) */
void s_ric_sv_mpc(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hux, float **hpL, float *work, float *diag, int compute_pi, float **hpi)
	{
	
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	const int nz = nx+nu+1;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int ii, jj;

	/* factorization and backward substitution */

	// final stage 
	ssyrk_spotrf_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag);

/*	d_transpose_pmat_lo(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);*/
	strtr_l_lib(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);	

/*s_print_pmat(pnz, cnl, bs, hpL[N], cnl);*/
/*exit(1);*/

	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		strmm_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl); // TODO allow 'rectanguar' B
/*s_print_pmat(pnz, cnl, bs, hpL[N-ii-1], cnl);*/
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		ssyrk_spotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
/*s_print_mat(1, nu, diag, 1);*/
/*		d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);*/
		strtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
/*s_print_pmat(nz, cnl, bs, hpL[N-ii-1], cnl);*/
/*if(ii==1)*/
/*exit(1);*/
		}

	// first stage 
	strmm_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
	ssyrk_spotrf_lib(nz, nx, ((nu+2-1)/2)*2, hpL[0], cnl, hpQ[0], cnz, diag);
	for(jj=0; jj<nu; jj++) hpL[0][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal

/*s_print_pmat(pnz, cnl, bs, hpL[0], cnl);*/
/*s_print_pmat(pnz, cnl, bs, hpL[1], cnl);*/
/*exit(1);*/

	// forward substitution 
	for(ii=0; ii<N; ii++)
		{
/*printf("\n%d\n", ii);*/
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
/*s_print_mat(1, pnz, hux[ii], 1);*/
		strsv_sgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
/*s_print_mat(1, pnz, hux[ii], 1);*/
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
/*s_print_mat(1, pnz, hux[ii+1], 1);*/
		sgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
/*s_print_mat(1, pnz, hux[ii+1], 1);*/
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			for(jj=0; jj<nx; jj++) work[jj] = hpL[ii+1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*s_print_mat(1, nx, work, 1);*/
/*			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1); // TODO remove nu*/
			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
/*s_print_mat(1, nx, work, 1);*/
			strmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
/*s_print_mat(1, nx, work, 1);*/
			}
/*s_print_mat(1, pnz, hux[ii+1], 1);*/
/*if(ii==1)*/
/*exit(1);*/
		}
	
	}



void s_ric_trs_mpc(int nx, int nu, int N, float **hpBAbt, float **hpL, float **hq, float **hux, float *work, int compute_Pb, float ** hPb, int compute_pi, float **hpi)
	{
	
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int ii, jj;
	
	/* backward substitution */
	for(ii=0; ii<N; ii++)
		{
/*		strmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hux[N-ii]+nu, work+pnz, 0);*/
/*		for(jj=0; jj<nx; jj++) work[jj] = hq[N-ii][nu+jj]; // copy p*/
/*		strmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+pnz, work, 1); // L*(L'*b) + p*/
		if(compute_Pb)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
			strmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
			strmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[N-ii-1], 0); // L*(L'*b)
			}
		for(jj=0; jj<nx; jj++) work[jj] = hPb[N-ii-1][jj] + hq[N-ii][nu+jj]; // add p
		sgemv_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		strsv_sgemv_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);
/*s_print_mat(1, nx+nu, hq[N-ii-1], 1);*/
/*if(ii==1)*/
/*exit(1);*/
		}

/*d_print_pmat(nz, nz, bs, hpL[0]+(nx+pad)*bs, cnl);*/
/*exit(3);*/

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		strsv_sgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
/*		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
		sgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);*/
			strmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}



/* version tailored for mhe (x0 free) */
void s_ric_sv_mhe_old(int nx, int nu, int N, float **hpBAbt, float **hpQ, float **hux, float **hpL, float *work, float *diag, int compute_pi, float **hpi)
	{
	
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	const int nz = nx+nu+1;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int ii, jj;


	// factorization and backward substitution 

	// final stage 
	ssyrk_spotrf_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag);
	strtr_l_lib(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);	


	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		strmm_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl);
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		ssyrk_spotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
		strtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
		}

	// first stage 
	strmm_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
	ssyrk_spotrf_lib(nz, nx, nu+nx, hpL[0], cnl, hpQ[0], cnz, diag);
	for(jj=0; jj<nu+nx; jj++) hpL[0][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal


	// forward substitution 

	// first stage 
	for(jj=0; jj<nu+nx; jj++) hux[0][jj] = - hpL[0][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
	strsv_sgemv_t_lib(nx+nu, nx+nu, &hpL[0][(nx+pad)*bs], cnl, &hux[0][0]);
	for(jj=0; jj<nx; jj++) hux[1][nu+jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
	sgemv_t_lib(nx+nu, nx, 0, hpBAbt[0], cnx, &hux[0][0], &hux[1][nu], 1);
	if(compute_pi)
		{
		for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
		for(jj=0; jj<nx; jj++) work[jj] = hpL[1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*		strmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &hux[1][nu], &work[0], 1);*/
		strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
		strmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[1][0], 0); // L*(L'*b) + p
		}

	// later stages
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
		strsv_sgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		sgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			for(jj=0; jj<nx; jj++) work[jj] = hpL[ii+1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1);*/
			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
			strmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
	}



void s_ric_trs_mhe_old(int nx, int nu, int N, float **hpBAbt, float **hpL, float **hq, float **hux, float *work, int compute_Pb, float ** hPb, int compute_pi, float **hpi)
	{
	
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int ii, jj;
	
	// backward substitution 

	// later stages
	for(ii=0; ii<N-1; ii++)
		{
/*		//for(jj=0; jj<nx; jj++) work[jj] = hpBAbt[N-ii-1][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b*/
/*		//strmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);*/
/*		strmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hux[N-ii]+nu, work+pnz, 0);*/
/*		for(jj=0; jj<nx; jj++) work[jj] = hq[N-ii][nu+jj]; // copy p*/
/*		strmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+pnz, work, 1); // L*(L'*b) + p*/
		if(compute_Pb)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
			strmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
			strmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[N-ii-1], 0); // L*(L'*b)
			}
		for(jj=0; jj<nx; jj++) work[jj] = hPb[N-ii-1][jj] + hq[N-ii][nu+jj]; // add p
		sgemv_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		strsv_sgemv_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);
		}

	// first stage 
/*	//for(jj=0; jj<nx; jj++) work[jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b*/
/*	//strmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);*/
/*	strmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, hux[1]+nu, work+pnz, 0);*/
/*	for(jj=0; jj<nx; jj++) work[jj] = hq[1][nu+jj]; // copy p*/
/*	strmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work+pnz, work, 1); // L*(L'*b) + p*/
	if(compute_Pb)
		{
		for(jj=0; jj<nx; jj++) work[jj] = hux[1][nu+jj]; // copy b in aligned memory
		strmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
		strmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[0], 0); // L*(L'*b)
		}
	for(jj=0; jj<nx; jj++) work[jj] = hPb[0][jj] + hq[N-ii][nu+jj]; // add p
	sgemv_n_lib(nx+nu, nx, hpBAbt[0], cnx, work, hq[0], 1);
	strsv_sgemv_n_lib(nu+nx, nu+nx, hpL[0]+(nx+pad)*bs, cnl, hq[0]);


	// forward substitution 

	// first stage 
	for(jj=0; jj<nu+nx; jj++) hux[0][jj] = - hq[0][jj];
	strsv_sgemv_t_lib(nx+nu, nx+nu, &hpL[0][(nx+pad)*bs], cnl, &hux[0][0]);
/*	for(jj=0; jj<nx; jj++) hux[1][nu+jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
	sgemv_t_lib(nx+nu, nx, 0, hpBAbt[0], cnx, &hux[0][0], &hux[1][nu], 1);
	if(compute_pi)
		{
		for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
		strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*		strmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &hux[1][nu], &work[0], 0);*/
		strmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[1][0], 0); // L*(L'*b) + p
		for(jj=0; jj<nx; jj++) hpi[1][jj] += hq[1][nu+jj];
		}

	// later stages
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		strsv_sgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
/*		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
		sgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*			strmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);*/
			strmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}

