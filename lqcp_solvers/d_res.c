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



void d_res_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	static double temp[D_MR] = {};

	int ii, jj;
	
	int nxu = nx+nu;

	// first block
	for(jj=0; jj<nu; jj++) hrq[0][jj] = - hq[0][jj];
	for(jj=0; jj<nu%bs; jj++) { temp[jj] = hux[0][(nu/bs)*bs+jj]; hux[0][(nu/bs)*bs+jj] = 0.0; }
	dgemv_t_lib(nx, nu, hpQ[0]+(nu/bs)*bs*cnz+nu%bs, cnz, hux[0]+nu, hrq[0], -1);
	for(jj=0; jj<nu%bs; jj++) hux[0][(nu/bs)*bs+jj] = temp[jj];
	dsymv_lib(nu, nu, hpQ[0], cnz, hux[0], hrq[0], -1);
	dgemv_n_lib(nu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dgemv_t_lib(nxu, nx, hpBAbt[0], cnx, hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj];
		dsymv_lib(nxu, nxu, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		dmvmv_lib(nxu, nx, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}

	// last block
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs, -1);
	
	}



void d_res_mhe_if(int nx, int nw, int N, double **hpQA, double **hpRG, double *L0_inv, double **hq, double **hr, double **hf, double *p0, double **hx, double **hw, double **hlam, double **hrq, double **hrr, double **hrf, double *work)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;
	
	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);

	int ii, jj;

	double *ptr = work;

	double *x_temp; x_temp = ptr; //d_zeros_align(&x_temp, 2*anx, 1);
	//double *x_temp; d_zeros_align(&x_temp, 2*anx, 1);
	ptr += 2*anx;
	double *x_temp2; x_temp2 = ptr; //d_zeros_align(&x_temp, 2*anx, 1);
	//double *x_temp2; d_zeros_align(&x_temp2, 2*anx, 1);
	ptr += 2*anx;

	double *wx_temp; wx_temp = ptr; //d_zeros_align(&wx_temp, anw+anx, 1); // TODO too large 
	//double *wx_temp; d_zeros_align(&wx_temp, anw+anx, 1); // TODO too large 
	ptr += anw+anx;
	double *wx_temp2; wx_temp2 = ptr; //d_zeros_align(&wx_temp, anw+anx, 1); // TODO too large 
	//double *wx_temp2; d_zeros_align(&wx_temp2, anw+anx, 1); // TODO too large 
	ptr += anw+anx;

	// first stage
	for(jj=0; jj<nx; jj++) hrq[0][jj] = hq[0][jj] - p0[jj];
	for(jj=0; jj<nw; jj++) hrr[0][jj] = hr[0][jj];
	for(jj=0; jj<nx; jj++) hrf[0][jj] = hf[0][jj] - hx[1][jj];

	//dsymv_lib(nx, nx, L0_inv, cnx, hx[0], hrq[0], 1);
	dtrmv_u_t_lib(nx, L0_inv, cnx, hx[0], x_temp, 0);
	dtrmv_u_n_lib(nx, L0_inv, cnx, x_temp, hrq[0], 1);
	//dtrmv_u_n_lib(nx, L0_inv, cnx, x_temp, x_temp2, 0);
	//d_print_mat(1, nx, x_temp2, 1);

	for(jj=0; jj<nx; jj++) x_temp[jj] = hx[0][jj];
	for(jj=0; jj<nx; jj++) x_temp[nx+jj] = hlam[0][jj];
	dsymv_lib(2*nx, nx, hpQA[0], cnx, x_temp, x_temp2, 0);
	for(jj=0; jj<nx; jj++) hrq[0][jj] += x_temp2[jj];
	for(jj=0; jj<nx; jj++) hrf[0][jj] += x_temp2[nx+jj];

	for(jj=0; jj<nw; jj++) wx_temp[jj] = hw[0][jj];
	for(jj=0; jj<nx; jj++) wx_temp[nw+jj] = hlam[0][jj];
	//d_print_mat(nx+nw, 1, wx_temp, nx+nw);
	//d_print_pmat(nx+nw, nw, bs, hpRG[0], cnw);
	dsymv_lib(nw+nx, nw, hpRG[0], cnw, wx_temp, wx_temp2, 0);
	//d_print_mat(nx+nw, 1, wx_temp2, nx+nw);
	for(jj=0; jj<nw; jj++) hrr[0][jj] += wx_temp2[jj];
	for(jj=0; jj<nx; jj++) hrf[0][jj] += wx_temp2[nw+jj];

	//d_print_mat(1, nx, hrq[0], 1);
	//d_print_mat(1, nw, hrr[0], 1);
	//d_print_mat(1, nx, hrf[0], 1);
	//exit(2);

	// middle stages
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nx; jj++) hrq[ii][jj] = hq[ii][jj] - hlam[ii-1][jj];
		for(jj=0; jj<nw; jj++) hrr[ii][jj] = hr[ii][jj];
		for(jj=0; jj<nx; jj++) hrf[ii][jj] = hf[ii][jj] - hx[ii+1][jj];

		for(jj=0; jj<nx; jj++) x_temp[jj] = hx[ii][jj];
		for(jj=0; jj<nx; jj++) x_temp[nx+jj] = hlam[ii][jj];
		dsymv_lib(2*nx, nx, hpQA[ii], cnx, x_temp, x_temp2, 0);
		for(jj=0; jj<nx; jj++) hrq[ii][jj] += x_temp2[jj];
		for(jj=0; jj<nx; jj++) hrf[ii][jj] += x_temp2[nx+jj];

		for(jj=0; jj<nw; jj++) wx_temp[jj] = hw[ii][jj];
		for(jj=0; jj<nx; jj++) wx_temp[nw+jj] = hlam[ii][jj];
		dsymv_lib(nw+nx, nw, hpRG[ii], cnw, wx_temp, wx_temp2, 0);
		for(jj=0; jj<nw; jj++) hrr[ii][jj] += wx_temp2[jj];
		for(jj=0; jj<nx; jj++) hrf[ii][jj] += wx_temp2[nw+jj];

		//d_print_mat(1, nx, hrq[ii], 1);
		//d_print_mat(1, nw, hrr[ii], 1);
		//d_print_mat(1, nx, hrf[ii], 1);
		//exit(1);
		}
	
	// last stage
	for(jj=0; jj<nx; jj++) hrq[N][jj] = hq[N][jj] - hlam[N-1][jj];
	dsymv_lib(nx, nx, hpQA[N], cnx, hx[N], hrq[N], 1);
	//d_print_mat(1, nx, hrr[N], 1);

	//free(x_temp);
	//free(x_temp2);
	//free(wx_temp);
	//free(wx_temp2);

	//exit(1);
	


	return;

	}




void d_res_mhe_old(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hq, double **hux, double **hpi, double **hrq, double **hrb)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int pnz = bs*((nx+nu+1+bs-1)/bs);
	const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	int ii, jj;
	
	int nxu = nx+nu;

	// first block
	for(jj=0; jj<nu+nx; jj++) hrq[0][jj] = - hq[0][jj];
	dsymv_lib(nxu, nxu, hpQ[0], cnz, hux[0], hrq[0], -1);
	for(jj=0; jj<nx; jj++) hrb[0][jj] = hux[1][nu+jj] - hpBAbt[0][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
	dmvmv_lib(nxu, nx, hpBAbt[0], cnx, hpi[1], hrq[0], hux[0], hrb[0], -1);

	// middle blocks
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hrq[ii][jj] = - hq[ii][jj];
		for(jj=0; jj<nx; jj++) hrq[ii][nu+jj] = hpi[ii][jj] - hq[ii][nu+jj];
		dsymv_lib(nxu, nxu, hpQ[ii], cnz, hux[ii], hrq[ii], -1);
		for(jj=0; jj<nx; jj++) hrb[ii][jj] = hux[ii+1][nu+jj] - hpBAbt[ii][(nxu/bs)*bs*cnx+nxu%bs+bs*jj];
		dmvmv_lib(nxu, nx, hpBAbt[ii], cnx, hpi[ii+1], hrq[ii], hux[ii], hrb[ii], -1);
		}

	// last block
	for(jj=0; jj<nx; jj++) hrq[N][nu+jj] = hpi[N][jj] - hq[N][nu+jj];
	dsymv_lib(nx+nu%bs, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hux[N]+(nu/bs)*bs, hrq[N]+(nu/bs)*bs, -1);
	
	}

