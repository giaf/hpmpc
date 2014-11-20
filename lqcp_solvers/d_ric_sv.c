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

/*#include "../include/aux_d.h"*/
#include "../include/blas_d.h"
#include "../include/block_size.h"



/* version tailored for mpc (x0 fixed) */
void d_ric_sv_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
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
	dsyrk_dpotrf_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag, 1);

/*	d_transpose_pmat_lo(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);*/
	dtrtr_l_lib(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);	

/*d_print_pmat(pnz, cnl, bs, hpL[N], cnl);*/

	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		dtrmm_l_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl); // TODO allow 'rectanguar' B
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		dsyrk_dpotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag, 1);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
/*		d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);*/
		dtrtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
/*d_print_pmat(pnz, cnl, bs, hpL[N-ii-1], cnl);*/
/*exit(1);*/
		}

	// first stage 
	dtrmm_l_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
	dsyrk_dpotrf_lib(nz, nx, ((nu+2-1)/2)*2, hpL[0], cnl, hpQ[0], cnz, diag, 1);
	for(jj=0; jj<nu; jj++) hpL[0][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal

/*d_print_pmat(pnz, cnl, bs, hpL[0], cnl);*/
/*exit(1);*/

	// forward substitution 
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		dgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			for(jj=0; jj<nx; jj++) work[jj] = hpL[ii+1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1); // TODO remove nu*/
			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
			dtrmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
	}



void d_ric_trs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
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
		if(compute_Pb)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
			dtrmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
			dtrmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[N-ii-1], 0); // L*(L'*b)
			}
		for(jj=0; jj<nx; jj++) work[jj] = hPb[N-ii-1][jj] + hq[N-ii][nu+jj]; // add p
		dgemv_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		dtrsv_dgemv_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);
		}

/*d_print_pmat(nz, nz, bs, hpL[0]+(nx+pad)*bs, cnl);*/
/*exit(3);*/

	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
/*		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
		dgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);*/
			dtrmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}



// xp is the vector of predictions, xe is the vector of estimates
//#if 0
void d_ric_trs_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double **hq, double **hr, double **hf, double **hxp, double **hxe, double **hy, double *work)
	{

	//printf("\nhola\n");

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nz = nx+ny;
	const int anw = nal*((nw+nal-1)/nal);
	const int any = nal*((ny+nal-1)/nal);
	const int anz = nal*((nz+nal-1)/nal);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnz = ncl*((nz+ncl-1)/ncl);

	int ii, jj, ll;
	double *ptr;

	ptr = work;

	double *y_temp = ptr; //; d_zeros_align(&y_temp, anz, 1);
	ptr += anz;

	double *w_temp = ptr; //; d_zeros_align(&w_temp, anw, 1);
	ptr += anw;

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		// copy y
		for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[ii][jj];
		//d_print_mat(1, nz, y_temp, 1);
	
		// copy xp
		for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[ii][jj];
		//d_print_mat(1, nz, y_temp, 1);
	
		// compute y + R*r
		dsymv_lib(ny, 0, hpR[ii], cny, hr[ii], y_temp, -1);
		//d_print_mat(1, nz, y_temp, 1);

		// compute y + R*r - C*xp
		dgemv_n_lib(ny, nx, hpC[ii], cnx, hxp[ii], y_temp, 1);
		//d_print_mat(1, nz, y_temp, 1);

		//d_print_pmat(nz, ny, bs, hpLe[ii], cnz);

		// compute xe
		dtrsv_dgemv_n_lib(ny, ny+nx, hpLe[ii], cnz, y_temp);
		//d_print_mat(1, nz, y_temp, 1);

		// copy xe
		for(jj=0; jj<nx; jj++) hxe[ii][jj] = y_temp[ny+jj];
		//d_print_mat(1, nx, hxe[ii], 1);

		// initialize w_temp with 0
		for(jj=0; jj<nw; jj++) w_temp[jj] = 0.0;
		//d_print_mat(1, nw, w_temp, 1);
	
		// compute Q*q
		dsymv_lib(nw, 0, hpQ[ii], cnw, hq[ii], w_temp, -1);
		//d_print_mat(1, nw, w_temp, 1);

		// copy f in xp
		for(jj=0; jj<nx; jj++) hxp[ii+1][jj] = hf[ii][jj];
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		// xp += G*w_temp
		dgemv_n_lib(nx, nw, hpG[ii], cnw, w_temp, hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		// xp += A*xe
		dgemv_n_lib(nx, nx, hpA[ii], cnx, hxe[ii], hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);

		}
	
	// stage N

	// copy y
	for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// copy xp
	for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// compute y + R*r
	dsymv_lib(ny, 0, hpR[N], cny, hr[N], y_temp, -1);
	//d_print_mat(1, nz, y_temp, 1);

	// compute y + R*r - C*xp
	dgemv_n_lib(ny, nx, hpC[N], cnx, hxp[N], y_temp, 1);
	//d_print_mat(1, nz, y_temp, 1);

	//d_print_pmat(nz, ny, bs, hpLe[N], cnz);

	// compute xe
	dtrsv_dgemv_n_lib(ny, ny+nx, hpLe[N], cnz, y_temp);
	//d_print_mat(1, nz, y_temp, 1);

	// copy xe
	for(jj=0; jj<nx; jj++) hxe[N][jj] = y_temp[ny+jj];
	//d_print_mat(1, nx, hxe[N], 1);

	//exit(1);

	}
//#endif



//#if defined(TARGET_C99_4X4)
// version tailored for MHE
void d_ric_trf_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double *work)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nz = nx+ny;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnw = bs*((nw+bs-1)/bs);
	const int pny = bs*((ny+bs-1)/bs);
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnf = cnz<cnx+ncl ? cnx+ncl : cnz;

	const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
	const int cnl = nx+nw+pad+cnx;

	int ii, jj, ll;
	double *ptr;

	ptr = work;

	double *CL = ptr; //d_zeros_align(&CL, pny, cnx);
	ptr += pny*cnx;

	double *CLLt = ptr; //d_zeros_align(&CLLt, pny, cnx);
	ptr += pny*cnx;

	double *Lam = ptr; // d_zeros_align(&Lam, pnz, cnz);
	ptr += pnz*cnz;

	double *diag = ptr; // d_zeros_align(&diag, anz, 1);
	ptr += anz;

	double *Lam_w = ptr; // d_zeros_align(&Lam_w, pnw, cnw);
	ptr += pnw*cnw;
	
	double *Pi_p = ptr; //  d_zeros_align(&Pi_p, pnx, cnx);
	ptr += pnx*cnx;

	//double *CLLt = ptr; d_zeros_align(&CLLt, pny, cnx);
	static double buffer[6] = {};

	double *ptr1;

	// compute /Pi_p from its cholesky factor
	//d_print_pmat(nx, nx, bs, hpLp[0]+(nx+nw+pad)*bs, cnl);
	dttmm_ll_lib(nx, hpLp[ii]+(nx+nw+pad)*bs, cnl, hpLp[0]+(nx+nw+pad)*bs, cnl, Pi_p, cnx);
	//d_print_pmat(nx, nx, bs, Pi_p, cnx);

	// copy /Pi_p on the bottom right block of Lam
	dtrma_lib(nx, ny, Pi_p, cnx, Lam+(ny/bs)*bs*cnz+ny%bs+ny*bs, cnz);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		// backup the top-left part of the bottom right block of Lam
		ptr1 = buffer;
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
			ptr1[0] = ptr[0];
			ptr += 1;
			ptr1 += 1;
			for(ll=jj+1; ll<((ny+bs-1)/bs)*bs; ll+=1)
				{
				ptr1[0] = ptr[0];
				ptr += 1;
				ptr1 += 1;
				}
			}
		//d_print_mat(6, 1, buffer, 1);

		// compute C*U', with U upper cholesky factor of /Pi_p
		dtrmm_l_lib(ny, nx, hpC[ii], cnx, hpLp[ii]+(nx+nw+pad)*bs, cnl, CL, cnx);
		//d_print_pmat(ny, nx, bs, CL, cnx);

		// compute R + (C*U')*(C*U')' on the top left of Lam
		dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[ii], cny, Lam, cnz, 1);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// recover overwritten part of I in bottom right part of Lam
		ptr1 = buffer;
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
			ptr[0] = ptr1[0];
			ptr += 1;
			ptr1 += 1;
			for(ll=jj+1; ll<((ny+bs-1)/bs)*bs; ll+=1)
				{
				ptr[0] = ptr1[0];
				ptr += 1;
				ptr1 += 1;
				}
			}
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// compute C*U'*L'
		dtrmm_u_lib(ny, nx, CL, cnx, hpLp[ii]+(nx+nw+pad)*bs, cnl, CLLt, cnx);
		//d_print_pmat(ny, nx, bs, CLLt, cnx);

		// copy C*U'*L' on the bottom left of Lam
		dgetr_lib(ny, 0, nx, ny, CLLt, cnx, Lam+(ny/bs)*bs*cnz+ny%bs, cnz);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// cholesky factorization of Lam
		dpotrf_lib(nz, nz, Lam, cnz, hpLe[ii], cnf, diag);
		//d_print_pmat(nz, nz, bs, hpLe[ii], cnf);
		//d_print_pmat(nz, nz, bs, Lam, cnz);
		//d_print_mat(nz, 1, diag, 1);

		// inverted diagonal of top-left part of hpLe
		for(jj=0; jj<ny; jj++) hpLe[ii][(jj/bs)*bs*cnf+jj%bs+jj*bs] = diag[jj];

		// transpose and align /Pi_e
		dtrtr_l_lib(nx, ny, hpLe[ii]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, hpLe[ii]+ncl*bs, cnf);	
		//d_print_pmat(nz, nz, bs, hpLe[ii], cnf);

		// compute A*U', with U' upper cholesky factor of /Pi_e
		// d_print_pmat(nx, nx, bs, hpA[ii], cnx);
		dtrmm_l_lib(nx, nx, hpA[ii], cnx, hpLe[ii]+ncl*bs, cnf, hpLp[ii+1], cnl);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// compute lower cholesky factor of Q
		dpotrf_lib(nw, nw, hpQ[ii], cnw, Lam_w, cnw, diag);
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// transpose in place the lower cholesky factor of Q
		dtrtr_l_lib(nw, 0, Lam_w, cnw, Lam_w, cnw);	
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// compute G*U', with U' upper cholesky factor of Q
		// d_print_pmat(nx, nw, bs, hpG[ii], cnw);
		dtrmm_l_lib(nx, nw, hpG[ii], cnw, Lam_w, cnw, hpLp[ii+1]+nx*bs, cnl);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// compute /Pi_p
		dsyrk_lib(nx, nx, nx+nw, hpLp[ii+1], cnl, hpLp[ii+1], cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, 0);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// copy /Pi_p on the bottom right block of Lam
		dtrma_lib(nx, ny, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, Lam+(ny/bs)*bs*cnz+ny%bs+ny*bs, cnz);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// factorize Pi_p
		dpotrf_lib(nx, nx, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, diag);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// transpose in place the lower cholesky factor of /Pi_p
		dtrtr_l_lib(nx, 0, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);	
		//d_print_pmat(nx, cnl, bs, hpLp[ii+1], cnl);

		//exit(1);
		}

	// stage N

	// backup the top-left part of the bottom right block of Lam
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
		ptr1[0] = ptr[0];
		ptr += 1;
		ptr1 += 1;
		for(ll=jj+1; ll<((ny+bs-1)/bs)*bs; ll+=1)
			{
			ptr1[0] = ptr[0];
			ptr += 1;
			ptr1 += 1;
			}
		}
	//d_print_mat(6, 1, buffer, 1);

	// compute C*U', with U upper cholesky factor of /Pi_p
	dtrmm_l_lib(ny, nx, hpC[N], cnx, hpLp[N]+(nx+nw+pad)*bs, cnl, CL, cnx);
	//d_print_pmat(ny, nx, bs, CL, cnx);

	// compute R + (C*U')*(C*U')' on the top left of Lam
	dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[N], cny, Lam, cnz, 1);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// recover overwritten part of I in bottom right part of Lam
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
		ptr[0] = ptr1[0];
		ptr += 1;
		ptr1 += 1;
		for(ll=jj+1; ll<((ny+bs-1)/bs)*bs; ll+=1)
			{
			ptr[0] = ptr1[0];
			ptr += 1;
			ptr1 += 1;
			}
		}
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// compute C*U'*L'
	dtrmm_u_lib(ny, nx, CL, cnx, hpLp[N]+(nx+nw+pad)*bs, cnl, CLLt, cnx);
	//d_print_pmat(ny, nx, bs, CLLt, cnx);

	// copy C*U'*L' on the bottom left of Lam
	dgetr_lib(ny, 0, nx, ny, CLLt, cnx, Lam+(ny/bs)*bs*cnz+ny%bs, cnz);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// cholesky factorization of Lam
	dpotrf_lib(nz, nz, Lam, cnz, hpLe[N], cnf, diag);
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);
	//d_print_pmat(nz, nz, bs, Lam, cnz);
	//d_print_mat(nz, 1, diag, 1);

	// inverted diagonal of top-left part of hpLe
	for(jj=0; jj<ny; jj++) hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs] = diag[jj];

	// transpose and align /Pi_e
	dtrtr_l_lib(nx, ny, hpLe[N]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, hpLe[N]+ncl*bs, cnf);	
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);

	//exit(1);

	}
//#endif




//#if defined(TARGET_C99_4X4)
// version tailored for MHE (test)
void d_ric_trf_mhe_test(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double *work)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nz = nx+ny;
	const int anz = nal*((nz+nal-1)/nal);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnw = bs*((nw+bs-1)/bs);
	const int pny = bs*((ny+bs-1)/bs);
	const int pnz = bs*((nz+bs-1)/bs);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnf = cnz<cnx+ncl ? cnx+ncl : cnz;

	const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
	const int cnl = nx+nw+pad+cnx;

	int ii, jj, ll;
	double *ptr;

	ptr = work;

	double *CL = ptr; //d_zeros_align(&CL, pny, cnx);
	ptr += pny*cnx;

	double *Lam = ptr; // d_zeros_align(&Lam, pnz, cnz);
	ptr += pnz*cnz;

	double *diag = ptr; // d_zeros_align(&diag, anz, 1);
	ptr += anz;

	double *Fam = ptr; // d_zeros_align(&Fam, pnz, cnf);
	ptr += pnz*cnf;

	double *Lam_w = ptr; // d_zeros_align(&Lam_w, pnw, cnw);
	ptr += pnw*cnw;
	
	// initialize bottom right part of Lam with identity
	for(ii=0; ii<pnz*cnz; ii++)
		Lam[ii] = 0.0;
	for(ii=0; ii<nx; ii++)
		Lam[((ny+ii)/bs)*bs*cnz+(ny+ii)%bs+(ny+ii)*bs] = 1.0;
	// d_print_pmat(nz, nz, bs, Lam, cnz);

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		// compute C*U', with U upper cholesky factor of /Pi_p
		dtrmm_l_lib(ny, nx, hpC[ii], cnx, hpLp[ii]+(nx+nw+pad)*bs, cnl, CL, cnx);
		//d_print_pmat(ny, nx, bs, CL, cnx);

		// compute R + (C*U')*(C*U')' on the top left of Lam
		dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[ii], cny, Lam, cnz, 1);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// copy C*U' on the bottom left of Lam
		dgetr_lib(ny, 0, nx, ny, CL, cnx, Lam+(ny/bs)*bs*cnz+ny%bs, cnz);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// recover overwritten part of I in bottom right part of Lam
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
			*ptr = 1.0;
			ptr += 1;
			for(ll=jj+1; ll<((ny+bs-1)/bs)*bs; ll+=1)
				{
				*ptr = 0.0;
				ptr += 1;
				}
			}
		// d_print_pmat(nz, nz, bs, Lam, cnz);

		// cholesky factorization of Lam
		dpotrf_lib(nz, nz, Lam, cnz, Fam, cnf, diag);
		//d_print_pmat(nz, nz, bs, Fam, cnf);
		//d_print_pmat(nz, nz, bs, Lam, cnz);
		//d_print_mat(nz, 1, diag, 1);

		// transpose and align the bottom right part of Lam
		dtrtr_l_lib(nx, ny, Fam+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, Fam+ncl*bs, cnf);	
		//d_print_pmat(nz, nz, bs, Fam, cnf);

		// compute upper cholesky factor of /Pi_e using triangular-triangular matrix multiplication
		// d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii], cnl);
		dttmm_uu_lib(nx, Fam+ncl*bs, cnf, hpLp[ii]+(nx+nw+pad)*bs, cnl, hpLe[ii], cnx);
		//d_print_pmat(nx, nx, bs, hpLe[ii], cnx);

		// compute A*U', with U' upper cholesky factor of /Pi_e
		// d_print_pmat(nx, nx, bs, hpA[ii], cnx);
		dtrmm_l_lib(nx, nx, hpA[ii], cnx, hpLe[ii], cnx, hpLp[ii+1], cnl);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// compute lower cholesky factor of Q
		dpotrf_lib(nw, nw, hpQ[ii], cnw, Lam_w, cnw, diag);
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// transpose in place the lower cholesky factor of Q
		dtrtr_l_lib(nw, 0, Lam_w, cnw, Lam_w, cnw);	
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// compute G*U', with U' upper cholesky factor of Q
		// d_print_pmat(nx, nw, bs, hpG[ii], cnw);
		dtrmm_l_lib(nx, nw, hpG[ii], cnw, Lam_w, cnw, hpLp[ii+1]+nx*bs, cnl);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// compute /Pi_p and factorize it
		dsyrk_dpotrf_lib(nx, nx+nw, nx, hpLp[ii+1], cnl, hpLp[ii+1], cnl, diag, 0);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// transpose in place the lower cholesky factor of /Pi_p
		dtrtr_l_lib(nx, 0, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);	
		//d_print_pmat(nx, cnl, bs, hpLp[ii+1], cnl);

		//exit(1);
		}

	//exit(1);

	}
//#endif




/* version tailored for mhe (x0 free) */
void d_ric_sv_mhe_old(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
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
	dsyrk_dpotrf_lib(nx+nu%bs+1, 0, nx+nu%bs, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, diag, 1);
	dtrtr_l_lib(nx, nu, hpL[N]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(nx+pad+ncl)*bs, cnl);	


	// middle stages 
	for(ii=0; ii<N-1; ii++)
		{	
		dtrmm_l_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl);
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		dsyrk_dpotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag, 1);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
		dtrtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
		}

	// first stage 
	dtrmm_l_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
	for(jj=0; jj<nx; jj++) hpL[0][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
	dsyrk_dpotrf_lib(nz, nx, nu+nx, hpL[0], cnl, hpQ[0], cnz, diag, 1);
	for(jj=0; jj<nu+nx; jj++) hpL[0][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal


	// forward substitution 

	// first stage 
	for(jj=0; jj<nu+nx; jj++) hux[0][jj] = - hpL[0][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
	dtrsv_dgemv_t_lib(nx+nu, nx+nu, &hpL[0][(nx+pad)*bs], cnl, &hux[0][0]);
	for(jj=0; jj<nx; jj++) hux[1][nu+jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
	dgemv_t_lib(nx+nu, nx, 0, hpBAbt[0], cnx, &hux[0][0], &hux[1][nu], 1);
	if(compute_pi)
		{
		for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
		for(jj=0; jj<nx; jj++) work[jj] = hpL[1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*		dtrmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &hux[1][nu], &work[0], 1);*/
		dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
		dtrmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[1][0], 0); // L*(L'*b) + p
		}

	// later stages
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hpL[ii][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		dgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			for(jj=0; jj<nx; jj++) work[jj] = hpL[ii+1][(nx+pad)*bs+((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
/*			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 1);*/
			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 1);
			dtrmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			}
		}
	
	}



void d_ric_trs_mhe_old(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
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
/*		//dtrmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);*/
/*		dtrmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hux[N-ii]+nu, work+anz, 0);*/
/*		for(jj=0; jj<nx; jj++) work[jj] = hq[N-ii][nu+jj]; // copy p*/
/*		dtrmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, work, 1); // L*(L'*b) + p*/
		if(compute_Pb)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
			dtrmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
			dtrmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[N-ii-1], 0); // L*(L'*b)
			}
		for(jj=0; jj<nx; jj++) work[jj] = hPb[N-ii-1][jj] + hq[N-ii][nu+jj]; // add p
		dgemv_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		dtrsv_dgemv_n_lib(nu, nu+nx, hpL[N-ii-1]+(nx+pad)*bs, cnl, hq[N-ii-1]);
		}

	// first stage 
/*	//for(jj=0; jj<nx; jj++) work[jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b*/
/*	//dtrmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work, work+pnz, 0);*/
/*	dtrmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, hux[1]+nu, work+anz, 0);*/
/*	for(jj=0; jj<nx; jj++) work[jj] = hq[1][nu+jj]; // copy p*/
/*	dtrmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, work+anz, work, 1); // L*(L'*b) + p*/
	if(compute_Pb)
		{
		for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
		dtrmv_u_n_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work, work+anz, 0);
		dtrmv_u_t_lib(nx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, work+anz, hPb[0], 0); // L*(L'*b)
		}
	for(jj=0; jj<nx; jj++) work[jj] = hPb[0][jj] + hq[N-ii][nu+jj]; // add p
	dgemv_n_lib(nx+nu, nx, hpBAbt[0], cnx, work, hq[0], 1);
	dtrsv_dgemv_n_lib(nu+nx, nu+nx, hpL[0]+(nx+pad)*bs, cnl, hq[0]);


	// forward substitution 

	// first stage 
	for(jj=0; jj<nu+nx; jj++) hux[0][jj] = - hq[0][jj];
	dtrsv_dgemv_t_lib(nx+nu, nx+nu, &hpL[0][(nx+pad)*bs], cnl, &hux[0][0]);
/*	for(jj=0; jj<nx; jj++) hux[1][nu+jj] = hpBAbt[0][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
	dgemv_t_lib(nx+nu, nx, 0, hpBAbt[0], cnx, &hux[0][0], &hux[1][nu], 1);
	if(compute_pi)
		{
		for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
		dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*		dtrmv_u_n_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &hux[1][nu], &work[0], 0);*/
		dtrmv_u_t_lib(nx, hpL[1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[1][0], 0); // L*(L'*b) + p
		for(jj=0; jj<nx; jj++) hpi[1][jj] += hq[1][nu+jj];
		}

	// later stages
	for(ii=1; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, &hpL[ii][(nx+pad)*bs], cnl, &hux[ii][0]);
/*		for(jj=0; jj<nx; jj++) hux[ii+1][nu+jj] = hpBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];*/
		dgemv_t_lib(nx+nu, nx, 0, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[anz], &work[0], 0);
/*			dtrmv_u_n_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &hux[ii+1][nu], &work[0], 0);*/
			dtrmv_u_t_lib(nx, hpL[ii+1]+(nx+pad+ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}

