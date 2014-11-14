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
		dtrmm_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl); // TODO allow 'rectanguar' B
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		dsyrk_dpotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag, 1);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
/*		d_transpose_pmat_lo(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);*/
		dtrtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
/*d_print_pmat(pnz, cnl, bs, hpL[N-ii-1], cnl);*/
/*exit(1);*/
		}

	// first stage 
	dtrmm_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
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



#if defined(TARGET_C99_4X4)
// version tailored for MHE
void d_ric_trf_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpL, double **hpQ, double **hpR, double **hpLp, double *work)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nz = nx+ny;
	const int anz = bs*((nz+nal-1)/nal);
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

	int ii, jj;
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

		// compute C*U', with U upper cholesky factor of /Pi
		dtrmm_lib(ny, nx, hpC[ii], cnx, hpL[ii]+(nx+nw+pad)*bs, cnl, CL, cnx);
		// d_print_pmat(ny, nx, bs, CL, cnx);

		// compute R + (C*U')*(C*U')' on the top left of Lam
		dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[ii], cny, Lam, cnz, 1);
		// d_print_pmat(nz, nz, bs, Lam, cnz);

		// copy C*U' on the bottom left of Lam
		dgetr_lib(ny, 0, nx, ny, CL, cnx, Lam+(ny/bs)*bs*cnz+ny%bs, cnz);
		// d_print_pmat(nz, nz, bs, Lam, cnz);

		// recover overwritten part of I in bottom right part of Lam
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
			*ptr = 1.0;
			ptr += 1;
			for(ii=jj+1; ii<((ny+bs-1)/bs)*bs; ii+=1)
				{
				*ptr = 0.0;
				ptr += 1;
				}
			}
		// d_print_pmat(nz, nz, bs, Lam, cnz);

		// cholesky factorization of Lam
		dpotrf_lib(nz, nz, Lam, cnz, Fam, cnf, diag);
		// d_print_pmat(nz, nz, bs, Fam, cnf);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// transpose and align the bottom right part of Lam
		dtrtr_l_lib(nx, ny, Fam+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, Fam+ncl*bs, cnf);	
		// d_print_pmat(nz, nz, bs, Fam, cnf);

		// compute upper cholesky factor of /Pi+ using triangular-triangular matrix multiplication
		// d_print_pmat(nx, nx+nw+pad+nx, bs, hpL[ii], cnl);
		dttmm_uu_lib(nx, Fam+ncl*bs, cnf, hpL[ii]+(nx+nw+pad)*bs, cnl, hpLp[ii], cnx);
		// d_print_pmat(nx, nx, bs, hpLp[ii], cnx);

		// compute A*U', with U' upper cholesky factor of /Pi+
		// d_print_pmat(nx, nx, bs, hpA[ii], cnx);
		dtrmm_lib(nx, nx, hpA[ii], cnx, hpLp[ii], cnx, hpL[ii+1], cnl);
		// d_print_pmat(nx, nx+nw+pad+nx, bs, hpL[ii+1], cnl);

		// compute lower cholesky factor of Q
		dpotrf_lib(nw, nw, hpQ[ii], cnw, Lam_w, cnw, diag);
		// d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// transpose in place the lower cholesky factor of Q
		dtrtr_l_lib(nw, 0, Lam_w, cnw, Lam_w, cnw);	
		// d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// compute G*U', with U' upper cholesky factor of Q
		// d_print_pmat(nx, nw, bs, hpG[ii], cnw);
		dtrmm_lib(nx, nw, hpG[ii], cnw, Lam_w, cnw, hpL[ii+1]+nx*bs, cnl);
		// d_print_pmat(nx, nx+nw+pad+nx, bs, hpL[ii+1], cnl);

		// compute /Pi and factorize it
		dsyrk_dpotrf_lib(nx, nx+nw, nx, hpL[ii+1], cnl, hpL[ii+1], cnl, diag, 0);
		// d_print_pmat(nx, nx+nw+pad+nx, bs, hpL[ii+1], cnl);

		// transpose in place the lower cholesky factor of /Pi
		dtrtr_l_lib(nx, 0, hpL[ii+1]+(nx+nw+pad)*bs, cnl, hpL[ii+1]+(nx+nw+pad)*bs, cnl);	
		d_print_pmat(nx, cnl, bs, hpL[ii+1], cnl);

		}

	exit(1);

	}
#endif




/* version tailored for mhe (x0 free) */
void d_ric_sv_mhe(int nx, int nu, int N, double **hpBAbt, double **hpQ, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi)
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
		dtrmm_lib(nz, nx, hpBAbt[N-ii-1], cnx, hpL[N-ii]+(nx+pad+ncl)*bs, cnl, hpL[N-ii-1], cnl);
		for(jj=0; jj<nx; jj++) hpL[N-ii-1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+jj*bs] += hpL[N-ii][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nx+pad+nu+jj)*bs];
		dsyrk_dpotrf_lib(nz, nx, nu+nx, hpL[N-ii-1], cnl, hpQ[N-ii-1], cnz, diag, 1);
		for(jj=0; jj<nu; jj++) hpL[N-ii-1][(nx+pad)*bs+(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
		dtrtr_l_lib(nx, nu, hpL[N-ii-1]+(nx+pad)*bs+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-ii-1]+(nx+pad+ncl)*bs, cnl);	
		}

	// first stage 
	dtrmm_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(nx+pad+ncl)*bs, cnl, hpL[0], cnl);
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



void d_ric_trs_mhe(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi)
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

