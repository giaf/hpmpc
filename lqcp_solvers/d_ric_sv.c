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

#include <math.h>

/*#include "../include/aux_d.h"*/
#include "../include/aux_d.h"
#include "../include/blas_d.h"
#include "../include/block_size.h"
#include "../include/lqcp_aux.h"



/* version tailored for mpc (x0 fixed) */
void d_ric_sv_mpc(int nx, int nu, int N, double **hpBAbt, double **hpQ, int update_hessian, double **hQd, double **hQl, double **hux, double **hpL, double *work, double *diag, int compute_pi, double **hpi, int nb, int ng, int ngN, double **hpDCt, double **Qx, double **qx, int fast_rsqrt)
	{
	
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line
	
	const int nz   = nx+nu+1;
	const int anz  = nal*((nz+nal-1)/nal);
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int pnb  = bs*((nb+bs-1)/bs);
	const int png  = bs*((ng+bs-1)/bs);
	const int pngN = bs*((ngN+bs-1)/bs);
	const int cnz  = ncl*((nz+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	const int cnxg = ncl*((ng+nx+ncl-1)/ncl);

	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

	// number of general constraints TODO
	//const int ng = 0;

	int nu0 = (nu/bs)*bs;

	int ii, jj, ll, nn;

	double temp;

	// factorization and backward substitution 

	// final stage 
	if(ngN>0)
		{
		//d_print_pmat(nx+nu, ngN, bs, hpDCt[N], cngN);
		//d_print_mat(1, 2*pnb+2*pngN, Qx[N], 1);
		dgemv_n_lib(nx+nu, ngN, hpDCt[N], cngN, qx[N]+2*pnb, hQl[N], 1);
		// copy and scale DCt
		// TODO // routine for this
		for(ii=0; ii<nx+nu-3; ii+=4)
			{
			for(jj=0; jj<ngN; jj++)
				{
				temp = Qx[N][2*pnb+jj];
				//d_print_mat(1, 1, &temp, 1);
				work[0+ii*cngN+jj*bs] = temp * hpDCt[N][0+ii*cngN+jj*bs];
				work[1+ii*cngN+jj*bs] = temp * hpDCt[N][1+ii*cngN+jj*bs];
				work[2+ii*cngN+jj*bs] = temp * hpDCt[N][2+ii*cngN+jj*bs];
				work[3+ii*cngN+jj*bs] = temp * hpDCt[N][3+ii*cngN+jj*bs];
				}
			}
		//d_print_pmat(nz, ngN, bs, work, cngN);
		for(ll=0; ll<nx+nu-ii; ll++)
			{
			for(jj=0; jj<ngN; jj++)
				{
				work[ll+ii*cngN+jj*bs] = Qx[N][2*pnb+jj] * hpDCt[N][ll+ii*cngN+jj*bs];
				}
			}
		for(jj=0; jj<nx+nu; jj++)
			work[(nu+nx)/bs*cngN*bs+(nu+nx)%bs+jj*bs] = 0.0;
		//d_print_pmat(nz, ngN, bs, work, cngN);
		//exit(1);
		}
	if(update_hessian)
		{
		d_update_hessian_ric_sv(nu%bs+nx, hpQ[N]+nu0*cnz+nu0*bs, cnz, hQd[N]+nu0);
		d_update_jacobian_ric_sv(nu%bs+nx, hpQ[N]+nu0*bs+((nx+nu)/bs)*bs*cnz+(nx+nu)%bs, hQl[N]+nu0);
		}
	//printf("\nQl\n");
	//d_print_mat(1, nx+nu, hQl[N], 1);
	//dpotrf_lib(nx+nu%bs+1, nx+nu%bs, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hpL[N]+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, diag);
	//d_print_pmat(nz, ng, bs, work, cnxg);
	//d_print_pmat(nz, nz, bs, hpQ[N], cnz);
	if(nx+nu%bs+1<128)
		{
		dsyrk_dpotrf_lib(nx+nu%bs+1, nx+nu%bs, ngN, work+(nu/bs)*bs*cngN, cngN, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hpL[N]+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, diag, 1, fast_rsqrt);
		}
	else
		{
		dsyrk_lib(nx+nu%bs+1, nx+nu%bs, ngN, work+(nu/bs)*bs*cngN, cngN, work+(nu/bs)*bs*cngN, cngN, hpQ[N]+(nu/bs)*bs*cnz+(nu/bs)*bs*bs, cnz, hpL[N]+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, 1);
		dpotrf_lib(nx+nu%bs+1, nx+nu%bs, hpL[N]+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, hpL[N]+(nu/bs)*bs*cnl+(nu/bs)*bs*bs, cnl, diag);
		}
#if 0
	d_print_pmat(nz, nx+ng, bs, work, cnxg);
	d_print_pmat(nz, nz, bs, hpL[N], cnl);
	d_print_mat(1, ng, Qx[N]+2*pnb, 1);
	d_print_mat(1, ng, Qx[N]+2*pnb+png, 1);
	d_print_mat(1, ng, qx[N]+2*pnb, 1);
	d_print_mat(1, ng, qx[N]+2*pnb+png, 1);
	d_print_mat(1, ng, Qx[N-1]+2*pnb, 1);
	d_print_mat(1, ng, Qx[N-1]+2*pnb+png, 1);
	d_print_mat(1, ng, qx[N-1]+2*pnb, 1);
	d_print_mat(1, ng, qx[N-1]+2*pnb+png, 1);
	exit(1);
#endif

	dtrtr_l_lib(nx, nu, hpL[N]+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N]+(ncl)*bs, cnl);	


	// middle stages 
	for(nn=0; nn<N-1; nn++)
		{	
		dtrmm_l_lib(nz, nx, hpBAbt[N-nn-1], cnx, hpL[N-nn]+(ncl)*bs, cnl, work, cnxg);
		for(jj=0; jj<nx; jj++) work[((nx+nu)/bs)*bs*cnxg+(nx+nu)%bs+(jj)*bs] += hpL[N-nn][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nu+jj)*bs];
		if(ng>0)
			{
			dgemv_n_lib(nx+nu, ng, hpDCt[N-nn-1], cng, qx[N-nn-1]+2*pnb, hQl[N-nn-1], 1);
			// copy and scale DCt
			// TODO // routine for this
			for(ii=0; ii<nx+nu-3; ii+=4)
				{
				for(jj=0; jj<ng; jj++)
					{
					//temp = sqrt(Qx[N-nn-1][2*pnb+jj]+Qx[N-nn-1][2*pnb+png+jj]);
					temp = Qx[N-nn-1][2*pnb+jj];
					work[0+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[N-nn-1][0+ii*cng+jj*bs];
					work[1+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[N-nn-1][1+ii*cng+jj*bs];
					work[2+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[N-nn-1][2+ii*cng+jj*bs];
					work[3+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[N-nn-1][3+ii*cng+jj*bs];
					}
				}
			for(ll=0; ll<nx+nu-ii; ll++)
				{
				for(jj=0; jj<ng; jj++)
					{
					//work[ll+ii*cnxg+(nx+jj)*bs] = sqrt(Qx[N-nn-1][2*pnb+jj]+Qx[N-nn-1][2*pnb+png+jj]) * hpDCt[N-nn-1][ll+ii*cng+jj*bs];
					work[ll+ii*cnxg+(nx+jj)*bs] = Qx[N-nn-1][2*pnb+jj] * hpDCt[N-nn-1][ll+ii*cng+jj*bs];
					}
				}
			//d_print_pmat(nz, nx+ng, bs, work, cnxg);
			for(jj=0; jj<nx+nu; jj++)
				work[(nu+nx)/bs*cnxg*bs+(nu+nx)%bs+(nx+jj)*bs] = 0.0;
			}
		if(update_hessian)
			{
			d_update_hessian_ric_sv(nx+nu, hpQ[N-nn-1], cnz, hQd[N-nn-1]);
			d_update_jacobian_ric_sv(nx+nu, hpQ[N-nn-1]+((nx+nu)/bs)*bs*cnz+(nx+nu)%bs, hQl[N-nn-1]);
			}
		//dsyrk_dpotrf_lib(nz, nu+nx, nx, work+ng*bs, cnx, hpQ[N-nn-1], cnz, hpL[N-nn-1], cnl, diag, 1);
		if(nz<128)
			{
			dsyrk_dpotrf_lib(nz, nu+nx, nx+ng, work, cnxg, hpQ[N-nn-1], cnz, hpL[N-nn-1], cnl, diag, 1, fast_rsqrt);
			}
		else
			{
			dsyrk_lib(nz, nu+nx, nx+ng, work, cnxg, work, cnxg, hpQ[N-nn-1], cnz, hpL[N-nn-1], cnl, 1);
			dpotrf_lib(nz, nu+nx, hpL[N-nn-1], cnl, hpL[N-nn-1], cnl, diag);
			}
#if 0
	d_print_pmat(nz, nx+ng, bs, work, cnxg);
	d_print_pmat(nz, nz, bs, hpL[N-nn-1], cnl);
	d_print_mat(1, ng, Qx[N]+2*pnb, 1);
	d_print_mat(1, ng, Qx[N]+2*pnb+png, 1);
	d_print_mat(1, ng, qx[N]+2*pnb, 1);
	d_print_mat(1, ng, qx[N]+2*pnb+png, 1);
	d_print_mat(1, ng, Qx[N-1]+2*pnb, 1);
	d_print_mat(1, ng, Qx[N-1]+2*pnb+png, 1);
	d_print_mat(1, ng, qx[N-1]+2*pnb, 1);
	d_print_mat(1, ng, qx[N-1]+2*pnb+png, 1);
	exit(1);
#endif

		for(jj=0; jj<nu; jj++) hpL[N-nn-1][(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal
		dtrtr_l_lib(nx, nu, hpL[N-nn-1]+(nu/bs)*bs*cnl+nu%bs+nu*bs, cnl, hpL[N-nn-1]+(ncl)*bs, cnl);	
		}

	// first stage 
	dtrmm_l_lib(nz, nx, hpBAbt[0], cnx, hpL[1]+(ncl)*bs, cnl, work, cnxg);
	for(jj=0; jj<nx; jj++) work[((nx+nu)/bs)*bs*cnxg+(nx+nu)%bs+(jj)*bs] += hpL[1][((nx+nu)/bs)*bs*cnl+(nx+nu)%bs+(nu+jj)*bs];
	if(ng>0)
		{
		dgemv_n_lib(nx+nu, ng, hpDCt[0], cng, qx[0]+2*pnb, hQl[0], 1);
		// copy and scale DCt
		// TODO // routine for this
		for(ii=0; ii<nx+nu-3; ii+=4)
			{
			for(jj=0; jj<ng; jj++)
				{
				//temp = sqrt(Qx[0][2*pnb+jj]+Qx[0][2*pnb+png+jj]);
				temp = Qx[0][2*pnb+jj];
				work[0+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[0][0+ii*cng+jj*bs];
				work[1+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[0][1+ii*cng+jj*bs];
				work[2+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[0][2+ii*cng+jj*bs];
				work[3+ii*cnxg+(nx+jj)*bs] = temp * hpDCt[0][3+ii*cng+jj*bs];
				}
			}
		for(ll=0; ll<nx+nu-ii; ll++)
			{
			for(jj=0; jj<ng; jj++)
				{
				//work[ll+ii*cnxg+(nx+jj)*bs] = sqrt(Qx[0][2*pnb+jj]+Qx[0][2*pnb+png+jj]) * hpDCt[0][ll+ii*cng+jj*bs];
				work[ll+ii*cnxg+(nx+jj)*bs] = Qx[0][2*pnb+jj] * hpDCt[0][ll+ii*cng+jj*bs];
				}
			}
		//d_print_pmat(nz, nx+ng, bs, work, cnxg);
		for(jj=0; jj<nx+nu; jj++)
			work[(nu+nx)/bs*cnxg*bs+(nu+nx)%bs+(nx+jj)*bs] = 0.0;
		}
	if(update_hessian)
		{
		d_update_hessian_ric_sv(nu, hpQ[0], cnz, hQd[0]);
		d_update_jacobian_ric_sv(nu, hpQ[0]+((nx+nu)/bs)*bs*cnz+(nx+nu)%bs, hQl[0]);
		}
	//dsyrk_dpotrf_lib(nz, ((nu+2-1)/2)*2, nx, work+ng*bs, cnx, hpQ[0], cnz, hpL[0], cnl, diag, 1);
	dsyrk_dpotrf_lib(nz, ((nu+2-1)/2)*2, nx+ng, work, cnxg, hpQ[0], cnz, hpL[0], cnl, diag, 1, fast_rsqrt);
	for(jj=0; jj<nu; jj++) hpL[0][(jj/bs)*bs*cnl+jj%bs+jj*bs] = diag[jj]; // copy reciprocal of diagonal


#if 0
	d_print_pmat(nz, nz, bs, hpL[0], cnl);
	d_print_pmat(nz, nz, bs, hpL[1], cnl);
	d_print_pmat(nz, nz, bs, hpL[2], cnl);
	d_print_pmat(nz, nz, bs, hpL[3], cnl);
	d_print_pmat(nz, nz, bs, hpL[N-1], cnl);
	d_print_pmat(nz, nz, bs, hpL[N], cnl);
	//exit(1);
#endif

	// forward substitution 
	for(nn=0; nn<N; nn++)
		{
		for(jj=0; jj<nu; jj++) hux[nn][jj] = - hpL[nn][((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, hpL[nn], cnl, &hux[nn][0]);
		for(jj=0; jj<nx; jj++) hux[nn+1][nu+jj] = hpBAbt[nn][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj];
		dgemv_t_lib(nx+nu, nx, hpBAbt[nn], cnx, &hux[nn][0], &hux[nn+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[nn+1][nu+jj]; // copy x into aligned memory
			for(jj=0; jj<nx; jj++) work[jj] = hpL[nn+1][((nu+nx)/bs)*bs*cnl+(nu+nx)%bs+bs*(nu+jj)]; // work space
			dtrmv_u_n_lib(nx, hpL[nn+1]+(ncl)*bs, cnl, &work[anz], &work[0], 1);
			dtrmv_u_t_lib(nx, hpL[nn+1]+(ncl)*bs, cnl, &work[0], &hpi[nn+1][0], 0); // L*(L'*b) + p
			}
		}
	
	
	}



void d_ric_trs_mpc(int nx, int nu, int N, double **hpBAbt, double **hpL, double **hq, double **hux, double *work, int compute_Pb, double ** hPb, int compute_pi, double **hpi, int nb, int ng, int ngN, double **hpDCt, double **qx)
	{
	
	const int bs  = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz   = nx+nu+1;
	const int anz  = nal*((nz+nal-1)/nal);
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int pnb  = bs*((nb+bs-1)/bs);
//	const int png  = bs*((ng+bs-1)/bs);
	const int cnz  = ncl*((nz+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);

	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

	int ii, jj;
	
	/* backward substitution */
	// general constraints
	if(ngN>0)
		{
		dgemv_n_lib(nx+nu, ngN, hpDCt[N], cngN, qx[N]+2*pnb, hq[N], 1);
		}
	for(ii=0; ii<N; ii++)
		{
		if(compute_Pb)
			{
			for(jj=0; jj<nx; jj++) work[jj] = hux[N-ii][nu+jj]; // copy b in aligned memory
			dtrmv_u_n_lib(nx, hpL[N-ii]+(ncl)*bs, cnl, work, work+anz, 0);
			dtrmv_u_t_lib(nx, hpL[N-ii]+(ncl)*bs, cnl, work+anz, hPb[N-ii-1], 0); // L*(L'*b)
			}
		// general constraints
		if(ng>0)
			{
			dgemv_n_lib(nx+nu, ng, hpDCt[N-ii-1], cng, qx[N-ii-1]+2*pnb, hq[N-ii-1], 1);
			}
		for(jj=0; jj<nx; jj++) work[jj] = hPb[N-ii-1][jj] + hq[N-ii][nu+jj]; // add p
		dgemv_n_lib(nx+nu, nx, hpBAbt[N-ii-1], cnx, work, hq[N-ii-1], 1);
		dtrsv_dgemv_n_lib(nu, nu+nx, hpL[N-ii-1], cnl, hq[N-ii-1]);
		}


	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu; jj++) hux[ii][jj] = - hq[ii][jj];
		dtrsv_dgemv_t_lib(nu, nx+nu, hpL[ii], cnl, &hux[ii][0]);
		dgemv_t_lib(nx+nu, nx, hpBAbt[ii], cnx, &hux[ii][0], &hux[ii+1][nu], 1);
		if(compute_pi)
			{
			for(jj=0; jj<nx; jj++) work[anz+jj] = hux[ii+1][nu+jj]; // copy x into aligned memory
			dtrmv_u_n_lib(nx, hpL[ii+1]+(ncl)*bs, cnl, &work[anz], &work[0], 0);
			dtrmv_u_t_lib(nx, hpL[ii+1]+(ncl)*bs, cnl, &work[0], &hpi[ii+1][0], 0); // L*(L'*b) + p
			for(jj=0; jj<nx; jj++) hpi[ii+1][jj] += hq[ii+1][nu+jj];
			}
		}

	}



// version exploiting A=I
// L = chol(R + B'*P*B)
// K = (P*B)\L
// P = Q + P + K*K'
void d_ric_eye_sv_mpc(int nx, int nu, int N, double **hpBt, double **hpR, double **hpS, double **hpQ, double **hpL, double **hpP, double *work, double *diag)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
//	const int nal = bs*ncl; // number of doubles per cache line
	
//	const int nz   = nx+nu+1;
//	const int anz  = nal*((nz+nal-1)/nal);
//	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int pnu  = bs*((nu+bs-1)/bs);
//	const int pnb  = bs*((nb+bs-1)/bs);
//	const int png  = bs*((ng+bs-1)/bs);
//	const int pngN = bs*((ngN+bs-1)/bs);
//	const int cnz  = ncl*((nz+ncl-1)/ncl);
	const int cnu  = ncl*((nu+ncl-1)/ncl);
	const int cnx  = ncl*((nx+ncl-1)/ncl);
//	const int cng  = ncl*((ng+ncl-1)/ncl);
//	const int cngN = ncl*((ngN+ncl-1)/ncl);
//	const int cnxg = ncl*((ng+nx+ncl-1)/ncl);

//	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

	// number of general constraints TODO
	//const int ng = 0;

	int nu_m = (nu/bs)*bs;
	int nu_r = nu%bs;

	int ii, jj, ll, nn;

	double temp;

	double *pPB, *pPBt, *pK;

	pPB = work;
	work += pnx*cnu;

	pPBt = work;
	work += pnu*cnx;

	pK = work;
	work += pnx*cnu;

	// last stage: inintialize P with Q_N
	d_copy_pmat(nx, nx, bs, hpQ[N], cnx, hpP[N], cnx);

	// factorization and backward substitution 
	for(nn=0; nn<N; nn++)
		{
		// PB = P*(B')'
#if defined(TARGET_C99_4X4) || defined(TARGET_C99_4X4_PREFETCH) || defined(TARGET_X64_AVX) || defined(TARGET_X64_AVX2)
		dgemm_nt_lib(nx, nu, nx, hpP[N-nn], cnx, hpBt[N-nn-1], cnx, pPB, cnu, pPBt, cnx, 0, 0, 1); // TODO embed transpose of result in dgemm_nt
#else
		dgemm_nt_lib(nx, nu, nx, hpP[N-nn], cnx, hpBt[N-nn-1], cnx, pPB, cnu, pPB, cnu, 0, 0, 0);
		//d_print_pmat(nx, nx, bs, hpP[N-nn], cnx);
		//d_print_pmat(nu, nx, bs, hpBt[N-nn-1], cnx);
		//d_print_pmat(nx, nu, bs, pPB, cnu);
		//exit(1);

		// PBt = (PB)'
		dgetr_lib(nx, 0, nu, 0, pPB, cnu, pPBt, cnx);
		//d_print_pmat(nu, nx, bs, pPBt, cnx);
		//exit(1);
#endif

		// R + PBt*B'
		//dgemm_nt_lib(nu, nu, nx, pPBt, cnx, hpBt[N-nn-1], cnx, hpL[N-nn-1], cnu, 0);
		dsyrk_lib(nu, nu, nx, pPBt, cnx, hpBt[N-nn-1], cnx, hpR[N-nn-1], cnu, hpL[N-nn-1], cnu, 1);
		//d_print_pmat(nu, nu, bs, hpL[N-nn-1], cnu);
		//exit(1);

		// S + PBt
		for(ii=0; ii<pnu*cnx; ii++) pPBt[ii] = pPBt[ii] + hpS[N-nn-1][ii]; // TODO routine for this
		//d_print_pmat(nu, nx, bs, pPBt, cnx);
		//exit(1);

		// PB on bottom of L
		dgetr_lib(nu, 0, nx, nu, pPBt, cnx, hpL[N-nn-1]+nu_m*cnu+nu_r, cnu);
		//d_print_pmat(nu+nx, nu, bs, hpL[N-nn-1], cnu);
		//exit(1);
		
		// [L; K] = chol([R + B'*P*B; P*B])
		dpotrf_lib(nx+nu, nu, hpL[N-nn-1], cnu, hpL[N-nn-1], cnu, diag);
		//d_print_pmat(nu+nx, nu, bs, hpL[N-nn-1], cnu);
		//exit(1);

		// copy K to alinged memory
		d_align_pmat(nx, nu, nu, bs, hpL[N-nn-1]+nu/bs*bs*cnu, cnu, pK, cnu); // TODO make kernel for this
		//d_print_pmat(nx, nu, bs, pK, cnu);
		//exit(1);

		// P_n = Q_n + P_{n+1}
		for(ii=0; ii<pnx*cnx; ii++) hpP[N-nn-1][ii] = hpP[N-nn][ii] + hpQ[N-nn-1][ii]; // TODO routine for this
		//d_print_pmat(nx, nx, bs, hpP[N-nn-1], cnx);

		// TODO if nu small, low-rank update
		dsyrk_lib(nx, nx, nu, pK, cnu, pK, cnu, hpP[N-nn-1], cnx, hpP[N-nn-1], cnx, -1);
		//d_print_pmat(nx, nx, bs, hpP[N-nn-1], cnx);

		// copy lower triangular to upper triangular
		dtrtr_l_lib(nx, 0, hpP[N-nn-1], cnx, hpP[N-nn-1], cnx);	
		//d_print_pmat(nx, nx, bs, hpP[N-nn-1], cnx);
		//exit(1);

		}
		
	}



// information filter version
int d_ric_trf_mhe_if(int nx, int nw, int N, double **hpQA, double **hpRG, double **hpALe, double **hpGLr, double *work)
	{

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nwx = nw+nx;
	const int anx = nal*((nx+nal-1)/nal);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnw = bs*((nw+bs-1)/bs);
	const int pnx2 = bs*((2*nx+bs-1)/bs);
	const int pnwx = bs*((nw+nx+bs-1)/bs);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));
	//const int cnwx = ncl*((nw+nx+ncl-1)/ncl);

	const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing
	const int cnl = nx+nw+pad+cnx;

	float diag_min;
	diag_min = 1.0;

	double *ptr;
	ptr = work;

	double *GLrALeLp; GLrALeLp = ptr;
	ptr += pnx*cnl;

	double *diag; diag = ptr;
	ptr += anx;

	int ii, jj;	

	for(ii=0; ii<N; ii++)
		{
		//d_print_pmat(2*nx, cnx, bs, hpQA[ii], cnx);
		//dtsyrk_dpotrf_lib(2*nx, nx, nx, hpALe[ii], cnx2, hpQA[ii], cnx, diag, 1);
		//d_print_pmat(2*nx, cnx2, bs, hpALe[ii], cnx2);
		dtsyrk_dpotrf_lib(2*nx, nx, nx, hpALe[ii], cnx2, hpQA[ii], cnx, hpALe[ii]+(cnx)*bs, cnx2, diag, 1);
		//d_print_pmat(2*nx, cnx2, bs, hpALe[ii], cnx2);
		//exit(1);
		// copy reciprocal of diagonal
		//d_print_pmat(2*nx, cnx2, bs, hpALe[ii], cnx2);
		for(jj=0; jj<nx; jj++) 
			{
			diag_min = fmin(diag_min, diag[jj]);
			hpALe[ii][cnx*bs+(jj/bs)*bs*cnx2+jj%bs+jj*bs] = diag[jj]; 
			}
		//d_print_pmat(2*nx, cnx2, bs, hpALe[ii], cnx2);

		dpotrf_lib(nwx, nw, hpRG[ii], cnw, hpGLr[ii], cnw, diag);
		//d_print_pmat(nwx, nw, bs, hpGLr[ii], cnw);

		// copy reciprocal of diagonal
		for(jj=0; jj<nw; jj++) hpGLr[ii][(jj/bs)*bs*cnw+jj%bs+jj*bs] = diag[jj]; 
		//d_print_pmat(nwx, nw, bs, hpGLr[0], cnw);
		//exit(1);

		d_align_pmat(nx, nw, nw, bs, hpGLr[ii], cnw, GLrALeLp, cnl);
		d_align_pmat(nx, nx, nx, bs, hpALe[ii]+cnx*bs, cnx2, GLrALeLp+nw*bs, cnl);
		//d_print_pmat(nx, cnl, bs, GLrALeLp, cnl);

		//dsyrk_dpotrf_dtrinv_lib(nx, nx, nwx, GLrALeLp, cnl, ptr, 0, hpALe[ii+1], cnx2, diag, 0);
		dsyrk_dpotrf_dtrinv_lib(nx, nx, nwx, GLrALeLp, cnl, ptr, 0, GLrALeLp+(nw+nx+pad)*bs, cnl, hpALe[ii+1], cnx2, diag, 0);
		//d_print_pmat(nx, cnl, bs, GLrALeLp, cnl);
		//d_print_pmat(2*nx, cnx2, bs, hpALe[ii+1], cnx2);
		for(jj=0; jj<nx; jj++) 
			{
			diag_min = fmin(diag_min, diag[jj]);
			}

		//exit(1);

		if(diag_min==0.0)
			return ii+1;

		//if(ii==2)
		//exit(1);
		}

	//d_print_pmat(nx, nx, bs, GLrALeLp+(nx+nw+pad)*bs, cnl);
	//dtsyrk_dpotrf_lib(nx, nx, nx, hpALe[N], cnx2, hpQA[N], cnx, diag, 1);
	dtsyrk_dpotrf_lib(nx, nx, nx, hpALe[N], cnx2, hpQA[N], cnx, hpALe[N]+(cnx)*bs, cnx2, diag, 1);
	//d_print_pmat(nx, cnx2, bs, hpALe[ii], cnx2);
	// copy reciprocal of diagonal
	for(jj=0; jj<nx; jj++) 
		{
		diag_min = fmin(diag_min, diag[jj]);
		hpALe[N][cnx*bs+(jj/bs)*bs*cnx2+jj%bs+jj*bs] = diag[jj]; 
		}

	if(diag_min==0.0)
		return ii+1;

	//exit(1);

	return 0;

	}



// information filter version
void d_ric_trs_mhe_if(int nx, int nw, int N, double **hpALe, double **hpGLr, double **hq, double **hr, double **hf, double **hxp, double **hx, double **hw, double **hlam, double *work)
	{

	//printf("\nin solver\n");

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));

	int ii, jj;

	double *ptr = work;

	double *x_temp; x_temp = ptr; //d_zeros_align(&x_temp, 2*anx, 1);
	ptr += 2*anx;

	double *wx_temp; wx_temp = ptr; //d_zeros_align(&wx_temp, anw+anx, 1); // TODO too large 
	ptr += anw+anx;



	// forward substitution
	for(ii=0; ii<N; ii++)
		{

		//printf("\nii = %d\n", ii);

		// compute sigma
		for(jj=0; jj<nx; jj++) hx[ii][jj] = hq[ii][jj];
		//d_print_mat(1, nx, hx[ii], 1);
		//d_print_mat(1, nx, hxp[ii], 1);
		//d_print_pmat(nx, nx, bs, hpALe[ii], cnx2); 
		dtrmv_u_t_lib(nx, hpALe[ii], cnx2, hxp[ii], x_temp, 0); // L*(L'*b) + p
		//d_print_mat(1, nx, x_temp, 1);
		dtrmv_u_n_lib(nx, hpALe[ii], cnx2, x_temp, hx[ii], -1);
		//d_print_mat(1, nx, hx[ii], 1);

		// compute hxp
		for(jj=0; jj<nx; jj++) x_temp[jj] = hx[ii][jj];
		for(jj=0; jj<nx; jj++) x_temp[nx+jj] = hf[ii][jj];
		//d_print_pmat(2*nx, nx, bs, hpALe[ii]+cnx*bs, cnx2);
		dtrsv_dgemv_n_lib(nx, 2*nx, hpALe[ii]+cnx*bs, cnx2, x_temp);
		for(jj=0; jj<nx; jj++) hx[ii][jj] = - x_temp[jj]; // restore sign
		//d_print_mat(1, 2*nx, x_temp, 1);
		for(jj=0; jj<nw; jj++) wx_temp[jj] = hr[ii][jj];
		for(jj=0; jj<nx; jj++) wx_temp[nw+jj] = x_temp[nx+jj];
		dtrsv_dgemv_n_lib(nw, nw+nx, hpGLr[ii], cnw, wx_temp);
		//d_print_mat(1, nw+nx, wx_temp, 1);
		for(jj=0; jj<nw; jj++) hw[ii][jj] = wx_temp[jj];
		for(jj=0; jj<nx; jj++) hxp[ii+1][jj] = wx_temp[nw+jj];
		//d_print_mat(1, nx, hxp[ii+1], 1);
		//d_print_mat(1, nw, hw[ii], 1);
	
		//if(ii==1)
		//return 0;
		//exit(1);
		}

	// compute - sigma !!! - !!!
	for(jj=0; jj<nx; jj++) hx[N][jj] = - hq[N][jj];
	//d_print_pmat(nx, nx, bs, hpALe[N], cnx2); 
	//d_print_mat(1, nx, hx[N], 1);
	//d_print_mat(1, nx, hxp[N], 1);
	dtrmv_u_t_lib(nx, hpALe[N], cnx2, hxp[N], x_temp, 0); // L*(L'*b) + p
	//d_print_mat(1, nx, x_temp, 1);
	dtrmv_u_n_lib(nx, hpALe[N], cnx2, x_temp, hx[N], 1);
	//d_print_mat(1, nx, hx[N], 1);

	// backwars substitution
	//d_print_pmat(nx, nx, bs, hpALe[N]+cnx*bs, cnx2); 
	//d_print_mat(1, nx, hx[N], 1);
	dtrsv_dgemv_n_lib(nx, nx, hpALe[N]+cnx*bs, cnx2, hx[N]);
	//d_print_mat(1, nx, hx[N], 1);
	dtrsv_dgemv_t_lib(nx, nx, hpALe[N]+cnx*bs, cnx2, hx[N]);
	//d_print_mat(1, nx, hx[N], 1);

	for(ii=0; ii<N; ii++)
		{

		// compute lambda
		for(jj=0; jj<nx; jj++) x_temp[jj] = hxp[N-ii][jj] - hx[N-ii][jj];
		dtrmv_u_t_lib(nx, hpALe[N-ii], cnx2, x_temp, x_temp+anx, 0); // L*(L'*b) + p
		dtrmv_u_n_lib(nx, hpALe[N-ii], cnx2, x_temp+anx, hlam[N-ii-1], 0);

		// compute x
		for(jj=0; jj<nx; jj++) x_temp[jj] = hx[N-ii-1][jj];
		for(jj=0; jj<nx; jj++) x_temp[nx+jj] = hlam[N-ii-1][jj];
		dtrsv_dgemv_t_lib(nx, nx+nx, hpALe[N-ii-1]+cnx*bs, cnx2, x_temp);
		for(jj=0; jj<nx; jj++) hx[N-ii-1][jj] = x_temp[jj];

		// compute w
		for(jj=0; jj<nw; jj++) wx_temp[jj] = -hw[N-ii-1][jj];
		for(jj=0; jj<nx; jj++) wx_temp[nw+jj] = hlam[N-ii-1][jj];
		dtrsv_dgemv_t_lib(nw, nw+nx, hpGLr[N-ii-1], cnw, wx_temp);
		for(jj=0; jj<nw; jj++) hw[N-ii-1][jj] = wx_temp[jj];

		}



	// free memory TODO remove !!!
	//free(*sigma);
	//free(x_temp);
	//free(x2_temp);
	//free(wx_temp);

	//exit(1);

	return;

	}



// xp is the vector of predictions, xe is the vector of estimates
//#if 0
int d_ric_trs_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hdLp, double **hpQ, double **hpR, double **hpLe, double **hq, double **hr, double **hf, double **hxp, double **hxe, double **hw, double **hy, int smooth, double **hlam, double *work)
	{

	//printf("\nhola\n");

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl;

	const int nz = nx+ny;
	const int anx = nal*((nx+nal-1)/nal);
	const int anw = nal*((nw+nal-1)/nal);
	const int any = nal*((ny+nal-1)/nal);
	const int anz = nal*((nz+nal-1)/nal);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cnw = ncl*((nw+ncl-1)/ncl);
	const int cny = ncl*((ny+ncl-1)/ncl);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnf = cnz<cnx+ncl ? cnx+ncl : cnz;

	const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
	const int cnl = nx+nw+pad+cnx;

	int ii, jj, ll;
	int return_value = 0;
	double *ptr;
	double Lmin;

	ptr = work;

	double *y_temp = ptr; //; d_zeros_align(&y_temp, anz, 1);
	ptr += anz;

	double *w_temp = ptr; //; d_zeros_align(&w_temp, anw, 1);
	ptr += anw;

	double *x_temp = ptr; //; d_zeros_align(&w_temp, anw, 1);
	ptr += 2*anx;

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		//printf("\nii = %d\n", ii);

		// copy y
		for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[ii][jj];
		//d_print_mat(1, nz, y_temp, 1);
	
		// compute y + R*r
		dsymv_lib(ny, ny, hpR[ii], cny, hr[ii], y_temp, -1);
		//d_print_mat(1, nz, y_temp, 1);

		// compute y + R*r - C*xp
		//int pny = bs*((ny+bs-1)/bs);
		//d_print_pmat(pny, cnx, bs, hpC[ii], cnx);
		//d_print_mat(1, anx, hxp[ii], 1);
		dgemv_n_lib(ny, nx, hpC[ii], cnx, hxp[ii], y_temp, 1);
		//d_print_mat(1, nz, y_temp, 1);

		//d_print_pmat(nz, ny, bs, hpLe[ii], cnf);

		// copy xp
		for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[ii][jj];
		//d_print_mat(1, nz, y_temp, 1);
	
		// compute xe
		dtrsv_dgemv_n_lib(ny, ny+nx, hpLe[ii], cnf, y_temp);
		//d_print_mat(1, nz, y_temp, 1);

		// copy xe
		for(jj=0; jj<nx; jj++) hxe[ii][jj] = y_temp[ny+jj];
		//d_print_mat(1, nx, hxe[ii], 1);
		//exit(1);

		// copy f in xp
		for(jj=0; jj<nx; jj++) hxp[ii+1][jj] = hf[ii][jj];
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		// xp += A*xe
		dgemv_n_lib(nx, nx, hpA[ii], cnx, hxe[ii], hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);

		// initialize w with 0
		for(jj=0; jj<nw; jj++) hw[ii][jj] = 0.0;
		//d_print_mat(1, nw, w_temp, 1);
	
		// compute Q*q
		dsymv_lib(nw, nw, hpQ[ii], cnw, hq[ii], hw[ii], -1);
		//d_print_mat(1, nw, w_temp, 1);

		// xp += G*w
		dgemv_n_lib(nx, nw, hpG[ii], cnw, hw[ii], hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		//if(ii==1)
		//return 0;
		//exit(1);

		}
	
	// stage N

	// copy y
	for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// compute y + R*r
	dsymv_lib(ny, ny, hpR[N], cny, hr[N], y_temp, -1);
	//d_print_mat(1, nz, y_temp, 1);

	// compute y + R*r - C*xp
	dgemv_n_lib(ny, nx, hpC[N], cnx, hxp[N], y_temp, 1);
	//d_print_mat(1, nz, y_temp, 1);

	//d_print_pmat(nz, ny, bs, hpLe[N], cnf);

	// copy xp
	for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// compute xe
	dtrsv_dgemv_n_lib(ny, ny+nx, hpLe[N], cnf, y_temp);
	//d_print_mat(1, nz, y_temp, 1);

	// copy xe
	for(jj=0; jj<nx; jj++) hxe[N][jj] = y_temp[ny+jj];
	//d_print_mat(1, nx, hxe[N], 1);

	//return 0;
	//exit(1);

	if(smooth==0)
		return return_value;
	
	// backward recursion to compute smoothed values

	for(ii=0; ii<N; ii++)
		{

		//printf("\nii = %d\n", ii);

		// check for singular covariance
		Lmin = 1;
		for(jj=0; jj<nx; jj++) Lmin = fmin(Lmin, hdLp[N-ii][jj]);
		//printf("\nL_min = %f\n", Lmin);

		// if singular, keep the current estimate as smooth value and go to the next iteration
		if(Lmin==0.0)
			{

			// the N-ii th prediction covariance matrix is singular
			return_value = N-ii;

			}
		// else compute smooth values
		else
			{

			// backup diagonal and overwrite with inverted diagonal
			//d_print_pmat(nx, nx, bs, hpLp[N-ii]+(nx+nw+pad)*bs, cnl);
			for(jj=0; jj<nx; jj++)
				{
				x_temp[jj] = hpLp[N-ii][(jj/bs)*bs*cnl+jj%bs+(nx+nw+pad+jj)*bs];
				hpLp[N-ii][(jj/bs)*bs*cnl+jj%bs+(nx+nw+pad+jj)*bs] = hdLp[N-ii][jj];
				}
			//d_print_pmat(nx, nx, bs, hpLp[N-ii]+(nx+nw+pad)*bs, cnl);

			// lam = xp - xe
			for(jj=0; jj<nx; jj++) hlam[N-ii-1][jj] = hxp[N-ii][jj] - hxe[N-ii][jj];
			//d_print_mat(1, nx, hlam[N-ii-1], 1);

			// lam = \Pi^{-1}*lam
			dtrsv_dgemv_n_lib(nx, nx, hpLp[N-ii]+(nx+nw+pad)*bs, cnl, hlam[N-ii-1]);
			//d_print_mat(1, nx, hlam[N-ii-1], 1);
			dtrsv_dgemv_t_lib(nx, nx, hpLp[N-ii]+(nx+nw+pad)*bs, cnl, hlam[N-ii-1]);
			//d_print_mat(1, nx, hlam[N-ii-1], 1);

			// restore diagonal
			for(jj=0; jj<nx; jj++)
				hpLp[N-ii][(jj/bs)*bs*cnl+jj%bs+(nx+nw+pad+jj)*bs] = x_temp[jj];
			//d_print_pmat(nx, nx, bs, hpLp[N-ii]+(nx+nw+pad)*bs, cnl);

			// G'*lam
			//d_print_pmat(nx, nw, bs, hpG[N-ii-1], cnw);
			dgemv_t_lib(nx, nw, hpG[N-ii-1], cnw, hlam[N-ii-1], w_temp, 0);
			//d_print_mat(nw, 1, w_temp, 1);

			// w = w - Q*G'*lam
			//d_print_pmat(nw, nw, bs, hpQ[N-ii-1], cnw);
			dsymv_lib(nw, nw, hpQ[N-ii-1], cnw, w_temp, hw[N-ii-1], -1);
			//d_print_mat(nw, 1, hw[N-ii-1], 1);

			// A'*lam
			//d_print_pmat(nx, nx, bs, hpA[N-ii-1], cnx);
			dgemv_t_lib(nx, nx, hpA[N-ii-1], cnx, hlam[N-ii-1], x_temp, 0);
			//d_print_mat(nx, 1, x_temp, 1);

			// xe = xe - Pi_e*A'*lam
			//d_print_pmat(nx, nx, bs, hpLe[N-ii-1]+ncl*bs, cnf);
			dtrmv_u_n_lib(nx, hpLe[N-ii-1]+ncl*bs, cnf, x_temp, x_temp+anx, 0);
			//d_print_mat(nx, 1, x_temp+anx, 1);
			//d_print_mat(nx, 1, hxe[N-ii-1], 1);
			dtrmv_u_t_lib(nx, hpLe[N-ii-1]+ncl*bs, cnf, x_temp+anx, hxe[N-ii-1], -1); // L*(L'*b) + p
			//d_print_mat(nx, 1, hxe[N-ii-1], 1);

			//exit(1);

			}

		//if(ii==40)
		//	exit(1);

		}

	return return_value;

	}
//#endif



// xp is the vector of predictions, xe is the vector of estimates; explicitly computes estimates only at the last stage
//#if 0
void d_ric_trs_mhe_end(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double **hq, double **hr, double **hf, double **hxp, double **hxe, double **hy, double *work)
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
	const int cnf = cnz<cnx+ncl ? cnx+ncl : cnz;

	const int pad = (ncl-(nx)%ncl)%ncl; // packing between AGL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

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

		//printf("\nii = %d\n", ii);

		// copy y
		for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[ii][jj];
		//d_print_mat(1, nz, y_temp, 1);
	
		// copy A*xp
		//for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[ii][jj];
		dgemv_n_lib(nx, nx, hpA[ii], cnx, hxp[ii], y_temp+ny, 0);
		//d_print_mat(1, nz, y_temp, 1);
	
		// compute y + R*r
		dsymv_lib(ny, ny, hpR[ii], cny, hr[ii], y_temp, -1);
		//d_print_mat(1, nz, y_temp, 1);

		// compute y + R*r - C*xp
		dgemv_n_lib(ny, nx, hpC[ii], cnx, hxp[ii], y_temp, 1);
		//d_print_mat(1, nz, y_temp, 1);

		//d_print_pmat(nz, ny, bs, hpLp[ii+1]+(nx+pad)*bs, cnl);
		//d_print_pmat(nz, cnl, bs, hpLp[ii+1], cnl);

		// compute A*xe
		dtrsv_dgemv_n_lib(ny, ny+nx, hpLp[ii+1]+(nx+pad)*bs, cnl, y_temp);
		//d_print_mat(1, nz, y_temp, 1);

		// copy A*xe in xp
		for(jj=0; jj<nx; jj++) hxp[ii+1][jj] = y_temp[ny+jj];
		//d_print_mat(1, nx, hxp[ii+1], 1);

		// add f to xp
		for(jj=0; jj<nx; jj++) hxp[ii+1][jj] += hf[ii][jj];
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		// initialize w_temp with 0
		for(jj=0; jj<nw; jj++) w_temp[jj] = 0.0;
		//d_print_mat(1, nw, w_temp, 1);
	
		// compute Q*q
		dsymv_lib(nw, nw, hpQ[ii], cnw, hq[ii], w_temp, -1);
		//d_print_mat(1, nw, w_temp, 1);

		// xp += G*w_temp
		dgemv_n_lib(nx, nw, hpG[ii], cnw, w_temp, hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);
	
		// xp += A*xe
		//dgemv_n_lib(nx, nx, hpA[ii], cnx, hxe[ii], hxp[ii+1], 1);
		//d_print_mat(1, nx, hxp[ii+1], 1);

		//if(ii==1)
		//exit(1);

		}
	
	// stage N

	// copy y
	for(jj=0; jj<ny; jj++) y_temp[jj] = - hy[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// copy xp
	for(jj=0; jj<nx; jj++) y_temp[ny+jj] = hxp[N][jj];
	//d_print_mat(1, nz, y_temp, 1);
	
	// compute y + R*r
	dsymv_lib(ny, ny, hpR[N], cny, hr[N], y_temp, -1);
	//d_print_mat(1, nz, y_temp, 1);

	// compute y + R*r - C*xp
	dgemv_n_lib(ny, nx, hpC[N], cnx, hxp[N], y_temp, 1);
	//d_print_mat(1, nz, y_temp, 1);

	//d_print_pmat(nz, ny, bs, hpLe[N], cnf);

	// compute xe
	dtrsv_dgemv_n_lib(ny, ny+nx, hpLe[N], cnf, y_temp);
	//d_print_mat(1, nz, y_temp, 1);

	// copy xe
	for(jj=0; jj<nx; jj++) hxe[N][jj] = y_temp[ny+jj];
	//d_print_mat(1, nx, hxe[N], 1);

	//exit(1);

	}
//#endif



//#if defined(TARGET_C99_4X4)
// version tailored for MHE
void d_ric_trf_mhe(int nx, int nw, int ny, int N, double **hpA, double **hpG, double **hpC, double **hpLp, double **hdLp, double **hpQ, double **hpR, double **hpLe, double *work)
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
	dsyttmm_lu_lib(nx, hpLp[0]+(nx+nw+pad)*bs, cnl, Pi_p, cnx);
	//d_print_pmat(nx, nx, bs, Pi_p, cnx);

	// copy /Pi_p on the bottom right block of Lam
	dtrma_lib(nx, ny, Pi_p, cnx, hpLe[0]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		// backup the top-left part of the bottom right block of Lam
		ptr1 = buffer;
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &hpLe[ii][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
		dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[ii], cny, hpLe[ii], cnf, 1);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// recover overwritten part of /Pi_p in bottom right part of Lam
		ptr1 = buffer;
		for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
			{
			ptr = &hpLe[ii][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
		dgetr_lib(ny, 0, nx, ny, CLLt, cnx, hpLe[ii]+(ny/bs)*bs*cnf+ny%bs, cnf);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// cholesky factorization of Lam
		dpotrf_lib(nz, nz, hpLe[ii], cnf, hpLe[ii], cnf, diag);
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
		//d_print_pmat(nx, nx, bs, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);

		// copy /Pi_p on the bottom right block of Lam
		dtrma_lib(nx, ny, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLe[ii+1]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf);
		//d_print_pmat(nz, nz, bs, Lam, cnz);

		// factorize Pi_p
		dpotrf_lib(nx, nx, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hdLp[ii+1]); //diag);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// transpose in place the lower cholesky factor of /Pi_p
		dtrtr_l_lib(nx, 0, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);	
		//d_print_pmat(nx, cnl, bs, hpLp[ii+1], cnl);
		//d_print_pmat(nx, nx, bs, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);


		//dsyttmm_lu_lib(nx, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, Pi_p, cnx);
		//d_print_pmat(nx, nx, bs, Pi_p, cnx);
		//exit(1);
		}

	// stage N

	// backup the top-left part of the bottom right block of Lam
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		ptr = &hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
	dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[N], cny, hpLe[N], cnf, 1);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// recover overwritten part of I in bottom right part of Lam
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		ptr = &hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
	dgetr_lib(ny, 0, nx, ny, CLLt, cnx, hpLe[N]+(ny/bs)*bs*cnf+ny%bs, cnf);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// cholesky factorization of Lam
	dpotrf_lib(nz, nz, hpLe[N], cnf, hpLe[N], cnf, diag);
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
// version tailored for MHE; explicitly computes estimates only at the last stage
void d_ric_trf_mhe_end(int nx, int nw, int ny, int N, double **hpCA, double **hpG, double **hpC, double **hpLp, double **hpQ, double **hpR, double **hpLe, double *work)
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

	const int pad = (ncl-(nx)%ncl)%ncl; // packing between AGL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

	int ii, jj, ll;
	double *ptr;

	ptr = work;

	double *Lam_w = ptr; // d_zeros_align(&Lam_w, pnw, cnw);
	ptr += pnw*cnw;
	
	double *GLam_w = ptr; // d_zeros_align(&Lam_w, pnw, cnw);
	ptr += pnx*cnw;
	
	double *GQGt = ptr; //d_zeros_align(&GQGt, pnx, cnx);
	ptr += pnx*cnx;

	double *diag = ptr; // d_zeros_align(&diag, anz, 1);
	ptr += anz;

	double *Pi_p = ptr; //  d_zeros_align(&Pi_p, pnx, cnx);
	ptr += pnx*cnx;

	double *CL = ptr; //d_zeros_align(&CL, pny, cnx);
	ptr += pny*cnx;

	double *CLLt = ptr; //d_zeros_align(&CLLt, pny, cnx);
	ptr += pny*cnx;

	static double buffer[6] = {};

	double *ptr1;

	// loop over horizon
	for(ii=0; ii<N; ii++)
		{

		// zero cross term in bottom left block
		for(jj=(ny/bs)*bs; jj<pnz; jj+=4)
			for(ll=0; ll<bs*ny; ll++)
				hpLp[ii+1][jj*cnl+(nx+pad)*bs+ll] = 0.0;

		// copy R in top left block
		d_copy_pmat_l(ny, bs, hpR[ii], cny, hpLp[ii+1]+(nx+pad)*bs, cnl);
		//d_print_pmat(nz, cnl, bs, hpLp[ii+1], cnl);
		//exit(1);

		// compute lower cholesky factor of Q
		dpotrf_lib(nw, nw, hpQ[ii], cnw, Lam_w, cnw, diag);
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// transpose in place the lower cholesky factor of Q
		dtrtr_l_lib(nw, 0, Lam_w, cnw, Lam_w, cnw);	
		//d_print_pmat(nw, nw, bs, Lam_w, cnw);

		// compute G*U', with U' upper cholesky factor of Q
		// d_print_pmat(nx, nw, bs, hpG[ii], cnw);
		dtrmm_l_lib(nx, nw, hpG[ii], cnw, Lam_w, cnw, GLam_w, cnw);
		//d_print_pmat(nx, nw, bs, GLam_w, cnw);

		// compute GQGt
		dsyrk_lib(nx, nx, nw, GLam_w, cnw, GLam_w, cnw, GQGt, cnx, GQGt, cnx, 0);
		//d_print_pmat(nx, nx, bs, GQGt, cnx);
		//d_print_pmat(nx, nx, bs, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);

		// copy GQGt on the bottom right block of Lam
		dtrma_lib(nx, ny, GQGt, cnx, hpLp[ii+1]+(ny/bs)*bs*cnz+ny%bs+(nx+pad+ny)*bs, cnl);
		//d_print_pmat(nz, cnl-1, bs, hpLp[ii+1], cnl);
	
		// compute CA*U', with U upper cholesky factor of /Pi_p
		//d_print_pmat(nz, cnl-1, bs, hpLp[ii], cnl);
		//d_print_pmat(nz, nx, bs, hpCA[ii], cnx);
		dtrmm_l_lib(nz, nx, hpCA[ii], cnx, hpLp[ii]+(nx+pad+ncl)*bs, cnl, hpLp[ii+1], cnl);
		//d_print_pmat(nz, cnl-1, bs, hpLp[ii+1], cnl);

		// compute Lp
		//dsyrk_dpotrf_lib(nz, nz, nx, hpLp[ii+1], cnl, hpLp[ii+1]+(nx+pad)*bs, cnl, diag, 1);
		dsyrk_dpotrf_lib(nz, nz, nx, hpLp[ii+1], cnl, hpLp[ii+1]+(nx+pad)*bs, cnl, hpLp[ii+1]+(nx+pad)*bs, cnl, diag, 1, 0);

		// inverted diagonal of top-left part of hpLe
		for(jj=0; jj<ny; jj++) hpLp[ii+1][(jj/bs)*bs*cnl+jj%bs+(nx+pad+jj)*bs] = diag[jj];

		// transpose and align Lp
		dtrtr_l_lib(nx, ny, hpLp[ii+1]+(ny/bs)*bs*cnl+ny%bs+(nx+pad+ny)*bs, cnl, hpLp[ii+1]+(nx+pad+ncl)*bs, cnl);	
		//d_print_pmat(nz, cnl-1, bs, hpLp[ii+1], cnl);

		//exit(1);
		}

	//exit(1);

	// stage N

	//d_print_pmat(nx, nx, bs, hpLp[N]+(nx+pad+ncl)*bs, cnl);
	dtrtr_u_lib(nx, hpLp[N]+(nx+pad+ncl)*bs, cnl, GQGt, cnx);
	//d_print_pmat(nx, nx, bs, GQGt, cnx);

	dsyttmm_lu_lib(nx, GQGt, cnx, Pi_p, cnx);
	//d_print_pmat(nx, nx, bs, Pi_p, cnx);

	// copy /Pi_p on the bottom right block of Lam
	//dtrma_lib(nx, ny, Pi_p, cnx, Lam+(ny/bs)*bs*cnz+ny%bs+ny*bs, cnz);
	dtrma_lib(nx, ny, Pi_p, cnx, hpLe[N]+(ny/bs)*bs*cnz+ny%bs+ny*bs, cnf);
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);

	// backup the top-left part of the bottom right block of Lam
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		//ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
		ptr = &hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
	dtrmm_l_lib(ny, nx, hpC[N], cnx, hpLp[N]+(nx+pad+ncl)*bs, cnl, CL, cnx);
	//d_print_pmat(ny, nx, bs, CL, cnx);

	// compute R + (C*U')*(C*U')' on the top left of hpLe
	dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[N], cny, hpLe[N], cnf, 1);
	//dsyrk_lib(ny, ny, nx, CL, cnx, CL, cnx, hpR[N], cny, Lam, cnz, 1);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

	// recover overwritten part of I in bottom right part of hpLe
	ptr1 = buffer;
	for(jj=ny; jj<((ny+bs-1)/bs)*bs; jj+=1)
		{
		//ptr = &Lam[(jj/bs)*bs*cnz+jj%bs+jj*bs];
		ptr = &hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs];
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
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);

	// compute C*U'*L'
	dtrmm_u_lib(ny, nx, CL, cnx, GQGt, cnx, CLLt, cnx);
	//d_print_pmat(ny, nx, bs, CLLt, cnx);

	// copy C*U'*L' on the bottom left of hpLe
	dgetr_lib(ny, 0, nx, ny, CLLt, cnx, hpLe[N]+(ny/bs)*bs*cnf+ny%bs, cnf);
	//d_print_pmat(nz, nz, bs, Lam, cnz);
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);

	// cholesky factorization of hpLe
	dpotrf_lib(nz, nz, hpLe[N], cnf, hpLe[N], cnf, diag);
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);
	//d_print_pmat(nz, nz, bs, Lam, cnz);
	//d_print_mat(nz, 1, diag, 1);

	// inverted diagonal of top-left part of hpLe
	for(jj=0; jj<ny; jj++) hpLe[N][(jj/bs)*bs*cnf+jj%bs+jj*bs] = diag[jj];

	// transpose and align /Pi_e
	dtrtr_l_lib(nx, ny, hpLe[N]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, hpLe[N]+ncl*bs, cnf);	
	//d_print_pmat(nz, nz, bs, hpLe[N], cnf);

	//dpotrf_lib(nz, nz, Lam, cnz, Lam, cnz, diag);
	//d_print_pmat(nz, nz, bs, Lam, cnz);

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
		//dsyrk_dpotrf_lib(nx, nx, nx+nw, hpLp[ii+1], cnl, hpLp[ii+1], cnl, diag, 0);
		dsyrk_dpotrf_lib(nx, nx, nx+nw, hpLp[ii+1], cnl, hpLp[ii+1], cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, diag, 0, 0);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[ii+1], cnl);

		// transpose in place the lower cholesky factor of /Pi_p
		dtrtr_l_lib(nx, 0, hpLp[ii+1]+(nx+nw+pad)*bs, cnl, hpLp[ii+1]+(nx+nw+pad)*bs, cnl);	
		//d_print_pmat(nx, cnl, bs, hpLp[ii+1], cnl);

		//exit(1);
		}

	//exit(1);

	}
//#endif





