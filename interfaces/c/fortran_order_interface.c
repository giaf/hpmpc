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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../include/target.h"
#include "../../include/block_size.h"
#include "../../include/aux_d.h"
#include "../../include/aux_s.h"
#include "../../include/blas_d.h"
#include "../../include/lqcp_solvers.h"
#include "../../include/mpc_solvers.h"

// Debug flag
#ifndef PC_DEBUG
#define PC_DEBUG 0
#endif /* PC_DEBUG */



int fortran_order_d_ip_ocp_hard_tv( 
							int *kk, int k_max, double mu0, double mu_tol,
							int N, int *nx, int *nu, int *nb, int *ng,
							int warm_start,
							double **A, double **B, double **b, 
							double **Q, double **S, double **R, double **q, double **r, 
							double **lb, double **ub,
							double **C, double **D, double **lg, double **ug,
							double **x, double **u, double **pi, double **lam, double **t,
							double *inf_norm_res,
							double *work0, 
							double *stat)

	{

//printf("\nstart of wrapper\n");

	int hpmpc_status = -1;

	int ii, jj, ll, nbu;

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cng[N+1];

	for(ii=0; ii<=N; ii++)
		{
		pnx[ii] = (nx[ii]+bs-1)/bs*bs;
		pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
		pnb[ii] = (nb[ii]+bs-1)/bs*bs;
		png[ii] = (ng[ii]+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;
		}



	double alpha_min = 1e-8; // minimum accepted step length
	double temp;
	
	int info = 0;




	// time-variant quantities
	int *ptr_int;
	ptr_int = (int *) work0;

	int *idxb[N+1];

	idxb[0] = ptr_int;
	ptr_int += nb[0];
	for(jj=0; jj<nb[0]; jj++) idxb[0][jj] = jj;
	for(ii=1; ii<N; ii++)
		{
		idxb[ii] = ptr_int;
		ptr_int += nb[ii];
		for(jj=0; jj<nb[ii]; jj++) idxb[ii][jj] = jj;
		}
	idxb[N] = ptr_int;
	ptr_int += nb[N];
	for(jj=0; jj<nb[N]; jj++) idxb[N][jj] = nu[N]+jj;

	work0 = (double *) ptr_int;



//printf("\n%d\n", ((size_t) work0) & 63);

	/* align work space */
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	double *ptr = (double *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

	/* array or pointers */
	double *hpBAbt[N];
	double *hb[N];
	double *hpQ[N+1];
	double *hq[N+1];
	double *hpDCt[N+1];
	double *hd[N+1];
	double *hux[N+1];
	double *hpi[N];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrb[N];
	double *hrq[N+1];
	double *hrd[N+1];
	double *work;



	for(ii=0; ii<N; ii++)
		{
		hpBAbt[ii] = ptr;
		ptr += pnz[ii]*cnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hpDCt[ii] = ptr;
		ptr += pnz[ii]*cng[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hpQ[ii] = ptr;
		ptr += pnz[ii]*cnux[ii];
		}

	work = ptr;
	ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nx, nu, nb, ng)/sizeof(double);

	for(ii=0; ii<N; ii++)
		{
		hb[ii] = ptr;
		ptr += pnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hq[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hd[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hux[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<N; ii++)
		{
		hpi[ii] = ptr;
		ptr += pnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hlam[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		ht[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<N; ii++)
		{
		hrb[ii] = ptr;
		ptr += pnx[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hrq[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hrd[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}



	/* pack matrices */

	// TODO use pointers to exploit time invariant !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// dynamic system
	for(ii=0; ii<N; ii++)
		{
		d_cvt_tran_mat2pmat(nx[ii+1], nu[ii], B[ii], nx[ii+1], 0, hpBAbt[ii], cnx[ii+1]);
		d_cvt_tran_mat2pmat(nx[ii+1], nx[ii], A[ii], nx[ii+1], nu[ii], hpBAbt[ii]+nu[ii]/bs*cnx[ii+1]*bs+nu[ii]%bs, cnx[ii+1]);
		for(jj=0; jj<nx[ii+1]; jj++)
			{
			hb[ii][jj] = b[ii][jj];
			hpBAbt[ii][(nx[ii]+nu[ii])/bs*cnx[ii+1]*bs+(nx[ii]+nu[ii])%bs+jj*bs] = b[ii][jj];
			}
		}
//	for(ii=0; ii<N; ii++)
//		d_print_pmat(nu[ii]+nx[ii]+1, nx[ii+1], bs, hpBAbt[ii], cnx[ii+1]);
//	exit(1);

	// general constraints
	for(ii=0; ii<N; ii++)
		{
		d_cvt_tran_mat2pmat(ng[ii], nu[ii], D[ii], ng[ii], 0, hpDCt[ii], cng[ii]);
		d_cvt_tran_mat2pmat(ng[ii], nx[ii], C[ii], ng[ii], nu[ii], hpDCt[ii]+nu[ii]/bs*cng[ii]*bs+nu[ii]%bs, cng[ii]);
		}
	ii = N;
	d_cvt_tran_mat2pmat(ng[ii], nx[ii], C[ii], ng[ii], 0, hpDCt[ii], cng[ii]);
//	for(ii=0; ii<=N; ii++)
//		d_print_pmat(nu[ii]+nx[ii], ng[ii], bs, hpDCt[ii], cng[ii]);
//	exit(1);

	// cost function
	for(ii=0; ii<N; ii++)
		{
		d_cvt_mat2pmat(nu[ii], nu[ii], R[ii], nu[ii], 0, hpQ[ii], cnux[ii]);
		d_cvt_tran_mat2pmat(nu[ii], nx[ii], S[ii], nu[ii], nu[ii], hpQ[ii]+nu[ii]/bs*cnux[ii]*bs+nu[ii]%bs, cnux[ii]);
		d_cvt_mat2pmat(nx[ii], nx[ii], Q[ii], nx[ii], nu[ii], hpQ[ii]+nu[ii]/bs*cnux[ii]*bs+nu[ii]%bs+nu[ii]*bs, cnux[ii]);
		for(jj=0; jj<nu[ii]; jj++)
			{
			hq[ii][jj] = r[ii][jj];
			hpQ[ii][(nx[ii]+nu[ii])/bs*cnux[ii]*bs+(nx[ii]+nu[ii])%bs+jj*bs] = r[ii][jj];
			}
		for(jj=0; jj<nx[ii]; jj++)
			{
			hq[ii][nu[ii]+jj] = q[ii][jj];
			hpQ[ii][(nx[ii]+nu[ii])/bs*cnux[ii]*bs+(nx[ii]+nu[ii])%bs+(nu[ii]+jj)*bs] = q[ii][jj];
			}
		}
	ii = N;
	d_cvt_mat2pmat(nx[ii], nx[ii], Q[ii], nx[ii], 0, hpQ[ii], cnux[ii]);
	for(jj=0; jj<nx[ii]; jj++)
		{
		hq[ii][nu[ii]+jj] = q[ii][jj];
		hpQ[ii][nx[ii]/bs*cnux[ii]*bs+nx[ii]%bs+jj*bs] = q[ii][jj];
		}
//	for(ii=0; ii<=N; ii++)
//		d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, hpQ[ii], cnux[ii]);
//	exit(1);

	// estimate mu0 if not user-provided
//	printf("%f\n", mu0);
	if(mu0<=0)
		{
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nu[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, R[ii][jj*nu[ii]+ll]);
			for(jj=0; jj<nx[ii]*nu[ii]; jj++) mu0 = fmax(mu0, S[ii][jj]);
			for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, Q[ii][jj*nx[ii]+ll]);
			for(jj=0; jj<nu[ii]; jj++) mu0 = fmax(mu0, r[ii][jj]);
			for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, q[ii][jj]);
			}
		ii=N;
		for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, Q[ii][jj*nx[ii]+ll]);
		for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, q[ii][jj]);
		}
//	printf("%f\n", mu0);
//	exit(1);

	// input box constraints
	for(ii=0; ii<N; ii++)
		{
		nbu = nb[ii]<nu[ii] ? nb[ii] : nu[ii];
		for(jj=0; jj<nbu; jj++)
			{
			if(lb[ii][jj]!=ub[ii][jj]) // equality constraint
				{
				hd[ii][jj+0]       = lb[ii][jj];
				hd[ii][jj+pnb[ii]] = ub[ii][jj];
				}
			else
				{
				for(ll=0; ll<nx[ii+1]; ll++)
					{
					// update linear term
					hpBAbt[ii][(nx[ii]+nu[ii])/bs*cnx[ii+1]*bs+(nx[ii]+nu[ii])%bs+ll*bs] += hpBAbt[ii][jj/bs*cnx[ii+1]*bs+jj%bs+ll*bs]*lb[ii][jj];
					// zero corresponding B column
					hpBAbt[ii][jj/bs*cnx[ii+1]*bs+jj%bs+ll*bs] = 0.0;
					}
				
				// inactive box constraints
				hd[ii][jj+0]       = lb[ii][jj] + 1e3;
				hd[ii][jj+pnb[ii]] = ub[ii][jj] - 1e3;

				}
			}
		}
	// state box constraints 
	for(ii=0; ii<=N; ii++)
		{
		for(jj=nu[ii]; jj<nb[ii]; jj++)
			{
			hd[ii][jj+0]       = lb[ii][jj];
			hd[ii][jj+pnb[ii]] = ub[ii][jj];
			}
		}
	// general constraints
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++)
			{
			hd[ii][2*pnb[ii]+jj+0]       = lg[ii][jj];
			hd[ii][2*pnb[ii]+jj+png[ii]] = ug[ii][jj];
			}
		}
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, 2*pnb[ii]+2*png[ii], hd[ii], 1);
//	exit(1);



	// initial guess 
	if(warm_start)
		{

		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nu[ii]; jj++)
				hux[ii][jj] = u[ii][jj];

		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx[ii]; jj++)
				hux[ii][nu[ii]+jj] = x[ii][jj];

		}
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
//	exit(1);



	// call the IP solver
	hpmpc_status = d_ip2_mpc_hard_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, idxb, ng, hpBAbt, hpQ, hpDCt, hd, hux, 1, hpi, hlam, ht, work);
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
//	exit(1);



	// copy back inputs and states
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]; jj++)
			u[ii][jj] = hux[ii][jj];

	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx[ii]; jj++)
			x[ii][jj] = hux[ii][nu[ii]+jj];



	// check for input equality constraints
	for(ii=0; ii<N; ii++)
		{
		nbu = nb[ii]<nu[ii] ? nb[ii] : nu[ii];
		for(jj=0; jj<nbu; jj++)
			{
			if(lb[ii][jj]==ub[ii][jj]) // equality constraint
				{
				u[ii][jj] = lb[ii][jj];
				}
			}
		}



	// compute infinity norm of residuals on exit

	double mu;

	d_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, hpBAbt, hb, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

	temp = fabs(hrq[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]+nx[ii]; jj++) 
			temp = fmax( temp, fabs(hrq[ii][jj]) );
	ii = N;
	for(jj=0; jj<nx[ii]; jj++) 
		temp = fmax( temp, fabs(hrq[ii][jj]) );
	inf_norm_res[0] = temp;

	temp = fabs(hrb[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++) 
			temp = fmax( temp, fabs(hrb[ii][jj]) );
	inf_norm_res[1] = temp;

	temp = fabs(hrd[0][0]);
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][jj+pnb[ii]]) );
			}
		}
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+png[ii]]) );
			}
		}
	inf_norm_res[2] = temp;

	inf_norm_res[3] = mu;



	// copy back multipliers

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++)
			pi[ii][jj] = hpi[ii][jj];

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++)
			{
			lam[ii][jj+0]      = hlam[ii][jj+0];
			lam[ii][jj+nb[ii]] = hlam[ii][jj+pnb[ii]];
			t[ii][jj+0]      = ht[ii][jj+0];
			t[ii][jj+nb[ii]] = ht[ii][jj+pnb[ii]];
			}
		}

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++)
			{
			lam[ii][2*nb[ii]+jj+0]      = hlam[ii][2*pnb[ii]+jj+0];
			lam[ii][2*nb[ii]+jj+ng[ii]] = hlam[ii][2*pnb[ii]+jj+png[ii]];
			t[ii][2*nb[ii]+jj+0]      = ht[ii][2*pnb[ii]+jj+0];
			t[ii][2*nb[ii]+jj+ng[ii]] = ht[ii][2*pnb[ii]+jj+png[ii]];
			}
		}

//	printf("\nend of wrapper\n");

    return hpmpc_status;

	}



void fortran_order_d_solve_kkt_new_rhs_ocp_hard_tv(
							int N, int *nx, int *nu, int *nb, int *ng,
							double **A, double **B, double **b, 
							double **Q, double **S, double **R, double **q, double **r, 
							double **lb, double **ub,
							double **C, double **D, double **lg, double **ug,
							double tau,
							double **x, double **u, double **pi, double **lam, double **t,
							double *inf_norm_res,
							double *work0) 

	{

//printf("\nstart of wrapper\n");

	int hpmpc_status = -1;

	int ii, jj, ll, nbu;

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];
	int cng[N+1];

	for(ii=0; ii<=N; ii++)
		{
		pnx[ii] = (nx[ii]+bs-1)/bs*bs;
		pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
		pnb[ii] = (nb[ii]+bs-1)/bs*bs;
		png[ii] = (ng[ii]+bs-1)/bs*bs;
		cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
		cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;
		}



	double temp;
	
	int info = 0;




	// time-variant quantities
	int *ptr_int;
	ptr_int = (int *) work0;

	int *(idxb[N+1]);

	idxb[0] = ptr_int;
	ptr_int += nb[0];
	for(jj=0; jj<nb[0]; jj++) idxb[0][jj] = jj;
	for(ii=1; ii<N; ii++)
		{
		idxb[ii] = ptr_int;
		ptr_int += nb[ii];
		for(jj=0; jj<nb[ii]; jj++) idxb[ii][jj] = jj;
		}
	idxb[N] = ptr_int;
	ptr_int += nb[N];
	for(jj=0; jj<nb[N]; jj++) idxb[N][jj] = nu[N]+jj;

	work0 = (double *) ptr_int;



//printf("\n%d\n", ((size_t) work0) & 63);

	/* align work space */
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	double *ptr = (double *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

	/* array or pointers */
	double *hpBAbt[N];
	double *hb[N];
	double *hpQ[N+1];
	double *hq[N+1];
	double *hpDCt[N+1];
	double *hd[N+1];
	double *hux[N+1];
	double *hpi[N];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrb[N];
	double *hrq[N+1];
	double *hrd[N+1];
	double *work;



	for(ii=0; ii<N; ii++)
		{
		hpBAbt[ii] = ptr;
		ptr += pnz[ii]*cnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hpDCt[ii] = ptr;
		ptr += pnz[ii]*cng[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hpQ[ii] = ptr;
		ptr += pnz[ii]*cnux[ii];
		}

	work = ptr;
	ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nx, nu, nb, ng)/sizeof(double);

	for(ii=0; ii<N; ii++)
		{
		hb[ii] = ptr;
		ptr += pnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hq[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hd[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hux[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<N; ii++)
		{
		hpi[ii] = ptr;
		ptr += pnx[ii+1];
		}

	for(ii=0; ii<=N; ii++)
		{
		hlam[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		ht[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}

	for(ii=0; ii<N; ii++)
		{
		hrb[ii] = ptr;
		ptr += pnx[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hrq[ii] = ptr;
		ptr += pnz[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hrd[ii] = ptr;
		ptr += 2*pnb[ii]+2*png[ii];
		}



	/* pack matrices */

	// TODO use pointers to exploit time invariant !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// dynamic system
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nx[ii+1]; jj++)
			{
			hb[ii][jj] = b[ii][jj];
			}
		}

	// cost function
	for(ii=0; ii<N; ii++)
		{
		for(jj=0; jj<nu[ii]; jj++)
			hq[ii][jj] = r[ii][jj];
		for(jj=0; jj<nx[ii]; jj++)
			hq[ii][nu[ii]+jj] = q[ii][jj];
		}
	ii = N;
	for(jj=0; jj<nx[ii]; jj++)
		hq[ii][nu[ii]+jj] = q[ii][jj];

	// input box constraints
	for(ii=0; ii<N; ii++)
		{
		nbu = nb[ii]<nu[ii] ? nb[ii] : nu[ii];
		for(jj=0; jj<nbu; jj++)
			{
			hd[ii][jj+0]       = lb[ii][jj];
			hd[ii][jj+pnb[ii]] = ub[ii][jj];
			}
		}
	// state box constraints 
	for(ii=0; ii<=N; ii++)
		{
		for(jj=nu[ii]; jj<nb[ii]; jj++)
			{
			hd[ii][jj+0]       = lb[ii][jj];
			hd[ii][jj+pnb[ii]] = ub[ii][jj];
			}
		}
	// general constraints
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++)
			{
			hd[ii][2*pnb[ii]+jj+0]       = lg[ii][jj];
			hd[ii][2*pnb[ii]+jj+png[ii]] = ug[ii][jj];
			}
		}



	// initial guess TODO remove
//	if(warm_start)
//		{
//		for(ii=0; ii<N; ii++)
//			for(jj=0; jj<nu[ii]; jj++)
//				hux[ii][jj] = u[ii][jj];
//		for(ii=0; ii<=N; ii++)
//			for(jj=0; jj<nx[ii]; jj++)
//				hux[ii][nu[ii]+jj] = x[ii][jj];
//		}



	// call the IP solver
//	hpmpc_status = d_ip2_mpc_hard_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nx, nu, nb, idxb, ng, hpBAbt, hpQ, hpDCt, hd, hux, 1, hpi, hlam, ht, work);
	d_kkt_solve_new_rhs_mpc_hard_tv(N, nx, nu, nb, idxb, ng, hpBAbt, hb, hpQ, hq, hpDCt, hd, tau, hux, 1, hpi, hlam, ht, work); // TODO B 
//	for(ii=0; ii<=N; ii++)
//		d_print_mat(1, nu[ii]+nx[ii], hux[ii], 1);
//	exit(1);



	// copy back inputs and states
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]; jj++)
			u[ii][jj] = hux[ii][jj];

	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx[ii]; jj++)
			x[ii][jj] = hux[ii][nu[ii]+jj];



	// compute infinity norm of residuals on exit

	double mu;

	d_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, hpBAbt, hb, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

#if 0
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nu[ii]+nx[ii], hrq[ii], 1);
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nx[ii+1], hrb[ii], 1);
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, 2*pnb[ii]+2*png[ii], hrd[ii], 1);
	d_print_mat(1, 1, &mu, 1);
	exit(1);
#endif

	temp = fabs(hrq[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]+nx[ii]; jj++) 
			temp = fmax( temp, fabs(hrq[ii][jj]) );
	ii = N;
	for(jj=0; jj<nx[ii]; jj++) 
		temp = fmax( temp, fabs(hrq[ii][jj]) );
	inf_norm_res[0] = temp;

	temp = fabs(hrb[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++) 
			temp = fmax( temp, fabs(hrb[ii][jj]) );
	inf_norm_res[1] = temp;

	temp = fabs(hrd[0][0]);
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][jj+pnb[ii]]) );
			}
		}
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+0]) );
			temp = fmax( temp, fabs(hrd[ii][2*pnb[ii]+jj+png[ii]]) );
			}
		}
	inf_norm_res[2] = temp;

	inf_norm_res[3] = mu;



	// copy back multipliers

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++)
			pi[ii][jj] = hpi[ii][jj];

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++)
			{
			lam[ii][jj+0]      = hlam[ii][jj+0];
			lam[ii][jj+nb[ii]] = hlam[ii][jj+pnb[ii]];
			t[ii][jj+0]      = ht[ii][jj+0];
			t[ii][jj+nb[ii]] = ht[ii][jj+pnb[ii]];
			}
		}

	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<ng[ii]; jj++)
			{
			lam[ii][2*nb[ii]+jj+0]      = hlam[ii][2*pnb[ii]+jj+0];
			lam[ii][2*nb[ii]+jj+ng[ii]] = hlam[ii][2*pnb[ii]+jj+png[ii]];
			t[ii][2*nb[ii]+jj+0]      = ht[ii][2*pnb[ii]+jj+0];
			t[ii][2*nb[ii]+jj+ng[ii]] = ht[ii][2*pnb[ii]+jj+png[ii]];
			}
		}

//	printf("\nend of wrapper\n");

    return;

	}



int fortran_order_d_ip_mpc_hard_tv( 
							int *kk, int k_max, double mu0, double mu_tol,
							int N, int nx, int nu, int nb, int ng, int ngN, 
							int time_invariant, int free_x0, int warm_start,
							double* A, double* B, double* b, 
							double* Q, double* Qf, double* S, double* R, 
							double* q, double* qf, double* r, 
							double *lb, double *ub,
							double *C, double *D, double *lg, double *ug,
							double *Cf, double *lgf, double *ugf,
							double* x, double* u, double *pi, double *lam, double *t,
							double *inf_norm_res,
							double *work0, 
							double *stat)

	{

//printf("\nstart of wrapper\n");

	//printf("\n%d %d %d %d %d %d\n", N, nx, nu, nb, ng, ngN);

	int hpmpc_status = -1;

	const int nbu = nb<nu ? nb : nu ;

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int nz   = nx+nu+1;
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnu  = bs*((nu+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int pnb  = bs*((nb+bs-1)/bs);
	const int png  = bs*((ng+bs-1)/bs);
	const int pngN = bs*((ngN+bs-1)/bs);
	const int cnux = (nu+nx+ncl-1)/ncl*ncl;
	const int cnu  = (nu+ncl-1)/ncl*ncl;
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);

	int pnb0;
	int png0;


	double alpha_min = 1e-8; // minimum accepted step length
	double temp;
	
	int info = 0;

	int i, ii, jj, ll;



	// time-variant quantities
	int *ptr_int;
	ptr_int = (int *) work0;

	int nxx[N+1];
	int nuu[N+1];
	int nbb[N+1];
	int ngg[N+1];
	int *(idxb[N+1]);

	nxx[0] = 0;
	nuu[0] = nu;
	nbb[0] = nbu;
	idxb[0] = ptr_int;
	ptr_int += nbb[0];
	for(jj=0; jj<nbb[0]; jj++) idxb[0][jj] = jj;
	ngg[0] = ng;
	for(ii=1; ii<N; ii++)
		{
		nxx[ii] = nx;
		nuu[ii] = nu;
		nbb[ii] = nb;
		idxb[ii] = ptr_int;
		ptr_int += nbb[ii];
		for(jj=0; jj<nbb[ii]; jj++) idxb[ii][jj] = jj;
		ngg[ii] = ng;
		}
	nxx[N] = nx;
	nuu[N] = 0;
	nbb[N] = nb-nu>0 ? nb-nu : 0;
	idxb[N] = ptr_int;
	ptr_int += nbb[N];
	for(jj=0; jj<nbb[N]; jj++) idxb[N][jj] = nuu[N]+jj;
	ngg[N] = ngN;

	work0 = (double *) ptr_int;



//printf("\n%d\n", ((size_t) work0) & 63);

	/* align work space */
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	double *ptr = (double *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

	/* array or pointers */
	double *hpBAbt[N];
	double *hb[N];
	double *hpQ[N+1];
	double *hq[N+1];
	double *hpDCt[N+1];
	double *hd[N+1];
	double *hux[N+1];
	double *hpi[N];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrb[N];
	double *hrq[N+1];
	double *hrd[N+1];
	double *work;



	if(time_invariant) // TODO time-invariant work space ?????
		{

		hpBAbt[0] = ptr;
		ptr += pnz*cnx;
		for(ii=1; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			}
		ptr += pnz*cnx;

		if(ng>0)
			{
			hpDCt[0] = ptr;
			ptr += pnz*cng;
			for(ii=1; ii<N; ii++)
				{
				hpDCt[ii] = ptr;
				}
			ptr += pnz*cng;
			}
		if(ngN>0)
			{
			hpDCt[N] = ptr;
			ptr += pnz*cngN;
			}

		hpQ[0] = ptr;
		ptr += pnz*cnux;
		for(ii=1; ii<N; ii++) // time variant and copied again internally in the IP !!!
			{
			hpQ[ii] = ptr;
			}
		ptr += pnz*cnux;
		hpQ[N] = ptr;
		ptr += pnz*cnux;

		work = ptr;
		ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nxx, nuu, nbb, ngg)/sizeof(double);

		hb[0] = ptr;
		ptr += pnx;
		for(ii=1; ii<N; ii++)
			{
			hb[ii] = ptr;
			}
		ptr += pnx;

		hq[0] = ptr;
		ptr += pnz;
		for(ii=1; ii<N; ii++)
			{
			hq[ii] = ptr;
			}
		ptr += pnz;
		hq[N] = ptr;
		ptr += pnz;

		hd[0] = ptr;
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		for(ii=1; ii<N; ii++) // time Variant box constraints
			{
			hd[ii] = ptr;
			}
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hpi[ii] = ptr;
			ptr += pnx; // for alignment of ptr
			}

		for(ii=0; ii<N; ii++) // time variant Lagrangian multipliers
			{
			hlam[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		hlam[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++) // time variant slack variables
			{
			ht[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		ht[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++)
			{
			hrb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hrq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hrd[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hrd[N] = ptr;
		ptr += 2*pnb+2*pngN;



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx); // pack A into (temporary) buffer
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hb[0]); // result

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
		for (jj = 0; jj<nx; jj++)
			hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hb[0][jj];

		// middle stages
		ii=1;
		if(jj<N)
			{
			d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, A, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for(jj=0; jj<nx; jj++)
				{
				hb[ii][jj] = b[jj];
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[jj];
				}
			}
		//d_print_mat(nx, nu, B, nx);
		//d_print_mat(nx, nx, A, nx);
		//d_print_pmat(nz, nx, bs, hpBAbt[0], cnx);
		//d_print_pmat(nz, nx, bs, hpBAbt[N-1], cnx);
		//exit(1);

		// general constraints
		if(ng>0)
			{
			// first stage
			ii=0;
			d_cvt_tran_mat2pmat(ng, nu, D, ng, 0, hpDCt[0], cng);
			// middle stages
			ii=1;
			if(jj<N)
				{
				d_cvt_tran_mat2pmat(ng, nu, D, ng, 0, hpDCt[ii], cng);
				d_cvt_tran_mat2pmat(ng, nx, C, ng, nu, hpDCt[ii]+nu/bs*cng*bs+nu%bs, cng);
				}
			}
		// last stage
		if(ngN>0)
			{
			//for(ii=0; ii<pnu*cngN; ii++) hpDCt[N][ii] = 0.0; // make sure D is zero !!!!!
			//d_cvt_tran_mat2pmat(ngN, nx, nu, bs, Cf, ngN, hpDCt[N]+nu/bs*cngN*bs+nu%bs, cngN);
			d_cvt_tran_mat2pmat(ngN, nx, Cf, ngN, 0, hpDCt[N], cngN);
			}
		//d_print_mat(ngN, nx, Cf, ngN);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ngN, bs, hpDCt[N], cngN);
		//exit(1);
		//printf("\n%d %d\n", nb, ng);
		//d_print_mat(ng, nx, C, ng);
		//d_print_mat(ng, nx, C+nx*ng, ng);
		//d_print_mat(ng, nu, D, ng);
		//d_print_mat(ng, nu, D+nu*ng, ng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[N], cng);
		//exit(1);

		// cost function
		// first stage
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nu, nx, S, nu, 0, hpQ[0], cnx); // pack S into a (temporary) buffer
		dgemv_n_lib(nu, nx, hpQ[0], cnx, hux[1], 1, r, hpi[1]); // result in (temporary) buffer

		jj = 0;
		d_cvt_mat2pmat(nu, nu, R, nu, 0, hpQ[0], cnu);
		for(ii=0; ii<nu; ii++)
			hpQ[0][(nu)/bs*cnu*bs+(nu)%bs+ii*bs] = hpi[1][ii];

		// middle stages
		jj=1;
		if(jj<N)
			{
			d_cvt_mat2pmat(nu, nu, R, nu, 0, hpQ[jj], cnux);
			d_cvt_tran_mat2pmat(nu, nx, S, nu, nu, hpQ[jj]+nu/bs*cnux*bs+nu%bs, cnux);
			d_cvt_mat2pmat(nx, nx, Q, nx, nu, hpQ[jj]+nu/bs*cnux*bs+nu%bs+nu*bs, cnux);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnux*bs+(nx+nu)%bs+ii*bs] = r[ii];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnux*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii];
			}

		// last stage
		d_cvt_mat2pmat(nx, nx, Qf, nx, 0, hpQ[N], cnx);
		for(jj=0; jj<nx; jj++)
			hpQ[N][nx/bs*cnx*bs+nx%bs+jj*bs] = qf[jj];

		// estimate mu0 if not user-provided
		if(mu0<=0)
			{
			// first stage
			jj=0;
			for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, abs(R[jj*nu*nu+ii*nu+ll]));
			for(ii=0; ii<nu; ii++) mu0 = fmax(mu0, abs(r[jj*nu+ii]));
			// middle stages
			jj=1;
			if(jj<N)
				{
				for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, abs(R[jj*nu*nu+ii*nu+ll]));
				for(ii=0; ii<nx*nu; ii++) mu0 = fmax(mu0, abs(S[jj*nu*nx+ii]));
				for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, abs(Q[jj*nx*nx+ii*nx+ll]));
				for(ii=0; ii<nu; ii++) mu0 = fmax(mu0, abs(r[jj*nu+ii]));
				for(ii=0; ii<nx; ii++) mu0 = fmax(mu0, abs(q[jj*nx+ii]));
				}
			// last stage
			for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, abs(Qf[ii*nx+ll]));
			for(jj=0; jj<nx; jj++) mu0 = fmax(mu0, abs(qf[ii]));
			}

		//d_print_pmat(nz, nz, bs, hpQ[0], cnu);
		//d_print_pmat(nz, nz, bs, hpQ[1], cnux);
		//d_print_pmat(nz, nz, bs, hpQ[N], cnx);
		//exit(1);

		// input constraints
		jj=0;
		pnb0 = (nbb[jj]+bs-1)/bs*bs;
		for(ii=0; ii<nbu; ii++)
			{
			if(lb[ii]!=ub[ii]) // equality constraint
				{
				hd[jj][ii+0]    = lb[ii];
				hd[jj][ii+pnb0] = ub[ii];
				}
			else
				{
				for(ll=0; ll<nx; ll++)
					{
					// update linear term
					hpBAbt[jj][(nxx[jj]+nuu[jj])/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii];
					// zero corresponding B column
					hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
					}
				
				// inactive box constraints
				hd[jj][ii+0]    = lb[ii] + 1e3;
				hd[jj][ii+pnb0] = ub[ii] - 1e3;

				}
			}
		jj=1;
		if(jj<N)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii]!=ub[ii]) // equality constraint
					{
					hd[jj][ii+0]    = lb[ii];
					hd[jj][ii+pnb0] = ub[ii];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nxx[jj]+nuu[jj])/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hd[jj][ii+0]    = lb[ii] + 1e3;
					hd[jj][ii+pnb0] = ub[ii] - 1e3;

					}
				}
			}
		// state constraints 
		jj=1;
		if(jj<N)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=nuu[jj]; ii<nbb[jj]; ii++)
				{
				hd[jj][ii+0]    = lb[ii];
				hd[jj][ii+pnb0] = ub[ii];
				//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    = lb[nu+ii];
			hd[N][ii+pnb0] = ub[nu+ii];
			//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				pnb0 = (nbb[jj]+bs-1)/bs*bs;
				png0 = (ngg[jj]+bs-1)/bs*bs;
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb0+ii+0]    = lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    = lgf[ii];
				hd[N][2*pnb0+ii+png0] = ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

		} // end of time invariant
	else // time variant
		{

		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		if(ng>0)
			{
			for(ii=0; ii<N; ii++)
				{
				hpDCt[ii] = ptr;
				ptr += pnz*cng;
				}
			}
		if(ngN>0)
			{
			hpDCt[N] = ptr;
			ptr += pnz*cngN;
			}

		for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnux; // TODO use cnux instrad
			}

		work = ptr;
		ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nxx, nuu, nbb, ngg)/sizeof(double);

		for(ii=0; ii<N; ii++)
			{
			hb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hd[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN;

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hpi[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<N; ii++)
			{
			hlam[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hlam[N] = ptr;
		ptr += 2*pnb+2*pngN;

		for(ii=0; ii<N; ii++)
			{
			ht[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		ht[N] = ptr;
		ptr += 2*pnb+2*pngN;

		for(ii=0; ii<N; ii++)
			{
			hrb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hrq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hrd[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hrd[N] = ptr;
		ptr += 2*pnb+2*pngN;



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx);
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hb[0]);

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
		for (jj = 0; jj<nx; jj++)
			hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hb[0][jj];

		// middle stages
		for(ii=1; ii<N; ii++)
			{
			d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				{
				hb[ii][jj] = b[ii*nx+jj];
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
				}
			}
		//d_print_mat(nx, nu, B, nx);
		//d_print_mat(nx, nx, A, nx);
		//d_print_pmat(nz, nx, bs, hpBAbt[0], cnx);
		//d_print_pmat(nz, nx, bs, hpBAbt[N-1], cnx);
		//exit(1);

		// general constraints
		if(ng>0)
			{
			// first stage
			ii=0;
			d_cvt_tran_mat2pmat(ng, nu, D, ng, 0, hpDCt[0], cng);
			// middle stages
			for(ii=1; ii<N; ii++)
				{
				d_cvt_tran_mat2pmat(ng, nu, D+ii*nu*ng, ng, 0, hpDCt[ii], cng);
				d_cvt_tran_mat2pmat(ng, nx, C+ii*nx*ng, ng, nu, hpDCt[ii]+nu/bs*cng*bs+nu%bs, cng);
				}
			}
		// last stage
		if(ngN>0)
			{
			//for(ii=0; ii<pnu*cngN; ii++) hpDCt[N][ii] = 0.0; // make sure D is zero !!!!!
			//d_cvt_tran_mat2pmat(ngN, nx, nu, bs, Cf, ngN, hpDCt[N]+nu/bs*cngN*bs+nu%bs, cngN);
			d_cvt_tran_mat2pmat(ngN, nx, Cf, ngN, 0, hpDCt[N], cngN);
			}
		//d_print_mat(ngN, nx, Cf, ngN);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ngN, bs, hpDCt[N], cngN);
		//exit(1);
		//printf("\n%d %d\n", nb, ng);
		//d_print_mat(ng, nx, C, ng);
		//d_print_mat(ng, nx, C+nx*ng, ng);
		//d_print_mat(ng, nu, D, ng);
		//d_print_mat(ng, nu, D+nu*ng, ng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[0], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[1], cng);
		//d_print_pmat(nx+nu, ng, bs, hpDCt[N], cng);
		//exit(1);

		// cost function
		// first stage
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nu, nx, S, nu, 0, hpQ[0], cnx);
		dgemv_n_lib(nu, nx, hpQ[0], cnx, hux[1], 1, r, hpi[1]);

		jj = 0;
		d_cvt_mat2pmat(nu, nu, R, nu, 0, hpQ[0], cnu);
		for(ii=0; ii<nu; ii++)
			hpQ[0][(nu)/bs*cnu*bs+(nu)%bs+ii*bs] = hpi[1][ii];

		// middle stages
		for(jj=1; jj<N; jj++)
			{
//			d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnux);
			d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnux);
			d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnux*bs+nu%bs, cnux);
			d_cvt_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnux*bs+nu%bs+nu*bs, cnux);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnux*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnux*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
			}

		// last stage
		d_cvt_mat2pmat(nx, nx, Qf, nx, 0, hpQ[N], cnx);
		for(jj=0; jj<nx; jj++)
			hpQ[N][nx/bs*cnx*bs+nx%bs+jj*bs] = qf[jj];

		// estimate mu0 if not user-provided
		if(mu0<=0)
			{
			jj=0;
			for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, R[jj*nu*nu+ii*nu+ll]);
			for(ii=0; ii<nu; ii++) mu0 = fmax(mu0, r[jj*nu+ii]);
			for(jj=1; jj<N; jj++)
				{
				for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, R[jj*nu*nu+ii*nu+ll]);
				for(ii=0; ii<nx*nu; ii++) mu0 = fmax(mu0, S[jj*nu*nx+ii]);
				for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Q[jj*nx*nx+ii*nx+ll]);
				for(ii=0; ii<nu; ii++) mu0 = fmax(mu0, r[jj*nu+ii]);
				for(ii=0; ii<nx; ii++) mu0 = fmax(mu0, q[jj*nx+ii]);
				}
			for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Qf[ii*nx+ll]);
			for(jj=0; jj<nx; jj++) mu0 = fmax(mu0, qf[ii]);
			}

		//d_print_pmat(nz, nz, bs, hpQ[0], cnu);
		//d_print_pmat(nz, nz, bs, hpQ[1], cnux);
		//d_print_pmat(nz, nz, bs, hpQ[N], cnx);
		//exit(1);

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii+nb*jj]!=ub[ii+nb*jj]) // equality constraint
					{
					hd[jj][ii+0]    = lb[ii+nb*jj];
					hd[jj][ii+pnb0] = ub[ii+nb*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nxx[jj]+nuu[jj])/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nb*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hd[jj][ii+0]    = lb[ii+nb*jj] + 1e3;
					hd[jj][ii+pnb0] = ub[ii+nb*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=1; jj<N; jj++)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=nuu[jj]; ii<nbb[jj]; ii++)
				{
				hd[jj][ii+0]    = lb[ii+nb*jj];
				hd[jj][ii+pnb0] = ub[ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    = lb[nu+ii+nb*jj];
			hd[N][ii+pnb0] = ub[nu+ii+nb*jj];
			//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				pnb0 = (nbb[jj]+bs-1)/bs*bs;
				png0 = (ngg[jj]+bs-1)/bs*bs;
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb0+ii+0]    = lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    = lgf[ii];
				hd[N][2*pnb0+ii+png0] = ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

		} // end of time variant



	// initial guess 
	if(warm_start)
		{

		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				hux[jj][ii] = u[ii+nu*jj];

		for(jj=1; jj<=N; jj++)
			for(ii=0; ii<nx; ii++)
				hux[jj][nuu[jj]+ii] = x[ii+nx*jj];

		}



#if 0
for(ii=0; ii<=N; ii++)
{
for(jj=0; jj<nbb[ii]; jj++)
	printf("%d ", idxb[ii][jj]);
printf("\n");
}
for(ii=0; ii<=N; ii++)
d_print_mat(1, 2*pnb, hd[ii], 1);
exit(1);
#endif

	// call the IP solver
	hpmpc_status = d_ip2_mpc_hard_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, stat, N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hpQ, hpDCt, hd, hux, 1, hpi, hlam, ht, work);

#if 0
for(ii=0; ii<=N; ii++)
d_print_mat(1, nuu[ii]+nxx[ii], hux[ii], 1);
exit(1);
#endif



	// copy back inputs and states
	for(jj=0; jj<N; jj++)
		for(ii=0; ii<nu; ii++)
			u[ii+nu*jj] = hux[jj][ii];

	for(jj=1; jj<=N; jj++)
		for(ii=0; ii<nx; ii++)
			x[ii+nx*jj] = hux[jj][nuu[jj]+ii];



	// check for input equality constraints
	if(time_invariant)
		{
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii]==ub[ii]) // equality constraint
					{
					u[ii+nu*jj] = lb[ii];
					}
				}
			}
		}
	else // time-variant
		{
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii+nb*jj]==ub[ii+nb*jj]) // equality constraint
					{
					u[ii+nu*jj] = lb[ii+nb*jj];
					}
				}
			}
		}



	// compute infinity norm of residuals on exit

	// restore linear part of cost function 
	if(time_invariant)
		{
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nuu[ii]; jj++) 
				hq[ii][jj]    = r[jj];
			for(jj=0; jj<nxx[ii]; jj++) 
				hq[ii][nuu[ii]+jj] = q[jj];
			}
		}
	else // time-variant
		{
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nuu[ii]; jj++) 
				hq[ii][jj]    = r[jj+nu*ii];
			for(jj=0; jj<nxx[ii]; jj++) 
				hq[ii][nuu[ii]+jj] = q[jj+nx*ii];
			}
		}
	for(jj=0; jj<nx; jj++) 
		hq[N][jj] = qf[jj];

	double mu;

	d_res_mpc_hard_tv(N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hb, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

#if 0
	printf("\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hrq[ii], 1);
	printf("\n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nxx[ii+1], hrb[ii], 1);
	printf("\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, (nbb[ii]+bs-1)/bs*bs*2+(ngg[ii]+bs-1)/bs*bs*2, hrd[ii], 1);
	printf("\n");
	printf("%f\n", mu);
	exit(1);
#endif

	temp = fabs(hrq[0][0]);
	for(jj=0; jj<nu; jj++) 
		temp = fmax( temp, fabs(hrq[0][jj]) );
	for(ii=1; ii<N; ii++)
		for(jj=0; jj<nu+nx; jj++) 
			temp = fmax( temp, fabs(hrq[ii][jj]) );
	for(jj=0; jj<nx; jj++) 
		temp = fmax( temp, fabs(hrq[N][jj]) );
	inf_norm_res[0] = temp;

	temp = fabs(hrb[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx; jj++) 
			temp = fmax( temp, fabs(hrb[ii][jj]) );
	inf_norm_res[1] = temp;

	temp = fabs(hrd[0][0]);
	pnb0 = (nbb[0]+bs-1)/bs*bs;
	for(jj=0; jj<nbu; jj++) 
		{
		temp = fmax( temp, fabs(hrd[0][jj]) );
		temp = fmax( temp, fabs(hrd[0][pnb0+jj]) );
		}
	for(ii=1; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		for(jj=0; jj<nb; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj]) );
			temp = fmax( temp, fabs(hrd[ii][pnb0+jj]) );
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	for(jj=0; jj<nbb[N]; jj++) 
		{
		temp = fmax( temp, fabs(hrq[N][jj]) );
		temp = fmax( temp, fabs(hrq[N][pnb0+jj]) );
		}
	for(ii=0; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		png0 = (ngg[ii]+bs-1)/bs*bs;
		for(jj=2*pnb0; jj<2*pnb0+ng; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj]) );
			temp = fmax( temp, fabs(hrd[ii][png0+jj]) );
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	png0 = (ngg[N]+bs-1)/bs*bs;
	for(jj=2*pnb0; jj<2*pnb0+ngg[N]; jj++) 
		{
		temp = fmax( temp, fabs(hrd[N][jj]) );
		temp = fmax( temp, fabs(hrd[N][png0+jj]) );
		}
	inf_norm_res[2] = temp;

	inf_norm_res[3] = mu;

	//printf("\n%e %e %e %e\n", norm_res[0], norm_res[1], norm_res[2], norm_res[3]);



	// copy back multipliers

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx; jj++)
			pi[jj+ii*nx] = hpi[ii][jj];

	ii = 0;
	pnb0 = (nbb[0]+bs-1)/bs*bs;
	for(jj=0; jj<nbu; jj++)
		{
		lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
		lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb0+jj];
		t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
		t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb0+jj];
		}
	for(ii=1; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		for(jj=0; jj<nb; jj++)
			{
			lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
			lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb0+jj];
			t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
			t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb0+jj];
			}
		}
	ii = N;
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	for(jj=0; jj<nbb[N]; jj++)
		{
		lam[nu+jj+ii*(2*nb+2*ng)]        = hlam[ii][jj];
		lam[nu+jj+ii*(2*nb+2*ng)+nb+ngN] = hlam[ii][pnb0+jj];
		t[nu+jj+ii*(2*nb+2*ng)]        = ht[ii][jj];
		t[nu+jj+ii*(2*nb+2*ng)+nb+ngN] = ht[ii][pnb0+jj];
		}

	for(ii=0; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		png0 = (ngg[ii]+bs-1)/bs*bs;
		for(jj=0; jj<ng; jj++)
			{
			lam[jj+ii*(2*nb+2*ng)+nb]       = hlam[ii][2*pnb0+jj];
			lam[jj+ii*(2*nb+2*ng)+nb+ng+nb] = hlam[ii][2*pnb0+png0+jj];
			t[jj+ii*(2*nb+2*ng)+nb]       = ht[ii][2*pnb0+jj];
			t[jj+ii*(2*nb+2*ng)+nb+ng+nb] = ht[ii][2*pnb0+png0+jj];
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	png0 = (ngg[N]+bs-1)/bs*bs;
	for(jj=0; jj<ngN; jj++)
		{
		lam[jj+N*(2*nb+2*ng)+nb]        = hlam[N][2*pnb0+jj];
		lam[jj+N*(2*nb+2*ng)+nb+ngN+nb] = hlam[N][2*pnb0+png0+jj];
		t[jj+N*(2*nb+2*ng)+nb]        = ht[N][2*pnb0+jj];
		t[jj+N*(2*nb+2*ng)+nb+ngN+nb] = ht[N][2*pnb0+png0+jj];
		}

//	printf("\nend of wrapper\n");

    return hpmpc_status;

	}



void fortran_order_d_solve_kkt_new_rhs_mpc_hard_tv(
							int N, int nx, int nu, int nb, int ng, int ngN, 
							int time_invariant, int free_x0,
							double* A, double* B, double* b, 
							double* Q, double* Qf, double* S, double* R, 
							double* q, double* qf, double* r, 
							double *lb, double *ub,
							double *C, double *D, double *lg, double *ug,
							double *Cf, double *lgf, double *ugf,
							double tau, 
							double* x, double* u, double *pi, double *lam, double *t,
							double *inf_norm_res,
							double *work0) 

	{

//printf("\nstart of wrapper\n");

	//printf("\n%d %d %d %d %d %d\n", N, nx, nu, nb, ng, ngN);

	const int nbu = nb<nu ? nb : nu ;

	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;

	const int nz   = nx+nu+1;
	const int pnz  = bs*((nz+bs-1)/bs);
	const int pnu  = bs*((nu+bs-1)/bs);
	const int pnx  = bs*((nx+bs-1)/bs);
	const int pnb  = bs*((nb+bs-1)/bs);
	const int png  = bs*((ng+bs-1)/bs);
	const int pngN = bs*((ngN+bs-1)/bs);
	const int cnux = (nu+nx+ncl-1)/ncl*ncl;
	const int cnu  = (nu+ncl-1)/ncl*ncl;
	const int cnx  = ncl*((nx+ncl-1)/ncl);
	const int cng  = ncl*((ng+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);

	int pnb0;
	int png0;


	int i, ii, jj, ll;

	double temp;



	// time-variant quantities
	int *ptr_int;
	ptr_int = (int *) work0;

	int nxx[N+1];
	int nuu[N+1];
	int nbb[N+1];
	int ngg[N+1];
	int *(idxb[N+1]);

	nxx[0] = 0;
	nuu[0] = nu;
	nbb[0] = nbu;
	idxb[0] = ptr_int;
	ptr_int += nbb[0];
	for(jj=0; jj<nbb[0]; jj++) idxb[0][jj] = jj;
	ngg[0] = ng;
	for(ii=1; ii<N; ii++)
		{
		nxx[ii] = nx;
		nuu[ii] = nu;
		nbb[ii] = nb;
		idxb[ii] = ptr_int;
		ptr_int += nbb[ii];
		for(jj=0; jj<nbb[ii]; jj++) idxb[ii][jj] = jj;
		ngg[ii] = ng;
		}
	nxx[N] = nx;
	nuu[N] = 0;
	nbb[N] = nb-nu>0 ? nb-nu : 0;
	idxb[N] = ptr_int;
	ptr_int += nbb[N];
	for(jj=0; jj<nbb[N]; jj++) idxb[N][jj] = nuu[N]+jj;
	ngg[N] = ngN;

	work0 = (double *) ptr_int;



//printf("\n%d\n", ((size_t) work0) & 63);

	/* align work space */
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	double *ptr = (double *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

	/* array or pointers */
	double *hpBAbt[N];
	double *hb[N];
	double *hpQ[N+1];
	double *hq[N+1];
	double *hpDCt[N+1];
	double *hd[N+1];
	double *hux[N+1];
	double *hpi[N];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrb[N];
	double *hrq[N+1];
	double *hrd[N+1];
	double *work;



	if(time_invariant) // TODO time-invariant work space ?????
		{

		hpBAbt[0] = ptr;
		ptr += pnz*cnx;
		for(ii=1; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			}
		ptr += pnz*cnx;

		if(ng>0)
			{
			hpDCt[0] = ptr;
			ptr += pnz*cng;
			for(ii=1; ii<N; ii++)
				{
				hpDCt[ii] = ptr;
				}
			ptr += pnz*cng;
			}
		if(ngN>0)
			{
			hpDCt[N] = ptr;
			ptr += pnz*cngN;
			}

		hpQ[0] = ptr;
		ptr += pnz*cnux;
		for(ii=1; ii<N; ii++) // time variant and copied again internally in the IP !!!
			{
			hpQ[ii] = ptr;
			}
		ptr += pnz*cnux;
		hpQ[N] = ptr;
		ptr += pnz*cnux;

		work = ptr;
		ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nxx, nuu, nbb, ngg)/sizeof(double);

		hb[0] = ptr;
		ptr += pnx;
		for(ii=1; ii<N; ii++)
			{
			hb[ii] = ptr;
			}
		ptr += pnx;

		hq[0] = ptr;
		ptr += pnz;
		for(ii=1; ii<N; ii++)
			{
			hq[ii] = ptr;
			}
		ptr += pnz;
		hq[N] = ptr;
		ptr += pnz;

		hd[0] = ptr;
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		for(ii=1; ii<N; ii++) // time Variant box constraints
			{
			hd[ii] = ptr;
			}
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hpi[ii] = ptr;
			ptr += pnx; // for alignment of ptr
			}

		for(ii=0; ii<N; ii++) // time variant Lagrangian multipliers
			{
			hlam[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		hlam[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++) // time variant slack variables
			{
			ht[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		ht[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++)
			{
			hrb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hrq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hrd[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hrd[N] = ptr;
		ptr += 2*pnb+2*pngN;



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx); // pack A into (temporary) buffer
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hb[0]); // result 

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx); // restore B0

		// middle stages
		ii=1;
		if(jj<N)
			{
			d_copy_mat(nx, 1, b, nx, hb[ii], nx);
			}


		// cost function
		// first stage
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nu, nx, S, nu, 0, hpQ[0], cnx); // pack S into a (temporary) buffer
		dgemv_n_lib(nu, nx, hpQ[0], cnx, hux[1], 1, r, hq[0]); // result

		jj = 0;
		d_cvt_mat2pmat(nu, nu, R, nu, 0, hpQ[0], cnu); // restore R0

		// middle stages
		jj=1;
		if(jj<N)
			{
			d_copy_mat(nu, 1, r, nu, hq[jj], nu+nx);
			d_copy_mat(nx, 1, q, nx, hq[jj]+nu, nu+nx);
			}

		// last stage
		d_copy_mat(nx, 1, qf, nx, hq[N], nx);


		// input constraints
		jj=0;
		pnb0 = (nbb[jj]+bs-1)/bs*bs;
		for(ii=0; ii<nbu; ii++)
			{
			hd[jj][ii+0]    = lb[ii];
			hd[jj][ii+pnb0] = ub[ii];
			}
		jj=1;
		if(jj<N)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=0; ii<nbu; ii++)
				{
				hd[jj][ii+0]    = lb[ii];
				hd[jj][ii+pnb0] = ub[ii];
				}
			}
		// state constraints 
		jj=1;
		if(jj<N)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=nuu[jj]; ii<nbb[jj]; ii++)
				{
				hd[jj][ii+0]    = lb[ii];
				hd[jj][ii+pnb0] = ub[ii];
				//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    = lb[nu+ii];
			hd[N][ii+pnb0] = ub[nu+ii];
			//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				pnb0 = (nbb[jj]+bs-1)/bs*bs;
				png0 = (ngg[jj]+bs-1)/bs*bs;
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb0+ii+0]    = lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    = lgf[ii];
				hd[N][2*pnb0+ii+png0] = ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

		} // end of time invariant
	else // time variant
		{

		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		if(ng>0)
			{
			for(ii=0; ii<N; ii++)
				{
				hpDCt[ii] = ptr;
				ptr += pnz*cng;
				}
			}
		if(ngN>0)
			{
			hpDCt[N] = ptr;
			ptr += pnz*cngN;
			}

		for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnux; // TODO use cnux instrad
			}

		work = ptr;
		ptr += d_ip2_mpc_hard_tv_work_space_size_bytes(N, nxx, nuu, nbb, ngg)/sizeof(double);

		for(ii=0; ii<N; ii++)
			{
			hb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hd[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hpi[ii] = ptr;
			ptr += pnx; // for alignment of ptr
			}

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hlam[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		hlam[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			ht[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		ht[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

		for(ii=0; ii<N; ii++)
			{
			hrb[ii] = ptr;
			ptr += pnx;
			}

		for(ii=0; ii<=N; ii++)
			{
			hrq[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<N; ii++)
			{
			hrd[ii] = ptr;
			ptr += 2*pnb+2*png;
			}
		hrd[N] = ptr;
		ptr += 2*pnb+2*pngN;



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx);
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b+0*nx, hb[0]);

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B+0*nx*nu, nx, 0, hpBAbt[0], cnx); // restore B0

		// middle stages
		for(ii=1; ii<N; ii++)
			{
			d_copy_mat(nx, 1, b+ii*nx, nx, hb[ii], nx);
			}


		// cost function
		// first stage
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nu, nx, S, nu, 0, hpQ[0], cnx);
		dgemv_n_lib(nu, nx, hpQ[0], cnx, hux[1], 1, r+0*nu, hq[0]);

		jj = 0;
		d_cvt_mat2pmat(nu, nu, R, nu, 0, hpQ[0], cnu); // restore Q0

		// middle stages
		for(jj=1; jj<N; jj++)
			{
			d_copy_mat(nu, 1, r+jj*nu, nu, hq[jj], nu+nx);
			d_copy_mat(nx, 1, q+jj*nx, nx, hq[jj]+nu, nu+nx);
			}

		// last stage
		d_copy_mat(nx, 1, qf, nx, hq[N], nx);


		// input constraints
		for(jj=0; jj<N; jj++)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=0; ii<nbu; ii++)
				{
				hd[jj][ii+0]    = lb[ii+nb*jj];
				hd[jj][ii+pnb0] = ub[ii+nb*jj];
				}
			}
		// state constraints 
		for(jj=1; jj<N; jj++)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=nuu[jj]; ii<nbb[jj]; ii++)
				{
				hd[jj][ii+0]    = lb[ii+nb*jj];
				hd[jj][ii+pnb0] = ub[ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    = lb[nu+ii+nb*jj];
			hd[N][ii+pnb0] = ub[nu+ii+nb*jj];
			//hd[jj][2*nu+2*ii+0] = lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = ub[N*nu+ii+nx*jj];
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				pnb0 = (nbb[jj]+bs-1)/bs*bs;
				png0 = (ngg[jj]+bs-1)/bs*bs;
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb0+ii+0]    = lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    = lgf[ii];
				hd[N][2*pnb0+ii+png0] = ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

		} // end of time variant



	// initial guess TODO remove
//	for(jj=0; jj<N; jj++)
//		for(ii=0; ii<nu; ii++)
//			hux[jj][ii] = u[ii+nu*jj];

//	for(jj=1; jj<=N; jj++)
//		for(ii=0; ii<nx; ii++)
//			hux[jj][nuu[jj]+ii] = x[ii+nx*jj];



	// call the IP solver
	d_kkt_solve_new_rhs_mpc_hard_tv(N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hb, hpQ, hq, hpDCt, hd, tau, hux, 1, hpi, hlam, ht, work); // TODO B 



	// copy back inputs and states
	for(jj=0; jj<N; jj++)
		for(ii=0; ii<nu; ii++)
			u[ii+nu*jj] = hux[jj][ii];

	for(jj=1; jj<=N; jj++)
		for(ii=0; ii<nx; ii++)
			x[ii+nx*jj] = hux[jj][nuu[jj]+ii];



	// compute infinity norm of residuals on exit

	// restore linear part of cost function 
	if(time_invariant)
		{
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nuu[ii]; jj++) 
				hq[ii][jj]    = r[jj];
			for(jj=0; jj<nxx[ii]; jj++) 
				hq[ii][nuu[ii]+jj] = q[jj];
			}
		}
	else // time-variant
		{
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nuu[ii]; jj++) 
				hq[ii][jj]    = r[jj+nu*ii];
			for(jj=0; jj<nxx[ii]; jj++) 
				hq[ii][nuu[ii]+jj] = q[jj+nx*ii];
			}
		}
	for(jj=0; jj<nx; jj++) 
		hq[N][jj] = qf[jj];

	double mu;

	d_res_mpc_hard_tv(N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hb, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

#if 0
	printf("\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, nuu[ii]+nxx[ii], hrq[ii], 1);
	printf("\n");
	for(ii=0; ii<N; ii++)
		d_print_mat(1, nxx[ii+1], hrb[ii], 1);
	printf("\n");
	for(ii=0; ii<=N; ii++)
		d_print_mat(1, (nbb[ii]+bs-1)/bs*bs*2+(ngg[ii]+bs-1)/bs*bs*2, hrd[ii], 1);
	printf("\n");
	printf("%f\n", mu);
	exit(1);
#endif

	temp = fabs(hrq[0][0]);
	for(jj=0; jj<nu; jj++) 
		temp = fmax( temp, fabs(hrq[0][jj]) );
	for(ii=1; ii<N; ii++)
		for(jj=0; jj<nu+nx; jj++) 
			temp = fmax( temp, fabs(hrq[ii][jj]) );
	for(jj=0; jj<nx; jj++) 
		temp = fmax( temp, fabs(hrq[N][jj]) );
	inf_norm_res[0] = temp;

	temp = fabs(hrb[0][0]);
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx; jj++) 
			temp = fmax( temp, fabs(hrb[ii][jj]) );
	inf_norm_res[1] = temp;

	temp = fabs(hrd[0][0]);
	pnb0 = (nbb[0]+bs-1)/bs*bs;
	for(jj=0; jj<nbu; jj++) 
		{
		temp = fmax( temp, fabs(hrd[0][jj]) );
		temp = fmax( temp, fabs(hrd[0][pnb0+jj]) );
		}
	for(ii=1; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		for(jj=0; jj<nb; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj]) );
			temp = fmax( temp, fabs(hrd[ii][pnb0+jj]) );
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	for(jj=0; jj<nbb[N]; jj++) 
		{
		temp = fmax( temp, fabs(hrq[N][jj]) );
		temp = fmax( temp, fabs(hrq[N][pnb0+jj]) );
		}
	for(ii=0; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		png0 = (ngg[ii]+bs-1)/bs*bs;
		for(jj=2*pnb0; jj<2*pnb0+ng; jj++) 
			{
			temp = fmax( temp, fabs(hrd[ii][jj]) );
			temp = fmax( temp, fabs(hrd[ii][png0+jj]) );
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	png0 = (ngg[N]+bs-1)/bs*bs;
	for(jj=2*pnb0; jj<2*pnb0+ngg[N]; jj++) 
		{
		temp = fmax( temp, fabs(hrd[N][jj]) );
		temp = fmax( temp, fabs(hrd[N][png0+jj]) );
		}
	inf_norm_res[2] = temp;

	inf_norm_res[3] = mu;

	//printf("\n%e %e %e %e\n", norm_res[0], norm_res[1], norm_res[2], norm_res[3]);



	// copy back multipliers

	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx; jj++)
			pi[jj+ii*nx] = hpi[ii][jj];

	ii = 0;
	pnb0 = (nbb[0]+bs-1)/bs*bs;
	for(jj=0; jj<nbu; jj++)
		{
		lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
		lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb0+jj];
		t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
		t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb0+jj];
		}
	for(ii=1; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		for(jj=0; jj<nb; jj++)
			{
			lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
			lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb0+jj];
			t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
			t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb0+jj];
			}
		}
	ii = N;
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	for(jj=0; jj<nbb[N]; jj++)
		{
		lam[nu+jj+ii*(2*nb+2*ng)]        = hlam[ii][jj];
		lam[nu+jj+ii*(2*nb+2*ng)+nb+ngN] = hlam[ii][pnb0+jj];
		t[nu+jj+ii*(2*nb+2*ng)]        = ht[ii][jj];
		t[nu+jj+ii*(2*nb+2*ng)+nb+ngN] = ht[ii][pnb0+jj];
		}

	for(ii=0; ii<N; ii++)
		{
		pnb0 = (nbb[ii]+bs-1)/bs*bs;
		png0 = (ngg[ii]+bs-1)/bs*bs;
		for(jj=0; jj<ng; jj++)
			{
			lam[jj+ii*(2*nb+2*ng)+nb]       = hlam[ii][2*pnb0+jj];
			lam[jj+ii*(2*nb+2*ng)+nb+ng+nb] = hlam[ii][2*pnb0+png0+jj];
			t[jj+ii*(2*nb+2*ng)+nb]       = ht[ii][2*pnb0+jj];
			t[jj+ii*(2*nb+2*ng)+nb+ng+nb] = ht[ii][2*pnb0+png0+jj];
			}
		}
	pnb0 = (nbb[N]+bs-1)/bs*bs;
	png0 = (ngg[N]+bs-1)/bs*bs;
	for(jj=0; jj<ngN; jj++)
		{
		lam[jj+N*(2*nb+2*ng)+nb]        = hlam[N][2*pnb0+jj];
		lam[jj+N*(2*nb+2*ng)+nb+ngN+nb] = hlam[N][2*pnb0+png0+jj];
		t[jj+N*(2*nb+2*ng)+nb]        = ht[N][2*pnb0+jj];
		t[jj+N*(2*nb+2*ng)+nb+ngN+nb] = ht[N][2*pnb0+png0+jj];
		}

/*printf("\nend of wrapper\n");*/

    return;

	}


