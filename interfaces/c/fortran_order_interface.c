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

/* ACADO auto-generated header */
/*#include "acado_common.h"*/
/* HPMPC configuration */
/*#include "hpmpc_pro/target.h"*/
/*#include "hpmpc_pro/block_size.h"*/
/*#include "hpmpc_pro/aux_d.h"*/
/*#include "hpmpc_pro/aux_s.h"*/
/*#include "hpmpc_pro/mpc_solvers.h"*/

#include "../../include/target.h"
#include "../../include/block_size.h"
#include "../../include/aux_d.h"
#include "../../include/aux_s.h"
#include "../../include/blas_d.h"
#include "../../include/lqcp_solvers.h"
#include "../../include/mpc_solvers.h"

// problem size (states, inputs, horizon)
/*#define NX ACADO_NX*/
/*#define NU ACADO_NU*/
/*#define NN ACADO_N*/

// free initial state: 0 mpc, 1 mhe
//#define FREE_X0 0 // TODO remve

// ip method: 1 primal-dual, 2 predictor-corrector primal-dual
#define IP 2

// warm-start with user-provided solution (otherwise initialize x and u with 0 or something feasible)
#define WARM_START 0

// double/single ('d'/'s') precision
#define PREC 'd' // TODO remve

// minimum accepted step length
#define ALPHA_MIN 1e-12

/*// threshold in the duality measure to switch from single to double precision*/
/*#define SP_THR 1e5*/

// assume the R matrices to be diagonal in MHE if
#define DIAG_R 0

// Debug flag
#ifndef PC_DEBUG
#define PC_DEBUG 0
#endif /* PC_DEBUG */








// OCP QP interface
// struct of arguments to the solver
struct ocp_qp_solver_args{
	double tol;
	int max_iter;
	double min_step;
	double mu0;
	double sigma_min;
	};



// emum of return values
enum return_values{
	ACADOS_SUCCESS,
	ACADOS_MAXITER,
	ACADOS_MINSTEP
	};



// work space size
int ocp_qp_hpmpc_workspace_double(int N, int *nx, int *nu, int *ng, int *nb, struct ocp_qp_solver_args *args)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int kmax = args->max_iter;

	int ii;

	int pnx[N+1]; for(ii=0; ii<=N; ii++) pnx[ii] = (nx[ii]+bs-1)/bs*bs;
	int pnz[N+1]; for(ii=0; ii<=N; ii++) pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
	int pnux[N+1]; for(ii=0; ii<=N; ii++) pnux[ii] = (nu[ii]+nx[ii]+bs-1)/bs*bs;
	int pnb[N+1]; for(ii=0; ii<=N; ii++) pnb[ii] = (nb[ii]+bs-1)/bs*bs;
	int png[N+1]; for(ii=0; ii<=N; ii++) png[ii] = (ng[ii]+bs-1)/bs*bs;
	int cnx[N+1]; for(ii=0; ii<=N; ii++) cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
	int cnux[N+1]; for(ii=0; ii<=N; ii++) cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
	int cng[N+1]; for(ii=0; ii<=N; ii++) cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;

	int work_space = 16; // enough to align twice to cache boundaries

	for(ii=0; ii<N; ii++)
		{
		work_space += pnz[ii]*cnx[ii+1] + pnz[ii]*cnux[ii] + pnux[ii]*cng[ii] + 6*pnb[ii] + 6*png[ii] + pnux[ii] + pnx[ii];
		}
	ii = N;
	work_space += pnz[ii]*cnux[ii] + pnux[ii]*cng[ii] + 6*pnb[ii] + 6*png[ii] + pnux[ii] + pnx[ii];

	work_space += 5*kmax;

	work_space += d_ip2_hard_mpc_tv_work_space_size_double(N, nx, nu, nb, ng); // work space needed by the IPM
	
	return work_space;

	}



/* version dealing with equality constratins: is lb=ub, then fix the variable (corresponding column in A or B set to zero, and updated b) */
int ocp_qp_solver(int N, int *nx, int *nu, int *nb, int *ng, double **hA, double **hB, double **hb, double **hQ, double **hS, double **hR, double **hq, double **hr, int **hidxb, double **hlb, double **hub, double **hC, double **hD, double **hlg, double **hug, double **hx, double **hu, struct ocp_qp_solver_args *args, double *work0)
	{

//	printf("\nstart of wrapper\n");

	int acados_status = ACADOS_SUCCESS;

	const int bs = D_MR;
	const int ncl = D_NCL;

	//const int anb = nal*((2*nb+nal-1)/nal);

	double mu_tol = args->tol;
	int k_max = args->max_iter;
	double alpha_min = args->min_step; // minimum accepted step length
	double sigma_par[] = {0.4, 0.1, args->sigma_min}; // control primal-dual IP behaviour
	double mu0 = args->mu0;
	int warm_start = 0;
	int compute_mult = 1; // compute multipliers TODO set to zero
	int kk = -1;
	
	int info = 0;

	int i, ii, jj, ll;



	// align work space (assume cache line <= 64 byte)
	size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
	double *ptr = (double *) addr;



	// array of pointers
	double *(hpBAbt[N]);
	double *(hpRSQrq[N+1]);
	double *(hpDCt[N+1]);
	double *(hd[N+1]);
	double *(hux[N+1]);
	double *(hpi[N+1]);
	double *(hlam[N+1]);
	double *(ht[N+1]);
	double *(hqq[N+1]);
	double *(hrb[N]);
	double *(hrq[N+1]);
	double *(hrd[N+1]);
	double *stat;
	double *work;



	// matrices size
	int pnx[N+1]; for(ii=0; ii<=N; ii++) pnx[ii] = (nx[ii]+bs-1)/bs*bs;
	int pnz[N+1]; for(ii=0; ii<=N; ii++) pnz[ii] = (nu[ii]+nx[ii]+1+bs-1)/bs*bs;
	int pnux[N+1]; for(ii=0; ii<=N; ii++) pnux[ii] = (nu[ii]+nx[ii]+bs-1)/bs*bs;
	int pnb[N+1]; for(ii=0; ii<=N; ii++) pnb[ii] = (nb[ii]+bs-1)/bs*bs;
	int png[N+1]; for(ii=0; ii<=N; ii++) png[ii] = (ng[ii]+bs-1)/bs*bs;
	int cnx[N+1]; for(ii=0; ii<=N; ii++) cnx[ii] = (nx[ii]+ncl-1)/ncl*ncl;
	int cnux[N+1]; for(ii=0; ii<=N; ii++) cnux[ii] = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
	int cng[N+1]; for(ii=0; ii<=N; ii++) cng[ii] = (ng[ii]+ncl-1)/ncl*ncl;



	// data structure

	ii = 0;
	hpBAbt[ii] = ptr;
	ptr += pnz[ii]*cnx[ii+1];
	d_cvt_tran_mat2pmat(nx[ii+1], nu[ii], hB[ii], nx[ii+1], 0, hpBAbt[ii], cnx[ii+1]);
	d_cvt_tran_mat2pmat(nx[ii+1], nx[ii], hA[ii], nx[ii+1], nu[ii], hpBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1]);
	d_cvt_tran_mat2pmat(nx[ii+1], 1, hb[ii], nx[ii+1], nu[ii]+nx[ii], hpBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1]);
//	d_print_pmat(nu[ii]+nx[ii]+1, nx[ii+1], bs, hpBAbt[ii], cnx[ii+1]);
	for(ii=1; ii<N; ii++)
		{
		if(hb[ii]==hb[ii-1] && hA[ii]==hA[ii-1] && hB[ii]==hB[ii-1])
			{
//			printf("\nsame\n");
			hpBAbt[ii] = hpBAbt[ii-1];
			}
		else
			{
//			printf("\ndifferent\n");
			hpBAbt[ii] = ptr;
			ptr += pnz[ii]*cnx[ii+1];
			d_cvt_tran_mat2pmat(nx[ii+1], nu[ii], hB[ii], nx[ii+1], 0, hpBAbt[ii], cnx[ii+1]);
			d_cvt_tran_mat2pmat(nx[ii+1], nx[ii], hA[ii], nx[ii+1], nu[ii], hpBAbt[ii]+nu[ii]/bs*bs*cnx[ii+1]+nu[ii]%bs, cnx[ii+1]);
			d_cvt_tran_mat2pmat(nx[ii+1], 1, hb[ii], nx[ii+1], nu[ii]+nx[ii], hpBAbt[ii]+(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs, cnx[ii+1]);
			}
//		d_print_pmat(nu[ii]+nx[ii]+1, nx[ii+1], bs, hpBAbt[ii], cnx[ii+1]);
		}
	
	ii = 0;
	hpRSQrq[ii] = ptr;
	ptr += pnz[ii]*cnux[ii];
	d_cvt_mat2pmat(nu[ii], nu[ii], hR[ii], nu[ii], 0, hpRSQrq[ii], cnux[ii]);
	d_cvt_tran_mat2pmat(nx[ii], nu[ii], hS[ii], nu[ii], nu[ii], hpRSQrq[ii]+nu[ii]/bs*bs*cnux[ii]+nu[ii]%bs, cnux[ii]);
	d_cvt_mat2pmat(nx[ii], nx[ii], hQ[ii], nx[ii], nu[ii], hpRSQrq[ii]+nu[ii]/bs*bs*cnux[ii]+nu[ii]%bs+nu[ii]*bs, cnux[ii]);
	d_cvt_tran_mat2pmat(nu[ii], 1, hr[ii], nu[ii], nu[ii]+nx[ii], hpRSQrq[ii]+(nu[ii]+nx[ii])/bs*bs*cnux[ii]+(nu[ii]+nx[ii])%bs, cnux[ii]);
	d_cvt_tran_mat2pmat(nx[ii], 1, hq[ii], nx[ii], nu[ii]+nx[ii], hpRSQrq[ii]+(nu[ii]+nx[ii])/bs*bs*cnux[ii]+(nu[ii]+nx[ii])%bs+nu[ii]*bs, cnux[ii]);
//	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, hpRSQrq[ii], cnux[ii]);
	for(ii=1; ii<N; ii++)
		{
		if(hq[ii]==hq[ii-1] && hr[ii]==hr[ii-1] && hQ[ii]==hQ[ii-1] && hS[ii]==hS[ii-1] && hR[ii]==hR[ii-1])
			{
//			printf("\nsame\n");
			hpRSQrq[ii] = hpRSQrq[ii-1];
			}
		else
			{
//			printf("\ndifferent\n");
			hpRSQrq[ii] = ptr;
			ptr += pnz[ii]*cnux[ii];
			d_cvt_mat2pmat(nu[ii], nu[ii], hR[ii], nu[ii], 0, hpRSQrq[ii], cnux[ii]);
			d_cvt_tran_mat2pmat(nx[ii], nu[ii], hS[ii], nu[ii], nu[ii], hpRSQrq[ii]+nu[ii]/bs*bs*cnux[ii]+nu[ii]%bs, cnux[ii]);
			d_cvt_mat2pmat(nx[ii], nx[ii], hQ[ii], nx[ii], nu[ii], hpRSQrq[ii]+nu[ii]/bs*bs*cnux[ii]+nu[ii]%bs+nu[ii]*bs, cnux[ii]);
			d_cvt_tran_mat2pmat(nu[ii], 1, hr[ii], nu[ii], nu[ii]+nx[ii], hpRSQrq[ii]+(nu[ii]+nx[ii])/bs*bs*cnux[ii]+(nu[ii]+nx[ii])%bs, cnux[ii]);
			d_cvt_tran_mat2pmat(nx[ii], 1, hq[ii], nx[ii], nu[ii]+nx[ii], hpRSQrq[ii]+(nu[ii]+nx[ii])/bs*bs*cnux[ii]+(nu[ii]+nx[ii])%bs+nu[ii]*bs, cnux[ii]);
			}
//		d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, hpRSQrq[ii], cnux[ii]);
		}
	ii = N;
	hpRSQrq[ii] = ptr;
	ptr += pnz[ii]*cnux[ii];
	d_cvt_mat2pmat(nx[ii], nx[ii], hQ[ii], nx[ii], 0, hpRSQrq[ii], cnux[ii]);
	d_cvt_tran_mat2pmat(nx[ii], 1, hq[ii], nx[ii], nx[ii], hpRSQrq[ii]+(nx[ii])/bs*bs*cnux[ii]+(nx[ii])%bs, cnux[ii]);
//	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, hpRSQrq[ii], cnux[ii]);

	ii = 0;
	hpDCt[ii] = ptr;
	ptr += pnz[ii]*cng[ii];
	d_cvt_tran_mat2pmat(ng[ii], nu[ii], hD[ii], ng[ii], 0, hpDCt[ii], cng[ii]);
	d_cvt_tran_mat2pmat(ng[ii], nx[ii], hC[ii], ng[ii], nu[ii], hpDCt[ii]+nu[ii]/bs*bs*cng[ii]+nu[ii]%bs, cng[ii]);
//	d_print_pmat(nu[ii]+nx[ii], ng[ii], bs, hpDCt[ii], cng[ii]);
	for(ii=1; ii<N; ii++)
		{
		if(hD[ii]==hD[ii-1] && hC[ii]==hC[ii-1])
			{
//			printf("\nsame\n");
			hpDCt[ii] = hpDCt[ii-1];
			}
		else
			{
//			printf("\ndifferent\n");
			hpDCt[ii] = ptr;
			ptr += pnux[ii]*cng[ii];
			d_cvt_tran_mat2pmat(ng[ii], nu[ii], hD[ii], ng[ii], 0, hpDCt[ii], cng[ii]);
			d_cvt_tran_mat2pmat(ng[ii], nx[ii], hC[ii], ng[ii], nu[ii], hpDCt[ii]+nu[ii]/bs*bs*cng[ii]+nu[ii]%bs, cng[ii]);
			}
//		d_print_pmat(nu[ii]+nx[ii], ng[ii], bs, hpDCt[ii], cng[ii]);
		}
	ii = N;
	hpDCt[ii] = ptr;
	ptr += pnux[ii]*cng[ii];
	d_cvt_tran_mat2pmat(ng[ii], nx[ii], hC[ii], ng[ii], 0, hpDCt[ii], cng[ii]);
//	d_print_pmat(nu[ii]+nx[ii], ng[ii], bs, hpDCt[ii], cng[ii]);

	ii = 0;
	hd[ii] = ptr;
	ptr += 2*pnb[ii]+2*png[ii];
	d_copy_mat(nb[ii], 1, hlb[ii], 1, hd[ii], 1);
	dax_mat(nb[ii], 1, -1.0, hub[ii], 1, hd[ii]+pnb[ii], 1); // TODO change in solver
	d_copy_mat(ng[ii], 1, hlg[ii], 1, hd[ii]+2*pnb[ii], 1);
	dax_mat(ng[ii], 1, -1.0, hug[ii], 1, hd[ii]+2*pnb[ii]+png[ii], 1); // TODO change in solver
//	d_print_mat(1, 2*pnb[ii]+2*png[ii], hd[ii], 1);
	for(ii=1; ii<N; ii++)
		{
		if(hlb[ii]==hlb[ii-1] && hub[ii]==hub[ii-1] && hlg[ii]==hlg[ii-1] && hug[ii]==hug[ii-1])
			{
//			printf("\nsame\n");
			hd[ii] = hd[ii-1];
			}
		else
			{
//			printf("\ndifferent\n");
			hd[ii] = ptr;
			ptr += 2*pnb[ii]+2*png[ii];
			d_copy_mat(nb[ii], 1, hlb[ii], 1, hd[ii], 1);
			dax_mat(nb[ii], 1, -1.0, hub[ii], 1, hd[ii]+pnb[ii], 1); // TODO change in solver
			d_copy_mat(ng[ii], 1, hlg[ii], 1, hd[ii]+2*pnb[ii], 1);
			dax_mat(ng[ii], 1, -1.0, hug[ii], 1, hd[ii]+2*pnb[ii]+png[ii], 1); // TODO change in solver
			}
//		d_print_mat(1, 2*pnb[ii]+2*png[ii], hd[ii], 1);
		}
	ii = N;
	hd[ii] = ptr;
	ptr += 2*pnb[ii]+2*png[ii];
	d_copy_mat(nb[ii], 1, hlb[ii], 1, hd[ii], 1);
	dax_mat(nb[ii], 1, -1.0, hub[ii], 1, hd[ii]+pnb[ii], 1); // TODO change in solver
	d_copy_mat(ng[ii], 1, hlg[ii], 1, hd[ii]+2*pnb[ii], 1);
	dax_mat(ng[ii], 1, -1.0, hug[ii], 1, hd[ii]+2*pnb[ii]+png[ii], 1); // TODO change in solver
//	d_print_mat(1, 2*pnb[ii]+2*png[ii], hd[ii], 1);

	for(ii=0; ii<=N; ii++)
		{
		hux[ii] = ptr;
		ptr += pnux[ii];
		}

	for(ii=0; ii<=N; ii++)
		{
		hpi[ii] = ptr;
		ptr += pnx[ii];
		}
	
	for(ii=0; ii<=N; ii++)
		{
		ht[ii] = ptr;
		hlam[ii] = ptr + 2*pnb[ii] + 2*png[ii];
		ptr += 4*pnb[ii] + 4*png[ii];
		}
	
	stat = ptr;
	ptr += 5*k_max;

	// align work space (again) (assume cache line <= 64 byte)
	addr = (( (size_t) ptr ) + 63 ) / 64 * 64;
	//size_t offset = addr % 64;
	//double *ptr = work0 + offset / 8;
	work = (double *) addr;


	// estimate mu0 if not user-provided
	if(mu0<=0)
		{
		// first stage
		ii=0;
		for(jj=0; jj<nu[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, abs(hR[ii][jj*nu[ii]+ll]));
		for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, abs(hS[ii][jj*nu[ii]+ll]));
		for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, abs(hQ[ii][jj*nx[ii]+ll]));
		for(jj=0; jj<nu[ii]; jj++) mu0 = fmax(mu0, abs(hr[ii][jj]));
		for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, abs(hq[ii][jj]));
		// middle stages
		for(jj=1; jj<N; jj++)
			{
			if(hq[ii]==hq[ii-1] && hr[ii]==hr[ii-1] && hQ[ii]==hQ[ii-1] && hS[ii]==hS[ii-1] && hR[ii]==hR[ii-1])
				{
				for(jj=0; jj<nu[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, abs(hR[ii][jj*nu[ii]+ll]));
				for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nu[ii]; ll++) mu0 = fmax(mu0, abs(hS[ii][jj*nu[ii]+ll]));
				for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, abs(hQ[ii][jj*nx[ii]+ll]));
				for(jj=0; jj<nu[ii]; jj++) mu0 = fmax(mu0, abs(hr[ii][jj]));
				for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, abs(hq[ii][jj]));
				}
			}
		// last stage
		ii = N;
		for(jj=0; jj<nx[ii]; jj++) for(ll=0; ll<nx[ii]; ll++) mu0 = fmax(mu0, abs(hQ[ii][jj*nx[ii]+ll]));
		for(jj=0; jj<nx[ii]; jj++) mu0 = fmax(mu0, abs(hq[ii][jj]));
		}
	


	// TODO check for equality constraints in the inputs



	// call the solver
	int hpmpc_status =  d_ip2_hard_mpc_tv(&kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, N, nx, nu, nb, hidxb, ng, hpBAbt, hpRSQrq, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);

	if(hpmpc_status==1)
		acados_status = ACADOS_MAXITER;

	if(hpmpc_status==2)
		acados_status = ACADOS_MINSTEP;



	// copy back inputs and states
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nu[ii]; jj++)
			hu[ii][jj] = hux[ii][jj];

	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nx[ii]; jj++)
			hx[ii][jj] = hux[ii][nu[ii]+jj];
	


#if 0
	printf("\n");
	for(ii=0; ii<kk; ii++)
		printf("%d %e %e %e %e %e\n", ii, stat[0+ii*5], stat[1+ii*5], stat[2+ii*5], stat[3+ii*5], stat[4+ii*5]);
#endif


	// TODO check for equality constraints in the inputs



	// TODO compute residuals ?????



	// return

	return acados_status;

	


#if 0





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

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<=N; ii++) // time Variant box constraints
			{
			hpi[ii] = ptr;
			ptr += pnx; // for alignment of ptr
			}

		work = ptr;
		ptr += d_ip2_hard_mpc_tv_work_space_size_double(N, nxx, nuu, nbb, ngg);

		hd[0] = ptr;
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		for(ii=1; ii<N; ii++) // time Variant box constraints
			{
			hd[ii] = ptr;
			}
		ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

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

		if(compute_res)
			{

			for(ii=0; ii<=N; ii++)
				{
				hq[ii] = ptr;
				ptr += pnz;
				}

			for(ii=0; ii<=N; ii++)
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

			}



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx); // pack A into (temporary) buffer
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hpi[1]); // result in (temporary) buffer

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
		for (jj = 0; jj<nx; jj++)
			hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hpi[1][jj];

		// middle stages
		ii=1;
		if(jj<N)
			{
			d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, A, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for(jj=0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[jj];
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
				hd[jj][ii+0]    =   lb[ii];
				hd[jj][ii+pnb0] = - ub[ii];
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
				hd[jj][ii+0]    =   lb[ii] + 1e3;
				hd[jj][ii+pnb0] = - ub[ii] - 1e3;

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
					hd[jj][ii+0]    =   lb[ii];
					hd[jj][ii+pnb0] = - ub[ii];
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
					hd[jj][ii+0]    =   lb[ii] + 1e3;
					hd[jj][ii+pnb0] = - ub[ii] - 1e3;

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
				hd[jj][ii+0]    =   lb[ii];
				hd[jj][ii+pnb0] = - ub[ii];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    =   lb[nu+ii];
			hd[N][ii+pnb0] = - ub[nu+ii];
			//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
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
					hd[jj][2*pnb0+ii+0]    =   lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = - ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    =   lgf[ii];
				hd[N][2*pnb0+ii+png0] = - ugf[ii];
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

		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += pnz;
			}

		for(ii=0; ii<=N; ii++) // time Variant box constraints
			{
			hpi[ii] = ptr;
			ptr += pnx; // for alignment of ptr
			}

		work = ptr;
		ptr += d_ip2_hard_mpc_tv_work_space_size_double(N, nxx, nuu, nbb, ngg);

		for(ii=0; ii<N; ii++) // time Variant box constraints
			{
			hd[ii] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			}
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

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

		if(compute_res)
			{

			for(ii=0; ii<=N; ii++)
				{
				hq[ii] = ptr;
				ptr += pnz;
				}

			for(ii=0; ii<=N; ii++)
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

			}



		/* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

		// dynamic system
		// first stage
		// compute A_0 * x_0 + b_0
		for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
		d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx);
		dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hpi[1]);

		ii = 0;
		d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
		for (jj = 0; jj<nx; jj++)
			hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hpi[1][jj];

		// middle stages
		for(ii=1; ii<N; ii++)
			{
			d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
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
//				d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnux);
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
					hd[jj][ii+0]    =   lb[ii+nb*jj];
					hd[jj][ii+pnb0] = - ub[ii+nb*jj];
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
					hd[jj][ii+0]    =   lb[ii+nb*jj] + 1e3;
					hd[jj][ii+pnb0] = - ub[ii+nb*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=1; jj<N; jj++)
			{
			pnb0 = (nbb[jj]+bs-1)/bs*bs;
			for(ii=nuu[jj]; ii<nbb[jj]; ii++)
				{
				hd[jj][ii+0]    =   lb[ii+nb*jj];
				hd[jj][ii+pnb0] = - ub[ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}
		pnb0 = (nbb[N]+bs-1)/bs*bs;
		for(ii=nuu[N]; ii<nbb[N]; ii++)
			{
			hd[N][ii+0]    =   lb[nu+ii+nb*jj];
			hd[N][ii+pnb0] = - ub[nu+ii+nb*jj];
			//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
			//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
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
					hd[jj][2*pnb0+ii+0]    =   lg[ii+ng*jj];
					hd[jj][2*pnb0+ii+png0] = - ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			png0 = (ngg[N]+bs-1)/bs*bs;
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb0+ii+0]    =   lgf[ii];
				hd[N][2*pnb0+ii+png0] = - ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

		} // end of time variant



	// initial guess 
	for(jj=0; jj<N; jj++)
		for(ii=0; ii<nu; ii++)
			hux[jj][ii] = u[ii+nu*jj];

	for(jj=1; jj<=N; jj++)
		for(ii=0; ii<nx; ii++)
			hux[jj][nuu[jj]+ii] = x[ii+nx*jj];



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
	//if(IP==1)
	//    hpmpc_status = d_ip_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
	//else
	//    hpmpc_status = d_ip2_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
		hpmpc_status = d_ip2_hard_mpc_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);

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
	if(compute_res)
		{

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

		d_res_ip_hard_mpc_tv(N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

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

		}

	if(compute_mult)
		{

		for(ii=0; ii<=N; ii++)
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

		}



#if PC_DEBUG == 1
	for (jj = 0; jj < *kk; jj++)
		printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
			   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
			   stat[5 * jj + 2]);
	printf("\n");
#endif /* PC_DEBUG == 1 */


/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

#endif

	}



/* version dealing with equality constratins: is lb=ub, then fix the variable (corresponding column in A or B set to zero, and updated b) */
int fortran_order_ip_hard_mpc_tv( int *kk, int k_max, double mu0, double mu_tol, char prec,
                          int N, int nx, int nu, int nb, int ng, int ngN, 
						  int time_invariant,
                          double* A, double* B, double* b, 
                          double* Q, double* Qf, double* S, double* R, 
                          double* q, double* qf, double* r, 
						  double *lb, double *ub,
                          double *C, double *D, double *lg, double *ug,
						  double *Cf, double *lgf, double *ugf,
                          double* x, double* u,
						  double *work0, 
                          double *stat,
						  int compute_res, double *inf_norm_res,
						  int compute_mult, double *pi, double *lam, double *t)

	{

//printf("\nstart of wrapper\n");

	//printf("\n%d %d %d %d %d %d\n", N, nx, nu, nb, ng, ngN);

	int hpmpc_status = -1;

    //char prec = PREC;

    if(prec=='d')
	    {
	    
		//const int nb = nx+nu; // number of box constraints
		//const int ng = 0; // number of general constraints // TODO remove when not needed any longer
		const int nbu = nb<nu ? nb : nu ;

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;
	
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
//		const int anz  = nal*((nz+nal-1)/nal);
//		const int anx  = nal*((nx+nal-1)/nal);

		int pnb0;
		int png0;

		//const int anb = nal*((2*nb+nal-1)/nal);

		double alpha_min = ALPHA_MIN; // minimum accepted step length
        static double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
/*      static double stat[5*K_MAX]; // statistics from the IP routine*/
//      double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(double));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
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
        size_t align = 64;
        size_t addr = (( (size_t) work0 ) + 63 ) / 64 * 64;
        //size_t offset = addr % 64;
        //double *ptr = work0 + offset / 8;
		double *ptr = (double *) addr;


//printf("\n%d\n", ((size_t) ptr) & 63);

        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpDCt[N+1]);
        double *(hpQ[N+1]);
        double *(hux[N+1]);
        double *(hd[N+1]);
        double *(hpi[N+1]);
        double *(hlam[N+1]);
        double *(ht[N+1]);
		double *(hq[N+1]);
		double *(hrb[N]);
		double *(hrq[N+1]);
		double *(hrd[N+1]);
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

			for(ii=0; ii<=N; ii++)
				{
				hux[ii] = ptr;
				ptr += pnz;
				}

			for(ii=0; ii<=N; ii++) // time Variant box constraints
				{
				hpi[ii] = ptr;
				ptr += pnx; // for alignment of ptr
				}

			work = ptr;
			ptr += d_ip2_hard_mpc_tv_work_space_size_double(N, nxx, nuu, nbb, ngg);

			hd[0] = ptr;
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			for(ii=1; ii<N; ii++) // time Variant box constraints
				{
				hd[ii] = ptr;
				}
			ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
			hd[N] = ptr;
			ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

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

			if(compute_res)
				{

				for(ii=0; ii<=N; ii++)
					{
					hq[ii] = ptr;
					ptr += pnz;
					}

				for(ii=0; ii<=N; ii++)
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

				}



			/* pack matrices 	*/

			//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

			// dynamic system
			// first stage
			// compute A_0 * x_0 + b_0
			for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
			d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx); // pack A into (temporary) buffer
			dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hpi[1]); // result in (temporary) buffer

			ii = 0;
			d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hpi[1][jj];

			// middle stages
			ii=1;
			if(jj<N)
				{
				d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[ii], cnx);
				d_cvt_tran_mat2pmat(nx, nx, A, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
				for(jj=0; jj<nx; jj++)
					hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[jj];
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
					hd[jj][ii+0]    =   lb[ii];
					hd[jj][ii+pnb0] = - ub[ii];
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
					hd[jj][ii+0]    =   lb[ii] + 1e3;
					hd[jj][ii+pnb0] = - ub[ii] - 1e3;

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
						hd[jj][ii+0]    =   lb[ii];
						hd[jj][ii+pnb0] = - ub[ii];
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
						hd[jj][ii+0]    =   lb[ii] + 1e3;
						hd[jj][ii+pnb0] = - ub[ii] - 1e3;

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
					hd[jj][ii+0]    =   lb[ii];
					hd[jj][ii+pnb0] = - ub[ii];
					//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
					//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
					}
				}
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			for(ii=nuu[N]; ii<nbb[N]; ii++)
				{
				hd[N][ii+0]    =   lb[nu+ii];
				hd[N][ii+pnb0] = - ub[nu+ii];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
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
						hd[jj][2*pnb0+ii+0]    =   lg[ii+ng*jj];
						hd[jj][2*pnb0+ii+png0] = - ug[ii+ng*jj];
						}
					}
				}
			if(ngN>0) // last stage
				{
				pnb0 = (nbb[N]+bs-1)/bs*bs;
				png0 = (ngg[N]+bs-1)/bs*bs;
				for(ii=0; ii<ngN; ii++)
					{
					hd[N][2*pnb0+ii+0]    =   lgf[ii];
					hd[N][2*pnb0+ii+png0] = - ugf[ii];
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

			for(ii=0; ii<=N; ii++)
				{
				hux[ii] = ptr;
				ptr += pnz;
				}

			for(ii=0; ii<=N; ii++) // time Variant box constraints
				{
				hpi[ii] = ptr;
				ptr += pnx; // for alignment of ptr
				}

			work = ptr;
			ptr += d_ip2_hard_mpc_tv_work_space_size_double(N, nxx, nuu, nbb, ngg);

			for(ii=0; ii<N; ii++) // time Variant box constraints
				{
				hd[ii] = ptr;
				ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
				}
			hd[N] = ptr;
			ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

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

			if(compute_res)
				{

				for(ii=0; ii<=N; ii++)
					{
					hq[ii] = ptr;
					ptr += pnz;
					}

				for(ii=0; ii<=N; ii++)
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

				}



			/* pack matrices 	*/

			//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

			// dynamic system
			// first stage
			// compute A_0 * x_0 + b_0
			for(ii=0; ii<nx; ii++) hux[1][ii] = x[ii]; // copy x0 into aligned memory
			d_cvt_mat2pmat(nx, nx, A, nx, 0, hpBAbt[0], cnx);
			dgemv_n_lib(nx, nx, hpBAbt[0], cnx, hux[1], 1, b, hpi[1]);

			ii = 0;
			d_cvt_tran_mat2pmat(nx, nu, B, nx, 0, hpBAbt[0], cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[0][(nu)/bs*cnx*bs+(nu)%bs+jj*bs] = hpi[1][jj];

			// middle stages
			for(ii=1; ii<N; ii++)
				{
				d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
				d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
				for (jj = 0; jj<nx; jj++)
					hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
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
//				d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnux);
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
						hd[jj][ii+0]    =   lb[ii+nb*jj];
						hd[jj][ii+pnb0] = - ub[ii+nb*jj];
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
						hd[jj][ii+0]    =   lb[ii+nb*jj] + 1e3;
						hd[jj][ii+pnb0] = - ub[ii+nb*jj] - 1e3;

						}
					}
				}
			// state constraints 
			for(jj=1; jj<N; jj++)
				{
				pnb0 = (nbb[jj]+bs-1)/bs*bs;
				for(ii=nuu[jj]; ii<nbb[jj]; ii++)
					{
					hd[jj][ii+0]    =   lb[ii+nb*jj];
					hd[jj][ii+pnb0] = - ub[ii+nb*jj];
					//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
					//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
					}
				}
			pnb0 = (nbb[N]+bs-1)/bs*bs;
			for(ii=nuu[N]; ii<nbb[N]; ii++)
				{
				hd[N][ii+0]    =   lb[nu+ii+nb*jj];
				hd[N][ii+pnb0] = - ub[nu+ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
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
						hd[jj][2*pnb0+ii+0]    =   lg[ii+ng*jj];
						hd[jj][2*pnb0+ii+png0] = - ug[ii+ng*jj];
						}
					}
				}
			if(ngN>0) // last stage
				{
				pnb0 = (nbb[N]+bs-1)/bs*bs;
				png0 = (ngg[N]+bs-1)/bs*bs;
				for(ii=0; ii<ngN; ii++)
					{
					hd[N][2*pnb0+ii+0]    =   lgf[ii];
					hd[N][2*pnb0+ii+png0] = - ugf[ii];
					}
				}
			//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
			//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
			//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
			//exit(1);

			} // end of time variant



		// initial guess 
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				hux[jj][ii] = u[ii+nu*jj];

		for(jj=1; jj<=N; jj++)
			for(ii=0; ii<nx; ii++)
				hux[jj][nuu[jj]+ii] = x[ii+nx*jj];



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
	    //if(IP==1)
	    //    hpmpc_status = d_ip_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
	    //else
	    //    hpmpc_status = d_ip2_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
	        hpmpc_status = d_ip2_hard_mpc_tv(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);

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
		if(compute_res)
			{

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

			d_res_ip_hard_mpc_tv(N, nxx, nuu, nbb, idxb, ngg, hpBAbt, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

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

			}

		if(compute_mult)
			{

			for(ii=0; ii<=N; ii++)
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

			}



#if PC_DEBUG == 1
        for (jj = 0; jj < *kk; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


	    }
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

	}



/* version dealing with equality constratins: is lb=ub, then fix the variable (corresponding column in A or B set to zero, and updated b) */
int fortran_order_ip_hard_mpc( int *kk, int k_max, double mu0, double mu_tol, char prec,
                          int N, int nx, int nu, int nb, int ng, int ngN, 
                          double* A, double* B, double* b, 
                          double* Q, double* Qf, double* S, double* R, 
                          double* q, double* qf, double* r, 
						  double *lb, double *ub,
                          double *C, double *D, double *lg, double *ug,
						  double *Cf, double *lgf, double *ugf,
                          double* x, double* u,
						  double *work0, 
                          double *stat,
						  int compute_res, double *inf_norm_res,
						  int compute_mult, double *pi, double *lam, double *t)

	{

//printf("\nstart of wrapper\n");

	//printf("\n%d %d %d %d %d %d\n", N, nx, nu, nb, ng, ngN);

	int hpmpc_status = -1;

    //char prec = PREC;

    if(prec=='d')
	    {
	    
		//const int nb = nx+nu; // number of box constraints
		//const int ng = 0; // number of general constraints // TODO remove when not needed any longer
		const int nbu = nb<nu ? nb : nu ;

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;
	
		const int nz   = nx+nu+1;
		const int pnz  = bs*((nz+bs-1)/bs);
		const int pnu  = bs*((nu+bs-1)/bs);
		const int pnx  = bs*((nx+bs-1)/bs);
		const int pnb  = bs*((nb+bs-1)/bs);
		const int png  = bs*((ng+bs-1)/bs);
		const int pngN = bs*((ngN+bs-1)/bs);
		const int cnz  = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx  = ncl*((nx+ncl-1)/ncl);
		const int cng  = ncl*((ng+ncl-1)/ncl);
		const int cngN = ncl*((ngN+ncl-1)/ncl);
		const int anz  = nal*((nz+nal-1)/nal);
		const int anx  = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		//const int anb = nal*((2*nb+nal-1)/nal);

		double alpha_min = ALPHA_MIN; // minimum accepted step length
        static double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
/*      static double stat[5*K_MAX]; // statistics from the IP routine*/
//      double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(double));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
		double temp;
        
        int info = 0;

        int i, ii, jj, ll;



        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;



        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpDCt[N+1]);
        double *(hpQ[N+1]);
        double *(hux[N+1]);
        double *(hd[N+1]);
        double *(hpi[N+1]);
        double *(hlam[N+1]);
        double *(ht[N+1]);
		double *(hq[N+1]);
		double *(hrb[N]);
		double *(hrq[N+1]);
		double *(hrd[N+1]);
		double *work;

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
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<N; ii++) // time Variant box constraints
	        {
            hd[ii] = ptr;
            ptr += 2*pnb+2*png; //anb; //nb; // for alignment of ptr
	        }
		hd[N] = ptr;
		ptr += 2*pnb+2*pngN; //anb; //nb; // for alignment of ptr

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
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

		if(compute_res)
			{

			for(ii=0; ii<=N; ii++)
				{
				hq[ii] = ptr;
				ptr += anz;
				}

			for(ii=0; ii<=N; ii++)
				{
				hrb[ii] = ptr;
				ptr += anx;
				}

			for(ii=0; ii<=N; ii++)
				{
				hrq[ii] = ptr;
				ptr += anz;
				}

			for(ii=0; ii<N; ii++)
				{
				hrd[ii] = ptr;
				ptr += 2*pnb+2*png;
				}
			hrd[N] = ptr;
			ptr += 2*pnb+2*pngN;

			}

		work = ptr;



        /* pack matrices 	*/

		//printf("\n%d %d %d %d %d\n", N, nx, nu, nb, ng);

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
		//d_print_mat(nx, nu, B, nx);
		//d_print_mat(nx, nx, A, nx);
		//d_print_pmat(nz, nx, bs, hpBAbt[0], cnx);
		//d_print_pmat(nz, nx, bs, hpBAbt[N-1], cnx);
		//exit(1);

		// general constraints
		if(ng>0)
			{
			for(ii=0; ii<N; ii++)
				{
				d_cvt_tran_mat2pmat(ng, nu, D+ii*nu*ng, ng, 0, hpDCt[ii], cng);
				d_cvt_tran_mat2pmat(ng, nx, C+ii*nx*ng, ng, nu, hpDCt[ii]+nu/bs*cng*bs+nu%bs, cng);
				}
			}
		if(ngN>0)
			{
			for(ii=0; ii<pnu*cngN; ii++) hpDCt[N][ii] = 0.0; // make sure D is zero !!!!!
			d_cvt_tran_mat2pmat(ngN, nx, Cf, ngN, nu, hpDCt[N]+nu/bs*cngN*bs+nu%bs, cngN);
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
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_mat2pmat(nx, nx, Qf, nx, nu, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// estimate mu0 if not user-provided
		if(mu0<=0)
			{
			for(jj=0; jj<N; jj++)
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

		//d_print_pmat(nz, nz, bs, hpQ[0], cnz);
		//d_print_pmat(nz, nz, bs, hpQ[1], cnz);
		//d_print_pmat(nz, nz, bs, hpQ[N], cnz);
		//exit(1);

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nbu; ii++)
				{
				if(lb[ii+nb*jj]!=ub[ii+nb*jj]) // equality constraint
					{
					hd[jj][ii+0]   =   lb[ii+nb*jj];
					hd[jj][ii+pnb] = - ub[ii+nb*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nb*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hd[jj][ii+0]   =   lb[ii+nb*jj] + 1e3;
					hd[jj][ii+pnb] = - ub[ii+nb*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=1; jj<=N; jj++)
			{
			for(ii=nbu; ii<nb; ii++)
				{
				hd[jj][ii+0]   =   lb[ii+nb*jj];
				hd[jj][ii+pnb] = - ub[ii+nb*jj];
				//hd[jj][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				//hd[jj][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}
		// general constraints
		if(ng>0)
			{
			for(jj=0; jj<N; jj++)
				{
				for(ii=0; ii<ng; ii++)
					{
					hd[jj][2*pnb+ii+0]   =   lg[ii+ng*jj];
					hd[jj][2*pnb+ii+png] = - ug[ii+ng*jj];
					}
				}
			}
		if(ngN>0) // last stage
			{
			for(ii=0; ii<ngN; ii++)
				{
				hd[N][2*pnb+ii+0]    =   lgf[ii];
				hd[N][2*pnb+ii+pngN] = - ugf[ii];
				}
			}
		//d_print_mat(1, 2*pnb+2*png, hd[0], 1);
		//d_print_mat(1, 2*pnb+2*png, hd[1], 1);
		//d_print_mat(1, 2*pnb+2*pngN, hd[N], 1);
		//exit(1);

        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



        // call the IP solver
	    if(IP==1)
	        hpmpc_status = d_ip_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);
	    else
	        hpmpc_status = d_ip2_hard_mpc(kk, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hpDCt, hd, hux, compute_mult, hpi, hlam, ht, work);



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];



		// check for input and states equality constraints
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

		// compute infinity norm of residuals on exit
		if(compute_res)
			{

			// restore linear part of cost function 
			for(ii=0; ii<N; ii++)
				{
				for(jj=0; jj<nu; jj++) 
					hq[ii][jj]    = r[jj+nu*ii];
				for(jj=0; jj<nx; jj++) 
					hq[ii][nu+jj] = q[jj+nx*ii];
				}
			for(jj=0; jj<nx; jj++) 
				hq[N][nu+jj] = qf[jj];

			double mu;

			d_res_ip_hard_mpc(nx, nu, N, nb, ng, ngN, hpBAbt, hpQ, hq, hux, hpDCt, hd, hpi, hlam, ht, hrq, hrb, hrd, &mu);

			temp = fabs(hrq[0][0]);
			for(jj=0; jj<nu; jj++) 
				temp = fmax( temp, fabs(hrq[0][jj]) );
			for(ii=1; ii<N; ii++)
				for(jj=0; jj<nu+nx; jj++) 
					temp = fmax( temp, fabs(hrq[ii][jj]) );
			for(jj=nu; jj<nu+nx; jj++) 
				temp = fmax( temp, fabs(hrq[N][jj]) );
			inf_norm_res[0] = temp;

			temp = fabs(hrb[0][0]);
			for(ii=0; ii<N; ii++)
				for(jj=0; jj<nx; jj++) 
					temp = fmax( temp, fabs(hrb[ii][jj]) );
			inf_norm_res[1] = temp;

			temp = fabs(hrd[0][0]);
			for(jj=0; jj<nbu; jj++) 
				{
				temp = fmax( temp, fabs(hrd[0][jj]) );
				temp = fmax( temp, fabs(hrd[0][pnb+jj]) );
				}
			for(ii=1; ii<N; ii++)
				for(jj=0; jj<nb; jj++) 
					{
					temp = fmax( temp, fabs(hrd[ii][jj]) );
					temp = fmax( temp, fabs(hrd[ii][pnb+jj]) );
					}
			for(jj=nbu; jj<nb; jj++) 
				{
				temp = fmax( temp, fabs(hrq[N][jj]) );
				temp = fmax( temp, fabs(hrq[N][pnb+jj]) );
				}
			for(ii=0; ii<N; ii++)
				for(jj=2*pnb; jj<2*pnb+ng; jj++) 
					{
					temp = fmax( temp, fabs(hrd[ii][jj]) );
					temp = fmax( temp, fabs(hrd[ii][png+jj]) );
					}
			for(jj=2*pnb; jj<2*pnb+ngN; jj++) 
				{
				temp = fmax( temp, fabs(hrd[N][jj]) );
				temp = fmax( temp, fabs(hrd[N][pngN+jj]) );
				}
			inf_norm_res[2] = temp;

			inf_norm_res[3] = mu;

			//printf("\n%e %e %e %e\n", norm_res[0], norm_res[1], norm_res[2], norm_res[3]);

			}

		if(compute_mult)
			{

			for(ii=0; ii<=N; ii++)
				for(jj=0; jj<nx; jj++)
					pi[jj+ii*nx] = hpi[ii][jj];

			ii = 0;
			for(jj=0; jj<nbu; jj++)
				{
				lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
				lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb+jj];
				t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
				t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb+jj];
				}
			for(ii=1; ii<N; ii++)
				{
				for(jj=0; jj<nb; jj++)
					{
					lam[jj+ii*(2*nb+2*ng)]       = hlam[ii][jj];
					lam[jj+ii*(2*nb+2*ng)+nb+ng] = hlam[ii][pnb+jj];
					t[jj+ii*(2*nb+2*ng)]       = ht[ii][jj];
					t[jj+ii*(2*nb+2*ng)+nb+ng] = ht[ii][pnb+jj];
					}
				}
			ii = N;
			for(jj=nbu; jj<nb; jj++)
				{
				lam[jj+ii*(2*nb+2*ng)]        = hlam[ii][jj];
				lam[jj+ii*(2*nb+2*ng)+nb+ngN] = hlam[ii][pnb+jj];
				t[jj+ii*(2*nb+2*ng)]        = ht[ii][jj];
				t[jj+ii*(2*nb+2*ng)+nb+ngN] = ht[ii][pnb+jj];
				}

			for(ii=0; ii<N; ii++)
				for(jj=0; jj<ng; jj++)
					{
					lam[jj+ii*(2*nb+2*ng)+nb]       = hlam[ii][2*pnb+jj];
					lam[jj+ii*(2*nb+2*ng)+nb+ng+nb] = hlam[ii][2*pnb+png+jj];
					t[jj+ii*(2*nb+2*ng)+nb]       = ht[ii][2*pnb+jj];
					t[jj+ii*(2*nb+2*ng)+nb+ng+nb] = ht[ii][2*pnb+png+jj];
					}
			for(jj=0; jj<ngN; jj++)
				{
				lam[jj+N*(2*nb+2*ng)+nb]        = hlam[N][2*pnb+jj];
				lam[jj+N*(2*nb+2*ng)+nb+ngN+nb] = hlam[N][2*pnb+pngN+jj];
				t[jj+N*(2*nb+2*ng)+nb]        = ht[N][2*pnb+jj];
				t[jj+N*(2*nb+2*ng)+nb+ngN+nb] = ht[N][2*pnb+pngN+jj];
				}

			}



#if PC_DEBUG == 1
        for (jj = 0; jj < *kk; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


	    }
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		float alpha_min = ALPHA_MIN; // minimum accepted step length
        static float sigma_par[] = {0.4, 0.1, 0.01}; // control primal-dual IP behaviour
/*        static float stat[5*K_MAX]; // statistics from the IP routine*/
        //float *work0 = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(float));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = ((float *) work0) + offset / 4;



        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
        float *(hdb[N + 1]);
        float *(hpi[N + 1]);
        float *(hlam[N + 1]);
        float *(ht[N + 1]);
		float *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj] - 1e3;

/*		            d_print_pmat(nx+nu, nx, bs, hpBAbt[jj], cnx);*/
					}
				}
			}
		// state constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] = (float)   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = (float) - ub[N*nu+ii+nx*jj];
				}
			}


        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the IP solver
		if(IP==1)
		    hpmpc_status = s_ip_box_mpc(kk, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
		else
		    hpmpc_status = s_ip2_box_mpc(kk, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];

		// check for input equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}


#if PC_DEBUG == 1
        for (jj = 0; jj < *kk; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

	}



int fortran_order_ip_soft_mpc( int k_max, double mu_tol, const char prec,
                               const int nx, const int nu, const int N,
                               double* A, double* B, double* b, 
                               double* Q, double* Qf, double* S, double* R, 
                               double* q, double* qf, double* r, 
                               double* lZ, double* uZ,
                               double* lz, double* uz,
                               double* lb, double* ub, 
                               double* x, double* u,
                               double* work0, 
                               int* nIt, double* stat )

	{

//printf("\nstart of wrapper\n");

	int hpmpc_status = -1;

    //char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;
	
		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		double alpha_min = ALPHA_MIN; // minimum accepted step length
        static double sigma_par[] = {0.4, 0.1, 0.001}; // control primal-dual IP behaviour
/*      static double stat[5*K_MAX]; // statistics from the IP routine*/
//      double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(double));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;



        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;



        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N + 1]);
		double *(hZ[N + 1]);
		double *(hz[N + 1]);
        double *(hux[N + 1]);
        double *(hdb[N + 1]);
        double *(hpi[N + 1]);
        double *(hlam[N + 1]);
        double *(ht[N + 1]);
		double *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

		for(ii=0; ii<=N; ii++)
	        {
			hZ[ii] = ptr;
			ptr += anb;
			}

		for(ii=0; ii<=N; ii++)
	        {
			hz[ii] = ptr;
			ptr += anb;
			}

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += 2*anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += 2*anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }

        // cost function
		double mu0 = 1.0;
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnz);
			for(ii=0; ii<nu; ii++) for(ll=0; ll<nu; ll++) mu0 = fmax(mu0, R[jj*nu*nu+ii*nu+ll]);
            d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			for(ii=0; ii<nx*nu; ii++) mu0 = fmax(mu0, S[jj*nu*nx+ii]);
            d_cvt_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Q[jj*nx*nx+ii*nx+ll]);
            for(ii=0; ii<nu; ii++)
				{
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
				mu0 = fmax(mu0, r[jj*nu+ii]);
				}
            for(ii=0; ii<nx; ii++)
				{
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
				mu0 = fmax(mu0, q[jj*nx+ii]);
				}
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_mat2pmat(nx, nx, Qf, nx, nu, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(ii=0; ii<nx; ii++) for(ll=0; ll<nx; ll++) mu0 = fmax(mu0, Qf[ii*nx+ll]);
        for(jj=0; jj<nx; jj++)
			{
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];
			mu0 = fmax(mu0, qf[ii]);
			}

		// cost function of soft constraint slack variable
		for(ii=0; ii<N; ii++)
			{
			for(jj=0; jj<nx; jj++)
				{
				hZ[ii+1][2*nu+2*jj+0] = lZ[nx*ii+jj];
				hZ[ii+1][2*nu+2*jj+1] = uZ[nx*ii+jj];
				mu0 = fmax(mu0, lZ[nx*ii+jj]);
				mu0 = fmax(mu0, uZ[nx*ii+jj]);
				}
			for(jj=0; jj<nx; jj++)
				{
				hz[ii+1][2*nu+2*jj+0] = lz[nx*ii+jj];
				hz[ii+1][2*nu+2*jj+1] = uz[nx*ii+jj];
				mu0 = fmax(mu0, lz[nx*ii+jj]);
				mu0 = fmax(mu0, uz[nx*ii+jj]);
				}
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] =   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] =   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = - ub[ii+nu*jj] - 1e3;

					}
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] =   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = - ub[N*nu+ii+nx*jj];
				}
			}

        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];




        // call the IP solver
	    if(IP==1)
	        hpmpc_status = d_ip_soft_mpc(nIt, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nu, nb-nu, hpBAbt, hpQ, hZ, hz, hdb, hux, compute_mult, hpi, hlam, ht, work);
	    else
	        hpmpc_status = d_ip2_soft_mpc(nIt, k_max, mu0, mu_tol, alpha_min, warm_start, sigma_par, stat, nx, nu, N, nu, nb-nu, hpBAbt, hpQ, hZ, hz, hdb, hux, compute_mult, hpi, hlam, ht, work);



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

		// check for input and states equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}



#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


	    }
#if 0
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
		const int anb = nal*((2*nb+nal-1)/nal);

		float alpha_min = ALPHA_MIN; // minimum accepted step length
        static float sigma_par[] = {0.4, 0.1, 0.01}; // control primal-dual IP behaviour
/*        static float stat[5*K_MAX]; // statistics from the IP routine*/
        //float *work0 = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 5*anz + 3*anx + 7*anb) + 3*anz)*sizeof(float));
        int warm_start = WARM_START;
        int compute_mult = 1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = ((float *) work0) + offset / 4;



        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
        float *(hdb[N + 1]);
        float *(hpi[N + 1]);
        float *(hlam[N + 1]);
        float *(ht[N + 1]);
		float *work;

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hdb[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hlam[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            ht[ii] = ptr;
            ptr += anb; //nb; // for alignment of ptr
	        }

		work = ptr;



        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]!=ub[ii+nu*jj]) // equality constraint
					{
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj];
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj];
					}
				else
					{
					for(ll=0; ll<nx; ll++)
						{
						// update linear term
						hpBAbt[jj][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+ll*bs] += hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs]*lb[ii+nu*jj];
						// zero corresponding B column
						hpBAbt[jj][ii/bs*cnx*bs+ii%bs+ll*bs] = 0;
						}
					
					// inactive box constraints
					hdb[jj][2*ii+0] = (float)   lb[ii+nu*jj] + 1e3;
					hdb[jj][2*ii+1] = (float) - ub[ii+nu*jj] - 1e3;

/*		            d_print_pmat(nx+nu, nx, bs, hpBAbt[jj], cnx);*/
					}
				}
			}
		// state constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hdb[jj+1][2*nu+2*ii+0] = (float)   lb[N*nu+ii+nx*jj];
				hdb[jj+1][2*nu+2*ii+1] = (float) - ub[N*nu+ii+nx*jj];
				}
			}


        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the IP solver
		if(IP==1)
		    hpmpc_status = s_ip_box_mpc(nIt, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);
		else
		    hpmpc_status = s_ip2_box_mpc(nIt, k_max, mu_tol, alpha_min, warm_start, sigma_par, (float *) stat, nx, nu, N, nb, hpBAbt, hpQ, hdb, hux, compute_mult, hpi, hlam, ht, work);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];

		// check for input equality constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				if(lb[ii+nu*jj]==ub[ii+nu*jj]) // equality constraint
					{
	                u[ii+nu*jj] = lb[ii+nu*jj];
					}
				}
			}


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */


		}
#endif
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
/*printf("\nend of wrapper\n");*/

    return hpmpc_status;

	}



int fortran_order_riccati_mpc( const char prec,
                               const int nx, const int nu, const int N,
                               double *A, double *B, double *b, 
                               double *Q, double *Qf, double *S, double *R, 
                               double *q, double *qf, double *r, 
                               double *x, double *u, double *pi, 
                               double *work_space )
	{

	//char prec = PREC;

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		//const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;

		//double *work = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz)*sizeof(double));

		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;


		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work_space;
		size_t offset = addr % 64;
		double *ptr = work_space + offset / 8;

		/* array or pointers */
		double *(hpBAbt[N]);
		double *(hpQ[N + 1]);
		double *(hpL[N + 1]);
		double *(hdL[N + 1]);
		double *(hpl[N + 1]);
		double *(hux[N + 1]);
		double *(hpi[N + 1]);
		double *work0;
		double *work1;

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		// cost function
		for(ii=0; ii<=N; ii++)
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnz;
			}

		// work space (matrices)
		for(jj=0; jj<=N; jj++)
			{
			hpL[jj] = ptr;
			hdL[jj] = ptr + pnz*cnl;
			ptr += pnz*cnl;
			}

		// work space (vectors)
		for(jj=0; jj<=N; jj++)
			{
			hpl[jj] = ptr;
			ptr += anz;
			}

		// states and inputs
		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += anz;
			}
		
		// eq. constr. multipliers
        for(ii=0; ii<=N; ii++) 
			{
			hpi[ii] = ptr;
			ptr += anx;
			}

		// inverted diagonal
		work1 = ptr;
		ptr += anz;

		// work space
		work0 = ptr;
		ptr += 2*anz;



		/* pack matrices 	*/

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
			d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
			}

		// cost function
		for(jj=0; jj<N; jj++)
			{
			d_cvt_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnz);
			d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			d_cvt_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
			}

		for(jj=0; jj<nu; jj++)
			for(ii=0; ii<nz; ii+=bs)
				for(i=0; i<bs; i++)
					hpQ[N][ii*cnz+i+jj*bs] = 0.0;
		for(jj=0; jj<nu; jj++)
			hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
		d_cvt_mat2pmat(nx, nx, Qf, nx, nu, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(jj=0; jj<nx; jj++)
			hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];



		// initial state
		for(ii=0; ii<nx; ii++)
            hux[0][nu+ii] = x[ii];


		// TODO
		double **dummy;

		// call Riccati solver
		d_back_ric_sv_new(N, nx, nu, hpBAbt, hpQ, 0, dummy, dummy, 1, hux, hpL, hdL, work0, work1, 0, dummy, compute_mult, hpi, 0, 0, 0, dummy, dummy, dummy);



		// copy back inputs
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				u[ii+nu*jj] = hux[jj][ii];

		// copy back states
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				x[ii+nx*(jj+1)] = hux[jj+1][nu+ii];

		// copy back lagrangian multipliers
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				pi[ii+nx*jj] = hpi[jj+1][ii];


		
		}
    else if(prec=='s')
	    {
	    
		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;
	
		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		//float *work = (float *) malloc((16 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 2*anz + 2*anx) + 3*anz)*sizeof(float));

		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;


		/* align work space */
		size_t align = 64; // max cache line size for all supported architectures
		size_t addr = (size_t) work_space;
		size_t offset = addr % 64;
		float *ptr = ((float *) work_space) + offset / 8;

		/* array or pointers */
		float *(hpBAbt[N]);
		float *(hpQ[N + 1]);
		float *(hpL[N + 1]);
		float *(hpl[N + 1]);
		float *(hux[N + 1]);
		float *(hpi[N + 1]);
		float *diag;
		float *work;

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			hpBAbt[ii] = ptr;
			ptr += pnz*cnx;
			}

		// cost function
		for(ii=0; ii<=N; ii++)
			{
			hpQ[ii] = ptr;
			ptr += pnz*cnz;
			}

		// work space (matrices)
		for(jj=0; jj<=N; jj++)
			{
			hpL[jj] = ptr;
			ptr += pnz*cnl;
			}

		// work space (vectors)
		for(jj=0; jj<=N; jj++)
			{
			hpl[jj] = ptr;
			ptr += anz;
			}

		// states and inputs
		for(ii=0; ii<=N; ii++)
			{
			hux[ii] = ptr;
			ptr += anz;
			}
		
		// eq. constr. multipliers
        for(ii=0; ii<=N; ii++) 
			{
			hpi[ii] = ptr;
			ptr += anx;
			}

		// inverted diagonal
		diag = ptr;
		ptr += anz;

		// work space
		work = ptr;
		ptr += 2*anz;



		/* pack matrices 	*/

		// dynamic system
		for(ii=0; ii<N; ii++)
			{
			cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
			cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
			for (jj = 0; jj<nx; jj++)
				hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
			}

		// cost function
		for(jj=0; jj<N; jj++)
			{
			cvt_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
			cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
			cvt_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
			for(ii=0; ii<nu; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
			for(ii=0; ii<nx; ii++)
				hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
			}

		for(jj=0; jj<nu; jj++)
			for(ii=0; ii<nz; ii+=bs)
				for(i=0; i<bs; i++)
					hpQ[N][ii*cnz+i+jj*bs] = 0.0;
		for(jj=0; jj<nu; jj++)
			hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
		cvt_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
		for(jj=0; jj<nx; jj++)
			hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];



		// initial state
		for(ii=0; ii<nx; ii++)
            hux[0][nu+ii] = (float) x[ii];
        


		// call Riccati solver
		s_ric_sv_mpc(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, compute_mult, hpi);



		// copy back inputs
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nu; ii++)
				u[ii+nu*jj] = (double) hux[jj][ii];

		// copy back states
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				x[ii+nx*(jj+1)] = (double) hux[jj+1][nu+ii];

		// copy back lagrangian multipliers
		for(jj=0; jj<N; jj++)
			for(ii=0; ii<nx; ii++)
				pi[ii+nx*jj] = (double) hpi[jj+1][ii];


		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}
	
	return 0;
	
	}



int fortran_order_riccati_mhe( const char prec, const int smooth,
                               const int nx, const int nw, const int ny, const int N,
                               double *A, double *G, double *C, double *f, 
                               double *Q, double *R, double *q, double *r, 
                               double *y, double *x0, double *L0,
                               double *xe, double *Le, double *w, double *lam,
                               double *work0 )
	{

//	printf("\nenter wrapper\n");

	int hpmpc_status = -1;

	//char prec = 'd';

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = bs*ncl;

		const int nt = nx+ny; 
		const int ant = nal*((nt+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);
		const int anw = nal*((nw+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnt = bs*((nt+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pnw = bs*((nw+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int cnt = ncl*((nt+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnw = ncl*((nw+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);
		const int cnf = cnt<cnx+ncl ? cnx+ncl : cnt;

		const int pad = (ncl-(nx+nw)%ncl)%ncl; // packing between AGL & P
		const int cnj = nx+nw+pad+cnx;

		//double *work0 = (double *) malloc((8 + (N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx)*sizeof(double));
//		printf("\nwork space allocated\n");

//		int compute_mult = 1; // compute multipliers

		int i, ii, jj, ll;



		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		double *ptr = work0 + offset / 8;

		//for(ii=0; ii<(N+1)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; ii++)
		//	ptr[ii] = 0.0;

		/* array or pointers */
		double *(hpA[N]);
		double *(hpG[N]);
		double *(hpC[N+1]);
		double *(hf[N]);
		double *(hpQ[N]);
		double *(hpR[N+1]);
		double *(hq[N]);
		double *(hr[N+1]);
		double *(hpLp[N+1]);
		double *(hdLp[N+1]);
		double *(hpLe[N+1]);
		double *(hxe[N+1]);
		double *(hxp[N+1]);
		double *(hw[N]);
		double *(hy[N+1]);
		double *(hlam[N]);

		double *diag;
		double *work;


		for(ii=0; ii<N; ii++)
			{
			// dynamic system
			hpA[ii] = ptr;
			ptr += pnx*cnx;
			hpG[ii] = ptr;
			ptr += pnx*cnw;
			hpC[ii] = ptr;
			ptr += pny*cnx;
			hf[ii] = ptr;
			ptr += anx;
			// cost function
			hpQ[ii] = ptr;
			ptr += pnw*cnw;
			hpR[ii] = ptr;
			ptr += pny*cny;
			hq[ii] = ptr;
			ptr += anw;
			hr[ii] = ptr;
			ptr += any;
			// covariances
			hpLp[ii] = ptr;
			ptr += pnx*cnj;
			hdLp[ii] = ptr;
			ptr += anx;
			hpLe[ii] = ptr;
			ptr += pnt*cnf;
			// estimates and measurements
			hxe[ii] = ptr;
			ptr += anx;
			hxp[ii] = ptr;
			ptr += anx;
			hw[ii] = ptr;
			ptr += anw;
			hy[ii] = ptr;
			ptr += any;
			hlam[ii] = ptr;
			ptr += anx;

		//if(ii==1)
		//for(jj=0; jj<(N-ii)*(pnx*cnx+pnx*cnw+pny*cnx+5*anx+pnw*cnw+pny*cny+2*anw+2*any+pnx*cnj+pnt*cnf) + 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; jj++)
		//	ptr[jj] = 0.0;

			}
		// stage N
		// dynamic system
		hpC[N] = ptr;
		ptr += pny*cnx;
		// cost function
		hpR[N] = ptr;
		ptr += pny*cny;
		hr[N] = ptr;
		ptr += any;
		// covariances
		hpLp[N] = ptr;
		ptr += pnx*cnj;
		hdLp[N] = ptr;
		ptr += anx;
		hpLe[N] = ptr;
		ptr += pnt*cnf;
		// estimates and measurements
		hxe[N] = ptr;
		ptr += anx;
		hxp[N] = ptr;
		ptr += anx;
		hy[N] = ptr;
		ptr += any;

		// diagonal backup
		diag = ptr;
		ptr += ant;

		// work space
		work = ptr;
		ptr += 2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx;
		//for(ii=0; ii<2*pny*cnx+pnt*cnt+ant+pnw*cnw+pnx*cnx; ii++)
		//	work[ii] = 0.0;

//		printf("\nmatrix space allocated\n");


		// convert into panel matrix format

		// stage 0
		// covariances
		//d_print_mat(nx, nx, L0, nx);
		d_cvt_mat2pmat(nx, nx, L0, nx, 0, hpLp[0]+(nx+nw+pad)*bs, cnj);
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[0], cnj);
		// estimates 
		for(jj=0; jj<nx; jj++) hxp[0][jj] = x0[jj];
		//d_print_mat(1, nx, hxp[0], 1);
		// stages 0 to N-1
		for(ii=0; ii<N; ii++)
			{
			//printf("\nii = %d\n", ii);
			// dynamic system
			//d_print_mat(nx, nx, A+ii*nx*nx, nx);
			d_cvt_mat2pmat(nx, nx, A+ii*nx*nx, nx, 0, hpA[ii], cnx);
			d_cvt_mat2pmat(nx, nw, G+ii*nx*nw, nx, 0, hpG[ii], cnw);
			d_cvt_mat2pmat(ny, nx, C+ii*ny*nx, ny, 0, hpC[ii], cnx);
			for(jj=0; jj<nx; jj++) hf[ii][jj] = f[ii*nx+jj];
			// cost function
			d_cvt_mat2pmat(nw, nw, Q+ii*nw*nw, nw, 0, hpQ[ii], cnw);
			d_cvt_mat2pmat(ny, ny, R+ii*ny*ny, ny, 0, hpR[ii], cny);
			for(jj=0; jj<nw; jj++) hq[ii][jj] = q[ii*nw+jj];
			for(jj=0; jj<ny; jj++) hr[ii][jj] = r[ii*ny+jj];
			// measurements
			for(jj=0; jj<ny; jj++) hy[ii][jj] = y[ii*ny+jj];
			}
		// stage N
		// dynamic system
		d_cvt_mat2pmat(ny, nx, C+N*ny*nx, ny, 0, hpC[N], cnx);
		// cost function
		d_cvt_mat2pmat(ny, ny, R+N*ny*ny, ny, 0, hpR[N], cny);
		for(jj=0; jj<ny; jj++) hr[N][jj] = r[N*ny+jj];
		// measurements
		for(jj=0; jj<ny; jj++) hy[N][jj] = y[N*ny+jj];
		//d_print_mat(1, ny, hy[0], 1);

#if 0
		printf("\nmatrices converted\n");
		printf("\nn = 0\n");
		d_print_pmat(nx, nx, bs, hpA[0], cnx);
		d_print_pmat(nx, nw, bs, hpG[0], cnw);
		d_print_pmat(ny, nx, bs, hpC[0], cnx);
		d_print_pmat(nw, nw, bs, hpQ[0], cnw);
		d_print_pmat(ny, ny, bs, hpR[0], cny);
		d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[0], cnj);
		d_print_pmat(nt, nt, bs, hpLe[0], cnf);
		printf("\nn = 1\n");
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
		d_print_pmat(nx, nx, bs, hpA[1], cnx);
		d_print_pmat(nx, nw, bs, hpG[1], cnw);
		d_print_pmat(ny, nx, bs, hpC[1], cnx);
		d_print_pmat(nw, nw, bs, hpQ[1], cnw);
		d_print_pmat(ny, ny, bs, hpR[1], cny);
		d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[1], cnj);
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
#endif


		// call Riccati solver
		// factorization
		d_ric_trf_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, work);
		// solution
		//d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, 0, hlam, work1);
		// smoothed solution
		hpmpc_status = d_ric_trs_mhe(nx, nw, ny, N, hpA, hpG, hpC, hpLp, hdLp, hpQ, hpR, hpLe, hq, hr, hf, hxp, hxe, hw, hy, smooth, hlam, work);

#if 0
		//d_print_pmat(nx, nx+nw+pad+nx, bs, hpLp[N], cnj);
		//d_print_pmat(nt, nt, bs, hpLe[N], cnf);
		printf("\nxp = \n");
		d_print_mat(1, nx, hxp[0], 1);
		d_print_mat(1, nx, hxp[1], 1);
		d_print_mat(1, nx, hxp[2], 1);
		printf("\nxe = \n");
		d_print_mat(1, nx, hxe[0], 1);
		d_print_mat(1, nx, hxe[1], 1);
		d_print_mat(1, nx, hxe[2], 1);
		printf("\nsystem solved\n");
		d_print_pmat(nx, nx, bs, hpLp[0]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nx, nx, bs, hpLp[1]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nx, nx, bs, hpLp[2]+(nx+nw+pad)*bs, cnj);
		d_print_pmat(nt, nt, bs, hpLe[0], cnf);
		d_print_pmat(nt, nt, bs, hpLe[1], cnf);
		d_print_pmat(nt, nt, bs, hpLe[2], cnf);
		return;
#endif


		// copy back estimate and covariance at first stage (Extended Kalma Filter update of initial condition)
		for(jj=0; jj<nx; jj++) x0[jj] = hxp[1][jj];
		d_cvt_pmat2mat(nx, nx, 0, hpLp[1]+(nx+nw+pad)*bs, cnj, L0, nx);


		// copy back estimates at all stages 0,1,...,N
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				xe[ii*nx+jj] = hxe[ii][jj];

		// copy back process disturbance at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nw; jj++)
				w[ii*nw+jj] = hw[ii][jj];
			
		// copy back mulipliers at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++)
				lam[ii*nx+jj] = hlam[ii][jj];

		// copy back covariance at last stage
		d_cvt_pmat2mat(nx, nx, ny, hpLe[N]+(ny/bs)*bs*cnf+ny%bs+ny*bs, cnf, Le, nx);


		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}

	return hpmpc_status;
	
	}



int fortran_order_riccati_mhe_if( char prec, int alg,
                                  int nx, int nw, int ny, int ndN, int N,
                                  double *A, double *G, double *C, double *f, 
								  double *D, double *d,
                                  double *R, double *Q, double *Qf, double *r, double *q, double *qf,
                                  double *y, double *x0, double *L0,
                                  double *xe, double *Le, double *w, double *lam, 
                                  double *work0 )
	{

//	printf("\nenter wrapper\n");

	if(alg!=0 && alg!=1 && alg!=2)
		{
		printf("\nUnsopported algorithm type: %d\n\n", alg);
		return -2;
		}

	int hpmpc_status = 0;

	//char prec = 'd';

	if(prec=='d')
		{

		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = bs*ncl;

		const int nwx = nw+nx; 
		const int anx = nal*((nx+nal-1)/nal);
		const int anw = nal*((nw+nal-1)/nal);
		const int any = nal*((ny+nal-1)/nal);
		const int pnx = bs*((nx+bs-1)/bs);
		const int pnw = bs*((nw+bs-1)/bs);
		const int pny = bs*((ny+bs-1)/bs);
		const int pnx2 = bs*((2*nx+bs-1)/bs);
		const int pnwx = bs*((nwx+bs-1)/bs);
		const int pndN = bs*((ndN+bs-1)/bs);
		const int pnxdN = bs*((nx+ndN+bs-1)/bs);
		const int pnwx1 = pnx>pnw ? 2*pnx : pnx+pnw;
		const int pnm = pnx>pnw ? pnx : pnw;
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int cnw = ncl*((nw+ncl-1)/ncl);
		const int cny = ncl*((ny+ncl-1)/ncl);
		const int cnx2 = 2*(ncl*((nx+ncl-1)/ncl));
		const int cndN = ncl*((ndN+ncl-1)/ncl);
		const int cnwx1 = ncl*((nw+nx+1+ncl-1)/ncl);

		const int pad = (ncl-(nx+nw)%ncl)%ncl; // padding
		const int cnj = nx+nw+pad+cnx;



		int i, ii, jj, ll;



		/* align work space */
		size_t align = 64;
		size_t addr = (size_t) work0;
		size_t offset = addr % 64;
		double *ptr = work0 + offset / 8;



		/* array or pointers */
		double *(hpCt[N+1]);
		double *(hpQRAG[N+1]);
		double *(hpLAG[N+1]);
		double *(hpLe[N+1]);
		double *Ld; 

		double *(hf[N+1]);
		double *(hr[N]);
		double *(hq[N+1]);
		double *(hy[N+1]);
		double *(hxe[N+1]);
		double *(hxp[N+1]);
		double *(hw[N]);
		double *(hlam[N]);

		double *Q_temp;
		double *q_temp;
		double *Ct_temp;

		double *diag;
		double *work;



		if(alg==0)
			{
			Q_temp = ptr;
			ptr += pny*cny;
			Ct_temp = ptr;
			ptr += pnx*cny;
			diag = ptr;
			ptr += anx;
			}

		for(ii=0; ii<N; ii++)
			{
			hpQRAG[ii] = ptr;
			ptr += pnwx1*cnwx1;
			if(nx>=nw)
				{
				if(alg==2)
					{
					d_cvt_mat2pmat(nx, nx, Q+ii*nx*nx, nx, 0, hpQRAG[ii], cnwx1);
					}
				else
					{
					for(jj=0; jj<pnx; jj+=4) // loop on panels 
						for(ll=0; ll<nx*bs; ll++) // loop within panels
							hpQRAG[ii][ll+jj*bs*cnwx1] = 0.0;
					d_cvt_mat2pmat(ny, ny, Q+ii*ny*ny, ny, 0, hpQRAG[ii], cnwx1);
					if(alg==0)
						{
						hpCt[ii] = ptr;
						ptr += pnx*cny;
						d_cvt_tran_mat2pmat(ny, nx, C+ii*ny*nx, ny, 0, hpCt[ii], cny);
						dpotrf_lib_old(ny, ny, hpQRAG[ii], cnwx1, Q_temp, cny, diag);
						dtrtr_l_lib(ny, 0, Q_temp, cny, 0, Q_temp, cny);	
						dtrmm_nt_u_lib(nx, ny, hpCt[ii], cny, Q_temp, cny, Ct_temp, cny);
						dsyrk_nt_lib(nx, nx, ny, Ct_temp, cny, Ct_temp, cny, 0, hpQRAG[ii], cnwx1, hpQRAG[ii], cnwx1);
						}
					}
				d_cvt_mat2pmat(nx, nx, A+ii*nx*nx, nx, 0, hpQRAG[ii]+pnx*cnwx1, cnwx1);
				d_cvt_mat2pmat(nw, nw, R+ii*nw*nw, nw, 0, hpQRAG[ii]+(pnx-pnw)*cnwx1+nx*bs, cnwx1);
				d_cvt_mat2pmat(nx, nw, G+ii*nw*nx, nx, 0, hpQRAG[ii]+pnx*cnwx1+nx*bs, cnwx1);
				//d_print_pmat(pnwx1, cnwx1, bs, hpQRAG[ii], cnwx1);
				if(nx>pnx-nx)
					d_cvt_mat2pmat(pnx-nx, nx, A+ii*nx*nx+(nx-pnx+nx), nx, nx, hpQRAG[ii]+nx/bs*bs*cnwx1+nx%bs, cnwx1);
				else
					d_cvt_mat2pmat(nx, nx, A+ii*nx*nx, nx, nx, hpQRAG[ii]+nx/bs*bs*cnwx1+nx%bs, cnwx1);
				if(nx>pnw-nw)
					d_cvt_mat2pmat(pnw-nw, nw, G+ii*nw*nx+(nx-pnw+nw), nx, nw, hpQRAG[ii]+(pnx-pnw+nw/bs*bs)*cnwx1+nw%bs+nx*bs, cnwx1);
				else
					d_cvt_mat2pmat(nx, nw, G+ii*nw*nx, nx, nw, hpQRAG[ii]+(pnx-pnw+nw/bs*bs)*cnwx1+nw%bs+nx*bs, cnwx1);
				//d_print_pmat(pnwx1, ncnwx1, bs, hpQRAG[ii], cnwx1);
				}
			else
				{
				if(alg==2)
					{
					d_cvt_mat2pmat(nx, nx, Q+ii*nx*nx, nx, 0, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1);
					}
				else
					{
					for(jj=0; jj<pnx; jj+=4) // loop on panels 
						for(ll=0; ll<nx*bs; ll++) // loop within panels
							hpQRAG[ii][ll+jj*bs*cnwx1] = 0.0;
					d_cvt_mat2pmat(ny, ny, Q+ii*ny*ny, ny, 0, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1);
					if(alg==0)
						{
						hpCt[ii] = ptr;
						ptr += pnx*cny;
						d_cvt_tran_mat2pmat(ny, nx, C+ii*ny*nx, ny, 0, hpCt[ii], cny);
						dpotrf_lib_old(ny, ny, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1, Q_temp, cny, diag);
						dtrtr_l_lib(ny, 0, Q_temp, cny, 0, Q_temp, cny);	
						dtrmm_nt_u_lib(nx, ny, hpCt[ii], cny, Q_temp, cny, Ct_temp, cny);
						dsyrk_nt_lib(nx, nx, ny, Ct_temp, cny, Ct_temp, cny, 0, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1);
						}
					}
				d_cvt_mat2pmat(nx, nx, A+ii*nx*nx, nx, 0, hpQRAG[ii]+pnw*cnwx1, cnwx1);
				d_cvt_mat2pmat(nw, nw, R+ii*nw*nw, nw, 0, hpQRAG[ii]+nx*bs, cnwx1);
				d_cvt_mat2pmat(nx, nw, G+ii*nw*nx, nx, 0, hpQRAG[ii]+pnw*cnwx1+nx*bs, cnwx1);
				//d_print_pmat(pnwx1, cnwx1, bs, hpQRAG[ii], cnwx1);
				if(nx>pnx-nx)
					d_cvt_mat2pmat(pnx-nx, nx, A+ii*nx*nx+(nx-pnx+nx), nx, nx, hpQRAG[ii]+(pnw-pnx+nx/bs*bs)*cnwx1+nx%bs, cnwx1);
				else
					d_cvt_mat2pmat(nx, nx, A+ii*nx*nx, nx, nx, hpQRAG[ii]+(pnw-pnx+nx/bs*bs)*cnwx1+nx%bs, cnwx1);
				if(nx>pnw-nw)
					d_cvt_mat2pmat(pnw-nw, nw, G+ii*nw*nx+(nx-pnw+nw), nx, nw, hpQRAG[ii]+nw/bs*bs*cnwx1+nw%bs+nx*bs, cnwx1);
				else
					d_cvt_mat2pmat(nx, nw, G+ii*nw*nx, nx, nw, hpQRAG[ii]+nw/bs*bs*cnwx1+nw%bs+nx*bs, cnwx1);
				//d_print_pmat(pnwx1, cnwx1, bs, hpQRAG[ii], cnwx1);
				}
			}
		hpQRAG[N] = ptr;
		ptr += (pnx+pndN)*cnx;
		if(alg==2)
			{
			d_cvt_mat2pmat(nx, nx, Qf, nx, 0, hpQRAG[N], cnx);
			}
		else // if(alg==0 || alg==1)
			{
			for(jj=0; jj<pnx*cnx; jj++) hpQRAG[N][jj] = 0.0;
			d_cvt_mat2pmat(ny, ny, Qf, ny, 0, hpQRAG[N], cnx);
			if(alg==0)
				{
				hpCt[N] = ptr;
				ptr += pnx*cny;
				d_cvt_tran_mat2pmat(ny, nx, C+N*ny*nx, ny, 0, hpCt[N], cny);
				dpotrf_lib_old(ny, ny, hpQRAG[N], cnx, Q_temp, cny, diag);
				dtrtr_l_lib(ny, 0, Q_temp, cny, 0, Q_temp, cny);	
				dtrmm_nt_u_lib(nx, ny, hpCt[N], cny, Q_temp, cny, Ct_temp, cny);
				dsyrk_nt_lib(nx, nx, ny, Ct_temp, cny, Ct_temp, cny, 0, hpQRAG[N], cnx, hpQRAG[N], cnx);
				}
			}


		if(ndN>0)
			{
			d_cvt_mat2pmat(ndN, nx, D, ndN, 0, hpQRAG[N]+pnx*cnx, cnx);
			//d_print_pmat(pnx+pndN, cnx, bs, pQD, cnx);
			if(ndN>pnx-nx)
				d_cvt_mat2pmat(pnx-nx, nx, D+(ndN-pnx+nx), ndN, nx, hpQRAG[N]+nx/bs*bs*cnx+nx%bs, cnx);
			else
				d_cvt_mat2pmat(ndN, nx, D, ndN, nx, hpQRAG[N]+nx/bs*bs*cnx+nx%bs, cnx);
			//d_print_pmat(pnx+pndN, cnx, bs, pQD, cnx);
			Ld = ptr;
			ptr += pndN*cndN;
			}
		//d_print_pmat(nx+nx, nx, bs, hpQA[0], cnx);
		//d_print_pmat(nx+ndN, nx, bs, hpQA[N], cnx);

		for(ii=0; ii<N; ii++)
			{
			hpLAG[ii] = ptr;
			ptr += pnwx1*cnwx1;
			}
		hpLAG[N] = ptr;
		ptr += (pnx+pndN)*cnx;

		for(ii=0; ii<=N; ii++)
			{
			hpLe[ii] = ptr;
			ptr += pnx*cnx;
			}
		d_cvt_mat2pmat(nx, nx, L0, nx, 0, hpLe[0], cnx);
		//d_print_pmat(nx, nx, bs, hpALe[0], cnx2);




		for(ii=0; ii<N; ii++)
			{
			hf[ii] = ptr;
			ptr += anx;
			for(jj=0; jj<nx; jj++) hf[ii][jj] = f[ii*nx+jj];
			}
		if(ndN>0)
			{
			hf[N] = ptr;
			ptr += anx;
			for(jj=0; jj<ndN; jj++) hf[N][jj] = d[jj];
			}

		for(ii=0; ii<N; ii++)
			{
			hr[ii] = ptr;
			ptr += anw;
			for(jj=0; jj<nw; jj++) hr[ii][jj] = r[ii*nw+jj];
			}

		if(alg==0 || alg==1)
			{
			for(ii=0; ii<=N; ii++)
				{
				hy[ii] = ptr;
				ptr += any;
				for(jj=0; jj<ny; jj++) hy[ii][jj] = y[ii*ny+jj];
				}
			}

		//d_print_mat(nx, N+1, hr[0], anx);
		if(alg==2)
			{
			//for(ii=0; ii<=N; ii++)
			for(ii=0; ii<N; ii++)
				{
				hq[ii] = ptr;
				ptr += anx;
				for(jj=0; jj<nx; jj++) hq[ii][jj] = q[ii*nx+jj];
				}
			hq[N] = ptr;
			ptr += anx;
			for(jj=0; jj<nx; jj++) hq[N][jj] = qf[jj];
			}
		else // if(alg==0 || alg==1)
			{
			q_temp = ptr;
			ptr += any;
			//for(ii=0; ii<=N; ii++)
			for(ii=0; ii<N; ii++)
				{
				hq[ii] = ptr;
				ptr += anx;
				for(jj=0; jj<ny; jj++) q_temp[jj] = - q[ii*ny+jj];
				//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
				//d_print_mat(1, nx, q_temp, 1);
				if(nx>=nw)
					dsymv_lib(ny, ny, hpQRAG[ii], cnwx1, hy[ii], -1, q_temp, q_temp);
				else
					dsymv_lib(ny, ny, hpQRAG[ii]+(pnw-pnx)*cnwx1, cnwx1, hy[ii], -1, q_temp, q_temp);
				//d_print_mat(1, nx, q_temp, 1);
				if(alg==0)
					{
					dgemv_n_lib(nx, ny, hpCt[ii], cny, q_temp, 0, hq[ii], hq[ii]);
					}
				else
					{
					for(jj=0; jj<ny; jj++) hq[ii][jj] = q_temp[jj];
					for( ; jj<nx; jj++) hq[ii][jj] = 0;
					}
				}
			hq[N] = ptr;
			ptr += anx;
			for(jj=0; jj<ny; jj++) q_temp[jj] = - qf[jj];
			//d_print_pmat(nx, nx, bs, hpQA[ii], cnx);
			//d_print_mat(1, nx, q_temp, 1);
			dsymv_lib(ny, ny, hpQRAG[N], cnx, hy[N], -1, q_temp, q_temp);
			//d_print_mat(1, nx, q_temp, 1);
			if(alg==0)
				{
				dgemv_n_lib(nx, ny, hpCt[N], cny, q_temp, 0, hq[N], hq[N]);
				}
			else
				{
				for(jj=0; jj<ny; jj++) hq[N][jj] = q_temp[jj];
				for( ; jj<nx; jj++) hq[N][jj] = 0;
				}
			}
		//d_print_pmat(nx, ny, bs, hpCt[0], cny);
		//d_print_mat(nx, N+1, hq[0], anx);

		for(ii=0; ii<=N; ii++)
			{
			hxp[ii] = ptr;
			ptr += anx;
			}
		for(jj=0; jj<nx; jj++) hxp[0][jj] = x0[jj];

		for(ii=0; ii<=N; ii++)
			{
			hxe[ii] = ptr;
			ptr += anx;
			}

		for(ii=0; ii<N; ii++)
			{
			hw[ii] = ptr;
			ptr += anw;
			}

		for(ii=0; ii<=N; ii++)
			{
			hlam[ii] = ptr;
			ptr += anx;
			}

		work = ptr;
		ptr += pnx*cnx+pnm+anx;



		int diag_R = DIAG_R;



		// factorize KKT matrix
		hpmpc_status = d_ric_trf_mhe_if(nx, nw, ndN, N, hpQRAG, diag_R, hpLe, hpLAG, Ld, work);

		if(hpmpc_status!=0)
			return hpmpc_status;

		// solve KKT system
		d_ric_trs_mhe_if(nx, nw, ndN, N, hpLe, hpLAG, Ld, hq, hr, hf, hxp, hxe, hw, hlam, work);



		// residuals computation
		//#if 1
		//#define DEBUG_MODE
		#ifdef DEBUG_MODE
		printf("\nstart of print residuals\n\n");
		double *(hq_res[N+1]);
		double *(hr_res[N]);
		double *(hf_res[N]);
		double *p_hq_res; d_zeros_align(&p_hq_res, anx, N+1);
		double *p_hr_res; d_zeros_align(&p_hr_res, anw, N);
		double *p_hf_res; d_zeros_align(&p_hf_res, anx, N+1);

		for(jj=0; jj<N; jj++)
			{
			hq_res[jj] = p_hq_res+jj*anx;
			hr_res[jj] = p_hr_res+jj*anw;
			hf_res[jj] = p_hf_res+jj*anx;
			}
		hq_res[N] = p_hq_res+N*anx;
		hf_res[N] = p_hf_res+N*anx;

		double *pL0_inv2; d_zeros_align(&pL0_inv2, pnx, cnx);
		//dtrinv_lib(nx, pL0, cnx, pL0_inv2, cnx);
		d_cvt_mat2pmat(nx, nx, L0, nx, 0, pL0_inv2, cnx); // XXX

		double *p0; d_zeros_align(&p0, anx, 1);
		double *x_temp; d_zeros_align(&x_temp, anx, 1);
		dtrmv_u_t_lib(nx, pL0_inv2, cnx, x0, x_temp, 0);
		dtrmv_u_n_lib(nx, pL0_inv2, cnx, x_temp, p0, 0);

		d_res_mhe_if(nx, nw, ndN, N, hpQA, hpRG, pL0_inv2, hq, hr, hf, p0, hxe, hw, hlam, hq_res, hr_res, hf_res, work);

		d_print_mat(nx, N+1, hq_res[0], anx);
		d_print_mat(nw, N, hr_res[0], anw);
		d_print_mat(nx, N, hf_res[0], anx);
		d_print_mat(ndN, 1, hf_res[0]+N*anx, anx);

		free(p_hq_res);
		free(p_hr_res);
		free(p_hf_res);
		free(pL0_inv2);
		free(p0);
		free(x_temp);
		printf("\nend of print residuals\n\n");
		#endif



		// copy back estimate and covariance at first stage (Extended Kalman Filter update of initial condition)
		for(jj=0; jj<nx; jj++) x0[jj] = hxp[1][jj];

		// save L0 for next step
		d_cvt_pmat2mat(nx, nx, 0, hpLe[1], cnx, L0, nx);


		// copy back estimates at all stages 0,1,...,N
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nx; jj++)
				xe[ii*nx+jj] = hxe[ii][jj];

		// copy back process disturbance at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nw; jj++)
				w[ii*nw+jj] = hw[ii][jj];
			
		// copy back multipliers at all stages 0,1,...,N-1
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++)
				lam[ii*nx+jj] = hlam[ii][jj];
		for(jj=0; jj<ndN; jj++)
			lam[N*nx+jj] = hlam[N][jj];

		// copy back cholesky factor of information matrix at last stage
		d_cvt_pmat2mat(nx, nx, 0, hpLAG[N], cnx, Le, nx);
		for(jj=0; jj<nx; jj++) Le[jj*(nx+1)] = 1.0/Le[jj*(nx+1)];


		//d_print_pmat(nx, nx, bs, hpALe[1], cnx2);
		//d_print_pmat(pnx2, cnx2, bs, hpALe[N-1], cnx2);
		//d_print_pmat(pnx2, cnx2, bs, hpALe[N], cnx2);
		//d_print_pmat(pnwx, cnw, bs, hpGLq[N-1], cnw);
		//d_print_pmat(nx, nx, bs, pL0_inv, cnx);
		//d_print_pmat(nx, nx, bs, pL0, cnx);
		//d_print_mat(nx, nx, L0, nx);
		//d_print_mat(nx, N+1, hxe[0], anx);


//		free(work0); // TODO remove
		
		}
	else
		{
		printf("\nUnsopported precision type: %s\n\n", &prec);
		return -1;
		}

	return hpmpc_status;
	
	}



int fortran_order_admm_box_mpc( int k_max, double tol,
                                     double rho, double alpha,
                                     int nx, int nu, int N,
                                     double* A, double* B, double* b, 
                                     double* Q, double* Qf, double* S, double* R, 
                                     double* q, double* qf, double* r, 
                                     double* lb, double* ub, 
                                     double* x, double* u,
                                     int* nIt, double *stat )

	{

/*printf("\nstart of wrapper\n");*/

    char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 10*anz + 7*anx) + 3*anz)*sizeof(double));

		// parameters
/*		double rho = 10.0; // penalty parameter*/
/*		double alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;

        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N+1]);
        double *(hux[N+1]);
		double *(hux_v[N+1]);
		double *(hux_w[N+1]);
		double *(hlb[N+1]);
		double *(hub[N+1]);
        double *(hpi[N+1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[0], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[1], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[N-1], cnx);*/

/*return 1;*/
        // cost function
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_tran_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_tran_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_tran_mat2pmat(nx, nx, Qf, nx, nu, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = lb[ii+nu*jj];
				hub[jj][ii] = ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



/*printf("\nstart of ip solver\n");*/

        // call the soft ADMM solver
		d_admm_box_mpc(nIt, k_max, tol, tol, warm_start, 1, rho, alpha, stat, nx, nu, N, hpBAbt, hpQ, hlb, hub, hux, hux_v, hux_w, compute_mult, hpi, ptr);

/*printf("\nend of ip solver\n");*/


        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

#if PC_DEBUG == 1
        for(jj=0; jj<*nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

	    }
    else if(prec=='s')
	    {

		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        float *work0 = (float *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 9*anz + 7*anx) + 3*anz)*sizeof(float));

		// parameters
/*		float rho = 10.0; // penalty parameter*/
/*		float alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = work0 + offset / 4;

        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
        float *(hux[N + 1]);
		float *(hux_v[N+1]);
		float *(hux_w[N+1]);
		float *(hlb[N+1]);
		float *(hub[N+1]);
        float *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_tran_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = (float) lb[ii+nu*jj];
				hub[jj][ii] = (float) ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = (float) lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = (float) ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the soft ADMM solver
		s_admm_box_mpc(nIt, k_max, (float) tol, (float) tol, warm_start, 1, (float) rho, (float) alpha, (float *)stat, nx, nu, N, hpBAbt, hpQ, hlb, hub, hux, hux_v, hux_w, compute_mult, hpi, ptr);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

 	   }

/*printf("\nend of wrapper\n");*/

    return 0;

	}



int fortran_order_admm_soft_wrapper( int k_max, double tol,
                                     double rho, double alpha,
                                     const int nx, const int nu, const int N,
                                     double* A, double* B, double* b, 
                                     double* Q, double* Qf, double* S, double* R, 
                                     double* q, double* qf, double* r, 
                                     double* Z, double *z,
                                     double* lb, double* ub, 
                                     double* x, double* u,
                                     int* nIt, double *stat )

	{

/*printf("\nstart of wrapper\n");*/

    char prec = PREC;

    if(prec=='d')
	    {
	    
		const int bs = D_MR; //d_get_mr();
		const int ncl = D_NCL;
		const int nal = D_MR*D_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        double *work0 = (double *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 10*anz + 17*anx) + 3*anz)*sizeof(double));

		// parameters
/*		double rho = 10.0; // penalty parameter*/
/*		double alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        double *ptr = work0 + offset / 8;

        /* array or pointers */
        double *(hpBAbt[N]);
        double *(hpQ[N + 1]);
		double *(hZ[N+1]);
		double *(hz[N+1]);
        double *(hux[N + 1]);
		double *(hux_v[N+1]);
		double *(hux_w[N+1]);
		double *(hlb[N+1]);
		double *(hub[N+1]);
		double *(hs_u[N+1]);
		double *(hs_v[N+1]);
		double *(hs_w[N+1]);
        double *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hZ[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hz[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_u[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_v[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_w[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            d_cvt_tran_mat2pmat(nx, nu, B+ii*nu*nx, nx, 0, hpBAbt[ii], cnx);
            d_cvt_tran_mat2pmat(nx, nx, A+ii*nx*nx, nx, nu, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = b[ii*nx+jj];
	        }
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[0], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[1], cnx);*/
/*	    d_print_pmat(nx+nu+1, nx, bs, hpBAbt[N-1], cnx);*/

/*return 1;*/
        // cost function
        for(jj=0; jj<N; jj++)
	        {
            d_cvt_tran_mat2pmat(nu, nu, R+jj*nu*nu, nu, 0, hpQ[jj], cnz);
            d_cvt_tran_mat2pmat(nu, nx, S+jj*nx*nu, nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            d_cvt_tran_mat2pmat(nx, nx, Q+jj*nx*nx, nx, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        d_cvt_tran_mat2pmat(nx, nx, Qf, nx, nu, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = qf[jj];

		// soft constraints cost function
        for(jj=0; jj<=N; jj++)
	        {
			for(ii=0; ii<nx; ii++) hZ[jj][ii]     = Z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hZ[jj][anx+ii] = Z[jj*2*nx+nx+ii]; // lower
			for(ii=0; ii<nx; ii++) hz[jj][ii]     = z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hz[jj][anx+ii] = z[jj*2*nx+nx+ii]; // lower
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = lb[ii+nu*jj];
				hub[jj][ii] = ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = x[ii+nx*jj];



/*printf("\nstart of ip solver\n");*/

        // call the soft ADMM solver
	        d_admm_soft_mpc(nIt, k_max, tol, tol, warm_start, 1, rho, alpha, stat, nx, nu, N, hpBAbt, hpQ, hZ, hz, hlb, hub, hux, hux_v, hux_w, hs_u, hs_v, hs_w, compute_mult, hpi, ptr);

/*printf("\nend of ip solver\n");*/


        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = hux[jj][nu+ii];

#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

	    }
    else if(prec=='s')
	    {

		const int bs = S_MR; //d_get_mr();
		const int ncl = S_NCL;
		const int nal = S_MR*S_NCL;

		const int nz = nx+nu+1;
		const int pnz = bs*((nz+bs-1)/bs);
		const int pnx = bs*((nx+bs-1)/bs);
		const int cnz = ncl*((nx+nu+1+ncl-1)/ncl);
		const int cnx = ncl*((nx+ncl-1)/ncl);
		const int anz = nal*((nz+nal-1)/nal);
		const int anx = nal*((nx+nal-1)/nal);

		const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
		const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;

		const int nb = nx+nu; // number of box constraints
/*		const int pnb = bs*((2*nb+bs-1)/bs);*/
		const int anb = nal*((2*nb+nal-1)/nal);

		// work space
        float *work0 = (float *) malloc((8 + (N+1)*(pnz*cnx + pnz*cnz + pnz*cnl + 9*anz + 17*anx) + 3*anz)*sizeof(float));

		// parameters
/*		float rho = 10.0; // penalty parameter*/
/*		float alpha = 1.9; // relaxation parameter*/
        int warm_start = 0;//WARM_START;
        int compute_mult = 0;//1; // compute multipliers
        
        int info = 0;

        int i, ii, jj, ll;


        /* align work space */
        size_t align = 64;
        size_t addr = (size_t) work0;
        size_t offset = addr % 64;
        float *ptr = work0 + offset / 4;

        /* array or pointers */
        float *(hpBAbt[N]);
        float *(hpQ[N + 1]);
		float *(hZ[N+1]);
		float *(hz[N+1]);
        float *(hux[N + 1]);
		float *(hux_v[N+1]);
		float *(hux_w[N+1]);
		float *(hlb[N+1]);
		float *(hub[N+1]);
		float *(hs_u[N+1]);
		float *(hs_v[N+1]);
		float *(hs_w[N+1]);
        float *(hpi[N + 1]);

        for(ii=0; ii<N; ii++)
	        {
            hpBAbt[ii] = ptr;
            ptr += pnz*cnx;
	        }

        for(ii=0; ii<=N; ii++) // time variant and copied again internally in the IP !!!
	        {
            hpQ[ii] = ptr;
            ptr += pnz*cnz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hZ[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hz[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_v[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hux_w[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hlb[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hub[ii] = ptr;
            ptr += anz;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_u[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_v[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++)
	        {
            hs_w[ii] = ptr;
            ptr += 2*anx;
	        }

        for(ii=0; ii<=N; ii++) // time Variant box constraints
	        {
            hpi[ii] = ptr;
            ptr += anx; // for alignment of ptr
	        }

        /* pack matrices 	*/

        // dynamic system
        for(ii=0; ii<N; ii++)
	        {
            cvt_tran_d2s_mat2pmat(nx, nu, 0, bs, B+ii*nu*nx, nx, hpBAbt[ii], cnx);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, A+ii*nx*nx, nx, hpBAbt[ii]+nu/bs*cnx*bs+nu%bs, cnx);
            for (jj = 0; jj<nx; jj++)
                hpBAbt[ii][(nx+nu)/bs*cnx*bs+(nx+nu)%bs+jj*bs] = (float) b[ii*nx+jj];
	        }

        // cost function
        for(jj=0; jj<N; jj++)
	        {
            cvt_tran_d2s_mat2pmat(nu, nu, 0, bs, R+jj*nu*nu, nu, hpQ[jj], cnz);
            cvt_tran_d2s_mat2pmat(nu, nx, nu, bs, S+jj*nx*nu, nu, hpQ[jj]+nu/bs*cnz*bs+nu%bs, cnz);
            cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Q+jj*nx*nx, nx, hpQ[jj]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
            for(ii=0; ii<nu; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+ii*bs] = (float) r[ii+jj*nu];
            for(ii=0; ii<nx; ii++)
                hpQ[jj][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+ii)*bs] = (float) q[ii+nx*jj];
	        }

        for(jj=0; jj<nu; jj++)
            for(ii=0; ii<nz; ii+=bs)
                for(i=0; i<bs; i++)
                    hpQ[N][ii*cnz+i+jj*bs] = 0.0;
        for(jj=0; jj<nu; jj++)
            hpQ[N][jj/bs*cnz*bs+jj%bs+jj*bs] = 1.0;
        cvt_tran_d2s_mat2pmat(nx, nx, nu, bs, Qf, nx, hpQ[N]+nu/bs*cnz*bs+nu%bs+nu*bs, cnz);
        for(jj=0; jj<nx; jj++)
            hpQ[N][(nx+nu)/bs*cnz*bs+(nx+nu)%bs+(nu+jj)*bs] = (float) qf[jj];

		// soft constraints cost function
        for(jj=0; jj<=N; jj++)
	        {
			for(ii=0; ii<nx; ii++) hZ[jj][ii]     = (float) Z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hZ[jj][anx+ii] = (float) Z[jj*2*nx+nx+ii]; // lower
			for(ii=0; ii<nx; ii++) hz[jj][ii]     = (float) z[jj*2*nx+ii]; // upper
			for(ii=0; ii<nx; ii++) hz[jj][anx+ii] = (float) z[jj*2*nx+nx+ii]; // lower
			}

		// input constraints
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nu; ii++)
				{
				hlb[jj][ii] = (float) lb[ii+nu*jj];
				hub[jj][ii] = (float) ub[ii+nu*jj];
				}
			}
		// state constraints 
		for(jj=0; jj<N; jj++)
			{
			for(ii=0; ii<nx; ii++)
				{
				hlb[jj+1][nu+ii] = (float) lb[N*nu+ii+nx*jj];
				hub[jj+1][nu+ii] = (float) ub[N*nu+ii+nx*jj];
				}
			}



        // initial guess
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                hux[jj][ii] = (float) u[ii+nu*jj];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                hux[jj][nu+ii] = (float) x[ii+nx*jj];



        // call the soft ADMM solver
		s_admm_soft_mpc(nIt, k_max, (float) tol, (float) tol, warm_start, 1, (float) rho, (float) alpha, (float *)stat, nx, nu, N, hpBAbt, hpQ, hZ, hz, hlb, hub, hux, hux_v, hux_w, hs_u, hs_v, hs_w, compute_mult, hpi, ptr);



		// convert stat into double (start fom end !!!)
		float *ptr_stat = (float *) stat;
		for(ii=5*k_max-1; ii>=0; ii--)
			{
			stat[ii] = (double) ptr_stat[ii];
			}



        // copy back inputs and states
        for(jj=0; jj<N; jj++)
            for(ii=0; ii<nu; ii++)
                u[ii+nu*jj] = (double) hux[jj][ii];

        for(jj=0; jj<=N; jj++)
            for(ii=0; ii<nx; ii++)
                x[ii+nx*jj] = (double) hux[jj][nu+ii];


#if PC_DEBUG == 1
        for (jj = 0; jj < *nIt; jj++)
            printf("k = %d\tsigma = %f\talpha = %f\tmu = %f\t\tmu = %e\n", jj,
                   stat[5 * jj], stat[5 * jj + 1], stat[5 * jj + 2],
                   stat[5 * jj + 2]);
        printf("\n");
#endif /* PC_DEBUG == 1 */

		free(work0);

 	   }

/*printf("\nend of wrapper\n");*/

    return 0;

	}

