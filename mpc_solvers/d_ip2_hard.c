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

#include <stdlib.h>
#include <math.h>

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"
#include "../include/mpc_aux.h"



/* computes work space size */
int d_ip2_mpc_hard_tv_work_space_size_doubles(int N, int *nx, int *nu, int *nb, int *ng)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int pnx, pnz, pnb, png, cnx, cnux; 

	int size = 0;
	int pnzM = 0;
	for(ii=0; ii<=N; ii++)
		{
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		if(pnz>pnzM) pnzM = pnz;
		pnb = (nb[ii]+bs-1)/bs*bs;
		png = (ng[ii]+bs-1)/bs*bs;
		cnx = (nx[ii]+ncl-1)/ncl*ncl;
		cnux = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		pnx = (nx[ii]+bs-1)/bs*bs;
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		size += pnz*(cnx+ncl>cnux ? cnx+ncl : cnux) + 3*pnx + 4*pnz + 15*pnb + 11*png;
		}

	int nxgM = ng[N];
	for(ii=0; ii<N; ii++)
		{
		if(nx[ii+1]+ng[ii]>nxgM) nxgM = nx[ii+1]+ng[ii];
		}

	size += pnzM*((nxgM+ncl-1)/ncl*ncl) + pnzM;

	return size;
	}



/* primal-dual interior-point method, hard constraints, time variant matrices, time variant size (mpc version) */
int d_ip2_mpc_hard_tv(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **pBAbt, double **pQ, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory)
	{
	
	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// matrices size
	int idx;
//	int nxM = 0;
	int nzM = 0;
//	int ngM = 0;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
//	int cnz[N+1];
	int cnux[N+1];

	for(jj=0; jj<=N; jj++)
		{
		pnx[jj] = (nx[jj]+bs-1)/bs*bs;
		pnz[jj] = (nu[jj]+nx[jj]+1+bs-1)/bs*bs;
		pnb[jj] = (nb[jj]+bs-1)/bs*bs;
		png[jj] = (ng[jj]+bs-1)/bs*bs;
		cnx[jj] = (nx[jj]+ncl-1)/ncl*ncl;
//		cnz[jj] = (nu[jj]+nx[jj]+1+ncl-1)/ncl*ncl;
		cnux[jj] = (nu[jj]+nx[jj]+ncl-1)/ncl*ncl;
//		if(nx[jj]>nxM) nxM = nx[jj];
		if(nu[jj]+nx[jj]+1>nzM) nzM = nu[jj]+nx[jj]+1;
//		if(ng[jj]>ngM) ngM = ng[jj];
//		if(nx[jj]+ng[jj]>nxgM) nxgM = nx[jj]+ng[jj];
		}

	int nxgM = ng[N];
	for(jj=0; jj<N; jj++)
		{
		if(nx[jj+1]+ng[jj]>nxgM) nxgM = nx[jj+1]+ng[jj];
		}


	// initialize work space
	// work_space_double_size_per_stage = pnz*cnl + 2*pnz + 2*pnx + 14*pnb + 10*png
	// work_space_double_size_const_max = pnz*cnxg + pnz
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *pL[N+1];
	double *dL[N+1];
	double *l[N+1];
	double *work;
	double *b[N];
	double *q[N+1];
	double *dux[N+1];
	double *dpi[N+1];
	double *pd[N+1]; // pointer to diagonal of Hessian
	double *pl[N+1]; // pointer to linear part of Hessian
	double *bd[N+1]; // backup diagonal of Hessian
	double *bl[N+1]; // backup linear part of Hessian
	double *diag;
	double *dlam[N+1];
	double *dt[N+1];
	double *lamt[N+1];
	double *t_inv[N+1];
	double *Qx[N+1];
	double *qx[N+1];
	double *qx2[N+1];
	double *Pb[N];

//	int size = 0;

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
//		ptr += pnz[jj] * ( cnx[jj]+ncl>cnz[jj] ? cnx[jj]+ncl : cnz[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
		ptr += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
//		size += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
		}

	for(jj=0; jj<=N; jj++)
		{
		dL[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		l[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	// work space
	work = ptr;
//	ptr += ((nzM+bs-1)/bs*bs) * ((nxM+ngM+ncl-1)/ncl*ncl); // pnzM*cnxgM
//	size += ((nzM+bs-1)/bs*bs) * ((nxM+ngM+ncl-1)/ncl*ncl); // pnzM*cnxgM
	ptr += ((nzM+bs-1)/bs*bs) * ((nxgM+ncl-1)/ncl*ncl); // pnzM*cnxgM
//	size += ((nzM+bs-1)/bs*bs) * ((nxgM+ncl-1)/ncl*ncl); // pnzM*cnxgM

	// b as vector
	for(jj=0; jj<N; jj++)
		{
		b[jj] = ptr;
		ptr += pnx[jj+1];
//		size += pnx[jj+1];
		d_copy_mat(1, nx[jj+1], pBAbt[jj]+(nu[jj]+nx[jj])/bs*bs*cnx[jj+1]+(nu[jj]+nx[jj])%bs, bs, b[jj], 1);
		}

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj];
//		size += pnx[jj];
		}
	
	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj+1];
//		size += pnx[jj+1];
		}

	// linear part of cost function
	for(jj=0; jj<=N; jj++)
		{
		q[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		for(ll=0; ll<nu[jj]+nx[jj]; ll++) q[jj][ll] = pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+ll*bs];
		}

	// Hessian backup
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = ptr;
		pl[jj] = ptr + pnb[jj];
		bd[jj] = ptr + 2*pnb[jj];
		bl[jj] = ptr + 3*pnb[jj];
		ptr += 4*pnb[jj];
//		size += 4*pnb[jj];
		// backup
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			bd[jj][ll] = pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs];
			bl[jj][ll] = q[jj][idx];
			}
		}

	diag = ptr;
	ptr += (nzM+bs-1)/bs*bs; // pnzM
//	size += (nzM+bs-1)/bs*bs; // pnzM

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb[jj]+2*png[jj];
		ptr += 4*pnb[jj]+4*png[jj];
//		size += 4*pnb[jj]+4*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
//		size += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
//		size += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+pnb[jj]+png[jj];
		qx2[jj] = ptr+2*pnb[jj]+2*png[jj];
		ptr += 3*pnb[jj]+3*png[jj];
//		size += 3*pnb[jj]+3*png[jj];
		}
	
//	printf("\nsize real = %d\n", size);

#if 0
	int cng[N+1];
	for(jj=0; jj<=N; jj++)
		{
		cng[jj] = (ng[jj]+ncl-1)/ncl*ncl;
		}
	d_print_pmat(nu[0]+nx[0]+1, nx[1], bs, pBAbt[0], cnx[1]);
	d_print_pmat(nu[1]+nx[1]+1, nx[2], bs, pBAbt[1], cnx[2]);
	d_print_pmat(nu[0]+nx[0], ng[0], bs, pDCt[0], cng[0]);
	d_print_pmat(nu[1]+nx[1], ng[1], bs, pDCt[1], cng[1]);
	d_print_pmat(nu[N]+nx[N], ng[N], bs, pDCt[N], cng[N]);
	exit(1);
#endif


	double temp0, temp1;
	double alpha, mu, mu_aff;
	double mu_scal = 0.0; 
	for(jj=0; jj<=N; jj++) mu_scal += 2*nb[jj] + 2*ng[jj];
	//printf("\nmu_scal = %f\n", mu_scal);
	mu_scal = 1.0 / mu_scal;
	//printf("\nmu_scal = %f\n", mu_scal);
	double sigma, sigma_decay, sigma_min;
	//for(ii=0; ii<=N; ii++)
	//	printf("\n%d %d\n", nb[ii], ng[ii]);
	//exit(1);

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	d_init_var_mpc_hard_tv(N, nx, nu, nb, idxb, ng, ux, pi, pDCt, d, t, lam, mu0, warm_start);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], ux[ii], 1);
exit(1);
#endif


	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx[jj]; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	for(ll=0; ll<nx[0]; ll++)
		dux[0][nu[0]+ll] = ux[0][nu[0]+ll];




	// compute the duality gap
	//alpha = 0.0; // needed to compute mu !!!!!
	//d_compute_mu_hard_mpc(N, nx, nu, nb, &mu, mu_scal, alpha, lam, dlam, t, dt);
	mu = mu0;

	// set to zero iteration count
	*kk = 0;	

	// larger than minimum accepted step size
	alpha = 1.0;

	// update hessian in Riccati routine
	const int update_hessian = 1;

	int fast_rsqrt = 0;



	// IP loop		
	while( *kk<k_max && mu>mu_tol && alpha>=alpha_min )
		{
						


		//update cost function matrices and vectors (box constraints)
		d_update_hessian_mpc_hard_tv(N, nx, nu, nb, ng, 0.0, t, t_inv, lam, lamt, dlam, Qx, qx, qx2, bd, bl, pd, pl, d);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii], pd[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii], pl[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, ng[ii], Qx[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, ng[ii], qx[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, ng[ii], qx2[ii], 1);
if(*kk==1)
exit(1);
#endif


		// compute the search direction: factorize and solve the KKT system
		d_back_ric_rec_sv_tv(N, nx, nu, pBAbt, pQ, dux, pL, dL, work, diag, 1, Pb, compute_mult, dpi, nb, idxb, pd, pl, ng, pDCt, Qx, qx2);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]+1, bs, pQ[ii], cnz[ii]);
//exit(1);
#endif
#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(pnz[ii], cnz[ii], bs, pL[ii], cnz[ii]);
//exit(1);
#endif
#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
if(*kk==1)
exit(1);
#endif


#if 1

		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		d_compute_alpha_mpc_hard_tv(N, nx, nu, nb, idxb, ng, &alpha, t, dt, lam, dlam, lamt, dux, pDCt, d);

		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;

#if 0
printf("\nalpha = %f\n", alpha);
exit(1);
#endif


		// compute the affine duality gap
		d_compute_mu_mpc_hard_tv(N, nx, nu, nb, ng, &mu_aff, mu_scal, alpha, lam, dlam, t, dt);

		stat[5*(*kk)+2] = mu_aff;
		//printf("\nmu = %f\n", mu_aff);



		// compute sigma
		sigma = mu_aff/mu;
		sigma = sigma*sigma*sigma;
//		if(sigma<sigma_min)
//			sigma = sigma_min;
//printf("\n%f %f %f %f\n", mu_aff, mu, sigma, mu_scal);
//exit(1);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii], dt[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii], dlam[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii], t_inv[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii], pl[ii], 1);
//exit(1);
#endif


		d_update_gradient_mpc_hard_tv(N, nx, nu, nb, ng, sigma*mu, dt, dlam, t_inv, pl, qx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii], pl[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, ng[ii], qx[ii], 1);
if(*kk==1)
exit(1);
#endif


//		// copy b into x
//		for(ii=0; ii<N; ii++)
//			for(jj=0; jj<nx[ii+1]; jj++) 
//				dux[ii+1][nu[ii+1]+jj] = pBAbt[ii][(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs+bs*jj]; // copy b



		// solve the system
		d_back_ric_rec_trs_tv(N, nx, nu, pBAbt, b, pL, dL, q, l, dux, work, 0, Pb, compute_mult, dpi, nb, idxb, pl, ng, pDCt, qx);

#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
if(*kk==1)
exit(1);
#endif



#endif


		// compute t & dlam & dt & alpha
		alpha = 1.0;
		d_compute_alpha_mpc_hard_tv(N, nx, nu, nb, idxb, ng, &alpha, t, dt, lam, dlam, lamt, dux, pDCt, d);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// update x, u, lam, t & compute the duality gap mu

		d_update_var_mpc_hard_tv(N, nx, nu, nb, ng, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);

		stat[5*(*kk)+4] = mu;
		
		// update sigma
/*		sigma *= sigma_decay;*/
/*		if(sigma<sigma_min)*/
/*			sigma = sigma_min;*/
/*		if(alpha<0.3)*/
/*			sigma = sigma_par[0];*/



		// increment loop index
		(*kk)++;


		} // end of IP loop
	
	// restore Hessian
	for(jj=0; jj<=N; jj++)
		{
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs] = bd[jj][ll];
			pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+idx*bs] = bl[jj][ll];
			}
		}



	// successful exit
	if(mu<=mu_tol)
		return 0;
	
	// max number of iterations reached
	if(*kk>=k_max)
		return 1;
	
	// no improvement
	if(alpha<alpha_min)
		return 2;
	
	// impossible
	return -1;

	} // end of ipsolver



/* primal-dual interior-point method, hard constraints, time variant matrices, time variant size (mpc version) */
void d_kkt_solve_new_rhs_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **pBAbt, double **r_A, double **pQ, double **r_H, double **pDCt, double **r_C, double r_T, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory)
	{

	
	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// matrices size
	int idx;
//	int nxM = 0;
	int nzM = 0;
//	int ngM = 0;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
//	int cnz[N+1];
	int cnux[N+1];

	for(jj=0; jj<=N; jj++)
		{
		pnx[jj] = (nx[jj]+bs-1)/bs*bs;
		pnz[jj] = (nu[jj]+nx[jj]+1+bs-1)/bs*bs;
		pnb[jj] = (nb[jj]+bs-1)/bs*bs;
		png[jj] = (ng[jj]+bs-1)/bs*bs;
		cnx[jj] = (nx[jj]+ncl-1)/ncl*ncl;
//		cnz[jj] = (nu[jj]+nx[jj]+1+ncl-1)/ncl*ncl;
		cnux[jj] = (nu[jj]+nx[jj]+ncl-1)/ncl*ncl;
//		if(nx[jj]>nxM) nxM = nx[jj];
		if(nu[jj]+nx[jj]+1>nzM) nzM = nu[jj]+nx[jj]+1;
//		if(ng[jj]>ngM) ngM = ng[jj];
//		if(nx[jj]+ng[jj]>nxgM) nxgM = nx[jj]+ng[jj];
		}

	int nxgM = ng[N];
	for(jj=0; jj<N; jj++)
		{
		if(nx[jj+1]+ng[jj]>nxgM) nxgM = nx[jj+1]+ng[jj];
		}


	// initialize work space
	// work_space_double_size_per_stage = pnz*cnl + 2*pnz + 2*pnx + 14*pnb + 10*png
	// work_space_double_size_const_max = pnz*cnxg + pnz
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *pL[N+1];
	double *dL[N+1];
	double *l[N+1];
	double *work;
	double *b[N];
	double *q[N+1];
	double *dux[N+1];
	double *dpi[N+1];
	double *pd[N+1]; // pointer to diagonal of Hessian
	double *pl[N+1]; // pointer to linear part of Hessian
	double *bd[N+1]; // backup diagonal of Hessian
	double *bl[N+1]; // backup linear part of Hessian
	double *diag;
	double *dlam[N+1];
	double *dt[N+1];
	double *lamt[N+1];
	double *t_inv[N+1];
	double *Qx[N+1];
	double *qx[N+1];
	double *qx2[N+1];
	double *Pb[N];

//	int size = 0;

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
//		ptr += pnz[jj] * ( cnx[jj]+ncl>cnz[jj] ? cnx[jj]+ncl : cnz[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
		ptr += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
//		size += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] ); // pnz*cnl // TODO use cnux instead of cnux !!!!!
		}

	for(jj=0; jj<=N; jj++)
		{
		dL[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		l[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	// work space
	work = ptr;
//	ptr += ((nzM+bs-1)/bs*bs) * ((nxM+ngM+ncl-1)/ncl*ncl); // pnzM*cnxgM
//	size += ((nzM+bs-1)/bs*bs) * ((nxM+ngM+ncl-1)/ncl*ncl); // pnzM*cnxgM
	ptr += ((nzM+bs-1)/bs*bs) * ((nxgM+ncl-1)/ncl*ncl); // pnzM*cnxgM
//	size += ((nzM+bs-1)/bs*bs) * ((nxgM+ncl-1)/ncl*ncl); // pnzM*cnxgM

	// b as vector
	for(jj=0; jj<N; jj++)
		{
		b[jj] = ptr;
		ptr += pnx[jj+1];
//		size += pnx[jj+1];
		d_copy_mat(1, nx[jj+1], pBAbt[jj]+(nu[jj]+nx[jj])/bs*bs*cnx[jj+1]+(nu[jj]+nx[jj])%bs, bs, b[jj], 1);
		}

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		}

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj];
//		size += pnx[jj];
		}
	
	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj+1];
//		size += pnx[jj+1];
		}

	// linear part of cost function
	for(jj=0; jj<=N; jj++)
		{
		q[jj] = ptr;
		ptr += pnz[jj];
//		size += pnz[jj];
		for(ll=0; ll<nu[jj]+nx[jj]; ll++) q[jj][ll] = pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+ll*bs];
		}

	// Hessian backup
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = ptr;
		pl[jj] = ptr + pnb[jj];
		bd[jj] = ptr + 2*pnb[jj];
		bl[jj] = ptr + 3*pnb[jj];
		ptr += 4*pnb[jj];
//		size += 4*pnb[jj];
		// backup
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			bd[jj][ll] = pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs];
			bl[jj][ll] = q[jj][idx];
			}
		}

	diag = ptr;
	ptr += (nzM+bs-1)/bs*bs; // pnzM
//	size += (nzM+bs-1)/bs*bs; // pnzM

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb[jj]+2*png[jj];
		ptr += 4*pnb[jj]+4*png[jj];
//		size += 4*pnb[jj]+4*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
//		size += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
//		size += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+pnb[jj]+png[jj];
		qx2[jj] = ptr+2*pnb[jj]+2*png[jj];
		ptr += 3*pnb[jj]+3*png[jj];
//		size += 3*pnb[jj]+3*png[jj];
		}
	
//	printf("\nsize real = %d\n", size);




	//update cost function vectors for a generic RHS (not tailored to IPM)
	d_update_gradient_new_rhs_mpc_hard_tv(N, nx, nu, nb, ng, r_T, t_inv, lamt, qx, bl, pl, r_C); 


	// solve the system
	d_back_ric_rec_trs_tv(N, nx, nu, pBAbt, r_A, pL, dL, r_H, l, ux, work, 1, Pb, compute_mult, pi, nb, idxb, pl, ng, pDCt, qx);


	// compute t & lam 
	d_compute_t_lam_new_rhs_mpc_hard_tv(N, nx, nu, nb, idxb, ng, r_T, t, lam, lamt, t_inv, ux, pDCt, r_C);

	


	} // end of final kkt solve



