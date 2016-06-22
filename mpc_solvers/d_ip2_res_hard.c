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
int d_ip2_res_mpc_hard_tv_work_space_size_bytes(int N, int *nx, int *nu, int *nb, int *ng)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int pnx, pnz, pnb, png, cnx, cnux; 

	int size = 0;
	int pnzM = 0;
	int pngM = 0;
	for(ii=0; ii<=N; ii++)
		{
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		if(pnz>pnzM) pnzM = pnz;
		pnb = (nb[ii]+bs-1)/bs*bs;
		png = (ng[ii]+bs-1)/bs*bs;
		if(png>pngM) pngM = png;
		cnx = (nx[ii]+ncl-1)/ncl*ncl;
		cnux = (nu[ii]+nx[ii]+ncl-1)/ncl*ncl;
		pnx = (nx[ii]+bs-1)/bs*bs;
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		size += pnz*(cnx+ncl>cnux ? cnx+ncl : cnux) + 5*pnx + 6*pnz + 18*pnb + 16*png;
		}

	size += 2*pngM;

	size *= sizeof(double);

	size += d_back_ric_rec_sv_tv_work_space_size_bytes(N, nx, nu, nb, ng);

	return size;
	}



/* primal-dual interior-point method computing residuals at each iteration, hard constraints, time variant matrices, time variant size (mpc version) */
int d_ip2_res_mpc_hard_tv(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *stat, int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **pBAbt, double **pQ, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory)
	{

	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// matrices size
	int idx;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];

	int pngM = 0;

	for(jj=0; jj<=N; jj++)
		{
		pnx[jj] = (nx[jj]+bs-1)/bs*bs;
		pnz[jj] = (nu[jj]+nx[jj]+1+bs-1)/bs*bs;
		pnb[jj] = (nb[jj]+bs-1)/bs*bs;
		png[jj] = (ng[jj]+bs-1)/bs*bs;
		if(png[jj]>pngM) pngM = png[jj];
		cnx[jj] = (nx[jj]+ncl-1)/ncl*ncl;
		cnux[jj] = (nu[jj]+nx[jj]+ncl-1)/ncl*ncl;
		}



	// initialize work space
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *pL[N+1];
	double *dL[N+1];
	double *l[N+1];
	double *work;
	double *b[N];
	double *q[N+1];
	double *dux[N+1];
	double *dpi[N];
	double *bd[N+1]; // backup diagonal of Hessian
	double *bl[N+1]; // backup gradient
	double *dlam[N+1];
	double *dt[N+1];
	double *t_inv[N+1];
	double *Qb[N+1];
	double *qb[N+1];
	double *Qx[N+1];
	double *qx[N+1];
	double *Pb[N];
	double *res_work;
	double *res_q[N+1];
	double *res_b[N];
	double *res_d[N+1];
	double *res_m[N+1];
	double *ux_bkp[N+1];
	double *pi_bkp[N];
	double *t_bkp[N+1];
	double *lam_bkp[N+1];

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		ptr += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] );
		}

	// work space
	work = ptr;
	ptr += d_back_ric_rec_sv_tv_work_space_size_bytes(N, nx, nu, nb, ng) / sizeof(double);

	for(jj=0; jj<=N; jj++)
		{
		dL[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		l[jj] = ptr;
		ptr += pnz[jj];
		}

	// b as vector
	for(jj=0; jj<N; jj++)
		{
		b[jj] = ptr;
		ptr += pnx[jj+1];
		d_copy_mat(1, nx[jj+1], pBAbt[jj]+(nu[jj]+nx[jj])/bs*bs*cnx[jj+1]+(nu[jj]+nx[jj])%bs, bs, b[jj], 1);
		}

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
		}

	// equality constr multipliers
	for(jj=0; jj<N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj+1];
		}
	
	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj+1];
		}

	// linear part of cost function (and copy it)
	for(jj=0; jj<=N; jj++)
		{
		q[jj] = ptr;
		ptr += pnz[jj];
		for(ll=0; ll<nu[jj]+nx[jj]; ll++) 
			q[jj][ll] = pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+ll*bs];
		}

	// diagonal of Hessian and gradient backup
	for(jj=0; jj<=N; jj++)
		{
		bd[jj] = ptr;
		bl[jj] = ptr+pnb[jj];
		ptr += 2*pnb[jj];
		// backup
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			bd[jj][ll] = pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs];
			bl[jj][ll] = q[jj][idx]; // XXX this has to come after q !!!
			}
		}

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb[jj]+2*png[jj];
		ptr += 4*pnb[jj]+4*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+pnb[jj]+png[jj];
		ptr += 2*pnb[jj]+2*png[jj];
		}
	
	// residuals
	res_work = ptr;
	ptr += 2*pngM;

	for(jj=0; jj<=N; jj++)
		{
		res_q[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<N; jj++)
		{
		res_b[jj] = ptr;
		ptr += pnx[jj+1];
		}

	for(jj=0; jj<=N; jj++)
		{
		res_d[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		res_m[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}
	
	for(jj=0; jj<=N; jj++)
		{
		ux_bkp[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<N; jj++)
		{
		pi_bkp[jj] = ptr;
		ptr += pnx[jj+1];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_bkp[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		lam_bkp[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}


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

	// check if there are inequality constraints
	double mu_scal = 0.0; 
	for(jj=0; jj<=N; jj++) mu_scal += 2*nb[jj] + 2*ng[jj];
	if(mu_scal!=0.0) // there are some constraints
		{
		mu_scal = 1.0 / mu_scal;
		}
	else // call the riccati solver and return
		{
		double **dummy;
		d_back_ric_rec_sv_tv(N, nx, nu, pBAbt, pQ, ux, pL, dL, work, 1, Pb, compute_mult, pi, nb, idxb, dummy, dummy, ng, dummy, dummy, dummy);
		// backup solution
		for(ii=0; ii<=N; ii++)
			for(jj=0; jj<nu[ii]+nx[ii]; jj++)
				ux_bkp[ii][jj] = ux[ii][jj];
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx[ii+1]; jj++)
				pi_bkp[ii][jj] = pi[ii][jj];
		// no IPM iterations
		*kk = 0;
		// return success
		return 0;
		}

	//printf("\nmu_scal = %f\n", mu_scal);
	double sigma = 0.0;
	//for(ii=0; ii<=N; ii++)
	//	printf("\n%d %d\n", nb[ii], ng[ii]);
	//exit(1);



	// initialize ux & pi & t>0 & lam>0
	d_init_var_mpc_hard_tv(N, nx, nu, nb, idxb, ng, ux, pi, pDCt, d, t, lam, mu0, warm_start);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], ux[ii], 1);
exit(1);
#endif



	// compute the duality gap
	mu = mu0;

	// set to zero iteration count
	*kk = 0;	

	// larger than minimum accepted step size
	alpha = 1.0;



	// compute residuals
	d_res_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, pBAbt, b, pQ, q, ux, pDCt, d, pi, lam, t, res_work, res_q, res_b, res_d, res_m, &mu);
#if 0
	printf("\nres_q\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, nu[jj]+nx[jj], res_q[jj], 1);
	printf("\nres_b\n");
	for(jj=0; jj<N; jj++)
		d_print_mat_e(1, nx[jj+1], res_b[jj], 1);
	printf("\nres_d\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_d[jj], 1);
	printf("\nres_m\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_m[jj], 1);
	printf("\nmu\n");
	d_print_mat_e(1, 1, &mu, 1);
	exit(2);
#endif



	// IP loop		
	while( *kk<k_max && mu>mu_tol && alpha>=alpha_min ) // XXX exit conditions on residuals???
		{



#if 0
printf("\nIPM it %d\n", *kk);
#endif
						


		// compute the update of Hessian and gradient from box and general constraints
		d_update_hessian_gradient_res_mpc_hard_tv(N, nx, nu, nb, ng, res_d, res_m, t, lam, t_inv, Qx, qx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, pnb[ii]+png[ii], Qx[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, pnb[ii]+png[ii], qx[ii], 1);
//if(*kk==1)
//exit(1);
#endif



		// compute the search direction: factorize and solve the KKT system
		// TODO sv
//		d_back_ric_rec_sv_tv(N, nx, nu, pBAbt, pQ, dux, pL, dL, work, 1, Pb, compute_mult, dpi, nb, idxb, pd, pl, ng, pDCt, Qx, qx2);
		d_back_ric_rec_trf_tv_res(N, nx, nu, pBAbt, pQ, pL, dL, work, nb, idxb, ng, pDCt, Qx, bd);
//		d_back_ric_rec_trs_tv(N, nx, nu, pBAbt, b, pL, dL, q, l, dux, work, 1, Pb, compute_mult, dpi, nb, idxb, pl, ng, pDCt, qx);
		d_back_ric_rec_trs_tv_res(N, nx, nu, pBAbt, res_b, pL, dL, res_q, l, dux, work, 1, Pb, compute_mult, dpi, nb, idxb, ng, pDCt, qx);

		
#if 0
printf("\npL\n");
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]+1, bs, pL[ii], cnux[ii]);
printf("\ndL\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dL[ii], 1);
exit(1);
#endif
#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
printf("\ndpi\n");
for(ii=0; ii<N; ii++)
	d_print_mat(1, nx[ii+1], dpi[ii], 1);
//if(*kk==1)
exit(1);
#endif



#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]+1, bs, pQ[ii], cnux[ii]);
//exit(1);
#endif
#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, pL[ii], cnux[ii]);
//exit(1);
#endif
#if 0
printf("\nux_aff\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
printf("\npi_aff\n");
for(ii=0; ii<N; ii++)
	d_print_mat(1, nx[ii+1], dpi[ii], 1);
if(*kk==1)
exit(1);
#endif


#if 1 // IPM1

		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		d_compute_alpha_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, dux, t, t_inv, lam, pDCt, res_d, res_m, dt, dlam, &alpha);

		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;

#if 0
printf("\nalpha = %f\n", alpha);
exit(1);
#endif


		// compute the affine duality gap
		d_compute_mu_res_mpc_hard_tv(N, nx, nu, nb, ng, alpha, lam, dlam, t, dt, mu_scal, &mu_aff);

		stat[5*(*kk)+2] = mu_aff;

#if 0
printf("\nmu = %f\n", mu_aff);
exit(1);
#endif



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


		// update res_m
		d_compute_centering_correction_res_mpc_hard_tv(N, nb, ng, sigma*mu, dt, dlam, res_m);



		// update gradient
		d_update_gradient_res_mpc_hard_tv(N, nx, nu, nb, ng, res_d, res_m, lam, t_inv, qx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, pnb[ii]+png[ii], qx[ii], 1);
if(*kk==1)
exit(1);
#endif



		// solve the system
		d_back_ric_rec_trs_tv_res(N, nx, nu, pBAbt, res_b, pL, dL, res_q, l, dux, work, 0, Pb, compute_mult, dpi, nb, idxb, ng, pDCt, qx);

#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
if(*kk==1)
exit(1);
#endif



#endif // end of IPM1


		// compute t & dlam & dt & alpha
		alpha = 1.0;
		d_compute_alpha_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, dux, t, t_inv, lam, pDCt, res_d, res_m, dt, dlam, &alpha);

#if 0
printf("\nalpha = %f\n", alpha);
printf("\nd\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], d[ii], 1);
printf("\nres_d\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], res_d[ii], 1);
printf("\ndt\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], dt[ii], 1);
printf("\ndlam\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], dlam[ii], 1);
exit(2);
#endif

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// backup & update x, u, pi, lam, t 
		d_backup_update_var_res_mpc_hard_tv(N, nx, nu, nb, ng, alpha, ux_bkp, ux, dux, pi_bkp, pi, dpi, t_bkp, t, dt, lam_bkp, lam, dlam);
//		d_update_var_res_mpc_hard_tv(N, nx, nu, nb, ng, alpha, ux, dux, pi, dpi, t, dt, lam, dlam);


#if 0
printf("\nux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], ux[ii], 1);
printf("\npi\n");
for(ii=0; ii<N; ii++)
	d_print_mat(1, nx[ii+1], pi[ii], 1);
printf("\nlam\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], lam[ii], 1);
printf("\nt\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], t[ii], 1);
//if(*kk==1)
//exit(1);
#endif


		// compute residuals
		// restore Hessian
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nb[jj]; ll++)
				{
				idx = idxb[jj][ll];
				pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs] = bd[jj][ll];
				}
			}
		d_res_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, pBAbt, b, pQ, q, ux, pDCt, d, pi, lam, t, res_work, res_q, res_b, res_d, res_m, &mu);
#if 0
		printf("\nres_q\n");
		for(jj=0; jj<=N; jj++)
			d_print_mat_e(1, nu[jj]+nx[jj], res_q[jj], 1);
		printf("\nres_b\n");
		for(jj=0; jj<N; jj++)
			d_print_mat_e(1, nx[jj+1], res_b[jj], 1);
		printf("\nres_d\n");
		for(jj=0; jj<=N; jj++)
			d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_d[jj], 1);
		printf("\nres_m\n");
		for(jj=0; jj<=N; jj++)
			d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_m[jj], 1);
		printf("\nmu\n");
		d_print_mat_e(1, 1, &mu, 1);
//		exit(2);
#endif

		stat[5*(*kk)+4] = mu;
		


		// increment loop index
		(*kk)++;


		} // end of IP loop
	


	// restore Hessian XXX and gradient
//	for(jj=0; jj<=N; jj++)
//		{
//		for(ll=0; ll<nb[jj]; ll++)
//			{
//			idx = idxb[jj][ll];
//			pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs] = bd[jj][ll];
//			pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+idx*bs] = bl[jj][ll];
//			}
//		}

#if 0
printf("\nQ\n");
for(jj=0; jj<=N; jj++)
	d_print_pmat(nu[jj]+nx[jj]+1, nu[jj]+nx[jj], bs, pQ[jj], cnux[jj]);
printf("\nux\n");
for(jj=0; jj<=N; jj++)
	d_print_mat(1, nu[jj]+nx[jj], ux[jj], 1);
exit(2);
#endif



	// TODO if mu is nan, recover solution !!!
//	if(mu==1.0/0.0 || mu==-1.0/0.0)
//		{
//		printf("\nnan!!!\n");
//		exit(3);
//		}



//exit(2);

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



void d_kkt_solve_new_rhs_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **pBAbt, double **b, double **pQ, double **q, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory)
	{
	
	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// matrices size
	int idx;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int cnx[N+1];
	int cnux[N+1];

	int pngM = 0;

	for(jj=0; jj<=N; jj++)
		{
		pnx[jj] = (nx[jj]+bs-1)/bs*bs;
		pnz[jj] = (nu[jj]+nx[jj]+1+bs-1)/bs*bs;
		pnb[jj] = (nb[jj]+bs-1)/bs*bs;
		png[jj] = (ng[jj]+bs-1)/bs*bs;
		if(png[jj]>pngM) pngM = png[jj];
		cnx[jj] = (nx[jj]+ncl-1)/ncl*ncl;
		cnux[jj] = (nu[jj]+nx[jj]+ncl-1)/ncl*ncl;
		}



	// initialize work space
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *pL[N+1];
	double *dL[N+1];
	double *l[N+1];
	double *work;
	double *b_old[N];
	double *q_old[N+1];
	double *dux[N+1];
	double *dpi[N];
	double *bd[N+1]; // backup diagonal of Hessian
	double *bl[N+1]; // backup diagonal of Hessian
	double *dlam[N+1];
	double *dt[N+1];
	double *t_inv[N+1];
	double *Qb[N+1];
	double *qb[N+1];
	double *Qx[N+1];
	double *qx[N+1];
	double *Pb[N];
	double *res_work;
	double *res_q[N+1];
	double *res_b[N];
	double *res_d[N+1];
	double *res_m[N+1];
	double *ux_bkp[N+1];
	double *pi_bkp[N];
	double *t_bkp[N+1];
	double *lam_bkp[N+1];

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		ptr += pnz[jj] * ( cnx[jj]+ncl>cnux[jj] ? cnx[jj]+ncl : cnux[jj] );
		}

	// work space
	work = ptr;
	ptr += d_back_ric_rec_sv_tv_work_space_size_bytes(N, nx, nu, nb, ng) / sizeof(double);

	for(jj=0; jj<=N; jj++)
		{
		dL[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		l[jj] = ptr;
		ptr += pnz[jj];
		}

	// b as vector
	for(jj=0; jj<N; jj++)
		{
		b_old[jj] = ptr;
		ptr += pnx[jj+1];
		}

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
		}

	// equality constr multipliers
	for(jj=0; jj<N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj+1];
		}
	
	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj+1];
		}

	// linear part of cost function (and copy it)
	for(jj=0; jj<=N; jj++)
		{
		q_old[jj] = ptr;
		ptr += pnz[jj];
		}

	// diagonal of Hessian backup
	for(jj=0; jj<=N; jj++)
		{
		bd[jj] = ptr;
		bl[jj] = ptr+pnb[jj];
		ptr += 2*pnb[jj];
//		// backup
//		for(ll=0; ll<nb[jj]; ll++)
//			{
//			idx = idxb[jj][ll];
//			bd[jj][ll] = pQ[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs];
//			bl[jj][ll] = q[jj][idx]; // XXX this has to come after q !!!
//			}
		}

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb[jj]+2*png[jj];
		ptr += 4*pnb[jj]+4*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+pnb[jj]+png[jj];
		ptr += 2*pnb[jj]+2*png[jj];
		}
	
	// residuals
	res_work = ptr;
	ptr += 2*pngM;

	for(jj=0; jj<=N; jj++)
		{
		res_q[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<N; jj++)
		{
		res_b[jj] = ptr;
		ptr += pnx[jj+1];
		}

	for(jj=0; jj<=N; jj++)
		{
		res_d[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		res_m[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}
	
	for(jj=0; jj<=N; jj++)
		{
		ux_bkp[jj] = ptr;
		ptr += pnz[jj];
		}

	for(jj=0; jj<N; jj++)
		{
		pi_bkp[jj] = ptr;
		ptr += pnx[jj+1];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_bkp[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		lam_bkp[jj] = ptr;
		ptr += 2*pnb[jj]+2*png[jj];
		}

#if 1

	double mu;

	// initialize solution with backup
	// bkp ux
	for(ii=0; ii<=N; ii++)
		for(jj=0; jj<nu[ii]+nx[ii]; jj++)
			ux[ii][jj] = ux_bkp[ii][jj];
	// bkp pi
	for(ii=0; ii<N; ii++)
		for(jj=0; jj<nx[ii+1]; jj++)
			pi[ii][jj] = pi_bkp[ii][jj];
	// bkp t
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++)
			{
			t[ii][jj] = t_bkp[ii][jj];
			t[ii][pnb[ii]+jj] = t_bkp[ii][pnb[ii]+jj];
			}
		for(jj=0; jj<ng[ii]; jj++)
			{
			t[ii][2*pnb[ii]+jj] = t_bkp[ii][2*pnb[ii]+jj];
			t[ii][2*pnb[ii]+png[ii]+jj] = t_bkp[ii][2*pnb[ii]+png[ii]+jj];
			}
		}
	// bkp lam
	for(ii=0; ii<=N; ii++)
		{
		for(jj=0; jj<nb[ii]; jj++)
			{
			lam[ii][jj] = lam_bkp[ii][jj];
			lam[ii][pnb[ii]+jj] = lam_bkp[ii][pnb[ii]+jj];
			}
		for(jj=0; jj<ng[ii]; jj++)
			{
			lam[ii][2*pnb[ii]+jj] = lam_bkp[ii][2*pnb[ii]+jj];
			lam[ii][2*pnb[ii]+png[ii]+jj] = lam_bkp[ii][2*pnb[ii]+png[ii]+jj];
			}
		}
	
#if 0
printf("\nux_bkp\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], ux[ii], 1);
printf("\npi_bkp\n");
for(ii=0; ii<N; ii++)
	d_print_mat(1, nx[ii+1], pi[ii], 1);
printf("\nlam_bkp\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], lam[ii], 1);
printf("\nt_bkp\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], t[ii], 1);
exit(2);
#endif

	// compute residuals
	d_res_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, pBAbt, b, pQ, q, ux, pDCt, d, pi, lam, t, res_work, res_q, res_b, res_d, res_m, &mu);

#if 0
printf("\nres_q\n");
for(jj=0; jj<=N; jj++)
	d_print_mat_e(1, nu[jj]+nx[jj], res_q[jj], 1);
printf("\nres_b\n");
for(jj=0; jj<N; jj++)
	d_print_mat_e(1, nx[jj+1], res_b[jj], 1);
printf("\nres_d\n");
for(jj=0; jj<=N; jj++)
	d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_d[jj], 1);
printf("\nres_m\n");
for(jj=0; jj<=N; jj++)
	d_print_mat_e(1, 2*pnb[jj]+2*png[jj], res_m[jj], 1);
printf("\nmu\n");
d_print_mat_e(1, 1, &mu, 1);
exit(2);
#endif

	// update gradient
	d_update_gradient_res_mpc_hard_tv(N, nx, nu, nb, ng, res_d, res_m, lam, t_inv, qx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, pnb[ii]+png[ii], qx[ii], 1);
exit(2);
#endif

	// solve the system
	d_back_ric_rec_trs_tv_res(N, nx, nu, pBAbt, res_b, pL, dL, res_q, l, dux, work, 1, Pb, compute_mult, dpi, nb, idxb, ng, pDCt, qx);

#if 0
printf("\nNEW dux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
//if(*kk==1)
//exit(1);
#endif

	// compute t & dlam & dt
	d_compute_dt_dlam_res_mpc_hard_tv(N, nx, nu, nb, idxb, ng, dux, t, t_inv, lam, pDCt, res_d, res_m, dt, dlam);

	// update x, u, lam, t & compute the duality gap mu
	d_update_var_res_mpc_hard_tv(N, nx, nu, nb, ng, 1.0, ux, dux, pi, dpi, t, dt, lam, dlam);

#else

	// XXX temporary allocation
	double *lamt[N+1];
	for(ii=0; ii<=N; ii++)
		{
		d_zeros_align(&lamt[ii], 2*pnb[ii]+2*png[ii], 1);
		for(jj=0; jj<nb[ii]; jj++)
			{
			lamt[ii][jj] = lam_bkp[ii][jj] * t_inv[ii][jj];
			lamt[ii][pnb[ii]+jj] = lam_bkp[ii][pnb[ii]+jj] * t_inv[ii][pnb[ii]+jj];
			}
		for(jj=0; jj<ng[ii]; jj++)
			{
			lamt[ii][jj] = lam_bkp[ii][jj] * t_inv[ii][jj];
			lamt[ii][2*pnb[ii]+png[ii]+jj] = lam_bkp[ii][2*pnb[ii]+png[ii]+jj] * t_inv[ii][2*pnb[ii]+png[ii]+jj];
			}
		}
	
	double *bl[N+1];
	for(ii=0; ii<=N; ii++)
		{
		d_zeros_align(&bl[ii], nb[ii], 1);
		for(jj=0; jj<nb[ii]; jj++)
			{
			bl[ii][jj] = r_H[ii][idxb[ii][jj]];
			}
		}

	double *pl[N+1];
	for(ii=0; ii<=N; ii++)
		d_zeros_align(&pl[ii], nb[ii], 1);



	//update cost function vectors for a generic RHS (not tailored to IPM)
	d_update_gradient_new_rhs_mpc_hard_tv(N, nx, nu, nb, ng, t_inv, lamt, qx, bl, pl, r_C); 


	// solve the system
	d_back_ric_rec_trs_tv(N, nx, nu, pBAbt, r_A, pL, dL, r_H, l, ux, work, 1, Pb, compute_mult, pi, nb, idxb, pl, ng, pDCt, qx);


	// compute t & lam 
	d_compute_t_lam_new_rhs_mpc_hard_tv(N, nx, nu, nb, idxb, ng, t, lam, lamt, t_inv, ux, pDCt, r_C);



	// free memory
	for(ii=0; ii<=N; ii++)
		{
		free(lamt[ii]);
		free(bl[ii]);
		free(pl[ii]);
		}

#endif
	


	} // end of final kkt solve


