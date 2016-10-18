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

#ifdef BLASFEO
#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>
#endif
//#else
#include "../include/blas_d.h"
//#endif

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"
#include "../include/mpc_aux.h"
#include "../include/d_blas_aux.h"



#define CORRECTOR_LOW 1



/* primal-dual interior-point method computing residuals at each iteration, hard constraints, time variant matrices, time variant size (mpc version) */
int d_tree_ip2_res_mpc_hard_libstr(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *stat, int N, int *nx, int *nu_N, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strmat *hsRSQrq, struct d_strmat *hsDCt, struct d_strvec *hsd, struct d_strvec *hsux, int compute_mult, struct d_strvec *hspi, struct d_strvec *hslam, struct d_strvec *hst, double *double_work_memory)
	{

	// indeces
	int jj, ll, ii, bs0, it_ref;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// nu with nu[N]=0
	int nu[N+1];
	for(ii=0; ii<N; ii++)
		nu[ii] = nu_N[ii];
	nu[N] = 0;



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



#if 0
printf("\nBAbt\n");
for(ii=0; ii<N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nx[ii+1], bs, pBAbt[ii], cnux[ii+1]);
printf("\nRSQrq\n");
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], bs, pQ[ii], cnux[ii]);
printf("\nd\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*pnb[ii]+2*png[ii], d[ii], 1);
exit(2);
#endif



	// initialize work space
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *work;
	double *memory;
	double *b[N+1];
	double *q[N+1];
	double *dux[N+1];
	double *dpi[N+1];
	double *bd[N+1]; // backup diagonal of Hessian
//	double *bl[N+1]; // backup gradient
	double *dlam[N+1];
	double *dt[N+1];
	double *t_inv[N+1];
	double *lamt[N+1];
	double *Qx[N+1];
	double *qx[N+1];
	double *Pb[N+1];
	double *res_work;
	double *res_q[N+1];
	double *res_b[N+1];
	double *res_d[N+1];
	double *res_m[N+1];

	// work space
	work = ptr;
	ptr += d_back_ric_rec_sv_tv_work_space_size_bytes(N, nx, nu, nb, ng) / sizeof(double);

	// memory space
	memory = ptr;
	ptr += d_back_ric_rec_sv_tv_memory_space_size_bytes(N, nx, nu, nb, ng) / sizeof(double);

	// b as vector
	for(jj=1; jj<=N; jj++)
		{
		b[jj] = ptr;
		ptr += pnx[jj];
//		for(ii=0; ii<nx[jj+1]; ii++)
//			b[jj][ii] = pBAbt[jj][(nu[jj]+nx[jj])/bs*bs*cnx[jj]+(nu[jj]+nx[jj])%bs+ii*bs];
		}

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
		}

	// equality constr multipliers
	for(jj=1; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj];
		}
	
	// backup of P*b
	for(jj=1; jj<=N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj];
		}

	// linear part of cost function (and copy it)
	for(jj=0; jj<=N; jj++)
		{
		q[jj] = ptr;
		ptr += pnz[jj];
//		for(ll=0; ll<nu[jj]+nx[jj]; ll++) 
//			q[jj][ll] = pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+ll*bs];
		}

	// diagonal of Hessian and gradient backup
	for(jj=0; jj<=N; jj++)
		{
		bd[jj] = ptr;
//		bl[jj] = ptr+pnb[jj];
//		ptr += 2*pnb[jj];
		ptr += pnb[jj];
		// backup
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
		lamt[jj] = ptr;
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

	for(jj=1; jj<=N; jj++)
		{
		res_b[jj] = ptr;
		ptr += pnx[jj];
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
	
#if 1 // libstr interface

		int cnl[N+1];
		for(ii=0; ii<N; ii++)
			{
			cnl[ii] = cnux[ii]<cnx[ii]+ncl ? cnx[ii]+ncl : cnux[ii];
			}
		cnl[ii] = cnux[ii]<cnx[ii]+ncl ? cnx[ii]+ncl : cnux[ii];

		int nzM  = 0;
		for(ii=0; ii<N; ii++)
			{
			if(nu[ii]+nx[ii]+1>nzM) nzM = nu[ii]+nx[ii]+1;
			}
		ii = N;
		if(nx[ii]+1>nzM) nzM = nx[ii]+1;

		int nxgM = ng[N];
		for(ii=0; ii<N; ii++)
			{
			if(nx[ii+1]+ng[ii]>nxgM) nxgM = nx[ii+1]+ng[ii];
			}
		
		int pnzM = (nzM+bs-1)/bs*bs;
		int cnxgM = (nxgM+ncl-1)/ncl*ncl;

		double *hpL[N+1];
		double *memory2 = memory;
		for(ii=0; ii<=N; ii++)
			{
			hpL[ii] = memory2;
			memory2 += pnz[ii]*cnl[ii];
			}

		double *hdL[N+1];
		for(ii=0; ii<=N; ii++)
			{
			hdL[ii] = memory2;
			memory2 += pnz[ii];
			}
		
//		struct d_strmat hsBAbt[N+1];
		struct d_strvec hsb[N+1];
//		struct d_strmat hsRSQrq[N+1];
		struct d_strvec hsrq[N+1];
		struct d_strvec hsdRSQ[N+1];
//		struct d_strmat hsDCt[N+1];
//		struct d_strvec hsd[N+1];
		struct d_strvec hsQx[N+1];
		struct d_strvec hsqx[N+1];
//		struct d_strvec hsux[N+1];
//		struct d_strvec hspi[N+1];
//		struct d_strvec hst[N+1];
//		struct d_strvec hslam[N+1];
		struct d_strvec hsdux[N+1];
		struct d_strvec hsdpi[N+1];
		struct d_strvec hsdt[N+1];
		struct d_strvec hsdlam[N+1];
		struct d_strvec hstinv[N+1];
		struct d_strvec hslamt[N+1];
		struct d_strvec hsPb[N+1];
		struct d_strmat hsL[N+1];
		struct d_strmat hsLxt[N+1];
		struct d_strvec hsres_q[N+1];
		struct d_strvec hsres_b[N+1];
		struct d_strvec hsres_d[N+1];
		struct d_strvec hsres_m[N+1];
		struct d_strmat hswork_mat[2];
		struct d_strvec hswork_vec[1];
		struct d_strvec hsres_work[2];
		struct d_strmat *hsmatdummy;
		struct d_strvec *hsvecdummy;

		for(ii=0; ii<=N; ii++)
			{
//			d_create_strmat(nu[ii]+nx[ii]+1, nx[ii+1], &hsBAbt[ii], (void *) pBAbt[ii]);
//			hsBAbt[ii].cn = cnx[ii];
			d_create_strvec(nx[ii], &hsb[ii], (void *) b[ii]);
//			d_create_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsRSQrq[ii], (void *) pQ[ii]);
//			hsRSQrq[ii].cn = cnux[ii];
			d_create_strvec(nu[ii]+nx[ii], &hsrq[ii], (void *) q[ii]);
			d_create_strvec(nb[ii], &hsdRSQ[ii], (void *) bd[ii]);
//			d_create_strvec(nb[ii]+ng[ii], &hsd[ii], (void *) d[ii]);
			d_create_strvec(nb[ii]+ng[ii], &hsQx[ii], (void *) Qx[ii]);
			d_create_strvec(nb[ii]+ng[ii], &hsqx[ii], (void *) qx[ii]);
			d_create_strvec(nx[ii+1], &hsPb[ii], (void *) Pb[ii]);
			d_create_strmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii], &hsL[ii], (void *) hpL[ii]);
			hsL[ii].dA = hdL[ii];
			hsL[ii].cn = cnl[ii];
			d_create_strmat(nx[ii], nx[ii], &hsLxt[ii], (void *) (hpL[ii]+ncl*bs));
			hsLxt[ii].cn = cnl[ii];
//			d_create_strvec(nu[ii]+nx[ii], &hsux[ii], (void *) ux[ii]);
//			d_create_strvec(nx[ii], &hspi[ii], (void *) pi[ii]);
//			d_create_strvec(2*nb[ii]+2*ng[ii], &hst[ii], (void *) t[ii]);
//			d_create_strvec(2*nb[ii]+2*ng[ii], &hslam[ii], (void *) lam[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hsdt[ii], (void *) dt[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hstinv[ii], (void *) t_inv[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hslamt[ii], (void *) lamt[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hsdlam[ii], (void *) dlam[ii]);
			d_create_strvec(nu[ii]+nx[ii], &hsdux[ii], (void *) dux[ii]);
			d_create_strvec(nx[ii], &hsdpi[ii], (void *) dpi[ii]);
			d_create_strvec(nu[ii]+nx[ii], &hsres_q[ii], (void *) res_q[ii]);
			d_create_strvec(nx[ii], &hsres_b[ii], (void *) res_b[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hsres_d[ii], (void *) res_d[ii]);
			d_create_strvec(2*nb[ii]+2*ng[ii], &hsres_m[ii], (void *) res_m[ii]);
			}
		d_create_strmat(pnzM, nxgM, &hswork_mat[0], (void *) work);
		hswork_mat[0].cn = cnxgM;
		d_create_strmat(pnzM, nxgM, &hswork_mat[1], (void *) (work+pnzM*cnxgM));
		hswork_mat[1].cn = cnxgM;
		d_create_strvec(pnzM, &hswork_vec[0], (void *) (work+2*pnzM*cnxgM));
		d_create_strvec(pnzM, &hsres_work[0], (void *) res_work);
		d_create_strvec(pnzM, &hsres_work[1], (void *) (res_work+pngM));

#endif

	// extract arrays
	double *hpRSQrq[N+1];
	for(jj=0; jj<=N; jj++)
		hpRSQrq[jj] = hsRSQrq[jj].pA;

	// extract b
	for(jj=1; jj<=N; jj++)
		{
		drowex_libstr(nx[jj], 1.0, &hsBAbt[jj], nu[jj-1]+nx[jj-1], 0, &hsb[jj], 0);
		}

	// extract q
	for(jj=0; jj<=N; jj++)
		{
		drowex_libstr(nu[jj]+nx[jj], 1.0, &hsRSQrq[jj], nu[jj]+nx[jj], 0, &hsrq[jj], 0);
		}

	// extract diagonal of Hessian and gradient 
	for(jj=0; jj<=N; jj++)
		{
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			bd[jj][ll] = hpRSQrq[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs];
//			bl[jj][ll] = q[jj][idx]; // XXX this has to come after q !!!
			}
		}








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
//		d_back_ric_rec_sv_tv_res(N, nx, nu, nb, idxb, ng, 0, pBAbt, b, 0, pQ, q, dummy, dummy, dummy, dummy, ux, compute_mult, pi, 1, Pb, memory, work);
		d_back_ric_rec_sv_libstr(N, nx, nu, nb, idxb, ng, 0, hsBAbt, hsb, 0, hsRSQrq, hsrq, hsvecdummy, hsmatdummy, hsvecdummy, hsvecdummy, hsdux, compute_mult, hsdpi, 1, hsPb, hsL, hsLxt, hswork_mat, hswork_vec);
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
	d_init_var_mpc_hard_libstr(N, nx, nu, nb, idxb, ng, hsux, hspi, hsDCt, hsd, hst, hslam, mu0, warm_start);

#if 0
printf("\nux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], ux[ii], 1);
printf("\npi\n");
for(ii=0; ii<N; ii++)
	d_print_mat(1, nx[ii+1], pi[ii+1], 1);
printf("\nlam\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], lam[ii], 1);
printf("\nt\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], t[ii], 1);
exit(1);
#endif



	// compute the duality gap
	mu = mu0;

	// set to zero iteration count
	*kk = 0;	

	// larger than minimum accepted step size
	alpha = 1.0;





	//
	// loop without residuals compuation at early iterations
	//

	double mu_tol_low = mu_tol;

#if 0
	if(0)
#else
	while( *kk<k_max && mu>mu_tol_low && alpha>=alpha_min )
#endif
		{

//		printf("\nkk = %d (no res)\n", *kk);
						


		//update cost function matrices and vectors (box constraints)
		d_update_hessian_mpc_hard_libstr(N, nx, nu, nb, ng, hsd, 0.0, hst, hstinv, hslam, hslamt, hsdlam, hsQx, hsqx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], lam[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], t[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii]+ng[ii], Qx[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii]+ng[ii], qx[ii], 1);
//if(*kk==1)
exit(1);
#endif


		// compute the search direction: factorize and solve the KKT system
#if 1
		d_back_ric_rec_sv_libstr(N, nx, nu, nb, idxb, ng, 0, hsBAbt, hsb, 1, hsRSQrq, hsrq, hsdRSQ, hsDCt, hsQx, hsqx, hsdux, compute_mult, hsdpi, 1, hsPb, hsL, hsLxt, hswork_mat, hswork_vec);
#else
		d_back_ric_rec_trf_tv_res(N, nx, nu, pBAbt, pQ, pL, dL, work, nb, idxb, ng, pDCt, Qx, bd);
		d_back_ric_rec_trs_tv_res(N, nx, nu, pBAbt, b, pL, dL, q, l, dux, work, 1, Pb, compute_mult, dpi, nb, idxb, ng, pDCt, qx);
#endif


#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(nu[ii]+nx[ii]+1, nu[ii]+nx[ii]+1, bs, pQ[ii], cnux[ii]);
exit(1);
#endif
#if 0
for(ii=0; ii<=N; ii++)
	d_print_pmat(pnz[ii], cnux[ii], hpL[ii], cnux[ii]);
if(*kk==2)
exit(1);
#endif
#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
printf("\ndpi\n");
for(ii=1; ii<=N; ii++)
	d_print_mat(1, nx[ii], dpi[ii], 1);
//if(*kk==2)
exit(1);
#endif


#if CORRECTOR_LOW==1 // IPM1

		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		d_compute_alpha_mpc_hard_libstr(N, nx, nu, nb, idxb, ng, &alpha, hst, hsdt, hslam, hsdlam, hslamt, hsdux, hsDCt, hsd);

		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;

#if 0
printf("\nalpha = %f\n", alpha);
exit(1);
#endif


		// compute the affine duality gap
		d_compute_mu_mpc_hard_libstr(N, nx, nu, nb, ng, &mu_aff, mu_scal, alpha, hslam, hsdlam, hst, hsdt);

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


//		d_update_gradient_mpc_hard_tv(N, nx, nu, nb, ng, sigma*mu, dt, dlam, t_inv, pl, qx);
		d_update_gradient_mpc_hard_libstr(N, nx, nu, nb, ng, sigma*mu, hsdt, hsdlam, hstinv, hsqx);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii]+ng[ii], qx[ii], 1);
//if(*kk==1)
exit(1);
#endif



//		// copy b into x
//		for(ii=0; ii<N; ii++)
//			for(jj=0; jj<nx[ii+1]; jj++) 
//				dux[ii+1][nu[ii+1]+jj] = pBAbt[ii][(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs+bs*jj]; // copy b



		// solve the system
		d_back_ric_rec_trs_libstr(N, nx, nu, nb, idxb, ng, hsBAbt, hsb, hsrq, hsDCt, hsqx, hsdux, compute_mult, hsdpi, 0, hsPb, hsL, hsLxt, hswork_vec);

#if 0
printf("\ndux\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nu[ii]+nx[ii], dux[ii], 1);
//if(*kk==1)
exit(1);
#endif



#endif // end of IPM1


		// compute t & dlam & dt & alpha
		alpha = 1.0;
		d_compute_alpha_mpc_hard_libstr(N, nx, nu, nb, idxb, ng, &alpha, hst, hsdt, hslam, hsdlam, hslamt, hsdux, hsDCt, hsd);

#if 0
printf("\nalpha = %f\n", alpha);
printf("\ndt\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], dt[ii], 1);
printf("\ndlam\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], dlam[ii], 1);
exit(2);
#endif

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// compute step & update x, u, lam, t & compute the duality gap mu
		d_update_var_mpc_hard_libstr(N, nx, nu, nb, ng, &mu, mu_scal, alpha, hsux, hsdux, hst, hsdt, hslam, hsdlam, hspi, hsdpi);

		stat[5*(*kk)+4] = mu;
		

#if 0
printf("\nux\n");
for(ii=0; ii<=N; ii++)
	d_print_tran_strvec(nu[ii]+nx[ii], &hsux[ii], 0);
printf("\npi\n");
for(ii=1; ii<=N; ii++)
	d_print_tran_strvec(nx[ii], &hspi[ii], 0);
printf("\nlam\n");
for(ii=0; ii<=N; ii++)
	d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hslam[ii], 0);
printf("\nt\n");
for(ii=0; ii<=N; ii++)
	d_print_tran_strvec(2*nb[ii]+2*ng[ii], &hst[ii], 0);
//if(*kk==1)
exit(1);
#endif


		// increment loop index
		(*kk)++;


		} // end of IP loop
	

	// restore Hessian
	for(jj=0; jj<=N; jj++)
		{
		for(ll=0; ll<nb[jj]; ll++)
			{
			idx = idxb[jj][ll];
			hpRSQrq[jj][idx/bs*bs*cnux[jj]+idx%bs+idx*bs] = bd[jj][ll];
//			pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+idx*bs] = bl[jj][ll];
			}
		for(ll=0; ll<nu[jj]+nx[jj]; ll++) 
			hpRSQrq[jj][(nu[jj]+nx[jj])/bs*bs*cnux[jj]+(nu[jj]+nx[jj])%bs+ll*bs] = q[jj][ll];
		}

#if 0
printf("\nux\n");
for(jj=0; jj<=N; jj++)
	d_print_mat(1, nu[jj]+nx[jj], ux[jj], 1);
printf("\npi\n");
for(jj=1; jj<=N; jj++)
	d_print_mat(1, nx[jj], pi[jj], 1);
printf("\nlam\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], lam[ii], 1);
printf("\nt\n");
for(ii=0; ii<=N; ii++)
	d_print_mat(1, 2*nb[ii]+2*ng[ii], t[ii], 1);
exit(2);
#endif


	//
	// loop with residuals computation and iterative refinement for high-accuracy result
	//

	// compute residuals
	d_res_res_mpc_hard_libstr(N, nx, nu, nb, idxb, ng, hsBAbt, hsb, hsRSQrq, hsrq, hsux, hsDCt, hsd, hspi, hslam, hst, hsres_work, hsres_q, hsres_b, hsres_d, hsres_m, &mu);
#if 0
	printf("\nres_q\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, nu[jj]+nx[jj], res_q[jj], 1);
	printf("\nres_b\n");
	for(jj=1; jj<=N; jj++)
		d_print_mat_e(1, nx[jj], res_b[jj], 1);
	printf("\nres_d\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, 2*nb[jj]+2*ng[jj], res_d[jj], 1);
	printf("\nres_m\n");
	for(jj=0; jj<=N; jj++)
		d_print_mat_e(1, 2*nb[jj]+2*ng[jj], res_m[jj], 1);
	printf("\nmu\n");
	d_print_mat_e(1, 1, &mu, 1);
	exit(2);
#endif


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


