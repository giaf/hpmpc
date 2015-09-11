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

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"
#include "../include/mpc_aux.h"






int d_ip2_soft_mpc_tv_work_space_size_double(int N, int *nx, int *nu, int *nb, int *ng, int *ns)
	{

	const int bs = D_MR;
	const int ncl = D_NCL;

	int ii;

	int pnx, pnz, pnb, png, pns, cnx, cnz;

	int size = 0;
	int pnzM = 0;
	int nxgM = 0;
	for(ii=0; ii<=N; ii++)
		{
		if(nx[ii]+ng[ii]>nxgM) nxgM = nx[ii]+ng[ii];
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		if(pnz>pnzM) pnzM = pnz;
		pnb = (nb[ii]+bs-1)/bs*bs;
		png = (ng[ii]+bs-1)/bs*bs;
		pns = (ns[ii]+bs-1)/bs*bs;
		cnx = (nx[ii]+ncl-1)/ncl*ncl;
		cnz = (nx[ii]+nu[ii]+1+ncl-1)/ncl*ncl;
		pnx = (nx[ii]+bs-1)/bs*bs;
		pnz = (nx[ii]+nu[ii]+1+bs-1)/bs*bs;
		size += pnz*(cnx+ncl>cnz ? cnx+ncl : cnz) + 2*pnx + 4*pnz + 12*pnb + 11*png + 24*pns;
		}
	size += pnzM*((nxgM+ncl-1)/ncl*ncl) + pnzM;

	return size;
	}



/* primal-dual interior-point method, hard constraints, time variant matrices, time variant size (mpc version) */
int d_ip2_soft_mpc_tv(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int N, int *nx, int *nu, int *nb, int **idxb, int *ng, int *ns, double **pBAbt, double **pQ, double **Z, double **z, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *double_work_memory)
	{

	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;



	// matrices size
	// work_space_int_size_per_stage = 7
	int idx;
	int nxM = 0;
	int nzM = 0;
	int ngM = 0;

	int pnx[N+1];
	int pnz[N+1];
	int pnb[N+1];
	int png[N+1];
	int pns[N+1];
	int cnx[N+1];
	int cnz[N+1];

	for(jj=0; jj<=N; jj++)
		{
		pnx[jj] = (nx[jj]+bs-1)/bs*bs;
		pnz[jj] = (nu[jj]+nx[jj]+1+bs-1)/bs*bs;
		pnb[jj] = (nb[jj]+bs-1)/bs*bs;
		png[jj] = (ng[jj]+bs-1)/bs*bs;
		pns[jj] = (ns[jj]+bs-1)/bs*bs;
		cnx[jj] = (nx[jj]+ncl-1)/ncl*ncl;
		cnz[jj] = (nu[jj]+nx[jj]+1+ncl-1)/ncl*ncl;
		if(nx[jj]>nxM) nxM = nx[jj];
		if(nu[jj]+nx[jj]+1>nzM) nzM = nu[jj]+nx[jj]+1;
		if(ng[jj]>ngM) ngM = ng[jj];
		}



	// initialize work space
	// work_space_double_size_per_stage = pnz*cnl + 2*pnz + 2*pnx + 14*pnb + 10*png
	// work_space_double_size_const_max = pnz*cnxg + pnz
	double *ptr;
	ptr = double_work_memory; // supposed to be aligned to cache line boundaries

	double *(pL[N+1]);
	double *(dL[N+1]);
	double *(l[N+1]);
	double *work;
	double *(q[N+1]);
	double *(dux[N+1]);
	double *(dpi[N+1]);
	double *(pd[N+1]); // pointer to diagonal of Hessian
	double *(pl[N+1]); // pointer to linear part of Hessian
	double *(bd[N+1]); // backup diagonal of Hessian
	double *(bl[N+1]); // backup linear part of Hessian
	double *diag;
	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);
	double *(t_inv[N+1]);
	double *(Qx[N+1]);
	double *(qx[N+1]);
	double *(qx2[N+1]);
	double *(Pb[N]);
	double *(Zl[N+1]); // inverse of the diagonal of the matrix of the cost funciton of the soft constraint slack variables as updated in the IP
	double *(zl[N+1]); // linear part of the cost funciton of the soft constraint slack variables as updated in the IP

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		ptr += pnz[jj] * ( cnx[jj]+ncl>cnz[jj] ? cnx[jj]+ncl : cnz[jj] ); // pnz*cnl
		}

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

	work = ptr;
	ptr += ((nzM+bs-1)/bs*bs) * ((nxM+ngM+ncl-1)/ncl*ncl); // pnzM*cnxgM

	
	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz[jj];
		}

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx[jj];
		}
	
	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += pnx[jj+1];
		}

	// linear part of cost function
	for(jj=0; jj<=N; jj++)
		{
		q[jj] = ptr;
		ptr += pnz[jj];
		for(ll=0; ll<nu[jj]+nx[jj]; ll++) q[jj][ll] = pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnz[jj]+(nu[jj]+nx[jj])%bs+ll*bs];
		}

	// Hessian backup
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = ptr;
		pl[jj] = ptr + pnb[jj] + pns[jj];
		bd[jj] = ptr + 2*pnb[jj] + 2*pns[jj];
		bl[jj] = ptr + 3*pnb[jj] + 3*pns[jj];
		ptr += 4*pnb[jj] + 4*pns[jj];
		// backup
		for(ll=0; ll<nb[jj]+ns[jj]; ll++)
			{
			idx = idxb[jj][ll];
			bd[jj][ll] = pQ[jj][idx/bs*bs*cnz[jj]+idx%bs+idx*bs];
			bl[jj][ll] = q[jj][idx];
			}
		}

	diag = ptr;
	ptr += (nzM+bs-1)/bs*bs; // pnzM

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb[jj] + 2*png[jj] + 4*pns[jj];
		ptr += 4*pnb[jj] + 4*png[jj] + 8*pns[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb[jj] + 2*png[jj] + 4*pns[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb[jj] + 2*png[jj] + 4*pns[jj];
		}

	for(jj=0; jj<=N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+png[jj];
		qx2[jj] = ptr+2*png[jj];
		ptr += 3*png[jj];
		}

	// updated cost function of soft constraint slack variables
	for(jj=0; jj<=N; jj++)
		{
		Zl[jj] = ptr;
		zl[jj] = ptr + 2*pns[jj];
		ptr += 4*pns[jj];
		}
	


	double temp0, temp1;
	double alpha, mu, mu_aff;
	double mu_scal = 0.0; 
	for(jj=0; jj<=N; jj++) mu_scal += 2*nb[jj] + 2*ng[jj] + 4*ns[jj];
	mu_scal = 1.0 / mu_scal;
	double sigma, sigma_decay, sigma_min;

	// TODO remove
	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	d_init_var_soft_mpc_tv(N, nx, nu, nb, idxb, ng, ns, ux, pi, pDCt, d, t, lam, mu0, warm_start);



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
		d_update_hessian_soft_mpc_tv(N, nx, nu, nb, ng, ns, 0.0, t, t_inv, lam, lamt, dlam, Qx, qx, qx2, bd, bl, pd, pl, d, Z, z, Zl, zl);

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
		d_back_ric_sv_tv(N, nx, nu, pBAbt, pQ, dux, pL, dL, work, diag, 1, Pb, compute_mult, dpi, nb, idxb, pd, pl, ng, pDCt, Qx, qx2);

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
		d_compute_alpha_soft_mpc_tv(N, nx, nu, nb, idxb, ng, ns, &alpha, t, dt, lam, dlam, lamt, dux, pDCt, d, Zl, zl);

		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;
		//printf("\nalpha = %f\n", alpha);



		// compute the affine duality gap
		d_compute_mu_soft_mpc_tv(N, nx, nu, nb, ng, ns, &mu_aff, mu_scal, alpha, lam, dlam, t, dt);

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


		d_update_gradient_soft_mpc_tv(N, nx, nu, nb, ng, ns, sigma*mu, dt, dlam, t_inv, lamt, pl, qx, Zl, zl);

#if 0
for(ii=0; ii<=N; ii++)
	d_print_mat(1, nb[ii], pl[ii], 1);
for(ii=0; ii<=N; ii++)
	d_print_mat(1, ng[ii], qx[ii], 1);
if(*kk==1)
exit(1);
#endif


		// copy b into x
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx[ii+1]; jj++) 
				dux[ii+1][nu[ii+1]+jj] = pBAbt[ii][(nu[ii]+nx[ii])/bs*bs*cnx[ii+1]+(nu[ii]+nx[ii])%bs+bs*jj]; // copy b



		// solve the system
		d_back_ric_trs_tv(N, nx, nu, pBAbt, pL, dL, q, l, dux, work, 0, Pb, compute_mult, dpi, nb, idxb, pl, ng, pDCt, qx);

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
		d_compute_alpha_soft_mpc_tv(N, nx, nu, nb, idxb, ng, ns, &alpha, t, dt, lam, dlam, lamt, dux, pDCt, d, Zl, zl);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// update x, u, lam, t & compute the duality gap mu

		d_update_var_soft_mpc_tv(N, nx, nu, nb, ng, ns, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);

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
		for(ll=0; ll<nb[jj]+ns[jj]; ll++)
			{
			idx = idxb[jj][ll];
			pQ[jj][idx/bs*bs*cnz[jj]+idx%bs+idx*bs] = bd[jj][ll];
			pQ[jj][(nu[jj]+nx[jj])/bs*bs*cnz[jj]+(nu[jj]+nx[jj])%bs+idx*bs] = bl[jj][ll];
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



/* primal-dual predictor-corrector interior-point method, hard and soft box constraints, mpc version */
int d_ip2_soft_mpc(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nh, int ns, double **pBAbt, double **pQ, double **Z, double **z, double **db, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work_memory)
	{

	// number of either hard or soft (box) constraints
	int nb = nh + ns;
	
	int nbu = nu<nb ? nu : nb ;
	int nbx = nb-nu>0 ? nb-nu : 0 ;

	// indeces
	int jj, ll, ii, bs0;

	double **dummy;


	// constants
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int nxu = nx+nu;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnb = bs*((2*nb+bs-1)/bs); // cache aligned number of box constraints
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);

//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	
	

	// initialize work space
	double *ptr;
	ptr = work_memory;

	double *(dux[N+1]);
	double *(dpi[N+1]);
	double *(pL[N+1]);
	double *(pd[N+1]); // pointer to diagonal of Hessian
	double *(pl[N+1]); // pointer to linear part of Hessian
	double *(bd[N+1]); // backup diagonal of Hessian
	double *(bl[N+1]); // backup linear part of Hessian
	double *work;
	double *diag;
	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);
	double *(t_inv[N+1]);
	double *(Pb[N]);
	double *(Zl[N+1]); // inverse of the diagonal of the matrix of the cost funciton of the soft constraint slack variables as updated in the IP
	double *(zl[N+1]); // linear part of the cost funciton of the soft constraint slack variables as updated in the IP

//	ptr += (N+1)*(pnx + pnz*cnl + 12*pnz) + 3*pnz;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += anz;
		}

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += anx;
		}
	
	// Hessian
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = ptr; //pQ[jj];
		pl[jj] = ptr + anz; //pQ[jj] + ((nu+nx)/bs)*bs*cnz + (nu+nx)%bs;
		bd[jj] = ptr + 2*anz;
		bl[jj] = ptr + 3*anz;
		ptr += 4*anz;
		// backup
		for(ll=0; ll<nx+nu; ll++)
			{
			bd[jj][ll] = pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs];
			bl[jj][ll] = pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs];
			}
		}

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		ptr += pnz*cnl;
		}
	
	work = ptr;
//	ptr += 2*anz;
	ptr += pnz*cnx;

	diag = ptr;
	ptr += anz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb;;
		ptr += 4*pnb;
		}
	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb;
		}
	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb;
		}

	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += anx;
		}

	// updated cost function of soft constraint slack variables
	for(jj=0; jj<=N; jj++)
		{
		Zl[jj] = ptr;
		zl[jj] = ptr + pnb;;
		ptr += 2*pnb;
		}



	double temp0, temp1;
	double alpha, mu, mu_aff;
	double mu_scal = 1.0/(N*2*(nb+nbx));
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	d_init_var_soft_mpc(N, nx, nu, nh, ns, ux, pi, db, t, lam, mu0, warm_start);



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	//alpha = 0.0; // needed to compute mu !!!!!
	//d_compute_mu_soft_mpc(N, nx, nu, nb, &mu, mu_scal, alpha, lam, dlam, t, dt);
	mu = mu0;

	// set to zero iteration count
	*kk = 0;	

	// larger than minimum accepted step size
	alpha = 1.0;

	// update hessian in Riccati routine
	const int update_hessian = 1;



	// IP loop		
	while( *kk<k_max && mu>mu_tol && alpha>=alpha_min )
		{
						


		//update cost function matrices and vectors (box constraints)

		// update hessian
		d_update_hessian_soft_mpc(N, nx, nu, nh, ns, cnz, 0.0, t, t_inv, lam, lamt, dlam, bd, bl, pd, pl, db, Z, z, Zl, zl);



		// compute the search direction: factorize and solve the KKT system
		d_back_ric_sv(N, nx, nu, pBAbt, pQ, update_hessian, pd, pl, 1, dux, pL, work, diag, 1, Pb, compute_mult, dpi, 0, 0, 0, dummy, dummy, dummy);



		// compute t_aff & dlam_aff & dt_aff & alpha
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<2*nb; ll++)
				{
				dlam[jj][ll] = 0.0;
				dlam[jj][pnb+ll] = 0.0;
				}

		alpha = 1.0;
		d_compute_alpha_soft_mpc(N, nbu, nu, nh, ns, &alpha, t, dt, lam, dlam, lamt, dux, db, Zl, zl);
		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;



		// compute the affine duality gap
		d_compute_mu_soft_mpc(N, nx, nu, nh, ns, &mu_aff, mu_scal, alpha, lam, dlam, t, dt);
		
		stat[5*(*kk)+2] = mu_aff;



		// compute sigma
		sigma = mu_aff/mu;
		sigma = sigma*sigma*sigma;
		if(sigma<sigma_min)
			sigma = sigma_min;



		// update Jacobian
		d_update_gradient_soft_mpc(N, nx, nu, nh, ns, sigma*mu, dt, dlam, t_inv, lamt, pl, Zl, zl);




		// copy b into x
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++) 
				dux[ii+1][nu+jj] = pBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b



		// solve the system
		d_ric_trs_mpc(nx, nu, N, pBAbt, pL, pl, dux, work, 0, Pb, compute_mult, dpi, 0, 0, 0, dummy, dummy);



		// compute t & dlam & dt & alpha
		alpha = 1.0;
		d_compute_alpha_soft_mpc(N, nx, nu, nh, ns, &alpha, t, dt, lam, dlam, lamt, dux, db, Zl, zl);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// update x, u, lam, t & compute the duality gap mu

		d_update_var_soft_mpc(N, nx, nu, nh, ns, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);

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
		for(ll=0; ll<nx+nu; ll++)
			{
			pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs] = bd[jj][ll];
			pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs] = bl[jj][ll];
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


