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






/* primal-dual interior-point method, hard constraints, time invariant matrices (mpc version) */
int d_ip_hard_mpc(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, int ng, int ngN, double **pBAbt, double **pQ, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work_memory)
	{

/*printf("\ncazzo\n");*/

/*	int nbx = nb - nu;*/
/*	if(nbx<0)*/
/*		nbx = 0;*/
	int nbu = nu<nb ? nu : nb ;

	// indeces
	int jj, ll, ii, bs0;


	// constants
	const int bs = D_MR; //d_get_mr();
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int nxu = nx+nu;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int pnb = bs*((nb+bs-1)/bs); // cache aligned number of box constraints
	const int png = bs*((ng+bs-1)/bs); // cache aligned number of general constraints
	const int pngN = bs*((ngN+bs-1)/bs); // cache aligned number of general constraints
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int cngN = ncl*((ngN+ncl-1)/ncl);
	const int cnxg= ncl*((ng+nx+ncl-1)/ncl);
	//const int anb = nal*((nb+nal-1)/nal); // cache aligned number of box constraints

//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
//	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	const int cnl = cnz<cnx+ncl ? cnx+ncl : cnz;
	
	
	// initialize work space
	double *ptr;
	ptr = work_memory;

	double *(dux[N+1]);
	double *(dpi[N+1]);
	double *(dL[N+1]);
	double *(pL[N+1]);
	double *(pd[N+1]); // pointer to diagonal of Hessian
	double *(pl[N+1]); // pointer to linear part of Hessian
	double *(bd[N+1]); // backup diagonal of Hessian
	double *(bl[N+1]); // backup linear part of Hessian
	double *work0;
	double *work1;
	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);
	double *(t_inv[N+1]);
	double *(qx[N+1]);
	double *(Qx[N+1]);

//	ptr += (N+1)*(pnz + pnx + pnz*cnl + 8*pnz) + 3*pnz;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr;
		ptr += pnz;
		}

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr;
		ptr += pnx;
		}
	
	// Hessian
	for(jj=0; jj<=N; jj++)
		{
		// TODO
		pd[jj] = ptr;
		pl[jj] = ptr + pnz;
		bd[jj] = ptr + 2*pnz;
		bl[jj] = ptr + 3*pnz;
		ptr += 4*pnz;
		// backup of diagonal of Hessian and Jacobian
		for(ll=0; ll<nx+nu; ll++)
			{
			bd[jj][ll] = pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs];
			bl[jj][ll] = pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs];
			}
		}
//	d_print_mat(nx+nu, 1, bd[1], 1);
//	d_print_mat(nx+nu, 1, bl[1], 1);
//	exit(1);

	// work space
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		dL[jj] = ptr + pnz*cnl;
		ptr += pnz*cnl + pnz;
		}
	
	work0 = ptr;
	if(cngN<=cnxg)
		ptr += pnz*cnxg;
	else
		ptr += pnz*cngN;

	work1 = ptr;
	ptr += pnz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb+2*png;
		ptr += 4*pnb+4*png;
		}
	dlam[N] = ptr;
	dt[N]   = ptr + 2*pnb+2*pngN;
	ptr += 4*pnb+4*pngN;

	for(jj=0; jj<N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb+2*png;
		}
	lamt[N] = ptr;
	ptr += 2*pnb+2*pngN;

	for(jj=0; jj<N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb+2*png;
		}
	t_inv[N] = ptr;
	ptr += 2*pnb+2*pngN;

	for(jj=0; jj<N; jj++)
		{
		Qx[jj] = ptr;
		qx[jj] = ptr+png;
		ptr += 2*pnb+2*png;
		}
	Qx[N] = ptr;
	qx[N] = ptr+png;
	ptr += 2*pnb+2*pngN;

/*dlam[0][0] = 1;*/
/*printf("\ncazzo %f\n", dlam[0][0]);*/

	double temp0, temp1;
	double alpha, mu;
	double mu_scal = 1.0/((N-1)*2*(nb+ng)+2*(nb+ngN));
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	d_init_var_hard_mpc(N, nx, nu, nb, ng, ngN, ux, pi, pDCt, d, t, lam, mu0, warm_start);



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



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

	double **dummy;



	// IP loop		
	while( *kk<k_max && mu>mu_tol && alpha>=alpha_min )
		{
						


		//update cost function matrices and vectors (box constraints)
		d_update_hessian_hard_mpc(N, nx, nu, nb, ng, ngN, cnz, sigma*mu, t, t_inv, lam, lamt, dlam, Qx, qx, bd, bl, pd, pl, d);



		// compute the search direction: factorize and solve the KKT system
#if defined(FAST_RSQRT)
		if(mu>1e-2)
			fast_rsqrt = 2;
		else
			{
			if(mu>1e-4)
				fast_rsqrt = 1;
			else
				fast_rsqrt = 0;
			}
#else
		fast_rsqrt = 0;
#endif
		//printf("\n%d %f\n", fast_rsqrt, mu);
		//d_back_ric_sv(N, nx, nu, pBAbt, pQ, update_hessian, pd, pl, 1, dux, pL, work, diag, 0, dummy, compute_mult, dpi, nb, ng, ngN, pDCt, Qx, qx);
		d_back_ric_sv_new(N, nx, nu, pBAbt, pQ, update_hessian, pd, pl, 1, dux, pL, dL, work0, work1, 0, dummy, compute_mult, dpi, nb, ng, ngN, pDCt, Qx, qx);



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		d_compute_alpha_hard_mpc(N, nx, nu, nb, ng, ngN, &alpha, t, dt, lam, dlam, lamt, dux, pDCt, d);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		d_update_var_hard_mpc(N, nx, nu, nb, ng, ngN, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);
		
		stat[5*(*kk)+2] = mu;
		


		// update sigma
		sigma *= sigma_decay;
		if(sigma<sigma_min)
			sigma = sigma_min;
		
		if(alpha<0.3)
			sigma = sigma_par[0];



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

/*printf("\nfinal iteration %d, mu %f\n", *kk, mu);*/



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




