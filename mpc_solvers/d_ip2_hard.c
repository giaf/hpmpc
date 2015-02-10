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






/* primal-dual interior-point method, hard constraints, time variant matrices (mpc version) */
int d_ip2_hard_mpc(int *kk, int k_max, double mu0, double mu_tol, double alpha_min, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, int ng, double **pBAbt, double **pQ, double **pDCt, double **d, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work_memory)
	{
	
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
	const int pnb = bs*((nb+bs-1)/bs); // simd aligned number of two-sided box constraints !!!!!!!!!!!!!!!!!!
	const int png = bs*((ng+bs-1)/bs); // simd aligned number of two-sided general constraints !!!!!!!!!!!!!!!!!!
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
//	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box constraints
	//const int anb = nal*((nb+nal-1)/nal); // cache aligned number of two-sided box constraints !!!!!!!!!!!!!!!!!!

//	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	//const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
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
	double *(qx[N+1]);
	double *(Pb[N]);

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
	//ptr += 2*anz;
	ptr += pnz*cnx;

	diag = ptr;
	ptr += anz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnb+2*png;
		ptr += 4*pnb+4*png;
		}
	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnb+2*png;
		}
	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnb+2*png;
		}
	for(jj=0; jj<=N; jj++)
		{
		qx[jj] = ptr;
		ptr += 2*pnb+2*png;
		}

	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += anx;
		}



	double temp0, temp1;
	double alpha, mu, mu_aff;
	double mu_scal = 1.0/(N*2*nb);
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	d_init_var_hard_mpc(N, nx, nu, nb, ng, ux, pi, d, t, lam, mu0, warm_start);




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



	// IP loop		
	while( *kk<k_max && mu>mu_tol && alpha>=alpha_min )
		{
						


		//update cost function matrices and vectors (box constraints)

		d_update_hessian_hard_mpc(N, nx, nu, nb, ng, cnz, 0.0, t, t_inv, lam, lamt, dlam, qx, bd, bl, pd, pl, d);



		// compute the search direction: factorize and solve the KKT system
		d_ric_sv_mpc(nx, nu, N, pBAbt, pQ, update_hessian, pd, pl, dux, pL, work, diag, compute_mult, dpi);



		// compute t_aff & dlam_aff & dt_aff & alpha
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<2*nb; ll++)
				dlam[jj][ll] = 0.0;

		alpha = 1.0;
		d_compute_alpha_hard_mpc(N, nbu, nu, nb, &alpha, t, dt, lam, dlam, lamt, dux, d);
		

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;



		// compute the affine duality gap
		d_compute_mu_hard_mpc(N, nx, nu, nb, &mu_aff, mu_scal, alpha, lam, dlam, t, dt);
		
		stat[5*(*kk)+2] = mu_aff;



		// compute sigma
		sigma = mu_aff/mu;
		sigma = sigma*sigma*sigma;
//		if(sigma<sigma_min)
//			sigma = sigma_min;



		d_update_gradient_hard_mpc(N, nx, nu, nb, sigma*mu, dt, dlam, t_inv, pl);

#if 0
		// first stage
		for(ii=0; ii<2*nbu; ii+=2)
			{
			dlam[0][ii+0] = t_inv[0][ii+0]*(sigma*mu - dlam[0][ii+0]*dt[0][ii+0]); // !!!!!
			dlam[0][ii+1] = t_inv[0][ii+1]*(sigma*mu - dlam[0][ii+1]*dt[0][ii+1]); // !!!!!
			pl[0][ii/2] += dlam[0][ii+1] - dlam[0][ii+0];
			}

		// middle stages
		for(jj=1; jj<N; jj++)
			{
			for(ii=0; ii<2*nb; ii+=2)
				{
				dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma*mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
				dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma*mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
				pl[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
				}
			}

		// last stages
		for(ii=2*nu; ii<2*nb; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma*mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma*mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			pl[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
			}
#endif



		// copy b into x
		for(ii=0; ii<N; ii++)
			for(jj=0; jj<nx; jj++) 
				dux[ii+1][nu+jj] = pBAbt[ii][((nu+nx)/bs)*bs*cnx+(nu+nx)%bs+bs*jj]; // copy b



		// solve the system
		d_ric_trs_mpc(nx, nu, N, pBAbt, pL, pl, dux, work, 1, Pb, compute_mult, dpi);



		// compute t & dlam & dt & alpha
		alpha = 1.0;
		d_compute_alpha_hard_mpc(N, nx, nu, nb, &alpha, t, dt, lam, dlam, lamt, dux, d);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// update x, u, lam, t & compute the duality gap mu

		d_update_var_hard_mpc(N, nx, nu, nb, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);

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




