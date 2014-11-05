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

#include <math.h>

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"
#include "../include/mpc_aux.h"






/* primal-dual interior-point method, box constraints, time invariant matrices (mpc version) */
void s_ip_box_mpc(int *kk, int k_max, float tol, int warm_start, float *sigma_par, float *stat, int nx, int nu, int N, int nb, float **pBAbt, float **pQ, float **db, float **ux, int compute_mult, float **pi, float **lam, float **t, float *work_memory)
	{

/*printf("\ncazzo\n");*/

/*	int nbx = nb - nu;*/
/*	if(nbx<0)*/
/*		nbx = 0;*/
	int nbu = nu<nb ? nu : nb ;

	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int nxu = nx+nu;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box constraints

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	
	
	// initialize work space
	float *ptr;
	ptr = work_memory;

	float *(dux[N+1]);
	float *(dpi[N+1]);
	float *(pL[N+1]);
	float *(pd[N+1]); // pointer to diagonal of Hessian
	float *(pl[N+1]); // pointer to linear part of Hessian
	float *(pl2[N+1]); // pointer to linear part of Hessian (backup)
	float *(bd[N+1]); // backup diagonal of Hessian
	float *(bl[N+1]); // backup linear part of Hessian
	float *work;
	float *diag;
	float *(dlam[N+1]);
	float *(dt[N+1]);
	float *(lamt[N+1]);
	float *(t_inv[N+1]);

//	ptr += (N+1)*(pnz + pnx + pnz*cnl + 8*pnz) + 3*pnz;

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
		ptr += pnx;
		}
	
	// Hessian
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = pQ[jj];
		pl[jj] = pQ[jj] + ((nu+nx)/bs)*bs*cnz + (nu+nx)%bs;
		bd[jj] = ptr;
		bl[jj] = ptr + anz;
		ptr += 2*anz;
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
	
	for(jj=0; jj<=N; jj++)
		{
		pl2[jj] = ptr;
		ptr += anz;
		}

	work = ptr;
	ptr += 2*anz;

	diag = ptr;
	ptr += anz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + anb;;
		ptr += 2*anb;
		}
	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += anb;
		}
	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += anb;
		}
	
	float temp0, temp1;
	float alpha, mu;
	float mu_scal = 1.0/(N*2*nb);
	float sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & pi & t>0 (slack variable)
	s_init_ux_pi_t_box_mpc(N, nx, nu, nbu, nb, ux, pi, db, t, warm_start);



	// initialize lambda>0 (multiplier of the inequality constraint)
	s_init_lam_mpc(N, nu, nbu, nb, t, lam);



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	// float precision
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	alpha = 0;
	s_compute_mu_mpc(N, nbu, nu, nb, &mu, mu_scal, alpha, lam, dlam, t, dt);

	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						
//printf("\nk = %d\n", *kk);						

		//update cost function matrices and vectors (box constraints)

		// box constraints

		s_update_hessian_box_mpc(N, nbu, (nu/bs)*bs, nb, cnz, sigma*mu, t, t_inv, lam, lamt, dlam, bd, bl, pd, pl, pl2, db);



		// compute the search direction: factorize and solve the KKT system
		s_ric_sv_mpc(nx, nu, N, pBAbt, pQ, dux, pL, work, diag, compute_mult, dpi);



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		s_compute_alpha_box_mpc(N, 2*nbu, 2*nu, 2*nb, &alpha, t, dt, lam, dlam, lamt, dux, db);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		s_update_var_mpc(nx, nu, N, nb, nbu, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);
		
		stat[5*(*kk)+2] = mu;
		

#if 0
printf("\npi = \n");
s_print_mat(1, nx, pi[0], 1);
s_print_mat(1, nx, pi[1], 1);
s_print_mat(1, nx, pi[2], 1);
s_print_mat(1, nx, pi[3], 1);
s_print_mat(1, nx, pi[4], 1);
s_print_mat(1, nx, pi[5], 1);
s_print_mat(1, nx, pi[6], 1);
s_print_mat(1, nx, pi[7], 1);
s_print_mat(1, nx, pi[8], 1);
s_print_mat(1, nx, pi[9], 1);
s_print_mat(1, nx, pi[N], 1);
#endif

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

	return;

	} // end of ipsolver



/* primal-dual interior-point method, box constraints, time invariant matrices (mpc version) */
void s_ip_box_mhe(int *kk, int k_max, float tol, int warm_start, float *sigma_par, float *stat, int nx, int nu, int N, int nb, float **pBAbt, float **pQ, float **db, float **ux, int compute_mult, float **pi, float **lam, float **t, float *work_memory)
	{

/*printf("\ncazzo\n");*/

/*	int nbx = nb - nu;*/
/*	if(nbx<0)*/
/*		nbx = 0;*/
	int nbu = nu<nb ? nu : nb ;

	// indeces
	int jj, ll, ii, bs0;

	// constants
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int nz = nx+nu+1;
	const int nxu = nx+nu;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box constraints

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	
	
	// initialize work space
	float *ptr;
	ptr = work_memory;

	float *(dux[N+1]);
	float *(dpi[N+1]);
	float *(pL[N+1]);
	float *(pd[N+1]); // pointer to diagonal of Hessian
	float *(pl[N+1]); // pointer to linear part of Hessian
	float *(pl2[N+1]); // pointer to linear part of Hessian (backup)
	float *(bd[N+1]); // backup diagonal of Hessian
	float *(bl[N+1]); // backup linear part of Hessian
	float *work;
	float *diag;
	float *(dlam[N+1]);
	float *(dt[N+1]);
	float *(lamt[N+1]);
	float *(t_inv[N+1]);

//	ptr += (N+1)*(pnz + pnx + pnz*cnl + 8*pnz) + 3*pnz;

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
		ptr += pnx;
		}
	
	// Hessian
	for(jj=0; jj<=N; jj++)
		{
		pd[jj] = pQ[jj];
		pl[jj] = pQ[jj] + ((nu+nx)/bs)*bs*cnz + (nu+nx)%bs;
		bd[jj] = ptr;
		bl[jj] = ptr + anz;
		ptr += 2*anz;
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
	
	for(jj=0; jj<=N; jj++)
		{
		pl2[jj] = ptr;
		ptr += anz;
		}

	work = ptr;
	ptr += 2*anz;

	diag = ptr;
	ptr += anz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + anb;;
		ptr += 2*anb;
		}
	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += anb;
		}
	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += anb;
		}
	
	float temp0, temp1;
	float alpha, mu;
	float mu_scal = 1.0/(N*2*nb);
	float sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	


	// initialize ux & t>0 (slack variable)
	s_init_ux_pi_t_box_mhe(N, nx, nu, nbu, nb, ux, pi, db, t, warm_start);



	// initialize lambda>0 (multiplier of the inequality constraint)
	s_init_lam_mhe(N, nu, nbu, nb, t, lam);



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	// float precision
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	alpha = 0;
	s_compute_mu_mhe(N, nbu, nu, nb, &mu, mu_scal, alpha, lam, dlam, t, dt);

	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						

		//update cost function matrices and vectors (box constraints)

		// box constraints

		s_update_hessian_box_mhe(N, nbu, (nu/bs)*bs, nb, cnz, sigma*mu, t, t_inv, lam, lamt, dlam, bd, bl, pd, pl, pl2, db);



		// compute the search direction: factorize and solve the KKT system
		s_ric_sv_mhe(nx, nu, N, pBAbt, pQ, dux, pL, work, diag, compute_mult, dpi);



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		s_compute_alpha_box_mhe(N, 2*nbu, 2*nu, 2*nb, &alpha, t, dt, lam, dlam, lamt, dux, db);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		s_update_var_mhe(nx, nu, N, nb, nbu, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);
		
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

	return;

	} // end of ipsolver


