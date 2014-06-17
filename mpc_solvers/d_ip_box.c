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






/* primal-dual interior-point method, box constraints, time invariant matrices */
void d_ip_box(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **db, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work_memory)
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

	const int nz = nx+nu+1;
	const int nxu = nx+nu;
	const int pnz = bs*((nz+bs-1)/bs);
	const int pnx = bs*((nx+bs-1)/bs);
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	
	
	// initialize work space
	double *ptr;
	ptr = work_memory;

	double *(dux[N+1]);
	double *(dpi[N+1]);
	double *(pL[N+1]);
	double *(pd[N+1]); // pointer to diagonal of Hessian
	double *(pl[N+1]); // pointer to linear part of Hessian
	double *(pl2[N+1]); // pointer to linear part of Hessian (backup)
	double *(bd[N+1]); // backup diagonal of Hessian
	double *(bl[N+1]); // backup linear part of Hessian
	double *work;
	double *diag;
	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);
	double *(t_inv[N+1]);

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
		pd[jj] = pQ[jj];
		pl[jj] = pQ[jj] + ((nu+nx)/bs)*bs*cnz + (nu+nx)%bs;
		bd[jj] = ptr;
		bl[jj] = ptr + pnz;
		ptr += 2*pnz;
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
		ptr += pnz;
		}

	work = ptr;
	ptr += 2*pnz;

	diag = ptr;
	ptr += pnz;

	// slack variables, Lagrangian multipliers for inequality constraints and work space (assume # box constraints <= 2*(nx+nu) < 2*pnz)
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr;
		dt[jj]   = ptr + 2*pnz;;
		ptr += 4*pnz;
		}
	for(jj=0; jj<=N; jj++)
		{
		lamt[jj] = ptr;
		ptr += 2*pnz;
		}
	for(jj=0; jj<=N; jj++)
		{
		t_inv[jj] = ptr;
		ptr += 2*pnz;
		}
	
	double temp0, temp1;
	double alpha, mu;
	double mu_scal = 1.0/(N*2*nb);
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	
	// initialize t>0 (slack variable)
/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		for(ll=0; ll<2*nb; ll++)*/
/*			t[jj][ll] = 1;*/
/*		}*/
/*	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!*/
/*		t[N][ll] = 1;*/
/*	for(ll=2*nu; ll<2*nb; ll++)*/
/*		t[N][ll] = 1;*/

	if(warm_start==1)
		{
		double thr0 = 1e-3;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
			t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = ( - db[0][ll+1] + db[0][ll+0])*0.5;
					t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
					t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
					}
				else
					{
					t[0][ll+0] = thr0;
					ux[0][ll/2] = db[0][ll+0] + thr0;
					}
				}
			else if(t[0][ll+1] < thr0)
				{
				t[0][ll+1] = thr0;
				ux[0][ll/2] = - db[0][ll+1] - thr0;
				}
			}
		for(; ll<2*nb; ll++)
			t[0][ll] = 1.0; // this has to be strictly positive !!!
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				t[jj][ll+0] = ux[jj][ll/2] - db[jj][ll+0];
				t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = ( - db[jj][ll+1] + db[jj][ll+0])*0.5;
						t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
						t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
						}
					else
						{
						t[jj][ll+0] = thr0;
						ux[jj][ll/2] = db[jj][ll+0] + thr0;
						}
					}
				else if(t[jj][ll+1] < thr0)
					{
					t[jj][ll+1] = thr0;
					ux[jj][ll/2] = - db[jj][ll+1] - thr0;
					}
				}
			}
		for(ll=0; ll<2*nbu; ll++) // this has to be strictly positive !!!
			t[N][ll] = 1;
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
			t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = ( - db[N][ll+1] + db[N][ll+0])*0.5;
					t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
					t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = db[N][ll+0] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = - db[N][ll+1] - thr0;
				}
			}

/*printf("\nt = \n");*/
/*for(jj=0; jj<=N; jj++)*/
/*	d_print_mat(1, 2*nb, t[jj], 1);*/

		}
	else
		{
		double thr0 = 1e-3;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			ux[0][ll/2] = 0.0;
/*			t[0][ll+0] = 1.0;*/
/*			t[0][ll+1] = 1.0;*/
			t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
			t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = ( - db[0][ll+1] + db[0][ll+0])*0.5;
					t[0][ll+0] =   ux[0][ll/2] - db[0][ll+0];
					t[0][ll+1] = - db[0][ll+1] - ux[0][ll/2];
					}
				else
					{
					t[0][ll+0] = thr0;
					ux[0][ll/2] = db[0][ll+0] + thr0;
					}
				}
			else if(t[0][ll+1] < thr0)
				{
				t[0][ll+1] = thr0;
				ux[0][ll/2] = - db[0][ll+1] - thr0;
				}
			}
		for(; ll<2*nb; ll++)
			t[0][ll] = 1.0; // this has to be strictly positive !!!
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				ux[jj][ll/2] = 0.0;
/*				t[jj][ll+0] = 1.0;*/
/*				t[jj][ll+1] = 1.0;*/
				t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
				t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = ( - db[jj][ll+1] + db[jj][ll+0])*0.5;
						t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
						t[jj][ll+1] = - db[jj][ll+1] - ux[jj][ll/2];
						}
					else
						{
						t[jj][ll+0] = thr0;
						ux[jj][ll/2] = db[jj][ll+0] + thr0;
						}
					}
				else if(t[jj][ll+1] < thr0)
					{
					t[jj][ll+1] = thr0;
					ux[jj][ll/2] = - db[jj][ll+1] - thr0;
					}
				}
			}
		for(ll=0; ll<2*nbu; ll++)
			t[N][ll] = 1.0; // this has to be strictly positive !!!
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
/*			t[N][ll+0] = 1.0;*/
/*			t[N][ll+1] = 1.0;*/
			t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
			t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = ( - db[N][ll+1] + db[N][ll+0])*0.5;
					t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
					t[N][ll+1] = - db[N][ll+1] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = db[N][ll+0] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = - db[N][ll+1] - thr0;
				}
			}

/*printf("\nt = \n");*/
/*for(jj=0; jj<=N; jj++)*/
/*	d_print_mat(1, 2*nb, t[jj], 1);*/

		}



	// TODO approximate reciprocal
	// initialize lambda>0 (multiplier of the inequality constr)
/*	for(jj=0; jj<N; jj++)*/
/*		{*/
/*		for(ll=0; ll<2*nb; ll++)*/
/*			lam[jj][ll] = 1;*/
/*		}*/
/*	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!*/
/*		lam[N][ll] = 1;*/
/*	for(ll=2*nu; ll<2*nb; ll++)*/
/*		lam[N][ll] = 1;*/
	
	for(ll=0; ll<2*nbu; ll++)
		lam[0][ll] = 1/t[0][ll];
	for(; ll<2*nb; ll++)
		lam[0][ll] = 1.0; // this has to be strictly positive !!!
	for(jj=1; jj<N; jj++)
		{
		for(ll=0; ll<2*nb; ll++)
			lam[jj][ll] = 1/t[jj][ll];
/*			lam[jj][ll] = thr0/t[jj][ll];*/
		}
	for(ll=0; ll<2*nu; ll++)
		lam[N][ll] = 1.0; // this has to be strictly positive !!!
	for(ll=2*nu; ll<2*nb; ll++)
		lam[N][ll] = 1/t[jj][ll];
/*		lam[N][ll] = thr0/t[jj][ll];*/



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[jj][ll] = 0.0;



	// initialize dux
	// double precision
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	d_compute_mu_mpc(N, nbu, nu, nb, &mu, mu_scal, alpha, lam, dlam, t, dt);

	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						

		//update cost function matrices and vectors (box constraints)

		// box constraints

		d_update_hessian_box_mpc(N, nbu, (nu/bs)*bs, nb, cnz, sigma*mu, t, t_inv, lam, lamt, dlam, bd, bl, pd, pl, pl2, db);



		// compute the search direction: factorize and solve the KKT system
		d_ric_sv_mpc(nx, nu, N, pBAbt, pQ, dux, pL, work, diag, compute_mult, dpi);



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1.0;
		d_compute_alpha_box_mpc(N, 2*nbu, 2*nu, 2*nb, &alpha, t, dt, lam, dlam, lamt, dux, db);

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		d_update_var_mpc(nx, nu, N, nb, nbu, &mu, mu_scal, alpha, ux, dux, t, dt, lam, dlam, pi, dpi);
		
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


