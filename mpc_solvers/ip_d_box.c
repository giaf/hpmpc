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






/* primal-dual interior-point method, box constraints, time invariant matrices */
void ip_d_box(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **lb, double **ub, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work, int *info)
	{
	
/*	int idx_alpha_max = -1;*/
/*	int idx_alpha_max_ll = -1;*/
/*	int idx_alpha_max_jj = -1;*/
/*	int idx_alpha_max_case = -1;*/
/*	int itemp;*/
/*	double dlam_temp, dt_temp;*/
	
	int nbx = nb - nu;
	if(nbx<0)
		nbx = 0;

	// indeces
	int jj, ll, ii, bs0;

	const int dbs = D_MR; //d_get_mr();
	int dsda = dbs*((nx+nu+1+dbs-nu%dbs+dbs-1)/dbs); // second (orizontal) dimension of matrices
	
	// compound quantities
	int nz = nx+nu+1;
	int nxu = nx+nu;
	
	// initialize work space
	double *ptr;
	ptr = work;

	double *(dux[N+1]);
	double *(dpi[N+1]);
	double *(pL[N+1]);
	double *(pl[N+1]);
	double *pBAbtL;
	double *pLt;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr+jj*dsda;
		}
	ptr += (N+1)*dsda;

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr+jj*dsda;
		}
	ptr += (N+1)*dsda;

	// cost function
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr+jj*dsda*dsda;
		pl[jj] = pL[jj] + nxu%dbs + (nxu/dbs)*dbs*dsda;
		}
	ptr += (N+1)*dsda*dsda;

	// work space
	pBAbtL = ptr;
	ptr += dsda*dsda;

	pLt = ptr;
	ptr += dsda*dsda;
	for(jj=0; jj<dsda*dsda; jj++)
		pLt[jj] = 0.0;
		


	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj] = ptr + jj*3*2*nb + 0*2*nb;
		dt[jj]   = ptr + jj*3*2*nb + 1*2*nb;
		lamt[jj] = ptr + jj*3*2*nb + 2*2*nb;
		}
	ptr += (N+1)*3*2*nb;
	
	double temp0, temp1;
	double alpha, mu;
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
		for(ll=0; ll<( nu<nb ? 2*nu : 2*nb); ll+=2)
			{
			t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
			t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = (ub[0][ll/2] - ub[0][ll/2])/2.0;
					t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
					t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
					}
				else
					{
					t[0][ll+0] = thr0;
					ux[0][ll/2] = lb[0][ll/2] + thr0;
					}
				}
			else if(t[0][ll+1] < thr0)
				{
				t[0][ll+1] = thr0;
				ux[0][ll/2] = ub[0][ll/2] - thr0;
				}
			}
		for(; ll<2*nb; ll++)
			t[0][ll] = 1.0; // this has to be strictly positive !!!
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				t[jj][ll+0] = ux[jj][ll/2] - lb[jj][ll/2];
				t[jj][ll+1] = ub[jj][ll/2] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = (ub[jj][ll/2] - ub[jj][ll/2])/2.0;
						t[jj][ll+0] = ux[jj][ll/2] - lb[jj][ll/2];
						t[jj][ll+1] = ub[jj][ll/2] - ux[jj][ll/2];
						}
					else
						{
						t[jj][ll+0] = thr0;
						ux[jj][ll/2] = lb[jj][ll/2] + thr0;
						}
					}
				else if(t[jj][ll+1] < thr0)
					{
					t[jj][ll+1] = thr0;
					ux[jj][ll/2] = ub[jj][ll/2] - thr0;
					}
				}
			}
		for(ll=0; ll<( nu<nb ? 2*nu : 2*nb); ll++) // this has to be strictly positive !!!
			t[N][ll] = 1;
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				t[N][ll+0] = thr0;
				ux[N][ll/2] = lb[N][ll/2] + thr0;
				}

			t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
			if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = ub[N][ll/2] - thr0;
				}
			}
		}
	else
		{
		double thr0 = 1e-6;
		for(ll=0; ll<( nu<nb ? 2*nu : 2*nb); ll+=2)
			{
			ux[0][ll/2] = 0.0;
			t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
			t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
			if(t[0][ll+0] < thr0 || t[0][ll+1] < thr0)
				{
				ux[0][ll/2] = (ub[0][ll/2] - lb[0][ll/2])/2.0;
				t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
				t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
				}
			}
		for(; ll<2*nb; ll++)
			t[0][ll] = 1.0; // this has to be strictly positive !!!
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				ux[jj][ll/2] = 0.0;
				t[jj][ll+0] = ux[jj][ll/2] - lb[jj][ll/2];
				t[jj][ll+1] = ub[jj][ll/2] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0 || t[jj][ll+1] < thr0)
					{
					ux[jj][ll/2] = (ub[jj][ll/2] - lb[jj][ll/2])/2.0;
					t[jj][ll+0] = ux[jj][ll/2] - lb[jj][ll/2];
					t[jj][ll+1] = ub[jj][ll/2] - ux[jj][ll/2];
					}
				}
			}
		for(ll=0; ll<( nu<nb ? 2*nu : 2*nb); ll++)
			t[N][ll] = 1.0; // this has to be strictly positive !!!
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
			t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
			t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
			if(t[N][ll+0] < thr0 || t[N][ll+1] < thr0)
				{
				ux[N][ll/2] = (ub[N][ll/2] - lb[N][ll/2])/2.0;
				t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
				t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
				}
			}
		}



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
	
	for(jj=0; jj<N; jj++)
		{
		for(ll=0; ll<2*nb; ll++)
			lam[jj][ll] = 1/t[jj][ll];
/*			lam[jj][ll] = thr0/t[jj][ll];*/
		}
	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!
		lam[N][ll] = 1;
	for(ll=2*nu; ll<2*nb; ll++)
		lam[N][ll] = 1/t[jj][ll];
/*		lam[N][ll] = thr0/t[jj][ll];*/



	// initialize pi
	for(jj=0; jj<=N; jj++)
		for(ll=0; ll<nx; ll++)
			dpi[0][ll] = 0.0;



	// initialize dux
	// double precision
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	mu = 0;
	for(jj=0; jj<N; jj++)
		{
		for(ll=0 ; ll<2*nb; ll+=2)
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
		}
	for(ll=2*nu ; ll<2*nb; ll+=2)
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
	mu /= N*2*nb + 2*nbx;
	
	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						


		//update cost function matrices and vectors (box constraints)

		for(jj=0; jj<=N; jj++)
			{

			// copy Q in L
			d_copy_pmat_lo(nz, dbs, pQ[jj], dsda, pL[jj], dsda);
/*				d_copy_pmat(nz, nz, dbs, pQ[jj], dsda, pL[jj], dsda);*/
		
			// box constraints
			for(ii=0; ii<2*nb; ii+=2*dbs)
				{
				bs0 = 2*nb-ii;
				if(2*dbs<bs0) bs0 = 2*dbs;
				for(ll=0; ll<bs0; ll+=2)
					{
					temp0 = 1.0/t[jj][ii+ll+0];
					temp1 = 1.0/t[jj][ii+ll+1];
					lamt[jj][ii+ll+0] = lam[jj][ii+ll+0]*temp0;
					lamt[jj][ii+ll+1] = lam[jj][ii+ll+1]*temp1;
					dlam[jj][ii+ll+0] = temp0*(sigma*mu); // !!!!!
					dlam[jj][ii+ll+1] = temp1*(sigma*mu); // !!!!!
					pL[jj][ll/2+(ii+ll)/2*dbs+ii/2*dsda] += lamt[jj][ii+ll+0] + lamt[jj][ii+ll+1];
					pl[jj][(ii+ll)/2*dbs] += lam[jj][ii+ll+1] - lamt[jj][ii+ll+1]*ub[jj][ii/2+ll/2] + dlam[jj][ii+ll+1] 
					                       - lam[jj][ii+ll+0] - lamt[jj][ii+ll+0]*lb[jj][ii/2+ll/2] - dlam[jj][ii+ll+0];
					}
				}
			}


			// compute the search direction: factorize and solve the KKT system
		dricposv_mpc(nx, nu, N, dsda, pBAbt, pL, dux, pLt, pBAbtL, compute_mult, dpi, info);
		if(*info!=0) return;


		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1;
/*		if(0)*/
/*		if(idx_alpha_max_ll>=0)*/
/*			{*/
/*			if(idx_alpha_max_case==0)*/
/*				{*/
/*				dt_temp =   dux[idx_alpha_max_jj][idx_alpha_max_ll/2] - db[idx_alpha_max_jj][idx_alpha_max_ll+0];*/
/*				dlam_temp = dlam[idx_alpha_max_jj][idx_alpha_max_ll+0] - lamt[idx_alpha_max_jj][idx_alpha_max_ll+0] * dt_temp;*/
/*				if( dlam_temp<0 && -alpha*dlam_temp>lam[idx_alpha_max_jj][idx_alpha_max_ll+0] )*/
/*					{*/
/*					alpha = - lam[idx_alpha_max_jj][idx_alpha_max_ll+0] / dlam_temp;*/
/*					}*/
/*				}*/
/*			else if(idx_alpha_max_case==1)*/
/*				{*/
/*				dt_temp = - dux[idx_alpha_max_jj][idx_alpha_max_ll/2] - db[idx_alpha_max_jj][idx_alpha_max_ll+1];*/
/*				dlam_temp = dlam[idx_alpha_max_jj][idx_alpha_max_ll+1] - lamt[idx_alpha_max_jj][idx_alpha_max_ll+1] * dt_temp;*/
/*				if( dlam_temp<0 && -alpha*dlam_temp>lam[idx_alpha_max_jj][idx_alpha_max_ll+1] )*/
/*					{*/
/*					alpha = - lam[idx_alpha_max_jj][idx_alpha_max_ll+1] / dlam_temp;*/
/*					}*/
/*				}*/
/*			else if(idx_alpha_max_case==2)*/
/*				{*/
/*				dt_temp =   dux[idx_alpha_max_jj][idx_alpha_max_ll/2] - db[idx_alpha_max_jj][idx_alpha_max_ll+0];*/
/*				dlam_temp = dlam[idx_alpha_max_jj][idx_alpha_max_ll+0] - lamt[idx_alpha_max_jj][idx_alpha_max_ll+0] * dt_temp;*/
/*				dt_temp -= t[idx_alpha_max_jj][idx_alpha_max_ll+0];*/
/*				if( dt_temp<0 && -alpha*dt_temp>t[idx_alpha_max_jj][idx_alpha_max_ll+0] )*/
/*					{*/
/*					alpha = - t[idx_alpha_max_jj][idx_alpha_max_ll+0] / dt_temp;*/
/*					}*/
/*				}*/
/*			else if(idx_alpha_max_case==3)*/
/*				{*/
/*				dt_temp = - dux[idx_alpha_max_jj][idx_alpha_max_ll/2] - db[idx_alpha_max_jj][idx_alpha_max_ll+1];*/
/*				dlam_temp = dlam[idx_alpha_max_jj][idx_alpha_max_ll+1] - lamt[idx_alpha_max_jj][idx_alpha_max_ll+1] * dt_temp;*/
/*				dt_temp -= t[idx_alpha_max_jj][idx_alpha_max_ll+1];*/
/*				if( dt_temp<0 && -alpha*dt_temp>t[idx_alpha_max_jj][idx_alpha_max_ll+1] )*/
/*					{*/
/*					alpha = - t[idx_alpha_max_jj][idx_alpha_max_ll+1] / dt_temp;*/
/*					}*/
/*				}*/
/*			}*/
		for(jj=0; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				dt[jj][ll+0] =   dux[jj][ll/2] - lb[jj][ll/2];
				dt[jj][ll+1] = - dux[jj][ll/2] + ub[jj][ll/2];
				dlam[jj][ll+0] -= lamt[jj][ll+0] * dt[jj][ll+0];
				dlam[jj][ll+1] -= lamt[jj][ll+1] * dt[jj][ll+1];
				if( dlam[jj][ll+0]<0 && -alpha*dlam[jj][ll+0]>lam[jj][ll+0] )
					{
					alpha = - lam[jj][ll+0] / dlam[jj][ll+0];
/*					idx_alpha_max_ll = ll;*/
/*					idx_alpha_max_jj = jj;*/
/*					idx_alpha_max_case = 0;*/
					}
				if( dlam[jj][ll+1]<0 && -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )
					{
					alpha = - lam[jj][ll+1] / dlam[jj][ll+1];
/*					idx_alpha_max_ll = ll;*/
/*					idx_alpha_max_jj = jj;*/
/*					idx_alpha_max_case = 1;*/
					}
				dt[jj][ll+0] -= t[jj][ll+0];
				dt[jj][ll+1] -= t[jj][ll+1];
				if( dt[jj][ll+0]<0 && -alpha*dt[jj][ll+0]>t[jj][ll+0] )
					{
					alpha = - t[jj][ll+0] / dt[jj][ll+0];
/*					idx_alpha_max_ll = ll;*/
/*					idx_alpha_max_jj = jj;*/
/*					idx_alpha_max_case = 2;*/
					}
				if( dt[jj][ll+1]<0 && -alpha*dt[jj][ll+1]>t[jj][ll+1] )
					{
					alpha = - t[jj][ll+1] / dt[jj][ll+1];
/*					idx_alpha_max_ll = ll;*/
/*					idx_alpha_max_jj = jj;*/
/*					idx_alpha_max_case = 3;*/
					}
				}
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			dt[N][ll+0] =   dux[N][ll/2] - lb[N][ll/2];
			dt[N][ll+1] = - dux[N][ll/2] + ub[N][ll/2];
			dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];
			dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];
			if( dlam[N][ll+0]<0 && -alpha*dlam[N][ll+0]>lam[N][ll+0] )
				{
				alpha = - lam[N][ll+0] / dlam[N][ll+0];
/*				idx_alpha_max_ll = ll;*/
/*				idx_alpha_max_jj = N;*/
/*				idx_alpha_max_case = 0;*/
				}
			if( dlam[N][ll+1]<0 && -alpha*dlam[N][ll+1]>lam[N][ll+1] )
				{
				alpha = - lam[N][ll+1] / dlam[N][ll+1];
/*				idx_alpha_max_ll = ll;*/
/*				idx_alpha_max_jj = N;*/
/*				idx_alpha_max_case = 1;*/
				}
			dt[N][ll+0] -= t[N][ll+0];
			dt[N][ll+1] -= t[N][ll+1];
			if( dt[N][ll+0]<0 && -alpha*dt[N][ll+0]>t[N][ll+0] )
				{
				alpha = - t[N][ll+0] / dt[N][ll+0];
/*				idx_alpha_max_ll = ll;*/
/*				idx_alpha_max_jj = N;*/
/*				idx_alpha_max_case = 2;*/
				}
			if( dt[N][ll+1]<0 && -alpha*dt[N][ll+1]>t[N][ll+1] )
				{
				alpha = - t[N][ll+1] / dt[N][ll+1];
/*				idx_alpha_max_ll = ll;*/
/*				idx_alpha_max_jj = N;*/
/*				idx_alpha_max_case = 3;*/
				}
			}

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		mu = 0;
		for(jj=0; jj<N; jj++)
			{
			// update inputs
			for(ll=0; ll<nu; ll++)
				ux[jj][ll] += alpha*(dux[jj][ll] - ux[jj][ll]);
			// update states
			for(ll=0; ll<nx; ll++)
				ux[jj][nu+ll] += alpha*(dux[jj][nu+ll] - ux[jj][nu+ll]);
			// update equality constrained multipliers
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] += alpha*(dpi[jj][ll] - pi[jj][ll]);
			// box constraints
			for(ll=0; ll<2*nb; ll+=2)
				{
				lam[jj][ll+0] += alpha*dlam[jj][ll+0];
				lam[jj][ll+1] += alpha*dlam[jj][ll+1];
				t[jj][ll+0] += alpha*dt[jj][ll+0];
				t[jj][ll+1] += alpha*dt[jj][ll+1];
				mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
				}
			}
		// update states
		for(ll=0; ll<nx; ll++)
			ux[N][nu+ll] += alpha*(dux[N][nu+ll] - ux[N][nu+ll]);
		// update equality constrained multipliers
		for(ll=0; ll<nx; ll++)
			pi[N][ll] += alpha*(dpi[N][ll] - pi[N][ll]);
		// box constraints
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			lam[N][ll+0] += alpha*dlam[N][ll+0];
			lam[N][ll+1] += alpha*dlam[N][ll+1];
			t[N][ll+0] += alpha*dt[N][ll+0];
			t[N][ll+1] += alpha*dt[N][ll+1];
			mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
			}
		mu /= N*2*nb + 2*nbx;

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
	


	return;

	} // end of ipsolver


