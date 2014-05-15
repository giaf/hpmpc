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
void d_ip2_box(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **lb, double **ub, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work, int *info)
	{
	
	int nbu = nu<nb ? nu : nb ;

	// indeces
	int jj, ll, ii, bs0;

	const int bs = D_MR; //d_get_mr();
	int sda = bs*((nx+nu+1+bs-nu%bs+bs-1)/bs); // second (orizontal) dimension of matrices
	
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
	double *(pl2[N+1]);
	double *pBAbtL;
	double *pLt;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr+jj*sda;
		}
	ptr += (N+1)*sda;

	// equality constr multipliers
	for(jj=0; jj<=N; jj++)
		{
		dpi[jj] = ptr+jj*sda;
		}
	ptr += (N+1)*sda;

	// cost function
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr+jj*sda*sda;
		pl[jj] = pL[jj] + nxu%bs + (nxu/bs)*bs*sda;
		}
	ptr += (N+1)*sda*sda;
	for(jj=0; jj<=N; jj++)
		{
		pl2[jj] = ptr+jj*sda;
		}
	ptr += (N+1)*sda;

	// work space
	pBAbtL = ptr;
	ptr += sda*sda;

	pLt = ptr;
	ptr += sda*sda;
	for(jj=0; jj<sda*sda; jj++)
		pLt[jj] = 0.0;
		


	double *(dlam[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);
	double *(t_inv[N+1]);

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		dlam[jj]  = ptr + jj*4*2*nb + 0*2*nb;
		dt[jj]    = ptr + jj*4*2*nb + 1*2*nb;
		lamt[jj]  = ptr + jj*4*2*nb + 2*2*nb;
		t_inv[jj] = ptr + jj*4*2*nb + 3*2*nb;
		}
	ptr += (N+1)*4*2*nb;
	
	double temp0, temp1;
	double alpha, mu, mu_aff;
	double mu_scal = 1.0/(N*2*nb);
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	
	// initialize t>0 (slack variable)
	if(warm_start==1)
		{
		double thr0 = 1e-3;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
			t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = (ub[0][ll/2] + ub[0][ll/2])*0.5;
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
						ux[jj][ll/2] = (ub[jj][ll/2] + ub[jj][ll/2])*0.5;
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
		for(ll=0; ll<2*nbu; ll++) // this has to be strictly positive !!!
			t[N][ll] = 1;
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
			t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = (ub[N][ll/2] + ub[N][ll/2])*0.5;
					t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
					t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = lb[N][ll/2] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = ub[N][ll/2] - thr0;
				}
			}
		}
	else
		{
		double thr0 = 1e-3;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			ux[0][ll/2] = 0.0;
			t[0][ll+0] = ux[0][ll/2] - lb[0][ll/2];
			t[0][ll+1] = ub[0][ll/2] - ux[0][ll/2];
			if(t[0][ll+0] < thr0)
				{
				if(t[0][ll+1] < thr0)
					{
					ux[0][ll/2] = (ub[0][ll/2] + ub[0][ll/2])*0.5;
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
				ux[jj][ll/2] = 0.0;
				t[jj][ll+0] = ux[jj][ll/2] - lb[jj][ll/2];
				t[jj][ll+1] = ub[jj][ll/2] - ux[jj][ll/2];
				if(t[jj][ll+0] < thr0)
					{
					if(t[jj][ll+1] < thr0)
						{
						ux[jj][ll/2] = (ub[jj][ll/2] + ub[jj][ll/2])*0.5;
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
		for(ll=0; ll<2*nbu; ll++)
			t[N][ll] = 1.0; // this has to be strictly positive !!!
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
			t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
			t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
			if(t[N][ll+0] < thr0)
				{
				if(t[N][ll+1] < thr0)
					{
					ux[N][ll/2] = (ub[N][ll/2] + ub[N][ll/2])*0.5;
					t[N][ll+0] = ux[N][ll/2] - lb[N][ll/2];
					t[N][ll+1] = ub[N][ll/2] - ux[N][ll/2];
					}
				else
					{
					t[N][ll+0] = thr0;
					ux[N][ll/2] = lb[N][ll/2] + thr0;
					}
				}
			else if(t[N][ll+1] < thr0)
				{
				t[N][ll+1] = thr0;
				ux[N][ll/2] = ub[N][ll/2] - thr0;
				}
			}
		}



	// initialize lambda>0 (multiplier of the inequality constr)
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
			dpi[0][ll] = 0.0;



	// initialize dux
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	mu = 0;
	for(ll=0 ; ll<2*nbu; ll+=2)
		mu += lam[0][ll+0] * t[0][ll+0] + lam[0][ll+1] * t[0][ll+1];
	for(jj=1; jj<N; jj++)
		for(ll=0 ; ll<2*nb; ll+=2)
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
	for(ll=2*nu ; ll<2*nb; ll+=2)
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
	mu *= mu_scal;



	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						


		//update cost function matrices and vectors (box constraints)

		// first stage
		// copy Q in L
		d_copy_pmat_lo(nz, bs, pQ[0], sda, pL[0], sda);
	
		// box constraints
		for(ii=0; ii<2*nbu; ii+=2*bs)
			{
			bs0 = 2*nb-ii;
			if(2*bs<bs0) bs0 = 2*bs;
			for(ll=0; ll<bs0; ll+=2)
				{
				t_inv[0][ii+ll+0] = 1.0/t[0][ii+ll+0];
				t_inv[0][ii+ll+1] = 1.0/t[0][ii+ll+1];
				lamt[0][ii+ll+0] = lam[0][ii+ll+0]*t_inv[0][ii+ll+0];
				lamt[0][ii+ll+1] = lam[0][ii+ll+1]*t_inv[0][ii+ll+1];
				pL[0][ll/2+(ii+ll)/2*bs+ii/2*sda] += lamt[0][ii+ll+0] + lamt[0][ii+ll+1];
				pl[0][(ii+ll)/2*bs] += lam[0][ii+ll+1] - lamt[0][ii+ll+1]*ub[0][ii/2+ll/2]
				                     - lam[0][ii+ll+0] - lamt[0][ii+ll+0]*lb[0][ii/2+ll/2];
				pl2[0][(ii+ll)/2] = pl[0][(ii+ll)/2*bs]; // backup for correction step
				}
			}


		for(jj=1; jj<N; jj++)
			{

			// copy Q in L
			d_copy_pmat_lo(nz, bs, pQ[jj], sda, pL[jj], sda);

			// box constraints
			for(ii=0; ii<2*nb; ii+=2*bs)
				{
				bs0 = 2*nb-ii;
				if(2*bs<bs0) bs0 = 2*bs;
				for(ll=0; ll<bs0; ll+=2)
					{
					t_inv[jj][ii+ll+0] = 1.0/t[jj][ii+ll+0];
					t_inv[jj][ii+ll+1] = 1.0/t[jj][ii+ll+1];
					lamt[jj][ii+ll+0] = lam[jj][ii+ll+0]*t_inv[jj][ii+ll+0];
					lamt[jj][ii+ll+1] = lam[jj][ii+ll+1]*t_inv[jj][ii+ll+1];
					pL[jj][ll/2+(ii+ll)/2*bs+ii/2*sda] += lamt[jj][ii+ll+0] + lamt[jj][ii+ll+1];
					pl[jj][(ii+ll)/2*bs] += lam[jj][ii+ll+1] - lamt[jj][ii+ll+1]*ub[jj][ii/2+ll/2] 
					                      - lam[jj][ii+ll+0] - lamt[jj][ii+ll+0]*lb[jj][ii/2+ll/2];
					pl2[jj][(ii+ll)/2] = pl[jj][(ii+ll)/2*bs]; // backup for correction step
					}
				}

			}
		// last stage
		// copy Q in L
		d_copy_pmat_lo(nz, bs, pQ[N], sda, pL[N], sda);
	
		// box constraints
		for(ii=0*nu; ii<2*nb; ii+=2*bs)
			{
			bs0 = 2*nb-ii;
			if(2*bs<bs0) bs0 = 2*bs;
			for(ll=0; ll<bs0; ll+=2)
				{
				t_inv[N][ii+ll+0] = 1.0/t[N][ii+ll+0];
				t_inv[N][ii+ll+1] = 1.0/t[N][ii+ll+1];
				lamt[N][ii+ll+0] = lam[N][ii+ll+0]*t_inv[N][ii+ll+0];
				lamt[N][ii+ll+1] = lam[N][ii+ll+1]*t_inv[N][ii+ll+1];
				pL[N][ll/2+(ii+ll)/2*bs+ii/2*sda] += lamt[N][ii+ll+0] + lamt[N][ii+ll+1];
				pl[N][(ii+ll)/2*bs] += lam[N][ii+ll+1] - lamt[N][ii+ll+1]*ub[N][ii/2+ll/2] 
				                     - lam[N][ii+ll+0] - lamt[N][ii+ll+0]*lb[N][ii/2+ll/2];
				pl2[N][(ii+ll)/2] = pl[N][(ii+ll)/2*bs]; // backup for correction step
				}
			}



		// compute the search direction: factorize and solve the KKT system
		dricposv_mpc(nx, nu, N, sda, pBAbt, pL, dux, pLt, pBAbtL, compute_mult, dpi, info);
		if(*info!=0) return;



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			dt[0][ll+0] =   dux[0][ll/2] - lb[0][ll/2];
			dt[0][ll+1] = - dux[0][ll/2] + ub[0][ll/2];
			dlam[0][ll+0] = - lamt[0][ll+0] * dt[0][ll+0];
			dlam[0][ll+1] = - lamt[0][ll+1] * dt[0][ll+1];
			if( dlam[0][ll+0]<0 && -alpha*dlam[0][ll+0]>lam[0][ll+0] )
				{
				alpha = - lam[0][ll+0] / dlam[0][ll+0];
				}
			if( dlam[0][ll+1]<0 && -alpha*dlam[0][ll+1]>lam[0][ll+1] )
				{
				alpha = - lam[0][ll+1] / dlam[0][ll+1];
				}
			dt[0][ll+0] -= t[0][ll+0];
			dt[0][ll+1] -= t[0][ll+1];
			if( dt[0][ll+0]<0 && -alpha*dt[0][ll+0]>t[0][ll+0] )
				{
				alpha = - t[0][ll+0] / dt[0][ll+0];
				}
			if( dt[0][ll+1]<0 && -alpha*dt[0][ll+1]>t[0][ll+1] )
				{
				alpha = - t[0][ll+1] / dt[0][ll+1];
				}
			}
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				dt[jj][ll+0] =   dux[jj][ll/2] - lb[jj][ll/2];
				dt[jj][ll+1] = - dux[jj][ll/2] + ub[jj][ll/2];
				dlam[jj][ll+0] = - lamt[jj][ll+0] * dt[jj][ll+0];
				dlam[jj][ll+1] = - lamt[jj][ll+1] * dt[jj][ll+1];
				if( dlam[jj][ll+0]<0 && -alpha*dlam[jj][ll+0]>lam[jj][ll+0] )
					{
					alpha = - lam[jj][ll+0] / dlam[jj][ll+0];
					}
				if( dlam[jj][ll+1]<0 && -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )
					{
					alpha = - lam[jj][ll+1] / dlam[jj][ll+1];
					}
				dt[jj][ll+0] -= t[jj][ll+0];
				dt[jj][ll+1] -= t[jj][ll+1];
				if( dt[jj][ll+0]<0 && -alpha*dt[jj][ll+0]>t[jj][ll+0] )
					{
					alpha = - t[jj][ll+0] / dt[jj][ll+0];
					}
				if( dt[jj][ll+1]<0 && -alpha*dt[jj][ll+1]>t[jj][ll+1] )
					{
					alpha = - t[jj][ll+1] / dt[jj][ll+1];
					}
				}
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			dt[N][ll+0] =   dux[N][ll/2] - lb[N][ll/2];
			dt[N][ll+1] = - dux[N][ll/2] + ub[N][ll/2];
			dlam[N][ll+0] = - lamt[N][ll+0] * dt[N][ll+0];
			dlam[N][ll+1] = - lamt[N][ll+1] * dt[N][ll+1];
			if( dlam[N][ll+0]<0 && -alpha*dlam[N][ll+0]>lam[N][ll+0] )
				{
				alpha = - lam[N][ll+0] / dlam[N][ll+0];
				}
			if( dlam[N][ll+1]<0 && -alpha*dlam[N][ll+1]>lam[N][ll+1] )
				{
				alpha = - lam[N][ll+1] / dlam[N][ll+1];
				}
			dt[N][ll+0] -= t[N][ll+0];
			dt[N][ll+1] -= t[N][ll+1];
			if( dt[N][ll+0]<0 && -alpha*dt[N][ll+0]>t[N][ll+0] )
				{
				alpha = - t[N][ll+0] / dt[N][ll+0];
				}
			if( dt[N][ll+1]<0 && -alpha*dt[N][ll+1]>t[N][ll+1] )
				{
				alpha = - t[N][ll+1] / dt[N][ll+1];
				}
			}

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;



		// compute the affine duality gap
		mu_aff = 0;
		for(ll=0 ; ll<2*nbu; ll+=2)
			mu_aff += (lam[0][ll+0] + alpha*dlam[0][ll+0]) * (t[0][ll+0] + alpha*dt[0][ll+0]) + (lam[0][ll+1] + alpha*dlam[0][ll+1]) * (t[0][ll+1] + alpha*dt[0][ll+1]);
		for(jj=1; jj<N; jj++)
			for(ll=0 ; ll<2*nb; ll+=2)
				mu_aff += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);
		for(ll=2*nu ; ll<2*nb; ll+=2)
			mu_aff += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);
		mu_aff *= mu_scal;

		stat[5*(*kk)+2] = mu_aff;



		// compute sigma
		sigma = mu_aff/mu;
		sigma = sigma*sigma*sigma;
		if(sigma<sigma_min)
			sigma = sigma_min;



		//update the rhs

		// first stage
		for(ii=0; ii<2*nbu; ii+=2)
			{
			dlam[0][ii+0] = t_inv[0][ii+0]*(sigma*mu - dlam[0][ii+0]*dt[0][ii+0]); // !!!!!
			dlam[0][ii+1] = t_inv[0][ii+1]*(sigma*mu - dlam[0][ii+1]*dt[0][ii+1]); // !!!!!
			pl2[0][ii/2] += dlam[0][ii+1] - dlam[0][ii+0];
			}

		// middle stages
		for(jj=1; jj<N; jj++)
			{
			for(ii=0; ii<2*nb; ii+=2)
				{
				dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma*mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
				dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma*mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
				pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
				}
			}

		// last stages
		for(ii=2*nu; ii<2*nb; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma*mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma*mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
			}



		// solve the system
		dricpotrs_mpc(nx, nu, N, sda, pBAbt, pL, pl2, dux, pBAbtL, compute_mult, dpi);



/*		// compute t & dlam & dt & alpha*/
		alpha = 1;
		for(ll=0; ll<2*nbu; ll+=2)
			{
			dt[0][ll+0] =   dux[0][ll/2] - lb[0][ll/2];
			dt[0][ll+1] = - dux[0][ll/2] + ub[0][ll/2];
			dlam[0][ll+0] -= lamt[0][ll+0] * dt[0][ll+0];
			dlam[0][ll+1] -= lamt[0][ll+1] * dt[0][ll+1];
			if( dlam[0][ll+0]<0 && -alpha*dlam[0][ll+0]>lam[0][ll+0] )
				{
				alpha = - lam[0][ll+0] / dlam[0][ll+0];
				}
			if( dlam[0][ll+1]<0 && -alpha*dlam[0][ll+1]>lam[0][ll+1] )
				{
				alpha = - lam[0][ll+1] / dlam[0][ll+1];
				}
			dt[0][ll+0] -= t[0][ll+0];
			dt[0][ll+1] -= t[0][ll+1];
			if( dt[0][ll+0]<0 && -alpha*dt[0][ll+0]>t[0][ll+0] )
				{
				alpha = - t[0][ll+0] / dt[0][ll+0];
				}
			if( dt[0][ll+1]<0 && -alpha*dt[0][ll+1]>t[0][ll+1] )
				{
				alpha = - t[0][ll+1] / dt[0][ll+1];
				}
			}
		for(jj=1; jj<N; jj++)
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
					}
				if( dlam[jj][ll+1]<0 && -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )
					{
					alpha = - lam[jj][ll+1] / dlam[jj][ll+1];
					}
				dt[jj][ll+0] -= t[jj][ll+0];
				dt[jj][ll+1] -= t[jj][ll+1];
				if( dt[jj][ll+0]<0 && -alpha*dt[jj][ll+0]>t[jj][ll+0] )
					{
					alpha = - t[jj][ll+0] / dt[jj][ll+0];
					}
				if( dt[jj][ll+1]<0 && -alpha*dt[jj][ll+1]>t[jj][ll+1] )
					{
					alpha = - t[jj][ll+1] / dt[jj][ll+1];
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
				}
			if( dlam[N][ll+1]<0 && -alpha*dlam[N][ll+1]>lam[N][ll+1] )
				{
				alpha = - lam[N][ll+1] / dlam[N][ll+1];
				}
			dt[N][ll+0] -= t[N][ll+0];
			dt[N][ll+1] -= t[N][ll+1];
			if( dt[N][ll+0]<0 && -alpha*dt[N][ll+0]>t[N][ll+0] )
				{
				alpha = - t[N][ll+0] / dt[N][ll+0];
				}
			if( dt[N][ll+1]<0 && -alpha*dt[N][ll+1]>t[N][ll+1] )
				{
				alpha = - t[N][ll+1] / dt[N][ll+1];
				}
			}

		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+3] = alpha;
			
		alpha *= 0.995;



		// update x, u, lam, t & compute the duality gap mu
		mu = 0;
		// update inputs
		for(ll=0; ll<nu; ll++)
			ux[0][ll] += alpha*(dux[0][ll] - ux[0][ll]);
		// box constraints
		for(ll=0; ll<2*nbu; ll+=2)
			{
			lam[0][ll+0] += alpha*dlam[0][ll+0];
			lam[0][ll+1] += alpha*dlam[0][ll+1];
			t[0][ll+0] += alpha*dt[0][ll+0];
			t[0][ll+1] += alpha*dt[0][ll+1];
			mu += lam[0][ll+0] * t[0][ll+0] + lam[0][ll+1] * t[0][ll+1];
			}
		for(jj=1; jj<N; jj++)
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
		mu *= mu_scal;

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
	


	return;

	} // end of ipsolver


