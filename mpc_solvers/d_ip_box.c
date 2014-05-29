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
void d_ip_box(int *kk, int k_max, double tol, int warm_start, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, double **pQ, double **lb, double **ub, double **ux, int compute_mult, double **pi, double **lam, double **t, double *work_memory)
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
	const int cnl = nx+pad+cnz;
	
	
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
/*				t[jj][ll+0] = 1.0;*/
/*				t[jj][ll+1] = 1.0;*/
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
/*			t[N][ll+0] = 1.0;*/
/*			t[N][ll+1] = 1.0;*/
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

/*printf("\nt = \n");*/
/*for(jj=0; jj<=N; jj++)*/
/*	d_print_mat(1, 2*nb, t[jj], 1);*/

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
	mu = 0;
	for(ll=0 ; ll<2*nbu; ll+=2)
		mu += lam[0][ll+0] * t[0][ll+0] + lam[0][ll+1] * t[0][ll+1];
	for(jj=1; jj<N; jj++)
		for(ll=0 ; ll<2*nb; ll+=2)
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
	for(ll=2*nu ; ll<2*nb; ll+=2)
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
	mu *= mu_scal;

/*printf("\nmu = %f\n", mu);*/
	
	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						

		//update cost function matrices and vectors (box constraints)

		// first stage
	
		// box constraints
		for(ii=0; ii<2*nbu; ii+=2*bs)
			{
			bs0 = 2*nb-ii;
			if(2*bs<bs0) bs0 = 2*bs;
			for(ll=0; ll<bs0; ll+=2)
				{
				temp0 = 1.0/t[0][ii+ll+0];
				temp1 = 1.0/t[0][ii+ll+1];
				lamt[0][ii+ll+0] = lam[0][ii+ll+0]*temp0;
				lamt[0][ii+ll+1] = lam[0][ii+ll+1]*temp1;
				dlam[0][ii+ll+0] = temp0*(sigma*mu); // !!!!!
				dlam[0][ii+ll+1] = temp1*(sigma*mu); // !!!!!
				pd[0][ll/2+(ii+ll)/2*bs+ii/2*cnz] = bd[0][(ii+ll)/2] + lamt[0][ii+ll+0] + lamt[0][ii+ll+1];
				pl[0][(ii+ll)/2*bs] = bl[0][(ii+ll)/2] + lam[0][ii+ll+1] - lamt[0][ii+ll+1]*ub[0][ii/2+ll/2] + dlam[0][ii+ll+1] 
				                                       - lam[0][ii+ll+0] - lamt[0][ii+ll+0]*lb[0][ii/2+ll/2] - dlam[0][ii+ll+0];
				}
			}


		// middle stages
		for(jj=1; jj<N; jj++)
			{

			// box constraints
			for(ii=0; ii<2*nb; ii+=2*bs)
				{
				bs0 = 2*nb-ii;
				if(2*bs<bs0) bs0 = 2*bs;
				for(ll=0; ll<bs0; ll+=2)
					{
					temp0 = 1.0/t[jj][ii+ll+0];
					temp1 = 1.0/t[jj][ii+ll+1];
					lamt[jj][ii+ll+0] = lam[jj][ii+ll+0]*temp0;
					lamt[jj][ii+ll+1] = lam[jj][ii+ll+1]*temp1;
					dlam[jj][ii+ll+0] = temp0*(sigma*mu); // !!!!!
					dlam[jj][ii+ll+1] = temp1*(sigma*mu); // !!!!!
					pd[jj][ll/2+(ii+ll)/2*bs+ii/2*cnz] = bd[jj][(ii+ll)/2] + lamt[jj][ii+ll+0] + lamt[jj][ii+ll+1];
					pl[jj][(ii+ll)/2*bs] = bl[jj][(ii+ll)/2] + lam[jj][ii+ll+1] - lamt[jj][ii+ll+1]*ub[jj][ii/2+ll/2] + dlam[jj][ii+ll+1] 
					                                         - lam[jj][ii+ll+0] - lamt[jj][ii+ll+0]*lb[jj][ii/2+ll/2] - dlam[jj][ii+ll+0];
					}
				}

			}
		// last stage

		// box constraints
		for(ii=0*nu; ii<2*nb; ii+=2*bs)
			{
			bs0 = 2*nb-ii;
			if(2*bs<bs0) bs0 = 2*bs;
			for(ll=0; ll<bs0; ll+=2)
				{
				temp0 = 1.0/t[N][ii+ll+0];
				temp1 = 1.0/t[N][ii+ll+1];
				lamt[N][ii+ll+0] = lam[N][ii+ll+0]*temp0;
				lamt[N][ii+ll+1] = lam[N][ii+ll+1]*temp1;
				dlam[N][ii+ll+0] = temp0*(sigma*mu); // !!!!!
				dlam[N][ii+ll+1] = temp1*(sigma*mu); // !!!!!
				pd[N][ll/2+(ii+ll)/2*bs+ii/2*cnz] = bd[N][(ii+ll)/2] + lamt[N][ii+ll+0] + lamt[N][ii+ll+1];
				pl[N][(ii+ll)/2*bs] = bl[N][(ii+ll)/2] + lam[N][ii+ll+1] - lamt[N][ii+ll+1]*ub[N][ii/2+ll/2] + dlam[N][ii+ll+1] 
				                                       - lam[N][ii+ll+0] - lamt[N][ii+ll+0]*lb[N][ii/2+ll/2] - dlam[N][ii+ll+0];
				}
			}




		// compute the search direction: factorize and solve the KKT system
		dricposv_mpc(nx, nu, N, pBAbt, pQ, dux, pL, work, diag, compute_mult, dpi);
	  //dricposv_mpc(nx, nu, N, hpBAbt, hpQ, hux, hpL, work, diag, COMPUTE_MULT, hpi);




		// compute t_aff & dlam_aff & dt_aff & alpha
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
		stat[5*(*kk)+1] = alpha;
			
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

		stat[5*(*kk)+2] = mu;
		

/*printf("\nmu = %f\n", mu);*/



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


