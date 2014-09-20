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
void d_admm_box_mpc(int *kk, int k_max, double tol_p, double tol_d, int warm_start, int compute_fact, double rho, double alpha, double *stat, int nx, int nu, int N, double **pBAbt, double **pQ, double **lb, double **ub, double **ux_u, double **ux_v, double **ux_w, int compute_mult, double **pi, double *work_memory)
	{

/*printf("\ncazzo\n");*/

/*	int nbx = nb - nu;*/
/*	if(nbx<0)*/
/*		nbx = 0;*/
/*	int nbu = nu<nb ? nu : nb ;*/

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
	const int cnz = ncl*((nz+ncl-1)/ncl);
	const int cnx = ncl*((nx+ncl-1)/ncl);
	const int anz = nal*((nz+nal-1)/nal);
	const int anx = nal*((nx+nal-1)/nal);
/*	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box constraints*/

	const int pad = (ncl-nx%ncl)%ncl; // packing between BAbtL & P
	const int cnl = cnz<cnx+ncl ? nx+pad+cnx+ncl : nx+pad+cnz;
	

	/* array or pointers */
	double *(ux_r[N+1]);
/*	double *(ux_v[N+1]);*/
/*	double *(ux_w[N+1]);*/
	double *(pL[N+1]);
	double *(pl[N+1]);
	double *(bd[N+1]);
	double *(bl[N+1]);
	double *(bb[N]);
	double *work1;
	double *diag;
	double *(Pb[N]);
	
	double *ptr = work_memory;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		ux_r[jj] = ptr;
		ptr += anz;
		}

/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		ux_v[jj] = ptr;*/
/*		ptr += anz;*/
/*		}*/

/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		ux_w[jj] = ptr;*/
/*		ptr += anz;*/
/*		}*/

	// work space (matrices)
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr;
		ptr += pnz*cnl;
		}

	// work space (vectors)
	for(jj=0; jj<=N; jj++)
		{
		pl[jj] = ptr;
		ptr += anz;
		}

	// work space
	work1 = ptr;
	ptr += 2*anz;

	diag = ptr;
	ptr += anz;

	// backup Hessian space
	for(jj=0; jj<=N; jj++)
		{
		bd[jj] = ptr;
		bl[jj] = ptr + anz;
		ptr += 2*anz;
		}
	
	// backup b
	for(jj=0; jj<N; jj++)
		{
		bb[jj] = ptr;
		ptr += anx;
		for(ll=0; ll<nx; ll++)
			{
			bb[jj][ll] = pBAbt[jj][((nx+nu)/bs)*bs*cnx+(nx+nu)%bs+ll*bs];
			}
		}

	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += anx;
		}

	double temp, v_temp, norm_p=1e3, norm_d=1e3;

	
	
	// initialize u and x (cold start)
	if(warm_start==0)
		{
		for(ll=0; ll<nu; ll++)
			{
			ux_u[0][ll] = 0.0;
			}
/*		for(ll=0; ll<nx; ll++)*/
/*			{*/
/*			ux_u[jj][nu+ll] = ux[jj][nu+ll];*/
/*			}*/
		for(jj=1; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				ux_u[jj][ll] = 0.0;
				}
			}
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				ux_v[jj][ll] = 0.0;
				}
			}
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				ux_w[jj][ll] = 0.0;
				}
			}
		}
	



	// first iteration (initial factorization)
	
	// reset iteration counter
	*kk = 0; 
	


	if(compute_fact==1) // factorize hessina in the first iteration
		{
		// dynamic
	
		// backup Hessian & add rho to diagonal
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				bd[jj][ll] = pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs];
				pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs] = bd[jj][ll] + rho;
				bl[jj][ll] = pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs];
				pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs] = bl[jj][ll] + rho*(ux_w[jj][ll] - ux_v[jj][ll]);
				}
			}
	
		// initial factorization
		d_ric_sv_mpc(nx, nu, N, pBAbt, pQ, ux_u, pL, work1, diag, compute_mult, pi);

		// constraints
		norm_p = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
/*				v_temp = - ( - ux_w[jj][ll] - ux_u[jj][ll] );*/
				ux_r[jj][ll] = alpha*ux_u[jj][ll] + (1.0-alpha)*ux_v[jj][ll]; // relaxation
				v_temp = - ( - ux_w[jj][ll] - ux_r[jj][ll] );
				v_temp = fmax(v_temp, lb[jj][ll]);
				v_temp = fmin(v_temp, ub[jj][ll]);
				temp = v_temp - ux_v[jj][ll];
				norm_p += temp*temp;
				ux_v[jj][ll] = v_temp;
				}
			}
		norm_p = sqrt(norm_p);
		stat[0+5*kk[0]] = norm_p;
	
		// integral of error
		norm_d = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
/*				temp = ux_u[jj][ll] - ux_v[jj][ll];*/
				temp = ux_r[jj][ll] - ux_v[jj][ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][ll] += temp;
				}
			}
		norm_d = rho*sqrt(norm_d);
		stat[1+5*kk[0]] = norm_d;

		// increment loop index
		(*kk)++;
		
		} // end of factorize hessian
	else
		{
		// backup Hessian
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				bl[jj][ll] = pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs];
				}
			}
		}



	// ADMM loop		
	int compute_Pb = compute_fact;
	while( (*kk<k_max && (norm_p>tol_p || norm_d>tol_d) ) || compute_Pb )
		{

		// dynamic

		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				pl[jj][ll] = bl[jj][ll] + rho*(ux_w[jj][ll] - ux_v[jj][ll]);
				}
			}

		// initialize x with b
		for(jj=0; jj<N; jj++)
			{
			for(ll=0; ll<nx; ll++)
				{
				ux_u[jj+1][nu+ll] = bb[jj][ll];
				}
			}

		// Riccati solver		
		d_ric_trs_mpc(nx, nu, N, pBAbt, pL, pl, ux_u, work1, compute_Pb, Pb, compute_mult, pi);
		compute_Pb = 0;

		// constraints
		norm_p = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
/*				v_temp = - ( - ux_w[jj][ll] - ux_u[jj][ll] );*/
				ux_r[jj][ll] = alpha*ux_u[jj][ll] + (1.0-alpha)*ux_v[jj][ll]; // relaxation
				v_temp = - ( - ux_w[jj][ll] - ux_r[jj][ll] );
				v_temp = fmax(v_temp, lb[jj][ll]);
				v_temp = fmin(v_temp, ub[jj][ll]);
				temp = v_temp - ux_v[jj][ll];
				norm_p += temp*temp;
				ux_v[jj][ll] = v_temp;
				}
			}
		norm_p = sqrt(norm_p);
		stat[0+5*kk[0]] = norm_p;
	
		// integral of error
		norm_d = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
/*				temp = ux_u[jj][ll] - ux_v[jj][ll];*/
				temp = ux_r[jj][ll] - ux_v[jj][ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][ll] += temp;
				}
			}
		norm_d = rho*sqrt(norm_d);
		stat[1+5*kk[0]] = norm_d;



		// increment loop index
		(*kk)++;



		} // end of ADMM loop



	// restore Hessian
	if(compute_fact==1)
		{
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx+nu; ll++)
				{
				pQ[jj][(ll/bs)*bs*cnz+ll%bs+ll*bs] = bd[jj][ll];
				pQ[jj][((nx+nu)/bs)*bs*cnz+(nx+nu)%bs+ll*bs] = bl[jj][ll];
				}
			}
		}

/*printf("\nfinal iteration %d, mu %f\n", *kk, mu);*/

	return;

	}
