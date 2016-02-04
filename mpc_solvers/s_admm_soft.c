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






/* primal-dual interior-point method, box constraints, time invariant matrices (mpc version) */
void s_admm_soft_mpc(int *kk, int k_max, float tol_p, float tol_d, int warm_start, int compute_fact, float rho, float alpha, float *stat, int nx, int nu, int N, float **pBAbt, float **pQ, float **Z, float **z, float **lb, float **ub, float **ux_u, float **ux_v, float **ux_w, float **s_u, float **s_v, float **s_w, int compute_mult, float **pi, float *work_memory)
	{

//alpha = 1.0; // no relaxation for the moment TODO remove 

/*printf("\ncazzo\n");*/

/*	int nbx = nb - nu;*/
/*	if(nbx<0)*/
/*		nbx = 0;*/
/*	int nbu = nu<nb ? nu : nb ;*/

	// indeces
	int jj, ll, ii, bs0;


	// constants
	const int bs = S_MR; //d_get_mr();
	const int ncl = S_NCL;
	const int nal = bs*ncl; // number of floats per cache line

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
	float *ux_r[N+1];
/*	float *ux_v[N+1];*/
/*	float *ux_w[N+1];*/
	float *pL[N+1];
	float *pl[N+1];
	float *bd[N+1];
	float *bl[N+1];
	float *bb[N];
	float *work1;
	float *diag;
	float *Zi[N+1]; // inverse of Hessian of soft constraints slack variables
/*	float *s_u[N+1]; // soft constraints slack variable*/
/*	float *s_v[N+1]; // soft constraints slack variable*/
/*	float *s_w[N+1]; // soft constraints slack variable*/
	float *s_r[N+1]; // soft constraints slack variable
	float *Pb[N];
	
	float *ptr = work_memory; // TODO + 10*anx

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

	// inverse (of diagonal) of Hessian of soft constraints slack variables
	for(jj=0; jj<=N; jj++)
		{
		Zi[jj] = ptr;
		ptr += 2*anx;
		}

/*	// soft constraints slack variables*/
/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		s_u[jj] = ptr;*/
/*		ptr += 2*anx;*/
/*		}*/

/*	// soft constraints slack variables*/
/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		s_v[jj] = ptr;*/
/*		ptr += 2*anx;*/
/*		}*/

/*	// soft constraints slack variables*/
/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		s_w[jj] = ptr;*/
/*		ptr += 2*anx;*/
/*		}*/

	// soft constraints slack variables
	for(jj=0; jj<=N; jj++)
		{
		s_r[jj] = ptr;
		ptr += 2*anx;
		}

	// backup of P*b
	for(jj=0; jj<N; jj++)
		{
		Pb[jj] = ptr;
		ptr += anx;
		}



	float temp, v_temp, norm_p=1e3, norm_d=1e3, x_temp;

	
	
	// initialize u and x (cold start)
	if(warm_start==0)
		{
		// states and inputs
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
		// slack variables of soft constraints
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<2*anx; ll++)
				{
				s_u[jj][ll] = 0.0;
				}
			}
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<2*anx; ll++)
				{
				s_v[jj][ll] = 0.0;
				}
			}
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<2*anx; ll++)
				{
				s_w[jj][ll] = 0.0;
				}
			}
		}
	



	// first iteration (initial factorization)
	
	// reset iteration counter
	*kk = 0; 
	


	if(compute_fact==1) // factorize hessina in the first iteration
		{

		// soft constraints cost function

		// invert Hessian of soft constraints slack variables
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx; ll++)
				{
				// upper
				Zi[jj][ll] = 1.0/(Z[jj][ll] + rho);
				s_u[jj][ll] = - Zi[jj][ll]*(z[jj][ll] + rho*(s_w[jj][ll] - s_v[jj][ll]));
				// lower
				Zi[jj][anx+ll] = 1.0/(Z[jj][anx+ll] + rho);
				s_u[jj][anx+ll] = - Zi[jj][anx+ll]*(z[jj][anx+ll] + rho*(s_w[jj][anx+ll] - s_v[jj][anx+ll]));
				}
			}

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
		s_ric_sv_mpc(nx, nu, N, pBAbt, pQ, ux_u, pL, work1, diag, compute_mult, pi);

		// constraints
		norm_p = 0;
		for(jj=0; jj<=N; jj++)
			{
			// hard constraints on inputs
			for(ll=0; ll<nu; ll++)
				{
				ux_r[jj][ll] = alpha*ux_u[jj][ll] + (1.0-alpha)*ux_v[jj][ll]; // relaxation
/*				v_temp = - ( - ux_w[jj][ll] - ux_u[jj][ll] );*/
				v_temp = - ( - ux_w[jj][ll] - ux_r[jj][ll] );
				v_temp = fmax(v_temp, lb[jj][ll]);
				v_temp = fmin(v_temp, ub[jj][ll]);
				temp = v_temp - ux_v[jj][ll];
				norm_p += temp*temp;
				ux_v[jj][ll] = v_temp;
				}
			// soft constraints on states
			for(ll=0; ll<nx; ll++)
				{
				ux_r[jj][nu+ll] = alpha*ux_u[jj][nu+ll] + (1.0-alpha)*ux_v[jj][nu+ll]; // relaxation
				s_r[jj][ll] = alpha*s_u[jj][ll] + (1.0-alpha)*s_v[jj][ll]; // relaxation
				s_r[jj][anx+ll] = alpha*s_u[jj][anx+ll] + (1.0-alpha)*s_v[jj][anx+ll]; // relaxation
/*				x_temp = - ux_w[jj][nu+ll] - ux_u[jj][nu+ll];*/
				x_temp = - ux_w[jj][nu+ll] - ux_r[jj][nu+ll];
				v_temp = - ( x_temp );
				v_temp = fmax(v_temp, lb[jj][nu+ll]);
				v_temp = fmin(v_temp, ub[jj][nu+ll]);
/*				s_v[jj][ll] = fmax( -0.5*( ub[jj][nu+ll] + x_temp + (- s_w[jj][ll] - s_u[jj][ll])), 0);*/
/*				s_v[jj][anx+ll] = fmax( -0.5*( - lb[jj][nu+ll] - x_temp + (- s_w[jj][anx+ll] - s_u[jj][anx+ll])), 0);*/
				s_v[jj][ll] = fmax( -0.5*( ub[jj][nu+ll] + x_temp + (- s_w[jj][ll] - s_r[jj][ll])), 0);
				s_v[jj][anx+ll] = fmax( -0.5*( - lb[jj][nu+ll] - x_temp + (- s_w[jj][anx+ll] - s_r[jj][anx+ll])), 0);
				v_temp = v_temp + s_v[jj][ll] - s_v[jj][anx+ll];
				temp = v_temp - ux_v[jj][nu+ll];
				norm_p += temp*temp;
				ux_v[jj][nu+ll] = v_temp;
				}
			}
		norm_p = sqrt(norm_p);
		stat[0+5*kk[0]] = norm_p;

		// integral of error
		norm_d = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nu; ll++)
				{
/*				temp = ux_u[jj][ll] - ux_v[jj][ll];*/
				temp = ux_r[jj][ll] - ux_v[jj][ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][ll] += temp;
				}
			for(ll=0; ll<nx; ll++)
				{
/*				temp = ux_u[jj][nu+ll] - ux_v[jj][nu+ll];*/
				temp = ux_r[jj][nu+ll] - ux_v[jj][nu+ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][nu+ll] += temp;
				}
			for(ll=0; ll<nx; ll++)
				{
/*				s_w[jj][ll] += s_u[jj][ll] - s_v[jj][ll];*/
/*				s_w[jj][anx+ll] += s_u[jj][anx+ll] - s_v[jj][anx+ll];*/
				s_w[jj][ll] += s_r[jj][ll] - s_v[jj][ll];
				s_w[jj][anx+ll] += s_r[jj][anx+ll] - s_v[jj][anx+ll];
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

		// soft constraints cost function

		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nx; ll++)
				{
				// upper
				s_u[jj][ll] = - Zi[jj][ll]*(z[jj][ll] + rho*(s_w[jj][ll] - s_v[jj][ll]));
				// lower
				s_u[jj][anx+ll] = - Zi[jj][anx+ll]*(z[jj][anx+ll] + rho*(s_w[jj][anx+ll] - s_v[jj][anx+ll]));
				}
			}


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
		s_ric_trs_mpc(nx, nu, N, pBAbt, pL, pl, ux_u, work1, compute_Pb, Pb, compute_mult, pi);
		compute_Pb = 0;

/*for(jj=0; jj<=N; jj++)*/
/*	d_print_mat(1, nu+nx, ux_u[jj], 1);*/
/*for(jj=0; jj<=N; jj++)*/
/*	d_print_mat(1, 2*anx, s_u[jj], 1);*/
/*exit(1);*/

		// constraints
		norm_p = 0;
		for(jj=0; jj<=N; jj++)
			{
			// hard constraints on inputs
			for(ll=0; ll<nu; ll++)
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
			// soft constraints on states
			for(ll=0; ll<nx; ll++)
				{
				ux_r[jj][nu+ll] = alpha*ux_u[jj][nu+ll] + (1.0-alpha)*ux_v[jj][nu+ll]; // relaxation
				s_r[jj][ll] = alpha*s_u[jj][ll] + (1.0-alpha)*s_v[jj][ll]; // relaxation
				s_r[jj][anx+ll] = alpha*s_u[jj][anx+ll] + (1.0-alpha)*s_v[jj][anx+ll]; // relaxation
/*				x_temp = - ux_w[jj][nu+ll] - ux_u[jj][nu+ll];*/
				x_temp = - ux_w[jj][nu+ll] - ux_r[jj][nu+ll];
				v_temp = - ( x_temp );
				v_temp = fmax(v_temp, lb[jj][nu+ll]);
				v_temp = fmin(v_temp, ub[jj][nu+ll]);
/*				s_v[jj][ll] = fmax( -0.5*( ub[jj][nu+ll] + x_temp + (- s_w[jj][ll] - s_u[jj][ll])), 0);*/
/*				s_v[jj][anx+ll] = fmax( -0.5*( - lb[jj][nu+ll] - x_temp + (- s_w[jj][anx+ll] - s_u[jj][anx+ll])), 0);*/
				s_v[jj][ll] = fmax( -0.5*( ub[jj][nu+ll] + x_temp + (- s_w[jj][ll] - s_r[jj][ll])), 0);
				s_v[jj][anx+ll] = fmax( -0.5*( - lb[jj][nu+ll] - x_temp + (- s_w[jj][anx+ll] - s_r[jj][anx+ll])), 0);
				v_temp = v_temp + s_v[jj][ll] - s_v[jj][anx+ll];
				temp = v_temp - ux_v[jj][nu+ll];
				norm_p += temp*temp;
				ux_v[jj][nu+ll] = v_temp;
				}
			}
		norm_p = sqrt(norm_p);
		stat[0+5*kk[0]] = norm_p;
	
		// integral of error
		norm_d = 0;
		for(jj=0; jj<=N; jj++)
			{
			for(ll=0; ll<nu; ll++)
				{
/*				temp = ux_u[jj][ll] - ux_v[jj][ll];*/
				temp = ux_r[jj][ll] - ux_v[jj][ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][ll] += temp;
				}
			for(ll=0; ll<nx; ll++)
				{
/*				temp = ux_u[jj][nu+ll] - ux_v[jj][nu+ll];*/
				temp = ux_r[jj][nu+ll] - ux_v[jj][nu+ll]; // relaxation
				norm_d += temp*temp;
				ux_w[jj][nu+ll] += temp;
				}
			for(ll=0; ll<nx; ll++)
				{
/*				s_w[jj][ll] += s_u[jj][ll] - s_v[jj][ll];*/
/*				s_w[jj][anx+ll] += s_u[jj][anx+ll] - s_v[jj][anx+ll];*/
				s_w[jj][ll] += s_r[jj][ll] - s_v[jj][ll];
				s_w[jj][anx+ll] += s_r[jj][anx+ll] - s_v[jj][anx+ll];
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
