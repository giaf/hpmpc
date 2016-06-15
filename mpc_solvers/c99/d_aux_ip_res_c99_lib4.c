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
#include "../../include/blas_d.h"
#include "../../include/block_size.h"



void d_update_hessian_gradient_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double **res_d, double **res_m, double **t, double **lam, double **t_inv, double **Qx, double **qx)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png;
	
	double temp0, temp1;
	
	double 
		*ptr_res_d, *ptr_Qx, *ptr_qx, *ptr_t, *ptr_lam, *ptr_res_m, *ptr_t_inv;
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_t_inv = t_inv[jj];
		ptr_res_d = res_d[jj];
		ptr_res_m = res_m[jj];
		ptr_Qx    = Qx[jj];
		ptr_qx    = qx[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0-3; ii+=4)
				{

				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
				ptr_Qx[ii+0] = ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+pnb+0]*ptr_lam[ii+pnb+0];
				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]+ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]);

				ptr_t_inv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_t_inv[ii+pnb+1] = 1.0/ptr_t[ii+pnb+1];
				ptr_Qx[ii+1] = ptr_t_inv[ii+1]*ptr_lam[ii+1] + ptr_t_inv[ii+pnb+1]*ptr_lam[ii+pnb+1];
				ptr_qx[ii+1] = ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]) - ptr_t_inv[ii+pnb+1]*(ptr_res_m[ii+pnb+1]+ptr_lam[ii+pnb+1]*ptr_res_d[ii+pnb+1]);

				ptr_t_inv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_t_inv[ii+pnb+2] = 1.0/ptr_t[ii+pnb+2];
				ptr_Qx[ii+2] = ptr_t_inv[ii+2]*ptr_lam[ii+2] + ptr_t_inv[ii+pnb+2]*ptr_lam[ii+pnb+2];
				ptr_qx[ii+2] = ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]) - ptr_t_inv[ii+pnb+2]*(ptr_res_m[ii+pnb+2]+ptr_lam[ii+pnb+2]*ptr_res_d[ii+pnb+2]);

				ptr_t_inv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_t_inv[ii+pnb+3] = 1.0/ptr_t[ii+pnb+3];
				ptr_Qx[ii+3] = ptr_t_inv[ii+3]*ptr_lam[ii+3] + ptr_t_inv[ii+pnb+3]*ptr_lam[ii+pnb+3];
				ptr_qx[ii+3] = ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]) - ptr_t_inv[ii+pnb+3]*(ptr_res_m[ii+pnb+3]+ptr_lam[ii+pnb+3]*ptr_res_d[ii+pnb+3]);

				}
			for(; ii<nb0; ii++)
				{

				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
				ptr_Qx[ii+0] = ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+pnb+0]*ptr_lam[ii+pnb+0];
				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]+ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]);

				}

			ptr_t     += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_t_inv += 2*pnb;
			ptr_res_d += 2*pnb;
			ptr_res_m += 2*pnb;
			ptr_Qx    += pnb;
			ptr_qx    += pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

		
			png = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_Qx[ii+0] = ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+png+0]*ptr_lam[ii+png+0];
				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]+ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]);

				ptr_t_inv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_t_inv[ii+png+1] = 1.0/ptr_t[ii+png+1];
				ptr_Qx[ii+1] = ptr_t_inv[ii+1]*ptr_lam[ii+1] + ptr_t_inv[ii+png+1]*ptr_lam[ii+png+1];
				ptr_qx[ii+1] = ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]) - ptr_t_inv[ii+png+1]*(ptr_res_m[ii+png+1]+ptr_lam[ii+png+1]*ptr_res_d[ii+png+1]);

				ptr_t_inv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_t_inv[ii+png+2] = 1.0/ptr_t[ii+png+2];
				ptr_Qx[ii+2] = ptr_t_inv[ii+2]*ptr_lam[ii+2] + ptr_t_inv[ii+png+2]*ptr_lam[ii+png+2];
				ptr_qx[ii+2] = ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]) - ptr_t_inv[ii+png+2]*(ptr_res_m[ii+png+2]+ptr_lam[ii+png+2]*ptr_res_d[ii+png+2]);

				ptr_t_inv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_t_inv[ii+png+3] = 1.0/ptr_t[ii+png+3];
				ptr_Qx[ii+3] = ptr_t_inv[ii+3]*ptr_lam[ii+3] + ptr_t_inv[ii+png+3]*ptr_lam[ii+png+3];
				ptr_qx[ii+3] = ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]) - ptr_t_inv[ii+png+3]*(ptr_res_m[ii+png+3]+ptr_lam[ii+png+3]*ptr_res_d[ii+png+3]);

				}
			for(; ii<ng0; ii++)
				{
				
				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_Qx[ii+0] = ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+png+0]*ptr_lam[ii+png+0];
				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]+ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]);

				}

			}

		}

	}



void d_compute_dt_dlam_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **dux, double **t, double **t_inv, double **lam, double **pDCt, double **res_d, double **res_m, double **dt, double **dlam)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, pnb, ng0, png, cng;

	double
		*ptr_res_d, *ptr_res_m, *ptr_dux, *ptr_t, *ptr_t_inv, *ptr_dt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	int jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		ptr_res_d = res_d[jj];
		ptr_res_m = res_m[jj];
		ptr_dux   = dux[jj];
		ptr_t     = t[jj];
		ptr_t_inv = t_inv[jj];
		ptr_dt    = dt[jj];
		ptr_lam   = lam[jj];
		ptr_dlam  = dlam[jj];
		ptr_idxb  = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb = (nb0+bs-1)/bs*bs;

			// box constraints
			for(ll=0; ll<nb0; ll++)
				{
				
				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_res_d[ll+0];
				ptr_dt[ll+pnb] = - ptr_dux[ptr_idxb[ll]] + ptr_res_d[ll+pnb];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+pnb] = - ptr_t_inv[ll+pnb] * ( ptr_lam[ll+pnb]*ptr_dt[ll+pnb] + ptr_res_m[ll+pnb] );

				}

			ptr_res_d += 2*pnb;
			ptr_res_m += 2*pnb;
			ptr_t     += 2*pnb;
			ptr_t_inv += 2*pnb;
			ptr_dt    += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_dlam  += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			png = (ng0+bs-1)/bs*bs;
			cng = (ng0+ncl-1)/ncl*ncl;

			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt, ptr_dt);

			for(ll=0; ll<ng0; ll++)
				{

				ptr_dt[ll+png] = - ptr_dt[ll];

				ptr_dt[ll+0]   -= ptr_res_d[ll+0];
				ptr_dt[ll+png] += ptr_res_d[ll+png];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+png] = - ptr_t_inv[ll+png] * ( ptr_lam[ll+png]*ptr_dt[ll+png] + ptr_res_m[ll+png] );

				}

			}

		}		

	return;
	
	}



void d_compute_alpha_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **dux, double **t, double **t_inv, double **lam, double **pDCt, double **res_d, double **res_m, double **dt, double **dlam, double *ptr_alpha)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, pnb, ng0, png, cng;

	double alpha = ptr_alpha[0];
	
	double
		*ptr_res_d, *ptr_res_m, *ptr_dux, *ptr_t, *ptr_t_inv, *ptr_dt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	int jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		ptr_res_d = res_d[jj];
		ptr_res_m = res_m[jj];
		ptr_dux   = dux[jj];
		ptr_t     = t[jj];
		ptr_t_inv = t_inv[jj];
		ptr_dt    = dt[jj];
		ptr_lam   = lam[jj];
		ptr_dlam  = dlam[jj];
		ptr_idxb  = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb = (nb0+bs-1)/bs*bs;

			// box constraints
			for(ll=0; ll<nb0; ll++)
				{
				
				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_res_d[ll+0];
				ptr_dt[ll+pnb] = - ptr_dux[ptr_idxb[ll]] + ptr_res_d[ll+pnb];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+pnb] = - ptr_t_inv[ll+pnb] * ( ptr_lam[ll+pnb]*ptr_dt[ll+pnb] + ptr_res_m[ll+pnb] );

				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+pnb]>ptr_lam[ll+pnb] )
					{
					alpha = - ptr_lam[ll+pnb] / ptr_dlam[ll+pnb];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+pnb]>ptr_t[ll+pnb] )
					{
					alpha = - ptr_t[ll+pnb] / ptr_dt[ll+pnb];
					}

				}

			ptr_res_d += 2*pnb;
			ptr_res_m += 2*pnb;
			ptr_t     += 2*pnb;
			ptr_t_inv += 2*pnb;
			ptr_dt    += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_dlam  += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			png = (ng0+bs-1)/bs*bs;
			cng = (ng0+ncl-1)/ncl*ncl;

			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt, ptr_dt);

			for(ll=0; ll<ng0; ll++)
				{

				ptr_dt[ll+png] = - ptr_dt[ll];

				ptr_dt[ll+0]   -= ptr_res_d[ll+0];
				ptr_dt[ll+png] += ptr_res_d[ll+png];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+png] = - ptr_t_inv[ll+png] * ( ptr_lam[ll+png]*ptr_dt[ll+png] + ptr_res_m[ll+png] );

				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+png]>ptr_lam[ll+png] )
					{
					alpha = - ptr_lam[ll+png] / ptr_dlam[ll+png];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+png]>ptr_t[ll+png] )
					{
					alpha = - ptr_t[ll+png] / ptr_dt[ll+png];
					}

				}

			}

		}		

	// store alpha
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_update_var_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nx1, nb0, pnb, ng0, png;

	int jj, ll;
	
	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		pnb = bs*((nb0+bs-1)/bs); // cache aligned number of box constraints
		ng0 = ng[jj];
		png = bs*((ng0+bs-1)/bs); // cache aligned number of box constraints
		if(jj<N)
			nx1 = nx[jj+1];
		else
			nx1 = 0;
		
		ptr_pi   = pi[jj];
		ptr_dpi  = dpi[jj];
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

		// update inputs and states
		for(ll=0; ll<nu0+nx0-3; ll+=4)
			{
			ptr_ux[ll+0] += alpha*ptr_dux[ll+0];
			ptr_ux[ll+1] += alpha*ptr_dux[ll+1];
			ptr_ux[ll+2] += alpha*ptr_dux[ll+2];
			ptr_ux[ll+3] += alpha*ptr_dux[ll+3];
			}
		for(; ll<nu0+nx0; ll++)
			ptr_ux[ll] += alpha*ptr_dux[ll];

		// update equality constrained multipliers
		for(ll=0; ll<nx1-3; ll+=4)
			{
			ptr_pi[ll+0] += alpha*ptr_dpi[ll+0];
			ptr_pi[ll+1] += alpha*ptr_dpi[ll+1];
			ptr_pi[ll+2] += alpha*ptr_dpi[ll+2];
			ptr_pi[ll+3] += alpha*ptr_dpi[ll+3];
			}
		for(; ll<nx1; ll++)
			ptr_pi[ll] += alpha*ptr_dpi[ll];

		// box constraints
		for(ll=0; ll<nb0; ll++)
			{
			ptr_lam[ll+0]   += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+pnb] += alpha*ptr_dlam[ll+pnb];
			ptr_t[ll+0]   += alpha*ptr_dt[ll+0];
			ptr_t[ll+pnb] += alpha*ptr_dt[ll+pnb];
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// genreal constraints
		for(ll=0; ll<ng0; ll++)
			{
			ptr_lam[ll+0]   += alpha*ptr_dlam[ll+0];
			ptr_lam[ll+png] += alpha*ptr_dlam[ll+png];
			ptr_t[ll+0]   += alpha*ptr_dt[ll+0];
			ptr_t[ll+png] += alpha*ptr_dt[ll+png];
			}

		}

	return;
	
	}



void d_compute_mu_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double alpha, double **lam, double **dlam, double **t, double **dt, double *ptr_mu, double mu_scal)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png;

	int jj, ll;
	
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;
		
	double mu = 0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		nb0 = nb[jj];
		pnb = (nb0+bs-1)/bs*bs;
		ng0 = ng[jj];
		png = (ng0+bs-1)/bs*bs;
		
		ptr_t    = t[jj];
		ptr_lam  = lam[jj];
		ptr_dt   = dt[jj];
		ptr_dlam = dlam[jj];

		// box constraints
		for(ll=0 ; ll<nb0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+pnb] + alpha*ptr_dlam[ll+pnb]) * (ptr_t[ll+pnb] + alpha*ptr_dt[ll+pnb]);
			}

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// general constraints
		for(ll=0; ll<ng0; ll++)
			{
			mu += (ptr_lam[ll+0] + alpha*ptr_dlam[ll+0]) * (ptr_t[ll+0] + alpha*ptr_dt[ll+0]) + (ptr_lam[ll+png] + alpha*ptr_dlam[ll+png]) * (ptr_t[ll+png] + alpha*ptr_dt[ll+png]);
			}

		}

	// scale mu
	mu *= mu_scal;
		
	ptr_mu[0] = mu;

	return;

	}



void d_compute_centering_correction_res_mpc_hard_tv(int N, int *nb, int *ng, double sigma_mu, double **dt, double **dlam, double **res_m)
	{

	const int bs = D_MR;

	int pnb, png;

	int ii, jj;

	double
		*ptr_res_m, *ptr_dt, *ptr_dlam;
	
	for(ii=0; ii<=N; ii++)
		{

		pnb = (nb[ii]+bs-1)/bs*bs;
		png = (ng[ii]+bs-1)/bs*bs;

		ptr_res_m = res_m[ii];
		ptr_dt    = dt[ii];
		ptr_dlam  = dlam[ii];

		for(jj=0; jj<nb[ii]; jj++)
			{
			ptr_res_m[jj]     += ptr_dt[jj]     * ptr_dlam[jj]     - sigma_mu;
			ptr_res_m[pnb+jj] += ptr_dt[pnb+jj] * ptr_dlam[pnb+jj] - sigma_mu;
			}
		for(jj=0; jj<ng[ii]; jj++)
			{
			ptr_res_m[2*pnb+jj]     += ptr_dt[2*pnb+jj]     * ptr_dlam[2*pnb+jj]     - sigma_mu;
			ptr_res_m[2*pnb+png+jj] += ptr_dt[2*pnb+png+jj] * ptr_dlam[2*pnb+png+jj] - sigma_mu;
			}
		}

	}



void d_update_gradient_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double **res_d, double **res_m, double **lam, double **t_inv, double **qx)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png;
	
	double temp0, temp1;
	
	double 
		*ptr_res_d, *ptr_Qx, *ptr_qx, *ptr_lam, *ptr_res_m, *ptr_t_inv;
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		ptr_lam   = lam[jj];
		ptr_t_inv = t_inv[jj];
		ptr_res_d = res_d[jj];
		ptr_res_m = res_m[jj];
		ptr_qx    = qx[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0-3; ii+=4)
				{

				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]+ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]);

				ptr_qx[ii+1] = ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]) - ptr_t_inv[ii+pnb+1]*(ptr_res_m[ii+pnb+1]+ptr_lam[ii+pnb+1]*ptr_res_d[ii+pnb+1]);

				ptr_qx[ii+2] = ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]) - ptr_t_inv[ii+pnb+2]*(ptr_res_m[ii+pnb+2]+ptr_lam[ii+pnb+2]*ptr_res_d[ii+pnb+2]);

				ptr_qx[ii+3] = ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]) - ptr_t_inv[ii+pnb+3]*(ptr_res_m[ii+pnb+3]+ptr_lam[ii+pnb+3]*ptr_res_d[ii+pnb+3]);

				}
			for(; ii<nb0; ii++)
				{

				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]+ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]);

				}

			ptr_lam   += 2*pnb;
			ptr_t_inv += 2*pnb;
			ptr_res_d += 2*pnb;
			ptr_res_m += 2*pnb;
			ptr_qx    += pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			png = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]+ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]);

				ptr_qx[ii+1] = ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]) - ptr_t_inv[ii+png+1]*(ptr_res_m[ii+png+1]+ptr_lam[ii+png+1]*ptr_res_d[ii+png+1]);

				ptr_qx[ii+2] = ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]) - ptr_t_inv[ii+png+2]*(ptr_res_m[ii+png+2]+ptr_lam[ii+png+2]*ptr_res_d[ii+png+2]);

				ptr_qx[ii+3] = ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]) - ptr_t_inv[ii+png+3]*(ptr_res_m[ii+png+3]+ptr_lam[ii+png+3]*ptr_res_d[ii+png+3]);

				}
			for(; ii<ng0; ii++)
				{
				
				ptr_qx[ii+0] = ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]) - ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]+ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]);

				}

			}

		}

	}




