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

#include <math.h> // TODO remove if not needed

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#ifdef BLASFEO

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_aux.h>
#include <blasfeo_d_blas.h>

#include "../../include/block_size.h" // TODO remove !!!!!



// initialize variables

void d_init_var_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strvec *hsux, struct d_strvec *hspi, struct d_strmat *hsDCt, struct d_strvec *hsdb, struct d_strvec *hst, struct d_strvec *hslam, double mu0, int warm_start)
	{

	int jj, ll, ii;

	double *ptr_ux, *ptr_pi, *ptr_db, *ptr_t, *ptr_lam;

	int nb0, ng0;
	
	double thr0 = 0.1; // minimum vale of t (minimum distance from a constraint)


	// cold start
	if(warm_start==0)
		{
		for(jj=0; jj<=N; jj++)
			{
			ptr_ux = hsux[jj].pa;
			for(ll=0; ll<nu[jj]+nx[jj]; ll++)
				{
				ptr_ux[ll] = 0.0;
				}
			}
		}


	// check bounds & initialize multipliers
	for(jj=0; jj<=N; jj++)
		{
		nb0 = nb[jj];
		ptr_ux = hsux[jj].pa;
		ptr_db = hsdb[jj].pa;
		ptr_lam = hslam[jj].pa;
		ptr_t = hst[jj].pa;
		for(ll=0; ll<nb0; ll++)
			{
			ptr_t[ll]     = - ptr_db[ll]     + ptr_ux[hidxb[jj][ll]];
			ptr_t[nb0+ll] =   ptr_db[nb0+ll] - ptr_ux[hidxb[jj][ll]];
			if(ptr_t[ll] < thr0)
				{
				if(ptr_t[nb0+ll] < thr0)
					{
					ptr_ux[hidxb[jj][ll]] = ( - ptr_db[nb0+ll] + ptr_db[ll])*0.5;
					ptr_t[ll]     = thr0; //- hdb[jj][ll]     + hux[jj][hidxb[jj][ll]];
					ptr_t[nb0+ll] = thr0; //  hdb[jj][nb0+ll] - hux[jj][hidxb[jj][ll]];
					}
				else
					{
					ptr_t[ll] = thr0;
					ptr_ux[hidxb[jj][ll]] = ptr_db[ll] + thr0;
					}
				}
			else if(ptr_t[nb0+ll] < thr0)
				{
				ptr_t[nb0+ll] = thr0;
				ptr_ux[hidxb[jj][ll]] = ptr_db[nb0+ll] - thr0;
				}
			ptr_lam[ll]     = mu0/ptr_t[ll];
			ptr_lam[nb0+ll] = mu0/ptr_t[nb0+ll];
			}
		}


	// initialize pi
	for(jj=1; jj<=N; jj++)
		{
		ptr_pi = hspi[jj].pa;
		for(ll=0; ll<nx[jj]; ll++)
			ptr_pi[ll] = 0.0; // initialize multipliers to zero
		}


	// TODO find a better way to initialize general constraints
	for(jj=0; jj<=N; jj++)
		{
		nb0 = nb[jj];
		ng0 = ng[jj];
		if(ng0>0)
			{
			ptr_t   = hst[jj].pa;
			ptr_lam = hslam[jj].pa;
			ptr_db  = hsdb[jj].pa;
			dgemv_t_libstr(nu[jj]+nx[jj], ng0, 1.0, &hsDCt[jj], 0, 0, &hsux[jj], 0, 0.0, &hst[jj], 2*nb0, &hst[jj], 2*nb0);
			for(ll=2*nb0; ll<2*nb0+ng0; ll++)
				{
				ptr_t[ll+ng0] = - ptr_t[ll];
				ptr_t[ll]     -= ptr_db[ll];
				ptr_t[ll+ng0] += ptr_db[ll+ng0];
				ptr_t[ll]     = fmax( thr0, ptr_t[ll] );
				ptr_t[ng0+ll] = fmax( thr0, ptr_t[ng0+ll] );
				ptr_lam[ll]     = mu0/ptr_t[ll];
				ptr_lam[ng0+ll] = mu0/ptr_t[ng0+ll];
				}
			}
		}

	}



// IPM with no residuals

void d_update_hessian_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, struct d_strvec *hsdb, double sigma_mu, struct d_strvec *hst, struct d_strvec *hstinv, struct d_strvec *hslam, struct d_strvec *hslamt, struct d_strvec *hsdlam, struct d_strvec *hsQx, struct d_strvec *hsqx)
	{
	
	int ii, jj, bs0;

	__m256d
		v_ones, v_sigma_mu, v_mask, v_left,
		v_tmp, v_lam, v_lamt, v_dlam, v_db,
		v_tmp0, v_tmp1,
		v_lam0, v_lam1,
		v_lamt0, v_lamt1,
		v_dlam0, v_dlam1,
		v_Qx0, v_Qx1,
		v_qx0, v_qx1,
		v_bd0, v_bd2,
		v_db0, v_db2;
	
	__m256i
		i_mask;

	double 
		*ptr_db, *ptr_Qx, *ptr_qx,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );


	double ii_left;

	int nb0, ng0;
	
	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t     = hst[jj].pa;
		ptr_lam   = hslam[jj].pa;
		ptr_lamt  = hslamt[jj].pa;
		ptr_dlam  = hsdlam[jj].pa;
		ptr_tinv  = hstinv[jj].pa;
		ptr_db    = hsdb[jj].pa;
		ptr_Qx    = hsQx[jj].pa;
		ptr_qx    = hsqx[jj].pa;

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ii=0; ii<nb0-3; ii+=4)
				{

				v_tmp0  = _mm256_loadu_pd( &ptr_t[0*nb0+ii] );
				v_tmp1  = _mm256_loadu_pd( &ptr_t[1*nb0+ii] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ii] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ii] );
				v_qx0   = _mm256_loadu_pd( &ptr_db[0*nb0+ii] );
				v_qx1   = _mm256_loadu_pd( &ptr_db[1*nb0+ii] );
				_mm256_storeu_pd( &ptr_tinv[0*nb0+ii], v_tmp0 );
				_mm256_storeu_pd( &ptr_tinv[1*nb0+ii], v_tmp1 );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_storeu_pd( &ptr_lamt[0*nb0+ii], v_lamt0 );
				_mm256_storeu_pd( &ptr_lamt[1*nb0+ii], v_lamt1 );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				_mm256_storeu_pd( &ptr_dlam[0*nb0+ii], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[1*nb0+ii], v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_sub_pd( v_lam1, v_qx1 );
				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_storeu_pd( &ptr_Qx[ii], v_Qx0 );
				_mm256_storeu_pd( &ptr_qx[ii], v_qx0 );

				}
			if(ii<nb0)
				{

				ii_left = nb0-ii;
				v_left= _mm256_broadcast_sd( &ii_left );
				v_mask= _mm256_loadu_pd( d_mask );
				i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

				v_tmp0  = _mm256_loadu_pd( &ptr_t[0*nb0+ii] );
				v_tmp1  = _mm256_loadu_pd( &ptr_t[1*nb0+ii] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				_mm256_maskstore_pd( &ptr_tinv[0*nb0+ii], i_mask, v_tmp0 );
				_mm256_maskstore_pd( &ptr_tinv[1*nb0+ii], i_mask, v_tmp1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ii] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ii] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				_mm256_maskstore_pd( &ptr_lamt[0*nb0+ii], i_mask, v_lamt0 );
				_mm256_maskstore_pd( &ptr_lamt[1*nb0+ii], i_mask, v_lamt1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_maskstore_pd( &ptr_dlam[0*nb0+ii], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[1*nb0+ii], i_mask, v_dlam1 );
				v_qx0   = _mm256_loadu_pd( &ptr_db[0*nb0+ii] );
				v_qx1   = _mm256_loadu_pd( &ptr_db[1*nb0+ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_sub_pd( v_lam1, v_qx1 );
				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_maskstore_pd( &ptr_Qx[ii], i_mask, v_Qx0 );
				_mm256_maskstore_pd( &ptr_qx[ii], i_mask, v_qx0 );

				}
		
			ptr_t     += 2*nb0;
			ptr_lam   += 2*nb0;
			ptr_lamt  += 2*nb0;
			ptr_dlam  += 2*nb0;
			ptr_tinv  += 2*nb0;
			ptr_db    += 2*nb0;
			ptr_Qx    += nb0;
			ptr_qx    += nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			for(ii=0; ii<ng0-3; ii+=4)
				{

				v_tmp0  = _mm256_loadu_pd( &ptr_t[0*ng0+ii] );
				v_tmp1  = _mm256_loadu_pd( &ptr_t[1*ng0+ii] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ii] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ii] );
				v_qx0   = _mm256_loadu_pd( &ptr_db[0*ng0+ii] );
				v_qx1   = _mm256_loadu_pd( &ptr_db[1*ng0+ii] );
				_mm256_storeu_pd( &ptr_tinv[0*ng0+ii], v_tmp0 );
				_mm256_storeu_pd( &ptr_tinv[1*ng0+ii], v_tmp1 );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_storeu_pd( &ptr_lamt[0*ng0+ii], v_lamt0 );
				_mm256_storeu_pd( &ptr_lamt[1*ng0+ii], v_lamt1 );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				_mm256_storeu_pd( &ptr_dlam[0*ng0+ii], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[1*ng0+ii], v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_sub_pd( v_lam1, v_qx1 );
				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_storeu_pd( &ptr_Qx[ii], v_Qx0 );
				_mm256_storeu_pd( &ptr_qx[ii], v_qx0 );

				}
			if(ii<ng0)
				{

				ii_left = ng0 - ii;
				v_left  = _mm256_broadcast_sd( &ii_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

				v_tmp0  = _mm256_loadu_pd( &ptr_t[0*ng0+ii] );
				v_tmp1  = _mm256_loadu_pd( &ptr_t[1*ng0+ii] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
				_mm256_maskstore_pd( &ptr_tinv[0*ng0+ii], i_mask, v_tmp0 );
				_mm256_maskstore_pd( &ptr_tinv[1*ng0+ii], i_mask, v_tmp1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ii] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ii] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
				_mm256_maskstore_pd( &ptr_lamt[0*ng0+ii], i_mask, v_lamt0 );
				_mm256_maskstore_pd( &ptr_lamt[1*ng0+ii], i_mask, v_lamt1 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
				_mm256_maskstore_pd( &ptr_dlam[0*ng0+ii], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[1*ng0+ii], i_mask, v_dlam1 );
				v_qx0   = _mm256_loadu_pd( &ptr_db[0*ng0+ii] );
				v_qx1   = _mm256_loadu_pd( &ptr_db[1*ng0+ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
				v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
				v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
				v_qx1   = _mm256_sub_pd( v_lam1, v_qx1 );
				v_Qx0   = _mm256_add_pd( v_lamt0, v_lamt1 );
				v_qx0   = _mm256_sub_pd( v_qx1, v_qx0 );
				_mm256_maskstore_pd( &ptr_Qx[ii], i_mask, v_Qx0 );
				_mm256_maskstore_pd( &ptr_qx[ii], i_mask, v_qx0 );

				}

			}

		}

	}



void d_update_gradient_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double sigma_mu, struct d_strvec *hsdt, struct d_strvec *hsdlam, struct d_strvec *hstinv, struct d_strvec *hsqx)
	{

	int ii, jj;

	int nb0, ng0;

	double
		*ptr_dlam, *ptr_t_inv, *ptr_dt, *ptr_pl2, *ptr_qx;

	for(jj=0; jj<=N; jj++)
		{

		ptr_dlam  = hsdlam[jj].pa;
		ptr_dt    = hsdt[jj].pa;
		ptr_t_inv = hstinv[jj].pa;
		ptr_qx    = hsqx[jj].pa;

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ii=0; ii<nb0-3; ii+=4)
				{
				ptr_dlam[0*nb0+ii+0] = ptr_t_inv[0*nb0+ii+0]*(sigma_mu - ptr_dlam[0*nb0+ii+0]*ptr_dt[0*nb0+ii+0]);
				ptr_dlam[1*nb0+ii+0] = ptr_t_inv[1*nb0+ii+0]*(sigma_mu - ptr_dlam[1*nb0+ii+0]*ptr_dt[1*nb0+ii+0]);
				ptr_qx[ii+0] += ptr_dlam[1*nb0+ii+0] - ptr_dlam[0*nb0+ii+0];

				ptr_dlam[0*nb0+ii+1] = ptr_t_inv[0*nb0+ii+1]*(sigma_mu - ptr_dlam[0*nb0+ii+1]*ptr_dt[0*nb0+ii+1]);
				ptr_dlam[1*nb0+ii+1] = ptr_t_inv[1*nb0+ii+1]*(sigma_mu - ptr_dlam[1*nb0+ii+1]*ptr_dt[1*nb0+ii+1]);
				ptr_qx[ii+1] += ptr_dlam[1*nb0+ii+1] - ptr_dlam[0*nb0+ii+1];

				ptr_dlam[0*nb0+ii+2] = ptr_t_inv[0*nb0+ii+2]*(sigma_mu - ptr_dlam[0*nb0+ii+2]*ptr_dt[0*nb0+ii+2]);
				ptr_dlam[1*nb0+ii+2] = ptr_t_inv[1*nb0+ii+2]*(sigma_mu - ptr_dlam[1*nb0+ii+2]*ptr_dt[1*nb0+ii+2]);
				ptr_qx[ii+2] += ptr_dlam[1*nb0+ii+2] - ptr_dlam[0*nb0+ii+2];

				ptr_dlam[0*nb0+ii+3] = ptr_t_inv[0*nb0+ii+3]*(sigma_mu - ptr_dlam[0*nb0+ii+3]*ptr_dt[0*nb0+ii+3]);
				ptr_dlam[1*nb0+ii+3] = ptr_t_inv[1*nb0+ii+3]*(sigma_mu - ptr_dlam[1*nb0+ii+3]*ptr_dt[1*nb0+ii+3]);
				ptr_qx[ii+3] += ptr_dlam[1*nb0+ii+3] - ptr_dlam[0*nb0+ii+3];
				}
			for(; ii<nb0; ii++)
				{
				ptr_dlam[0*nb0+ii+0] = ptr_t_inv[0*nb0+ii+0]*(sigma_mu - ptr_dlam[0*nb0+ii+0]*ptr_dt[0*nb0+ii+0]);
				ptr_dlam[1*nb0+ii+0] = ptr_t_inv[1*nb0+ii+0]*(sigma_mu - ptr_dlam[1*nb0+ii+0]*ptr_dt[1*nb0+ii+0]);
				ptr_qx[ii+0] += ptr_dlam[1*nb0+ii+0] - ptr_dlam[0*nb0+ii+0];
				}

			ptr_dlam  += 2*nb0;
			ptr_dt    += 2*nb0;
			ptr_t_inv += 2*nb0;
			ptr_qx    += nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			for(ii=0; ii<ng0-3; ii+=4)
				{
				ptr_dlam[0*ng0+ii+0] = ptr_t_inv[0*ng0+ii+0]*(sigma_mu - ptr_dlam[0*ng0+ii+0]*ptr_dt[0*ng0+ii+0]);
				ptr_dlam[1*ng0+ii+0] = ptr_t_inv[1*ng0+ii+0]*(sigma_mu - ptr_dlam[1*ng0+ii+0]*ptr_dt[1*ng0+ii+0]);
				ptr_qx[ii+0] += ptr_dlam[1*ng0+ii+0] - ptr_dlam[0*ng0+ii+0];

				ptr_dlam[0*ng0+ii+1] = ptr_t_inv[0*ng0+ii+1]*(sigma_mu - ptr_dlam[0*ng0+ii+1]*ptr_dt[0*ng0+ii+1]);
				ptr_dlam[1*ng0+ii+1] = ptr_t_inv[1*ng0+ii+1]*(sigma_mu - ptr_dlam[1*ng0+ii+1]*ptr_dt[1*ng0+ii+1]);
				ptr_qx[ii+1] += ptr_dlam[1*ng0+ii+1] - ptr_dlam[0*ng0+ii+1];

				ptr_dlam[0*ng0+ii+2] = ptr_t_inv[0*ng0+ii+2]*(sigma_mu - ptr_dlam[0*ng0+ii+2]*ptr_dt[0*ng0+ii+2]);
				ptr_dlam[1*ng0+ii+2] = ptr_t_inv[1*ng0+ii+2]*(sigma_mu - ptr_dlam[1*ng0+ii+2]*ptr_dt[1*ng0+ii+2]);
				ptr_qx[ii+2] += ptr_dlam[1*ng0+ii+2] - ptr_dlam[0*ng0+ii+2];

				ptr_dlam[0*ng0+ii+3] = ptr_t_inv[0*ng0+ii+3]*(sigma_mu - ptr_dlam[0*ng0+ii+3]*ptr_dt[0*ng0+ii+3]);
				ptr_dlam[1*ng0+ii+3] = ptr_t_inv[1*ng0+ii+3]*(sigma_mu - ptr_dlam[1*ng0+ii+3]*ptr_dt[1*ng0+ii+3]);
				ptr_qx[ii+3] += ptr_dlam[1*ng0+ii+3] - ptr_dlam[0*ng0+ii+3];

				}
			for(; ii<ng0; ii++)
				{
				ptr_dlam[0*ng0+ii+0] = ptr_t_inv[0*ng0+ii+0]*(sigma_mu - ptr_dlam[0*ng0+ii+0]*ptr_dt[0*ng0+ii+0]);
				ptr_dlam[1*ng0+ii+0] = ptr_t_inv[1*ng0+ii+0]*(sigma_mu - ptr_dlam[1*ng0+ii+0]*ptr_dt[1*ng0+ii+0]);
				ptr_qx[ii+0] += ptr_dlam[1*ng0+ii+0] - ptr_dlam[0*ng0+ii+0];
				}

			}

		}

	}



void d_compute_alpha_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double *ptr_alpha, struct d_strvec *hst, struct d_strvec *hsdt, struct d_strvec *hslam, struct d_strvec *hsdlam, struct d_strvec *hslamt, struct d_strvec *hsdux, struct d_strmat *hsDCt, struct d_strvec *hsdb)
	{
	
	int ii, jj, ll;

	__m256
		t_sign, t_ones, t_zeros,
		t_mask0, t_mask1,
		t_lam, t_dlam, t_t, t_dt,
		t_tmp0, t_tmp1,
		t_alpha0, t_alpha1;

	__m128
		s_sign, s_ones, s_mask, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha0, s_alpha1;
	
	__m256d
		v_sign, v_alpha, v_mask, v_left,
		v_temp0, v_dt0, v_dux, v_db0, v_dlam0, v_lamt0, v_t0, v_lam0,
		v_temp1, v_dt1, v_db1, v_dlam1, v_lamt1, v_t1, v_lam1;
	
	__m128d
		u_sign, u_dux, u_alpha,
		u_dt0, u_temp0, u_db0, u_dlam0, u_lamt0, u_t0, u_lam0,
		u_dt1, u_temp1, u_db1, u_dlam1, u_lamt1, u_t1, u_lam1;
	
	__m256i
		i_mask;
	
	int nu0, nx0, nb0, ng0, cng;

	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha0 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_zeros = _mm256_setzero_ps( );

	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	
	double ll_left;

	static double d_mask[4]  = {0.5, 1.5, 2.5, 3.5};

	static double dux_tmp[4] = {};

	double alpha = ptr_alpha[0];

	double
		*ptr_db, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lamt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	for(jj=0; jj<=N; jj++)
		{

		ptr_db   = hsdb[jj].pa;
		ptr_dux  = hsdux[jj].pa;
		ptr_t    = hst[jj].pa;
		ptr_dt   = hsdt[jj].pa;
		ptr_lamt = hslamt[jj].pa;
		ptr_lam  = hslam[jj].pa;
		ptr_dlam = hsdlam[jj].pa;
		ptr_idxb = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ll=0; ll<nb0-3; ll+=4)
				{
				//v_dux   = _mm256_loadu_pd( &ptr_dux[ll] );
				u_temp0 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
				u_temp1 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
				u_temp0 = _mm_loadh_pd( u_temp0, &ptr_dux[ptr_idxb[ll+1]] );
				u_temp1 = _mm_loadh_pd( u_temp1, &ptr_dux[ptr_idxb[ll+3]] );
				v_dux   = _mm256_castpd128_pd256( u_temp0 );
				v_dux   = _mm256_insertf128_pd( v_dux, u_temp1, 0x1 );
				v_db0   = _mm256_loadu_pd( &ptr_db[0*nb0+ll] );
				v_db1   = _mm256_loadu_pd( &ptr_db[1*nb0+ll] );
				v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
				v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
				v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_storeu_pd( &ptr_dt[0*nb0+ll], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[1*nb0+ll], v_dt1 );

				v_lamt0 = _mm256_loadu_pd( &ptr_lamt[0*nb0+ll] );
				v_lamt1 = _mm256_loadu_pd( &ptr_lamt[1*nb0+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
				v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_storeu_pd( &ptr_dlam[0*nb0+ll], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[1*nb0+ll], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<nb0)
				{

				ll_left = nb0 - ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				u_temp0 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
				if(ll_left>1) u_temp0 = _mm_loadh_pd( u_temp0, &ptr_dux[ptr_idxb[ll+1]] );
				if(ll_left>2) u_temp1 = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
				//u_temp1 = _mm_loadh_pd( u_temp1, &ptr_dux[ptr_idxb[ll+3]] );
				v_dux   = _mm256_castpd128_pd256( u_temp0 );
				v_dux   = _mm256_insertf128_pd( v_dux, u_temp1, 0x1 );
				v_db0   = _mm256_loadu_pd( &ptr_db[0*nb0+ll] );
				v_db1   = _mm256_loadu_pd( &ptr_db[1*nb0+ll] );
				v_dt0   = _mm256_sub_pd ( v_dux, v_db0 );
				v_dt1   = _mm256_sub_pd ( v_db1, v_dux );
				v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
				v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_maskstore_pd( &ptr_dt[0*nb0+ll], i_mask, v_dt0 );
				_mm256_maskstore_pd( &ptr_dt[1*nb0+ll], i_mask, v_dt1 );

				v_lamt0 = _mm256_loadu_pd( &ptr_lamt[0*nb0+ll] );
				v_lamt1 = _mm256_loadu_pd( &ptr_lamt[1*nb0+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
				v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_maskstore_pd( &ptr_dlam[0*nb0+ll], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[1*nb0+ll], i_mask, v_dlam1 );

				if(ll<nb0-2) // 3 left
					{

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}
				else // 1 or 2 left
					{

					s_mask   = _mm256_cvtpd_ps( v_mask );
					s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
					t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
					t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

					v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
					v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
					v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
					v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				}

			ptr_db   += 2*nb0;
			ptr_t    += 2*nb0;
			ptr_dt   += 2*nb0;
			ptr_lamt += 2*nb0;
			ptr_lam  += 2*nb0;
			ptr_dlam += 2*nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];

			dgemv_t_libstr(nx0+nu0, ng0, 1.0, &hsDCt[jj], 0, 0, &hsdux[jj], 0, 0.0, &hsdt[jj], 0, &hsdt[jj], 0);

			for(ll=0; ll<ng0-3; ll+=4)
				{
				v_dt0   = _mm256_loadu_pd( &ptr_dt[0*ng0+ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_loadu_pd( &ptr_db[0*ng0+ll] );
				v_db1   = _mm256_loadu_pd( &ptr_db[1*ng0+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_add_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_loadu_pd( &ptr_t[0*ng0+ll] );
				v_t1    = _mm256_loadu_pd( &ptr_t[1*ng0+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_storeu_pd( &ptr_dt[0*ng0+ll], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[1*ng0+ll], v_dt1 );

				v_lamt0 = _mm256_loadu_pd( &ptr_lamt[0*ng0+ll] );
				v_lamt1 = _mm256_loadu_pd( &ptr_lamt[1*ng0+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*ng0+ll] );
				v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*ng0+ll] );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ll] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_storeu_pd( &ptr_dlam[0*ng0+ll], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[1*ng0+ll], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<ng0)
				{

				ll_left = ng0 - ll;
				v_left  = _mm256_broadcast_sd( &ll_left );
				v_mask  = _mm256_loadu_pd( d_mask );
				v_mask  = _mm256_sub_pd( v_mask, v_left );
				i_mask  = _mm256_castpd_si256( v_mask );

				v_dt0   = _mm256_loadu_pd( &ptr_dt[0*ng0+ll] );
				v_dt1   = _mm256_xor_pd( v_dt0, v_sign );
				v_db0   = _mm256_loadu_pd( &ptr_db[0*ng0+ll] );
				v_db1   = _mm256_loadu_pd( &ptr_db[1*ng0+ll] );
				v_dt0   = _mm256_sub_pd ( v_dt0, v_db0 );
				v_dt1   = _mm256_add_pd ( v_dt1, v_db1 );
				v_t0    = _mm256_loadu_pd( &ptr_t[0*ng0+ll] );
				v_t1    = _mm256_loadu_pd( &ptr_t[1*ng0+ll] );
				v_dt0   = _mm256_sub_pd( v_dt0, v_t0 );
				v_dt1   = _mm256_sub_pd( v_dt1, v_t1 );
				_mm256_maskstore_pd( &ptr_dt[0*ng0+ll], i_mask, v_dt0 );
				_mm256_maskstore_pd( &ptr_dt[1*ng0+ll], i_mask, v_dt1 );

				v_lamt0 = _mm256_loadu_pd( &ptr_lamt[0*ng0+ll] );
				v_lamt1 = _mm256_loadu_pd( &ptr_lamt[1*ng0+ll] );
				v_temp0 = _mm256_mul_pd( v_lamt0, v_dt0 );
				v_temp1 = _mm256_mul_pd( v_lamt1, v_dt1 );
				v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*ng0+ll] );
				v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*ng0+ll] );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ll] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ll] );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_lam0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_lam1 );
				v_dlam0 = _mm256_sub_pd( v_dlam0, v_temp0 );
				v_dlam1 = _mm256_sub_pd( v_dlam1, v_temp1 );
				_mm256_maskstore_pd( &ptr_dlam[0*ng0+ll], i_mask, v_dlam0 );
				_mm256_maskstore_pd( &ptr_dlam[1*ng0+ll], i_mask, v_dlam1 );

				if(ll<ng0-2) // 3 left
					{

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}
				else // 1 or 2 left
					{

					s_mask   = _mm256_cvtpd_ps( v_mask );
					s_mask   = _mm_shuffle_ps( s_mask, s_mask, 0x44 );
					t_mask1  = _mm256_permute2f128_ps( _mm256_castps128_ps256( s_mask ), _mm256_castps128_ps256( s_mask ), 0x20 );
					t_mask1  = _mm256_cmp_ps( t_mask1, t_zeros, 0x01 );

					v_dt0    = _mm256_permute2f128_pd( v_dt0, v_dt1, 0x20 );
					v_t0     = _mm256_permute2f128_pd( v_t0, v_t1, 0x20 );
					v_dlam0  = _mm256_permute2f128_pd( v_dlam0, v_dlam1, 0x20 );
					v_lam0   = _mm256_permute2f128_pd( v_lam0, v_lam1, 0x20 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_mask0  = _mm256_and_ps( t_mask0, t_mask1 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				}
			}

		}		

	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha0 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha1 = _mm256_extractf128_ps( t_alpha0, 0x1 );
//	s_alpha0  = _mm_min_ps( s_alpha0 , s_alpha1 );
	s_alpha1 = _mm256_castps256_ps128( t_alpha0 );
	s_alpha0 = _mm_min_ps( s_alpha0, s_alpha1 );
	
	v_alpha = _mm256_cvtps_pd( s_alpha0 );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_update_var_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, struct d_strvec *hsux, struct d_strvec *hsdux, struct d_strvec *hst, struct d_strvec *hsdt, struct d_strvec *hslam, struct d_strvec *hsdlam, struct d_strvec *hspi, struct d_strvec *hsdpi)
	{
	
	int ii;

	int nu0, nx0, nx1, nb0, ng0;

	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_mu0, u_tmp;

	__m256d
		v_mask, v_left, v_zeros,
		v_alpha, v_ux, v_dux, v_pi, v_dpi, 
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0,
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	__m256i
		i_mask;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	double
		*ptr_pi, *ptr_dpi, *ptr_ux, *ptr_dux, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	for(jj=1; jj<=N; jj++)
		{

		nx0 = nx[jj];

		ptr_pi   = hspi[jj].pa;
		ptr_dpi  = hsdpi[jj].pa;

		// equality constraints lagrange multipliers
		ll = 0;
		for(; ll<nx0-3; ll+=4)
			{
			v_pi  = _mm256_loadu_pd( &ptr_pi[ll] );
			v_dpi = _mm256_loadu_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_storeu_pd( &ptr_pi[ll], v_pi );
			}
		if(ll<nx0)
			{
			ll_left = nx0-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_pi  = _mm256_loadu_pd( &ptr_pi[ll] );
			v_dpi = _mm256_loadu_pd( &ptr_dpi[ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
#if defined(TARGET_X64_AVX2)
			v_pi  = _mm256_fmadd_pd( v_alpha, v_dpi, v_pi );
#else
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
#endif
			_mm256_maskstore_pd( &ptr_pi[ll], i_mask, v_pi );
			}

		}

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		ng0 = ng[jj];
		
		ptr_ux   = hsux[jj].pa;
		ptr_dux  = hsdux[jj].pa;
		ptr_t    = hst[jj].pa;
		ptr_dt   = hsdt[jj].pa;
		ptr_lam  = hslam[jj].pa;
		ptr_dlam = hsdlam[jj].pa;

		// inputs and states
		for(ll=0; ll<nu0+nx0-3; ll+=4)
			{
			v_ux  = _mm256_loadu_pd( &ptr_ux[ll] );
			v_dux = _mm256_loadu_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_storeu_pd( &ptr_ux[ll], v_ux );
			}
		if(ll<nu0+nx0)
			{
			ll_left = nu0+nx0-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );

			v_ux  = _mm256_loadu_pd( &ptr_ux[ll] );
			v_dux = _mm256_loadu_pd( &ptr_dux[ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
#if defined(TARGET_X64_AVX2)
			v_ux    = _mm256_fmadd_pd( v_alpha, v_dux, v_ux );
#else
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
#endif
			_mm256_maskstore_pd( &ptr_ux[ll], i_mask, v_ux );
			}

		// box constraints
		for(ll=0; ll<nb0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_storeu_pd( &ptr_t[0*nb0+ll], v_t0 );
			_mm256_storeu_pd( &ptr_t[1*nb0+ll], v_t1 );
			_mm256_storeu_pd( &ptr_lam[0*nb0+ll], v_lam0 );
			_mm256_storeu_pd( &ptr_lam[1*nb0+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb0)
			{
			ll_left = nb0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[0*nb0+ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[1*nb0+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[0*nb0+ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[1*nb0+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		ptr_t    += 2*nb0;
		ptr_dt   += 2*nb0;
		ptr_lam  += 2*nb0;
		ptr_dlam += 2*nb0;

		// genreal constraints
		for(ll=0; ll<ng0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[0*ng0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*ng0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*ng0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_storeu_pd( &ptr_t[0*ng0+ll], v_t0 );
			_mm256_storeu_pd( &ptr_t[1*ng0+ll], v_t1 );
			_mm256_storeu_pd( &ptr_lam[0*ng0+ll], v_lam0 );
			_mm256_storeu_pd( &ptr_lam[1*ng0+ll], v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<ng0)
			{

			ll_left = ng0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );
			i_mask  = _mm256_castpd_si256( v_mask );

			v_t0    = _mm256_loadu_pd( &ptr_t[0*ng0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*ng0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*ng0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*ng0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
#endif
			_mm256_maskstore_pd( &ptr_t[0*ng0+ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[1*ng0+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[0*ng0+ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[1*ng0+ll], i_mask, v_lam1 );
#if defined(TARGET_X64_AVX2)
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif

			}

		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );

	return;
	
	}



void d_compute_mu_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, struct d_strvec *hslam, struct d_strvec *hsdlam, struct d_strvec *hst, struct d_strvec *hsdt)
	{

	int ii;
	
	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_mu0, u_tmp;

	__m256d
		v_alpha, v_mask, v_left, v_zeros,
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0, 
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;

	int nb0, ng0;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t    = hst[jj].pa;
		ptr_lam  = hslam[jj].pa;
		ptr_dt   = hsdt[jj].pa;
		ptr_dlam = hsdlam[jj].pa;

		// box constraints
		nb0 = nb[jj];
		for(ll=0; ll<nb0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb0)
			{
			ll_left = nb0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		ptr_t    += 2*nb0;
		ptr_lam  += 2*nb0;
		ptr_dt   += 2*nb0;
		ptr_dlam += 2*nb0;

		// general constraints
		ng0 = ng[jj];
		for(ll=0; ll<ng0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<ng0)
			{
			ll_left = ng0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_loadu_pd( &ptr_t[ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#else
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );

	return;

	}



void d_update_gradient_new_rhs_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double **db, double **t_inv, double **lamt, double **qx)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, ng0;
	
	double temp0, temp1;
	
	double 
		*ptr_db, *ptr_qx,
		*ptr_t_inv, *ptr_lamt;
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t_inv = t_inv[jj];
		ptr_lamt  = lamt[jj];
		ptr_db    = db[jj];
		ptr_qx    = qx[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ii=0; ii<nb0-3; ii+=4)
				{

				ptr_qx[ii+0] = - ptr_lamt[ii+nb0+0]*ptr_db[ii+nb0+0] - ptr_lamt[ii+0]*ptr_db[ii+0];

				ptr_qx[ii+1] = - ptr_lamt[ii+nb0+1]*ptr_db[ii+nb0+1] - ptr_lamt[ii+1]*ptr_db[ii+1];

				ptr_qx[ii+2] = - ptr_lamt[ii+nb0+2]*ptr_db[ii+nb0+2] - ptr_lamt[ii+2]*ptr_db[ii+2];

				ptr_qx[ii+3] = - ptr_lamt[ii+nb0+3]*ptr_db[ii+nb0+3] - ptr_lamt[ii+3]*ptr_db[ii+3];

				}
			for(; ii<nb0; ii++)
				{

				ptr_qx[ii+0] = - ptr_lamt[ii+nb0+0]*ptr_db[ii+nb0+0] - ptr_lamt[ii+0]*ptr_db[ii+0];

				}

			ptr_t_inv += 2*nb0;
			ptr_lamt  += 2*nb0;
			ptr_db    += 2*nb0;
			ptr_qx    += nb0;

			} // end of if nb0>0

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_qx[ii+0] = - ptr_lamt[ii+ng0+0]*ptr_db[ii+ng0+0] - ptr_lamt[ii+0]*ptr_db[ii+0];

				ptr_qx[ii+1] = - ptr_lamt[ii+ng0+1]*ptr_db[ii+ng0+1] - ptr_lamt[ii+1]*ptr_db[ii+1];

				ptr_qx[ii+2] = - ptr_lamt[ii+ng0+2]*ptr_db[ii+ng0+2] - ptr_lamt[ii+2]*ptr_db[ii+2];

				ptr_qx[ii+3] = - ptr_lamt[ii+ng0+3]*ptr_db[ii+ng0+3] - ptr_lamt[ii+3]*ptr_db[ii+3];

				}
			for(; ii<ng0; ii++)
				{

				ptr_qx[ii+0] = - ptr_lamt[ii+ng0+0]*ptr_db[ii+ng0+0] - ptr_lamt[ii+0]*ptr_db[ii+0];

				}

			} // end of if ng0>0

		} // end of jj loop over N

	}



void d_compute_t_lam_new_rhs_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **t_aff, double **lam_aff, double **lamt, double **tinv, double **dux, double **pDCt, double **db)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, ng0, cng;

	double
		*ptr_db, *ptr_dux, *ptr_t_aff, *ptr_lam_aff, *ptr_lamt, *ptr_tinv;
	
	int
		*ptr_idxb;
	
	int jj, ll;

	for(jj=0; jj<=N; jj++)
		{

		ptr_db      = db[jj];
		ptr_dux     = dux[jj];
		ptr_t_aff   = t_aff[jj];
		ptr_lam_aff = lam_aff[jj];
		ptr_lamt    = lamt[jj];
		ptr_tinv    = tinv[jj];
		ptr_idxb    = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			// box constraints
			for(ll=0; ll<nb0; ll++)
				{

				ptr_t_aff[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_db[ll+0];
				ptr_t_aff[ll+nb0] = - ptr_dux[ptr_idxb[ll]] + ptr_db[ll+nb0];
				ptr_lam_aff[ll+0]   = - ptr_lamt[ll+0]   * ptr_t_aff[ll+0];
				ptr_lam_aff[ll+nb0] = - ptr_lamt[ll+nb0] * ptr_t_aff[ll+nb0];
				}

			ptr_db      += 2*nb0;
			ptr_t_aff   += 2*nb0;
			ptr_lam_aff += 2*nb0;
			ptr_lamt    += 2*nb0;
			ptr_tinv    += 2*nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			cng = (ng0+ncl-1)/ncl*ncl;

#ifdef BLASFEO
			dgemv_t_lib(nx0+nu0, ng0, 1.0, pDCt[jj], cng, ptr_dux, 0.0, ptr_t_aff, ptr_t_aff);
#else
			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_t_aff, ptr_t_aff);
#endif

			for(ll=0; ll<ng0; ll++)
				{
				ptr_t_aff[ll+ng0] = - ptr_t_aff[ll+0];
				ptr_t_aff[ll+0]   -= ptr_db[ll+0];
				ptr_t_aff[ll+ng0] += ptr_db[ll+ng0];
				ptr_lam_aff[ll+0]   = - ptr_lamt[ll+0]   * ptr_t_aff[ll+0];
				ptr_lam_aff[ll+ng0] = - ptr_lamt[ll+ng0] * ptr_t_aff[ll+ng0];
				}

			}

		}		

	return;
	
	}



// IPM with residuals

void d_update_hessian_gradient_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, struct d_strvec *hsres_d, struct d_strvec *hsres_m, struct d_strvec *hst, struct d_strvec *hslam, struct d_strvec *hstinv, struct d_strvec *hsQx, struct d_strvec *hsqx)
	{
	
	int ii, jj, bs0;
	
	int nb0, ng0;
	
	double 
		*ptr_res_d, *ptr_Qx, *ptr_qx, *ptr_t, *ptr_lam, *ptr_res_m, *ptr_t_inv;
	
	__m256d
		v_ones,
		v_tmp0, v_tinv0, v_lam0, v_resm0, v_resd0,
		v_tmp1, v_tinv1, v_lam1, v_resm1, v_resd1;
	
	__m256i
		i_mask;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double ii_left;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t     = hst[jj].pa;
		ptr_lam   = hslam[jj].pa;
		ptr_t_inv = hstinv[jj].pa;
		ptr_res_d = hsres_d[jj].pa;
		ptr_res_m = hsres_m[jj].pa;
		ptr_Qx    = hsQx[jj].pa;
		ptr_qx    = hsqx[jj].pa;

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ii=0; ii<nb0-3; ii+=4)
				{

				v_tinv0 = _mm256_loadu_pd( &ptr_t[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t[ii+nb0] );
				v_tinv0 = _mm256_div_pd( v_ones, v_tinv0 );
				v_tinv1 = _mm256_div_pd( v_ones, v_tinv1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+nb0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+nb0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+nb0] );
				v_tmp0  = _mm256_mul_pd( v_tinv0, v_lam0 );
				v_tmp1  = _mm256_mul_pd( v_tinv1, v_lam1 );
				_mm256_storeu_pd( &ptr_t_inv[ii+0], v_tinv0 );
				_mm256_storeu_pd( &ptr_t_inv[ii+nb0], v_tinv1 );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_Qx[ii+0], v_tmp0 );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_qx[ii+0], v_tmp0 );

				}
			if(ii<nb0)
				{

				ii_left = nb0-ii;
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

				v_tinv0 = _mm256_loadu_pd( &ptr_t[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t[ii+nb0] );
				v_tinv0 = _mm256_div_pd( v_ones, v_tinv0 );
				v_tinv1 = _mm256_div_pd( v_ones, v_tinv1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+nb0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+nb0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+nb0] );
				v_tmp0  = _mm256_mul_pd( v_tinv0, v_lam0 );
				v_tmp1  = _mm256_mul_pd( v_tinv1, v_lam1 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tinv0 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+nb0], i_mask, v_tinv1 );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_Qx[ii+0], i_mask, v_tmp0 );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_tmp0 );

				}

			ptr_t     += 2*nb0;
			ptr_lam   += 2*nb0;
			ptr_t_inv += 2*nb0;
			ptr_res_d += 2*nb0;
			ptr_res_m += 2*nb0;
			ptr_Qx    += nb0;
			ptr_qx    += nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			for(ii=0; ii<ng0-3; ii+=4)
				{

				v_tinv0 = _mm256_loadu_pd( &ptr_t[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t[ii+ng0] );
				v_tinv0 = _mm256_div_pd( v_ones, v_tinv0 );
				v_tinv1 = _mm256_div_pd( v_ones, v_tinv1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+ng0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+ng0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+ng0] );
				v_tmp0  = _mm256_mul_pd( v_tinv0, v_lam0 );
				v_tmp1  = _mm256_mul_pd( v_tinv1, v_lam1 );
				_mm256_storeu_pd( &ptr_t_inv[ii+0], v_tinv0 );
				_mm256_storeu_pd( &ptr_t_inv[ii+ng0], v_tinv1 );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_Qx[ii+0], v_tmp0 );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_qx[ii+0], v_tmp0 );

				}
			if(ii<ng0)
				{

				ii_left = ng0-ii;
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

				v_tinv0 = _mm256_loadu_pd( &ptr_t[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t[ii+ng0] );
				v_tinv0 = _mm256_div_pd( v_ones, v_tinv0 );
				v_tinv1 = _mm256_div_pd( v_ones, v_tinv1 );
				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+ng0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+ng0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+ng0] );
				v_tmp0  = _mm256_mul_pd( v_tinv0, v_lam0 );
				v_tmp1  = _mm256_mul_pd( v_tinv1, v_lam1 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+0], i_mask, v_tinv0 );
				_mm256_maskstore_pd( &ptr_t_inv[ii+ng0], i_mask, v_tinv1 );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_Qx[ii+0], i_mask, v_tmp0 );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_tmp0 );

				}

			}

		}

	}



void d_compute_alpha_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strvec *hsdux, struct d_strvec *hst, struct d_strvec *hstinv, struct d_strvec *hslam, struct d_strmat *hsDCt, struct d_strvec *hsres_d, struct d_strvec *hsres_m, struct d_strvec *hsdt, struct d_strvec *hsdlam, double *ptr_alpha)
	{
	
	int ii, jj, ll;

	int nu0, nx0, nb0, ng0, cng;

	double alpha = ptr_alpha[0];
	
	double
		*ptr_res_d, *ptr_res_m, *ptr_dux, *ptr_t, *ptr_t_inv, *ptr_dt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	__m128d
		u_dux, u_alpha,
		u_resm0, u_resd0, u_dt0, u_dlam0, u_tmp0, u_tinv0, u_lam0, u_t0,
		u_resm1, u_resd1, u_dt1, u_dlam1, u_tmp1, u_tinv1, u_lam1, u_t1;
	
	__m256d
		v_dux, v_sign, v_alpha,
		v_resm0, v_resd0, v_dt0, v_dlam0, v_tmp0, v_tinv0, v_lam0, v_t0,
		v_resm1, v_resd1, v_dt1, v_dlam1, v_tmp1, v_tinv1, v_lam1, v_t1;
	
	__m128
		s_dlam, s_lam, s_mask0, s_tmp0,
		s_alpha0,
		s_alpha1;

	__m256
		t_dlam, t_dt, t_lam, t_t, t_sign, t_ones, t_zeros,
		t_mask0, t_tmp0, t_alpha0,
		t_mask1, t_tmp1, t_alpha1;
	
	__m256i
		i_mask;

	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );

	int int_sign = 0x80000000;
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );

	t_zeros = _mm256_setzero_ps( );

	// initialize alpha with 1.0
	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );

	for(jj=0; jj<=N; jj++)
		{

		ptr_res_d = hsres_d[jj].pa;
		ptr_res_m = hsres_m[jj].pa;
		ptr_dux   = hsdux[jj].pa;
		ptr_t     = hst[jj].pa;
		ptr_t_inv = hstinv[jj].pa;
		ptr_dt    = hsdt[jj].pa;
		ptr_lam   = hslam[jj].pa;
		ptr_dlam  = hsdlam[jj].pa;
		ptr_idxb  = idxb[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			// box constraints
			ll = 0;
#if 1
			for(; ll<nb0-3; ll+=4)
				{

				u_tmp0  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
				u_tmp1  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
				u_tmp0  = _mm_loadh_pd( u_tmp0, &ptr_dux[ptr_idxb[ll+1]] );
				u_tmp1  = _mm_loadh_pd( u_tmp1, &ptr_dux[ptr_idxb[ll+3]] );
				v_dux   = _mm256_castpd128_pd256( u_tmp0 );
				v_dux   = _mm256_insertf128_pd( v_dux, u_tmp1, 0x1 );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+nb0] );
				v_dt0   = _mm256_sub_pd( v_dux, v_resd0 );
				v_dt1   = _mm256_sub_pd( v_resd1, v_dux );
				_mm256_storeu_pd( &ptr_dt[ll+0], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[ll+nb0], v_dt1 );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+nb0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+nb0] );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
				v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+nb0] );
				v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
				v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
				v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
				v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
				_mm256_storeu_pd( &ptr_dlam[ll+0], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[ll+nb0], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				v_t0  = _mm256_loadu_pd( &ptr_t[ll+0] );
				v_t1  = _mm256_loadu_pd( &ptr_t[ll+nb0] );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<nb0)
				{
				
				if(nb0-ll==1)
					{

					u_dux    = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
					u_resd0  = _mm_load_sd( &ptr_res_d[ll+0] );
					u_resd1  = _mm_load_sd( &ptr_res_d[ll+nb0] );
					u_dt0    = _mm_sub_pd( u_dux, u_resd0 );
					u_dt1    = _mm_sub_pd( u_resd1, u_dux );
					_mm_store_sd( &ptr_dt[ll+0], u_dt0 );
					_mm_store_sd( &ptr_dt[ll+nb0], u_dt1 );

					u_lam0   = _mm_load_sd( &ptr_lam[ll+0] );
					u_lam1   = _mm_load_sd( &ptr_lam[ll+nb0] );
					u_tmp0   = _mm_mul_pd( u_lam0, u_dt0 );
					u_tmp1   = _mm_mul_pd( u_lam1, u_dt1 );
					u_resm0  = _mm_load_sd( &ptr_res_m[ll+0] );
					u_resm1  = _mm_load_sd( &ptr_res_m[ll+nb0] );
					u_tmp0   = _mm_add_pd( u_tmp0, u_resm0 );
					u_tmp1   = _mm_add_pd( u_tmp1, u_resm1 );
					u_tinv0  = _mm_load_sd( &ptr_t_inv[ll+0] );
					u_tinv1  = _mm_load_sd( &ptr_t_inv[ll+nb0] );
					u_tinv0  = _mm_xor_pd( u_tinv0, _mm256_castpd256_pd128( v_sign ) );
					u_tinv1  = _mm_xor_pd( u_tinv1, _mm256_castpd256_pd128( v_sign ) );
					u_dlam0  = _mm_mul_pd( u_tinv0, u_tmp0 );
					u_dlam1  = _mm_mul_pd( u_tinv1, u_tmp1 );
					_mm_store_sd( &ptr_dlam[ll+0], u_dlam0 );
					_mm_store_sd( &ptr_dlam[ll+nb0], u_dlam1 );

					u_dt1    = _mm_movedup_pd( u_dt1 );
					u_dt0    = _mm_move_sd( u_dt1, u_dt0 );
					u_t1     = _mm_loaddup_pd( &ptr_t[ll+nb0] );
					u_t0     = _mm_load_sd( &ptr_t[ll+0] );
					u_t0     = _mm_move_sd( u_t1, u_t0 );
					u_dlam1  = _mm_movedup_pd( u_dlam1 );
					u_dlam0  = _mm_move_sd( u_dlam1, u_dlam0 );
					u_lam1   = _mm_movedup_pd( u_lam1 );
					u_lam0   = _mm_move_sd( u_lam1, u_lam0 );

					v_dlam0  = _mm256_castpd128_pd256( u_dlam0 );
					v_dlam0  = _mm256_insertf128_pd( v_dlam0, u_dt0, 0x1 );
					v_lam0   = _mm256_castpd128_pd256( u_lam0 );
					v_lam0   = _mm256_insertf128_pd( v_lam0, u_t0, 0x1 );

					s_dlam   = _mm256_cvtpd_ps( v_dlam0 );
					s_lam    = _mm256_cvtpd_ps( v_lam0 );
					s_mask0  = _mm_cmp_ps( s_dlam, _mm256_castps256_ps128( t_zeros ), 0x01 );
					s_lam    = _mm_xor_ps( s_lam, _mm256_castps256_ps128( t_sign ) );
					s_tmp0   = _mm_div_ps( s_lam, s_dlam );
					s_tmp0   = _mm_blendv_ps( _mm256_castps256_ps128( t_ones ), s_tmp0, s_mask0 );
					t_tmp0   = _mm256_blend_ps( t_ones, _mm256_castps128_ps256( s_tmp0 ), 0xf );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				else if(nb0-ll==2)
					{

					u_dux    = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
					u_dux    = _mm_loadh_pd( u_dux, &ptr_dux[ptr_idxb[ll+1]] );
					u_resd0  = _mm_loadu_pd( &ptr_res_d[ll+0] );
					u_resd1  = _mm_loadu_pd( &ptr_res_d[ll+nb0] );
					u_dt0    = _mm_sub_pd( u_dux, u_resd0 );
					u_dt1    = _mm_sub_pd( u_resd1, u_dux );
					_mm_storeu_pd( &ptr_dt[ll+0], u_dt0 );
					_mm_storeu_pd( &ptr_dt[ll+nb0], u_dt1 );

					u_lam0   = _mm_loadu_pd( &ptr_lam[ll+0] );
					u_lam1   = _mm_loadu_pd( &ptr_lam[ll+nb0] );
					u_tmp0   = _mm_mul_pd( u_lam0, u_dt0 );
					u_tmp1   = _mm_mul_pd( u_lam1, u_dt1 );
					u_resm0  = _mm_loadu_pd( &ptr_res_m[ll+0] );
					u_resm1  = _mm_loadu_pd( &ptr_res_m[ll+nb0] );
					u_tmp0   = _mm_add_pd( u_tmp0, u_resm0 );
					u_tmp1   = _mm_add_pd( u_tmp1, u_resm1 );
					u_tinv0  = _mm_loadu_pd( &ptr_t_inv[ll+0] );
					u_tinv1  = _mm_loadu_pd( &ptr_t_inv[ll+nb0] );
					u_tinv0  = _mm_xor_pd( u_tinv0, _mm256_castpd256_pd128( v_sign ) );
					u_tinv1  = _mm_xor_pd( u_tinv1, _mm256_castpd256_pd128( v_sign ) );
					u_dlam0  = _mm_mul_pd( u_tinv0, u_tmp0 );
					u_dlam1  = _mm_mul_pd( u_tinv1, u_tmp1 );
					_mm_storeu_pd( &ptr_dlam[ll+0], u_dlam0 );
					_mm_storeu_pd( &ptr_dlam[ll+nb0], u_dlam1 );

					v_dt0    = _mm256_castpd128_pd256( u_dt0 );
					v_dt0    = _mm256_insertf128_pd( v_dt0, u_dt1, 0x1 );
					v_t0     = _mm256_castpd128_pd256( _mm_loadu_pd( &ptr_t[ll+0] ) );
					v_t0     = _mm256_insertf128_pd( v_t0, _mm_loadu_pd( &ptr_t[ll+nb0]), 0x1 );
					v_dlam0  = _mm256_castpd128_pd256( u_dlam0 );
					v_dlam0  = _mm256_insertf128_pd( v_dlam0, u_dlam1, 0x1 );
					v_lam0   = _mm256_castpd128_pd256( u_lam0 );
					v_lam0   = _mm256_insertf128_pd( v_lam0, u_lam1, 0x1 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				else // if(nb-ll==3)
					{

					i_mask = _mm256_castpd_si256( _mm256_set_pd( 1.0, -1.0, -1.0, -1.0 ) );

					u_tmp0  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
					u_tmp1  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
					u_tmp0  = _mm_loadh_pd( u_tmp0, &ptr_dux[ptr_idxb[ll+1]] );
					v_dux   = _mm256_castpd128_pd256( u_tmp0 );
					v_dux   = _mm256_insertf128_pd( v_dux, u_tmp1, 0x1 );
					v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
					v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+nb0] );
					v_dt0   = _mm256_sub_pd( v_dux, v_resd0 );
					v_dt1   = _mm256_sub_pd( v_resd1, v_dux );
					_mm256_maskstore_pd( &ptr_dt[ll+0], i_mask, v_dt0 );
					_mm256_maskstore_pd( &ptr_dt[ll+nb0], i_mask, v_dt1 );

					v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
					v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+nb0] );
					v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
					v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
					v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
					v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+nb0] );
					v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
					v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
					v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
					v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+nb0] );
					v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
					v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
					v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
					v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
					_mm256_maskstore_pd( &ptr_dlam[ll+0], i_mask, v_dlam0 );
					_mm256_maskstore_pd( &ptr_dlam[ll+nb0], i_mask, v_dlam1 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					v_t0  = _mm256_loadu_pd( &ptr_t[ll+0] );
					v_t1  = _mm256_loadu_pd( &ptr_t[ll+nb0] );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}

				}

#else
			for(; ll<nb0; ll++)
				{
				
				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_res_d[ll+0];
				ptr_dt[ll+nb0] = - ptr_dux[ptr_idxb[ll]] + ptr_res_d[ll+nb0];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+nb0] = - ptr_t_inv[ll+nb0] * ( ptr_lam[ll+nb0]*ptr_dt[ll+nb0] + ptr_res_m[ll+nb0] );

				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+nb0]>ptr_lam[ll+nb0] )
					{
					alpha = - ptr_lam[ll+nb0] / ptr_dlam[ll+nb0];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+nb0]>ptr_t[ll+nb0] )
					{
					alpha = - ptr_t[ll+nb0] / ptr_dt[ll+nb0];
					}

				}
#endif

			ptr_res_d += 2*nb0;
			ptr_res_m += 2*nb0;
			ptr_t     += 2*nb0;
			ptr_t_inv += 2*nb0;
			ptr_dt    += 2*nb0;
			ptr_lam   += 2*nb0;
			ptr_dlam  += 2*nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];

			dgemv_t_libstr(nx0+nu0, ng0, 1.0, &hsDCt[jj], 0, 0, &hsdux[jj], 0, 0.0, &hsdt[jj], 0, &hsdt[jj], 0);

			ll = 0;
#if 1
			for(; ll<ng0-3; ll+=4)
				{

				v_tmp0  = _mm256_loadu_pd( &ptr_dt[ll+0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+ng0] );
				v_dt0   = _mm256_sub_pd( v_tmp0, v_resd0 );
				v_dt1   = _mm256_sub_pd( v_resd1, v_tmp0 );
				_mm256_storeu_pd( &ptr_dt[ll+0], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[ll+ng0], v_dt1 );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+ng0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+ng0] );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
				v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+ng0] );
				v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
				v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
				v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
				v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
				_mm256_storeu_pd( &ptr_dlam[ll+0], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[ll+ng0], v_dlam1 );

				t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
				t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
				t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
				t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
				v_t0  = _mm256_loadu_pd( &ptr_t[ll+0] );
				v_t1  = _mm256_loadu_pd( &ptr_t[ll+ng0] );
				t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
				t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
				t_lam    = _mm256_xor_ps( t_lam, t_sign );
				t_t      = _mm256_xor_ps( t_t, t_sign );
				t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
				t_tmp1   = _mm256_div_ps( t_t, t_dt );
				t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
				t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
				t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
				t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

				}
			if(ll<ng0)
				{
				
				if(ng0-ll==1)
					{

					u_tmp0  = _mm_loadu_pd( &ptr_dt[ll+0] );
					u_resd0 = _mm_loadu_pd( &ptr_res_d[ll+0] );
					u_resd1 = _mm_loadu_pd( &ptr_res_d[ll+ng0] );
					u_dt0   = _mm_sub_pd( u_tmp0, u_resd0 );
					u_dt1   = _mm_sub_pd( u_resd1, u_tmp0 );
					_mm_store_sd( &ptr_dt[ll+0], u_dt0 );
					_mm_store_sd( &ptr_dt[ll+ng0], u_dt1 );

					u_lam0   = _mm_load_sd( &ptr_lam[ll+0] );
					u_lam1   = _mm_load_sd( &ptr_lam[ll+ng0] );
					u_tmp0   = _mm_mul_pd( u_lam0, u_dt0 );
					u_tmp1   = _mm_mul_pd( u_lam1, u_dt1 );
					u_resm0  = _mm_load_sd( &ptr_res_m[ll+0] );
					u_resm1  = _mm_load_sd( &ptr_res_m[ll+ng0] );
					u_tmp0   = _mm_add_pd( u_tmp0, u_resm0 );
					u_tmp1   = _mm_add_pd( u_tmp1, u_resm1 );
					u_tinv0  = _mm_load_sd( &ptr_t_inv[ll+0] );
					u_tinv1  = _mm_load_sd( &ptr_t_inv[ll+ng0] );
					u_tinv0  = _mm_xor_pd( u_tinv0, _mm256_castpd256_pd128( v_sign ) );
					u_tinv1  = _mm_xor_pd( u_tinv1, _mm256_castpd256_pd128( v_sign ) );
					u_dlam0  = _mm_mul_pd( u_tinv0, u_tmp0 );
					u_dlam1  = _mm_mul_pd( u_tinv1, u_tmp1 );
					_mm_store_sd( &ptr_dlam[ll+0], u_dlam0 );
					_mm_store_sd( &ptr_dlam[ll+ng0], u_dlam1 );

					u_dt1    = _mm_movedup_pd( u_dt1 );
					u_dt0    = _mm_move_sd( u_dt1, u_dt0 );
					u_t1     = _mm_loaddup_pd( &ptr_t[ll+ng0] );
					u_t0     = _mm_load_sd( &ptr_t[ll+0] );
					u_t0     = _mm_move_sd( u_t1, u_t0 );
					u_dlam1  = _mm_movedup_pd( u_dlam1 );
					u_dlam0  = _mm_move_sd( u_dlam1, u_dlam0 );
					u_lam1   = _mm_movedup_pd( u_lam1 );
					u_lam0   = _mm_move_sd( u_lam1, u_lam0 );

					v_dlam0  = _mm256_castpd128_pd256( u_dlam0 );
					v_dlam0  = _mm256_insertf128_pd( v_dlam0, u_dt0, 0x1 );
					v_lam0   = _mm256_castpd128_pd256( u_lam0 );
					v_lam0   = _mm256_insertf128_pd( v_lam0, u_t0, 0x1 );

					s_dlam   = _mm256_cvtpd_ps( v_dlam0 );
					s_lam    = _mm256_cvtpd_ps( v_lam0 );
					s_mask0  = _mm_cmp_ps( s_dlam, _mm256_castps256_ps128( t_zeros ), 0x01 );
					s_lam    = _mm_xor_ps( s_lam, _mm256_castps256_ps128( t_sign ) );
					s_tmp0   = _mm_div_ps( s_lam, s_dlam );
					s_tmp0   = _mm_blendv_ps( _mm256_castps256_ps128( t_ones ), s_tmp0, s_mask0 );
					t_tmp0   = _mm256_blend_ps( t_ones, _mm256_castps128_ps256( s_tmp0 ), 0xf );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				else if(ng0-ll==2)
					{

					u_tmp0  = _mm_loadu_pd( &ptr_dt[ll+0] );
					u_resd0 = _mm_loadu_pd( &ptr_res_d[ll+0] );
					u_resd1 = _mm_loadu_pd( &ptr_res_d[ll+ng0] );
					u_dt0   = _mm_sub_pd( u_tmp0, u_resd0 );
					u_dt1   = _mm_sub_pd( u_resd1, u_tmp0 );
					_mm_storeu_pd( &ptr_dt[ll+0], u_dt0 );
					_mm_storeu_pd( &ptr_dt[ll+ng0], u_dt1 );

					u_lam0   = _mm_loadu_pd( &ptr_lam[ll+0] );
					u_lam1   = _mm_loadu_pd( &ptr_lam[ll+ng0] );
					u_tmp0   = _mm_mul_pd( u_lam0, u_dt0 );
					u_tmp1   = _mm_mul_pd( u_lam1, u_dt1 );
					u_resm0  = _mm_loadu_pd( &ptr_res_m[ll+0] );
					u_resm1  = _mm_loadu_pd( &ptr_res_m[ll+ng0] );
					u_tmp0   = _mm_add_pd( u_tmp0, u_resm0 );
					u_tmp1   = _mm_add_pd( u_tmp1, u_resm1 );
					u_tinv0  = _mm_loadu_pd( &ptr_t_inv[ll+0] );
					u_tinv1  = _mm_loadu_pd( &ptr_t_inv[ll+ng0] );
					u_tinv0  = _mm_xor_pd( u_tinv0, _mm256_castpd256_pd128( v_sign ) );
					u_tinv1  = _mm_xor_pd( u_tinv1, _mm256_castpd256_pd128( v_sign ) );
					u_dlam0  = _mm_mul_pd( u_tinv0, u_tmp0 );
					u_dlam1  = _mm_mul_pd( u_tinv1, u_tmp1 );
					_mm_storeu_pd( &ptr_dlam[ll+0], u_dlam0 );
					_mm_storeu_pd( &ptr_dlam[ll+ng0], u_dlam1 );

					v_dt0    = _mm256_castpd128_pd256( u_dt0 );
					v_dt0    = _mm256_insertf128_pd( v_dt0, u_dt1, 0x1 );
					v_t0     = _mm256_castpd128_pd256( _mm_loadu_pd( &ptr_t[ll+0] ) );
					v_t0     = _mm256_insertf128_pd( v_t0, _mm_loadu_pd( &ptr_t[ll+ng0]), 0x1 );
					v_dlam0  = _mm256_castpd128_pd256( u_dlam0 );
					v_dlam0  = _mm256_insertf128_pd( v_dlam0, u_dlam1, 0x1 );
					v_lam0   = _mm256_castpd128_pd256( u_lam0 );
					v_lam0   = _mm256_insertf128_pd( v_lam0, u_lam1, 0x1 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

					}
				else // if(ng-ll==3)
					{

					i_mask = _mm256_castpd_si256( _mm256_set_pd( 1.0, -1.0, -1.0, -1.0 ) );

					v_tmp0  = _mm256_loadu_pd( &ptr_dt[ll+0] );
					v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
					v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+ng0] );
					v_dt0   = _mm256_sub_pd( v_tmp0, v_resd0 );
					v_dt1   = _mm256_sub_pd( v_resd1, v_tmp0 );
					_mm256_maskstore_pd( &ptr_dt[ll+0], i_mask, v_dt0 );
					_mm256_maskstore_pd( &ptr_dt[ll+ng0], i_mask, v_dt1 );

					v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
					v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+ng0] );
					v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
					v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
					v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
					v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+ng0] );
					v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
					v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
					v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
					v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+ng0] );
					v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
					v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
					v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
					v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
					_mm256_maskstore_pd( &ptr_dlam[ll+0], i_mask, v_dlam0 );
					_mm256_maskstore_pd( &ptr_dlam[ll+ng0], i_mask, v_dlam1 );

					t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam1 ) ), 0x20 );
					t_dt     = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt1 ) ), 0x20 );
					t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
					t_mask1  = _mm256_cmp_ps( t_dt, t_zeros, 0x01 );
					v_t0  = _mm256_loadu_pd( &ptr_t[ll+0] );
					v_t1  = _mm256_loadu_pd( &ptr_t[ll+ng0] );
					t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam1 ) ), 0x20 );
					t_t      = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t0 ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t1 ) ), 0x20 );
					t_lam    = _mm256_xor_ps( t_lam, t_sign );
					t_t      = _mm256_xor_ps( t_t, t_sign );
					t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
					t_tmp1   = _mm256_div_ps( t_t, t_dt );
					t_mask0  = _mm256_blend_ps( t_zeros, t_mask0, 0x77 );
					t_mask1  = _mm256_blend_ps( t_zeros, t_mask1, 0x77 );
					t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
					t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
					t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
					t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

					}

				}

#else
			for(; ll<ng0; ll++)
				{

				ptr_dt[ll+ng0] = - ptr_dt[ll];

				ptr_dt[ll+0]   -= ptr_res_d[ll+0];
				ptr_dt[ll+ng0] += ptr_res_d[ll+ng0];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+ng0] = - ptr_t_inv[ll+ng0] * ( ptr_lam[ll+ng0]*ptr_dt[ll+ng0] + ptr_res_m[ll+ng0] );

				if( -alpha*ptr_dlam[ll+0]>ptr_lam[ll+0] )
					{
					alpha = - ptr_lam[ll+0] / ptr_dlam[ll+0];
					}
				if( -alpha*ptr_dlam[ll+ng0]>ptr_lam[ll+ng0] )
					{
					alpha = - ptr_lam[ll+ng0] / ptr_dlam[ll+ng0];
					}
				if( -alpha*ptr_dt[ll+0]>ptr_t[ll+0] )
					{
					alpha = - ptr_t[ll+0] / ptr_dt[ll+0];
					}
				if( -alpha*ptr_dt[ll+ng0]>ptr_t[ll+ng0] )
					{
					alpha = - ptr_t[ll+ng0] / ptr_dt[ll+ng0];
					}

				}
#endif

			}

		}		

	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha0 = _mm256_extractf128_ps( t_alpha0, 0x1 );
	s_alpha1 = _mm256_castps256_ps128( t_alpha0 );
	s_alpha0 = _mm_min_ps( s_alpha0, s_alpha1 );
	
	v_alpha = _mm256_cvtps_pd( s_alpha0 );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );
//	u_alpha = _mm_min_sd( u_alpha, _mm_set_sd( 1.0 ) );
	_mm_store_sd( ptr_alpha, u_alpha );

	// store alpha
//	ptr_alpha[0] = alpha;

	return;
	
	}



void d_compute_dt_dlam_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, double **dux, double **t, double **t_inv, double **lam, double **pDCt, double **res_d, double **res_m, double **dt, double **dlam)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nx0, nb0, ng0, cng;

	double
		*ptr_res_d, *ptr_res_m, *ptr_dux, *ptr_t, *ptr_t_inv, *ptr_dt, *ptr_lam, *ptr_dlam;
	
	int
		*ptr_idxb;
	
	__m128d
		u_tmp0,
		u_tmp1;
	
	__m256d
		v_dux, v_sign,
		v_resm0, v_resd0, v_dt0, v_dlam0, v_tmp0, v_tinv0, v_lam0, v_t0,
		v_resm1, v_resd1, v_dt1, v_dlam1, v_tmp1, v_tinv1, v_lam1, v_t1;

	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );

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

			// box constraints
			ll = 0;
			for(; ll<nb0-3; ll+=4)
				{

				u_tmp0  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+0]] );
				u_tmp1  = _mm_load_sd( &ptr_dux[ptr_idxb[ll+2]] );
				u_tmp0  = _mm_loadh_pd( u_tmp0, &ptr_dux[ptr_idxb[ll+1]] );
				u_tmp1  = _mm_loadh_pd( u_tmp1, &ptr_dux[ptr_idxb[ll+3]] );
				v_dux   = _mm256_castpd128_pd256( u_tmp0 );
				v_dux   = _mm256_insertf128_pd( v_dux, u_tmp1, 0x1 );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+nb0] );
				v_dt0   = _mm256_sub_pd( v_dux, v_resd0 );
				v_dt1   = _mm256_sub_pd( v_resd1, v_dux );
				_mm256_storeu_pd( &ptr_dt[ll+0], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[ll+nb0], v_dt1 );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+nb0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+nb0] );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
				v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+nb0] );
				v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
				v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
				v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
				v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
				_mm256_storeu_pd( &ptr_dlam[ll+0], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[ll+nb0], v_dlam1 );

				}
			for(; ll<nb0; ll++)
				{
				
				ptr_dt[ll+0]   =   ptr_dux[ptr_idxb[ll]] - ptr_res_d[ll+0];
				ptr_dt[ll+nb0] = - ptr_dux[ptr_idxb[ll]] + ptr_res_d[ll+nb0];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+nb0] = - ptr_t_inv[ll+nb0] * ( ptr_lam[ll+nb0]*ptr_dt[ll+nb0] + ptr_res_m[ll+nb0] );

				}

			ptr_res_d += 2*nb0;
			ptr_res_m += 2*nb0;
			ptr_t     += 2*nb0;
			ptr_t_inv += 2*nb0;
			ptr_dt    += 2*nb0;
			ptr_lam   += 2*nb0;
			ptr_dlam  += 2*nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			nu0 = nu[jj];
			nx0 = nx[jj];
			cng = (ng0+ncl-1)/ncl*ncl;

#ifdef BLASFEO
			dgemv_t_lib(nx0+nu0, ng0, 1.0, pDCt[jj], cng, ptr_dux, 0.0, ptr_dt, ptr_dt);
#else
			dgemv_t_lib(nx0+nu0, ng0, pDCt[jj], cng, ptr_dux, 0, ptr_dt, ptr_dt);
#endif

			ll = 0;
			for(; ll<ng0-3; ll+=4)
				{

				v_tmp0  = _mm256_loadu_pd( &ptr_dt[ll+0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ll+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ll+ng0] );
				v_dt0   = _mm256_sub_pd( v_tmp0, v_resd0 );
				v_dt1   = _mm256_sub_pd( v_resd1, v_tmp0 );
				_mm256_storeu_pd( &ptr_dt[ll+0], v_dt0 );
				_mm256_storeu_pd( &ptr_dt[ll+ng0], v_dt1 );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ll+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ll+ng0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_dt0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_dt1 );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ll+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ll+ng0] );
				v_tmp0  = _mm256_add_pd( v_tmp0, v_resm0 );
				v_tmp1  = _mm256_add_pd( v_tmp1, v_resm1 );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ll+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ll+ng0] );
				v_tinv0 = _mm256_xor_pd( v_tinv0, v_sign );
				v_tinv1 = _mm256_xor_pd( v_tinv1, v_sign );
				v_dlam0  = _mm256_mul_pd( v_tinv0, v_tmp0 );
				v_dlam1  = _mm256_mul_pd( v_tinv1, v_tmp1 );
				_mm256_storeu_pd( &ptr_dlam[ll+0], v_dlam0 );
				_mm256_storeu_pd( &ptr_dlam[ll+ng0], v_dlam1 );

				}
			for(; ll<ng0; ll++)
				{

				ptr_dt[ll+ng0] = - ptr_dt[ll];

				ptr_dt[ll+0]   -= ptr_res_d[ll+0];
				ptr_dt[ll+ng0] += ptr_res_d[ll+ng0];

				ptr_dlam[ll+0]   = - ptr_t_inv[ll+0]   * ( ptr_lam[ll+0]*ptr_dt[ll+0]     + ptr_res_m[ll+0] );
				ptr_dlam[ll+ng0] = - ptr_t_inv[ll+ng0] * ( ptr_lam[ll+ng0]*ptr_dt[ll+ng0] + ptr_res_m[ll+ng0] );

				}

			}

		}		

	return;
	
	}



void d_update_var_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double alpha, double **ux, double **dux, double **pi, double **dpi, double **t, double **dt, double **lam, double **dlam)
	{

	// constants
	const int bs = D_MR;

	int nu0, nx0, nx1, nb0, ng0;

	int jj, ll;
	
	double
		*ptr_ux, *ptr_dux, *ptr_pi, *ptr_dpi, *ptr_t, *ptr_dt, *ptr_lam, *ptr_dlam;

	for(jj=1; jj<=N; jj++)
		{

		nx0 = nx[jj];

		// update equality constrained multipliers
		ptr_pi     = pi[jj];
		ptr_dpi    = dpi[jj];
		daxpy_lib(nx1, alpha, ptr_dpi, ptr_pi);

		}

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		ng0 = ng[jj];
		
		// update inputs and states
		ptr_ux     = ux[jj];
		ptr_dux    = dux[jj];
		daxpy_lib(nu0+nx0, alpha, ptr_dux, ptr_ux);

		// box constraints
		ptr_t       = t[jj];
		ptr_dt      = dt[jj];
		ptr_lam     = lam[jj];
		ptr_dlam    = dlam[jj];
		daxpy_lib(nb0, alpha, &ptr_dlam[0], &ptr_lam[0]);
		daxpy_lib(nb0, alpha, &ptr_dlam[nb0], &ptr_lam[nb0]);
		daxpy_lib(nb0, alpha, &ptr_dt[0], &ptr_t[0]);
		daxpy_lib(nb0, alpha, &ptr_dt[nb0], &ptr_t[nb0]);

		// general constraints
		ptr_t       += 2*nb0;
		ptr_dt      += 2*nb0;
		ptr_lam     += 2*nb0;
		ptr_dlam    += 2*nb0;
		daxpy_lib(ng0, alpha, &ptr_dlam[0], &ptr_lam[0]);
		daxpy_lib(ng0, alpha, &ptr_dlam[ng0], &ptr_lam[ng0]);
		daxpy_lib(ng0, alpha, &ptr_dt[0], &ptr_t[0]);
		daxpy_lib(ng0, alpha, &ptr_dt[ng0], &ptr_t[ng0]);

		}

	return;
	
	}



void d_backup_update_var_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double alpha, struct d_strvec *hsux_bkp, struct d_strvec *hsux, struct d_strvec *hsdux, struct d_strvec *hspi_bkp, struct d_strvec *hspi, struct d_strvec *hsdpi, struct d_strvec *hst_bkp, struct d_strvec *hst, struct d_strvec *hsdt, struct d_strvec *hslam_bkp, struct d_strvec *hslam, struct d_strvec *hsdlam)
	{

	int ii;

	int nu0, nx0, nx1, nb0, ng0;

	int jj, ll;
	
	double
		*ptr_ux_bkp, *ptr_ux, *ptr_dux, *ptr_pi_bkp, *ptr_pi, *ptr_dpi, *ptr_t_bkp, *ptr_t, *ptr_dt, *ptr_lam_bkp, *ptr_lam, *ptr_dlam;

	for(jj=1; jj<=N; jj++)
		{

		nx0 = nx[jj];

		// update equality constrained multipliers
		ptr_pi_bkp = hspi_bkp[jj].pa;
		ptr_pi     = hspi[jj].pa;
		ptr_dpi    = hsdpi[jj].pa;
		daxpy_bkp_lib(nx0, alpha, ptr_dpi, ptr_pi, ptr_pi_bkp);

		}

	for(jj=0; jj<=N; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		ng0 = ng[jj];
		if(jj<N)
			nx1 = nx[jj+1];
		else
			nx1 = 0;
		
		// update inputs and states
		ptr_ux_bkp = hsux_bkp[jj].pa;
		ptr_ux     = hsux[jj].pa;
		ptr_dux    = hsdux[jj].pa;
		daxpy_bkp_lib(nu0+nx0, alpha, ptr_dux, ptr_ux, ptr_ux_bkp);

		// box constraints
		ptr_t_bkp   = hst_bkp[jj].pa;
		ptr_t       = hst[jj].pa;
		ptr_dt      = hsdt[jj].pa;
		ptr_lam_bkp = hslam_bkp[jj].pa;
		ptr_lam     = hslam[jj].pa;
		ptr_dlam    = hsdlam[jj].pa;
		daxpy_bkp_lib(nb0, alpha, &ptr_dlam[0], &ptr_lam[0], &ptr_lam_bkp[0]);
		daxpy_bkp_lib(nb0, alpha, &ptr_dlam[nb0], &ptr_lam[nb0], &ptr_lam_bkp[nb0]);
		daxpy_bkp_lib(nb0, alpha, &ptr_dt[0], &ptr_t[0], &ptr_t_bkp[0]);
		daxpy_bkp_lib(nb0, alpha, &ptr_dt[nb0], &ptr_t[nb0], &ptr_t_bkp[nb0]);

		// general constraints
		ptr_t_bkp   += 2*nb0;
		ptr_t       += 2*nb0;
		ptr_dt      += 2*nb0;
		ptr_lam_bkp += 2*nb0;
		ptr_lam     += 2*nb0;
		ptr_dlam    += 2*nb0;
		daxpy_bkp_lib(ng0, alpha, &ptr_dlam[0], &ptr_lam[0], &ptr_lam_bkp[0]);
		daxpy_bkp_lib(ng0, alpha, &ptr_dlam[ng0], &ptr_lam[ng0], &ptr_lam_bkp[ng0]);
		daxpy_bkp_lib(ng0, alpha, &ptr_dt[0], &ptr_t[0], &ptr_t_bkp[0]);
		daxpy_bkp_lib(ng0, alpha, &ptr_dt[ng0], &ptr_t[ng0], &ptr_t_bkp[ng0]);

		}

	return;
	
	}



void d_compute_mu_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, double alpha, struct d_strvec *hslam, struct d_strvec *hsdlam, struct d_strvec *hst, struct d_strvec *hsdt, double *ptr_mu, double mu_scal)
	{

	int ii;
	
	int jj, ll, ll_bkp, ll_end;
	double ll_left;
	
	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_mu0, u_tmp;

	__m256d
		v_alpha, v_mask, v_left, v_zeros,
		v_t0, v_dt0, v_lam0, v_dlam0, v_mu0, 
		v_t1, v_dt1, v_lam1, v_dlam1, v_mu1;
		
	double
		*ptr_t, *ptr_lam, *ptr_dt, *ptr_dlam;

	int nb0, ng0;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_zeros = _mm256_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();

	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t    = hst[jj].pa;
		ptr_lam  = hslam[jj].pa;
		ptr_dt   = hsdt[jj].pa;
		ptr_dlam = hsdlam[jj].pa;

		// box constraints
		nb0 = nb[jj];
		for(ll=0; ll<nb0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#endif
#if defined(TARGET_X64_AVX)
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<nb0)
			{
			ll_left = nb0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_loadu_pd( &ptr_t[0*nb0+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*nb0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*nb0+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*nb0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*nb0+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*nb0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*nb0+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*nb0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#endif
#if defined(TARGET_X64_AVX)
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		ptr_t    += 2*nb0;
		ptr_lam  += 2*nb0;
		ptr_dt   += 2*nb0;
		ptr_dlam += 2*nb0;

		// general constraints
		ng0 = ng[jj];
		for(ll=0; ll<ng0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#endif
#if defined(TARGET_X64_AVX)
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}
		if(ll<ng0)
			{
			ll_left = ng0-ll;
			v_left  = _mm256_broadcast_sd( &ll_left );
			v_mask  = _mm256_loadu_pd( d_mask );
			v_mask  = _mm256_sub_pd( v_mask, v_left );

			v_t0    = _mm256_loadu_pd( &ptr_t[ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[ng0+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[ng0+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[ng0+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[ng0+ll] );
#if defined(TARGET_X64_AVX2)
			v_t0    = _mm256_fmadd_pd( v_alpha, v_dt0, v_t0 );
			v_t1    = _mm256_fmadd_pd( v_alpha, v_dt1, v_t1 );
			v_lam0  = _mm256_fmadd_pd( v_alpha, v_dlam0, v_lam0 );
			v_lam1  = _mm256_fmadd_pd( v_alpha, v_dlam1, v_lam1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_fmadd_pd( v_lam0, v_t0, v_mu0 );
			v_mu1   = _mm256_fmadd_pd( v_lam1, v_t1, v_mu1 );
#endif
#if defined(TARGET_X64_AVX)
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_lam0  = _mm256_blendv_pd( v_zeros, v_lam0, v_mask );
			v_lam1  = _mm256_blendv_pd( v_zeros, v_lam1, v_mask );
			v_mu0   = _mm256_add_pd( v_mu0, v_lam0 );
			v_mu1   = _mm256_add_pd( v_mu1, v_lam1 );
#endif
			}

		}

	v_mu0 = _mm256_add_pd( v_mu0, v_mu1 );
	u_mu0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
	u_mu0 = _mm_hadd_pd( u_mu0, u_mu0 );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu0 = _mm_mul_sd( u_mu0, u_tmp );
	_mm_store_sd( ptr_mu, u_mu0 );

	return;

	}



void d_compute_centering_correction_res_mpc_hard_libstr(int N, int *nb, int *ng, double sigma_mu, struct d_strvec *hsdt, struct d_strvec *hsdlam, struct d_strvec *hsres_m)
	{

	int ii, jj;

	int nb0, ng0;

	double
		*ptr_res_m, *ptr_dt, *ptr_dlam;
	
	__m256d
		v_sigma_mu,
		v_dt0, v_dlam0, v_tmp0, v_resm0,
		v_dt1, v_dlam1, v_tmp1, v_resm1;
	
	__m256i
		i_mask;
	
	double ii_left;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	v_sigma_mu = _mm256_broadcast_sd( &sigma_mu );

	for(ii=0; ii<=N; ii++)
		{

		nb0 = nb[ii];

		ng0 = ng[ii]; //(ng[ii]+bs-1)/bs*bs;

		ptr_res_m = hsres_m[ii].pa;
		ptr_dt    = hsdt[ii].pa;
		ptr_dlam  = hsdlam[ii].pa;

		for(jj=0; jj<nb[ii]-3; jj+=4)
			{
			v_dt0   = _mm256_loadu_pd( &ptr_dt[jj+0] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[jj+nb0] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[jj+0] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[jj+nb0] );
			v_tmp0  = _mm256_mul_pd( v_dt0, v_dlam0 );
			v_tmp1  = _mm256_mul_pd( v_dt1, v_dlam1 );
			v_tmp0  = _mm256_sub_pd( v_tmp0, v_sigma_mu );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_sigma_mu );
			v_resm0 = _mm256_loadu_pd( &ptr_res_m[jj+0] );
			v_resm1 = _mm256_loadu_pd( &ptr_res_m[jj+nb0] );
			v_resm0 = _mm256_add_pd( v_resm0, v_tmp0 );
			v_resm1 = _mm256_add_pd( v_resm1, v_tmp1 );
			_mm256_storeu_pd( &ptr_res_m[jj+0], v_resm0 );
			_mm256_storeu_pd( &ptr_res_m[jj+nb0], v_resm1 );
			}
		if(jj<nb[ii])
			{
			ii_left = nb[ii]-jj;
			i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

			v_dt0   = _mm256_loadu_pd( &ptr_dt[jj+0] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[jj+nb0] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[jj+0] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[jj+nb0] );
			v_tmp0  = _mm256_mul_pd( v_dt0, v_dlam0 );
			v_tmp1  = _mm256_mul_pd( v_dt1, v_dlam1 );
			v_tmp0  = _mm256_sub_pd( v_tmp0, v_sigma_mu );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_sigma_mu );
			v_resm0 = _mm256_loadu_pd( &ptr_res_m[jj+0] );
			v_resm1 = _mm256_loadu_pd( &ptr_res_m[jj+nb0] );
			v_resm0 = _mm256_add_pd( v_resm0, v_tmp0 );
			v_resm1 = _mm256_add_pd( v_resm1, v_tmp1 );
			_mm256_maskstore_pd( &ptr_res_m[jj+0], i_mask, v_resm0 );
			_mm256_maskstore_pd( &ptr_res_m[jj+nb0], i_mask, v_resm1 );
			}

		ptr_res_m += 2*nb0;
		ptr_dt    += 2*nb0;
		ptr_dlam  += 2*nb0;

		for(jj=0; jj<ng[ii]-3; jj+=4)
			{
			v_dt0   = _mm256_loadu_pd( &ptr_dt[jj+0] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[jj+ng0] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[jj+0] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[jj+ng0] );
			v_tmp0  = _mm256_mul_pd( v_dt0, v_dlam0 );
			v_tmp1  = _mm256_mul_pd( v_dt1, v_dlam1 );
			v_tmp0  = _mm256_sub_pd( v_tmp0, v_sigma_mu );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_sigma_mu );
			v_resm0 = _mm256_loadu_pd( &ptr_res_m[jj+0] );
			v_resm1 = _mm256_loadu_pd( &ptr_res_m[jj+ng0] );
			v_resm0 = _mm256_add_pd( v_resm0, v_tmp0 );
			v_resm1 = _mm256_add_pd( v_resm1, v_tmp1 );
			_mm256_storeu_pd( &ptr_res_m[jj+0], v_resm0 );
			_mm256_storeu_pd( &ptr_res_m[jj+ng0], v_resm1 );
			}
		if(jj<ng[ii])
			{
			ii_left = ng[ii]-jj;
			i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

			v_dt0   = _mm256_loadu_pd( &ptr_dt[jj+0] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[jj+ng0] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[jj+0] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[jj+ng0] );
			v_tmp0  = _mm256_mul_pd( v_dt0, v_dlam0 );
			v_tmp1  = _mm256_mul_pd( v_dt1, v_dlam1 );
			v_tmp0  = _mm256_sub_pd( v_tmp0, v_sigma_mu );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_sigma_mu );
			v_resm0 = _mm256_loadu_pd( &ptr_res_m[jj+0] );
			v_resm1 = _mm256_loadu_pd( &ptr_res_m[jj+ng0] );
			v_resm0 = _mm256_add_pd( v_resm0, v_tmp0 );
			v_resm1 = _mm256_add_pd( v_resm1, v_tmp1 );
			_mm256_maskstore_pd( &ptr_res_m[jj+0], i_mask, v_resm0 );
			_mm256_maskstore_pd( &ptr_res_m[jj+ng0], i_mask, v_resm1 );
			}

		}

	}



void d_update_gradient_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int *ng, struct d_strvec *hsres_d, struct d_strvec *hsres_m, struct d_strvec *hslam, struct d_strvec *hstinv, struct d_strvec *hsqx)
	{
	
	int ii, jj, bs0;

	int nb0, ng0;
	
	double temp0, temp1;
	
	double 
		*ptr_res_d, *ptr_Qx, *ptr_qx, *ptr_lam, *ptr_res_m, *ptr_t_inv;
	
	__m256d
		v_ones,
		v_tmp0, v_tinv0, v_lam0, v_resm0, v_resd0,
		v_tmp1, v_tinv1, v_lam1, v_resm1, v_resd1;
	
	__m256i
		i_mask;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );

	double ii_left;

	static double d_mask[4] = {0.5, 1.5, 2.5, 3.5};

	for(jj=0; jj<=N; jj++)
		{
		
		ptr_lam   = hslam[jj].pa;
		ptr_t_inv = hstinv[jj].pa;
		ptr_res_d = hsres_d[jj].pa;
		ptr_res_m = hsres_m[jj].pa;
		ptr_qx    = hsqx[jj].pa;

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			for(ii=0; ii<nb0-3; ii+=4)
				{

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+nb0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+nb0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+nb0] );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ii+nb0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_qx[ii+0], v_tmp0 );

				}
			if(ii<nb0)
				{

				ii_left = nb0-ii;
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+nb0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+nb0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+nb0] );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ii+nb0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_tmp0 );

				}

			ptr_lam   += 2*nb0;
			ptr_t_inv += 2*nb0;
			ptr_res_d += 2*nb0;
			ptr_res_m += 2*nb0;
			ptr_qx    += nb0;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			for(ii=0; ii<ng0-3; ii+=4)
				{

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+ng0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+ng0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+ng0] );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ii+ng0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_storeu_pd( &ptr_qx[ii+0], v_tmp0 );

				}
			if(ii<ng0)
				{

				ii_left = ng0-ii;
				i_mask  = _mm256_castpd_si256( _mm256_sub_pd( _mm256_loadu_pd( d_mask ), _mm256_broadcast_sd( &ii_left ) ) );

				v_lam0  = _mm256_loadu_pd( &ptr_lam[ii+0] );
				v_lam1  = _mm256_loadu_pd( &ptr_lam[ii+ng0] );
				v_resm0 = _mm256_loadu_pd( &ptr_res_m[ii+0] );
				v_resm1 = _mm256_loadu_pd( &ptr_res_m[ii+ng0] );
				v_resd0 = _mm256_loadu_pd( &ptr_res_d[ii+0] );
				v_resd1 = _mm256_loadu_pd( &ptr_res_d[ii+ng0] );
				v_tinv0 = _mm256_loadu_pd( &ptr_t_inv[ii+0] );
				v_tinv1 = _mm256_loadu_pd( &ptr_t_inv[ii+ng0] );
				v_tmp0  = _mm256_mul_pd( v_lam0, v_resd0 );
				v_tmp1  = _mm256_mul_pd( v_lam1, v_resd1 );
				v_tmp0  = _mm256_sub_pd( v_resm0, v_tmp0 );
				v_tmp1  = _mm256_add_pd( v_resm1, v_tmp1 );
				v_tmp0  = _mm256_mul_pd( v_tmp0, v_tinv0 );
				v_tmp1  = _mm256_mul_pd( v_tmp1, v_tinv1 );
				v_tmp0  = _mm256_sub_pd( v_tmp0, v_tmp1 );
				_mm256_maskstore_pd( &ptr_qx[ii+0], i_mask, v_tmp0 );

				}

			}

		}

	}



#endif

