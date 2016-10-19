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
#endif

#include "../../include/tree.h"



// initialize variables

void d_init_var_tree_mpc_hard_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **hidxb, int *ng, struct d_strvec *hsux, struct d_strvec *hspi, struct d_strmat *hsDCt, struct d_strvec *hsdb, struct d_strvec *hst, struct d_strvec *hslam, double mu0, int warm_start)
	{

	int jj, ll, ii;
	int nkids, idxkid;

	double *hux[Nn];
	double *hpi[Nn];
	double *hdb[Nn];
	double *ht[Nn];
	double *hlam[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		hux[ii] = hsux[ii].pa;
		hpi[ii] = hspi[ii].pa;
		hdb[ii] = hsdb[ii].pa;
		ht[ii] = hst[ii].pa;
		hlam[ii] = hslam[ii].pa;
		}

	int nb0, ng0;
	
	double
		*ptr_t, *ptr_lam, *ptr_db;

	double thr0 = 0.1; // minimum vale of t (minimum distance from a constraint)


	// cold start
	if(warm_start==0)
		{
		for(jj=0; jj<Nn; jj++)
			{
			for(ll=0; ll<nu[jj]+nx[jj]; ll++)
				{
				hux[jj][ll] = 0.0;
				}
			}
		}


	// check bounds & initialize multipliers
	for(jj=0; jj<Nn; jj++)
		{
		nb0 = nb[jj];
		for(ll=0; ll<nb0; ll++)
			{
			ht[jj][ll]     = - hdb[jj][ll]     + hux[jj][hidxb[jj][ll]];
			ht[jj][nb0+ll] =   hdb[jj][nb0+ll] - hux[jj][hidxb[jj][ll]];
			if(ht[jj][ll] < thr0)
				{
				if(ht[jj][nb0+ll] < thr0)
					{
					hux[jj][hidxb[jj][ll]] = ( - hdb[jj][nb0+ll] + hdb[jj][ll])*0.5;
					ht[jj][ll]     = thr0; //- hdb[jj][ll]     + hux[jj][hidxb[jj][ll]];
					ht[jj][nb0+ll] = thr0; //  hdb[jj][nb0+ll] - hux[jj][hidxb[jj][ll]];
					}
				else
					{
					ht[jj][ll] = thr0;
					hux[jj][hidxb[jj][ll]] = hdb[jj][ll] + thr0;
					}
				}
			else if(ht[jj][nb0+ll] < thr0)
				{
				ht[jj][nb0+ll] = thr0;
				hux[jj][hidxb[jj][ll]] = hdb[jj][nb0+ll] - thr0;
				}
			hlam[jj][ll]     = mu0/ht[jj][ll];
			hlam[jj][nb0+ll] = mu0/ht[jj][nb0+ll];
			}
		}


	// initialize pi
	for(ii=0; ii<Nn; ii++)
		{
		nkids = tree[ii].nkids;
		for(jj=0; jj<nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			for(ll=0; ll<nx[idxkid]; ll++)
				hpi[idxkid][ll] = 0.0; // initialize multipliers to zero
			}
		}

	// TODO find a better way to initialize general constraints
	for(jj=0; jj<Nn; jj++)
		{
		nb0 = nb[jj];
		ng0 = ng[jj];
		if(ng0>0)
			{
			ptr_t   = ht[jj];
			ptr_lam = hlam[jj];
			ptr_db  = hdb[jj];
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



void d_update_var_tree_mpc_hard_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int *ng, double *ptr_mu, double mu_scal, double alpha, struct d_strvec *hsux, struct d_strvec *hsdux, struct d_strvec *hst, struct d_strvec *hsdt, struct d_strvec *hslam, struct d_strvec *hsdlam, struct d_strvec *hspi, struct d_strvec *hsdpi)
	{
	
	int ii;
	int nkids, idxkid;

	double *ux[Nn];
	double *dux[Nn];
	double *pi[Nn];
	double *dpi[Nn];
	double *t[Nn];
	double *dt[Nn];
	double *lam[Nn];
	double *dlam[Nn];
	for(ii=0; ii<Nn; ii++)
		{
		ux[ii] = hsux[ii].pa;
		dux[ii] = hsdux[ii].pa;
		pi[ii] = hspi[ii].pa;
		dpi[ii] = hsdpi[ii].pa;
		t[ii] = hst[ii].pa;
		dt[ii] = hsdt[ii].pa;
		lam[ii] = hslam[ii].pa;
		dlam[ii] = hsdlam[ii].pa;
		}

	int nu0, nx0, nx1, nb0, pnb, ng0, png;

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

	// equality constraints lagrange multipliers
	for(ii=0; ii<Nn; ii++)
		{
		nkids = tree[ii].nkids;
		for(jj=0; jj<nkids; jj++)
			{
			idxkid = tree[ii].kids[jj];
			nx0 = nx[idxkid];
			ptr_pi   = pi[idxkid];
			ptr_dpi  = dpi[idxkid];
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
		}
	
	for(jj=0; jj<Nn; jj++)
		{

		nx0 = nx[jj];
		nu0 = nu[jj];
		nb0 = nb[jj];
		pnb  = nb0; //bs*((nb0+bs-1)/bs); // cache aligned number of box constraints
		ng0 = ng[jj];
		png  = ng0; //bs*((ng0+bs-1)/bs); // cache aligned number of box constraints
		
		ptr_ux   = ux[jj];
		ptr_dux  = dux[jj];
		ptr_t    = t[jj];
		ptr_dt   = dt[jj];
		ptr_lam  = lam[jj];
		ptr_dlam = dlam[jj];

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
			v_t0    = _mm256_loadu_pd( &ptr_t[0*pnb+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*pnb+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*pnb+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*pnb+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*pnb+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*pnb+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*pnb+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*pnb+ll] );
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
			_mm256_storeu_pd( &ptr_t[0*pnb+ll], v_t0 );
			_mm256_storeu_pd( &ptr_t[1*pnb+ll], v_t1 );
			_mm256_storeu_pd( &ptr_lam[0*pnb+ll], v_lam0 );
			_mm256_storeu_pd( &ptr_lam[1*pnb+ll], v_lam1 );
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

			v_t0    = _mm256_loadu_pd( &ptr_t[0*pnb+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*pnb+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*pnb+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*pnb+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*pnb+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*pnb+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*pnb+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*pnb+ll] );
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
			_mm256_maskstore_pd( &ptr_t[0*pnb+ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[1*pnb+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[0*pnb+ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[1*pnb+ll], i_mask, v_lam1 );
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

		ptr_t    += 2*pnb;
		ptr_dt   += 2*pnb;
		ptr_lam  += 2*pnb;
		ptr_dlam += 2*pnb;

		// genreal constraints
		for(ll=0; ll<ng0-3; ll+=4)
			{
			v_t0    = _mm256_loadu_pd( &ptr_t[0*png+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*png+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*png+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*png+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*png+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*png+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*png+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*png+ll] );
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
			_mm256_storeu_pd( &ptr_t[0*png+ll], v_t0 );
			_mm256_storeu_pd( &ptr_t[1*png+ll], v_t1 );
			_mm256_storeu_pd( &ptr_lam[0*png+ll], v_lam0 );
			_mm256_storeu_pd( &ptr_lam[1*png+ll], v_lam1 );
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

			v_t0    = _mm256_loadu_pd( &ptr_t[0*png+ll] );
			v_t1    = _mm256_loadu_pd( &ptr_t[1*png+ll] );
			v_lam0  = _mm256_loadu_pd( &ptr_lam[0*png+ll] );
			v_lam1  = _mm256_loadu_pd( &ptr_lam[1*png+ll] );
			v_dt0   = _mm256_loadu_pd( &ptr_dt[0*png+ll] );
			v_dt1   = _mm256_loadu_pd( &ptr_dt[1*png+ll] );
			v_dlam0 = _mm256_loadu_pd( &ptr_dlam[0*png+ll] );
			v_dlam1 = _mm256_loadu_pd( &ptr_dlam[1*png+ll] );
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
			_mm256_maskstore_pd( &ptr_t[0*png+ll], i_mask, v_t0 );
			_mm256_maskstore_pd( &ptr_t[1*png+ll], i_mask, v_t1 );
			_mm256_maskstore_pd( &ptr_lam[0*png+ll], i_mask, v_lam0 );
			_mm256_maskstore_pd( &ptr_lam[1*png+ll], i_mask, v_lam1 );
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



