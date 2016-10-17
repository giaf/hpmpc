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


#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#ifdef BLASFEO
#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>
#endif

#include "../../include/blas_d.h"
#include "../../include/block_size.h"



/* supports the problem size to change stage-wise */
void d_res_res_mpc_hard_libstr(int N, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsQ, struct d_strvec *hsq, struct d_strvec *hsux, struct d_strmat *hsDCt, struct d_strvec *hsd, struct d_strvec *hspi, struct d_strvec *hslam, struct d_strvec *hst, struct d_strvec *hswork, struct d_strvec *hsrq, struct d_strvec *hsrb, struct d_strvec *hsrd, struct d_strvec *hsrm, double *mu)
	{

	int ii, jj;

	// extract pointers to vectors
	double *hb[N+1];
	double *hq[N+1];
	double *hux[N+1];
	double *hd[N+1];
	double *hpi[N+1];
	double *hlam[N+1];
	double *ht[N+1];
	double *hrq[N+1];
	double *hrb[N+1];
	double *hrd[N+1];
	double *hrm[N+1];
	double *work0 = hswork[0].pa;
	double *work1 = hswork[1].pa;
	for(ii=0; ii<N; ii++)
		{
		hb[ii+1] = hsb[ii+1].pa;
		hq[ii] = hsq[ii].pa;
		hux[ii] = hsux[ii].pa;
//		hd[ii] = hsd[ii].pa;
		hpi[ii+1] = hspi[ii+1].pa;
//		hlam[ii] = hslam[ii].pa;
//		ht[ii] = hst[ii].pa;
		hrq[ii] = hsrq[ii].pa;
		hrb[ii+1] = hsrb[ii+1].pa;
//		hrd[ii] = hsrd[ii].pa;
//		hrm[ii] = hsrm[ii].pa;
		}
	ii = N;
//	hb[ii] = hsb[ii].pa;
	hq[ii] = hsq[ii].pa;
	hux[ii] = hsux[ii].pa;
//	hd[ii] = hsd[ii].pa;
//	hpi[ii] = hspi[ii].pa;
//	hlam[ii] = hslam[ii].pa;
//	ht[ii] = hst[ii].pa;
	hrq[ii] = hsrq[ii].pa;
//	hrb[ii] = hsrb[ii].pa;
//	hrd[ii] = hsrd[ii].pa;
//	hrm[ii] = hsrm[ii].pa;

	const int bs = D_MR;
	const int ncl = D_NCL;

	int nu0, nu1, cnux0, nx0, nx1, nxm, cnx0, cnx1, nb0, pnb, ng0, png, cng, nb_tot;

	__m256d
		v_ux, v_tmp2,
		v_tmp0, v_lam0, v_t0, v_mu0,
		v_tmp1, v_lam1, v_t1, v_mu1;
	
	__m128d
		u_ux, u_tmp2,
		u_tmp0, u_lam0, u_t0, u_mu0,
		u_tmp1, u_lam1, u_t1, u_mu1;

	double
		*ptr_q, *ptr_d, *ptr_ux, *ptr_pi, *ptr_lam, *ptr_t, *ptr_rq, *ptr_rd, *ptr_rm;
	
	int
		*ptr_idxb;
	
	double
		mu2;

	// initialize mu
	nb_tot = 0;
	mu2 = 0;
	u_mu0 = _mm_setzero_pd();
	u_mu1 = _mm_setzero_pd();
	v_mu0 = _mm256_setzero_pd();
	v_mu1 = _mm256_setzero_pd();



	// first stage
	ii = 0;
	nu0 = nu[ii];
	nu1 = nu[ii+1];
	nx0 = nx[ii]; // nx1;
	nx1 = nx[ii+1];
	cnx1  = (nx1+ncl-1)/ncl*ncl;
	cnux0 = (nu0+nx0+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = nb0; //(nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = ng0; //(ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;

	if(nb0>0 | ng0>0)
		{
		hd[ii] = hsd[ii].pa;
		hlam[ii] = hslam[ii].pa;
		ht[ii] = hst[ii].pa;
		hrd[ii] = hsrd[ii].pa;
		hrm[ii] = hsrm[ii].pa;
		}

//	for(jj=0; jj<nu0; jj++) 
//		hrq[ii][jj] = hq[ii][jj];

//	for(jj=0; jj<nx0; jj++) 
//		hrq[ii][nu0+jj] = hq[ii][nu0+jj]; // - hpi[ii-1][jj];

	for(jj=0; jj<nu0+nx0; jj++) 
		hrq[ii][jj] = hq[ii][jj];

	if(nb0>0)
		{

		nb_tot += nb0;

		for(jj=0; jj<nb0; jj++) 
			{
			hrq[ii][idxb[ii][jj]] += - hlam[ii][jj] + hlam[ii][pnb+jj];

			hrd[ii][jj]     = hd[ii][jj]     - hux[ii][idxb[ii][jj]] + ht[ii][jj];
			hrd[ii][pnb+jj] = hd[ii][pnb+jj] - hux[ii][idxb[ii][jj]] - ht[ii][pnb+jj];

			hrm[ii][jj]     = hlam[ii][jj]     * ht[ii][jj];
			hrm[ii][pnb+jj] = hlam[ii][pnb+jj] * ht[ii][pnb+jj];
			mu2 += hrm[ii][jj] + hrm[ii][pnb+jj];
			}
		}

	dsymv_l_libstr(nu0+nx0, nu0+nx0, 1.0, &hsQ[ii], 0, 0, &hsux[ii], 0, 1.0, &hsrq[ii], 0, &hsrq[ii], 0);

	for(jj=0; jj<nx1; jj++) 
		hrb[ii+1][jj] = hb[ii+1][jj] - hux[ii+1][nu1+jj];

	dgemv_nt_libstr(nu0+nx0, nx1, 1.0, 1.0, &hsBAbt[ii], 0, 0, &hspi[ii+1], 0, &hsux[ii], 0, 1.0, 1.0, &hsrq[ii], 0, &hsrb[ii+1], 0, &hsrq[ii], 0, &hsrb[ii+1], 0);

	if(ng0>0)
		{

		nb_tot += ng0;

		for(jj=0; jj<ng0; jj++)
			{
			work0[jj] = hlam[ii][jj+2*pnb+png] - hlam[ii][jj+2*pnb+0];

			hrd[ii][2*pnb+jj]     = hd[ii][2*pnb+jj]     + ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] = hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];

			hrm[ii][2*pnb+jj]     = hlam[ii][2*pnb+jj]     * ht[ii][2*pnb+jj];
			hrm[ii][2*pnb+png+jj] = hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
			mu2 += hrm[ii][2*pnb+jj] + hrm[ii][2*pnb+png+jj];
			}

		dgemv_nt_libstr(nu0+nx0, ng0, 1.0, 1.0, &hsDCt[ii], 0, 0, &hswork[0], 0, &hsux[ii], 0, 1.0, 0.0, &hsrq[ii], 0, &hswork[1], 0, &hsrq[ii], 0, &hswork[1], 0);

		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+jj]     -= work1[jj];
			hrd[ii][2*pnb+png+jj] -= work1[jj];
			}

		}


	


	// middle stages
	for(ii=1; ii<N; ii++)
		{
		nu0 = nu1;
		nu1 = nu[ii+1];
		nx0 = nx1;
		nx1 = nx[ii+1];
		cnx0 = cnx1;
		cnx1  = (nx1+ncl-1)/ncl*ncl;
		cnux0  = (nu0+nx0+ncl-1)/ncl*ncl;
		nb0 = nb[ii];
		pnb = nb0; //(nb0+bs-1)/bs*bs;
		ng0 = ng[ii];
		png = ng0; //(ng0+bs-1)/bs*bs;
		cng = (ng0+ncl-1)/ncl*ncl;

		if(nb0>0 | ng0>0)
			{
			hd[ii] = hsd[ii].pa;
			hlam[ii] = hslam[ii].pa;
			ht[ii] = hst[ii].pa;
			hrd[ii] = hsrd[ii].pa;
			hrm[ii] = hsrm[ii].pa;
			}

		ptr_q  = hq[ii];
		ptr_rq = hrq[ii];
		ptr_pi = hpi[ii];

		for(jj=0; jj<nu0-3; jj+=4)
			{
			v_tmp0 = _mm256_loadu_pd( &ptr_q[jj] );
			_mm256_storeu_pd( &ptr_rq[jj], v_tmp0 );
			}
		for(; jj<nu0; jj++) // TODO mask ?
			{
			u_tmp0 = _mm_load_sd( &ptr_q[jj] );
			_mm_store_sd( &ptr_rq[jj], u_tmp0 );
			}

		for(jj=0; jj<nx0-3; jj+=4)
			{
			v_tmp0 = _mm256_loadu_pd( &ptr_q[nu0+jj] );
			v_tmp1 = _mm256_loadu_pd( &ptr_pi[jj] );
			v_tmp0 = _mm256_sub_pd( v_tmp0, v_tmp1 );
			_mm256_storeu_pd( &ptr_rq[nu0+jj], v_tmp0 );
			}
		for(; jj<nx0; jj++) // TODO mask ?
			{
			u_tmp0 = _mm_load_sd( &ptr_q[nu0+jj] );
			u_tmp1 = _mm_load_sd( &ptr_pi[jj] );
			u_tmp0 = _mm_sub_sd( u_tmp0, u_tmp1 );
			_mm_store_sd( &ptr_rq[nu0+jj], u_tmp0 );
			}

		if(nb0+ng0>0)
			{
			ptr_ux   = hux[ii];
			ptr_idxb = idxb[ii];
			ptr_lam  = hlam[ii];
			ptr_t    = ht[ii];
			ptr_d    = hd[ii];
			ptr_rd   = hrd[ii];
			ptr_rm   = hrm[ii];
			}

		if(nb0>0)
			{

			nb_tot += nb0;

			for(jj=0; jj<nb0-3; jj+=4)
				{
				v_lam0 = _mm256_loadu_pd( &ptr_lam[jj+0] );
				v_lam1 = _mm256_loadu_pd( &ptr_lam[jj+pnb] );
				v_tmp0 = _mm256_sub_pd( v_lam1, v_lam0 );

				u_tmp0 = _mm_load_sd( &ptr_rq[ptr_idxb[jj+0]] );
				u_tmp1 = _mm_load_sd( &ptr_rq[ptr_idxb[jj+2]] );
				u_tmp0 = _mm_loadh_pd( u_tmp0, &ptr_rq[ptr_idxb[jj+1]] );
				u_tmp1 = _mm_loadh_pd( u_tmp1, &ptr_rq[ptr_idxb[jj+3]] );
				v_tmp1 = _mm256_castpd128_pd256( u_tmp0 );
				v_tmp1 = _mm256_insertf128_pd( v_tmp1, u_tmp1, 0x1 );

				v_tmp0 = _mm256_add_pd( v_tmp0, v_tmp1 );

				u_tmp1 = _mm256_extractf128_pd( v_tmp0, 0x1 );
				u_tmp0 = _mm256_castpd256_pd128( v_tmp0 );

				_mm_store_sd( &ptr_rq[ptr_idxb[jj+0]], u_tmp0 );
				_mm_storeh_pd( &ptr_rq[ptr_idxb[jj+1]], u_tmp0 );
				_mm_store_sd( &ptr_rq[ptr_idxb[jj+2]], u_tmp1 );
				_mm_storeh_pd( &ptr_rq[ptr_idxb[jj+3]], u_tmp1 );

				u_tmp0 = _mm_load_sd( &ptr_ux[ptr_idxb[jj+0]] );
				u_tmp1 = _mm_load_sd( &ptr_ux[ptr_idxb[jj+2]] );
				u_tmp0 = _mm_loadh_pd( u_tmp0, &ptr_ux[ptr_idxb[jj+1]] );
				u_tmp1 = _mm_loadh_pd( u_tmp1, &ptr_ux[ptr_idxb[jj+3]] );
				v_ux   = _mm256_castpd128_pd256( u_tmp0 );
				v_ux   = _mm256_insertf128_pd( v_ux, u_tmp1, 0x1 );

				v_t0   = _mm256_loadu_pd( &ptr_t[jj+0] );
				v_t1   = _mm256_loadu_pd( &ptr_t[jj+pnb] );

				v_tmp0   = _mm256_loadu_pd( &ptr_d[jj+0] );
				v_tmp1   = _mm256_loadu_pd( &ptr_d[jj+pnb] );
				v_tmp0   = _mm256_sub_pd( v_tmp0, v_ux );
				v_tmp1   = _mm256_sub_pd( v_tmp1, v_ux );
				v_tmp0   = _mm256_add_pd( v_tmp0, v_t0 );
				v_tmp1   = _mm256_sub_pd( v_tmp1, v_t1 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+pnb], v_tmp1 );

				v_tmp0 = _mm256_mul_pd( v_lam0, v_t0 );
				v_tmp1 = _mm256_mul_pd( v_lam1, v_t1 );
				_mm256_storeu_pd( &ptr_rm[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rm[jj+pnb], v_tmp1 );
				v_mu0  = _mm256_add_pd( v_mu0, v_tmp0 );
				v_mu1  = _mm256_add_pd( v_mu1, v_tmp1 );
				}
			for(; jj<nb0; jj++)
				{
				u_lam0 = _mm_load_sd( &ptr_lam[jj+0] );
				u_lam1 = _mm_load_sd( &ptr_lam[jj+pnb] );
				u_tmp0 = _mm_sub_sd( u_lam1, u_lam0 );
				u_tmp1 = _mm_load_sd( &ptr_rq[ptr_idxb[jj]] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_tmp1 );
				_mm_store_sd( &ptr_rq[ptr_idxb[jj]], u_tmp0 );

				u_ux   = _mm_load_sd( &ptr_ux[ptr_idxb[jj]] );

				u_t0   = _mm_load_sd( &ptr_t[jj+0] );
				u_t1   = _mm_load_sd( &ptr_t[jj+pnb] );

				u_tmp0   = _mm_load_sd( &ptr_d[jj+0] );
				u_tmp1   = _mm_load_sd( &ptr_d[jj+pnb] );
				u_tmp0   = _mm_sub_sd( u_tmp0, u_ux );
				u_tmp1   = _mm_sub_sd( u_tmp1, u_ux );
				u_tmp0   = _mm_add_sd( u_tmp0, u_t0 );
				u_tmp1   = _mm_sub_sd( u_tmp1, u_t1 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+pnb], u_tmp1 );

				u_tmp0 = _mm_mul_sd( u_lam0, u_t0 );
				u_tmp1 = _mm_mul_sd( u_lam1, u_t1 );
				_mm_store_sd( &ptr_rm[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rm[jj+pnb], u_tmp1 );
				u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
				u_mu1  = _mm_add_sd( u_mu1, u_tmp1 );
				}
			}

		dsymv_l_libstr(nu0+nx0, nu0+nx0, 1.0, &hsQ[ii], 0, 0, &hsux[ii], 0, 1.0, &hsrq[ii], 0, &hsrq[ii], 0);

		for(jj=0; jj<nx1; jj++) // TODO
			hrb[ii+1][jj] = hb[ii+1][jj] - hux[ii+1][nu1+jj];

		dgemv_nt_libstr(nu0+nx0, nx1, 1.0, 1.0, &hsBAbt[ii], 0, 0, &hspi[ii+1], 0, &hsux[ii], 0, 1.0, 1.0, &hsrq[ii], 0, &hsrb[ii+1], 0, &hsrq[ii], 0, &hsrb[ii+1], 0);

		if(ng0>0)
			{

			ptr_d   += 2*pnb;
			ptr_lam += 2*pnb;
			ptr_t   += 2*pnb;
			ptr_rd  += 2*pnb;
			ptr_rm  += 2*pnb;

			nb_tot += ng0;

			for(jj=0; jj<ng0-3; jj+=4)
				{
				v_lam0 = _mm256_loadu_pd( &ptr_lam[jj+0] );
				v_lam1 = _mm256_loadu_pd( &ptr_lam[jj+png] );
				v_tmp0 = _mm256_sub_pd( v_lam1, v_lam0 );
				_mm256_storeu_pd( &work0[jj], v_tmp0 );

				v_t0   = _mm256_loadu_pd( &ptr_t[jj+0] );
				v_t1   = _mm256_loadu_pd( &ptr_t[jj+png] );
				v_tmp0 = _mm256_loadu_pd( &ptr_d[jj+0] );
				v_tmp1 = _mm256_loadu_pd( &ptr_d[jj+png] );
				v_tmp0 = _mm256_add_pd( v_tmp0, v_t0 );
				v_tmp1 = _mm256_sub_pd( v_tmp1, v_t1 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+png], v_tmp1 );

				v_tmp0 = _mm256_mul_pd( v_lam0, v_t0 );
				v_tmp1 = _mm256_mul_pd( v_lam1, v_t1 );
				_mm256_storeu_pd( &ptr_rm[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rm[jj+png], v_tmp1 );
				v_mu0  = _mm256_add_pd( v_mu0, v_tmp0 );
				v_mu1  = _mm256_add_pd( v_mu1, v_tmp1 );
				}
			for(; jj<ng0; jj++) // TODO mask ?
				{
				u_lam0 = _mm_load_sd( &ptr_lam[jj+0] );
				u_lam1 = _mm_load_sd( &ptr_lam[jj+png] );
				u_tmp0 = _mm_sub_sd( u_lam1, u_lam0 );
				_mm_store_sd( &work0[jj], u_tmp0 );

				u_t0   = _mm_load_sd( &ptr_t[jj+0] );
				u_t1   = _mm_load_sd( &ptr_t[jj+png] );
				u_tmp0 = _mm_load_sd( &ptr_d[jj+0] );
				u_tmp1 = _mm_load_sd( &ptr_d[jj+png] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_t0 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_t1 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+png], u_tmp1 );

				u_tmp0 = _mm_mul_sd( u_lam0, u_t0 );
				u_tmp1 = _mm_mul_sd( u_lam1, u_t1 );
				_mm_store_sd( &ptr_rm[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rm[jj+png], u_tmp1 );
				u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
				u_mu1  = _mm_add_sd( u_mu1, u_tmp1 );
				}

			dgemv_nt_libstr(nu0+nx0, ng0, 1.0, 1.0, &hsDCt[ii], 0, 0, &hswork[0], 0, &hsux[ii], 0, 1.0, 0.0, &hsrq[ii], 0, &hswork[1], 0, &hsrq[ii], 0, &hswork[1], 0);

			for(jj=0; jj<ng0-3; jj+=4)
				{
				v_tmp2 = _mm256_loadu_pd( &work1[jj] );
				v_tmp0 = _mm256_loadu_pd( &ptr_rd[jj+0] );
				v_tmp1 = _mm256_loadu_pd( &ptr_rd[jj+png] );
				v_tmp0 = _mm256_sub_pd( v_tmp0, v_tmp2 );
				v_tmp1 = _mm256_sub_pd( v_tmp1, v_tmp2 );
				_mm256_storeu_pd( &ptr_rd[jj+0], v_tmp0 );
				_mm256_storeu_pd( &ptr_rd[jj+png], v_tmp1 );
				}
			for(; jj<ng0; jj++) // TODO mask ?
				{
				u_tmp2 = _mm_load_sd( &work1[jj] );
				u_tmp0 = _mm_load_sd( &ptr_rd[jj+0] );
				u_tmp1 = _mm_load_sd( &ptr_rd[jj+png] );
				u_tmp0 = _mm_sub_sd( u_tmp0, u_tmp2 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_tmp2 );
				_mm_store_sd( &ptr_rd[jj+0], u_tmp0 );
				_mm_store_sd( &ptr_rd[jj+png], u_tmp1 );
				}

			}

		}
	


	// last stage
	ii = N;
	nu0 = nu1;
	nx0 = nx1;
	cnux0  = (nu0+nx0+ncl-1)/ncl*ncl;
	nb0 = nb[ii];
	pnb = nb0; (nb0+bs-1)/bs*bs;
	ng0 = ng[ii];
	png = ng0; //(ng0+bs-1)/bs*bs;
	cng = (ng0+ncl-1)/ncl*ncl;

	if(nb0>0 | ng0>0)
		{
		hd[ii] = hsd[ii].pa;
		hlam[ii] = hslam[ii].pa;
		ht[ii] = hst[ii].pa;
		hrd[ii] = hsrd[ii].pa;
		hrm[ii] = hsrm[ii].pa;
		}

	for(jj=0; jj<nx0; jj++) 
		hrq[ii][nu0+jj] = - hpi[ii][jj] + hq[ii][nu0+jj];

	if(nb0>0)
		{

		nb_tot += nb0;

		for(jj=0; jj<nb0; jj++) 
			{
			hrq[ii][idxb[ii][jj]] += - hlam[ii][jj] + hlam[ii][pnb+jj];

			hrd[ii][jj]     = hd[ii][jj]     - hux[ii][idxb[ii][jj]] + ht[ii][jj];
			hrd[ii][pnb+jj] = hd[ii][pnb+jj] - hux[ii][idxb[ii][jj]] - ht[ii][pnb+jj];

			hrm[ii][jj]     = hlam[ii][jj]     * ht[ii][jj];
			hrm[ii][pnb+jj] = hlam[ii][pnb+jj] * ht[ii][pnb+jj];
			mu2 += hrm[ii][jj] + hrm[ii][pnb+jj];
			}
		}

	dsymv_l_libstr(nx0+nu0%bs, nx0+nu0%bs, 1.0, &hsQ[ii], nu0/bs*bs, nu0/bs*bs, &hsux[ii], nu0/bs*bs, 1.0, &hsrq[ii], nu0/bs*bs, &hsrq[ii], nu0/bs*bs);
	
	if(ng0>0)
		{

		nb_tot += ng0;

		for(jj=0; jj<ng0; jj++)
			{
			work0[jj] = hlam[ii][jj+2*pnb+png] - hlam[ii][jj+2*pnb+0];

			hrd[ii][2*pnb+jj]     = hd[ii][2*pnb+jj]     + ht[ii][2*pnb+jj];
			hrd[ii][2*pnb+png+jj] = hd[ii][2*pnb+png+jj] - ht[ii][2*pnb+png+jj];

			hrm[ii][2*pnb+jj]     = hlam[ii][2*pnb+jj]     * ht[ii][2*pnb+jj];
			hrm[ii][2*pnb+png+jj] = hlam[ii][2*pnb+png+jj] * ht[ii][2*pnb+png+jj];
			mu2 += hrm[ii][2*pnb+jj] + hrm[ii][2*pnb+png+jj];
			}

		dgemv_nt_libstr(nu0+nx0, ng0, 1.0, 1.0, &hsDCt[ii], 0, 0, &hswork[0], 0, &hsux[ii], 0, 1.0, 0.0, &hsrq[ii], 0, &hswork[1], 0, &hsrq[ii], 0, &hswork[1], 0);

		for(jj=0; jj<ng0; jj++)
			{
			hrd[ii][2*pnb+jj]     -= work1[jj];
			hrd[ii][2*pnb+png+jj] -= work1[jj];
			}

		}


	// normalize mu
	double mu_scal = 0.0;
	if(nb_tot!=0)
		{
		mu_scal = 1.0 / (2.0*nb_tot);

		v_mu0  = _mm256_add_pd( v_mu0, v_mu1 );
		u_mu0  = _mm_add_sd( u_mu0, u_mu1 );
		u_tmp0 = _mm_add_pd( _mm256_castpd256_pd128( v_mu0 ), _mm256_extractf128_pd( v_mu0, 0x1 ) );
		u_tmp0 = _mm_hadd_pd( u_tmp0, u_tmp0);
		u_mu0  = _mm_add_sd( u_mu0, u_tmp0 );
		u_mu1  = _mm_load_sd( &mu2 );
		u_mu0  = _mm_add_sd( u_mu0, u_mu1 );
		u_tmp0 = _mm_load_sd( &mu_scal );
		u_mu0  = _mm_mul_sd( u_mu0, u_tmp0 );
		_mm_store_sd( &mu[0], u_mu0 );
		}



	return;

	}


