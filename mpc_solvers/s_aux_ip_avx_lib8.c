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

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4
#include <immintrin.h>  // AVX

#include "../include/block_size.h"



void s_init_ux_t_box_mpc(int N, int nu, int nbu, int nb, float **ux, float **db, float **t, int warm_start)
	{
	
	int jj, ll;
	
	float thr0 = 1e-3; // minimum distance from a constraint

	if(warm_start==1)
		{
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

		}
	else // cold start
		{
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

		}
	
	}



void s_init_lam_mpc(int N, int nu, int nbu, int nb, float **t, float **lam)	// TODO approximate reciprocal
	{
	
	int jj, ll;
	
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
	
	}



void s_update_hessian_box_mpc(int N, int k0, int k1, int kmax, int cnz, float sigma_mu, float **t, float **t_inv, float **lam, float **lamt, float **dlam, float **bd, float **bl, float **pd, float **pl, float **pl2, float **db)

/*void d_update_hessian_box(int k0, int kmax, int nb, int cnz, float sigma_mu, float *t, float *lam, float *lamt, float *dlam, float *bd, float *bl, float *pd, float *pl, float *lb, float *ub)*/
	{
	
	const int bs = 8; //d_get_mr();
	
	float temp0, temp1;
	
	float *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv, *ptr_pd, *ptr_pl, *ptr_pl2, *ptr_bd, *ptr_bl, *ptr_db;
	
	int ii, jj, ll, bs0;
	
	// first stage
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_tinv  = t_inv[0];
	ptr_pd    = pd[0];
	ptr_pl    = pl[0];
	ptr_pl2   = pl2[0];
	ptr_bd    = bd[0];
	ptr_bl    = bl[0];
	ptr_db    = db[0];
	
	ii = 0;
	for(; ii<k0-7; ii+=8)
		{

/*		v_tmp  = _mm256_load_ps( &ptr_t[0] );*/
/*		v_tmp  = _mm256_div_ps( v_ones, v_tmp );*/
/*		_mm256_store_ps( &ptr_t_inv[0], v_tmp ); // store t_inv*/
/*		v_lam  = _mm256_load_ps( &ptr_lam[0] );*/
/*		v_lamt = _mm256_mul_ps( v_tmp, v_lam );*/
/*		_mm256_store_ps( &ptr_lamt[0], v_lamt );*/
/*		v_dlam = _mm256_mul_ps( v_tmp, v_sigma_mu );*/
/*		_mm256_store_ps( &ptr_dlam[0], v_dlam );*/
/*		u_lamt = _mm256_extractf128_ps( v_lamt, 0x1 );*/
/*		u_lamt = _mm_hadd_ps( _mm256_castps256_ps128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] , lamt[4]+lamt[5] , lamt[6]+lamt[7] ]*/
/*		u_bd   = _mm_load_ps( &bd[0][ii] );*/
/*		u_bd   = _mm_add_ps( u_bd, u_lamt );*/
/*		_mm_store_ss( &pd[0][0+(ii+0)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe5 );*/
/*		_mm_store_ss( &pd[0][1+(ii+1)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe6 );*/
/*		_mm_store_ss( &pd[0][2+(ii+2)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe7 );*/
/*		_mm_store_ss( &pd[0][3+(ii+3)*bs+ii*cnz], u_bd );*/
/*		v_db   = _mm256_load_ps( &db[0][2*ii+0] );*/
/*		v_db   = _mm256_mul_ps( v_db, v_lamt );*/
/*		v_lam  = _mm256_add_ps( v_lam, v_dlam );*/
/*		v_lam  = _mm256_add_ps( v_lam, v_db );*/
/*		u_lam  = _mm256_extractf128_ps( v_lam, 0x1 );*/
/*		u_lam  = _mm_hsub_ps( _mm256_castps256_ps128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] , ... ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] , ... ]*/
/*		u_bl   = _mm_load_ps( &bl[0][ii] );*/
/*		u_bl   = _mm_sub_ps( u_bl, u_lam );*/
/*		_mm_store_ps( &pl2[0][ii+0], u_bl );*/
/*		_mm_store_ss( &pl[0][(ii+0)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe5 );*/
/*		_mm_store_ss( &pl[0][(ii+1)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe6 );*/
/*		_mm_store_ss( &pl[0][(ii+2)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe7 );*/
/*		_mm_store_ss( &pl[0][(ii+3)*bs], u_bl );*/

/*		v_tmp  = _mm256_load_ps( &ptr_t[8] );*/
/*		v_tmp  = _mm256_div_ps( v_ones, v_tmp );*/
/*		_mm256_store_ps( &ptr_t_inv[8], v_tmp ); // store t_inv*/
/*		v_lam  = _mm256_load_ps( &ptr_lam[8] );*/
/*		v_lamt = _mm256_mul_ps( v_tmp, v_lam );*/
/*		_mm256_store_ps( &ptr_lamt[8], v_lamt );*/
/*		v_dlam = _mm256_mul_ps( v_tmp, v_sigma_mu );*/
/*		_mm256_store_ps( &ptr_dlam[8], v_dlam );*/
/*		u_lamt = _mm256_extractf128_ps( v_lamt, 0x1 );*/
/*		u_lamt = _mm_hadd_ps( _mm256_castps256_ps128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]*/
/*		u_bd   = _mm_load_ps( &bd[0][ii+4] );*/
/*		u_bd   = _mm_add_ps( u_bd, u_lamt );*/
/*		_mm_store_ss( &pd[0][4+(ii+4)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe5 );*/
/*		_mm_store_ss( &pd[0][5+(ii+5)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe6 );*/
/*		_mm_store_ss( &pd[0][6+(ii+6)*bs+ii*cnz], u_bd );*/
/*		_mm_shuffle_ps( u_bd, u_bd, 0xe7 );*/
/*		_mm_store_ss( &pd[0][7+(ii+7)*bs+ii*cnz], u_bd );*/
/*		v_db   = _mm256_load_ps( &db[0][2*ii+8] );*/
/*		v_db   = _mm256_mul_ps( v_db, v_lamt );*/
/*		v_lam  = _mm256_add_ps( v_lam, v_dlam );*/
/*		v_lam  = _mm256_add_ps( v_lam, v_db );*/
/*		u_lam  = _mm256_extractf128_ps( v_lam, 0x1 );*/
/*		u_lam  = _mm_hsub_ps( _mm256_castps256_ps128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]*/
/*		u_bl   = _mm_load_ps( &bl[0][ii+4] );*/
/*		u_bl   = _mm_sub_ps( u_bl, u_lam );*/
/*		_mm_store_ps( &pl2[0][ii+4], u_bl );*/
/*		_mm_store_ss( &pl[0][(ii+4)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe5 );*/
/*		_mm_store_ss( &pl[0][(ii+5)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe6 );*/
/*		_mm_store_ss( &pl[0][(ii+6)*bs], u_bl );*/
/*		_mm_shuffle_ps( u_bl, u_bl, 0xe7 );*/
/*		_mm_store_ss( &pl[0][(ii+7)*bs], u_bl );*/




		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[0+(ii+0)*bs+ii*cnz] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[(ii+0)*bs] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+0] - ptr_dlam[0];
		ptr_pl2[ii+0] = ptr_pl[(ii+0)*bs];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[1+(ii+1)*bs+ii*cnz] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[(ii+1)*bs] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[2*ii+3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2*ii+2] - ptr_dlam[2];
		ptr_pl2[ii+1] = ptr_pl[(ii+1)*bs];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[2+(ii+2)*bs+ii*cnz] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[(ii+2)*bs] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[2*ii+5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[2*ii+4] - ptr_dlam[4];
		ptr_pl2[ii+2] = ptr_pl[(ii+2)*bs];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[3+(ii+3)*bs+ii*cnz] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[(ii+3)*bs] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[2*ii+7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[2*ii+6] - ptr_dlam[6];
		ptr_pl2[ii+3] = ptr_pl[(ii+3)*bs];

		ptr_tinv[8] = 1.0/ptr_t[8];
		ptr_tinv[9] = 1.0/ptr_t[9];
		ptr_lamt[8] = ptr_lam[8]*ptr_tinv[8];
		ptr_lamt[9] = ptr_lam[9]*ptr_tinv[9];
		ptr_dlam[8] = ptr_tinv[8]*sigma_mu; // !!!!!
		ptr_dlam[9] = ptr_tinv[9]*sigma_mu; // !!!!!
		ptr_pd[4+(ii+4)*bs+ii*cnz] = ptr_bd[ii+4] + ptr_lamt[8] + ptr_lamt[9];
		ptr_pl[(ii+4)*bs] = ptr_bl[ii+4] + ptr_lam[9] + ptr_lamt[9]*ptr_db[2*ii+9] + ptr_dlam[9] - ptr_lam[8] - ptr_lamt[8]*ptr_db[2*ii+8] - ptr_dlam[8];
		ptr_pl2[ii+4] = ptr_pl[(ii+4)*bs];

		ptr_tinv[10] = 1.0/ptr_t[10];
		ptr_tinv[11] = 1.0/ptr_t[11];
		ptr_lamt[10] = ptr_lam[10]*ptr_tinv[10];
		ptr_lamt[11] = ptr_lam[11]*ptr_tinv[11];
		ptr_dlam[10] = ptr_tinv[10]*sigma_mu; // !!!!!
		ptr_dlam[11] = ptr_tinv[11]*sigma_mu; // !!!!!
		ptr_pd[5+(ii+5)*bs+ii*cnz] = ptr_bd[ii+5] + ptr_lamt[10] + ptr_lamt[11];
		ptr_pl[(ii+5)*bs] = ptr_bl[ii+5] + ptr_lam[11] + ptr_lamt[11]*ptr_db[2*ii+11] + ptr_dlam[11] - ptr_lam[10] - ptr_lamt[10]*ptr_db[2*ii+10] - ptr_dlam[10];
		ptr_pl2[ii+5] = ptr_pl[(ii+5)*bs];

		ptr_tinv[12] = 1.0/ptr_t[12];
		ptr_tinv[13] = 1.0/ptr_t[13];
		ptr_lamt[12] = ptr_lam[12]*ptr_tinv[12];
		ptr_lamt[13] = ptr_lam[13]*ptr_tinv[13];
		ptr_dlam[12] = ptr_tinv[12]*sigma_mu; // !!!!!
		ptr_dlam[13] = ptr_tinv[13]*sigma_mu; // !!!!!
		ptr_pd[6+(ii+6)*bs+ii*cnz] = ptr_bd[ii+6] + ptr_lamt[12] + ptr_lamt[13];
		ptr_pl[(ii+6)*bs] = ptr_bl[ii+6] + ptr_lam[13] + ptr_lamt[13]*ptr_db[2*ii+13] + ptr_dlam[13] - ptr_lam[12] - ptr_lamt[12]*ptr_db[2*ii+12] - ptr_dlam[12];
		ptr_pl2[ii+6] = ptr_pl[(ii+6)*bs];

		ptr_tinv[14] = 1.0/ptr_t[14];
		ptr_tinv[15] = 1.0/ptr_t[15];
		ptr_lamt[14] = ptr_lam[14]*ptr_tinv[14];
		ptr_lamt[15] = ptr_lam[15]*ptr_tinv[15];
		ptr_dlam[14] = ptr_tinv[14]*sigma_mu; // !!!!!
		ptr_dlam[15] = ptr_tinv[15]*sigma_mu; // !!!!!
		ptr_pd[7+(ii+7)*bs+ii*cnz] = ptr_bd[ii+7] + ptr_lamt[14] + ptr_lamt[15];
		ptr_pl[(ii+7)*bs] = ptr_bl[ii+7] + ptr_lam[15] + ptr_lamt[15]*ptr_db[2*ii+15] + ptr_dlam[15] - ptr_lam[14] - ptr_lamt[14]*ptr_db[2*ii+14] - ptr_dlam[14];
		ptr_pl2[ii+7] = ptr_pl[(ii+7)*bs];

		ptr_t     += 16;
		ptr_lam   += 16;
		ptr_lamt  += 16;
		ptr_dlam  += 16;
		ptr_tinv  += 16;

		}
	if(ii<k0)
		{
		bs0 = k0-ii;
		for(ll=0; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ll+(ii+ll)*bs+ii*cnz] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[(ii+ll)*bs] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+2*ll+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+2*ll+0] - ptr_dlam[0];
			ptr_pl2[ii+ll+0] = ptr_pl[(ii+ll)*bs];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			}
		}

	// middle stages

	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_tinv  = t_inv[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];
		ptr_pl2   = pl2[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_db    = db[jj];

		ii = 0;
		for(; ii<kmax-7; ii+=8)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[0+(ii+0)*bs+ii*cnz] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[(ii+0)*bs] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+0] - ptr_dlam[0];
			ptr_pl2[ii+0] = ptr_pl[(ii+0)*bs];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_pd[1+(ii+1)*bs+ii*cnz] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
			ptr_pl[(ii+1)*bs] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[2*ii+3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2*ii+2] - ptr_dlam[2];
			ptr_pl2[ii+1] = ptr_pl[(ii+1)*bs];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_pd[2+(ii+2)*bs+ii*cnz] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
			ptr_pl[(ii+2)*bs] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[2*ii+5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[2*ii+4] - ptr_dlam[4];
			ptr_pl2[ii+2] = ptr_pl[(ii+2)*bs];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_pd[3+(ii+3)*bs+ii*cnz] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
			ptr_pl[(ii+3)*bs] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[2*ii+7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[2*ii+6] - ptr_dlam[6];
			ptr_pl2[ii+3] = ptr_pl[(ii+3)*bs];

			ptr_tinv[8] = 1.0/ptr_t[8];
			ptr_tinv[9] = 1.0/ptr_t[9];
			ptr_lamt[8] = ptr_lam[8]*ptr_tinv[8];
			ptr_lamt[9] = ptr_lam[9]*ptr_tinv[9];
			ptr_dlam[8] = ptr_tinv[8]*sigma_mu; // !!!!!
			ptr_dlam[9] = ptr_tinv[9]*sigma_mu; // !!!!!
			ptr_pd[4+(ii+4)*bs+ii*cnz] = ptr_bd[ii+4] + ptr_lamt[8] + ptr_lamt[9];
			ptr_pl[(ii+4)*bs] = ptr_bl[ii+4] + ptr_lam[9] + ptr_lamt[9]*ptr_db[2*ii+9] + ptr_dlam[9] - ptr_lam[8] - ptr_lamt[8]*ptr_db[2*ii+8] - ptr_dlam[8];
			ptr_pl2[ii+4] = ptr_pl[(ii+4)*bs];

			ptr_tinv[10] = 1.0/ptr_t[10];
			ptr_tinv[11] = 1.0/ptr_t[11];
			ptr_lamt[10] = ptr_lam[10]*ptr_tinv[10];
			ptr_lamt[11] = ptr_lam[11]*ptr_tinv[11];
			ptr_dlam[10] = ptr_tinv[10]*sigma_mu; // !!!!!
			ptr_dlam[11] = ptr_tinv[11]*sigma_mu; // !!!!!
			ptr_pd[5+(ii+5)*bs+ii*cnz] = ptr_bd[ii+5] + ptr_lamt[10] + ptr_lamt[11];
			ptr_pl[(ii+5)*bs] = ptr_bl[ii+5] + ptr_lam[11] + ptr_lamt[11]*ptr_db[2*ii+11] + ptr_dlam[11] - ptr_lam[10] - ptr_lamt[10]*ptr_db[2*ii+10] - ptr_dlam[10];
			ptr_pl2[ii+5] = ptr_pl[(ii+5)*bs];

			ptr_tinv[12] = 1.0/ptr_t[12];
			ptr_tinv[13] = 1.0/ptr_t[13];
			ptr_lamt[12] = ptr_lam[12]*ptr_tinv[12];
			ptr_lamt[13] = ptr_lam[13]*ptr_tinv[13];
			ptr_dlam[12] = ptr_tinv[12]*sigma_mu; // !!!!!
			ptr_dlam[13] = ptr_tinv[13]*sigma_mu; // !!!!!
			ptr_pd[6+(ii+6)*bs+ii*cnz] = ptr_bd[ii+6] + ptr_lamt[12] + ptr_lamt[13];
			ptr_pl[(ii+6)*bs] = ptr_bl[ii+6] + ptr_lam[13] + ptr_lamt[13]*ptr_db[2*ii+13] + ptr_dlam[13] - ptr_lam[12] - ptr_lamt[12]*ptr_db[2*ii+12] - ptr_dlam[12];
			ptr_pl2[ii+6] = ptr_pl[(ii+6)*bs];

			ptr_tinv[14] = 1.0/ptr_t[14];
			ptr_tinv[15] = 1.0/ptr_t[15];
			ptr_lamt[14] = ptr_lam[14]*ptr_tinv[14];
			ptr_lamt[15] = ptr_lam[15]*ptr_tinv[15];
			ptr_dlam[14] = ptr_tinv[14]*sigma_mu; // !!!!!
			ptr_dlam[15] = ptr_tinv[15]*sigma_mu; // !!!!!
			ptr_pd[7+(ii+7)*bs+ii*cnz] = ptr_bd[ii+7] + ptr_lamt[14] + ptr_lamt[15];
			ptr_pl[(ii+7)*bs] = ptr_bl[ii+7] + ptr_lam[15] + ptr_lamt[15]*ptr_db[2*ii+15] + ptr_dlam[15] - ptr_lam[14] - ptr_lamt[14]*ptr_db[2*ii+14] - ptr_dlam[14];
			ptr_pl2[ii+7] = ptr_pl[(ii+7)*bs];

			ptr_t     += 16;
			ptr_lam   += 16;
			ptr_lamt  += 16;
			ptr_dlam  += 16;
			ptr_tinv  += 16;

			}
		if(ii<kmax)
			{
			bs0 = kmax-ii;
			for(ll=0; ll<bs0; ll++)
				{
				ptr_tinv[0] = 1.0/ptr_t[0];
				ptr_tinv[1] = 1.0/ptr_t[1];
				ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
				ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
				ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
				ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
				ptr_pd[ll+(ii+ll)*bs+ii*cnz] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
				ptr_pl[(ii+ll)*bs] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+2*ll+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+2*ll+0] - ptr_dlam[0];
				ptr_pl2[ii+ll+0] = ptr_pl[(ii+ll)*bs];

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_tinv  += 2;
				}
			}
	
		}

	// last stage

	ptr_t     = t[N]     + 2*k1;
	ptr_lam   = lam[N]   + 2*k1;
	ptr_lamt  = lamt[N]  + 2*k1;
	ptr_dlam  = dlam[N]  + 2*k1;
	ptr_tinv  = t_inv[N] + 2*k1;
	ptr_pd    = pd[N];
	ptr_pl    = pl[N];
	ptr_pl2   = pl2[N];
	ptr_bd    = bd[N];
	ptr_bl    = bl[N];
	ptr_db    = db[N];

	ii=k1; // k1 supposed to be multiple of bs !!!!!!!!!!

	for(; ii<kmax-7; ii+=8)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[0+(ii+0)*bs+ii*cnz] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[(ii+0)*bs] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+0] - ptr_dlam[0];
		ptr_pl2[ii+0] = ptr_pl[(ii+0)*bs];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[1+(ii+1)*bs+ii*cnz] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[(ii+1)*bs] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[2*ii+3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2*ii+2] - ptr_dlam[2];
		ptr_pl2[ii+1] = ptr_pl[(ii+1)*bs];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[2+(ii+2)*bs+ii*cnz] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[(ii+2)*bs] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[2*ii+5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[2*ii+4] - ptr_dlam[4];
		ptr_pl2[ii+2] = ptr_pl[(ii+2)*bs];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[3+(ii+3)*bs+ii*cnz] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[(ii+3)*bs] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[2*ii+7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[2*ii+6] - ptr_dlam[6];
		ptr_pl2[ii+3] = ptr_pl[(ii+3)*bs];

		ptr_tinv[8] = 1.0/ptr_t[8];
		ptr_tinv[9] = 1.0/ptr_t[9];
		ptr_lamt[8] = ptr_lam[8]*ptr_tinv[8];
		ptr_lamt[9] = ptr_lam[9]*ptr_tinv[9];
		ptr_dlam[8] = ptr_tinv[8]*sigma_mu; // !!!!!
		ptr_dlam[9] = ptr_tinv[9]*sigma_mu; // !!!!!
		ptr_pd[4+(ii+4)*bs+ii*cnz] = ptr_bd[ii+4] + ptr_lamt[8] + ptr_lamt[9];
		ptr_pl[(ii+4)*bs] = ptr_bl[ii+4] + ptr_lam[9] + ptr_lamt[9]*ptr_db[2*ii+9] + ptr_dlam[9] - ptr_lam[8] - ptr_lamt[8]*ptr_db[2*ii+8] - ptr_dlam[8];
		ptr_pl2[ii+4] = ptr_pl[(ii+4)*bs];

		ptr_tinv[10] = 1.0/ptr_t[10];
		ptr_tinv[11] = 1.0/ptr_t[11];
		ptr_lamt[10] = ptr_lam[10]*ptr_tinv[10];
		ptr_lamt[11] = ptr_lam[11]*ptr_tinv[11];
		ptr_dlam[10] = ptr_tinv[10]*sigma_mu; // !!!!!
		ptr_dlam[11] = ptr_tinv[11]*sigma_mu; // !!!!!
		ptr_pd[5+(ii+5)*bs+ii*cnz] = ptr_bd[ii+5] + ptr_lamt[10] + ptr_lamt[11];
		ptr_pl[(ii+5)*bs] = ptr_bl[ii+5] + ptr_lam[11] + ptr_lamt[11]*ptr_db[2*ii+11] + ptr_dlam[11] - ptr_lam[10] - ptr_lamt[10]*ptr_db[2*ii+10] - ptr_dlam[10];
		ptr_pl2[ii+5] = ptr_pl[(ii+5)*bs];

		ptr_tinv[12] = 1.0/ptr_t[12];
		ptr_tinv[13] = 1.0/ptr_t[13];
		ptr_lamt[12] = ptr_lam[12]*ptr_tinv[12];
		ptr_lamt[13] = ptr_lam[13]*ptr_tinv[13];
		ptr_dlam[12] = ptr_tinv[12]*sigma_mu; // !!!!!
		ptr_dlam[13] = ptr_tinv[13]*sigma_mu; // !!!!!
		ptr_pd[6+(ii+6)*bs+ii*cnz] = ptr_bd[ii+6] + ptr_lamt[12] + ptr_lamt[13];
		ptr_pl[(ii+6)*bs] = ptr_bl[ii+6] + ptr_lam[13] + ptr_lamt[13]*ptr_db[2*ii+13] + ptr_dlam[13] - ptr_lam[12] - ptr_lamt[12]*ptr_db[2*ii+12] - ptr_dlam[12];
		ptr_pl2[ii+6] = ptr_pl[(ii+6)*bs];

		ptr_tinv[14] = 1.0/ptr_t[14];
		ptr_tinv[15] = 1.0/ptr_t[15];
		ptr_lamt[14] = ptr_lam[14]*ptr_tinv[14];
		ptr_lamt[15] = ptr_lam[15]*ptr_tinv[15];
		ptr_dlam[14] = ptr_tinv[14]*sigma_mu; // !!!!!
		ptr_dlam[15] = ptr_tinv[15]*sigma_mu; // !!!!!
		ptr_pd[7+(ii+7)*bs+ii*cnz] = ptr_bd[ii+7] + ptr_lamt[14] + ptr_lamt[15];
		ptr_pl[(ii+7)*bs] = ptr_bl[ii+7] + ptr_lam[15] + ptr_lamt[15]*ptr_db[2*ii+15] + ptr_dlam[15] - ptr_lam[14] - ptr_lamt[14]*ptr_db[2*ii+14] - ptr_dlam[14];
		ptr_pl2[ii+7] = ptr_pl[(ii+7)*bs];

		ptr_t     += 16;
		ptr_lam   += 16;
		ptr_lamt  += 16;
		ptr_dlam  += 16;
		ptr_tinv  += 16;

		}
	if(ii<kmax)
		{
		bs0 = kmax-ii;
		for(ll=0; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ll+(ii+ll)*bs+ii*cnz] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[(ii+ll)*bs] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+2*ll+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+2*ll+0] - ptr_dlam[0];
			ptr_pl2[ii+ll+0] = ptr_pl[(ii+ll)*bs];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			}
		}


	}



void s_compute_alpha_box_mpc(int N, int k0, int k1, int kmax, float *ptr_alpha, float **t, float **dt, float **lam, float **dlam, float **lamt, float **dux, float **db)
	{
	
	const int bs = 8; //d_get_mr();
	
	float alpha = ptr_alpha[0];
	
	int kna = ((k1+bs-1)/bs)*bs;

	int jj, ll;


	// first stage

	ll = 0;
	for(; ll<k0; ll+=2)
		{

		dt[0][ll+0] =   dux[0][ll/2] - db[0][ll+0];
		dt[0][ll+1] = - dux[0][ll/2] - db[0][ll+1];
		dlam[0][ll+0] -= lamt[0][ll+0] * dt[0][ll+0];
		dlam[0][ll+1] -= lamt[0][ll+1] * dt[0][ll+1];
		if( -alpha*dlam[0][ll+0]>lam[0][ll+0] )
			{
			alpha = - lam[0][ll+0] / dlam[0][ll+0];
			}
		if( -alpha*dlam[0][ll+1]>lam[0][ll+1] )
			{
			alpha = - lam[0][ll+1] / dlam[0][ll+1];
			}
		dt[0][ll+0] -= t[0][ll+0];
		dt[0][ll+1] -= t[0][ll+1];
		if( -alpha*dt[0][ll+0]>t[0][ll+0] )
			{
			alpha = - t[0][ll+0] / dt[0][ll+0];
			}
		if( -alpha*dt[0][ll+1]>t[0][ll+1] )
			{
			alpha = - t[0][ll+1] / dt[0][ll+1];
			}

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		for(; ll<kmax; ll+=2)
			{

			dt[jj][ll+0] =   dux[jj][ll/2] - db[jj][ll+0];
			dt[jj][ll+1] = - dux[jj][ll/2] - db[jj][ll+1];
			dlam[jj][ll+0] -= lamt[jj][ll+0] * dt[jj][ll+0];
			dlam[jj][ll+1] -= lamt[jj][ll+1] * dt[jj][ll+1];
			if( -alpha*dlam[jj][ll+0]>lam[jj][ll+0] )
				{
				alpha = - lam[jj][ll+0] / dlam[jj][ll+0];
				}
			if( -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )
				{
				alpha = - lam[jj][ll+1] / dlam[jj][ll+1];
				}
			dt[jj][ll+0] -= t[jj][ll+0];
			dt[jj][ll+1] -= t[jj][ll+1];
			if( -alpha*dt[jj][ll+0]>t[jj][ll+0] )
				{
				alpha = - t[jj][ll+0] / dt[jj][ll+0];
				}
			if( -alpha*dt[jj][ll+1]>t[jj][ll+1] )
				{
				alpha = - t[jj][ll+1] / dt[jj][ll+1];
				}

			}

		}		

	// last stage
	ll = k1;
	for(; ll<kmax; ll+=2)
		{

		dt[N][ll+0] =   dux[N][ll/2] - db[N][ll+0];
		dt[N][ll+1] = - dux[N][ll/2] - db[N][ll+1];
		dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];
		dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];
		if( -alpha*dlam[N][ll+0]>lam[N][ll+0] )
			{
			alpha = - lam[N][ll+0] / dlam[N][ll+0];
			}
		if( -alpha*dlam[N][ll+1]>lam[N][ll+1] )
			{
			alpha = - lam[N][ll+1] / dlam[N][ll+1];
			}
		dt[N][ll+0] -= t[N][ll+0];
		dt[N][ll+1] -= t[N][ll+1];
		if( -alpha*dt[N][ll+0]>t[N][ll+0] )
			{
			alpha = - t[N][ll+0] / dt[N][ll+0];
			}
		if( -alpha*dt[N][ll+1]>t[N][ll+1] )
			{
			alpha = - t[N][ll+1] / dt[N][ll+1];
			}

		}
	
	ptr_alpha[0] = alpha;

	return;
	
	}



void s_update_var_mpc(int nx, int nu, int N, int nb, int nbu, float *ptr_mu, float mu_scal, float alpha, float **ux, float **dux, float **t, float **dt, float **lam, float **dlam, float **pi, float **dpi)
	{
	
	int 
		jj, ll, ll_left;
	
	float 
		ll_left_f;

	__m128
		u_mu, u_tmp;

	__m256
		mask, zeros, alpha_mask,
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_ps( alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_ps();

	zeros = _mm256_setzero_ps();

	const float mask_f[] = {7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5};
	mask = _mm256_loadu_ps( mask_f );



	// update inputs
	ll = 0;
	for(; ll<nu-7; ll+=8)
		{
		v_ux  = _mm256_load_ps( &ux[jj][ll] );
		v_dux = _mm256_load_ps( &dux[jj][ll] );
		v_dux = _mm256_sub_ps( v_dux, v_ux );
		v_dux = _mm256_mul_ps( v_alpha, v_dux );
		v_ux  = _mm256_add_ps( v_ux, v_dux );
		_mm256_store_ps( &ux[jj][ll], v_ux );
		}
	ll_left = nu - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_ux  = _mm256_load_ps( &ux[jj][ll] );
		v_dux = _mm256_load_ps( &dux[jj][ll] );
		v_dux = _mm256_sub_ps( v_dux, v_ux );
		v_dux = _mm256_mul_ps( alpha_mask, v_dux );
		v_ux  = _mm256_add_ps( v_ux, v_dux );
		_mm256_store_ps( &ux[jj][ll], v_ux );
		}

	// box constraints
	ll = 0;
	for(; ll<2*nbu-7; ll+=8)
		{
		v_t    = _mm256_load_ps( &t[0][ll] );
		v_lam  = _mm256_load_ps( &lam[0][ll] );
		v_dt   = _mm256_load_ps( &dt[0][ll] );
		v_dlam = _mm256_load_ps( &dlam[0][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
		_mm256_store_ps( &t[0][ll], v_t );
		_mm256_store_ps( &lam[0][ll], v_lam );
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}
	ll_left = 2*nbu - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_t    = _mm256_load_ps( &t[0][ll] );
		v_lam  = _mm256_load_ps( &lam[0][ll] );
		v_dt   = _mm256_load_ps( &dt[0][ll] );
		v_dlam = _mm256_load_ps( &dlam[0][ll] );
		v_dt   = _mm256_mul_ps( alpha_mask, v_dt );
		v_dlam = _mm256_mul_ps( alpha_mask, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
		_mm256_store_ps( &t[0][ll], v_t );
		_mm256_store_ps( &lam[0][ll], v_lam );
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}


	for(jj=1; jj<N; jj++)
		{
		ll = 0;
		for(; ll<nu+nx-7; ll+=8)
			{
			v_ux  = _mm256_load_ps( &ux[jj][ll] );
			v_dux = _mm256_load_ps( &dux[jj][ll] );
			v_dux = _mm256_sub_ps( v_dux, v_ux );
			v_dux = _mm256_mul_ps( v_alpha, v_dux );
			v_ux  = _mm256_add_ps( v_ux, v_dux );
			_mm256_store_ps( &ux[jj][ll], v_ux );
			}
		ll_left = nu + nx - ll;
		if( ll_left>0 )
			{
			ll_left_f = 8.0 - ll_left;
			alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_ux  = _mm256_load_ps( &ux[jj][ll] );
			v_dux = _mm256_load_ps( &dux[jj][ll] );
			v_dux = _mm256_sub_ps( v_dux, v_ux );
			v_dux = _mm256_mul_ps( alpha_mask, v_dux );
			v_ux  = _mm256_add_ps( v_ux, v_dux );
			_mm256_store_ps( &ux[jj][ll], v_ux );
			}

		// update equality constrained multipliers
		ll = 0;
		for(; ll<nx-7; ll+=8)
			{
			v_pi  = _mm256_load_ps( &pi[jj][ll] );
			v_dpi = _mm256_load_ps( &dpi[jj][ll] );
			v_dpi = _mm256_sub_ps( v_dpi, v_pi );
			v_dpi = _mm256_mul_ps( v_alpha, v_dpi );
			v_pi  = _mm256_add_ps( v_pi, v_dpi );
			_mm256_store_ps( &pi[jj][ll], v_pi );
			}
		ll_left = nx - ll;
		if( ll_left>0 )
			{
			ll_left_f = 8.0 - ll_left;
			alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_pi  = _mm256_load_ps( &pi[jj][ll] );
			v_dpi = _mm256_load_ps( &dpi[jj][ll] );
			v_dpi = _mm256_sub_ps( v_dpi, v_pi );
			v_dpi = _mm256_mul_ps( alpha_mask, v_dpi );
			v_pi  = _mm256_add_ps( v_pi, v_dpi );
			_mm256_store_ps( &pi[jj][ll], v_pi );
			}

		// box constraints
		ll = 0;
		for(; ll<2*nb-7; ll+=8)
			{
			v_t    = _mm256_load_ps( &t[jj][ll] );
			v_lam  = _mm256_load_ps( &lam[jj][ll] );
			v_dt   = _mm256_load_ps( &dt[jj][ll] );
			v_dlam = _mm256_load_ps( &dlam[jj][ll] );
			v_dt   = _mm256_mul_ps( v_alpha, v_dt );
			v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
			v_t    = _mm256_add_ps( v_t, v_dt );
			v_lam  = _mm256_add_ps( v_lam, v_dlam );
			_mm256_store_ps( &t[jj][ll], v_t );
			_mm256_store_ps( &lam[jj][ll], v_lam );
			v_lam  = _mm256_mul_ps( v_lam, v_t );
			v_mu   = _mm256_add_ps( v_mu, v_lam );
			}
		ll_left = 2*nb - ll;
		if( ll_left>0 )
			{
			ll_left_f = 8.0 - ll_left;
			alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_t    = _mm256_load_ps( &t[jj][ll] );
			v_lam  = _mm256_load_ps( &lam[jj][ll] );
			v_dt   = _mm256_load_ps( &dt[jj][ll] );
			v_dlam = _mm256_load_ps( &dlam[jj][ll] );
			v_dt   = _mm256_mul_ps( alpha_mask, v_dt );
			v_dlam = _mm256_mul_ps( alpha_mask, v_dlam );
			v_t    = _mm256_add_ps( v_t, v_dt );
			v_lam  = _mm256_add_ps( v_lam, v_dlam );
			_mm256_store_ps( &t[jj][ll], v_t );
			_mm256_store_ps( &lam[jj][ll], v_lam );
			v_lam  = _mm256_mul_ps( v_lam, v_t );
			v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_mu   = _mm256_add_ps( v_mu, v_lam );
			}

		}

	// update states
	ll = 0;
	for(; ll<nx-7; ll+=8)
		{
		v_ux  = _mm256_loadu_ps( &ux[N][nu+ll] );
		v_dux = _mm256_loadu_ps( &dux[N][nu+ll] );
		v_dux = _mm256_sub_ps( v_dux, v_ux );
		v_dux = _mm256_mul_ps( v_alpha, v_dux );
		v_ux  = _mm256_add_ps( v_ux, v_dux );
		_mm256_storeu_ps( &ux[N][nu+ll], v_ux );
		}
	ll_left = nx - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_ux  = _mm256_loadu_ps( &ux[N][nu+ll] );
		v_dux = _mm256_loadu_ps( &dux[N][nu+ll] );
		v_dux = _mm256_sub_ps( v_dux, v_ux );
		v_dux = _mm256_mul_ps( alpha_mask, v_dux );
		v_ux  = _mm256_add_ps( v_ux, v_dux );
		_mm256_storeu_ps( &ux[N][nu+ll], v_ux );
		}

	// update equality constrained multipliers
	ll = 0;
	for(; ll<nx-7; ll+=8)
		{
		v_pi  = _mm256_load_ps( &pi[N][ll] );
		v_dpi = _mm256_load_ps( &dpi[N][ll] );
		v_dpi = _mm256_sub_ps( v_dpi, v_pi );
		v_dpi = _mm256_mul_ps( v_alpha, v_dpi );
		v_pi  = _mm256_add_ps( v_pi, v_dpi );
		_mm256_store_ps( &pi[N][ll], v_pi );
		}
	ll_left = nx - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_pi  = _mm256_load_ps( &pi[N][ll] );
		v_dpi = _mm256_load_ps( &dpi[N][ll] );
		v_dpi = _mm256_sub_ps( v_dpi, v_pi );
		v_dpi = _mm256_mul_ps( alpha_mask, v_dpi );
		v_dpi  = _mm256_add_ps( v_pi, v_dpi );
		v_pi = _mm256_blendv_ps( v_dpi, v_pi, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		_mm256_store_ps( &pi[N][ll], v_pi );
		}

	// box constraints
	ll = 2*nu;
	for(; ll<2*nb-7; ll+=8)
		{
		v_t    = _mm256_loadu_ps( &t[N][ll] );
		v_lam  = _mm256_loadu_ps( &lam[N][ll] );
		v_dt   = _mm256_loadu_ps( &dt[N][ll] );
		v_dlam = _mm256_loadu_ps( &dlam[N][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
		_mm256_storeu_ps( &t[N][ll], v_t );
		_mm256_storeu_ps( &lam[N][ll], v_lam );
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}
	ll_left = 2*nb - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_t    = _mm256_loadu_ps( &t[N][ll] );
		v_lam  = _mm256_loadu_ps( &lam[N][ll] );
		v_dt   = _mm256_loadu_ps( &dt[N][ll] );
		v_dlam = _mm256_loadu_ps( &dlam[N][ll] );
		v_dt   = _mm256_mul_ps( alpha_mask, v_dt );
		v_dlam = _mm256_mul_ps( alpha_mask, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
		_mm256_storeu_ps( &t[N][ll], v_t );
		_mm256_storeu_ps( &lam[N][ll], v_lam );
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}

	u_tmp = _mm256_extractf128_ps( v_mu, 0x1 );
	u_mu  = _mm_add_ps( u_tmp, _mm256_castps256_ps128( v_mu ) );

	u_mu  = _mm_hadd_ps( u_mu, u_mu );
	u_mu  = _mm_hadd_ps( u_mu, u_mu );

	u_tmp = _mm_load_ss( &mu_scal );
	u_mu  = _mm_mul_ss( u_mu, u_tmp );
	_mm_store_ss( ptr_mu, u_mu );

	return;
	
	}



void s_compute_mu_mpc(int N, int nbu, int nu, int nb, float *ptr_mu, float mu_scal, float alpha, float **lam, float **dlam, float **t, float **dt)
	{

	int 
		jj, ll, ll_left;
	
	float 
		ll_left_f;

	__m128
		u_mu, u_tmp;

	__m256
		mask, zeros, alpha_mask,
		v_alpha, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_ps( alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_ps();

	zeros = _mm256_setzero_ps();

	const float mask_f[] = {7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5};
	mask = _mm256_loadu_ps( mask_f );

	// box constraints
	ll = 0;
	for(; ll<2*nbu-7; ll+=8)
		{
		v_t    = _mm256_load_ps( &t[0][ll] );
		v_lam  = _mm256_load_ps( &lam[0][ll] );
		v_dt   = _mm256_load_ps( &dt[0][ll] );
		v_dlam = _mm256_load_ps( &dlam[0][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*		_mm256_store_ps( &t[0][ll], v_t );*/
/*		_mm256_store_ps( &lam[0][ll], v_lam );*/
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}
	ll_left = 2*nbu - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
/*		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );*/
		v_t    = _mm256_load_ps( &t[0][ll] );
		v_lam  = _mm256_load_ps( &lam[0][ll] );
		v_dt   = _mm256_load_ps( &dt[0][ll] );
		v_dlam = _mm256_load_ps( &dlam[0][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*		_mm256_store_ps( &t[0][ll], v_t );*/
/*		_mm256_store_ps( &lam[0][ll], v_lam );*/
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}


	for(jj=1; jj<N; jj++)
		{

		// box constraints
		ll = 0;
		for(; ll<2*nb-7; ll+=8)
			{
			v_t    = _mm256_load_ps( &t[jj][ll] );
			v_lam  = _mm256_load_ps( &lam[jj][ll] );
			v_dt   = _mm256_load_ps( &dt[jj][ll] );
			v_dlam = _mm256_load_ps( &dlam[jj][ll] );
			v_dt   = _mm256_mul_ps( v_alpha, v_dt );
			v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
			v_t    = _mm256_add_ps( v_t, v_dt );
			v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*			_mm256_store_ps( &t[jj][ll], v_t );*/
/*			_mm256_store_ps( &lam[jj][ll], v_lam );*/
			v_lam  = _mm256_mul_ps( v_lam, v_t );
			v_mu   = _mm256_add_ps( v_mu, v_lam );
			}
		ll_left = 2*nb - ll;
		if( ll_left>0 )
			{
			ll_left_f = 8.0 - ll_left;
			alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_t    = _mm256_load_ps( &t[jj][ll] );
			v_lam  = _mm256_load_ps( &lam[jj][ll] );
			v_dt   = _mm256_load_ps( &dt[jj][ll] );
			v_dlam = _mm256_load_ps( &dlam[jj][ll] );
			v_dt   = _mm256_mul_ps( v_alpha, v_dt );
			v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
			v_t    = _mm256_add_ps( v_t, v_dt );
			v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*			_mm256_store_ps( &t[jj][ll], v_t );*/
/*			_mm256_store_ps( &lam[jj][ll], v_lam );*/
			v_lam  = _mm256_mul_ps( v_lam, v_t );
			v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
			v_mu   = _mm256_add_ps( v_mu, v_lam );
			}

		}

	// box constraints
	ll = 2*nu;
	for(; ll<2*nb-7; ll+=8)
		{
		v_t    = _mm256_loadu_ps( &t[N][ll] );
		v_lam  = _mm256_loadu_ps( &lam[N][ll] );
		v_dt   = _mm256_loadu_ps( &dt[N][ll] );
		v_dlam = _mm256_loadu_ps( &dlam[N][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*		_mm256_storeu_ps( &t[N][ll], v_t );*/
/*		_mm256_storeu_ps( &lam[N][ll], v_lam );*/
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}
	ll_left = 2*nb - ll;
	if( ll_left>0 )
		{
		ll_left_f = 8.0 - ll_left;
		alpha_mask = _mm256_blendv_ps( v_alpha, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_t    = _mm256_loadu_ps( &t[N][ll] );
		v_lam  = _mm256_loadu_ps( &lam[N][ll] );
		v_dt   = _mm256_loadu_ps( &dt[N][ll] );
		v_dlam = _mm256_loadu_ps( &dlam[N][ll] );
		v_dt   = _mm256_mul_ps( v_alpha, v_dt );
		v_dlam = _mm256_mul_ps( v_alpha, v_dlam );
		v_t    = _mm256_add_ps( v_t, v_dt );
		v_lam  = _mm256_add_ps( v_lam, v_dlam );
/*		_mm256_storeu_ps( &t[N][ll], v_t );*/
/*		_mm256_storeu_ps( &lam[N][ll], v_lam );*/
		v_lam  = _mm256_mul_ps( v_lam, v_t );
		v_lam = _mm256_blendv_ps( v_lam, zeros, _mm256_sub_ps( mask, _mm256_broadcast_ss( &ll_left_f) ) );
		v_mu   = _mm256_add_ps( v_mu, v_lam );
		}

	u_tmp = _mm256_extractf128_ps( v_mu, 0x1 );
	u_mu  = _mm_add_ps( u_tmp, _mm256_castps256_ps128( v_mu ) );

	u_mu  = _mm_hadd_ps( u_mu, u_mu );
	u_mu  = _mm_hadd_ps( u_mu, u_mu );

	u_tmp = _mm_load_ss( &mu_scal );
	u_mu  = _mm_mul_ss( u_mu, u_tmp );
	_mm_store_ss( ptr_mu, u_mu );

	return;

	}



void s_init_ux_t_box_mhe(int N, int nu, int nbu, int nb, float **ux, float **db, float **t, int warm_start)
	{
	
	int jj, ll;
	
	float thr0 = 1e-3; // minimum distance from a constraint

	if(warm_start==1)
		{
		for(jj=0; jj<N; jj++)
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

		}
	else // cold start
		{
		for(jj=0; jj<N; jj++)
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

		}
	
	}



void s_init_lam_mhe(int N, int nu, int nbu, int nb, float **t, float **lam)	// TODO approximate reciprocal
	{
	
	int jj, ll;
	
	for(jj=0; jj<N; jj++)
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
	
	}



void s_update_hessian_box_mhe(int N, int k0, int k1, int kmax, int cnz, float sigma_mu, float **t, float **t_inv, float **lam, float **lamt, float **dlam, float **bd, float **bl, float **pd, float **pl, float **pl2, float **db)

/*void d_update_hessian_box(int k0, int kmax, int nb, int cnz, float sigma_mu, float *t, float *lam, float *lamt, float *dlam, float *bd, float *bl, float *pd, float *pl, float *lb, float *ub)*/
	{
	
	const int bs = 8; //d_get_mr();
	
	float temp0, temp1;
	
	float *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv, *ptr_pd, *ptr_pl, *ptr_pl2, *ptr_bd, *ptr_bl, *ptr_db;
	
	int ii, jj, ll, bs0;
	
	// first & middle stages

	for(jj=0; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_tinv  = t_inv[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];
		ptr_pl2   = pl2[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_db    = db[jj];

		ii = 0;
		for(; ii<kmax-7; ii+=8)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[0+(ii+0)*bs+ii*cnz] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[(ii+0)*bs] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+0] - ptr_dlam[0];
			ptr_pl2[ii+0] = ptr_pl[(ii+0)*bs];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_pd[1+(ii+1)*bs+ii*cnz] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
			ptr_pl[(ii+1)*bs] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[2*ii+3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2*ii+2] - ptr_dlam[2];
			ptr_pl2[ii+1] = ptr_pl[(ii+1)*bs];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_pd[2+(ii+2)*bs+ii*cnz] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
			ptr_pl[(ii+2)*bs] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[2*ii+5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[2*ii+4] - ptr_dlam[4];
			ptr_pl2[ii+2] = ptr_pl[(ii+2)*bs];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_pd[3+(ii+3)*bs+ii*cnz] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
			ptr_pl[(ii+3)*bs] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[2*ii+7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[2*ii+6] - ptr_dlam[6];
			ptr_pl2[ii+3] = ptr_pl[(ii+3)*bs];

			ptr_tinv[8] = 1.0/ptr_t[8];
			ptr_tinv[9] = 1.0/ptr_t[9];
			ptr_lamt[8] = ptr_lam[8]*ptr_tinv[8];
			ptr_lamt[9] = ptr_lam[9]*ptr_tinv[9];
			ptr_dlam[8] = ptr_tinv[8]*sigma_mu; // !!!!!
			ptr_dlam[9] = ptr_tinv[9]*sigma_mu; // !!!!!
			ptr_pd[4+(ii+4)*bs+ii*cnz] = ptr_bd[ii+4] + ptr_lamt[8] + ptr_lamt[9];
			ptr_pl[(ii+4)*bs] = ptr_bl[ii+4] + ptr_lam[9] + ptr_lamt[9]*ptr_db[2*ii+9] + ptr_dlam[9] - ptr_lam[8] - ptr_lamt[8]*ptr_db[2*ii+8] - ptr_dlam[8];
			ptr_pl2[ii+4] = ptr_pl[(ii+4)*bs];

			ptr_tinv[10] = 1.0/ptr_t[10];
			ptr_tinv[11] = 1.0/ptr_t[11];
			ptr_lamt[10] = ptr_lam[10]*ptr_tinv[10];
			ptr_lamt[11] = ptr_lam[11]*ptr_tinv[11];
			ptr_dlam[10] = ptr_tinv[10]*sigma_mu; // !!!!!
			ptr_dlam[11] = ptr_tinv[11]*sigma_mu; // !!!!!
			ptr_pd[5+(ii+5)*bs+ii*cnz] = ptr_bd[ii+5] + ptr_lamt[10] + ptr_lamt[11];
			ptr_pl[(ii+5)*bs] = ptr_bl[ii+5] + ptr_lam[11] + ptr_lamt[11]*ptr_db[2*ii+11] + ptr_dlam[11] - ptr_lam[10] - ptr_lamt[10]*ptr_db[2*ii+10] - ptr_dlam[10];
			ptr_pl2[ii+5] = ptr_pl[(ii+5)*bs];

			ptr_tinv[12] = 1.0/ptr_t[12];
			ptr_tinv[13] = 1.0/ptr_t[13];
			ptr_lamt[12] = ptr_lam[12]*ptr_tinv[12];
			ptr_lamt[13] = ptr_lam[13]*ptr_tinv[13];
			ptr_dlam[12] = ptr_tinv[12]*sigma_mu; // !!!!!
			ptr_dlam[13] = ptr_tinv[13]*sigma_mu; // !!!!!
			ptr_pd[6+(ii+6)*bs+ii*cnz] = ptr_bd[ii+6] + ptr_lamt[12] + ptr_lamt[13];
			ptr_pl[(ii+6)*bs] = ptr_bl[ii+6] + ptr_lam[13] + ptr_lamt[13]*ptr_db[2*ii+13] + ptr_dlam[13] - ptr_lam[12] - ptr_lamt[12]*ptr_db[2*ii+12] - ptr_dlam[12];
			ptr_pl2[ii+6] = ptr_pl[(ii+6)*bs];

			ptr_tinv[14] = 1.0/ptr_t[14];
			ptr_tinv[15] = 1.0/ptr_t[15];
			ptr_lamt[14] = ptr_lam[14]*ptr_tinv[14];
			ptr_lamt[15] = ptr_lam[15]*ptr_tinv[15];
			ptr_dlam[14] = ptr_tinv[14]*sigma_mu; // !!!!!
			ptr_dlam[15] = ptr_tinv[15]*sigma_mu; // !!!!!
			ptr_pd[7+(ii+7)*bs+ii*cnz] = ptr_bd[ii+7] + ptr_lamt[14] + ptr_lamt[15];
			ptr_pl[(ii+7)*bs] = ptr_bl[ii+7] + ptr_lam[15] + ptr_lamt[15]*ptr_db[2*ii+15] + ptr_dlam[15] - ptr_lam[14] - ptr_lamt[14]*ptr_db[2*ii+14] - ptr_dlam[14];
			ptr_pl2[ii+7] = ptr_pl[(ii+7)*bs];

			ptr_t     += 16;
			ptr_lam   += 16;
			ptr_lamt  += 16;
			ptr_dlam  += 16;
			ptr_tinv  += 16;

			}
		if(ii<kmax)
			{
			bs0 = kmax-ii;
			for(ll=0; ll<bs0; ll++)
				{
				ptr_tinv[0] = 1.0/ptr_t[0];
				ptr_tinv[1] = 1.0/ptr_t[1];
				ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
				ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
				ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
				ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
				ptr_pd[ll+(ii+ll)*bs+ii*cnz] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
				ptr_pl[(ii+ll)*bs] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+2*ll+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+2*ll+0] - ptr_dlam[0];
				ptr_pl2[ii+ll+0] = ptr_pl[(ii+ll)*bs];

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_tinv  += 2;
				}
			}
	
		}

	// last stage

	ptr_t     = t[N]     + 2*k1;
	ptr_lam   = lam[N]   + 2*k1;
	ptr_lamt  = lamt[N]  + 2*k1;
	ptr_dlam  = dlam[N]  + 2*k1;
	ptr_tinv  = t_inv[N] + 2*k1;
	ptr_pd    = pd[N];
	ptr_pl    = pl[N];
	ptr_pl2   = pl2[N];
	ptr_bd    = bd[N];
	ptr_bl    = bl[N];
	ptr_db    = db[N];

	ii=k1; // k1 supposed to be multiple of bs !!!!!!!!!!

	for(; ii<kmax-7; ii+=8)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[0+(ii+0)*bs+ii*cnz] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[(ii+0)*bs] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+0] - ptr_dlam[0];
		ptr_pl2[ii+0] = ptr_pl[(ii+0)*bs];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[1+(ii+1)*bs+ii*cnz] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[(ii+1)*bs] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[2*ii+3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2*ii+2] - ptr_dlam[2];
		ptr_pl2[ii+1] = ptr_pl[(ii+1)*bs];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[2+(ii+2)*bs+ii*cnz] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[(ii+2)*bs] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[2*ii+5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[2*ii+4] - ptr_dlam[4];
		ptr_pl2[ii+2] = ptr_pl[(ii+2)*bs];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[3+(ii+3)*bs+ii*cnz] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[(ii+3)*bs] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[2*ii+7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[2*ii+6] - ptr_dlam[6];
		ptr_pl2[ii+3] = ptr_pl[(ii+3)*bs];

		ptr_tinv[8] = 1.0/ptr_t[8];
		ptr_tinv[9] = 1.0/ptr_t[9];
		ptr_lamt[8] = ptr_lam[8]*ptr_tinv[8];
		ptr_lamt[9] = ptr_lam[9]*ptr_tinv[9];
		ptr_dlam[8] = ptr_tinv[8]*sigma_mu; // !!!!!
		ptr_dlam[9] = ptr_tinv[9]*sigma_mu; // !!!!!
		ptr_pd[4+(ii+4)*bs+ii*cnz] = ptr_bd[ii+4] + ptr_lamt[8] + ptr_lamt[9];
		ptr_pl[(ii+4)*bs] = ptr_bl[ii+4] + ptr_lam[9] + ptr_lamt[9]*ptr_db[2*ii+9] + ptr_dlam[9] - ptr_lam[8] - ptr_lamt[8]*ptr_db[2*ii+8] - ptr_dlam[8];
		ptr_pl2[ii+4] = ptr_pl[(ii+4)*bs];

		ptr_tinv[10] = 1.0/ptr_t[10];
		ptr_tinv[11] = 1.0/ptr_t[11];
		ptr_lamt[10] = ptr_lam[10]*ptr_tinv[10];
		ptr_lamt[11] = ptr_lam[11]*ptr_tinv[11];
		ptr_dlam[10] = ptr_tinv[10]*sigma_mu; // !!!!!
		ptr_dlam[11] = ptr_tinv[11]*sigma_mu; // !!!!!
		ptr_pd[5+(ii+5)*bs+ii*cnz] = ptr_bd[ii+5] + ptr_lamt[10] + ptr_lamt[11];
		ptr_pl[(ii+5)*bs] = ptr_bl[ii+5] + ptr_lam[11] + ptr_lamt[11]*ptr_db[2*ii+11] + ptr_dlam[11] - ptr_lam[10] - ptr_lamt[10]*ptr_db[2*ii+10] - ptr_dlam[10];
		ptr_pl2[ii+5] = ptr_pl[(ii+5)*bs];

		ptr_tinv[12] = 1.0/ptr_t[12];
		ptr_tinv[13] = 1.0/ptr_t[13];
		ptr_lamt[12] = ptr_lam[12]*ptr_tinv[12];
		ptr_lamt[13] = ptr_lam[13]*ptr_tinv[13];
		ptr_dlam[12] = ptr_tinv[12]*sigma_mu; // !!!!!
		ptr_dlam[13] = ptr_tinv[13]*sigma_mu; // !!!!!
		ptr_pd[6+(ii+6)*bs+ii*cnz] = ptr_bd[ii+6] + ptr_lamt[12] + ptr_lamt[13];
		ptr_pl[(ii+6)*bs] = ptr_bl[ii+6] + ptr_lam[13] + ptr_lamt[13]*ptr_db[2*ii+13] + ptr_dlam[13] - ptr_lam[12] - ptr_lamt[12]*ptr_db[2*ii+12] - ptr_dlam[12];
		ptr_pl2[ii+6] = ptr_pl[(ii+6)*bs];

		ptr_tinv[14] = 1.0/ptr_t[14];
		ptr_tinv[15] = 1.0/ptr_t[15];
		ptr_lamt[14] = ptr_lam[14]*ptr_tinv[14];
		ptr_lamt[15] = ptr_lam[15]*ptr_tinv[15];
		ptr_dlam[14] = ptr_tinv[14]*sigma_mu; // !!!!!
		ptr_dlam[15] = ptr_tinv[15]*sigma_mu; // !!!!!
		ptr_pd[7+(ii+7)*bs+ii*cnz] = ptr_bd[ii+7] + ptr_lamt[14] + ptr_lamt[15];
		ptr_pl[(ii+7)*bs] = ptr_bl[ii+7] + ptr_lam[15] + ptr_lamt[15]*ptr_db[2*ii+15] + ptr_dlam[15] - ptr_lam[14] - ptr_lamt[14]*ptr_db[2*ii+14] - ptr_dlam[14];
		ptr_pl2[ii+7] = ptr_pl[(ii+7)*bs];

		ptr_t     += 16;
		ptr_lam   += 16;
		ptr_lamt  += 16;
		ptr_dlam  += 16;
		ptr_tinv  += 16;

		}
	if(ii<kmax)
		{
		bs0 = kmax-ii;
		for(ll=0; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ll+(ii+ll)*bs+ii*cnz] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[(ii+ll)*bs] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[2*ii+2*ll+1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[2*ii+2*ll+0] - ptr_dlam[0];
			ptr_pl2[ii+ll+0] = ptr_pl[(ii+ll)*bs];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			}
		}


	}



void s_compute_alpha_box_mhe(int N, int k0, int k1, int kmax, float *ptr_alpha, float **t, float **dt, float **lam, float **dlam, float **lamt, float **dux, float **db)
	{
	
	const int bs = 8; //d_get_mr();
	
	float alpha = ptr_alpha[0];
	
	int kna = ((k1+bs-1)/bs)*bs;

	int jj, ll;


	// first & middle stages
	for(jj=0; jj<N; jj++)
		{

		ll = 0;
		for(; ll<kmax; ll+=2)
			{

			dt[jj][ll+0] =   dux[jj][ll/2] - db[jj][ll+0];
			dt[jj][ll+1] = - dux[jj][ll/2] - db[jj][ll+1];
			dlam[jj][ll+0] -= lamt[jj][ll+0] * dt[jj][ll+0];
			dlam[jj][ll+1] -= lamt[jj][ll+1] * dt[jj][ll+1];
			if( -alpha*dlam[jj][ll+0]>lam[jj][ll+0] )
				{
				alpha = - lam[jj][ll+0] / dlam[jj][ll+0];
				}
			if( -alpha*dlam[jj][ll+1]>lam[jj][ll+1] )
				{
				alpha = - lam[jj][ll+1] / dlam[jj][ll+1];
				}
			dt[jj][ll+0] -= t[jj][ll+0];
			dt[jj][ll+1] -= t[jj][ll+1];
			if( -alpha*dt[jj][ll+0]>t[jj][ll+0] )
				{
				alpha = - t[jj][ll+0] / dt[jj][ll+0];
				}
			if( -alpha*dt[jj][ll+1]>t[jj][ll+1] )
				{
				alpha = - t[jj][ll+1] / dt[jj][ll+1];
				}

			}

		}		

	// last stage
	ll = k1;
	for(; ll<kmax; ll+=2)
		{

		dt[N][ll+0] =   dux[N][ll/2] - db[N][ll+0];
		dt[N][ll+1] = - dux[N][ll/2] - db[N][ll+1];
		dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];
		dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];
		if( -alpha*dlam[N][ll+0]>lam[N][ll+0] )
			{
			alpha = - lam[N][ll+0] / dlam[N][ll+0];
			}
		if( -alpha*dlam[N][ll+1]>lam[N][ll+1] )
			{
			alpha = - lam[N][ll+1] / dlam[N][ll+1];
			}
		dt[N][ll+0] -= t[N][ll+0];
		dt[N][ll+1] -= t[N][ll+1];
		if( -alpha*dt[N][ll+0]>t[N][ll+0] )
			{
			alpha = - t[N][ll+0] / dt[N][ll+0];
			}
		if( -alpha*dt[N][ll+1]>t[N][ll+1] )
			{
			alpha = - t[N][ll+1] / dt[N][ll+1];
			}

		}
	
	ptr_alpha[0] = alpha;

	return;
	
	}



void s_update_var_mhe(int nx, int nu, int N, int nb, int nbu, float *ptr_mu, float mu_scal, float alpha, float **ux, float **dux, float **t, float **dt, float **lam, float **dlam, float **pi, float **dpi)
	{
	
	int jj, ll;
	
	float mu = 0;

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
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}



void s_compute_mu_mhe(int N, int nbu, int nu, int nb, float *ptr_mu, float mu_scal, float alpha, float **lam, float **dlam, float **t, float **dt)
	{
	
	int jj, ll;
	
	float mu = 0;
	
	for(jj=0; jj<N; jj++)
		for(ll=0 ; ll<2*nb; ll+=2)
			mu += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);

	for(ll=2*nu ; ll<2*nb; ll+=2)
		mu += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);

	mu *= mu_scal;
		
	ptr_mu[0] = mu;

	return;

	}

