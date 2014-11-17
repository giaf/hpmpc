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



void d_init_ux_pi_t_box_mpc(int N, int nx, int nu, int nbu, int nb, double **ux, double **pi, double **db, double **t, int warm_start)
	{
	
	int jj, ll, ii;
	
	double thr0 = 1e-3; // minimum distance from a constraint

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
		for(ii=ll/2; ii<nu; ii++)
			ux[0][ii] = 0.0; // initialize remaining components of u to zero
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
			for(ii=ll/2; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
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
		for(ii=ll/2; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] = 0.0; // initialize multipliers to zero

		}
	
	}



void d_init_lam_mpc(int N, int nu, int nbu, int nb, double **t, double **lam)	// TODO approximate reciprocal
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



void d_update_hessian_box_mpc(int N, int k0, int k1, int kmax, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db)
	{

	__m256d
		v_ones, v_sigma_mu,
		v_tmp, v_lam, v_lamt, v_dlam, v_db;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	const int bs = 4; //d_get_mr();
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv;
	
	int ii, jj, ll, bs0;
	
	// first stage
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_t_inv = t_inv[0];
	
	ii = 0;
	for(; ii<k0-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &ptr_t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[0][ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[0][0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[0][1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[0][(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[0][(ii+1)*bs], u_bl );
		_mm_store_pd( &pl2[0][ii+0], u_bl );

		v_tmp  = _mm256_load_pd( &ptr_t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[0][ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[0][2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[0][3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[0][(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[0][(ii+3)*bs], u_bl );
		_mm_store_pd( &pl2[0][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<k0)
		{

/*		bs0 = nb-ii;*/
		bs0 = k0-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[0][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[0][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[0][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[0][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[0][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[0][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[0][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[0][ii+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp  = _mm_load_pd( &ptr_t[0] );
			u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
			_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );
			u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
			u_bd   = _mm_load_sd( &bd[0][ii+ll] );
			u_bd   = _mm_add_sd( u_bd, u_tmp );
			_mm_store_sd( &pd[0][ll+(ii+ll)*bs+ii*cnz], u_bd );
			u_db   = _mm_load_pd( &db[0][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[0][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[0][(ii+ll)*bs], u_bl );
			_mm_store_sd( &pl2[0][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}
	
	// middle stages
	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];

		ii = 0;
		for(; ii<kmax-3; ii+=4)
			{
		
			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+0], u_bl );

			v_tmp  = _mm256_load_pd( &ptr_t[4] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[4], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[4], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii+2] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][2+(ii+2)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][3+(ii+3)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+4] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii+2] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+2)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+3)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+2], u_bl );


			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			}
		if(ii<kmax)
			{

/*			bs0 = nb-ii;*/
			bs0 = kmax-ii;
			ll = 0;
		
			if(bs0>=2)
				{

				v_tmp  = _mm256_load_pd( &ptr_t[0] );
				v_tmp  = _mm256_div_pd( v_ones, v_tmp );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
				v_lam  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt = _mm256_mul_pd( v_tmp, v_lam );
				_mm256_store_pd( &ptr_lamt[0], v_lamt );
				v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam );
				u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
				u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
				u_bd   = _mm_load_pd( &bd[jj][ii] );
				u_bd   = _mm_add_pd( u_bd, u_lamt );
				_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
				_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
				v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
				v_db   = _mm256_mul_pd( v_db, v_lamt );
				v_lam  = _mm256_add_pd( v_lam, v_dlam );
				v_lam  = _mm256_add_pd( v_lam, v_db );
				u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
				u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
				u_bl   = _mm_load_pd( &bl[jj][ii] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
				_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+0], u_bl );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
			
				ll   += 2;
				bs0  -= 2;

				}
		
			if(bs0>0)
				{
			
				u_tmp  = _mm_load_pd( &ptr_t[0] );
				u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
				_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
				u_lam  = _mm_load_pd( &ptr_lam[0] );
				u_lamt = _mm_mul_pd( u_tmp, u_lam );
				_mm_store_pd( &ptr_lamt[0], u_lamt );
				u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam );
				u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
				u_bd   = _mm_load_sd( &bd[jj][ii+ll] );
				u_bd   = _mm_add_sd( u_bd, u_tmp );
				_mm_store_sd( &pd[jj][ll+(ii+ll)*bs+ii*cnz], u_bd );
				u_db   = _mm_load_pd( &db[jj][2*ii+2*ll+0] );
				u_db   = _mm_mul_pd( u_db, u_lamt );
				u_lam  = _mm_add_pd( u_lam, u_dlam );
				u_lam  = _mm_add_pd( u_lam, u_db );
				u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
				u_bl   = _mm_load_sd( &bl[jj][ii+ll] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_store_sd( &pl[jj][(ii+ll)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+ll], u_bl );

	/*			t    += 2;*/
	/*			lam  += 2;*/
	/*			lamt += 2;*/
	/*			dlam += 2;*/

				}
			}
		
		}

	// last stage

	ptr_t     = t[N]     + 2*k1;
	ptr_lam   = lam[N]   + 2*k1;
	ptr_lamt  = lamt[N]  + 2*k1;
	ptr_dlam  = dlam[N]  + 2*k1;
	ptr_t_inv = t_inv[N] + 2*k1;

	ii=k1; // k1 supposed to be multiple of bs !!!!!!!!!!

	for(; ii<kmax-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &ptr_t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+0], u_bl );

		v_tmp  = _mm256_load_pd( &ptr_t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+3)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<kmax)
		{

/*		bs0 = nb-ii;*/
		bs0 = kmax-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[N][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[N][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[N][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp  = _mm_load_pd( &ptr_t[0] );
			u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
			_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );
			u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
			u_bd   = _mm_load_sd( &bd[N][ii+ll] );
			u_bd   = _mm_add_sd( u_bd, u_tmp );
			_mm_store_sd( &pd[N][ll+(ii+ll)*bs+ii*cnz], u_bd );
			u_db   = _mm_load_pd( &db[N][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[N][(ii+ll)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}

	
	return;

	}



void d_compute_alpha_box_mpc(int N, int k0, int k1, int kmax, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db)
	{
	
	__m128
		s_sign, s_ones, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha;
	
	__m256d
		v_sign, v_temp,
		v_dt, v_dux, v_db, v_dlam, v_lamt, v_t, v_alpha, v_lam;
	
	__m128d
		u_sign,	u_temp,
		u_dt, u_dux, u_db, u_dlam, u_lamt, u_t, u_alpha, u_lam;
	
	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );



	const int bs = 4; //d_get_mr();
	
	double alpha = ptr_alpha[0];
	
	int kna = ((k1+bs-1)/bs)*bs;

	int jj, ll;


	// first stage
	ll = 0;
	for(; ll<k0-3; ll+=4) // TODO avx single prec
		{

		v_db    = _mm256_load_pd( &db[0][ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll/2+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll/2+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[0][ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[0][ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[0][ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[0][ll] );
		v_lam   = _mm256_load_pd( &lam[0][ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[0][ll], v_dlam );

		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_dt    = _mm256_cvtpd_ps( v_dt );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_t     = _mm256_cvtpd_ps( v_t );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_t     = _mm_xor_ps( s_t, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp1  = _mm_div_ps( s_t, s_dt );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );

		}

	for(; ll<k0; ll+=2)
		{

		u_db    = _mm_load_pd( &db[0][ll] );
		u_dux   = _mm_loaddup_pd( &dux[0][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[0][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[0][ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[0][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[0][ll] );
		u_lam   = _mm_load_pd( &lam[0][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[0][ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		for(; ll<kmax-3; ll+=4)
			{

			v_db    = _mm256_load_pd( &db[jj][ll] );
			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_t     = _mm256_load_pd( &t[jj][ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][ll], v_dt );

			v_lamt  = _mm256_load_pd( &lamt[jj][ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dt );
			v_dlam  = _mm256_load_pd( &dlam[jj][ll] );
			v_lam   = _mm256_load_pd( &lam[jj][ll] );
			v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
			_mm256_store_pd( &dlam[jj][ll], v_dlam );

			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_dt    = _mm256_cvtpd_ps( v_dt );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
			s_lam   = _mm256_cvtpd_ps( v_lam );
			s_t     = _mm256_cvtpd_ps( v_t );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_t     = _mm_xor_ps( s_t, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp1  = _mm_div_ps( s_t, s_dt );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp1 );
			}

		for(; ll<kmax; ll+=2)
			{

			u_db    = _mm_load_pd( &db[jj][ll] );
			u_dux   = _mm_loaddup_pd( &dux[jj][ll/2+0] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_t     = _mm_load_pd( &t[jj][ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][ll], u_dt );

			u_lamt  = _mm_load_pd( &lamt[jj][ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dt );
			u_dlam  = _mm_load_pd( &dlam[jj][ll] );
			u_lam   = _mm_load_pd( &lam[jj][ll] );
			u_dlam  = _mm_sub_pd( u_dlam, u_lam );
			u_dlam  = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][ll], u_dlam );

			v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			s_lam   = _mm256_cvtpd_ps( v_lam );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

			}

		}		

	// last stage
	ll = k1;
	for(; ll<kna; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_lam   = _mm_load_pd( &lam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		_mm_store_pd( &dlam[N][ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

		}
		
	for(; ll<kmax-3; ll+=4)
		{

		v_db    = _mm256_load_pd( &db[N][ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[N][ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[N][ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[N][ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[N][ll] );
		v_lam   = _mm256_load_pd( &lam[N][ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[N][ll], v_dlam );

		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_dt    = _mm256_cvtpd_ps( v_dt );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_t     = _mm256_cvtpd_ps( v_t );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_t     = _mm_xor_ps( s_t, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp1  = _mm_div_ps( s_t, s_dt );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );

		}

	for(; ll<kmax; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_lam   = _mm_load_pd( &lam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

		}

	v_alpha = _mm256_cvtps_pd( s_alpha );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_update_var_mpc(int nx, int nu, int N, int nb, int nbu, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


	// first stage
	jj = 0;
	
	// update inputs
	ll = 0;
	for(; ll<nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu-1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	if(ll<nu)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}

	// box constraints
	ll = 0;
	for(; ll<2*nbu-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		_mm256_store_pd( &t[jj][ll], v_t );
		_mm256_store_pd( &lam[jj][ll], v_lam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nbu-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<nx+nu-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			}
		if(ll<nx+nu-1)
			{
			u_ux  = _mm_load_pd( &ux[jj][ll] );
			u_dux = _mm_load_pd( &dux[jj][ll] );
			u_dux = _mm_sub_pd( u_dux, u_ux );
			u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_pd( u_ux, u_dux );
			_mm_store_pd( &ux[jj][ll], u_ux );
			ll += 2;
			}
		if(ll<nx+nu)
			{
			u_ux  = _mm_load_sd( &ux[jj][ll] );
			u_dux = _mm_load_sd( &dux[jj][ll] );
			u_dux = _mm_sub_sd( u_dux, u_ux );
			u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_sd( u_ux, u_dux );
			_mm_store_sd( &ux[jj][ll], u_ux );
			}
		
		ll = 0;
		for(; ll<nx-3; ll+=4)
			{
			v_pi  = _mm256_load_pd( &pi[jj][ll] );
			v_dpi = _mm256_load_pd( &dpi[jj][ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
			_mm256_store_pd( &pi[jj][ll], v_pi );
			}
		if(ll<nx-1)
			{
			u_pi  = _mm_load_pd( &pi[jj][ll] );
			u_dpi = _mm_load_pd( &dpi[jj][ll] );
			u_dpi = _mm_sub_pd( u_dpi, u_pi );
			u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_pd( u_pi, u_dpi );
			_mm_store_pd( &pi[jj][ll], u_pi );
			ll += 2;
			}
		if(ll<nx)
			{
			u_pi  = _mm_load_sd( &pi[jj][ll] );
			u_dpi = _mm_load_sd( &dpi[jj][ll] );
			u_dpi = _mm_sub_sd( u_dpi, u_pi );
			u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_sd( u_pi, u_dpi );
			_mm_store_sd( &pi[jj][ll], u_pi );
			}

		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			_mm256_store_pd( &t[jj][ll], v_t );
			_mm256_store_pd( &lam[jj][ll], v_lam );
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][ll], u_t );
			_mm_store_pd( &lam[jj][ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	// update states
	ll = nu;
	if(nu%2==1)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		ll++;
		}
	if((4-nu%4)%4>1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	for(; ll<nx+nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nx+nu-1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	if(ll<nx+nu)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}

	ll = 0;
	for(; ll<nx-3; ll+=4)
		{
		v_pi  = _mm256_load_pd( &pi[jj][ll] );
		v_dpi = _mm256_load_pd( &dpi[jj][ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
		_mm256_store_pd( &pi[jj][ll], v_pi );
		}
	if(ll<nx-1)
		{
		u_pi  = _mm_load_pd( &pi[jj][ll] );
		u_dpi = _mm_load_pd( &dpi[jj][ll] );
		u_dpi = _mm_sub_pd( u_dpi, u_pi );
		u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_pd( u_pi, u_dpi );
		_mm_store_pd( &pi[jj][ll], u_pi );
		ll += 2;
		}
	if(ll<nx)
		{
		u_pi  = _mm_load_sd( &pi[jj][ll] );
		u_dpi = _mm_load_sd( &dpi[jj][ll] );
		u_dpi = _mm_sub_sd( u_dpi, u_pi );
		u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_sd( u_pi, u_dpi );
		_mm_store_sd( &pi[jj][ll], u_pi );
		}

	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		_mm256_store_pd( &t[jj][ll], v_t );
		_mm256_store_pd( &lam[jj][ll], v_lam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;
	
	}


void d_compute_mu_mpc(int N, int nbu, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


	// first stage
	jj = 0;
	
	// box constraints
	ll = 0;
	for(; ll<2*nbu-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*		_mm256_store_pd( &t[jj][ll], v_t );*/
/*		_mm256_store_pd( &lam[jj][ll], v_lam );*/
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nbu-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	// middle stage
	for(jj=1; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*			_mm256_store_pd( &t[jj][ll], v_t );*/
/*			_mm256_store_pd( &lam[jj][ll], v_lam );*/
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
/*			_mm_store_pd( &t[jj][ll], u_t );*/
/*			_mm_store_pd( &lam[jj][ll], u_lam );*/
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*		_mm256_store_pd( &t[jj][ll], v_t );*/
/*		_mm256_store_pd( &lam[jj][ll], v_lam );*/
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;

	}



void d_init_ux_pi_t_box_mhe_old(int N, int nx, int nu, int nbu, int nb, double **ux, double **pi, double **db, double **t, int warm_start)
	{
	
	int jj, ll, ii;
	
	double thr0 = 1e-3; // minimum distance from a constraint

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
			for(ii=ll/2; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
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
		for(ii=ll/2; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] = 0.0; // initialize multipliers to zero

		}
	
	}



void d_init_lam_mhe_old(int N, int nu, int nbu, int nb, double **t, double **lam) // TODO approximate reciprocal
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



void d_update_hessian_box_mhe_old(int N, int k0, int k1, int kmax, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **pl2, double **db)
	{

	__m256d
		v_ones, v_sigma_mu,
		v_tmp, v_lam, v_lamt, v_dlam, v_db;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	const int bs = 4; //d_get_mr();
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv;
	
	int ii, jj, ll, bs0;
	
	// first & middle stages
	for(jj=0; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];

		ii = 0;
		for(; ii<kmax-3; ii+=4)
			{
		
			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+0], u_bl );

			v_tmp  = _mm256_load_pd( &ptr_t[4] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[4], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[4], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[jj][ii+2] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[jj][2+(ii+2)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[jj][3+(ii+3)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+4] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii+2] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[jj][(ii+2)*bs], u_bl );
			_mm_storeh_pd( &pl[jj][(ii+3)*bs], u_bl );
			_mm_store_pd( &pl2[jj][ii+2], u_bl );


			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			}
		if(ii<kmax)
			{

/*			bs0 = nb-ii;*/
			bs0 = kmax-ii;
			ll = 0;
		
			if(bs0>=2)
				{

				v_tmp  = _mm256_load_pd( &ptr_t[0] );
				v_tmp  = _mm256_div_pd( v_ones, v_tmp );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
				v_lam  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt = _mm256_mul_pd( v_tmp, v_lam );
				_mm256_store_pd( &ptr_lamt[0], v_lamt );
				v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam );
				u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
				u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
				u_bd   = _mm_load_pd( &bd[jj][ii] );
				u_bd   = _mm_add_pd( u_bd, u_lamt );
				_mm_storel_pd( &pd[jj][0+(ii+0)*bs+ii*cnz], u_bd );
				_mm_storeh_pd( &pd[jj][1+(ii+1)*bs+ii*cnz], u_bd );
				v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
				v_db   = _mm256_mul_pd( v_db, v_lamt );
				v_lam  = _mm256_add_pd( v_lam, v_dlam );
				v_lam  = _mm256_add_pd( v_lam, v_db );
				u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
				u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
				u_bl   = _mm_load_pd( &bl[jj][ii] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_storel_pd( &pl[jj][(ii+0)*bs], u_bl );
				_mm_storeh_pd( &pl[jj][(ii+1)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+0], u_bl );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
			
				ll   += 2;
				bs0  -= 2;

				}
		
			if(bs0>0)
				{
			
				u_tmp  = _mm_load_pd( &ptr_t[0] );
				u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
				_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
				u_lam  = _mm_load_pd( &ptr_lam[0] );
				u_lamt = _mm_mul_pd( u_tmp, u_lam );
				_mm_store_pd( &ptr_lamt[0], u_lamt );
				u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam );
				u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
				u_bd   = _mm_load_sd( &bd[jj][ii+ll] );
				u_bd   = _mm_add_sd( u_bd, u_tmp );
				_mm_store_sd( &pd[jj][ll+(ii+ll)*bs+ii*cnz], u_bd );
				u_db   = _mm_load_pd( &db[jj][2*ii+2*ll+0] );
				u_db   = _mm_mul_pd( u_db, u_lamt );
				u_lam  = _mm_add_pd( u_lam, u_dlam );
				u_lam  = _mm_add_pd( u_lam, u_db );
				u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
				u_bl   = _mm_load_sd( &bl[jj][ii+ll] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_store_sd( &pl[jj][(ii+ll)*bs], u_bl );
				_mm_store_pd( &pl2[jj][ii+ll], u_bl );

	/*			t    += 2;*/
	/*			lam  += 2;*/
	/*			lamt += 2;*/
	/*			dlam += 2;*/

				}
			}
		
		}

	// last stage

	ptr_t     = t[N]     + 2*k1;
	ptr_lam   = lam[N]   + 2*k1;
	ptr_lamt  = lamt[N]  + 2*k1;
	ptr_dlam  = dlam[N]  + 2*k1;
	ptr_t_inv = t_inv[N] + 2*k1;

	ii=k1; // k1 supposed to be multiple of bs !!!!!!!!!!

	for(; ii<kmax-3; ii+=4)
		{
		
		v_tmp  = _mm256_load_pd( &ptr_t[0] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[0] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[0], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+0], u_bl );

		v_tmp  = _mm256_load_pd( &ptr_t[4] );
		v_tmp  = _mm256_div_pd( v_ones, v_tmp );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp ); // store t_inv
		v_lam  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt = _mm256_mul_pd( v_tmp, v_lam );
		_mm256_store_pd( &ptr_lamt[4], v_lamt );
		v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[4], v_dlam );
		u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
		u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
		u_bd   = _mm_load_pd( &bd[N][ii+2] );
		u_bd   = _mm_add_pd( u_bd, u_lamt );
		_mm_storel_pd( &pd[N][2+(ii+2)*bs+ii*cnz], u_bd );
		_mm_storeh_pd( &pd[N][3+(ii+3)*bs+ii*cnz], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_storel_pd( &pl[N][(ii+2)*bs], u_bl );
		_mm_storeh_pd( &pl[N][(ii+3)*bs], u_bl );
		_mm_store_pd( &pl2[N][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<kmax)
		{

/*		bs0 = nb-ii;*/
		bs0 = kmax-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp  = _mm256_load_pd( &ptr_t[0] );
			v_tmp  = _mm256_div_pd( v_ones, v_tmp );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp ); // store t_inv
			v_lam  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt = _mm256_mul_pd( v_tmp, v_lam );
			_mm256_store_pd( &ptr_lamt[0], v_lamt );
			v_dlam = _mm256_mul_pd( v_tmp, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam );
			u_lamt = _mm256_extractf128_pd( v_lamt, 0x1 );
			u_lamt = _mm_hadd_pd( _mm256_castpd256_pd128( v_lamt ), u_lamt ); // [ lamt[0]+lamt[1] , lamt[2]+lamt[3] ]
			u_bd   = _mm_load_pd( &bd[N][ii] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_storel_pd( &pd[N][0+(ii+0)*bs+ii*cnz], u_bd );
			_mm_storeh_pd( &pd[N][1+(ii+1)*bs+ii*cnz], u_bd );
			v_db   = _mm256_load_pd( &db[N][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[N][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_storel_pd( &pl[N][(ii+0)*bs], u_bl );
			_mm_storeh_pd( &pl[N][(ii+1)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp  = _mm_load_pd( &ptr_t[0] );
			u_tmp  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp );
			_mm_store_pd( &ptr_t_inv[0], u_tmp ); // store t_inv
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );
			u_tmp  = _mm_hadd_pd( u_lamt, u_lamt ); // [ lamt[0]+lamt[1] , xxx ]
			u_bd   = _mm_load_sd( &bd[N][ii+ll] );
			u_bd   = _mm_add_sd( u_bd, u_tmp );
			_mm_store_sd( &pd[N][ll+(ii+ll)*bs+ii*cnz], u_bd );
			u_db   = _mm_load_pd( &db[N][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[N][(ii+ll)*bs], u_bl );
			_mm_store_pd( &pl2[N][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}

	
	return;

	}



void d_compute_alpha_box_mhe_old(int N, int k0, int k1, int kmax, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db)
	{
	
	__m128
		s_sign, s_ones, s_mask0, s_mask1, s_zeros,
		s_lam, s_dlam, s_t, s_dt, s_tmp0, s_tmp1, s_alpha;
	
	__m256d
		v_sign, v_temp,
		v_dt, v_dux, v_db, v_dlam, v_lamt, v_t, v_alpha, v_lam;
	
	__m128d
		u_sign,	u_temp,
		u_dt, u_dux, u_db, u_dlam, u_lamt, u_t, u_alpha, u_lam;
	
	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	s_alpha  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );



	const int bs = 4; //d_get_mr();
	
	double alpha = ptr_alpha[0];
	
	int kna = ((k1+bs-1)/bs)*bs;

	int jj, ll;


	// first & middle stages
	for(jj=0; jj<N; jj++)
		{

		ll = 0;
		for(; ll<kmax-3; ll+=4) // TODO avx single prec
			{

			v_db    = _mm256_load_pd( &db[jj][ll] );
			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll/2+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_t     = _mm256_load_pd( &t[jj][ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][ll], v_dt );

			v_lamt  = _mm256_load_pd( &lamt[jj][ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dt );
			v_dlam  = _mm256_load_pd( &dlam[jj][ll] );
			v_lam   = _mm256_load_pd( &lam[jj][ll] );
			v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
			_mm256_store_pd( &dlam[jj][ll], v_dlam );

			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_dt    = _mm256_cvtpd_ps( v_dt );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
			s_lam   = _mm256_cvtpd_ps( v_lam );
			s_t     = _mm256_cvtpd_ps( v_t );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_t     = _mm_xor_ps( s_t, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp1  = _mm_div_ps( s_t, s_dt );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp1 );
			}

		for(; ll<kmax; ll+=2)
			{

			u_db    = _mm_load_pd( &db[jj][ll] );
			u_dux   = _mm_loaddup_pd( &dux[jj][ll/2+0] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_t     = _mm_load_pd( &t[jj][ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][ll], u_dt );

			u_lamt  = _mm_load_pd( &lamt[jj][ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dt );
			u_dlam  = _mm_load_pd( &dlam[jj][ll] );
			u_lam   = _mm_load_pd( &lam[jj][ll] );
			u_dlam  = _mm_sub_pd( u_dlam, u_lam );
			u_dlam  = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][ll], u_dlam );

			v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			s_dlam  = _mm256_cvtpd_ps( v_dlam );
			s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
			v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			s_lam   = _mm256_cvtpd_ps( v_lam );
			s_lam   = _mm_xor_ps( s_lam, s_sign );
			s_tmp0  = _mm_div_ps( s_lam, s_dlam );
			s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

			}

		}		

	// last stage
	ll = k1;
	for(; ll<kna; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_lam   = _mm_load_pd( &lam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

		}
		
	for(; ll<kmax-3; ll+=4)
		{

		v_db    = _mm256_load_pd( &db[N][ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll/2+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[N][ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[N][ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[N][ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[N][ll] );
		v_lam   = _mm256_load_pd( &lam[N][ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[N][ll], v_dlam );

		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_dt    = _mm256_cvtpd_ps( v_dt );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		s_mask1 = _mm_cmplt_ps( s_dt, s_zeros );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_t     = _mm256_cvtpd_ps( v_t );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_t     = _mm_xor_ps( s_t, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp1  = _mm_div_ps( s_t, s_dt );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_tmp1  = _mm_blendv_ps( s_ones, s_tmp1, s_mask1 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp1 );

		}

	for(; ll<kmax; ll+=2)
		{

		u_db    = _mm_load_pd( &db[N][ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll/2+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][ll] );
		u_lam   = _mm_load_pd( &lam[N][ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][ll], u_dlam );

		v_dlam  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam  = _mm256_cvtpd_ps( v_dlam );
		s_mask0 = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam   = _mm256_cvtpd_ps( v_lam );
		s_lam   = _mm_xor_ps( s_lam, s_sign );
		s_tmp0  = _mm_div_ps( s_lam, s_dlam );
		s_tmp0  = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha = _mm_min_ps( s_alpha, s_tmp0 );

		}

	v_alpha = _mm256_cvtps_pd( s_alpha );
	u_alpha = _mm256_extractf128_pd( v_alpha, 0x1 );
	u_alpha = _mm_min_pd( u_alpha, _mm256_castpd256_pd128( v_alpha ) );
	u_alpha = _mm_min_sd( u_alpha, _mm_permute_pd( u_alpha, 0x1 ) );
/*	u_alpha = _mm_min_sd( u_alpha, _mm_load_sd( &alpha ) );*/
	_mm_store_sd( &alpha, u_alpha );

	
	ptr_alpha[0] = alpha;

	return;
	
	}


void d_update_var_mhe_old(int nx, int nu, int N, int nb, int nbu, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


		
	// first & middle stages
	for(jj=0; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<nx+nu-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			}
		if(ll<nx+nu-1)
			{
			u_ux  = _mm_load_pd( &ux[jj][ll] );
			u_dux = _mm_load_pd( &dux[jj][ll] );
			u_dux = _mm_sub_pd( u_dux, u_ux );
			u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_pd( u_ux, u_dux );
			_mm_store_pd( &ux[jj][ll], u_ux );
			ll += 2;
			}
		if(ll<nx+nu)
			{
			u_ux  = _mm_load_sd( &ux[jj][ll] );
			u_dux = _mm_load_sd( &dux[jj][ll] );
			u_dux = _mm_sub_sd( u_dux, u_ux );
			u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
			u_ux  = _mm_add_sd( u_ux, u_dux );
			_mm_store_sd( &ux[jj][ll], u_ux );
			}
		
		ll = 0;
		for(; ll<nx-3; ll+=4)
			{
			v_pi  = _mm256_load_pd( &pi[jj][ll] );
			v_dpi = _mm256_load_pd( &dpi[jj][ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
			_mm256_store_pd( &pi[jj][ll], v_pi );
			}
		if(ll<nx-1)
			{
			u_pi  = _mm_load_pd( &pi[jj][ll] );
			u_dpi = _mm_load_pd( &dpi[jj][ll] );
			u_dpi = _mm_sub_pd( u_dpi, u_pi );
			u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_pd( u_pi, u_dpi );
			_mm_store_pd( &pi[jj][ll], u_pi );
			ll += 2;
			}
		if(ll<nx)
			{
			u_pi  = _mm_load_sd( &pi[jj][ll] );
			u_dpi = _mm_load_sd( &dpi[jj][ll] );
			u_dpi = _mm_sub_sd( u_dpi, u_pi );
			u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
			u_pi  = _mm_add_sd( u_pi, u_dpi );
			_mm_store_sd( &pi[jj][ll], u_pi );
			}

		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			_mm256_store_pd( &t[jj][ll], v_t );
			_mm256_store_pd( &lam[jj][ll], v_lam );
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][ll], u_t );
			_mm_store_pd( &lam[jj][ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	// update states
	ll = nu;
	if(nu%2==1)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		ll++;
		}
	if((4-nu%4)%4>1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	for(; ll<nx+nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nx+nu-1)
		{
		u_ux  = _mm_load_pd( &ux[jj][ll] );
		u_dux = _mm_load_pd( &dux[jj][ll] );
		u_dux = _mm_sub_pd( u_dux, u_ux );
		u_dux = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_pd( u_ux, u_dux );
		_mm_store_pd( &ux[jj][ll], u_ux );
		ll += 2;
		}
	if(ll<nx+nu)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}

	ll = 0;
	for(; ll<nx-3; ll+=4)
		{
		v_pi  = _mm256_load_pd( &pi[jj][ll] );
		v_dpi = _mm256_load_pd( &dpi[jj][ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
		_mm256_store_pd( &pi[jj][ll], v_pi );
		}
	if(ll<nx-1)
		{
		u_pi  = _mm_load_pd( &pi[jj][ll] );
		u_dpi = _mm_load_pd( &dpi[jj][ll] );
		u_dpi = _mm_sub_pd( u_dpi, u_pi );
		u_dpi = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_pd( u_pi, u_dpi );
		_mm_store_pd( &pi[jj][ll], u_pi );
		ll += 2;
		}
	if(ll<nx)
		{
		u_pi  = _mm_load_sd( &pi[jj][ll] );
		u_dpi = _mm_load_sd( &dpi[jj][ll] );
		u_dpi = _mm_sub_sd( u_dpi, u_pi );
		u_dpi = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dpi );
		u_pi  = _mm_add_sd( u_pi, u_dpi );
		_mm_store_sd( &pi[jj][ll], u_pi );
		}

	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		_mm256_store_pd( &t[jj][ll], v_t );
		_mm256_store_pd( &lam[jj][ll], v_lam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][ll], u_t );
		_mm_store_pd( &lam[jj][ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;
	
	}


void d_compute_mu_mhe_old(int N, int nbu, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	int jj, ll;
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_t, v_dt, v_lam, v_dlam, v_mu;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();


	// first & middle stages
	for(jj=0; jj<N; jj++)
		{
		
		ll = 0;
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][ll] );
			v_lam  = _mm256_load_pd( &lam[jj][ll] );
			v_dt   = _mm256_load_pd( &dt[jj][ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*			_mm256_store_pd( &t[jj][ll], v_t );*/
/*			_mm256_store_pd( &lam[jj][ll], v_lam );*/
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][ll] );
			u_lam  = _mm_load_pd( &lam[jj][ll] );
			u_dt   = _mm_load_pd( &dt[jj][ll] );
			u_dlam = _mm_load_pd( &dlam[jj][ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
/*			_mm_store_pd( &t[jj][ll], u_t );*/
/*			_mm_store_pd( &lam[jj][ll], u_lam );*/
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}

	// last stage
	jj = N;
	
	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][ll] );
		v_lam  = _mm256_load_pd( &lam[jj][ll] );
		v_dt   = _mm256_load_pd( &dt[jj][ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
/*		_mm256_store_pd( &t[jj][ll], v_t );*/
/*		_mm256_store_pd( &lam[jj][ll], v_lam );*/
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][ll] );
		u_lam  = _mm_load_pd( &lam[jj][ll] );
		u_dt   = _mm_load_pd( &dt[jj][ll] );
		u_dlam = _mm_load_pd( &dlam[jj][ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
/*		_mm_store_pd( &t[jj][ll], u_t );*/
/*		_mm_store_pd( &lam[jj][ll], u_lam );*/
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
		
	u_tmp = _mm256_extractf128_pd( v_mu, 0x1 );
	u_mu  = _mm_add_pd( u_mu, _mm256_castpd256_pd128( v_mu ) );
	u_mu  = _mm_add_pd( u_mu, u_tmp );
	u_mu  = _mm_hadd_pd( u_mu, u_mu );
	u_tmp = _mm_load_sd( &mu_scal );
	u_mu  = _mm_mul_sd( u_mu, u_tmp );
	_mm_store_sd( ptr_mu, u_mu );
		

	return;

	}

