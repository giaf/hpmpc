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

#include "../../include/block_size.h"



void d_init_var_box_mpc(int N, int nx, int nu, int nb, double **ux, double **pi, double **db, double **t, double **lam, double mu0, int warm_start)
	{

	const int nbu = nu<nb ? nu : nb ;
	
	int jj, ll, ii;
	
	double thr0 = 0.1; // minimum distance from a constraint

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
			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0; // this has to be strictly positive !!!
			}
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
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			}
		for(ll=0; ll<2*nbu; ll++) // this has to be strictly positive !!!
			{
			t[N][ll] = 1.0;
			lam[N][ll] = 1.0;
			}
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
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
			}
		}
	else // cold start
		{
		for(ll=0; ll<2*nbu; ll+=2)
			{
			ux[0][ll/2] = 0.0;
			//ux[0][ll/2] = 0.5*( - db[0][ll+1] + db[0][ll+0] );
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
			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(ii=ll/2; ii<nu; ii++)
			ux[0][ii] = 0.0; // initialize remaining components of u to zero
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0; // this has to be strictly positive !!!
			}
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
				ux[jj][ll/2] = 0.0;
				//ux[jj][ll/2] = 0.5*( - db[jj][ll+1] + db[jj][ll+0] );
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
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			for(ii=ll/2; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
			}
		for(ll=0; ll<2*nbu; ll++)
			{
			t[N][ll] = 1.0; // this has to be strictly positive !!!
			lam[N][ll] = 1.0; // this has to be strictly positive !!!
			}
		for(ll=2*nu; ll<2*nb; ll+=2)
			{
			ux[N][ll/2] = 0.0;
			//ux[N][ll/2] = 0.5*( - db[N][ll+1] + db[N][ll+0] );
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
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
			}
		for(ii=ll/2; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] = 0.0; // initialize multipliers to zero

		}
	
	}



void d_init_var_soft_mpc(int N, int nx, int nu, int nb, double **ux, double **pi, double **db, double **t, double **lam, double mu0, int warm_start)
	{
	
	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll, ii;
	
	double thr0 = 0.1; // minimum distance from a constraint

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

			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0;
			}
		for(jj=1; jj<N; jj++)
			{
			for(ll=0; ll<2*nb; ll+=2)
				{
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
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			}
		for(ll=0; ll<2*nbu; ll++) // this has to be strictly positive !!!
			{
			t[N][ll] = 1.0;
			lam[N][ll] = 1.0;
			}
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
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
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
			lam[0][ll+0] = mu0/t[0][ll+0];
			lam[0][ll+1] = mu0/t[0][ll+1];
			}
		for(ii=ll/2; ii<nu; ii++)
			ux[0][ii] = 0.0; // initialize remaining components of u to zero
		for(; ll<2*nb; ll++)
			{
			t[0][ll] = 1.0; // this has to be strictly positive !!!
			lam[0][ll] = 1.0;
			}
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
				lam[jj][ll+0] = mu0/t[jj][ll+0];
				lam[jj][ll+1] = mu0/t[jj][ll+1];
				}
			for(ii=ll/2; ii<nx+nu; ii++)
				ux[jj][ii] = 0.0; // initialize remaining components of u and x to zero
			}
		for(ll=0; ll<2*nbu; ll++)
			{
			t[N][ll] = 1.0; // this has to be strictly positive !!!
			lam[N][ll] = 1.0;
			}
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
			lam[N][ll+0] = mu0/t[N][ll+0];
			lam[N][ll+1] = mu0/t[N][ll+1];
			}
		for(ii=ll/2; ii<nx+nu; ii++)
			ux[N][ii] = 0.0; // initialize remaining components of x to zero

		// inizialize t_theta (cold start only for the moment)
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<2*nbx; ll++)
				t[jj][anb+2*nu+ll] = 1.0;

		// initialize lam_theta (cold start only for the moment)
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<2*nbx; ll++)
				lam[jj][anb+2*nu+ll] = mu0/t[jj][anb+2*nu+ll];

		// initialize pi
		for(jj=0; jj<=N; jj++)
			for(ll=0; ll<nx; ll++)
				pi[jj][ll] = 0.0; // initialize multipliers to zero

		}
	
	}



void d_update_hessian_box_mpc(int N, int nx, int nu, int nb, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **db)
	{

	const int nbu = nu<nb ? nu : nb ;

	const int bs = 4; //d_get_mr();
	
	__m256d
		v_ones, v_sigma_mu,
		v_tmp, v_lam, v_lamt, v_dlam, v_db;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
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
	for(; ii<nbu-3; ii+=4)
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
		_mm_store_pd( &pd[0][ii+0], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_store_pd( &pl[0][ii+0], u_bl );

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
		_mm_store_pd( &pd[0][ii+2], u_bd );
		v_db   = _mm256_load_pd( &db[0][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[0][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_store_pd( &pl[0][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<nbu)
		{

/*		bs0 = nb-ii;*/
		bs0 = nbu-ii;
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
			_mm_store_pd( &pd[0][ii+0], u_bd );
			v_db   = _mm256_load_pd( &db[0][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[0][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_pd( &pl[0][ii+0], u_bl );

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
			_mm_store_sd( &pd[0][ii+ll], u_bd );
			u_db   = _mm_load_pd( &db[0][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[0][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[0][ii+ll], u_bl );

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
		for(; ii<nb-3; ii+=4)
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
			_mm_store_pd( &pd[jj][ii+0], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_pd( &pl[jj][ii+0], u_bl );

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
			_mm_store_pd( &pd[jj][ii+2], u_bd );
			v_db   = _mm256_load_pd( &db[jj][2*ii+4] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[jj][ii+2] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_pd( &pl[jj][ii+2], u_bl );


			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			}
		if(ii<nb)
			{

			bs0 = nb-ii;
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
				_mm_store_pd( &pd[jj][ii+0], u_bd );
				v_db   = _mm256_load_pd( &db[jj][2*ii+0] );
				v_db   = _mm256_mul_pd( v_db, v_lamt );
				v_lam  = _mm256_add_pd( v_lam, v_dlam );
				v_lam  = _mm256_add_pd( v_lam, v_db );
				u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
				u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
				u_bl   = _mm_load_pd( &bl[jj][ii] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_store_pd( &pl[jj][ii+0], u_bl );

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
				_mm_store_sd( &pd[jj][ii+ll], u_bd );
				u_db   = _mm_load_pd( &db[jj][2*ii+2*ll+0] );
				u_db   = _mm_mul_pd( u_db, u_lamt );
				u_lam  = _mm_add_pd( u_lam, u_dlam );
				u_lam  = _mm_add_pd( u_lam, u_db );
				u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
				u_bl   = _mm_load_sd( &bl[jj][ii+ll] );
				u_bl   = _mm_sub_pd( u_bl, u_lam );
				_mm_store_sd( &pl[jj][ii+ll], u_bl );

	/*			t    += 2;*/
	/*			lam  += 2;*/
	/*			lamt += 2;*/
	/*			dlam += 2;*/

				}
			}
		
		}

	// last stage

	//ptr_t     = t[N]     + 2*k1;
	//ptr_lam   = lam[N]   + 2*k1;
	//ptr_lamt  = lamt[N]  + 2*k1;
	//ptr_dlam  = dlam[N]  + 2*k1;
	//ptr_t_inv = t_inv[N] + 2*k1;

	//ii=k1; // k1 supposed to be multiple of bs !!!!!!!!!!

	ptr_t     = t[N]     + 2*nu;
	ptr_lam   = lam[N]   + 2*nu;
	ptr_lamt  = lamt[N]  + 2*nu;
	ptr_dlam  = dlam[N]  + 2*nu;
	ptr_t_inv = t_inv[N] + 2*nu;

	ii = (nu/4)*4;
	ll = nu%4;
	if(ll>0)
		{
		bs0 = 4 - ll;
		if(bs0%2==1)
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
			_mm_store_sd( &pd[N][ii+ll], u_bd );
			u_db   = _mm_load_pd( &db[N][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[N][ii+ll], u_bl );

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_t_inv += 2;

			ll++;
			bs0--;
			}
		if(bs0==2)
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
			u_bd   = _mm_load_pd( &bd[N][ii+ll] );
			u_bd   = _mm_add_pd( u_bd, u_lamt );
			_mm_store_pd( &pd[N][ii+ll+0], u_bd );
			v_db   = _mm256_load_pd( &db[N][2*ii+2*ll+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_pd( &pl[N][ii+ll+0], u_bl );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;
			
			ll   += 2;
			bs0  -= 2;


			bs0-=2;
			}
		ii += 4;
		}

	for(; ii<nb-3; ii+=4)
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
		_mm_store_pd( &pd[N][ii+0], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+0] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_store_pd( &pl[N][ii+0], u_bl );

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
		_mm_store_pd( &pd[N][ii+2], u_bd );
		v_db   = _mm256_load_pd( &db[N][2*ii+4] );
		v_db   = _mm256_mul_pd( v_db, v_lamt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_add_pd( v_lam, v_db );
		u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
		u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
		u_bl   = _mm_load_pd( &bl[N][ii+2] );
		u_bl   = _mm_sub_pd( u_bl, u_lam );
		_mm_store_pd( &pl[N][ii+2], u_bl );


		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<nb)
		{

/*		bs0 = nb-ii;*/
		bs0 = nb-ii;
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
			_mm_store_pd( &pd[N][ii+0], u_bd );
			v_db   = _mm256_load_pd( &db[N][2*ii+0] );
			v_db   = _mm256_mul_pd( v_db, v_lamt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_add_pd( v_lam, v_db );
			u_lam  = _mm256_extractf128_pd( v_lam, 0x1 );
			u_lam  = _mm_hsub_pd( _mm256_castpd256_pd128( v_lam ), u_lam ); // [ lam[1]-lam[0] , lam[3]-lam[2] ] + [ dlam[1]-dlam[0] , dlam[3]-dlam[2] ]
			u_bl   = _mm_load_pd( &bl[N][ii] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_pd( &pl[N][ii+0], u_bl );

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
			_mm_store_sd( &pd[N][ii+ll], u_bd );
			u_db   = _mm_load_pd( &db[N][2*ii+2*ll+0] );
			u_db   = _mm_mul_pd( u_db, u_lamt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_add_pd( u_lam, u_db );
			u_lam  = _mm_hsub_pd( u_lam, u_lam ); // [ lam[1]-lam[0] , xxx ] + [ dlam[1]-dlam[0] , xxx ]
			u_bl   = _mm_load_sd( &bl[N][ii+ll] );
			u_bl   = _mm_sub_pd( u_bl, u_lam );
			_mm_store_sd( &pl[N][ii+ll], u_bl );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/

			}
		}

	
	return;

	}



void d_update_hessian_soft_mpc(int N, int nx, int nu, int nb, int cnz, double sigma_mu, double **t, double **t_inv, double **lam, double **lamt, double **dlam, double **bd, double **bl, double **pd, double **pl, double **db, double **Z, double **z, double **Zl, double **zl)

/*void d_update_hessian_box(int k0, int kmax, int nb, int cnz, double sigma_mu, double *t, double *lam, double *lamt, double *dlam, double *bd, double *bl, double *pd, double *pl, double *lb, double *ub)*/
	{

	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	//const int k0 = nbu;
	//const int k1 = (nu/bs)*bs;
	//const int kmax = nb;
	
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv, *ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Z, *ptr_z, *ptr_Zl, *ptr_zl;

	static double Qx[8] = {};
	static double qx[8] = {};
	
	int ii, jj, ll, bs0;
	
	// first stage
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_tinv  = t_inv[0];
	ptr_pd    = pd[0];
	ptr_pl    = pl[0];
	ptr_bd    = bd[0];
	ptr_bl    = bl[0];
	ptr_db    = db[0];
	//ptr_Z     = Z[0];
	//ptr_z     = z[0];
	//ptr_Zl    = Zl[0];
	//ptr_zl    = zl[0];
	
	ii = 0;
	// hard constraints on u only
	for(; ii<nbu-3; ii+=4)
		{

		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
		ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
		ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2] - ptr_dlam[2];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
		ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[4] - ptr_dlam[4];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
		ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[6] - ptr_dlam[6];

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_tinv  += 8;
		ptr_db    += 8;

		}
	if(ii<nbu)
		{
		bs0 = nbu-ii;
		for(ll=0; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ii+ll] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[ii+ll] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			ptr_db    += 2;
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
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_db    = db[jj];
		ptr_Z     = Z[jj];
		ptr_z     = z[jj];
		ptr_Zl    = Zl[jj];
		ptr_zl    = zl[jj];

		ii = 0;
		// hard constraints on u
		for(; ii<nbu-3; ii+=4)
			{

			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_lamt[0] + ptr_lamt[1];
			ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_lamt[2] + ptr_lamt[3];
			ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_lam[3] + ptr_lamt[3]*ptr_db[3] + ptr_dlam[3] - ptr_lam[2] - ptr_lamt[2]*ptr_db[2] - ptr_dlam[2];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_lamt[4] + ptr_lamt[5];
			ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_lam[5] + ptr_lamt[5]*ptr_db[5] + ptr_dlam[5] - ptr_lam[4] - ptr_lamt[4]*ptr_db[4] - ptr_dlam[4];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_lamt[6] + ptr_lamt[7];
			ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_lam[7] + ptr_lamt[7]*ptr_db[7] + ptr_dlam[7] - ptr_lam[6] - ptr_lamt[6]*ptr_db[6] - ptr_dlam[6];

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_tinv  += 8;
			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nbu)
			{
			// clean-up loop
			bs0 = nbu-ii;
			ll = 0;
			for(; ll<bs0; ll++)
				{
				ptr_tinv[0] = 1.0/ptr_t[0];
				ptr_tinv[1] = 1.0/ptr_t[1];
				ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
				ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
				ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
				ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
				ptr_pd[ii+ll] = ptr_bd[ii+ll] + ptr_lamt[0] + ptr_lamt[1];
				ptr_pl[ii+ll] = ptr_bl[ii+ll] + ptr_lam[1] + ptr_lamt[1]*ptr_db[1] + ptr_dlam[1] - ptr_lam[0] - ptr_lamt[0]*ptr_db[0] - ptr_dlam[0];

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_tinv  += 2;
				ptr_db    += 2;
				ptr_Z     += 2;
				ptr_z     += 2;
				ptr_Zl    += 2;
				ptr_zl    += 2;
				}
			
			// soft constraints on x
			// clean-up loop
			bs0 = nb-ii<4 ? nb-ii : 4 ;
			for(; ll<bs0; ll++)
				{
				ptr_tinv[0] = 1.0/ptr_t[0];
				ptr_tinv[1] = 1.0/ptr_t[1];
				ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
				ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
				ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
				ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
				ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
				ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
				ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
				ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
				ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
				ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
				Qx[0] = ptr_lamt[0];
				Qx[1] = ptr_lamt[1];
				qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
				qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
				ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
				ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
				ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
				ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
				qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
				qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
				Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
				Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
				ptr_pd[ii+ll] = ptr_bd[ii+ll] + Qx[1] + Qx[0];
				ptr_pl[ii+ll] = ptr_bl[ii+ll] + qx[1] - qx[0];

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_tinv  += 2;
				ptr_db    += 2;
				ptr_Z     += 2;
				ptr_z     += 2;
				ptr_Zl    += 2;
				ptr_zl    += 2;
				}
			ii += ll;
			}
		// main loop
		for(; ii<nb-3; ii+=4)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
			ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
			ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
			ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
			Qx[0] = ptr_lamt[0];
			Qx[1] = ptr_lamt[1];
			qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
			ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
			ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
			ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
			qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
			Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
			ptr_pd[ii+0] = ptr_bd[ii+0] + Qx[1] + Qx[0];
			ptr_pl[ii+0] = ptr_bl[ii+0] + qx[1] - qx[0];

			ptr_tinv[2] = 1.0/ptr_t[2];
			ptr_tinv[3] = 1.0/ptr_t[3];
			ptr_tinv[anb+2] = 1.0/ptr_t[anb+2];
			ptr_tinv[anb+3] = 1.0/ptr_t[anb+3];
			ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
			ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
			ptr_lamt[anb+2] = ptr_lam[anb+2]*ptr_tinv[anb+2];
			ptr_lamt[anb+3] = ptr_lam[anb+3]*ptr_tinv[anb+3];
			ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
			ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
			ptr_dlam[anb+2] = ptr_tinv[anb+2]*sigma_mu; // !!!!!
			ptr_dlam[anb+3] = ptr_tinv[anb+3]*sigma_mu; // !!!!!
			Qx[2] = ptr_lamt[2];
			Qx[3] = ptr_lamt[3];
			qx[2] = ptr_lam[2] + ptr_dlam[2] + ptr_lamt[2]*ptr_db[2];
			qx[3] = ptr_lam[3] + ptr_dlam[3] + ptr_lamt[3]*ptr_db[3];
			ptr_Zl[2] = 1.0 / (ptr_Z[2] + Qx[2] + ptr_lamt[anb+2]); // inverted of updated diagonal !!!
			ptr_Zl[3] = 1.0 / (ptr_Z[3] + Qx[3] + ptr_lamt[anb+3]); // inverted of updated diagonal !!!
			ptr_zl[2] = - ptr_z[2] + qx[2] + ptr_lam[anb+2] + ptr_dlam[anb+2];
			ptr_zl[3] = - ptr_z[3] + qx[3] + ptr_lam[anb+3] + ptr_dlam[anb+3];
			qx[2] = qx[2] - Qx[2]*ptr_zl[2]*ptr_Zl[2]; // update this before Qx !!!!!!!!!!!
			qx[3] = qx[3] - Qx[3]*ptr_zl[3]*ptr_Zl[3]; // update this before Qx !!!!!!!!!!!
			Qx[2] = Qx[2] - Qx[2]*Qx[2]*ptr_Zl[2];
			Qx[3] = Qx[3] - Qx[3]*Qx[3]*ptr_Zl[3];
			ptr_pd[ii+1] = ptr_bd[ii+1] + Qx[3] + Qx[2];
			ptr_pl[ii+1] = ptr_bl[ii+1] + qx[3] - qx[2];

			ptr_tinv[4] = 1.0/ptr_t[4];
			ptr_tinv[5] = 1.0/ptr_t[5];
			ptr_tinv[anb+4] = 1.0/ptr_t[anb+4];
			ptr_tinv[anb+5] = 1.0/ptr_t[anb+5];
			ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
			ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
			ptr_lamt[anb+4] = ptr_lam[anb+4]*ptr_tinv[anb+4];
			ptr_lamt[anb+5] = ptr_lam[anb+5]*ptr_tinv[anb+5];
			ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
			ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
			ptr_dlam[anb+4] = ptr_tinv[anb+4]*sigma_mu; // !!!!!
			ptr_dlam[anb+5] = ptr_tinv[anb+5]*sigma_mu; // !!!!!
			Qx[4] = ptr_lamt[4];
			Qx[5] = ptr_lamt[5];
			qx[4] = ptr_lam[4] + ptr_dlam[4] + ptr_lamt[4]*ptr_db[4];
			qx[5] = ptr_lam[5] + ptr_dlam[5] + ptr_lamt[5]*ptr_db[5];
			ptr_Zl[4] = 1.0 / (ptr_Z[4] + Qx[4] + ptr_lamt[anb+4]); // inverted of updated diagonal !!!
			ptr_Zl[5] = 1.0 / (ptr_Z[5] + Qx[5] + ptr_lamt[anb+5]); // inverted of updated diagonal !!!
			ptr_zl[4] = - ptr_z[4] + qx[4] + ptr_lam[anb+4] + ptr_dlam[anb+4];
			ptr_zl[5] = - ptr_z[5] + qx[5] + ptr_lam[anb+5] + ptr_dlam[anb+5];
			qx[4] = qx[4] - Qx[4]*ptr_zl[4]*ptr_Zl[4]; // update this before Qx !!!!!!!!!!!
			qx[5] = qx[5] - Qx[5]*ptr_zl[5]*ptr_Zl[5]; // update this before Qx !!!!!!!!!!!
			Qx[4] = Qx[4] - Qx[4]*Qx[4]*ptr_Zl[4];
			Qx[5] = Qx[5] - Qx[5]*Qx[5]*ptr_Zl[5];
			ptr_pd[ii+2] = ptr_bd[ii+2] + Qx[5] + Qx[4];
			ptr_pl[ii+2] = ptr_bl[ii+2] + qx[5] - qx[4];

			ptr_tinv[6] = 1.0/ptr_t[6];
			ptr_tinv[7] = 1.0/ptr_t[7];
			ptr_tinv[anb+6] = 1.0/ptr_t[anb+6];
			ptr_tinv[anb+7] = 1.0/ptr_t[anb+7];
			ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
			ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
			ptr_lamt[anb+6] = ptr_lam[anb+6]*ptr_tinv[anb+6];
			ptr_lamt[anb+7] = ptr_lam[anb+7]*ptr_tinv[anb+7];
			ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
			ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
			ptr_dlam[anb+6] = ptr_tinv[anb+6]*sigma_mu; // !!!!!
			ptr_dlam[anb+7] = ptr_tinv[anb+7]*sigma_mu; // !!!!!
			Qx[6] = ptr_lamt[6];
			Qx[7] = ptr_lamt[7];
			qx[6] = ptr_lam[6] + ptr_dlam[6] + ptr_lamt[6]*ptr_db[6];
			qx[7] = ptr_lam[7] + ptr_dlam[7] + ptr_lamt[7]*ptr_db[7];
			ptr_Zl[6] = 1.0 / (ptr_Z[6] + Qx[6] + ptr_lamt[anb+6]); // inverted of updated diagonal !!!
			ptr_Zl[7] = 1.0 / (ptr_Z[7] + Qx[7] + ptr_lamt[anb+7]); // inverted of updated diagonal !!!
			ptr_zl[6] = - ptr_z[6] + qx[6] + ptr_lam[anb+6] + ptr_dlam[anb+6];
			ptr_zl[7] = - ptr_z[7] + qx[7] + ptr_lam[anb+7] + ptr_dlam[anb+7];
			qx[6] = qx[6] - Qx[6]*ptr_zl[6]*ptr_Zl[6]; // update this before Qx !!!!!!!!!!!
			qx[7] = qx[7] - Qx[7]*ptr_zl[7]*ptr_Zl[7]; // update this before Qx !!!!!!!!!!!
			Qx[6] = Qx[6] - Qx[6]*Qx[6]*ptr_Zl[6];
			Qx[7] = Qx[7] - Qx[7]*Qx[7]*ptr_Zl[7];
			ptr_pd[ii+3] = ptr_bd[ii+3] + Qx[7] + Qx[6];
			ptr_pl[ii+3] = ptr_bl[ii+3] + qx[7] - qx[6];

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_tinv  += 8;
			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nb)
			{
			bs0 = nb-ii;
			for(ll=0; ll<bs0; ll++)
				{
				ptr_tinv[0] = 1.0/ptr_t[0];
				ptr_tinv[1] = 1.0/ptr_t[1];
				ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
				ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
				ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
				ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
				ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
				ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
				ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
				ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
				ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
				ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
				Qx[0] = ptr_lamt[0];
				Qx[1] = ptr_lamt[1];
				qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
				qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
				ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
				ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
				ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
				ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
				qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
				qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
				Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
				Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
				ptr_pd[ii+ll] = ptr_bd[ii+ll] + Qx[1] + Qx[0];
				ptr_pl[ii+ll] = ptr_bl[ii+ll] + qx[1] - qx[0];

				ptr_t     += 2;
				ptr_lam   += 2;
				ptr_lamt  += 2;
				ptr_dlam  += 2;
				ptr_tinv  += 2;
				ptr_db    += 2;
				ptr_Z     += 2;
				ptr_z     += 2;
				ptr_Zl    += 2;
				ptr_zl    += 2;
				}
			}
	
		}

	// last stage

	ptr_t     = t[N]     + 2*nu;
	ptr_lam   = lam[N]   + 2*nu;
	ptr_lamt  = lamt[N]  + 2*nu;
	ptr_dlam  = dlam[N]  + 2*nu;
	ptr_tinv  = t_inv[N] + 2*nu;
	ptr_db    = db[N]    + 2*nu;
	ptr_Z     = Z[N]     + 2*nu;
	ptr_z     = z[N]     + 2*nu;
	ptr_Zl    = Zl[N]    + 2*nu;
	ptr_zl    = zl[N]    + 2*nu;
	ptr_pd    = pd[N];
	ptr_pl    = pl[N];
	ptr_bd    = bd[N];
	ptr_bl    = bl[N];

	ii=4*(nu/4); // k1 supposed to be multiple of bs !!!!!!!!!! NO MORE !!!!!!!
	if(ii<nu)
		{
		bs0 = nb-ii<4 ? nb-ii : 4 ;
		ll = nu-ii; //k0%4;
		for(; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
			ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
			ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
			ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
			Qx[0] = ptr_lamt[0];
			Qx[1] = ptr_lamt[1];
			qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
			ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
			ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
			ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
			qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
			Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
			ptr_pd[ii+ll] = ptr_bd[ii+ll] + Qx[1] + Qx[0];
			ptr_pl[ii+ll] = ptr_bl[ii+ll] + qx[1] - qx[0];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			ptr_db    += 2;
			ptr_Z     += 2;
			ptr_z     += 2;
			ptr_Zl    += 2;
			ptr_zl    += 2;
			}
		ii += 4;
		}

	for(; ii<nb-3; ii+=4)
		{
		ptr_tinv[0] = 1.0/ptr_t[0];
		ptr_tinv[1] = 1.0/ptr_t[1];
		ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
		ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
		ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
		ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
		ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
		ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
		ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
		ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
		ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
		ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
		Qx[0] = ptr_lamt[0];
		Qx[1] = ptr_lamt[1];
		qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
		qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
		ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
		ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
		ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
		ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
		qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
		qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
		Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
		Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
		ptr_pd[ii+0] = ptr_bd[ii+0] + Qx[1] + Qx[0];
		ptr_pl[ii+0] = ptr_bl[ii+0] + qx[1] - qx[0];

		ptr_tinv[2] = 1.0/ptr_t[2];
		ptr_tinv[3] = 1.0/ptr_t[3];
		ptr_tinv[anb+2] = 1.0/ptr_t[anb+2];
		ptr_tinv[anb+3] = 1.0/ptr_t[anb+3];
		ptr_lamt[2] = ptr_lam[2]*ptr_tinv[2];
		ptr_lamt[3] = ptr_lam[3]*ptr_tinv[3];
		ptr_lamt[anb+2] = ptr_lam[anb+2]*ptr_tinv[anb+2];
		ptr_lamt[anb+3] = ptr_lam[anb+3]*ptr_tinv[anb+3];
		ptr_dlam[2] = ptr_tinv[2]*sigma_mu; // !!!!!
		ptr_dlam[3] = ptr_tinv[3]*sigma_mu; // !!!!!
		ptr_dlam[anb+2] = ptr_tinv[anb+2]*sigma_mu; // !!!!!
		ptr_dlam[anb+3] = ptr_tinv[anb+3]*sigma_mu; // !!!!!
		Qx[2] = ptr_lamt[2];
		Qx[3] = ptr_lamt[3];
		qx[2] = ptr_lam[2] + ptr_dlam[2] + ptr_lamt[2]*ptr_db[2];
		qx[3] = ptr_lam[3] + ptr_dlam[3] + ptr_lamt[3]*ptr_db[3];
		ptr_Zl[2] = 1.0 / (ptr_Z[2] + Qx[2] + ptr_lamt[anb+2]); // inverted of updated diagonal !!!
		ptr_Zl[3] = 1.0 / (ptr_Z[3] + Qx[3] + ptr_lamt[anb+3]); // inverted of updated diagonal !!!
		ptr_zl[2] = - ptr_z[2] + qx[2] + ptr_lam[anb+2] + ptr_dlam[anb+2];
		ptr_zl[3] = - ptr_z[3] + qx[3] + ptr_lam[anb+3] + ptr_dlam[anb+3];
		qx[2] = qx[2] - Qx[2]*ptr_zl[2]*ptr_Zl[2]; // update this before Qx !!!!!!!!!!!
		qx[3] = qx[3] - Qx[3]*ptr_zl[3]*ptr_Zl[3]; // update this before Qx !!!!!!!!!!!
		Qx[2] = Qx[2] - Qx[2]*Qx[2]*ptr_Zl[2];
		Qx[3] = Qx[3] - Qx[3]*Qx[3]*ptr_Zl[3];
		ptr_pd[ii+1] = ptr_bd[ii+1] + Qx[3] + Qx[2];
		ptr_pl[ii+1] = ptr_bl[ii+1] + qx[3] - qx[2];

		ptr_tinv[4] = 1.0/ptr_t[4];
		ptr_tinv[5] = 1.0/ptr_t[5];
		ptr_tinv[anb+4] = 1.0/ptr_t[anb+4];
		ptr_tinv[anb+5] = 1.0/ptr_t[anb+5];
		ptr_lamt[4] = ptr_lam[4]*ptr_tinv[4];
		ptr_lamt[5] = ptr_lam[5]*ptr_tinv[5];
		ptr_lamt[anb+4] = ptr_lam[anb+4]*ptr_tinv[anb+4];
		ptr_lamt[anb+5] = ptr_lam[anb+5]*ptr_tinv[anb+5];
		ptr_dlam[4] = ptr_tinv[4]*sigma_mu; // !!!!!
		ptr_dlam[5] = ptr_tinv[5]*sigma_mu; // !!!!!
		ptr_dlam[anb+4] = ptr_tinv[anb+4]*sigma_mu; // !!!!!
		ptr_dlam[anb+5] = ptr_tinv[anb+5]*sigma_mu; // !!!!!
		Qx[4] = ptr_lamt[4];
		Qx[5] = ptr_lamt[5];
		qx[4] = ptr_lam[4] + ptr_dlam[4] + ptr_lamt[4]*ptr_db[4];
		qx[5] = ptr_lam[5] + ptr_dlam[5] + ptr_lamt[5]*ptr_db[5];
		ptr_Zl[4] = 1.0 / (ptr_Z[4] + Qx[4] + ptr_lamt[anb+4]); // inverted of updated diagonal !!!
		ptr_Zl[5] = 1.0 / (ptr_Z[5] + Qx[5] + ptr_lamt[anb+5]); // inverted of updated diagonal !!!
		ptr_zl[4] = - ptr_z[4] + qx[4] + ptr_lam[anb+4] + ptr_dlam[anb+4];
		ptr_zl[5] = - ptr_z[5] + qx[5] + ptr_lam[anb+5] + ptr_dlam[anb+5];
		qx[4] = qx[4] - Qx[4]*ptr_zl[4]*ptr_Zl[4]; // update this before Qx !!!!!!!!!!!
		qx[5] = qx[5] - Qx[5]*ptr_zl[5]*ptr_Zl[5]; // update this before Qx !!!!!!!!!!!
		Qx[4] = Qx[4] - Qx[4]*Qx[4]*ptr_Zl[4];
		Qx[5] = Qx[5] - Qx[5]*Qx[5]*ptr_Zl[5];
		ptr_pd[ii+2] = ptr_bd[ii+2] + Qx[5] + Qx[4];
		ptr_pl[ii+2] = ptr_bl[ii+2] + qx[5] - qx[4];

		ptr_tinv[6] = 1.0/ptr_t[6];
		ptr_tinv[7] = 1.0/ptr_t[7];
		ptr_tinv[anb+6] = 1.0/ptr_t[anb+6];
		ptr_tinv[anb+7] = 1.0/ptr_t[anb+7];
		ptr_lamt[6] = ptr_lam[6]*ptr_tinv[6];
		ptr_lamt[7] = ptr_lam[7]*ptr_tinv[7];
		ptr_lamt[anb+6] = ptr_lam[anb+6]*ptr_tinv[anb+6];
		ptr_lamt[anb+7] = ptr_lam[anb+7]*ptr_tinv[anb+7];
		ptr_dlam[6] = ptr_tinv[6]*sigma_mu; // !!!!!
		ptr_dlam[7] = ptr_tinv[7]*sigma_mu; // !!!!!
		ptr_dlam[anb+6] = ptr_tinv[anb+6]*sigma_mu; // !!!!!
		ptr_dlam[anb+7] = ptr_tinv[anb+7]*sigma_mu; // !!!!!
		Qx[6] = ptr_lamt[6];
		Qx[7] = ptr_lamt[7];
		qx[6] = ptr_lam[6] + ptr_dlam[6] + ptr_lamt[6]*ptr_db[6];
		qx[7] = ptr_lam[7] + ptr_dlam[7] + ptr_lamt[7]*ptr_db[7];
		ptr_Zl[6] = 1.0 / (ptr_Z[6] + Qx[6] + ptr_lamt[anb+6]); // inverted of updated diagonal !!!
		ptr_Zl[7] = 1.0 / (ptr_Z[7] + Qx[7] + ptr_lamt[anb+7]); // inverted of updated diagonal !!!
		ptr_zl[6] = - ptr_z[6] + qx[6] + ptr_lam[anb+6] + ptr_dlam[anb+6];
		ptr_zl[7] = - ptr_z[7] + qx[7] + ptr_lam[anb+7] + ptr_dlam[anb+7];
		qx[6] = qx[6] - Qx[6]*ptr_zl[6]*ptr_Zl[6]; // update this before Qx !!!!!!!!!!!
		qx[7] = qx[7] - Qx[7]*ptr_zl[7]*ptr_Zl[7]; // update this before Qx !!!!!!!!!!!
		Qx[6] = Qx[6] - Qx[6]*Qx[6]*ptr_Zl[6];
		Qx[7] = Qx[7] - Qx[7]*Qx[7]*ptr_Zl[7];
		ptr_pd[ii+3] = ptr_bd[ii+3] + Qx[7] + Qx[6];
		ptr_pl[ii+3] = ptr_bl[ii+3] + qx[7] - qx[6];

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_tinv  += 8;
		ptr_db    += 8;
		ptr_Z     += 8;
		ptr_z     += 8;
		ptr_Zl    += 8;
		ptr_zl    += 8;

		}
	if(ii<nb)
		{
		bs0 = nb-ii;
		for(ll=0; ll<bs0; ll++)
			{
			ptr_tinv[0] = 1.0/ptr_t[0];
			ptr_tinv[1] = 1.0/ptr_t[1];
			ptr_tinv[anb+0] = 1.0/ptr_t[anb+0];
			ptr_tinv[anb+1] = 1.0/ptr_t[anb+1];
			ptr_lamt[0] = ptr_lam[0]*ptr_tinv[0];
			ptr_lamt[1] = ptr_lam[1]*ptr_tinv[1];
			ptr_lamt[anb+0] = ptr_lam[anb+0]*ptr_tinv[anb+0];
			ptr_lamt[anb+1] = ptr_lam[anb+1]*ptr_tinv[anb+1];
			ptr_dlam[0] = ptr_tinv[0]*sigma_mu; // !!!!!
			ptr_dlam[1] = ptr_tinv[1]*sigma_mu; // !!!!!
			ptr_dlam[anb+0] = ptr_tinv[anb+0]*sigma_mu; // !!!!!
			ptr_dlam[anb+1] = ptr_tinv[anb+1]*sigma_mu; // !!!!!
			Qx[0] = ptr_lamt[0];
			Qx[1] = ptr_lamt[1];
			qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			ptr_Zl[0] = 1.0 / (ptr_Z[0] + Qx[0] + ptr_lamt[anb+0]); // inverted of updated diagonal !!!
			ptr_Zl[1] = 1.0 / (ptr_Z[1] + Qx[1] + ptr_lamt[anb+1]); // inverted of updated diagonal !!!
			ptr_zl[0] = - ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
			ptr_zl[1] = - ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
			qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			Qx[0] = Qx[0] - Qx[0]*Qx[0]*ptr_Zl[0];
			Qx[1] = Qx[1] - Qx[1]*Qx[1]*ptr_Zl[1];
			ptr_pd[ii+ll] = ptr_bd[ii+ll] + Qx[1] + Qx[0];
			ptr_pl[ii+ll] = ptr_bl[ii+ll] + qx[1] - qx[0];

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_tinv  += 2;
			ptr_db    += 2;
			ptr_Z     += 2;
			ptr_z     += 2;
			ptr_Zl    += 2;
			ptr_zl    += 2;
			}
		}


	}



void d_update_jacobian_box_mpc(int N, int nx, int nu, int nb, double sigma_mu, double **dt, double **dlam, double **t_inv, double **pl2)
	{

	const int nbu = nu<nb ? nu : nb ;

	int ii, jj;

	// first stage
	for(ii=0; ii<2*nbu; ii+=2)
		{
		dlam[0][ii+0] = t_inv[0][ii+0]*(sigma_mu - dlam[0][ii+0]*dt[0][ii+0]); // !!!!!
		dlam[0][ii+1] = t_inv[0][ii+1]*(sigma_mu - dlam[0][ii+1]*dt[0][ii+1]); // !!!!!
		pl2[0][ii/2] += dlam[0][ii+1] - dlam[0][ii+0];
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		for(ii=0; ii<2*nb; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
			}
		}

	// last stages
	for(ii=2*nu; ii<2*nb; ii+=2)
		{
		dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
		dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
		pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
		}

	}



void d_update_jacobian_soft_mpc(int N, int nx, int nu, int nb, double sigma_mu, double **dt, double **dlam, double **t_inv, double **lamt, double **pl2, double **Zl, double **zl)
	{

	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int ii, jj;
	
	static double Qx[2] = {};
	static double qx[2] = {};


	// first stage
	for(ii=0; ii<2*nbu; ii+=2)
		{
		dlam[0][ii+0] = t_inv[0][ii+0]*(sigma_mu - dlam[0][ii+0]*dt[0][ii+0]); // !!!!!
		dlam[0][ii+1] = t_inv[0][ii+1]*(sigma_mu - dlam[0][ii+1]*dt[0][ii+1]); // !!!!!
		pl2[0][ii/2] += dlam[0][ii+1] - dlam[0][ii+0];
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		ii=0;
		for(; ii<2*nbu; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			pl2[jj][ii/2] += dlam[jj][ii+1] - dlam[jj][ii+0];
			}
		for(; ii<2*nb; ii+=2)
			{
			dlam[jj][ii+0] = t_inv[jj][ii+0]*(sigma_mu - dlam[jj][ii+0]*dt[jj][ii+0]); // !!!!!
			dlam[jj][ii+1] = t_inv[jj][ii+1]*(sigma_mu - dlam[jj][ii+1]*dt[jj][ii+1]); // !!!!!
			dlam[jj][anb+ii+0] = t_inv[jj][anb+ii+0]*(sigma_mu - dlam[jj][anb+ii+0]*dt[jj][anb+ii+0]); // !!!!!
			dlam[jj][anb+ii+1] = t_inv[jj][anb+ii+1]*(sigma_mu - dlam[jj][anb+ii+1]*dt[jj][anb+ii+1]); // !!!!!
			Qx[0] = lamt[jj][ii+0];
			Qx[1] = lamt[jj][ii+1];
			//qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
			//qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
			qx[0] = dlam[jj][ii+0];
			qx[1] = dlam[jj][ii+1];
			//ptr_zl[0] = ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
			//ptr_zl[1] = ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
			zl[jj][ii+0] += qx[0] + dlam[jj][anb+ii+0];
			zl[jj][ii+1] += qx[1] + dlam[jj][anb+ii+1];
			//qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
			//qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
			qx[0] = qx[0] - Qx[0]*(qx[0] + dlam[jj][anb+ii+0])*Zl[jj][ii+0]; // update this before Qx !!!!!!!!!!!
			qx[1] = qx[1] - Qx[1]*(qx[1] + dlam[jj][anb+ii+1])*Zl[jj][ii+1]; // update this before Qx !!!!!!!!!!!
			pl2[jj][ii/2] += qx[1] - qx[0];
			}
		}

	// last stages
	for(ii=2*nu; ii<2*nb; ii+=2)
		{
		dlam[N][ii+0] = t_inv[N][ii+0]*(sigma_mu - dlam[N][ii+0]*dt[N][ii+0]); // !!!!!
		dlam[N][ii+1] = t_inv[N][ii+1]*(sigma_mu - dlam[N][ii+1]*dt[N][ii+1]); // !!!!!
		dlam[N][anb+ii+0] = t_inv[N][anb+ii+0]*(sigma_mu - dlam[N][anb+ii+0]*dt[N][anb+ii+0]); // !!!!!
		dlam[N][anb+ii+1] = t_inv[N][anb+ii+1]*(sigma_mu - dlam[N][anb+ii+1]*dt[N][anb+ii+1]); // !!!!!
		Qx[0] = lamt[N][ii+0];
		Qx[1] = lamt[N][ii+1];
		//qx[0] = ptr_lam[0] + ptr_dlam[0] + ptr_lamt[0]*ptr_db[0];
		//qx[1] = ptr_lam[1] + ptr_dlam[1] + ptr_lamt[1]*ptr_db[1];
		qx[0] = dlam[N][ii+0];
		qx[1] = dlam[N][ii+1];
		//ptr_zl[0] = ptr_z[0] + qx[0] + ptr_lam[anb+0] + ptr_dlam[anb+0];
		//ptr_zl[1] = ptr_z[1] + qx[1] + ptr_lam[anb+1] + ptr_dlam[anb+1];
		zl[N][ii+0] += qx[0] + dlam[N][anb+ii+0];
		zl[N][ii+1] += qx[1] + dlam[N][anb+ii+1];
		//qx[0] = qx[0] - Qx[0]*ptr_zl[0]*ptr_Zl[0]; // update this before Qx !!!!!!!!!!!
		//qx[1] = qx[1] - Qx[1]*ptr_zl[1]*ptr_Zl[1]; // update this before Qx !!!!!!!!!!!
		qx[0] = qx[0] - Qx[0]*(qx[0] + dlam[N][anb+ii+0])*Zl[N][ii+0]; // update this before Qx !!!!!!!!!!!
		qx[1] = qx[1] - Qx[1]*(qx[1] + dlam[N][anb+ii+1])*Zl[N][ii+1]; // update this before Qx !!!!!!!!!!!
		pl2[N][ii/2] += qx[1] - qx[0];
		}

	}



void d_compute_alpha_box_mpc(int N, int nx, int nu, int nb, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db)
	{
	
	const int nbu = nu<nb ? nu : nb ;

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
	
	int jj, ll;


	// first stage
	ll = 0;
	for(; ll<nbu-1; ll+=2) // TODO avx single prec
		{

		v_db    = _mm256_load_pd( &db[0][2*ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[0][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[0][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[0][2*ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[0][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[0][2*ll] );
		v_lam   = _mm256_load_pd( &lam[0][2*ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[0][2*ll], v_dlam );

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

	for(; ll<nbu; ll++)
		{

		u_db    = _mm_load_pd( &db[0][2*ll] );
		u_dux   = _mm_loaddup_pd( &dux[0][ll+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[0][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[0][2*ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[0][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[0][2*ll] );
		u_lam   = _mm_load_pd( &lam[0][2*ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[0][2*ll], u_dlam );

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
		for(; ll<nb-1; ll+=2)
			{

			v_db    = _mm256_load_pd( &db[jj][2*ll] );
			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_t     = _mm256_load_pd( &t[jj][2*ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][2*ll], v_dt );

			v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dt );
			v_dlam  = _mm256_load_pd( &dlam[jj][2*ll] );
			v_lam   = _mm256_load_pd( &lam[jj][2*ll] );
			v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
			_mm256_store_pd( &dlam[jj][2*ll], v_dlam );

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

		for(; ll<nb; ll++)
			{

			u_db    = _mm_load_pd( &db[jj][2*ll] );
			u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_t     = _mm_load_pd( &t[jj][2*ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );

			u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dt );
			u_dlam  = _mm_load_pd( &dlam[jj][2*ll] );
			u_lam   = _mm_load_pd( &lam[jj][2*ll] );
			u_dlam  = _mm_sub_pd( u_dlam, u_lam );
			u_dlam  = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );

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
	ll = nu;
	for(; ll<((nu+bs-1)/bs)*bs; ll++)
		{

		u_db    = _mm_load_pd( &db[N][2*ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][2*ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][2*ll] );
		u_lam   = _mm_load_pd( &lam[N][2*ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		_mm_store_pd( &dlam[N][2*ll], u_dlam );

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
		
	for(; ll<nb-1; ll+=2)
		{

		v_db    = _mm256_load_pd( &db[N][2*ll] );
		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[N][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_t     = _mm256_load_pd( &t[N][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[N][2*ll], v_dt );

		v_lamt  = _mm256_load_pd( &lamt[N][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dt );
		v_dlam  = _mm256_load_pd( &dlam[N][2*ll] );
		v_lam   = _mm256_load_pd( &lam[N][2*ll] );
		v_dlam  = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam  = _mm256_sub_pd( v_dlam, v_temp );
		_mm256_store_pd( &dlam[N][2*ll], v_dlam );

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

	for(; ll<nb; ll++)
		{

		u_db    = _mm_load_pd( &db[N][2*ll] );
		u_dux   = _mm_loaddup_pd( &dux[N][ll+0] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_t     = _mm_load_pd( &t[N][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[N][2*ll], u_dt );

		u_lamt  = _mm_load_pd( &lamt[N][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dt );
		u_dlam  = _mm_load_pd( &dlam[N][2*ll] );
		u_lam   = _mm_load_pd( &lam[N][2*ll] );
		u_dlam  = _mm_sub_pd( u_dlam, u_lam );
		u_dlam  = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[N][2*ll], u_dlam );

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


void d_compute_alpha_soft_mpc(int N, int nx, int nu, int nb, double *ptr_alpha, double **t, double **dt, double **lam, double **dlam, double **lamt, double **dux, double **db, double **Zl, double **zl)
	{
	
/*	const int bs = 4; //d_get_mr();*/

	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	
	double alpha = ptr_alpha[0];
	
/*	int kna = ((k1+bs-1)/bs)*bs;*/

	int jj, ll;


	// first stage

	ll = 0;
	// hard input constraints
	for(; ll<nbu; ll++)
		{

		dt[0][2*ll+0] =   dux[0][ll] - db[0][2*ll+0] - t[0][2*ll+0];
		dt[0][2*ll+1] = - dux[0][ll] - db[0][2*ll+1] - t[0][2*ll+1];
		dlam[0][2*ll+0] -= lamt[0][2*ll+0] * dt[0][2*ll+0] + lam[0][2*ll+0];
		dlam[0][2*ll+1] -= lamt[0][2*ll+1] * dt[0][2*ll+1] + lam[0][2*ll+1];
		if( -alpha*dlam[0][2*ll+0]>lam[0][2*ll+0] )
			{
			alpha = - lam[0][2*ll+0] / dlam[0][2*ll+0];
			}
		if( -alpha*dlam[0][2*ll+1]>lam[0][2*ll+1] )
			{
			alpha = - lam[0][2*ll+1] / dlam[0][2*ll+1];
			}
		if( -alpha*dt[0][2*ll+0]>t[0][2*ll+0] )
			{
			alpha = - t[0][2*ll+0] / dt[0][2*ll+0];
			}
		if( -alpha*dt[0][2*ll+1]>t[0][2*ll+1] )
			{
			alpha = - t[0][2*ll+1] / dt[0][2*ll+1];
			}

		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		// hard input constraints
		for(; ll<nbu; ll++)
			{

			dt[jj][2*ll+0] =   dux[jj][ll] - db[jj][2*ll+0] - t[jj][2*ll+0];
			dt[jj][2*ll+1] = - dux[jj][ll] - db[jj][2*ll+1] - t[jj][2*ll+1];
			dlam[jj][2*ll+0] -= lamt[jj][2*ll+0] * dt[jj][2*ll+0] + lam[jj][2*ll+0];
			dlam[jj][2*ll+1] -= lamt[jj][2*ll+1] * dt[jj][2*ll+1] + lam[jj][2*ll+1];
			if( -alpha*dlam[jj][2*ll+0]>lam[jj][2*ll+0] )
				{
				alpha = - lam[jj][2*ll+0] / dlam[jj][2*ll+0];
				}
			if( -alpha*dlam[jj][2*ll+1]>lam[jj][2*ll+1] )
				{
				alpha = - lam[jj][2*ll+1] / dlam[jj][2*ll+1];
				}
			if( -alpha*dt[jj][2*ll+0]>t[jj][2*ll+0] )
				{
				alpha = - t[jj][2*ll+0] / dt[jj][2*ll+0];
				}
			if( -alpha*dt[jj][2*ll+1]>t[jj][2*ll+1] )
				{
				alpha = - t[jj][2*ll+1] / dt[jj][2*ll+1];
				}

			}
		// soft state constraints
		for(; ll<nb; ll++)
			{

			dt[jj][anb+2*ll+0] = ( zl[jj][2*ll+0] - lamt[jj][2*ll+0]*dux[jj][ll] ) * Zl[jj][2*ll+0];
			dt[jj][anb+2*ll+1] = ( zl[jj][2*ll+1] + lamt[jj][2*ll+1]*dux[jj][ll] ) * Zl[jj][2*ll+1];
			dt[jj][2*ll+0] = dt[jj][anb+2*ll+0] + dux[jj][ll] - db[jj][2*ll+0] - t[jj][2*ll+0];
			dt[jj][2*ll+1] = dt[jj][anb+2*ll+1] - dux[jj][ll] - db[jj][2*ll+1] - t[jj][2*ll+1];
			dt[jj][anb+2*ll+0] -= t[jj][anb+2*ll+0];
			dt[jj][anb+2*ll+1] -= t[jj][anb+2*ll+1];
			dlam[jj][2*ll+0] -= lamt[jj][2*ll+0] * dt[jj][2*ll+0] + lam[jj][2*ll+0];
			dlam[jj][2*ll+1] -= lamt[jj][2*ll+1] * dt[jj][2*ll+1] + lam[jj][2*ll+1];
			dlam[jj][anb+2*ll+0] -= lamt[jj][anb+2*ll+0] * dt[jj][anb+2*ll+0] + lam[jj][anb+2*ll+0];
			dlam[jj][anb+2*ll+1] -= lamt[jj][anb+2*ll+1] * dt[jj][anb+2*ll+1] + lam[jj][anb+2*ll+1];
			if( -alpha*dlam[jj][2*ll+0]>lam[jj][2*ll+0] )
				{
				alpha = - lam[jj][2*ll+0] / dlam[jj][2*ll+0];
				}
			if( -alpha*dlam[jj][2*ll+1]>lam[jj][2*ll+1] )
				{
				alpha = - lam[jj][2*ll+1] / dlam[jj][2*ll+1];
				}
			if( -alpha*dt[jj][2*ll+0]>t[jj][2*ll+0] )
				{
				alpha = - t[jj][2*ll+0] / dt[jj][2*ll+0];
				}
			if( -alpha*dt[jj][2*ll+1]>t[jj][2*ll+1] )
				{
				alpha = - t[jj][2*ll+1] / dt[jj][2*ll+1];
				}
			if( -alpha*dlam[jj][anb+2*ll+0]>lam[jj][anb+2*ll+0] )
				{
				alpha = - lam[jj][anb+2*ll+0] / dlam[jj][anb+2*ll+0];
				}
			if( -alpha*dlam[jj][anb+2*ll+1]>lam[jj][anb+2*ll+1] )
				{
				alpha = - lam[jj][anb+2*ll+1] / dlam[jj][anb+2*ll+1];
				}
			if( -alpha*dt[jj][anb+2*ll+0]>t[jj][anb+2*ll+0] )
				{
				alpha = - t[jj][anb+2*ll+0] / dt[jj][anb+2*ll+0];
				}
			if( -alpha*dt[jj][anb+2*ll+1]>t[jj][anb+2*ll+1] )
				{
				alpha = - t[jj][anb+2*ll+1] / dt[jj][anb+2*ll+1];
				}

			}

		}		

	// last stage
	ll = nu;
	for(; ll<nb; ll++)
		{

		dt[N][anb+2*ll+0] = ( zl[N][2*ll+0] - lamt[N][2*ll+0]*dux[N][ll] ) * Zl[N][2*ll+0];
		dt[N][anb+2*ll+1] = ( zl[N][2*ll+1] + lamt[N][2*ll+1]*dux[N][ll] ) * Zl[N][2*ll+1];
		dt[N][2*ll+0] = dt[N][anb+2*ll+0] + dux[N][ll] - db[N][2*ll+0] - t[N][2*ll+0];
		dt[N][2*ll+1] = dt[N][anb+2*ll+1] - dux[N][ll] - db[N][2*ll+1] - t[N][2*ll+1];
		dt[N][anb+2*ll+0] -= t[N][anb+2*ll+0];
		dt[N][anb+2*ll+1] -= t[N][anb+2*ll+1];
		dlam[N][2*ll+0] -= lamt[N][2*ll+0] * dt[N][2*ll+0] + lam[N][2*ll+0];
		dlam[N][2*ll+1] -= lamt[N][2*ll+1] * dt[N][2*ll+1] + lam[N][2*ll+1];
		dlam[N][anb+2*ll+0] -= lamt[N][anb+2*ll+0] * dt[N][anb+2*ll+0] + lam[N][anb+2*ll+0];
		dlam[N][anb+2*ll+1] -= lamt[N][anb+2*ll+1] * dt[N][anb+2*ll+1] + lam[N][anb+2*ll+1];
		if( -alpha*dlam[N][2*ll+0]>lam[N][2*ll+0] )
			{
			alpha = - lam[N][2*ll+0] / dlam[N][2*ll+0];
			}
		if( -alpha*dlam[N][2*ll+1]>lam[N][2*ll+1] )
			{
			alpha = - lam[N][2*ll+1] / dlam[N][2*ll+1];
			}
		if( -alpha*dt[N][2*ll+0]>t[N][2*ll+0] )
			{
			alpha = - t[N][2*ll+0] / dt[N][2*ll+0];
			}
		if( -alpha*dt[N][2*ll+1]>t[N][2*ll+1] )
			{
			alpha = - t[N][2*ll+1] / dt[N][2*ll+1];
			}
		if( -alpha*dlam[N][anb+2*ll+0]>lam[N][anb+2*ll+0] )
			{
			alpha = - lam[N][anb+2*ll+0] / dlam[N][anb+2*ll+0];
			}
		if( -alpha*dlam[N][anb+2*ll+1]>lam[N][anb+2*ll+1] )
			{
			alpha = - lam[N][anb+2*ll+1] / dlam[N][anb+2*ll+1];
			}
		if( -alpha*dt[N][anb+2*ll+0]>t[N][anb+2*ll+0] )
			{
			alpha = - t[N][anb+2*ll+0] / dt[N][anb+2*ll+0];
			}
		if( -alpha*dt[N][anb+2*ll+1]>t[N][anb+2*ll+1] )
			{
			alpha = - t[N][anb+2*ll+1] / dt[N][anb+2*ll+1];
			}


		}
	
	ptr_alpha[0] = alpha;

	return;
	
	}



void d_update_var_box_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{
	
	const int nbu = nu<nb ? nu : nb ;

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


void d_update_var_soft_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **ux, double **dux, double **t, double **dt, double **lam, double **dlam, double **pi, double **dpi)
	{

	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll;
	
	double mu = 0;

	// update inputs
	for(ll=0; ll<nu; ll++)
		ux[0][ll] += alpha*(dux[0][ll] - ux[0][ll]);
	// box constraints on inputs
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
		// box constraints on inputs
		ll = 0;
		for(; ll<2*nbu; ll+=2)
			{
			lam[jj][ll+0] += alpha*dlam[jj][ll+0];
			lam[jj][ll+1] += alpha*dlam[jj][ll+1];
			t[jj][ll+0] += alpha*dt[jj][ll+0];
			t[jj][ll+1] += alpha*dt[jj][ll+1];
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
			}
		// soft constraints on states
		for(; ll<2*nb; ll+=2)
			{
			lam[jj][ll+0] += alpha*dlam[jj][ll+0];
			lam[jj][ll+1] += alpha*dlam[jj][ll+1];
			t[jj][ll+0] += alpha*dt[jj][ll+0];
			t[jj][ll+1] += alpha*dt[jj][ll+1];
			lam[jj][anb+ll+0] += alpha*dlam[jj][anb+ll+0];
			lam[jj][anb+ll+1] += alpha*dlam[jj][anb+ll+1];
			t[jj][anb+ll+0] += alpha*dt[jj][anb+ll+0];
			t[jj][anb+ll+1] += alpha*dt[jj][anb+ll+1];
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1] + lam[jj][anb+ll+0] * t[jj][anb+ll+0] + lam[jj][anb+ll+1] * t[jj][anb+ll+1];
			}
		}

	// update states
	for(ll=0; ll<nx; ll++)
		ux[N][nu+ll] += alpha*(dux[N][nu+ll] - ux[N][nu+ll]);
	// update equality constrained multipliers
	for(ll=0; ll<nx; ll++)
		pi[N][ll] += alpha*(dpi[N][ll] - pi[N][ll]);
	// soft constraints on states
	for(ll=2*nu; ll<2*nb; ll+=2)
		{
		lam[N][ll+0] += alpha*dlam[N][ll+0];
		lam[N][ll+1] += alpha*dlam[N][ll+1];
		t[N][ll+0] += alpha*dt[N][ll+0];
		t[N][ll+1] += alpha*dt[N][ll+1];
		lam[N][anb+ll+0] += alpha*dlam[N][anb+ll+0];
		lam[N][anb+ll+1] += alpha*dlam[N][anb+ll+1];
		t[N][anb+ll+0] += alpha*dt[N][anb+ll+0];
		t[N][anb+ll+1] += alpha*dt[N][anb+ll+1];
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1] + lam[N][anb+ll+0] * t[N][anb+ll+0] + lam[N][anb+ll+1] * t[N][anb+ll+1];
		}
	mu *= mu_scal;

	ptr_mu[0] = mu;

	return;
	
	}



void d_compute_mu_box_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	const int nbu = nu<nb ? nu : nb ;

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



void d_compute_mu_soft_mpc(int N, int nx, int nu, int nb, double *ptr_mu, double mu_scal, double alpha, double **lam, double **dlam, double **t, double **dt)
	{
	
	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	int jj, ll;
	
	double mu = 0;
	
	// fist stage: bounds on u only
	for(ll=0; ll<2*nbu; ll+=2)
		{
		mu += (lam[0][ll+0] + alpha*dlam[0][ll+0]) * (t[0][ll+0] + alpha*dt[0][ll+0]) + (lam[0][ll+1] + alpha*dlam[0][ll+1]) * (t[0][ll+1] + alpha*dt[0][ll+1]);
		}

	// middle stages: bounds on both u and x
	for(jj=1; jj<N; jj++)
		{
		for(ll=0; ll<2*nb; ll+=2)
			mu += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);
		for(ll=anb+2*nu; ll<anb+2*nb; ll+=2)
			mu += (lam[jj][ll+0] + alpha*dlam[jj][ll+0]) * (t[jj][ll+0] + alpha*dt[jj][ll+0]) + (lam[jj][ll+1] + alpha*dlam[jj][ll+1]) * (t[jj][ll+1] + alpha*dt[jj][ll+1]);
		}	

	// last stage: bounds on x only
	for(ll=2*nu; ll<2*nb; ll+=2)
		mu += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);
	for(ll=anb+2*nu; ll<anb+2*nb; ll+=2)
		mu += (lam[N][ll+0] + alpha*dlam[N][ll+0]) * (t[N][ll+0] + alpha*dt[N][ll+0]) + (lam[N][ll+1] + alpha*dlam[N][ll+1]) * (t[N][ll+1] + alpha*dt[N][ll+1]);

	mu *= mu_scal;
		
	ptr_mu[0] = mu;

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

