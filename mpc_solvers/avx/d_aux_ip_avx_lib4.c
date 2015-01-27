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
		v_tmp, v_lam, v_lamt, v_dlam, v_db,
		v_tmp0, v_tmp1, v_tmp2, v_tmp3, 
		v_lam0, v_lam1, v_lam2, v_lam3,
		v_lamt0, v_lamt1, v_lamt2, v_lamt3,
		v_dlam0, v_dlam1, v_dlam2, v_dlam3,
		v_Qx0, v_Qx1, v_qx0, v_qx1,
		v_Zl0, v_Zl1, v_zl0, v_zl1,
		v_bd0, v_bd2,
		v_db0, v_db2;
			
	__m128d
		u_tmp0, u_tmp1, u_Qx, u_qx,
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db;
	
	v_ones = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );
	
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv;
	
	int ii, jj, ll, bs0;
	
	// first stage
	jj = 0;
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_t_inv = t_inv[0];
	
	ii = 0;
	for(; ii<nbu-3; ii+=4)
		{
		
		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1   = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_tmp0  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1   = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0   = v_tmp0;
		v_tmp1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1   = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0   = v_tmp1;
		v_Qx0   = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

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

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

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
			
			u_tmp0 = _mm_load_pd( &ptr_t[0] );
			u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp0, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );

			u_Qx   = u_lamt;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt );
			u_qx   = _mm_add_pd( u_qx, u_dlam );
			u_qx   = _mm_add_pd( u_qx, u_lam );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/
			
			ll++;

			}
		ii += ll;
		}
	for( ; ii<nu; ii++)
		{
		pd[jj][ii] = bd[jj][ii];
		pl[jj][ii] = bl[jj][ii];
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
		
			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

			v_Qx0   = v_lamt0;
			v_Qx1   = v_lamt1;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1   = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

			v_tmp0  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1   = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0   = v_tmp0;
			v_tmp1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1   = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0   = v_tmp1;
			v_Qx0   = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0   = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

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

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

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
			
				u_tmp0 = _mm_load_pd( &ptr_t[0] );
				u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
				_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
				u_lam  = _mm_load_pd( &ptr_lam[0] );
				u_lamt = _mm_mul_pd( u_tmp0, u_lam );
				_mm_store_pd( &ptr_lamt[0], u_lamt );
				u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam );

				u_Qx   = u_lamt;
				u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
				u_qx   = _mm_mul_pd( u_qx, u_lamt );
				u_qx   = _mm_add_pd( u_qx, u_dlam );
				u_qx   = _mm_add_pd( u_qx, u_lam );

				u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
				u_qx   = _mm_hsub_pd( u_qx, u_qx );
				u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
				u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
				u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
				_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

	/*			t    += 2;*/
	/*			lam  += 2;*/
	/*			lamt += 2;*/
	/*			dlam += 2;*/
				
				ll++;

				}
			ii += ll;
			}
		for( ; ii<nu+nx; ii++)
			{
			pd[jj][ii] = bd[jj][ii];
			pl[jj][ii] = bl[jj][ii];
			}
			
		}

	// last stage
	jj = N;

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
			u_tmp0 = _mm_load_pd( &ptr_t[0] );
			u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp0, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );

			u_Qx   = u_lamt;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt );
			u_qx   = _mm_add_pd( u_qx, u_dlam );
			u_qx   = _mm_add_pd( u_qx, u_lam );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

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
			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

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
		
		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1   = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_tmp0  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1   = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0   = v_tmp0;
		v_tmp1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1   = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0   = v_tmp1;
		v_Qx0   = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

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

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

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
			
			u_tmp0 = _mm_load_pd( &ptr_t[0] );
			u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp0, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );

			u_Qx   = u_lamt;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt );
			u_qx   = _mm_add_pd( u_qx, u_dlam );
			u_qx   = _mm_add_pd( u_qx, u_lam );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/
			
			ll++;

			}
		ii += ll;
		}
		for( ; ii<nu+nx; ii++)
			{
			pd[jj][ii] = bd[jj][ii];
			pl[jj][ii] = bl[jj][ii];
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
	
	__m256d
		v_zeros, v_ones, v_sigma_mu,
		v_tmp0, v_tmp1, v_tmp2, v_tmp3, 
		v_lam0, v_lam1, v_lam2, v_lam3,
		v_lamt0, v_lamt1, v_lamt2, v_lamt3,
		v_dlam0, v_dlam1, v_dlam2, v_dlam3,
		v_Qx0, v_Qx1, v_qx0, v_qx1,
		v_Zl0, v_Zl1, v_zl0, v_zl1,
		v_bd0, v_bd2,
		v_db0, v_db2;
		
	__m128d
		u_tmp, u_lamt, u_bd, u_bl, u_lam, u_dlam, u_db,
		u_lam0, u_lam1, u_dlam0, u_dlam1, u_lamt0, u_lamt1,
		u_tmp0, u_tmp1, u_Qx, u_qx, u_Zl, u_zl;
	
	__m256i
		i_mask;
	
	v_zeros    = _mm256_setzero_pd();
	v_ones     = _mm256_set_pd( 1.0, 1.0, 1.0, 1.0 );
	v_sigma_mu = _mm256_set_pd( sigma_mu, sigma_mu, sigma_mu, sigma_mu );

	const long long mask2[] = { 1, 1, -1, -1 };
		
	double temp0, temp1;
	
	double *ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_t_inv, 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Z, *ptr_z, *ptr_Zl, *ptr_zl;

	static double Qx[8] = {};
	static double qx[8] = {};
	
	int ii, jj, ll, bs0;
	
	// first stage
	jj = 0;
	
	ptr_t     = t[0];
	ptr_lam   = lam[0];
	ptr_lamt  = lamt[0];
	ptr_dlam  = dlam[0];
	ptr_t_inv = t_inv[0];
	
	ii = 0;
	// hard constraints on u only
	for(; ii<nbu-3; ii+=4)
		{

		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

		v_Qx0   = v_lamt0;
		v_Qx1   = v_lamt1;
		v_qx0   = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1   = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1   = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1   = _mm256_add_pd( v_qx1, v_lam1 );

		v_tmp0  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1   = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0   = v_tmp0;
		v_tmp1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1   = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0   = v_tmp1;
		v_Qx0   = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0   = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;

		}
	if(ii<nbu)
		{
		bs0 = nbu-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

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
			
			u_tmp0 = _mm_load_pd( &ptr_t[0] );
			u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			u_lam  = _mm_load_pd( &ptr_lam[0] );
			u_lamt = _mm_mul_pd( u_tmp0, u_lam );
			_mm_store_pd( &ptr_lamt[0], u_lamt );
			u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam );

			u_Qx   = u_lamt;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt );
			u_qx   = _mm_add_pd( u_qx, u_dlam );
			u_qx   = _mm_add_pd( u_qx, u_lam );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

/*			t    += 2;*/
/*			lam  += 2;*/
/*			lamt += 2;*/
/*			dlam += 2;*/
			
			ll++;

			}
		ii += ll;
		}
	for( ; ii<nu; ii++)
		{
		pd[jj][ii] = bd[jj][ii];
		pl[jj][ii] = bl[jj][ii];
		}


	// middle stages

	for(jj=1; jj<N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_lamt  = lamt[jj];
		ptr_dlam  = dlam[jj];
		ptr_t_inv = t_inv[jj];

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

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 ); // store t_inv
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 ); // store t_inv
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );

			v_Qx0  = v_lamt0;
			v_Qx1  = v_lamt1;
			v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

			v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0  = v_tmp0;
			v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0  = v_tmp1;
			v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

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
			if(bs0>=2)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;
				
				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
				
				ll   += 2;
				bs0  -= 2;

				}
			
			if(bs0>0)
				{
				if(nbu<nb) // there are soft constraints afterwards
					{

					v_tmp0  = _mm256_load_pd( &ptr_t[0] );
					u_tmp1  = _mm_load_pd( &ptr_t[anb+2] );
					v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
					u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
					_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
					_mm_store_pd( &ptr_t_inv[anb+2], u_tmp1 );
					v_lam0  = _mm256_load_pd( &ptr_lam[0] );
					u_lam1  = _mm_load_pd( &ptr_lam[anb+2] );
					v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
					u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
					_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
					_mm_store_pd( &ptr_lamt[anb+2], u_lamt1 );
					v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
					u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
					_mm_store_pd( &ptr_dlam[anb+2], u_dlam1 );

					v_Qx0   = v_lamt0;
					v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
					v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
					v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

					u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
					u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
					u_Zl   = _mm_load_pd( &ptr_Z[2] );
					u_zl   = _mm_load_pd( &ptr_z[2] );
					u_Zl   = _mm_add_pd( u_Zl, u_Qx );
					u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
					u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
					u_zl   = _mm_sub_pd( u_qx, u_zl );
					u_zl   = _mm_add_pd( u_zl, u_lam1 );
					u_zl   = _mm_add_pd( u_zl, u_dlam1 );
					_mm_store_pd( &ptr_Zl[2], u_Zl );
					_mm_store_pd( &ptr_zl[2], u_zl );
					u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
					u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
					u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
					u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
					u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

					u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
					u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
					u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
					u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
					u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
					u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
					_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 4;
					ptr_lam   += 4;
					ptr_lamt  += 4;
					ptr_dlam  += 4;
					ptr_t_inv += 4;

					ptr_db    += 4;
					ptr_Z     += 4;
					ptr_z     += 4;
					ptr_Zl    += 4;
					ptr_zl    += 4;
				
					ll   += 2;
					}
				else // no soft constraints afterward
					{

					u_tmp0 = _mm_load_pd( &ptr_t[0] );
					u_tmp0 = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
					_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
					u_lam  = _mm_load_pd( &ptr_lam[0] );
					u_lamt = _mm_mul_pd( u_tmp0, u_lam );
					_mm_store_pd( &ptr_lamt[0], u_lamt );
					u_dlam = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
					_mm_store_pd( &ptr_dlam[0], u_dlam );

					u_Qx   = u_lamt;
					u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
					u_qx   = _mm_mul_pd( u_qx, u_lamt );
					u_qx   = _mm_add_pd( u_qx, u_dlam );
					u_qx   = _mm_add_pd( u_qx, u_lam );

					u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
					u_qx   = _mm_hsub_pd( u_qx, u_qx );
					u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
					u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
					u_tmp0 = _mm_add_sd( u_Qx, u_tmp0 );
					u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
					_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
					_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

					ptr_t     += 2;
					ptr_lam   += 2;
					ptr_lamt  += 2;
					ptr_dlam  += 2;
					ptr_t_inv += 2;

					ptr_db    += 2;
					ptr_Z     += 2;
					ptr_z     += 2;
					ptr_Zl    += 2;
					ptr_zl    += 2;
					
					ll++;

					}

				}
		
			// soft constraints on x
			// clean-up loop
			bs0 = nb-ii<4 ? nb-ii : 4 ; // either 0 ro 2 constraints to be done !!!

			if(ll<bs0)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll   += 2;

				}
			ii += ll;
			}
		// main loop
		for(; ii<nb-3; ii+=4)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp1  = _mm256_load_pd( &ptr_t[4] );
			v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
			v_tmp3  = _mm256_load_pd( &ptr_t[anb+4] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
			v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
			v_tmp3  = _mm256_div_pd( v_ones, v_tmp3 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[4], v_tmp1 );
			_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
			_mm256_store_pd( &ptr_t_inv[anb+4], v_tmp3 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam1  = _mm256_load_pd( &ptr_lam[4] );
			v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
			v_lam3  = _mm256_load_pd( &ptr_lam[anb+4] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
			v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
			v_lamt3 = _mm256_mul_pd( v_tmp3, v_lam3 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
			_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
			_mm256_store_pd( &ptr_lamt[anb+4], v_lamt3 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
			v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
			v_dlam3 = _mm256_mul_pd( v_tmp3, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[4], v_dlam1 );
			_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );
			_mm256_store_pd( &ptr_dlam[anb+4], v_dlam3 );

			v_Qx0 = v_lamt0;
			v_Qx1 = v_lamt1;
			v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
			v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
			v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
			v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
			v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

			v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
			v_Zl1  = _mm256_load_pd( &ptr_Z[4] );
			v_zl0  = _mm256_load_pd( &ptr_z[0] );
			v_zl1  = _mm256_load_pd( &ptr_z[4] );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
			v_Zl1  = _mm256_add_pd( v_Zl1, v_Qx1 );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
			v_Zl1  = _mm256_add_pd( v_Zl1, v_lamt3 );
			v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
			v_Zl1  = _mm256_div_pd( v_ones, v_Zl1 );
			v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
			v_zl1  = _mm256_sub_pd( v_qx1, v_zl1 );
			v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
			v_zl1  = _mm256_add_pd( v_zl1, v_lam3 );
			v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
			v_zl1  = _mm256_add_pd( v_zl1, v_dlam3 );
			_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
			_mm256_store_pd( &ptr_Zl[4], v_Zl1 );
			_mm256_store_pd( &ptr_zl[0], v_zl0 );
			_mm256_store_pd( &ptr_zl[4], v_zl1 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
			v_tmp1 = _mm256_mul_pd( v_Qx1, v_Zl1 );
			v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
			v_tmp3 = _mm256_mul_pd( v_tmp1, v_zl1 );
			v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
			v_qx1  = _mm256_sub_pd( v_qx1, v_tmp3 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
			v_tmp1 = _mm256_mul_pd( v_Qx1, v_tmp1 );
			v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );
			v_Qx1  = _mm256_sub_pd( v_Qx1, v_tmp1 );

			v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
			v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
			v_Qx0  = v_tmp0;
			v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
			v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
			v_qx0  = v_tmp1;
			v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
			v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
			v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
			v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
			v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
			v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
			_mm256_store_pd( &pd[jj][ii], v_tmp0 );
			_mm256_store_pd( &pl[jj][ii], v_tmp1 );

			ptr_t     += 8;
			ptr_lam   += 8;
			ptr_lamt  += 8;
			ptr_dlam  += 8;
			ptr_t_inv += 8;

			ptr_db    += 8;
			ptr_Z     += 8;
			ptr_z     += 8;
			ptr_Zl    += 8;
			ptr_zl    += 8;

			}
		if(ii<nb)
			{
			bs0 = nb-ii;
			ll = 0;
			
			if(bs0>=2)
				{

				v_tmp0  = _mm256_load_pd( &ptr_t[0] );
				v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
				v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
				v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
				_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
				_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
				v_lam0  = _mm256_load_pd( &ptr_lam[0] );
				v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
				v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
				v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
				_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
				_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
				v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
				v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
				_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
				_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );

				v_Qx0   = v_lamt0;
				v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
				v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
				v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

				v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
				v_zl0  = _mm256_load_pd( &ptr_z[0] );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
				v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
				v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
				v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
				v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
				v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
				_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
				_mm256_store_pd( &ptr_zl[0], v_zl0 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
				v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
				v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
				v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
				v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

				u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
				u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
				u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
				u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
				u_tmp0  = _mm_load_pd( &bd[jj][ii] );
				u_tmp1  = _mm_load_pd( &bl[jj][ii] );
				u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
				u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
				_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
				_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

				ptr_t     += 4;
				ptr_lam   += 4;
				ptr_lamt  += 4;
				ptr_dlam  += 4;
				ptr_t_inv += 4;

				ptr_db    += 4;
				ptr_Z     += 4;
				ptr_z     += 4;
				ptr_Zl    += 4;
				ptr_zl    += 4;
			
				ll   += 2;
				bs0  -= 2;

				}
			
			if(bs0>0)
				{
				
				u_tmp0  = _mm_load_pd( &ptr_t[0] );
				u_tmp1  = _mm_load_pd( &ptr_t[anb+0] );
				u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
				u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
				_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
				_mm_store_pd( &ptr_t_inv[anb+0], u_tmp1 );
				u_lam0  = _mm_load_pd( &ptr_lam[0] );
				u_lam1  = _mm_load_pd( &ptr_lam[anb+0] );
				u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
				u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
				_mm_store_pd( &ptr_lamt[0], u_lamt0 );
				_mm_store_pd( &ptr_lamt[anb+0], u_lamt1 );
				u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
				u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
				_mm_store_pd( &ptr_dlam[0], u_dlam0 );
				_mm_store_pd( &ptr_dlam[anb+0], u_dlam1 );

				u_Qx   = u_lamt0;
				u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
				u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
				u_qx   = _mm_add_pd( u_qx, u_dlam0 );
				u_qx   = _mm_add_pd( u_qx, u_lam0 );

				u_Zl   = _mm_load_pd( &ptr_Z[0] );
				u_zl   = _mm_load_pd( &ptr_z[0] );
				u_Zl   = _mm_add_pd( u_Zl, u_Qx );
				u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
				u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
				u_zl   = _mm_sub_pd( u_qx, u_zl );
				u_zl   = _mm_add_pd( u_zl, u_lam1 );
				u_zl   = _mm_add_pd( u_zl, u_dlam1 );
				_mm_store_pd( &ptr_Zl[0], u_Zl );
				_mm_store_pd( &ptr_zl[0], u_zl );
				u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
				u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
				u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
				u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
				u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

				u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
				u_qx   = _mm_hsub_pd( u_qx, u_qx );
				u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
				u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
				u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
				u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
				_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
				_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

//				ptr_t     += 2;
//				ptr_lam   += 2;
//				ptr_lamt  += 2;
//				ptr_dlam  += 2;
//				ptr_t_inv += 2;

//				ptr_db    += 2;
//				ptr_Z     += 2;
//				ptr_z     += 2;
//				ptr_Zl    += 2;
//				ptr_zl    += 2;

				ll++;
				}

			ii += ll;
			}
		for( ; ii<nu+nx; ii++)
			{
			pd[jj][ii] = bd[jj][ii];
			pl[jj][ii] = bl[jj][ii];
			}
	
		}

	// last stage
	jj = N;

	ptr_t     = t[N]     + 2*nu;
	ptr_lam   = lam[N]   + 2*nu;
	ptr_lamt  = lamt[N]  + 2*nu;
	ptr_dlam  = dlam[N]  + 2*nu;
	ptr_t_inv  = t_inv[N] + 2*nu;
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

		if(ll%2==1)
			{

			u_tmp0  = _mm_load_pd( &ptr_t[0] );
			u_tmp1  = _mm_load_pd( &ptr_t[anb+0] );
			u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			_mm_store_pd( &ptr_t_inv[anb+0], u_tmp1 );
			u_lam0  = _mm_load_pd( &ptr_lam[0] );
			u_lam1  = _mm_load_pd( &ptr_lam[anb+0] );
			u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
			u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
			_mm_store_pd( &ptr_lamt[0], u_lamt0 );
			_mm_store_pd( &ptr_lamt[anb+0], u_lamt1 );
			u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam0 );
			_mm_store_pd( &ptr_dlam[anb+0], u_dlam1 );

			u_Qx   = u_lamt0;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
			u_qx   = _mm_add_pd( u_qx, u_dlam0 );
			u_qx   = _mm_add_pd( u_qx, u_lam0 );

			u_Zl   = _mm_load_pd( &ptr_Z[0] );
			u_zl   = _mm_load_pd( &ptr_z[0] );
			u_Zl   = _mm_add_pd( u_Zl, u_Qx );
			u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
			u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
			u_zl   = _mm_sub_pd( u_qx, u_zl );
			u_zl   = _mm_add_pd( u_zl, u_lam1 );
			u_zl   = _mm_add_pd( u_zl, u_dlam1 );
			_mm_store_pd( &ptr_Zl[0], u_Zl );
			_mm_store_pd( &ptr_zl[0], u_zl );
			u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
			u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
			u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
			u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
			u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

			ptr_t     += 2;
			ptr_lam   += 2;
			ptr_lamt  += 2;
			ptr_dlam  += 2;
			ptr_t_inv  += 2;

			ptr_db    += 2;
			ptr_Z     += 2;
			ptr_z     += 2;
			ptr_Zl    += 2;
			ptr_zl    += 2;

			ll++;
			}
		if(ll<bs0)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii+2*ll] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
			v_zl0  = _mm256_load_pd( &ptr_z[0] );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
			v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
			v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
			v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
			v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
			_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
			_mm256_store_pd( &ptr_zl[0], v_zl0 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
			v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
			v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
			v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii+ll] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii+ll] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+ll], u_tmp1 );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;

			ptr_db    += 4;
			ptr_Z     += 4;
			ptr_z     += 4;
			ptr_Zl    += 4;
			ptr_zl    += 4;
		

			ll+=2;
			}

		ii += 4;
		}

	for(; ii<nb-3; ii+=4)
		{

		v_tmp0  = _mm256_load_pd( &ptr_t[0] );
		v_tmp1  = _mm256_load_pd( &ptr_t[4] );
		v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
		v_tmp3  = _mm256_load_pd( &ptr_t[anb+4] );
		v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
		v_tmp1  = _mm256_div_pd( v_ones, v_tmp1 );
		v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
		v_tmp3  = _mm256_div_pd( v_ones, v_tmp3 );
		_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
		_mm256_store_pd( &ptr_t_inv[4], v_tmp1 );
		_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
		_mm256_store_pd( &ptr_t_inv[anb+4], v_tmp3 );
		v_lam0  = _mm256_load_pd( &ptr_lam[0] );
		v_lam1  = _mm256_load_pd( &ptr_lam[4] );
		v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
		v_lam3  = _mm256_load_pd( &ptr_lam[anb+4] );
		v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
		v_lamt1 = _mm256_mul_pd( v_tmp1, v_lam1 );
		v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
		v_lamt3 = _mm256_mul_pd( v_tmp3, v_lam3 );
		_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
		_mm256_store_pd( &ptr_lamt[4], v_lamt1 );
		_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
		_mm256_store_pd( &ptr_lamt[anb+4], v_lamt3 );
		v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
		v_dlam1 = _mm256_mul_pd( v_tmp1, v_sigma_mu );
		v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
		v_dlam3 = _mm256_mul_pd( v_tmp3, v_sigma_mu );
		_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
		_mm256_store_pd( &ptr_dlam[4], v_dlam1 );
		_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );
		_mm256_store_pd( &ptr_dlam[anb+4], v_dlam3 );

		v_Qx0 = v_lamt0;
		v_Qx1 = v_lamt1;
		v_qx0  = _mm256_load_pd( &db[jj][2*ii+0] );
		v_qx1  = _mm256_load_pd( &db[jj][2*ii+4] );
		v_qx0  = _mm256_mul_pd( v_qx0, v_lamt0 );
		v_qx1  = _mm256_mul_pd( v_qx1, v_lamt1 );
		v_qx0  = _mm256_add_pd( v_qx0, v_dlam0 );
		v_qx1  = _mm256_add_pd( v_qx1, v_dlam1 );
		v_qx0  = _mm256_add_pd( v_qx0, v_lam0 );
		v_qx1  = _mm256_add_pd( v_qx1, v_lam1 );

		v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
		v_Zl1  = _mm256_load_pd( &ptr_Z[4] );
		v_zl0  = _mm256_load_pd( &ptr_z[0] );
		v_zl1  = _mm256_load_pd( &ptr_z[4] );
		v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
		v_Zl1  = _mm256_add_pd( v_Zl1, v_Qx1 );
		v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
		v_Zl1  = _mm256_add_pd( v_Zl1, v_lamt3 );
		v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
		v_Zl1  = _mm256_div_pd( v_ones, v_Zl1 );
		v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
		v_zl1  = _mm256_sub_pd( v_qx1, v_zl1 );
		v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
		v_zl1  = _mm256_add_pd( v_zl1, v_lam3 );
		v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
		v_zl1  = _mm256_add_pd( v_zl1, v_dlam3 );
		_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
		_mm256_store_pd( &ptr_Zl[4], v_Zl1 );
		_mm256_store_pd( &ptr_zl[0], v_zl0 );
		_mm256_store_pd( &ptr_zl[4], v_zl1 );
		v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
		v_tmp1 = _mm256_mul_pd( v_Qx1, v_Zl1 );
		v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
		v_tmp3 = _mm256_mul_pd( v_tmp1, v_zl1 );
		v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
		v_qx1  = _mm256_sub_pd( v_qx1, v_tmp3 );
		v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
		v_tmp1 = _mm256_mul_pd( v_Qx1, v_tmp1 );
		v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );
		v_Qx1  = _mm256_sub_pd( v_Qx1, v_tmp1 );

		v_tmp0 = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x20 );
		v_Qx1  = _mm256_permute2f128_pd( v_Qx0, v_Qx1, 0x31 );
		v_Qx0  = v_tmp0;
		v_tmp1 = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x20 );
		v_qx1  = _mm256_permute2f128_pd( v_qx0, v_qx1, 0x31 );
		v_qx0  = v_tmp1;
		v_Qx0  = _mm256_hadd_pd( v_Qx0, v_Qx1 );
		v_qx0  = _mm256_hsub_pd( v_qx0, v_qx1 );
		v_tmp0  = _mm256_load_pd( &bd[jj][ii] );
		v_tmp1  = _mm256_load_pd( &bl[jj][ii] );
		v_tmp0  = _mm256_add_pd( v_Qx0, v_tmp0 );
		v_tmp1  = _mm256_sub_pd( v_tmp1, v_qx0 );
		_mm256_store_pd( &pd[jj][ii], v_tmp0 );
		_mm256_store_pd( &pl[jj][ii], v_tmp1 );

		ptr_t     += 8;
		ptr_lam   += 8;
		ptr_lamt  += 8;
		ptr_dlam  += 8;
		ptr_t_inv += 8;
		ptr_db    += 8;
		ptr_Z     += 8;
		ptr_z     += 8;
		ptr_Zl    += 8;
		ptr_zl    += 8;

		}
	if(ii<nb)
		{
		bs0 = nb-ii;
		ll = 0;
		
		if(bs0>=2)
			{

			v_tmp0  = _mm256_load_pd( &ptr_t[0] );
			v_tmp2  = _mm256_load_pd( &ptr_t[anb+0] );
			v_tmp0  = _mm256_div_pd( v_ones, v_tmp0 );
			v_tmp2  = _mm256_div_pd( v_ones, v_tmp2 );
			_mm256_store_pd( &ptr_t_inv[0], v_tmp0 );
			_mm256_store_pd( &ptr_t_inv[anb+0], v_tmp2 );
			v_lam0  = _mm256_load_pd( &ptr_lam[0] );
			v_lam2  = _mm256_load_pd( &ptr_lam[anb+0] );
			v_lamt0 = _mm256_mul_pd( v_tmp0, v_lam0 );
			v_lamt2 = _mm256_mul_pd( v_tmp2, v_lam2 );
			_mm256_store_pd( &ptr_lamt[0], v_lamt0 );
			_mm256_store_pd( &ptr_lamt[anb+0], v_lamt2 );
			v_dlam0 = _mm256_mul_pd( v_tmp0, v_sigma_mu );
			v_dlam2 = _mm256_mul_pd( v_tmp2, v_sigma_mu );
			_mm256_store_pd( &ptr_dlam[0], v_dlam0 );
			_mm256_store_pd( &ptr_dlam[anb+0], v_dlam2 );

			v_Qx0   = v_lamt0;
			v_qx0   = _mm256_load_pd( &db[jj][2*ii] );
			v_qx0   = _mm256_mul_pd( v_qx0, v_lamt0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_dlam0 );
			v_qx0   = _mm256_add_pd( v_qx0, v_lam0 );

			v_Zl0  = _mm256_load_pd( &ptr_Z[0] );
			v_zl0  = _mm256_load_pd( &ptr_z[0] );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_Qx0 );
			v_Zl0  = _mm256_add_pd( v_Zl0, v_lamt2 );
			v_Zl0  = _mm256_div_pd( v_ones, v_Zl0 );
			v_zl0  = _mm256_sub_pd( v_qx0, v_zl0 );
			v_zl0  = _mm256_add_pd( v_zl0, v_lam2 );
			v_zl0  = _mm256_add_pd( v_zl0, v_dlam2 );
			_mm256_store_pd( &ptr_Zl[0], v_Zl0 );
			_mm256_store_pd( &ptr_zl[0], v_zl0 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_Zl0 );
			v_tmp2 = _mm256_mul_pd( v_tmp0, v_zl0 );
			v_qx0  = _mm256_sub_pd( v_qx0, v_tmp2 );
			v_tmp0 = _mm256_mul_pd( v_Qx0, v_tmp0 );
			v_Qx0  = _mm256_sub_pd( v_Qx0, v_tmp0 );

			u_Qx    = _mm256_extractf128_pd( v_Qx0, 0x1 );
			u_qx    = _mm256_extractf128_pd( v_qx0, 0x1 );
			u_Qx    = _mm_hadd_pd( _mm256_castpd256_pd128( v_Qx0 ), u_Qx );
			u_qx    = _mm_hsub_pd( _mm256_castpd256_pd128( v_qx0 ), u_qx );
			u_tmp0  = _mm_load_pd( &bd[jj][ii] );
			u_tmp1  = _mm_load_pd( &bl[jj][ii] );
			u_tmp0  = _mm_add_pd( u_tmp0, u_Qx );
			u_tmp1  = _mm_sub_pd( u_tmp1, u_qx );
			_mm_store_pd( &pd[jj][ii+0], u_tmp0 );
			_mm_store_pd( &pl[jj][ii+0], u_tmp1 );

			ptr_t     += 4;
			ptr_lam   += 4;
			ptr_lamt  += 4;
			ptr_dlam  += 4;
			ptr_t_inv += 4;

			ptr_db    += 4;
			ptr_Z     += 4;
			ptr_z     += 4;
			ptr_Zl    += 4;
			ptr_zl    += 4;
		
			ll   += 2;
			bs0  -= 2;

			}
		
		if(bs0>0)
			{
			
			u_tmp0  = _mm_load_pd( &ptr_t[0] );
			u_tmp1  = _mm_load_pd( &ptr_t[anb+0] );
			u_tmp0  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp0 );
			u_tmp1  = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_tmp1 );
			_mm_store_pd( &ptr_t_inv[0], u_tmp0 );
			_mm_store_pd( &ptr_t_inv[anb+0], u_tmp1 );
			u_lam0  = _mm_load_pd( &ptr_lam[0] );
			u_lam1  = _mm_load_pd( &ptr_lam[anb+0] );
			u_lamt0 = _mm_mul_pd( u_tmp0, u_lam0 );
			u_lamt1 = _mm_mul_pd( u_tmp1, u_lam1 );
			_mm_store_pd( &ptr_lamt[0], u_lamt0 );
			_mm_store_pd( &ptr_lamt[anb+0], u_lamt1 );
			u_dlam0 = _mm_mul_pd( u_tmp0, _mm256_castpd256_pd128( v_sigma_mu ) );
			u_dlam1 = _mm_mul_pd( u_tmp1, _mm256_castpd256_pd128( v_sigma_mu ) );
			_mm_store_pd( &ptr_dlam[0], u_dlam0 );
			_mm_store_pd( &ptr_dlam[anb+0], u_dlam1 );

			u_Qx   = u_lamt0;
			u_qx   = _mm_load_pd( &db[jj][2*ii+2*ll] );
			u_qx   = _mm_mul_pd( u_qx, u_lamt0 );
			u_qx   = _mm_add_pd( u_qx, u_dlam0 );
			u_qx   = _mm_add_pd( u_qx, u_lam0 );

			u_Zl   = _mm_load_pd( &ptr_Z[0] );
			u_zl   = _mm_load_pd( &ptr_z[0] );
			u_Zl   = _mm_add_pd( u_Zl, u_Qx );
			u_Zl   = _mm_add_pd( u_Zl, u_lamt1 );
			u_Zl   = _mm_div_pd( _mm256_castpd256_pd128( v_ones ), u_Zl );
			u_zl   = _mm_sub_pd( u_qx, u_zl );
			u_zl   = _mm_add_pd( u_zl, u_lam1 );
			u_zl   = _mm_add_pd( u_zl, u_dlam1 );
			_mm_store_pd( &ptr_Zl[0], u_Zl );
			_mm_store_pd( &ptr_zl[0], u_zl );
			u_tmp0 = _mm_mul_pd( u_Qx, u_Zl );
			u_tmp1 = _mm_mul_pd( u_tmp0, u_zl );
			u_qx  = _mm_sub_pd( u_qx, u_tmp1 );
			u_tmp0 = _mm_mul_pd( u_Qx, u_tmp0 );
			u_Qx  = _mm_sub_pd( u_Qx, u_tmp0 );

			u_Qx   = _mm_hadd_pd( u_Qx, u_Qx );
			u_qx   = _mm_hsub_pd( u_qx, u_qx );
			u_tmp0 = _mm_load_sd( &bd[jj][ii+ll] );
			u_tmp1 = _mm_load_sd( &bl[jj][ii+ll] );
			u_tmp0 = _mm_add_sd( u_tmp0, u_Qx );
			u_tmp1 = _mm_sub_sd( u_tmp1, u_qx );
			_mm_store_sd( &pd[jj][ii+ll], u_tmp0 );
			_mm_store_sd( &pl[jj][ii+ll], u_tmp1 );

//				ptr_t     += 2;
//				ptr_lam   += 2;
//				ptr_lamt  += 2;
//				ptr_dlam  += 2;
//				ptr_t_inv += 2;

//				ptr_db    += 2;
//				ptr_Z     += 2;
//				ptr_z     += 2;
//				ptr_Zl    += 2;
//				ptr_zl    += 2;

			ll++;
			}

		ii += ll;
		}
	for( ; ii<nu+nx; ii++)
		{
		pd[jj][ii] = bd[jj][ii];
		pl[jj][ii] = bl[jj][ii];
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
	
	const int nbu = nu<nb ? nu : nb ;
	const int nbx = nb-nu>0 ? nb-nu : 0 ;

	// constants
	const int bs = 4; //D_MR;
	const int ncl = D_NCL;
	const int nal = bs*ncl; // number of doubles per cache line

	const int anb = nal*((2*nb+nal-1)/nal); // cache aligned number of box and soft constraints // !!!!! doubled to include soft constraints !!!!!

	__m256
		t_sign, t_ones, t_zeros,
		t_mask0, t_mask1,
		t_lam, t_dlam, t_t, t_dt,
		t_lam_z, t_dlam_z,
		t_tmp0, t_tmp1,
		t_alpha0, t_alpha1;
		
	__m128
		s_sign, s_ones, s_zeros,
		s_mask0, s_mask1,
		s_lam, s_dlam, s_t, s_dt,
		s_lam_z, s_dlam_z, s_t_z, s_dt_z,
		s_tmp0, s_tmp1,
		s_alpha0, s_alpha1;
	
	__m256d
		v_sign, v_temp, v_tmp0, v_tmp1,
		v_dt, v_dux, v_db, v_dlam, v_lamt, v_t, v_alpha, v_lam,
		v_dt_z, v_dlam_z, v_lamt_z, v_t_z, v_lam_z;
	
	__m128d
		u_sign, u_temp, u_tmp0, u_tmp1,
		u_dux, u_db, u_alpha, 
		u_dt, u_dlam, u_lamt, u_t, u_lam,
		u_dt_z, u_dlam_z, u_lamt_z, u_t_z, u_lam_z;
	
	long long long_sign = 0x8000000000000000;
	v_sign = _mm256_broadcast_sd( (double *) &long_sign );
	u_sign = _mm_loaddup_pd( (double *) &long_sign );

	int int_sign = 0x80000000;
	s_sign = _mm_broadcast_ss( (float *) &int_sign );
	t_sign = _mm256_broadcast_ss( (float *) &int_sign );
	
	s_ones  = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_zeros = _mm_setzero_ps( );

	t_ones  = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_zeros = _mm256_setzero_ps( );

	s_alpha0 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );
	s_alpha1 = _mm_set_ps( 1.0, 1.0, 1.0, 1.0 );

	t_alpha0 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	t_alpha1 = _mm256_set_ps( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 );
	
	double alpha;

	int jj, ll;


	// first stage
	jj = 0;

	ll = 0;
	// hard input constraints
	for(; ll<nbu-1; ll+=2)
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

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		}

	for(; ll<nbu; ll++)
		{

		u_db     = _mm_load_pd( &db[jj][2*ll] );
		u_dux    = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_dt     = _mm_addsub_pd( u_db, u_dux );
		u_dt     = _mm_xor_pd( u_dt, u_sign );
		u_t      = _mm_load_pd( &t[jj][2*ll] );
		u_dt     = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );

		u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp   = _mm_mul_pd( u_lamt, u_dt );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam   = _mm_sub_pd( u_dlam, u_temp );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		s_dlam   = _mm256_cvtpd_ps( v_dlam );
		s_mask0  = _mm_cmplt_ps( s_dlam, s_zeros );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		s_lam    = _mm256_cvtpd_ps( v_lam );
		s_lam    = _mm_xor_ps( s_lam, s_sign );
		s_tmp0   = _mm_div_ps( s_lam, s_dlam );
		s_tmp0   = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
		s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

		}
	
	// TODO possibly soft constraints on u

	// middle stages
	for(jj=1; jj<N; jj++)
		{

		ll = 0;
		// hard input constraints
		for(; ll<nbu-1; ll+=2)
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

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		for(; ll<nbu; ll++)
			{

			u_db     = _mm_load_pd( &db[jj][2*ll] );
			u_dux    = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_dt     = _mm_addsub_pd( u_db, u_dux );
			u_dt     = _mm_xor_pd( u_dt, u_sign );
			u_t      = _mm_load_pd( &t[jj][2*ll] );
			u_dt     = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );

			u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp   = _mm_mul_pd( u_lamt, u_dt );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam   = _mm_sub_pd( u_dlam, u_temp );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			s_dlam   = _mm256_cvtpd_ps( v_dlam );
			s_mask0  = _mm_cmplt_ps( s_dlam, s_zeros );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			s_lam    = _mm256_cvtpd_ps( v_lam );
			s_lam    = _mm_xor_ps( s_lam, s_sign );
			s_tmp0   = _mm_div_ps( s_lam, s_dlam );
			s_tmp0   = _mm_blendv_ps( s_ones, s_tmp0, s_mask0 );
			s_alpha0 = _mm_min_ps( s_alpha0, s_tmp0 );

			}

		// soft state constraints
		if(ll<nb && ll%2==1)
			{

			u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dux );
			u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
			u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
			u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
			u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
			u_db    = _mm_load_pd( &db[jj][2*ll] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_dt    = _mm_add_pd( u_dt, u_dt_z );
			u_t     = _mm_load_pd( &t[jj][2*ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );
			u_t_z   = _mm_load_pd( &t[jj][anb+2*ll] );
			u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
			_mm_store_pd( &dt[jj][anb+2*ll], u_dt_z );

			//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_lamt_z = _mm_load_pd( &lamt[jj][anb+2*ll] );
			u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
			u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_dlam_z = _mm_load_pd( &dlam[jj][anb+2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_lam_z  = _mm_load_pd( &lam[jj][anb+2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
			u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );
			_mm_store_pd( &dlam[jj][anb+2*ll], u_dlam_z );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			ll++;

			}

		for(; ll<nb-1; ll+=2)
			{

			v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
			v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
			v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
			v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
			v_temp  = _mm256_mul_pd( v_lamt, v_dux );
			v_dt_z  = _mm256_load_pd( &zl[jj][2*ll] );
			v_dt_z  = _mm256_addsub_pd( v_dt_z, v_temp );
			v_temp  = _mm256_load_pd( &Zl[jj][2*ll] );
			v_dt_z  = _mm256_mul_pd( v_dt_z, v_temp );
			v_db    = _mm256_load_pd( &db[jj][2*ll] );
			v_dt    = _mm256_addsub_pd( v_db, v_dux );
			v_dt    = _mm256_xor_pd( v_dt, v_sign );
			v_dt    = _mm256_add_pd( v_dt, v_dt_z );
			v_t     = _mm256_load_pd( &t[jj][2*ll] );
			v_dt    = _mm256_sub_pd( v_dt, v_t );
			_mm256_store_pd( &dt[jj][2*ll], v_dt );
			v_t_z   = _mm256_load_pd( &t[jj][anb+2*ll] );
			v_dt_z  = _mm256_sub_pd( v_dt_z, v_t_z );
			_mm256_store_pd( &dt[jj][anb+2*ll], v_dt_z );

			//v_lamt   = _mm256_load_pd( &lamt[jj][2*ll] );
			v_lamt_z = _mm256_load_pd( &lamt[jj][anb+2*ll] );
			v_tmp0   = _mm256_mul_pd( v_lamt, v_dt );
			v_tmp1   = _mm256_mul_pd( v_lamt_z, v_dt_z );
			v_dlam   = _mm256_load_pd( &dlam[jj][2*ll] );
			v_dlam_z = _mm256_load_pd( &dlam[jj][anb+2*ll] );
			v_lam    = _mm256_load_pd( &lam[jj][2*ll] );
			v_lam_z  = _mm256_load_pd( &lam[jj][anb+2*ll] );
			v_dlam   = _mm256_sub_pd( v_dlam, v_lam );
			v_dlam_z = _mm256_sub_pd( v_dlam_z, v_lam_z );
			v_dlam   = _mm256_sub_pd( v_dlam, v_tmp0 );
			v_dlam_z = _mm256_sub_pd( v_dlam_z, v_tmp1 );
			_mm256_store_pd( &dlam[jj][2*ll], v_dlam );
			_mm256_store_pd( &dlam[jj][anb+2*ll], v_dlam_z );

			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
			t_dlam_z = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			t_mask1  = _mm256_cmp_ps( t_dlam_z, t_zeros, 0x01 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
			t_lam_z  = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_lam_z  = _mm256_xor_ps( t_lam_z, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp1   = _mm256_div_ps( t_lam_z, t_dlam_z );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
			t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

			}

		for(; ll<nb; ll++)
			{

			u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
			u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
			u_temp  = _mm_mul_pd( u_lamt, u_dux );
			u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
			u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
			u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
			u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
			u_db    = _mm_load_pd( &db[jj][2*ll] );
			u_dt    = _mm_addsub_pd( u_db, u_dux );
			u_dt    = _mm_xor_pd( u_dt, u_sign );
			u_dt    = _mm_add_pd( u_dt, u_dt_z );
			u_t     = _mm_load_pd( &t[jj][2*ll] );
			u_dt    = _mm_sub_pd( u_dt, u_t );
			_mm_store_pd( &dt[jj][2*ll], u_dt );
			u_t_z   = _mm_load_pd( &t[jj][anb+2*ll] );
			u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
			_mm_store_pd( &dt[jj][anb+2*ll], u_dt_z );

			//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
			u_lamt_z = _mm_load_pd( &lamt[jj][anb+2*ll] );
			u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
			u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
			u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
			u_dlam_z = _mm_load_pd( &dlam[jj][anb+2*ll] );
			u_lam    = _mm_load_pd( &lam[jj][2*ll] );
			u_lam_z  = _mm_load_pd( &lam[jj][anb+2*ll] );
			u_dlam   = _mm_sub_pd( u_dlam, u_lam );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
			u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
			u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
			_mm_store_pd( &dlam[jj][2*ll], u_dlam );
			_mm_store_pd( &dlam[jj][anb+2*ll], u_dlam_z );

			v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
			v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
			t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
			t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
			v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
			v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
			t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
			t_lam    = _mm256_xor_ps( t_lam, t_sign );
			t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
			t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
			t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

			}

		}		

	// last stage
	jj = N;
	ll = nu;

	// soft state constraints
	if(ll<nb && ll%2==1)
		{

		u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dux );
		u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
		u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
		u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
		u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
		u_db    = _mm_load_pd( &db[jj][2*ll] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_dt    = _mm_add_pd( u_dt, u_dt_z );
		u_t     = _mm_load_pd( &t[jj][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );
		u_t_z   = _mm_load_pd( &t[jj][anb+2*ll] );
		u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
		_mm_store_pd( &dt[jj][anb+2*ll], u_dt_z );

		//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_lamt_z = _mm_load_pd( &lamt[jj][anb+2*ll] );
		u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
		u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_dlam_z = _mm_load_pd( &dlam[jj][anb+2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_lam_z  = _mm_load_pd( &lam[jj][anb+2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
		u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );
		_mm_store_pd( &dlam[jj][anb+2*ll], u_dlam_z );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		ll++;

		}

	for(; ll<nb-1; ll+=2)
		{

		v_dux   = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+0] ) );
		v_temp  = _mm256_castpd128_pd256( _mm_loaddup_pd( &dux[jj][ll+1] ) );
		v_dux   = _mm256_permute2f128_pd( v_dux, v_temp, 0x20 );
		v_lamt  = _mm256_load_pd( &lamt[jj][2*ll] );
		v_temp  = _mm256_mul_pd( v_lamt, v_dux );
		v_dt_z  = _mm256_load_pd( &zl[jj][2*ll] );
		v_dt_z  = _mm256_addsub_pd( v_dt_z, v_temp );
		v_temp  = _mm256_load_pd( &Zl[jj][2*ll] );
		v_dt_z  = _mm256_mul_pd( v_dt_z, v_temp );
		v_db    = _mm256_load_pd( &db[jj][2*ll] );
		v_dt    = _mm256_addsub_pd( v_db, v_dux );
		v_dt    = _mm256_xor_pd( v_dt, v_sign );
		v_dt    = _mm256_add_pd( v_dt, v_dt_z );
		v_t     = _mm256_load_pd( &t[jj][2*ll] );
		v_dt    = _mm256_sub_pd( v_dt, v_t );
		_mm256_store_pd( &dt[jj][2*ll], v_dt );
		v_t_z   = _mm256_load_pd( &t[jj][anb+2*ll] );
		v_dt_z  = _mm256_sub_pd( v_dt_z, v_t_z );
		_mm256_store_pd( &dt[jj][anb+2*ll], v_dt_z );

		//v_lamt   = _mm256_load_pd( &lamt[jj][2*ll] );
		v_lamt_z = _mm256_load_pd( &lamt[jj][anb+2*ll] );
		v_tmp0   = _mm256_mul_pd( v_lamt, v_dt );
		v_tmp1   = _mm256_mul_pd( v_lamt_z, v_dt_z );
		v_dlam   = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dlam_z = _mm256_load_pd( &dlam[jj][anb+2*ll] );
		v_lam    = _mm256_load_pd( &lam[jj][2*ll] );
		v_lam_z  = _mm256_load_pd( &lam[jj][anb+2*ll] );
		v_dlam   = _mm256_sub_pd( v_dlam, v_lam );
		v_dlam_z = _mm256_sub_pd( v_dlam_z, v_lam_z );
		v_dlam   = _mm256_sub_pd( v_dlam, v_tmp0 );
		v_dlam_z = _mm256_sub_pd( v_dlam_z, v_tmp1 );
		_mm256_store_pd( &dlam[jj][2*ll], v_dlam );
		_mm256_store_pd( &dlam[jj][anb+2*ll], v_dlam_z );

		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt ) ), 0x20 );
		t_dlam_z = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dt_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		t_mask1  = _mm256_cmp_ps( t_dlam_z, t_zeros, 0x01 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t ) ), 0x20 );
		t_lam_z  = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_t_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_lam_z  = _mm256_xor_ps( t_lam_z, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp1   = _mm256_div_ps( t_lam_z, t_dlam_z );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_tmp1   = _mm256_blendv_ps( t_ones, t_tmp1, t_mask1 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );
		t_alpha1 = _mm256_min_ps( t_alpha1, t_tmp1 );

		}

	for(; ll<nb; ll++)
		{

		u_dux   = _mm_loaddup_pd( &dux[jj][ll+0] );
		u_lamt  = _mm_load_pd( &lamt[jj][2*ll] );
		u_temp  = _mm_mul_pd( u_lamt, u_dux );
		u_dt_z  = _mm_load_pd( &zl[jj][2*ll] );
		u_dt_z  = _mm_addsub_pd( u_dt_z, u_temp );
		u_temp  = _mm_load_pd( &Zl[jj][2*ll] );
		u_dt_z  = _mm_mul_pd( u_dt_z, u_temp );
		u_db    = _mm_load_pd( &db[jj][2*ll] );
		u_dt    = _mm_addsub_pd( u_db, u_dux );
		u_dt    = _mm_xor_pd( u_dt, u_sign );
		u_dt    = _mm_add_pd( u_dt, u_dt_z );
		u_t     = _mm_load_pd( &t[jj][2*ll] );
		u_dt    = _mm_sub_pd( u_dt, u_t );
		_mm_store_pd( &dt[jj][2*ll], u_dt );
		u_t_z   = _mm_load_pd( &t[jj][anb+2*ll] );
		u_dt_z  = _mm_sub_pd( u_dt_z, u_t_z );
		_mm_store_pd( &dt[jj][anb+2*ll], u_dt_z );

		//u_lamt   = _mm_load_pd( &lamt[jj][2*ll] );
		u_lamt_z = _mm_load_pd( &lamt[jj][anb+2*ll] );
		u_tmp0   = _mm_mul_pd( u_lamt, u_dt );
		u_tmp1   = _mm_mul_pd( u_lamt_z, u_dt_z );
		u_dlam   = _mm_load_pd( &dlam[jj][2*ll] );
		u_dlam_z = _mm_load_pd( &dlam[jj][anb+2*ll] );
		u_lam    = _mm_load_pd( &lam[jj][2*ll] );
		u_lam_z  = _mm_load_pd( &lam[jj][anb+2*ll] );
		u_dlam   = _mm_sub_pd( u_dlam, u_lam );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_lam_z );
		u_dlam   = _mm_sub_pd( u_dlam, u_tmp0 );
		u_dlam_z = _mm_sub_pd( u_dlam_z, u_tmp1 );
		_mm_store_pd( &dlam[jj][2*ll], u_dlam );
		_mm_store_pd( &dlam[jj][anb+2*ll], u_dlam_z );

		v_dlam   = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam ), _mm256_castpd128_pd256( u_dt ), 0x20 );
		v_dlam_z = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_dlam_z ), _mm256_castpd128_pd256( u_dt_z ), 0x20 );
		t_dlam   = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_dlam_z ) ), 0x20 );
		t_mask0  = _mm256_cmp_ps( t_dlam, t_zeros, 0x01 );
		v_lam    = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam ), _mm256_castpd128_pd256( u_t ), 0x20 );
		v_lam_z  = _mm256_permute2f128_pd( _mm256_castpd128_pd256( u_lam_z ), _mm256_castpd128_pd256( u_t_z ), 0x20 );
		t_lam    = _mm256_permute2f128_ps( _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam ) ), _mm256_castps128_ps256( _mm256_cvtpd_ps( v_lam_z ) ), 0x20 );
		t_lam    = _mm256_xor_ps( t_lam, t_sign );
		t_tmp0   = _mm256_div_ps( t_lam, t_dlam );
		t_tmp0   = _mm256_blendv_ps( t_ones, t_tmp0, t_mask0 );
		t_alpha0 = _mm256_min_ps( t_alpha0, t_tmp0 );

		}
	
	// reduce alpha
	t_alpha0 = _mm256_min_ps( t_alpha0, t_alpha1 );
	s_alpha1 = _mm256_extractf128_ps( t_alpha0, 0x1 );
	s_alpha0  = _mm_min_ps( s_alpha0 , s_alpha1 );
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

	int jj, ll, ll_bkp, ll_end;
	double ll_left;

	double d_mask[4] = {0.5, 1.5, 2.5, 3.5};
	
	__m128d
		u_ux, u_dux, u_pi, u_dpi, u_t, u_dt, u_lam, u_dlam, u_mu, u_tmp;

	__m256d
		v_mask, v_left,
		v_t0, v_dt0, v_lam0, v_dlam0, v_t1, v_dt1, v_lam1, v_dlam1, 
		v_alpha, v_ux, v_dux, v_pi, v_dpi, v_mu;
	
	__m256i
		i_mask;
		
	v_alpha = _mm256_set_pd( alpha, alpha, alpha, alpha );
	
	v_mu = _mm256_setzero_pd();
	u_mu = _mm_setzero_pd();



	// first stage
	jj = 0;
	
	ll = 0;
	// update inputs
	for(; ll<nu-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu)
		{
		ll_left = nu-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
		}
#if 0
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
#endif
	// box constraints
	ll = 0;
	for(; ll<nbu-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu   = _mm256_add_pd( v_mu, v_lam0 );
		}
	if(ll<nbu)
		{
		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// middle stages
	for(jj=1; jj<N; jj++)
		{
		// update equality constrained multipliers
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
		if(ll<nx)
			{
			ll_left = nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
			v_pi  = _mm256_load_pd( &pi[jj][ll] );
			v_dpi = _mm256_load_pd( &dpi[jj][ll] );
			v_dpi = _mm256_sub_pd( v_dpi, v_pi );
			v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
			v_pi  = _mm256_add_pd( v_pi, v_dpi );
			_mm256_maskstore_pd( &pi[jj][ll], i_mask, v_pi );
			}
		// update inputs & states
		// box constraints
		ll = 0;
		for(; ll<nb-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );

			v_t0    = _mm256_load_pd( &t[jj][2*ll+0] );
			v_t1    = _mm256_load_pd( &t[jj][2*ll+4] );
			v_lam0  = _mm256_load_pd( &lam[jj][2*ll+0] );
			v_lam1  = _mm256_load_pd( &lam[jj][2*ll+4] );
			v_dt0   = _mm256_load_pd( &dt[jj][2*ll+0] );
			v_dt1   = _mm256_load_pd( &dt[jj][2*ll+4] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll+0] );
			v_dlam1 = _mm256_load_pd( &dlam[jj][2*ll+4] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_t1    = _mm256_add_pd( v_t1, v_dt1 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
			_mm256_store_pd( &t[jj][2*ll+0], v_t0 );
			_mm256_store_pd( &t[jj][2*ll+4], v_t1 );
			_mm256_store_pd( &lam[jj][2*ll+0], v_lam0 );
			_mm256_store_pd( &lam[jj][2*ll+4], v_lam1 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
			v_mu   = _mm256_add_pd( v_mu, v_lam0 );
			v_mu   = _mm256_add_pd( v_mu, v_lam1 );
			}
		// backup ll
		ll_bkp = ll;
		// clean up inputs & states
		for(; ll<nu+nx-3; ll+=4)
			{
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_store_pd( &ux[jj][ll], v_ux );
			}
		if(ll<nu+nx)
			{
			ll_left = nu+nx-ll;
			v_left= _mm256_broadcast_sd( &ll_left );
			v_mask= _mm256_loadu_pd( d_mask );
			i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
			v_ux  = _mm256_load_pd( &ux[jj][ll] );
			v_dux = _mm256_load_pd( &dux[jj][ll] );
			v_dux = _mm256_sub_pd( v_dux, v_ux );
			v_dux = _mm256_mul_pd( v_alpha, v_dux );
			v_ux  = _mm256_add_pd( v_ux, v_dux );
			_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
			}
		// cleanup box constraints
		ll = ll_bkp;
		for(; ll<nb-1; ll+=2)
			{
			v_t0    = _mm256_load_pd( &t[jj][2*ll] );
			v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
			v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			_mm256_store_pd( &t[jj][2*ll], v_t0 );
			_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu   = _mm256_add_pd( v_mu, v_lam0 );
			}
		if(ll<nb)
			{
			u_t    = _mm_load_pd( &t[jj][2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][2*ll], u_t );
			_mm_store_pd( &lam[jj][2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}
		// soft constraints on states
		ll = nbu;
		if(nbu%2==1)
			{
			u_t    = _mm_load_pd( &t[jj][anb+2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][anb+2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][anb+2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][anb+2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][anb+2*ll], u_t );
			_mm_store_pd( &lam[jj][anb+2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );

			ll++;
			}
		for(; ll<nb-1; ll+=2)
			{
			v_t0    = _mm256_load_pd( &t[jj][anb+2*ll] );
			v_lam0  = _mm256_load_pd( &lam[jj][anb+2*ll] );
			v_dt0   = _mm256_load_pd( &dt[jj][anb+2*ll] );
			v_dlam0 = _mm256_load_pd( &dlam[jj][anb+2*ll] );
			v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
			v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
			v_t0    = _mm256_add_pd( v_t0, v_dt0 );
			v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
			_mm256_store_pd( &t[jj][anb+2*ll], v_t0 );
			_mm256_store_pd( &lam[jj][anb+2*ll], v_lam0 );
			v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
			v_mu   = _mm256_add_pd( v_mu, v_lam0 );
			}
		for(; ll<nb; ll++)
			{
			u_t    = _mm_load_pd( &t[jj][anb+2*ll] );
			u_lam  = _mm_load_pd( &lam[jj][anb+2*ll] );
			u_dt   = _mm_load_pd( &dt[jj][anb+2*ll] );
			u_dlam = _mm_load_pd( &dlam[jj][anb+2*ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			_mm_store_pd( &t[jj][anb+2*ll], u_t );
			_mm_store_pd( &lam[jj][anb+2*ll], u_lam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}
		}

	// last stage
	jj = N;
	// update equality constrained multipliers
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
	if(ll<nx)
		{
		ll_left = nx-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_pi  = _mm256_load_pd( &pi[jj][ll] );
		v_dpi = _mm256_load_pd( &dpi[jj][ll] );
		v_dpi = _mm256_sub_pd( v_dpi, v_pi );
		v_dpi = _mm256_mul_pd( v_alpha, v_dpi );
		v_pi  = _mm256_add_pd( v_pi, v_dpi );
		_mm256_maskstore_pd( &pi[jj][ll], i_mask, v_pi );
		}
	// cleanup at the beginning
	ll = nu;
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nb<ll_end)
		ll_end = nb;
	for(; ll<ll_end; ll++)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );

		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
	ll_end = ((nu+bs-1)/bs)*bs;
	if(nx+nu<ll_end)
		ll_end = nx+nu;
	for(; ll<ll_end; ll++)
		{
		u_ux  = _mm_load_sd( &ux[jj][ll] );
		u_dux = _mm_load_sd( &dux[jj][ll] );
		u_dux = _mm_sub_sd( u_dux, u_ux );
		u_dux = _mm_mul_sd( _mm256_castpd256_pd128( v_alpha ), u_dux );
		u_ux  = _mm_add_sd( u_ux, u_dux );
		_mm_store_sd( &ux[jj][ll], u_ux );
		}
	// update inputs & states
	// box constraints
	for(; ll<nb-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );

		v_t0    = _mm256_load_pd( &t[jj][2*ll+0] );
		v_t1    = _mm256_load_pd( &t[jj][2*ll+4] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll+0] );
		v_lam1  = _mm256_load_pd( &lam[jj][2*ll+4] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll+0] );
		v_dt1   = _mm256_load_pd( &dt[jj][2*ll+4] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll+0] );
		v_dlam1 = _mm256_load_pd( &dlam[jj][2*ll+4] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dt1   = _mm256_mul_pd( v_alpha, v_dt1 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_dlam1 = _mm256_mul_pd( v_alpha, v_dlam1 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_t1    = _mm256_add_pd( v_t1, v_dt1 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		v_lam1  = _mm256_add_pd( v_lam1, v_dlam1 );
		_mm256_store_pd( &t[jj][2*ll+0], v_t0 );
		_mm256_store_pd( &t[jj][2*ll+4], v_t1 );
		_mm256_store_pd( &lam[jj][2*ll+0], v_lam0 );
		_mm256_store_pd( &lam[jj][2*ll+4], v_lam1 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_lam1  = _mm256_mul_pd( v_lam1, v_t1 );
		v_mu   = _mm256_add_pd( v_mu, v_lam0 );
		v_mu   = _mm256_add_pd( v_mu, v_lam1 );
		}
	// backup ll
	ll_bkp = ll;
	// clean up inputs & states
	for(; ll<nu+nx-3; ll+=4)
		{
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_store_pd( &ux[jj][ll], v_ux );
		}
	if(ll<nu+nx)
		{
		ll_left = nu+nx-ll;
		v_left= _mm256_broadcast_sd( &ll_left );
		v_mask= _mm256_loadu_pd( d_mask );
		i_mask= _mm256_castpd_si256( _mm256_sub_pd( v_mask, v_left ) );
		v_ux  = _mm256_load_pd( &ux[jj][ll] );
		v_dux = _mm256_load_pd( &dux[jj][ll] );
		v_dux = _mm256_sub_pd( v_dux, v_ux );
		v_dux = _mm256_mul_pd( v_alpha, v_dux );
		v_ux  = _mm256_add_pd( v_ux, v_dux );
		_mm256_maskstore_pd( &ux[jj][ll], i_mask, v_ux );
		}
	// cleanup box constraints
	ll = ll_bkp;
	for(; ll<nb-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu   = _mm256_add_pd( v_mu, v_lam0 );
		}
	if(ll<nb)
		{
		u_t    = _mm_load_pd( &t[jj][2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][2*ll], u_t );
		_mm_store_pd( &lam[jj][2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}
	// soft constraints on states
	ll = nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][anb+2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][anb+2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][anb+2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][anb+2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][anb+2*ll], u_t );
		_mm_store_pd( &lam[jj][anb+2*ll], u_lam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );

		ll++;
		}
	for(; ll<nb-1; ll+=2)
		{
		v_t0    = _mm256_load_pd( &t[jj][anb+2*ll] );
		v_lam0  = _mm256_load_pd( &lam[jj][anb+2*ll] );
		v_dt0   = _mm256_load_pd( &dt[jj][anb+2*ll] );
		v_dlam0 = _mm256_load_pd( &dlam[jj][anb+2*ll] );
		v_dt0   = _mm256_mul_pd( v_alpha, v_dt0 );
		v_dlam0 = _mm256_mul_pd( v_alpha, v_dlam0 );
		v_t0    = _mm256_add_pd( v_t0, v_dt0 );
		v_lam0  = _mm256_add_pd( v_lam0, v_dlam0 );
		_mm256_store_pd( &t[jj][anb+2*ll], v_t0 );
		_mm256_store_pd( &lam[jj][anb+2*ll], v_lam0 );
		v_lam0  = _mm256_mul_pd( v_lam0, v_t0 );
		v_mu   = _mm256_add_pd( v_mu, v_lam0 );
		}
	for(; ll<nb; ll++)
		{
		u_t    = _mm_load_pd( &t[jj][anb+2*ll] );
		u_lam  = _mm_load_pd( &lam[jj][anb+2*ll] );
		u_dt   = _mm_load_pd( &dt[jj][anb+2*ll] );
		u_dlam = _mm_load_pd( &dlam[jj][anb+2*ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		_mm_store_pd( &t[jj][anb+2*ll], u_t );
		_mm_store_pd( &lam[jj][anb+2*ll], u_lam );
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
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// middle stages
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
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		// soft constraints
		ll = 2*nu;
		if(nu%2==1)
			{
			u_t    = _mm_load_pd( &t[jj][anb+ll] );
			u_lam  = _mm_load_pd( &lam[jj][anb+ll] );
			u_dt   = _mm_load_pd( &dt[jj][anb+ll] );
			u_dlam = _mm_load_pd( &dlam[jj][anb+ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			ll += 2;
			}
		for(; ll<2*nb-3; ll+=4)
			{
			v_t    = _mm256_load_pd( &t[jj][anb+ll] );
			v_lam  = _mm256_load_pd( &lam[jj][anb+ll] );
			v_dt   = _mm256_load_pd( &dt[jj][anb+ll] );
			v_dlam = _mm256_load_pd( &dlam[jj][anb+ll] );
			v_dt   = _mm256_mul_pd( v_alpha, v_dt );
			v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
			v_t    = _mm256_add_pd( v_t, v_dt );
			v_lam  = _mm256_add_pd( v_lam, v_dlam );
			v_lam  = _mm256_mul_pd( v_lam, v_t );
			v_mu   = _mm256_add_pd( v_mu, v_lam );
			}
		if(ll<2*nb-1)
			{
			u_t    = _mm_load_pd( &t[jj][anb+ll] );
			u_lam  = _mm_load_pd( &lam[jj][anb+ll] );
			u_dt   = _mm_load_pd( &dt[jj][anb+ll] );
			u_dlam = _mm_load_pd( &dlam[jj][anb+ll] );
			u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
			u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
			u_t    = _mm_add_pd( u_t, u_dt );
			u_lam  = _mm_add_pd( u_lam, u_dlam );
			u_lam  = _mm_mul_pd( u_lam, u_t );
			u_mu   = _mm_add_pd( u_mu, u_lam );
			}

		}	

	// last stage
	jj = N;
	
	// hard constraints
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
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		}

	// soft constraints
	ll = 2*nu;
	if(nu%2==1)
		{
		u_t    = _mm_load_pd( &t[jj][anb+ll] );
		u_lam  = _mm_load_pd( &lam[jj][anb+ll] );
		u_dt   = _mm_load_pd( &dt[jj][anb+ll] );
		u_dlam = _mm_load_pd( &dlam[jj][anb+ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
		u_lam  = _mm_mul_pd( u_lam, u_t );
		u_mu   = _mm_add_pd( u_mu, u_lam );
		ll += 2;
		}
	for(; ll<2*nb-3; ll+=4)
		{
		v_t    = _mm256_load_pd( &t[jj][anb+ll] );
		v_lam  = _mm256_load_pd( &lam[jj][anb+ll] );
		v_dt   = _mm256_load_pd( &dt[jj][anb+ll] );
		v_dlam = _mm256_load_pd( &dlam[jj][anb+ll] );
		v_dt   = _mm256_mul_pd( v_alpha, v_dt );
		v_dlam = _mm256_mul_pd( v_alpha, v_dlam );
		v_t    = _mm256_add_pd( v_t, v_dt );
		v_lam  = _mm256_add_pd( v_lam, v_dlam );
		v_lam  = _mm256_mul_pd( v_lam, v_t );
		v_mu   = _mm256_add_pd( v_mu, v_lam );
		}
	if(ll<2*nb-1)
		{
		u_t    = _mm_load_pd( &t[jj][anb+ll] );
		u_lam  = _mm_load_pd( &lam[jj][anb+ll] );
		u_dt   = _mm_load_pd( &dt[jj][anb+ll] );
		u_dlam = _mm_load_pd( &dlam[jj][anb+ll] );
		u_dt   = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dt );
		u_dlam = _mm_mul_pd( _mm256_castpd256_pd128( v_alpha ), u_dlam );
		u_t    = _mm_add_pd( u_t, u_dt );
		u_lam  = _mm_add_pd( u_lam, u_dlam );
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

