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



void d_update_hessian_gradient_res_mpc_hard_tv(int N, int *nx, int *nu, int *nb, int *ng, double **res_d, double res_m, double **t, double **lam, double **bd, double **bl, double **t_inv, double **pd, double **pl, double **Qx, double **qx, double **qx2)
	{
	
	// constants
	const int bs = D_MR;
	const int ncl = D_NCL;

	int nb0, pnb, ng0, png;
	
	double temp0, temp1;
	
	double 
		*ptr_pd, *ptr_pl, *ptr_bd, *ptr_bl, *ptr_db, *ptr_Qx, *ptr_qx, *ptr_qx2,
		*ptr_t, *ptr_lam, *ptr_lamt, *ptr_dlam, *ptr_tinv;
	
	int ii, jj, bs0;
	
	for(jj=0; jj<=N; jj++)
		{
		
		ptr_t     = t[jj];
		ptr_lam   = lam[jj];
		ptr_t_inv = t_inv[jj];
		ptr_res_d = res_d[jj];
		ptr_res_m = res_m[jj];
		ptr_bd    = bd[jj];
		ptr_bl    = bl[jj];
		ptr_pd    = pd[jj];
		ptr_pl    = pl[jj];

		// box constraints
		nb0 = nb[jj];
		if(nb0>0)
			{

			pnb  = (nb0+bs-1)/bs*bs; // simd aligned number of box constraints

			for(ii=0; ii<nb0-3; ii+=4)
				{

				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
				ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+pnb+0]*ptr_lam[ii+pnb+0];
				ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]-ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]) - ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]);

				ptr_t_inv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_t_inv[ii+pnb+1] = 1.0/ptr_t[ii+pnb+1];
				ptr_pd[ii+1] = ptr_bd[ii+1] + ptr_t_inv[ii+1]*ptr_lam[ii+1] + ptr_t_inv[ii+pnb+1]*ptr_lam[ii+pnb+1];
				ptr_pl[ii+1] = ptr_bl[ii+1] + ptr_t_inv[ii+pnb+1]*(ptr_res_m[ii+pnb+1]-ptr_lam[ii+pnb+1]*ptr_res_d[ii+pnb+1]) - ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]);

				ptr_t_inv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_t_inv[ii+pnb+2] = 1.0/ptr_t[ii+pnb+2];
				ptr_pd[ii+2] = ptr_bd[ii+2] + ptr_t_inv[ii+2]*ptr_lam[ii+2] + ptr_t_inv[ii+pnb+2]*ptr_lam[ii+pnb+2];
				ptr_pl[ii+2] = ptr_bl[ii+2] + ptr_t_inv[ii+pnb+2]*(ptr_res_m[ii+pnb+2]-ptr_lam[ii+pnb+2]*ptr_res_d[ii+pnb+2]) - ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]);

				ptr_t_inv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_t_inv[ii+pnb+3] = 1.0/ptr_t[ii+pnb+3];
				ptr_pd[ii+3] = ptr_bd[ii+3] + ptr_t_inv[ii+3]*ptr_lam[ii+3] + ptr_t_inv[ii+pnb+3]*ptr_lam[ii+pnb+3];
				ptr_pl[ii+3] = ptr_bl[ii+3] + ptr_t_inv[ii+pnb+3]*(ptr_res_m[ii+pnb+3]-ptr_lam[ii+pnb+3]*ptr_res_d[ii+pnb+3]) - ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]);

				}
			for(; ii<nb0; ii++)
				{

				ptr_t_inv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_t_inv[ii+pnb+0] = 1.0/ptr_t[ii+pnb+0];
				ptr_pd[ii+0] = ptr_bd[ii+0] + ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+pnb+0]*ptr_lam[ii+pnb+0];
				ptr_pl[ii+0] = ptr_bl[ii+0] + ptr_t_inv[ii+pnb+0]*(ptr_res_m[ii+pnb+0]-ptr_lam[ii+pnb+0]*ptr_res_d[ii+pnb+0]) - ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]);

				}

			ptr_t     += 2*pnb;
			ptr_lam   += 2*pnb;
			ptr_t_inv += 2*pnb;
			ptr_res_d += 2*pnb;
			ptr_res_m += 2*pnb;

			}

		// general constraints
		ng0 = ng[jj];
		if(ng0>0)
			{

			ptr_Qx    = Qx[jj];
			ptr_qx    = qx[jj];
			ptr_qx2   = qx2[jj];
		
			png = (ng0+bs-1)/bs*bs; // simd aligned number of general constraints

			for(ii=0; ii<ng0-3; ii+=4)
				{

				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_Qx[ii+0] = sqrt(ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+png+0]*ptr_lam[ii+png+0]);
				ptr_qx[ii+0] = ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]-ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]) - ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				ptr_tinv[ii+1] = 1.0/ptr_t[ii+1];
				ptr_tinv[ii+png+1] = 1.0/ptr_t[ii+png+1];
				ptr_Qx[ii+1] = sqrt(ptr_t_inv[ii+1]*ptr_lam[ii+1] + ptr_t_inv[ii+png+1]*ptr_lam[ii+png+1]);
				ptr_qx[ii+1] = ptr_t_inv[ii+png+1]*(ptr_res_m[ii+png+1]-ptr_lam[ii+png+1]*ptr_res_d[ii+png+1]) - ptr_t_inv[ii+1]*(ptr_res_m[ii+1]-ptr_lam[ii+1]*ptr_res_d[ii+1]);
				ptr_qx2[ii+1] = ptr_qx[ii+1] / ptr_Qx[ii+1];

				ptr_tinv[ii+2] = 1.0/ptr_t[ii+2];
				ptr_tinv[ii+png+2] = 1.0/ptr_t[ii+png+2];
				ptr_Qx[ii+2] = sqrt(ptr_t_inv[ii+2]*ptr_lam[ii+2] + ptr_t_inv[ii+png+2]*ptr_lam[ii+png+2]);
				ptr_qx[ii+2] = ptr_t_inv[ii+png+2]*(ptr_res_m[ii+png+2]-ptr_lam[ii+png+2]*ptr_res_d[ii+png+2]) - ptr_t_inv[ii+2]*(ptr_res_m[ii+2]-ptr_lam[ii+2]*ptr_res_d[ii+2]);
				ptr_qx2[ii+2] = ptr_qx[ii+2] / ptr_Qx[ii+2];

				ptr_tinv[ii+3] = 1.0/ptr_t[ii+3];
				ptr_tinv[ii+png+3] = 1.0/ptr_t[ii+png+3];
				ptr_Qx[ii+3] = sqrt(ptr_t_inv[ii+3]*ptr_lam[ii+3] + ptr_t_inv[ii+png+3]*ptr_lam[ii+png+3]);
				ptr_qx[ii+3] = ptr_t_inv[ii+png+3]*(ptr_res_m[ii+png+3]-ptr_lam[ii+png+3]*ptr_res_d[ii+png+3]) - ptr_t_inv[ii+3]*(ptr_res_m[ii+3]-ptr_lam[ii+3]*ptr_res_d[ii+3]);
				ptr_qx2[ii+3] = ptr_qx[ii+3] / ptr_Qx[ii+3];

				}
			for(; ii<ng0; ii++)
				{
				
				ptr_tinv[ii+0] = 1.0/ptr_t[ii+0];
				ptr_tinv[ii+png+0] = 1.0/ptr_t[ii+png+0];
				ptr_Qx[ii+0] = sqrt(ptr_t_inv[ii+0]*ptr_lam[ii+0] + ptr_t_inv[ii+png+0]*ptr_lam[ii+png+0]);
				ptr_qx[ii+0] = ptr_t_inv[ii+png+0]*(ptr_res_m[ii+png+0]-ptr_lam[ii+png+0]*ptr_res_d[ii+png+0]) - ptr_t_inv[ii+0]*(ptr_res_m[ii+0]-ptr_lam[ii+0]*ptr_res_d[ii+0]);
				ptr_qx2[ii+0] = ptr_qx[ii+0] / ptr_Qx[ii+0];

				}

			}

		}

	}




