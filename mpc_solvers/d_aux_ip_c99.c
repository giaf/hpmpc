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

#include "../include/block_size.h"



void d_update_hessian_box(int k0, int kmax, int nb, int cnz, double sigma_mu, double *t, double *lam, double *lamt, double *dlam, double *bd, double *bl, double *pd, double *pl, double *lb, double *ub)
	{
	
	const int bs = D_MR; //d_get_mr();
	
	double temp0, temp1;
	
	int ii, ll, bs0;
	
	k0   *= 2;
	kmax *= 2;
	
	for(ii=k0; ii<kmax; ii+=2*bs)
		{
		bs0 = 2*nb-ii;
		if(2*bs<bs0) bs0 = 2*bs;
		for(ll=0; ll<bs0; ll+=2)
			{
			temp0 = 1.0/t[ii+ll+0];
			temp1 = 1.0/t[ii+ll+1];
			lamt[ii+ll+0] = lam[ii+ll+0]*temp0;
			lamt[ii+ll+1] = lam[ii+ll+1]*temp1;
			dlam[ii+ll+0] = temp0*sigma_mu; // !!!!!
			dlam[ii+ll+1] = temp1*sigma_mu; // !!!!!
			pd[ll/2+(ii+ll)/2*bs+ii/2*cnz] = bd[(ii+ll)/2] + lamt[ii+ll+0] + lamt[ii+ll+1];
			pl[(ii+ll)/2*bs] = bl[(ii+ll)/2] + lam[ii+ll+1] - lamt[ii+ll+1]*ub[ii/2+ll/2] + dlam[ii+ll+1] 
			                                 - lam[ii+ll+0] - lamt[ii+ll+0]*lb[ii/2+ll/2] - dlam[ii+ll+0];
			}
		}

	}
