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


#ifdef BLASFEO

#include <blasfeo_target.h>
#include <blasfeo_common.h>
#include <blasfeo_d_blas.h>
#include <blasfeo_d_aux.h>

#include "../../include/tree.h"


int d_tree_res_res_mpc_hard_work_space_size_bytes_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int *ng)
	{

	int ii;

	int size = 0;

	int ngM = 0;
	for(ii=0; ii<Nn; ii++)
		{
		ngM = ng[ii]>ngM ? ng[ii] : ngM;
		}

	size += 2*d_size_strvec(ngM); // res_work[0], res_work[1]

	// make multiple of (typical) cache line size
	size = (size+63)/64*64;

	return size;

	}



void d_tree_res_res_mpc_hard_libstr(int Nn, struct node *tree, int *nx, int *nu, int *nb, int **idxb, int *ng, struct d_strmat *hsBAbt, struct d_strvec *hsb, struct d_strmat *hsRSQrq, struct d_strvec *hsq, struct d_strvec *hsux, struct d_strmat *hsDCt, struct d_strvec *hsd, struct d_strvec *hspi, struct d_strvec *hslam, struct d_strvec *hst, struct d_strvec *hsres_q, struct d_strvec *hsres_b, struct d_strvec *hsres_d, struct d_strvec *hsres_m, double *mu, void *work)
	{

	int ii, jj, ll;

	char *c_ptr;

	struct d_strvec hswork_0, hswork_1;
	double *work0, *work1;


	double
		*ptr_b, *ptr_q, *ptr_d, *ptr_ux, *ptr_pi, *ptr_lam, *ptr_t, *ptr_rb, *ptr_rq, *ptr_rd, *ptr_rm;
	
	int
		*ptr_idxb;
	
	int nu0, nu1, nx0, nx1, nxm, nb0, ng0, nb_tot;

	int nkids, idxkid;

	double
		mu2;

	// initialize mu
	nb_tot = 0;
	mu2 = 0;



	// loop over nodes
	for(ii=0; ii<Nn; ii++)
		{

		// work on node

		nu0 = nu[ii];
		nx0 = nx[ii];
		nb0 = nb[ii];
		ng0 = ng[ii];

		ptr_q = hsq[ii].pa;
		ptr_ux = hsux[ii].pa;
		ptr_pi = hspi[ii].pa;
		ptr_rq = hsres_q[ii].pa;

		if(nb0>0 | ng0>0)
			{
			ptr_d = hsd[ii].pa;
			ptr_lam = hslam[ii].pa;
			ptr_t = hst[ii].pa;
			ptr_rd = hsres_d[ii].pa;
			ptr_rm = hsres_m[ii].pa;
			}

		for(jj=0; jj<nu0; jj++) 
			ptr_rq[jj] = ptr_q[jj];

		for(jj=0; jj<nx0; jj++) 
			ptr_rq[nu0+jj] = ptr_q[nu0+jj] - ptr_pi[jj];

		dsymv_l_libstr(nu0+nx0, nu0+nx0, 1.0, &hsRSQrq[ii], 0, 0, &hsux[ii], 0, 1.0, &hsres_q[ii], 0, &hsres_q[ii], 0);

		if(nb0>0)
			{

			ptr_idxb = idxb[ii];
			nb_tot += nb0;

			for(jj=0; jj<nb0; jj++) 
				{
				ptr_rq[ptr_idxb[jj]] += - ptr_lam[jj] + ptr_lam[nb0+jj];

				ptr_rd[jj]     = ptr_d[jj]     - ptr_ux[ptr_idxb[jj]] + ptr_t[jj];
				ptr_rd[nb0+jj] = ptr_d[nb0+jj] - ptr_ux[ptr_idxb[jj]] - ptr_t[nb0+jj];

				ptr_rm[jj]     = ptr_lam[jj]     * ptr_t[jj];
				ptr_rm[nb0+jj] = ptr_lam[nb0+jj] * ptr_t[nb0+jj];
				mu2 += ptr_rm[jj] + ptr_rm[nb0+jj];
				}
			}

		if(ng0>0)
			{

			c_ptr = (char *) work;
			d_create_strvec(ng0, &hswork_0, (void *) c_ptr);
			c_ptr += hswork_0.memory_size;
			d_create_strvec(ng0, &hswork_1, (void *) c_ptr);
			c_ptr += hswork_1.memory_size;
			work0 = hswork_0.pa;
			work1 = hswork_1.pa;

			ptr_d   += 2*nb0;
			ptr_lam += 2*nb0;
			ptr_t   += 2*nb0;
			ptr_rd  += 2*nb0;
			ptr_rm  += 2*nb0;

			nb_tot += ng0;

			for(jj=0; jj<ng0; jj++)
				{
				work0[jj] = ptr_lam[jj+ng0] - ptr_lam[jj+0];

				ptr_rd[jj]     = ptr_d[jj]     + ptr_t[jj];
				ptr_rd[ng0+jj] = ptr_d[ng0+jj] - ptr_t[ng0+jj];

				ptr_rm[jj]     = ptr_lam[jj]     * ptr_t[jj];
				ptr_rm[ng0+jj] = ptr_lam[ng0+jj] * ptr_t[ng0+jj];
				mu2 += ptr_rm[jj] + ptr_rm[ng0+jj];
				}

			dgemv_nt_libstr(nu0+nx0, ng0, 1.0, 1.0, &hsDCt[ii], 0, 0, &hswork_0, 0, &hsux[ii], 0, 1.0, 0.0, &hsres_q[ii], 0, &hswork_1, 0, &hsres_q[ii], 0, &hswork_1, 0);

			for(jj=0; jj<ng0; jj++)
				{
				ptr_rd[jj]     -= work1[jj];
				ptr_rd[ng0+jj] -= work1[jj];
				}

			}

		// work on kids

		nkids = tree[ii].nkids;

		for(jj=0; jj<nkids; jj++)
			{

			idxkid = tree[ii].kids[jj];

			nu1 = nu[idxkid];
			nx1 = nx[idxkid];

			ptr_b  = hsb[idxkid-1].pa;
			ptr_rb = hsres_b[idxkid-1].pa;
			ptr_ux = hsux[idxkid].pa;

			for(ll=0; ll<nx1; ll++) 
				ptr_rb[ll] = ptr_b[ll] - ptr_ux[nu1+ll];

			dgemv_nt_libstr(nu0+nx0, nx1, 1.0, 1.0, &hsBAbt[idxkid-1], 0, 0, &hspi[idxkid], 0, &hsux[ii], 0, 1.0, 1.0, &hsres_q[ii], 0, &hsres_b[idxkid-1], 0, &hsres_q[ii], 0, &hsres_b[idxkid-1], 0);

			}

		}
	



	// normalize mu
	if(nb_tot!=0)
		{
		mu2 /= 2.0*nb_tot;
		mu[0] = mu2;
		}



	return;

	}



#endif

