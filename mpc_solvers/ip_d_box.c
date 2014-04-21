/**************************************************************************************************
*                                                                                                 *
* This file is part of HPMPC.                                                                     *
*                                                                                                 *
* HPMPC -- Library for High-Performance implementation of solvers for MPC.                        *
* Copyright (C) 2014 by Technical Univeristy of Denmark. All rights reserved.                     *
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

#include "../include/aux_d.h"
#include "../include/aux_s.h"
#include "../include/lqcp_solvers.h"
#include "../include/block_size.h"






/* primal-dual interior-point method, box constraints, time invariant matrices */
void ip_d_box(char prec, double sp_thr, int *kk, int k_max, double tol, double *sigma_par, double *stat, int nx, int nu, int N, int nb, double **pBAbt, float **psBAbt, double **pQ, float **psQ, double **db, double **ux, double *work, int *info)
	{
	
	int nl = 0; // set to zero for the moment
	
	int nbx = nb - 2*nu;
	if(nbx<0)
		nbx = 0;

	// indeces
	int jj, ll, ii, bs0;
	int it_ref;	

	const int dbs = D_MR; //d_get_mr();
	const int sbs = S_MR; //s_get_mr();
	int dsda = dbs*((nx+nu+1+dbs-nu%dbs+dbs-1)/dbs); // second (orizontal) dimension of matrices
	int ssda = sbs*((nx+nu+1+sbs-nu%sbs+sbs-1)/sbs); // second (orizontal) dimension of matrices
	
/*	printf("\n\n%d %d %d %d\n\n", dbs, dsda, sbs, ssda);*/

	// compound quantities
	int nz = nx+nu+1;
	int nxu = nx+nu;
	
	// initialize work space
	double *ptr;
	ptr = work;

	double *(dux[N+1]);
	double *(pL[N+1]);
	double *(pl[N+1]);
	double *pBAbtL;
	double *pLt;

	// inputs and states
	for(jj=0; jj<=N; jj++)
		{
		dux[jj] = ptr+jj*dsda;
		}
	ptr += (N+1)*dsda;

	// cost function
	for(jj=0; jj<=N; jj++)
		{
		pL[jj] = ptr+jj*dsda*dsda;
		pl[jj] = pL[jj] + nxu%dbs + (nxu/dbs)*dbs*dsda;
		}
	ptr += (N+1)*dsda*dsda;

	// work space
	pBAbtL = ptr;
	ptr += dsda*dsda;

	pLt = ptr;
	ptr += dsda*dsda;
		


	float *(sdux[N+1]);
	float *(psL[N+1]);
	float *(psl[N+1]);
	float *psBAbtL;
	float *psLt;

	float *sptr = (float *) ptr;

	// inputs and states, single prec
	for(jj=0; jj<=N; jj++)
		{
		sdux[jj] = sptr+jj*ssda;
		}
	sptr += (N+1)*ssda;


	// cost function
	for(jj=0; jj<=N; jj++)
		{
		psL[jj] = sptr+jj*ssda*ssda;
		psl[jj] = psL[jj] + nxu%sbs + (nxu/sbs)*sbs*ssda;
		}
	sptr += (N+1)*ssda*ssda;

	// work space
	psBAbtL = sptr;
	sptr += ssda*ssda;

	psLt = sptr;
	sptr += ssda*ssda;
	
	ptr = (double *) sptr;



	double *(lam[N+1]);
	double *(dlam[N+1]);
	double *(t[N+1]);
	double *(dt[N+1]);
	double *(lamt[N+1]);

	// slack variables, Lagrangian multipliers for inequality constraints and work space
	for(jj=0; jj<=N; jj++)
		{
		lam[jj]  = ptr + jj*5*nb;
		dlam[jj] = ptr + jj*5*nb + nb;
		t[jj]    = ptr + jj*5*nb + 2*nb;
		dt[jj]   = ptr + jj*5*nb + 3*nb;
		lamt[jj] = ptr + jj*5*nb + 4*nb;
		}
	ptr += (N+1)*5*nb;
	
	double temp0, temp1;
	double alpha, mu;
	double sigma, sigma_decay, sigma_min;

	sigma = sigma_par[0]; //0.4;
	sigma_decay = sigma_par[1]; //0.3;
	sigma_min = sigma_par[2]; //0.01;
	
/*	double d1 = 1;*/
/*	double dm1 = -1;*/
/*	char cl = 'l';*/
/*	char ct = 't';*/
/*	char cn = 'n';*/
/*	int i1 = 1;*/
	
/*	// initialize x0*/
/*	for(ll=0; ll<nx; ll++)*/
/*		ux[0][nu+ll] = x0[ll];*/
/*	// initialize x*/
/*	for(jj=1; jj<=N; jj++)*/
/*		for(ll=0; ll<nx; ll++)*/
/*			ux[jj][nu+ll] = 0;*/

/*	// initialize u*/
/*	for(jj=0; jj<N; jj++)*/
/*		for(ll=0; ll<nu; ll++)*/
/*			ux[jj][ll] = 0;*/



	// initialize t>0 (slack variable)
/*	for(jj=0; jj<=N; jj++)*/
/*		{*/
/*		for(ll=0; ll<nb; ll++)*/
/*			t[jj][ll] = 1;*/
/*		}*/
/*	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!*/
/*		t[N][ll] = 1;*/
/*	for(ll=2*nu; ll<nb; ll++)*/
/*		t[N][ll] = 1;*/

	double thr0 = 1e-3;
	for(jj=0; jj<N; jj++)
		{
		for(ll=0; ll<nb; ll+=2)
			{
			t[jj][ll+0] =   ux[jj][ll/2] - db[jj][ll+0];
			if(t[jj][ll+0] < thr0)
				{
				t[jj][ll+0] = thr0;
				ux[jj][ll/2] = thr0 + db[jj][ll+0];
				}

			t[jj][ll+1] = - ux[jj][ll/2] - db[jj][ll+1];
			if(t[jj][ll+1] < thr0)
				{
				t[jj][ll+1] = thr0;
				ux[jj][ll/2] = thr0 - db[jj][ll+0];
				}
			}
		}
	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!
		t[N][ll] = 1;
	for(ll=2*nu; ll<nb; ll+=2)
		{
		t[N][ll+0] =   ux[N][ll/2] - db[N][ll+0];
		if(t[N][ll+0] < thr0)
			{
			t[N][ll+0] = thr0;
			ux[N][ll/2] = thr0 + db[N][ll+0];
			}

		t[N][ll+1] = - ux[N][ll/2] - db[N][ll+1];
		if(t[N][ll+1] < thr0)
			{
			t[N][ll+1] = thr0;
			ux[N][ll/2] = thr0 - db[N][ll+0];
			}
		}



	// initialize lambda>0 (multiplier of the inequality constr)
/*	for(jj=0; jj<N; jj++)*/
/*		{*/
/*		for(ll=0; ll<nb; ll++)*/
/*			lam[jj][ll] = 1;*/
/*		}*/
/*	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!*/
/*		lam[N][ll] = 1;*/
/*	for(ll=2*nu; ll<nb; ll++)*/
/*		lam[N][ll] = 1;*/
	
	for(jj=0; jj<N; jj++)
		{
		for(ll=0; ll<nb; ll++)
			lam[jj][ll] = 1/t[jj][ll];
/*			lam[jj][ll] = thr0/t[jj][ll];*/
		}
	for(ll=0; ll<2*nu; ll++) // this has to be strictly positive !!!
		lam[N][ll] = 1;
	for(ll=2*nu; ll<nb; ll++)
		lam[N][ll] = 1/t[jj][ll];
/*		lam[N][ll] = thr0/t[jj][ll];*/



	// initialize dux
	// double precision
	for(ll=0; ll<nx; ll++)
		dux[0][nu+ll] = ux[0][nu+ll];
	// single precision
	for(ll=0; ll<nx; ll++)
		sdux[0][nu+ll] = ux[0][nu+ll];



	// compute the duality gap
	mu = 0;
	for(jj=0; jj<N; jj++)
		{
		for(ll=0 ; ll<nb; ll+=2)
			mu += lam[jj][ll+0] * t[jj][ll+0] + lam[jj][ll+1] * t[jj][ll+1];
		}
	for(ll=2*nu ; ll<nb; ll+=2)
		mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
	mu /= N*nb + nbx;
	
	*kk = 0;	
	


	// IP loop		
	while( *kk<k_max && mu>tol )
		{
						


		//update cost function matrices and vectors (box constraints)

		if(prec=='d' && mu<sp_thr)
			{

			for(jj=0; jj<=N; jj++)
				{

				// copy Q in L
				d_copy_pmat_lo(nz, dbs, pQ[jj], dsda, pL[jj], dsda);
/*				d_copy_pmat(nz, nz, dbs, pQ[jj], dsda, pL[jj], dsda);*/
			
/*printf("\n%d\n", jj);*/
/*d_print_pmat(nz, nz, dbs, pQ[jj], dsda);*/
/*d_print_pmat(nz, nz, dbs, pL[jj], dsda);*/

				// box constraints
				for(ii=0; ii<nb; ii+=2*dbs)
					{
					bs0 = nb-ii;
					if(2*dbs<bs0) bs0 = 2*dbs;
					for(ll=0; ll<bs0; ll+=2)
						{
						temp0 = 1.0/t[jj][ii+ll+0];
						temp1 = 1.0/t[jj][ii+ll+1];
						lamt[jj][ii+ll+0] = lam[jj][ii+ll+0]*temp0;
						lamt[jj][ii+ll+1] = lam[jj][ii+ll+1]*temp1;
						dlam[jj][ii+ll+0] = temp0*(sigma*mu); // !!!!!
						dlam[jj][ii+ll+1] = temp1*(sigma*mu); // !!!!!
						pL[jj][ll/2+(ii+ll)/2*dbs+ii/2*dsda] += lamt[jj][ii+ll+0] + lamt[jj][ii+ll+1];
						pl[jj][(ii+ll)/2*dbs] += lam[jj][ii+ll+1] + lamt[jj][ii+ll+1]*db[jj][ii+ll+1] + dlam[jj][ii+ll+1] 
						                       - lam[jj][ii+ll+0] - lamt[jj][ii+ll+0]*db[jj][ii+ll+0] - dlam[jj][ii+ll+0];
						}
					}
/*d_print_pmat(nz, nz, dbs, pL[jj], dsda);*/
				}

			}
		else
			{

			for(jj=0; jj<=N; jj++)
				{

				// convert Q in sL
				s_copy_pmat_lo(nz, sbs, psQ[jj], ssda, psL[jj], ssda);

				// box constraints
				for(ii=0; ii<nb; ii+=2*sbs)
					{
					bs0 = nb-ii;
					if(2*sbs<bs0) bs0 = 2*sbs;
					for(ll=0; ll<bs0; ll+=2)
						{
						temp0 = 1.0/t[jj][ii+ll+0];
						temp1 = 1.0/t[jj][ii+ll+1];
						lamt[jj][ii+ll+0] = lam[jj][ii+ll+0]*temp0;
						lamt[jj][ii+ll+1] = lam[jj][ii+ll+1]*temp1;
						dlam[jj][ii+ll+0] = temp0*(sigma*mu); // !!!!!
						dlam[jj][ii+ll+1] = temp1*(sigma*mu); // !!!!!
						psL[jj][ll/2+(ii+ll)/2*sbs+ii/2*ssda] += (float) lamt[jj][ii+ll+0] + lamt[jj][ii+ll+1];
						psl[jj][(ii+ll)/2*sbs] += (float) lam[jj][ii+ll+1] + lamt[jj][ii+ll+1]*db[jj][ii+ll+1] + dlam[jj][ii+ll+1] 
						                       - lam[jj][ii+ll+0] - lamt[jj][ii+ll+0]*db[jj][ii+ll+0] - dlam[jj][ii+ll+0];
						}
					}
				}

			}


		if(prec=='d' && mu<sp_thr)
			{
			// compute the search direction: factorize and solve the KKT system
			dricposv_mpc(nx, nu, N, dsda, pBAbt, pL, dux, pLt, pBAbtL, info);
			if(*info!=0) return;
			}
		else
			{
			// compute the search direction: factorize and solve the KKT system
			sricposv_mpc(nx, nu, N, ssda, psBAbt, psL, sdux, psLt, psBAbtL, info);
			if(*info!=0) return;

			// solution in double precision
			for(ll=0; ll<nu; ll++)
				dux[0][ll] = (double) sdux[0][ll];
			for(jj=1; jj<=N; jj++)
				for(ll=0; ll<nxu; ll++)
					dux[jj][ll] = (double) sdux[jj][ll];
			}

/*d_print_mat(nx+nu, N, dux[0], dsda);*/
/*exit(2);*/



		// compute t_aff & dlam_aff & dt_aff & alpha
		alpha = 1;
		temp0 = 2;
		temp1 = 2;
		for(jj=0; jj<N; jj++)
			{
			for(ll=0; ll<nb; ll+=2)
				{
				dt[jj][ll+0] =   dux[jj][ll/2] - db[jj][ll+0];
				dt[jj][ll+1] = - dux[jj][ll/2] - db[jj][ll+1];
				dlam[jj][ll+0] -= lamt[jj][ll+0] * dt[jj][ll+0];
				dlam[jj][ll+1] -= lamt[jj][ll+1] * dt[jj][ll+1];
				if(dlam[jj][ll+0]<0)
					temp0 = - lam[jj][ll+0] / dlam[jj][ll+0];
				if(dlam[jj][ll+1]<0)
					temp1 = - lam[jj][ll+1] / dlam[jj][ll+1];
				if(temp0<alpha)
					alpha = temp0;
				if(temp1<alpha)
					alpha = temp1;
				dt[jj][ll+0] -= t[jj][ll+0];
				dt[jj][ll+1] -= t[jj][ll+1];
				if(dt[jj][ll+0]<0)
					temp0 = - t[jj][ll+0] / dt[jj][ll+0];
				if(dt[jj][ll+1]<0)
					temp1 = - t[jj][ll+1] / dt[jj][ll+1];
				if(temp0<alpha)
					alpha = temp0;
				if(temp1<alpha)
					alpha = temp1;
				}
			}
		for(ll=2*nu; ll<nb; ll+=2)
			{
			dt[N][ll+0] =   dux[N][ll/2] - db[N][ll+0];
			dt[N][ll+1] = - dux[N][ll/2] - db[N][ll+1];
			dlam[N][ll+0] -= lamt[N][ll+0] * dt[N][ll+0];
			dlam[N][ll+1] -= lamt[N][ll+1] * dt[N][ll+1];
			if(dlam[N][ll+0]<0)
				temp0 = - lam[N][ll+0] / dlam[N][ll+0];
			if(dlam[N][ll+1]<0)
				temp1 = - lam[N][ll+1] / dlam[N][ll+1];
			if(temp0<alpha)
				alpha = temp0;
			if(temp1<alpha)
				alpha = temp1;
			dt[N][ll+0] -= t[N][ll+0];
			dt[N][ll+1] -= t[N][ll+1];
			if(dt[N][ll+0]<0)
				temp0 = - t[N][ll+0] / dt[N][ll+0];
			if(dt[N][ll+1]<0)
				temp1 = - t[N][ll+1] / dt[N][ll+1];
			if(temp0<alpha)
				alpha = temp0;
			if(temp1<alpha)
				alpha = temp1;
			}
		stat[5*(*kk)] = sigma;
		stat[5*(*kk)+1] = alpha;
			
		alpha *= 0.995;




		// update x, u, lam, t & compute the duality gap mu
		mu = 0;
		for(jj=0; jj<N; jj++)
			{
			// update inputs
			for(ll=0; ll<nu; ll++)
				ux[jj][ll] += alpha*(dux[jj][ll] - ux[jj][ll]);
			// update states
			for(ll=0; ll<nx; ll++)
				ux[jj][nu+ll] += alpha*(dux[jj][nu+ll] - ux[jj][nu+ll]);
			// box constraints
			for(ll=0; ll<nb; ll+=2)
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
		// box constraints
		for(ll=2*nu; ll<nb; ll+=2)
			{
			lam[N][ll+0] += alpha*dlam[N][ll+0];
			lam[N][ll+1] += alpha*dlam[N][ll+1];
			t[N][ll+0] += alpha*dt[N][ll+0];
			t[N][ll+1] += alpha*dt[N][ll+1];
			mu += lam[N][ll+0] * t[N][ll+0] + lam[N][ll+1] * t[N][ll+1];
			}
		mu /= N*nb + nbx;

		stat[5*(*kk)+2] = mu;
		




		// update sigma
		sigma *= sigma_decay;
		if(sigma<sigma_min)
			sigma = sigma_min;
		
		if(alpha<0.3)
			sigma = sigma_par[0];



		// increment loop index
		(*kk)++;



		} // end of IP loop
	


	return;

	} // end of ipsolver


