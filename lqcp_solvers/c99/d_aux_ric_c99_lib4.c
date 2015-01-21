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



void d_update_hessian_ric_sv(int kmax, double *pQ, int sda, double *Qd)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pQ[jj*sda+(jj+0)*4+0] = Qd[jj+0];
		pQ[jj*sda+(jj+1)*4+1] = Qd[jj+1];
		pQ[jj*sda+(jj+2)*4+2] = Qd[jj+2];
		pQ[jj*sda+(jj+3)*4+3] = Qd[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pQ[jj*sda+(jj+ll)*4+ll] = Qd[jj+ll];
		}
	
	}



void d_update_jacobian_ric_sv(int kmax, double *pQ, double *Ql)
	{

	int jj, ll;

	for(jj=0; jj<kmax-3; jj+=4)
		{
		pQ[(jj+0)*4] = Ql[jj+0];
		pQ[(jj+1)*4] = Ql[jj+1];
		pQ[(jj+2)*4] = Ql[jj+2];
		pQ[(jj+3)*4] = Ql[jj+3];
		}
	for(ll=0; ll<kmax-jj; ll++)
		{
		pQ[(jj+ll)*4] = Ql[jj+ll];
		}
	
	}


