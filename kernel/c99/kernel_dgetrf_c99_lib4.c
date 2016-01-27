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


#if 1

void corner_dgetrf_nn_4x4_lib4(double *C, double *LU, double *inv_diag_U)
	{

	const int bs = 4;

	double
		// elements of C
		c_00, c_01, c_02, c_03,
		c_10, c_11, c_12, c_13,
		c_20, c_21, c_22, c_23,
		c_30, c_31, c_32, c_33,
		// LU
		u_00, u_01, u_02, u_03,
		l_10, u_11, u_12, u_13,
		l_20, l_21, u_22, u_23,
		l_30, l_31, l_32, u_33,
		// inv diag U
		iu_0, iu_1, iu_2, iu_3;


	// first column
	c_00 = C[0+bs*0];
	c_10 = C[1+bs*0];
	c_20 = C[2+bs*0];
	c_30 = C[3+bs*0];

	u_00 = c_00;
	iu_0 = 1.0 / u_00;
	l_10 = c_10 * iu_0;
	l_20 = c_20 * iu_0;
	l_30 = c_30 * iu_0;


	// second column
	c_01 = C[0+bs*1];
	c_11 = C[1+bs*1];
	c_21 = C[2+bs*1];
	c_31 = C[3+bs*1];

	u_01 = c_01;
	c_11 -= l_10 * u_01;
	c_21 -= l_20 * u_01;
	c_31 -= l_30 * u_01;

	u_11 = c_11;
	iu_1 = 1.0 / u_11;
	l_21 = c_21 * iu_1;
	l_31 = c_31 * iu_1;


	// third column
	c_02 = C[0+bs*2];
	c_12 = C[1+bs*2];
	c_22 = C[2+bs*2];
	c_32 = C[3+bs*2];

	u_02 = c_02;
	c_12 -= l_10 * u_02;
	c_22 -= l_20 * u_02;
	c_32 -= l_30 * u_02;

	u_12 = c_12;
	c_22 -= l_21 * u_12;
	c_32 -= l_31 * u_12;

	u_22 = c_22;
	iu_2 = 1.0 / u_22;
	l_32 = c_32 * iu_2;


	// fourth column
	c_03 = C[0+bs*3];
	c_13 = C[1+bs*3];
	c_23 = C[2+bs*3];
	c_33 = C[3+bs*3];

	u_03 = c_03;
	c_13 -= l_10 * u_03;
	c_23 -= l_20 * u_03;
	c_33 -= l_30 * u_03;

	u_13 = c_13;
	c_23 -= l_21 * u_13;
	c_33 -= l_31 * u_13;

	u_23 = c_23;
	c_33 -= l_32 * u_23;

	u_33 = c_33;
	iu_3 = 1.0 / u_33;

	
	// store LU
	LU[0+bs*0] = u_00;
	LU[1+bs*0] = l_10;
	LU[2+bs*0] = l_20;
	LU[3+bs*0] = l_30;

	LU[0+bs*1] = u_01;
	LU[1+bs*1] = u_11;
	LU[2+bs*1] = l_21;
	LU[3+bs*1] = l_31;

	LU[0+bs*2] = u_02;
	LU[1+bs*2] = u_12;
	LU[2+bs*2] = u_22;
	LU[3+bs*2] = l_32;

	LU[0+bs*3] = u_03;
	LU[1+bs*3] = u_13;
	LU[2+bs*3] = u_23;
	LU[3+bs*3] = u_33;


	
	// return
	return;
	
	}

#else

void corner_dgetrf_nt_4x4_lib4(double *C, double *L, double *U, double *inv_diag_U)
	{

	const int bs = 4;

	double
		// elements of C
		c_00, c_01, c_02, c_03,
		c_10, c_11, c_12, c_13,
		c_20, c_21, c_22, c_23,
		c_30, c_31, c_32, c_33,
		// L
		l_10,
		l_20, l_21,
		l_30, l_31, l_32,
		// U^T
		u_00,
		u_01, u_11,
		u_02, u_12, u_22,
		u_03, u_13, u_23, u_33,
		// inv diag U
		iu_0, iu_1, iu_2, iu_3;


	// first column
	c_00 = C[0+bs*0];
	c_10 = C[1+bs*0];
	c_20 = C[2+bs*0];
	c_30 = C[3+bs*0];

	u_00 = c_00;
	iu_0 = 1.0 / u_00;
	l_10 = c_10 * iu_0;
	l_20 = c_20 * iu_0;
	l_30 = c_30 * iu_0;


	// second column
	c_01 = C[0+bs*1];
	c_11 = C[1+bs*1];
	c_21 = C[2+bs*1];
	c_31 = C[3+bs*1];

	u_01 = c_01;
	c_11 -= l_10 * u_01;
	c_21 -= l_20 * u_01;
	c_31 -= l_30 * u_01;

	u_11 = c_11;
	iu_1 = 1.0 / u_11;
	l_21 = c_21 * iu_1;
	l_31 = c_31 * iu_1;


	// third column
	c_02 = C[0+bs*2];
	c_12 = C[1+bs*2];
	c_22 = C[2+bs*2];
	c_32 = C[3+bs*2];

	u_02 = c_02;
	c_12 -= l_10 * u_02;
	c_22 -= l_20 * u_02;
	c_32 -= l_30 * u_02;

	u_12 = c_12;
	c_22 -= l_21 * u_12;
	c_32 -= l_31 * u_12;

	u_22 = c_22;
	iu_2 = 1.0 / u_22;
	l_32 = c_32 * iu_2;


	// fourth column
	c_03 = C[0+bs*3];
	c_13 = C[1+bs*3];
	c_23 = C[2+bs*3];
	c_33 = C[3+bs*3];

	u_03 = c_03;
	c_13 -= l_10 * u_03;
	c_23 -= l_20 * u_03;
	c_33 -= l_30 * u_03;

	u_13 = c_13;
	c_23 -= l_21 * u_13;
	c_33 -= l_31 * u_13;

	u_23 = c_23;
	c_33 -= l_32 * u_23;

	u_33 = c_33;
	iu_3 = 1.0 / u_33;

	
	// store L
	L[0+bs*0] = 1.0;
	L[1+bs*0] = l_10;
	L[2+bs*0] = l_20;
	L[3+bs*0] = l_30;

	L[1+bs*1] = 1.0;
	L[2+bs*1] = l_21;
	L[3+bs*1] = l_31;

	L[2+bs*2] = 1.0;
	L[3+bs*2] = l_32;

	L[3+bs*3] = 1.0;


	// store U^T
	U[0+bs*0] = u_00;
	U[1+bs*0] = u_01;
	U[2+bs*0] = u_02;
	U[3+bs*0] = u_03;

	U[1+bs*1] = u_11;
	U[2+bs*1] = u_12;
	U[3+bs*1] = u_13;

	U[2+bs*2] = u_22;
	U[3+bs*2] = u_23;

	U[3+bs*3] = u_33;

	
	// return
	return;
	
	}

#endif



// C numbering, starting from 0
void idamax_lib4(int n, int offset, double *pA, int sda, int *p_idamax, double *p_amax)
	{

	int idamax, ii;
	double tmp, amax;
		
	p_idamax[0] = -1;
	if(n<1)
		return;

	const int bs = 4;

	int na = (bs - offset%bs)%bs;
	na = n<na ? n : na;

	amax = -1.0;
	ii = 0;
	if(na>0)
		{
		for( ; ii<na; ii++)
			{
			tmp = fabs(pA[0]);
			if(tmp>amax)
				{
				idamax = ii+0;
				amax = tmp;
				}
			pA += 1;
			}
		pA += bs*(sda-1);
		}
	for( ; ii<n-3; ii+=4)
		{
		tmp = fabs(pA[0]);
		if(tmp>amax)
			{
			idamax = ii+0;
			amax = tmp;
			}
		tmp = fabs(pA[1]);
		if(tmp>amax)
			{
			idamax = ii+1;
			amax = tmp;
			}
		tmp = fabs(pA[2]);
		if(tmp>amax)
			{
			idamax = ii+2;
			amax = tmp;
			}
		tmp = fabs(pA[3]);
		if(tmp>amax)
			{
			idamax = ii+3;
			amax = tmp;
			}
		pA += bs*sda;
		}
	for( ; ii<n; ii++)
		{
		tmp = fabs(pA[0]);
		if(tmp>amax)
			{
			idamax = ii+1;
			}
		pA += 1;
		}
	
	p_amax[0] = amax;
	p_idamax[0] = idamax;

	return;

	}



// C numering (starting from zero) in the ipiv
void kernel_dgetrf_pivot_4_lib4(int m, double *pA, int sda, double *inv_diag_A, int* ipiv)
	{

	const int bs = 4;

	// assume m>=4
	int ma = m-4;

	double
		tmp0, tmp1, tmp2, tmp3,
		u_00, u_01, u_02, u_03,
		      u_11, u_12, u_13,
		            u_22, u_23,
		                  u_33;
	
	double
		*pB;
	
	int 
		k, idamax;
	
	// first column
	idamax_lib4(m-0, 0, &pA[0+bs*0], sda, &idamax, &tmp0);
	ipiv[0] = idamax;
	if(tmp0!=0.0)
		{
		if(ipiv[0]!=0)
			drowsw_lib(4, pA+0, pA+ipiv[0]/bs*bs*sda+ipiv[0]%bs);

		tmp0 = 1.0 / pA[0+bs*0];
		inv_diag_A[0] = tmp0;
		pA[1+bs*0] *= tmp0;
		pA[2+bs*0] *= tmp0;
		pA[3+bs*0] *= tmp0;
		pB = pA + bs*sda;
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*0] *= tmp0;
			pB[1+bs*0] *= tmp0;
			pB[2+bs*0] *= tmp0;
			pB[3+bs*0] *= tmp0;
			pB += bs*sda;
			}
		for( ; k<ma; k++)
			{
			pB[0+bs*0] *= tmp0;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[0] = 0.0;
		}

	// second column
	u_01  = pA[0+bs*1];
	tmp1  = pA[1+bs*1];
	tmp2  = pA[2+bs*1];
	tmp3  = pA[3+bs*1];
	tmp1 -= pA[1+bs*0] * u_01;
	tmp2 -= pA[2+bs*0] * u_01;
	tmp3 -= pA[3+bs*0] * u_01;
	pA[1+bs*1] = tmp1;
	pA[2+bs*1] = tmp2;
	pA[3+bs*1] = tmp3;
	pB = pA + bs*sda;
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*1];
		tmp1  = pB[1+bs*1];
		tmp2  = pB[2+bs*1];
		tmp3  = pB[3+bs*1];
		tmp0 -= pB[0+bs*0] * u_01;
		tmp1 -= pB[1+bs*0] * u_01;
		tmp2 -= pB[2+bs*0] * u_01;
		tmp3 -= pB[3+bs*0] * u_01;
		pB[0+bs*1] = tmp0;
		pB[1+bs*1] = tmp1;
		pB[2+bs*1] = tmp2;
		pB[3+bs*1] = tmp3;
		pB += bs*sda;
		}
	for( ; k<ma; k++)
		{
		tmp0 = pB[0+bs*1];
		tmp0 -= pB[0+bs*0] * u_01;
		pB[0+bs*1] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-1, 1, &pA[1+bs*1], sda, &idamax, &tmp1);
	ipiv[1] = idamax+1;
	if(tmp1!=0)
		{
		if(ipiv[1]!=1)
			drowsw_lib(4, pA+1, pA+ipiv[1]/bs*bs*sda+ipiv[1]%bs);

		tmp1 = 1.0 / pA[1+bs*1];
		inv_diag_A[1] = tmp1;
		pA[2+bs*1] *= tmp1;
		pA[3+bs*1] *= tmp1;
		pB = pA + bs*sda;
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*1] *= tmp1;
			pB[1+bs*1] *= tmp1;
			pB[2+bs*1] *= tmp1;
			pB[3+bs*1] *= tmp1;
			pB += bs*sda;
			}
		for( ; k<ma; k++)
			{
			pB[0+bs*1] *= tmp1;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[1] = 0.0;
		}

	// third column
	u_02  = pA[0+bs*2];
	u_12  = pA[1+bs*2];
	u_12 -= pA[1+bs*0] * u_02;
	pA[1+bs*2] = u_12;
	tmp2  = pA[2+bs*2];
	tmp3  = pA[3+bs*2];
	tmp2 -= pA[2+bs*0] * u_02;
	tmp3 -= pA[3+bs*0] * u_02;
	tmp2 -= pA[2+bs*1] * u_12;
	tmp3 -= pA[3+bs*1] * u_12;
	pA[2+bs*2] = tmp2;
	pA[3+bs*2] = tmp3;
	pB = pA + bs*sda;
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*2];
		tmp1  = pB[1+bs*2];
		tmp2  = pB[2+bs*2];
		tmp3  = pB[3+bs*2];
		tmp0 -= pB[0+bs*0] * u_02;
		tmp1 -= pB[1+bs*0] * u_02;
		tmp2 -= pB[2+bs*0] * u_02;
		tmp3 -= pB[3+bs*0] * u_02;
		tmp0 -= pB[0+bs*1] * u_12;
		tmp1 -= pB[1+bs*1] * u_12;
		tmp2 -= pB[2+bs*1] * u_12;
		tmp3 -= pB[3+bs*1] * u_12;
		pB[0+bs*2] = tmp0;
		pB[1+bs*2] = tmp1;
		pB[2+bs*2] = tmp2;
		pB[3+bs*2] = tmp3;
		pB += bs*sda;
		}
	for( ; k<ma; k++)
		{
		tmp0  = pB[0+bs*2];
		tmp0 -= pB[0+bs*0] * u_02;
		tmp0 -= pB[0+bs*1] * u_12;
		pB[0+bs*2] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-2, 2, &pA[2+bs*2], sda, &idamax, &tmp2);
	ipiv[2] = idamax+2;
	if(tmp2!=0)
		{
		if(ipiv[2]!=2)
			drowsw_lib(4, pA+2, pA+ipiv[2]/bs*bs*sda+ipiv[2]%bs);

		tmp2 = 1.0 / pA[2+bs*2];
		inv_diag_A[2] = tmp2;
		pA[3+bs*2] *= tmp2;
		pB = pA + bs*sda;
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*2] *= tmp2;
			pB[1+bs*2] *= tmp2;
			pB[2+bs*2] *= tmp2;
			pB[3+bs*2] *= tmp2;
			pB += bs*sda;
			}
		for( ; k<ma; k++)
			{
			pB[0+bs*2] *= tmp2;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[2] = 0.0;
		}

	// fourth column
	u_03  = pA[0+bs*3];
	u_13  = pA[1+bs*3];
	u_13 -= pA[1+bs*0] * u_03;
	pA[1+bs*3] = u_13;
	u_23  = pA[2+bs*3];
	u_23 -= pA[2+bs*0] * u_03;
	u_23 -= pA[2+bs*1] * u_13;
	pA[2+bs*3] = u_23;
	tmp3  = pA[3+bs*3];
	tmp3 -= pA[3+bs*0] * u_03;
	tmp3 -= pA[3+bs*1] * u_13;
	tmp3 -= pA[3+bs*2] * u_23;
	pA[3+bs*3] = tmp3;
	pB = pA + bs*sda;
	for(k=0; k<ma-3; k+=4)
		{
		tmp0  = pB[0+bs*3];
		tmp1  = pB[1+bs*3];
		tmp2  = pB[2+bs*3];
		tmp3  = pB[3+bs*3];
		tmp0 -= pB[0+bs*0] * u_03;
		tmp1 -= pB[1+bs*0] * u_03;
		tmp2 -= pB[2+bs*0] * u_03;
		tmp3 -= pB[3+bs*0] * u_03;
		tmp0 -= pB[0+bs*1] * u_13;
		tmp1 -= pB[1+bs*1] * u_13;
		tmp2 -= pB[2+bs*1] * u_13;
		tmp3 -= pB[3+bs*1] * u_13;
		tmp0 -= pB[0+bs*2] * u_23;
		tmp1 -= pB[1+bs*2] * u_23;
		tmp2 -= pB[2+bs*2] * u_23;
		tmp3 -= pB[3+bs*2] * u_23;
		pB[0+bs*3] = tmp0;
		pB[1+bs*3] = tmp1;
		pB[2+bs*3] = tmp2;
		pB[3+bs*3] = tmp3;
		pB += bs*sda;
		}
	for( ; k<ma; k++)
		{
		tmp0  = pB[0+bs*3];
		tmp0 -= pB[0+bs*0] * u_03;
		tmp0 -= pB[0+bs*1] * u_13;
		tmp0 -= pB[0+bs*2] * u_23;
		pB[0+bs*3] = tmp0;
		pB += 1;
		}

	idamax_lib4(m-3, 3, &pA[3+bs*3], sda, &idamax, &tmp3);
	ipiv[3] = idamax+3;
	if(tmp3!=0)
		{
		if(ipiv[3]!=3)
			drowsw_lib(4, pA+3, pA+ipiv[3]/bs*bs*sda+ipiv[3]%bs);

		tmp3 = 1.0 / pA[3+bs*3];
		inv_diag_A[3] = tmp3;
		pB = pA + bs*sda;
		for(k=0; k<ma-3; k+=4)
			{
			pB[0+bs*3] *= tmp3;
			pB[1+bs*3] *= tmp3;
			pB[2+bs*3] *= tmp3;
			pB[3+bs*3] *= tmp3;
			pB += bs*sda;
			}
		for( ; k<ma; k++)
			{
			pB[0+bs*3] *= tmp3;
			pB += 1;
			}
		}
	else
		{
		inv_diag_A[3] = 0.0;
		}
	
	return;

	}


	


