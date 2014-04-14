void corner_strmm_pp_nt_2x1_c99_lib2(float *A, float *B, float *C, int ldc)
	{
	
	const int bs = 2;
	
	float
		a_00, a_10,
		b_00;
	
	b_00 = B[0+bs*0];
	
	a_00 = A[0+bs*0];
	a_10 = A[1+bs*0];
	
	C[0+ldc*0] = a_00*b_00;
	C[1+ldc*0] = a_10*b_00;

	}

