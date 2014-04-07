/* return the number of rows of the C sub-matrix in the dgemm micro-kernel, double precision */
int d_get_mr()
	{
	int bs = 4;
	return bs;
	}

/* return the number of columns of the C sub-matrix in the dgemm micro-kernel, double precision */
int d_get_nr()
	{
	int bs = 4;
	return bs;
	}

/* return the number of rows of the C sub-matrix in the dgemm micro-kernel, single precision */
int s_get_mr()
	{
	int bs = 4;
	return bs;
	}

/* return the number of columns of the C sub-matrix in the dgemm micro-kernel, single precision */
int s_get_nr()
	{
	int bs = 4;
	return bs;
	}
