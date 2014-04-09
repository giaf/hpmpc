// normal-transposed, 2x2 with data packed in 2
void kernel_dgemm_nt_2x2_atom_lib2(int kmax, double *A, double *B, double *C, int bs_dummy, int alg)
	{

	if(kmax<=0)
		return;
	
	int k_iter = kmax/4;
	int k_left = kmax%4;

	__asm__ volatile
	(
		"                                \n\t"
		"                                \n\t"
		"movl   %2, %%eax                \n\t" // load address of A
		"movl   %3, %%edx                \n\t" // load address of B
		"                                \n\t"
		"                                \n\t"
		"prefetcht0 0(%%eax)             \n\t"
		"prefetcht0 0(%%edx)             \n\t"
		"prefetcht0 64(%%eax)            \n\t"
		"prefetcht0 64(%%edx)            \n\t"
		"                                \n\t"
		"                                \n\t"
		"movsd  (%%eax), %%xmm6          \n\t" // A[0]
/*		"movsd  (%%edx), %%xmm4          \n\t" // B[0]*/
		"                                \n\t"
		"                                \n\t" // zero registers
		"xorps  %%xmm0, %%xmm0           \n\t" // C_00
		"movsd  %%xmm0, %%xmm1           \n\t" // C_01
		"movsd  %%xmm0, %%xmm2           \n\t" // C_10
		"movsd  %%xmm0, %%xmm3           \n\t" // C_11
		"movsd  %%xmm0, %%xmm4           \n\t"
		"movsd  %%xmm0, %%xmm5           \n\t"
		"movsd  %%xmm0, %%xmm7           \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %0, %%ecx                \n\t" // i = k_iter;
		"testl  %%ecx, %%ecx             \n\t" // check i
		"je     .DCONSIDKLEFT2            \n\t" // if i==0, jump to k_left loop
		"                                \n\t"
		".DLOOPKITER2:                    \n\t" // MAIN LOOP
		"                                \n\t"
		"                                \n\t"
		"prefetcht0 128(%%eax)           \n\t"
		"prefetcht0 128(%%edx)           \n\t"
		"                                \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #1
		"movsd  0(%%edx), %%xmm4         \n\t"
		"mulsd  0(%%edx), %%xmm6         \n\t"
/*		"mulsd  %%xmm4, %%xmm6         \n\t"*/
		"addsd  %%xmm7, %%xmm3           \n\t"
		"movsd  8(%%eax), %%xmm7         \n\t"
		"mulsd  8(%%eax), %%xmm4         \n\t"
/*		"mulsd  %%xmm7, %%xmm4         \n\t"*/
		"                                \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"movsd  8(%%edx), %%xmm5         \n\t"
		"mulsd  8(%%edx), %%xmm7         \n\t"
/*		"mulsd  %%xmm5, %%xmm7         \n\t"*/
		"addsd  %%xmm6, %%xmm0           \n\t"
		"movsd  16(%%eax), %%xmm6        \n\t"
		"mulsd  0(%%eax), %%xmm5         \n\t"
/*		"mulsd  %%xmm6, %%xmm5         \n\t"*/
		"decl   %%ecx                    \n\t" // i -= 1;
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #2
		"movsd  16(%%edx), %%xmm4        \n\t"
		"mulsd  16(%%edx), %%xmm6        \n\t"
/*		"mulsd  %%xmm4, %%xmm6         \n\t"*/
		"addsd  %%xmm7, %%xmm3           \n\t"
		"movsd  24(%%eax), %%xmm7        \n\t"
		"mulsd  24(%%eax), %%xmm4        \n\t"
/*		"mulsd  %%xmm7, %%xmm4         \n\t"*/
		"                                \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"movsd  24(%%edx), %%xmm5        \n\t"
		"mulsd  24(%%edx), %%xmm7        \n\t"
/*		"mulsd  %%xmm5, %%xmm7         \n\t"*/
		"addsd  %%xmm6, %%xmm0           \n\t"
		"movsd  32(%%eax), %%xmm6        \n\t"
		"mulsd  16(%%eax), %%xmm5        \n\t"
/*		"mulsd  %%xmm6, %%xmm5         \n\t"*/
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #3
		"movsd  32(%%edx), %%xmm4        \n\t"
		"mulsd  32(%%edx), %%xmm6        \n\t"
/*		"mulsd  %%xmm4, %%xmm6         \n\t"*/
		"addsd  %%xmm7, %%xmm3           \n\t"
		"movsd  40(%%eax), %%xmm7        \n\t"
		"mulsd  40(%%eax), %%xmm4        \n\t"
/*		"mulsd  %%xmm7, %%xmm4         \n\t"*/
		"                                \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"movsd  40(%%edx), %%xmm5        \n\t"
		"mulsd  40(%%edx), %%xmm7        \n\t"
/*		"mulsd  %%xmm5, %%xmm7         \n\t"*/
		"addsd  %%xmm6, %%xmm0           \n\t"
		"movsd  48(%%eax), %%xmm6        \n\t"
		"mulsd  32(%%eax), %%xmm5        \n\t"
/*		"mulsd  %%xmm6, %%xmm5         \n\t"*/
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #4
		"movsd  48(%%edx), %%xmm4        \n\t"
		"mulsd  48(%%edx), %%xmm6        \n\t"
/*		"mulsd  %%xmm4, %%xmm6         \n\t"*/
		"addsd  %%xmm7, %%xmm3           \n\t"
		"movsd  56(%%eax), %%xmm7        \n\t"
		"mulsd  56(%%eax), %%xmm4        \n\t"
/*		"mulsd  %%xmm7, %%xmm4         \n\t"*/
		"                                \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"movsd  56(%%edx), %%xmm5        \n\t"
		"mulsd  56(%%edx), %%xmm7        \n\t"
/*		"mulsd  %%xmm5, %%xmm7         \n\t"*/
		"leal   64(%%edx), %%edx         \n\t"
		"addsd  %%xmm6, %%xmm0           \n\t"
		"movsd  64(%%eax), %%xmm6        \n\t"
		"mulsd  48(%%eax), %%xmm5        \n\t"
/*		"mulsd  %%xmm6, %%xmm5         \n\t"*/
		"leal   64(%%eax), %%eax         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKITER2              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DCONSIDKLEFT2:                  \n\t"
		"                                \n\t"
		"movl   %1, %%ecx                \n\t" // i = k_left;
		"testl  %%ecx, %%ecx             \n\t" // check i via logical AND.
		"je     .DPOSTACCUM2              \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".DLOOPKLEFT2:                    \n\t" // EDGE LOOP
		"                                \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #1
		"movsd  0(%%edx), %%xmm4         \n\t"
		"mulsd  0(%%edx), %%xmm6         \n\t"
		"addsd  %%xmm7, %%xmm3           \n\t"
		"movsd  8(%%eax), %%xmm7         \n\t"
		"mulsd  8(%%eax), %%xmm4         \n\t"
		"decl   %%ecx                    \n\t" // i -= 1;
		"                                \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"movsd  8(%%edx), %%xmm5         \n\t"
		"mulsd  8(%%edx), %%xmm7         \n\t"
		"leal   16(%%edx), %%edx         \n\t"
		"addsd  %%xmm6, %%xmm0           \n\t"
		"movsd  16(%%eax), %%xmm6        \n\t"
		"mulsd  0(%%eax), %%xmm5         \n\t"
		"leal   16(%%eax), %%eax         \n\t"
		"                                \n\t"
		"                                \n\t"
		"jne    .DLOOPKLEFT2              \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DPOSTACCUM2:                    \n\t"
		"                                \n\t"
		"addsd  %%xmm4, %%xmm2           \n\t" // unroll #1
		"addsd  %%xmm7, %%xmm3           \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl   %4, %%eax                \n\t" // load address of C
		"                                \n\t"
		"                                \n\t"
		"movl   %5, %%ecx                \n\t" // alg
		"testl  %%ecx, %%ecx             \n\t" // check alg
		"je     .D02                      \n\t" // if alg==0, jump
		"                                \n\t"
		"cmpl	$1, %%ecx                \n\t"
		"movsd  (%%eax), %%xmm4          \n\t"
		"movsd  8(%%eax), %%xmm6         \n\t"
		"movsd  16(%%eax),%%xmm5         \n\t"
		"movsd  24(%%eax), %%xmm7        \n\t"
		"je     .D12                      \n\t" // if alg==1, jump
		"                                \n\t"
		"                                \n\t"// alg==-1
		"subsd  %%xmm0, %%xmm4           \n\t"
		"subsd  %%xmm1, %%xmm5           \n\t"
		"subsd  %%xmm2, %%xmm6           \n\t"
		"subsd  %%xmm3, %%xmm7           \n\t"
		"                                \n\t"
		"movsd  %%xmm4, (%%eax)          \n\t"
		"movsd  %%xmm6, 8(%%eax)         \n\t"
		"movsd  %%xmm5, 16(%%eax)        \n\t"
		"movsd  %%xmm7, 24(%%eax)        \n\t"
		"                                \n\t"
		"jmp    .SDONE2                   \n\t" // jump to end
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".D12:                            \n\t" // alg==1
		"                                \n\t"
		"addsd  %%xmm4, %%xmm0           \n\t"
		"addsd  %%xmm5, %%xmm1           \n\t"
		"addsd  %%xmm6, %%xmm2           \n\t"
		"addsd  %%xmm7, %%xmm3           \n\t"
		"                                \n\t"
		".D02:                            \n\t" // alg==0
		"                                \n\t"
		"movsd  %%xmm0, (%%eax)          \n\t"
		"movsd  %%xmm2, 8(%%eax)         \n\t"
		"movsd  %%xmm1, 16(%%eax)        \n\t"
		"movsd  %%xmm3, 24(%%eax)        \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".SDONE2:                         \n\t" // end
		"                                \n\t"
		: // output operands (none)
		: // input operands
		  "m" (k_iter),		// %0
		  "m" (k_left),		// %1
		  "m" (A),			// %2
		  "m" (B),			// %3
		  "m" (C),			// %4
		  "m" (alg)			// %5
		: // register clobber list
		  "eax", "edx", "ecx",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);
}



// normal-transposed, 2x1 with data packed in 2
void kernel_dgemm_pp_nt_2x1_c99_lib2(int kmax, double *A, double *B, double *C, int bs, int alg)
	{
	
	if(kmax<=0)
		return;

/*	const int bs = 2;	*/

	int k;
	
	double
		a_0k, a_1k, b_0k,
		c_00=0, c_10=0;
	
	k = 0;
	for(; k<kmax-3; k+=4)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[2];
		a_1k = A[3];
		
		b_0k = B[2];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[4];
		a_1k = A[5];
		
		b_0k = B[4];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		
		a_0k = A[6];
		a_1k = A[7];
		
		b_0k = B[6];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		A += 8;
		B += 8;

		}
	
	for(; k<kmax; k++)
		{

		a_0k = A[0];
		a_1k = A[1];
		
		b_0k = B[0];
		
		c_00 += a_0k * b_0k;
		c_10 += a_1k * b_0k;
		
		A += 2;
		B += 2;
		
		}

	if(alg==0)
		{
		C[0+bs*0] = c_00;
		C[1+bs*0] = c_10;
		}
	else if(alg==1)
		{
		C[0+bs*0] += c_00;
		C[1+bs*0] += c_10;
		}
	else
		{
		C[0+bs*0] -= c_00;
		C[1+bs*0] -= c_10;
		}

	}




