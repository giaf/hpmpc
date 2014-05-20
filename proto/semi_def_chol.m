function A = semi_def_chol(C)

	n = size(C, 1);

	A = zeros(n,n);

	for jj=1:n
		for ii=jj:n
			A(ii,jj) = C(ii,jj);
		end
	end

	
	for ii=1:n
		
		% correction
		if(ii>1)
			A(ii:n,ii) -= A(ii:n,1:ii-1)*(A(ii,1:ii-1))';
		endif
		
		% factorization
		a_ii = A(ii, ii);
		if a_ii>1e-15
			temp = sqrt(a_ii);
			A(ii, ii) = temp;
			a_ii = 1/temp;
		else
			A(ii, ii) = 0;
			a_ii = 0;
		endif
		
		% solution
		if(ii<n)
			A(ii+1:n,ii) *= a_ii;
		endif
	
	end
		
