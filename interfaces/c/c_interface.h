// dynamic memory allocation & c (or row-major) order

//int c_order_dynamic_mem_ip_wrapper( int k_max, double tol, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, int* nIt, double *stat );
int c_order_dynamic_mem_riccati_wrapper_init( const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double **ptr_work );
int c_order_dynamic_mem_riccati_wrapper_free( double *work );
int c_order_dynamic_mem_riccati_wrapper_fact_solve( const int nx, const int nu, const int N, double *x, double *u, double *pi, double *work );
int c_order_dynamic_mem_riccati_wrapper_solve( const int nx, const int nu, const int N, double *b, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work );



// dynamic memory allocation & fortran (or column-major) order

//int fortran_order_dynamic_mem_ip_wrapper( int k_max, double tol, const int nx, const int nu, const int N, double* A, double* B, double* b, double* Q, double* Qf, double* S, double* R, double* q, double* qf, double* r, double* lb, double* ub, double* x, double* u, int* nIt, double *stat );
int fortran_order_dynamic_mem_riccati_wrapper_init( const int nx, const int nu, const int N, double *A, double *B, double *b, double *Q, double *Qf, double *S, double *R, double *q, double *qf, double *r, double **ptr_work );
int fortran_order_dynamic_mem_riccati_wrapper_free( double *work );
int fortran_order_dynamic_mem_riccati_wrapper_fact_solve( const int nx, const int nu, const int N, double *x, double *u, double *pi, double *work );
int fortran_order_dynamic_mem_riccati_wrapper_solve( const int nx, const int nu, const int N, double *b, double *q, double *qf, double *r, double *x, double *u, double *pi, double *work );
// mhe
int fortran_order_dynamic_mem_riccati_wrapper_mhe( const int nx, const int nw, const int ny, const int N, double *A, double *G, double *C, double *f, double *Q, double *R, double *q, double *r, double *y, double *x0, double *L0, double *xe, double *Le );



// static memory allocation & c (or row-major) order




// static memory allocation & fortran (or column-major) order

