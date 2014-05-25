
ADD_EXECUTABLE( test_dgemm_orig test_dgemm_orig.c )
TARGET_LINK_LIBRARIES( test_dgemm_orig hpmpc )
ADD_TEST(
	NAME test_dgemm_orig
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_dgemm_orig
)

ADD_EXECUTABLE( test_sgemm_orig test_sgemm_orig.c )
TARGET_LINK_LIBRARIES( test_sgemm_orig hpmpc )
ADD_TEST(
	NAME test_sgemm_orig
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_sgemm_orig
)

ADD_EXECUTABLE( test_blas_d test_blas_d.c )
TARGET_LINK_LIBRARIES( test_blas_d hpmpc )
ADD_TEST(
	NAME test_blas_d
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_blas_d
)
ADD_EXECUTABLE( test_blas_s test_blas_s.c )
TARGET_LINK_LIBRARIES( test_blas_s hpmpc )
ADD_TEST(
	NAME test_blas_s
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_blas_s
)

ADD_EXECUTABLE( test_dricposv test_dricposv.c tools.c)
TARGET_LINK_LIBRARIES( test_dricposv hpmpc )
ADD_TEST(
	NAME test_dricposv
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_dricposv
)
ADD_EXECUTABLE( test_sricposv test_sricposv.c tools.c)
TARGET_LINK_LIBRARIES( test_sricposv hpmpc )
ADD_TEST(
	NAME test_sricposv
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_sricposv
)

ADD_EXECUTABLE( test_d_ip_box test_d_ip_box.c tools.c)
TARGET_LINK_LIBRARIES( test_d_ip_box hpmpc )
ADD_TEST(
	NAME test_d_ip_box
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_d_ip_box
)
ADD_EXECUTABLE( test_s_ip_box test_s_ip_box.c tools.c)
TARGET_LINK_LIBRARIES( test_s_ip_box hpmpc )
ADD_TEST(
	NAME test_s_ip_box
	WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
	COMMAND test_s_ip_box
)
