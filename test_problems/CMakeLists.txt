
ADD_LIBRARY( hpmpc_tools STATIC tools.c )

SET( TEST_PROBLEMS
	test_blas_d_debug.c
	test_blas_s_debug.c
#	test_blas_d.c
	test_blas_s.c
	test_d_ric_sv.c
	test_s_ric_sv.c
	test_d_ric_mhe.c
	test_d_ip_box.c
	test_s_ip_box.c
	test_d_admm_box.c
	test_s_admm_box.c
	test_d_admm_soft.c
	test_s_admm_soft.c
)

FOREACH( SRC ${TEST_PROBLEMS} )
	GET_FILENAME_COMPONENT( EXEC_NAME ${SRC} NAME_WE )
	ADD_EXECUTABLE( ${EXEC_NAME} ${SRC} )
	TARGET_LINK_LIBRARIES(
		${EXEC_NAME}
		hpmpc hpmpc_tools
	)
	ADD_TEST(
		NAME test_${EXEC_NAME}
		WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		COMMAND ${EXEC_NAME}
	)
ENDFOREACH( )
