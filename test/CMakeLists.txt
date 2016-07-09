#
# HPMPC/test
#

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(googletest)

set(SRC 
    hpmpc_test.cpp
    test_qp.cpp
   )
   
add_executable(hpmpc_test ${SRC})

target_link_libraries(hpmpc_test 
	${PROJECT_SOURCE_DIR}/libhpmpc.a 
	gtest
	gtest_main
)

#enable_testing()
#add_test(AllTests mpc_test)
