#
# HPMPC/test
#

#enable_testing()
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SRC 
    hpmpc_test.cpp
    test_qp.cpp
   )
   
add_executable(hpmpc_test ${SRC})

target_link_libraries(hpmpc_test ${PROJECT_SOURCE_DIR}/libhpmpc.a ${GTEST_BOTH_LIBRARIES})

if (UNIX)
	# On UNIX, link to pthread -- gtest requires it.
	target_link_libraries(hpmpc_test pthread)
endif (UNIX)

#add_test(AllTests mpc_test)
