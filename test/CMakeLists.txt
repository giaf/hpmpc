#
# hpmpc/test
#

include(CompilerOptions.cmake)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(googletest)

set(SRC 
    hpmpc_test.cpp
    test_qp.cpp
   )
   
add_executable(hpmpc_test ${SRC})

if (${CMAKE_CXX_COMPILER_ID} MATCHES "(GNU|Clang)")
	target_compile_options(hpmpc_test
		PRIVATE	-std=c++14
	)
endif()
	

target_link_libraries(hpmpc_test 
	hpmpc 
	gtest
	gtest_main
)
