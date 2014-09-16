
SET( SRCS c_order_dynamic_mem_interface.c c_order_static_mem_interface.c fortran_order_dynamic_mem_interface.c fortran_order_static_mem_interface.c )

FOREACH( SRC ${SRCS} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_INTERFACES ${SRC_INTERFACES} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_INTERFACES} PARENT_SCOPE )
