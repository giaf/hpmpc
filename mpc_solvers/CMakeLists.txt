
SET( SRCS d_ip_box.c d_ip2_box.c dres_ip_box.c s_ip_box.c s_ip2_box.c sres_ip_box.c )

FOREACH( SRC ${SRCS} )
    GET_FILENAME_COMPONENT(ABS_SRC ${SRC} REALPATH)
	SET( SRC_LQCP ${SRC_LQCP} ${ABS_SRC} )
ENDFOREACH()
SET ( SOURCES ${SOURCES} ${SRC_LQCP} PARENT_SCOPE )
