ELSEIF ( ${HPMPC_TARGET} STREQUAL "CORTEX_A57" )
HPMPC_ADD_SOURCE_FILES(
	# Double precision
	kernel_dgemm_neon_lib4.c  kernel_dgemm_neon_assembly_lib4.c 	
	# Single precision
	kernel_sgemm_neon_assembly_lib4.c
	)

ELSE( )
    MESSAGE( ERROR "Impossible happened!" )
ENDIF( )

