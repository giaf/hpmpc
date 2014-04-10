#ifndef __HPMPC_BLOCK_SIZE__
#define __HPMPC_BLOCK_SIZE__

#if defined( TARGET_AVX )

#define D_MR 4
#define D_NR 4
#define S_MR 8
#define S_NR 4

#elif defined( TARGET_C99_4X4 )

#define D_MR 4
#define D_NR 4
#define S_MR 4
#define S_NR 4

//#elif defined( TARGET_SSE )

//#define D_MR 4
//#define D_NR 4
//#define S_MR 8
//#define S_NR 4

#elif defined( TARGET_ATOM )

#define D_MR 2
#define D_NR 2
#define S_MR 4
#define S_NR 4

#elif defined( TARGET_NEON )

#define D_MR 4
#define D_NR 4
#define S_MR 4
#define S_NR 4

#elif defined( TARGET_C99_2X2 )

#define D_MR 2
#define D_NR 2
#define S_MR 2
#define S_NR 2

//#elif defined( TARGET_SCAL )

//#define D_MR 4
//#define D_NR 2
//#define S_MR 4
//#define S_NR 2

#elif defined( TARGET_POWERPC_G2 )

#define D_MR 4
#define D_NR 4
#define S_MR 4
#define S_NR 4

#else
#error "Unknown architecture"
#endif /* __HPMPC_BLOCK_SIZE__ */

int d_get_mr();
int d_get_nr();
int s_get_mr();
int s_get_nr();

#endif /* __HPMPC_BLOCK_SIZE__ */
