#include <stdlib.h>
#include <stdio.h>

#include "../include/blas_d.h"
#include "../include/kernel_d_avx.h"

void dricposv_mpc(int nx, int nu, int N, int sda, double **hpBAbt, double **hpQ, double **hux, double *pL, double *pBAbtL)
	{
	if(!(nx==30 && nu==14 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	double *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* factorization and backward substitution */
	
	/* final stage */
	
	/* dpotrf */
	pC = hpQ[10];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(41, &pC[0], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[192], &pC[384], &pC[192], &pC[208], &pC[400], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[576], &pC[768], &pC[192], &pC[592], &pC[784], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[960], &pC[1152], &pC[192], &pC[976], &pC[1168], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1344], &pC[1536], &pC[192], &pC[1360], &pC[1552], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1728], &pC[1920], &pC[192], &pC[1744], &pC[1936], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[2112], &pC[192], &pC[2128], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(37, &pC[208], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[384], &pC[576], &pC[384], &pC[416], &pC[608], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[768], &pC[960], &pC[384], &pC[800], &pC[992], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1152], &pC[1344], &pC[384], &pC[1184], &pC[1376], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1536], &pC[1728], &pC[384], &pC[1568], &pC[1760], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1920], &pC[2112], &pC[384], &pC[1952], &pC[2144], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(33, &pC[416], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[576], &pC[768], &pC[576], &pC[624], &pC[816], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[960], &pC[1152], &pC[576], &pC[1008], &pC[1200], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[1344], &pC[1536], &pC[576], &pC[1392], &pC[1584], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[1728], &pC[1920], &pC[576], &pC[1776], &pC[1968], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[2112], &pC[576], &pC[2160], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(29, &pC[624], 48, 2, &pL[0], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[768], &pC[960], &pC[768], &pC[832], &pC[1024], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1152], &pC[1344], &pC[768], &pC[1216], &pC[1408], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1536], &pC[1728], &pC[768], &pC[1600], &pC[1792], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1920], &pC[2112], &pC[768], &pC[1984], &pC[2176], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(25, &pC[832], 48, 2, &pL[208], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[960], &pC[1152], &pC[960], &pC[1040], &pC[1232], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[1344], &pC[1536], &pC[960], &pC[1424], &pC[1616], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[1728], &pC[1920], &pC[960], &pC[1808], &pC[2000], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(20, &pC[2112], &pC[960], &pC[2192], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(21, &pC[1040], 48, 2, &pL[416], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1152], &pC[1344], &pC[1152], &pC[1248], &pC[1440], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1536], &pC[1728], &pC[1152], &pC[1632], &pC[1824], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1920], &pC[2112], &pC[1152], &pC[2016], &pC[2208], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(17, &pC[1248], 48, 2, &pL[624], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(28, &pC[1344], &pC[1536], &pC[1344], &pC[1456], &pC[1648], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(28, &pC[1728], &pC[1920], &pC[1344], &pC[1840], &pC[2032], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(28, &pC[2112], &pC[1344], &pC[2224], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(13, &pC[1456], 48, 2, &pL[832], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(32, &pC[1536], &pC[1728], &pC[1536], &pC[1664], &pC[1856], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(32, &pC[1920], &pC[2112], &pC[1536], &pC[2048], &pC[2240], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(9, &pC[1664], 48, 2, &pL[1040], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(36, &pC[1728], &pC[1920], &pC[1728], &pC[1872], &pC[2064], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(36, &pC[2112], &pC[1728], &pC[2256], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(5, &pC[1872], 48, 2, &pL[1248], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(40, &pC[1920], &pC[2112], &pC[1920], &pC[2080], &pC[2272], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(1, &pC[2080], 48, 2, &pL[1456], 48);
	kernel_dgemm_pp_nt_4x1_avx_lib4(44, &pC[2112], &pC[2112], &pC[2288], 4, -1);
	corner_dpotrf_dtrsv_dcopy_1x1_sse_lib4(&pC[2288], 48, 2, &pL[1664], 48);
	
	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{
		
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+208;
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[0], &pA[192], &pB[0], &pC[0], &pC[192], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[16], &pA[208], &pB[208], &pC[16], &pC[208], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[32], &pA[224], &pB[416], &pC[32], &pC[224], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[48], &pA[240], &pB[624], &pC[48], &pC[240], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[64], &pA[256], &pB[832], &pC[64], &pC[256], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[80], &pA[272], &pB[1040], &pC[80], &pC[272], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[96], &pA[288], &pB[1248], &pC[96], &pC[288], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[112], &pA[304], &pB[1456], &pC[112], &pC[304], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pB[0], &pC[384], &pC[576], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[400], &pA[592], &pB[208], &pC[400], &pC[592], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[416], &pA[608], &pB[416], &pC[416], &pC[608], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[432], &pA[624], &pB[624], &pC[432], &pC[624], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[448], &pA[640], &pB[832], &pC[448], &pC[640], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[464], &pA[656], &pB[1040], &pC[464], &pC[656], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[480], &pA[672], &pB[1248], &pC[480], &pC[672], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[496], &pA[688], &pB[1456], &pC[496], &pC[688], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pB[0], &pC[768], &pC[960], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[784], &pA[976], &pB[208], &pC[784], &pC[976], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[800], &pA[992], &pB[416], &pC[800], &pC[992], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[816], &pA[1008], &pB[624], &pC[816], &pC[1008], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[832], &pA[1024], &pB[832], &pC[832], &pC[1024], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[848], &pA[1040], &pB[1040], &pC[848], &pC[1040], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[864], &pA[1056], &pB[1248], &pC[864], &pC[1056], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[880], &pA[1072], &pB[1456], &pC[880], &pC[1072], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pB[0], &pC[1152], &pC[1344], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1168], &pA[1360], &pB[208], &pC[1168], &pC[1360], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1184], &pA[1376], &pB[416], &pC[1184], &pC[1376], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1200], &pA[1392], &pB[624], &pC[1200], &pC[1392], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1216], &pA[1408], &pB[832], &pC[1216], &pC[1408], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[1232], &pA[1424], &pB[1040], &pC[1232], &pC[1424], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[1248], &pA[1440], &pB[1248], &pC[1248], &pC[1440], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[1264], &pA[1456], &pB[1456], &pC[1264], &pC[1456], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pB[0], &pC[1536], &pC[1728], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1552], &pA[1744], &pB[208], &pC[1552], &pC[1744], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1568], &pA[1760], &pB[416], &pC[1568], &pC[1760], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1584], &pA[1776], &pB[624], &pC[1584], &pC[1776], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1600], &pA[1792], &pB[832], &pC[1600], &pC[1792], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[1616], &pA[1808], &pB[1040], &pC[1616], &pC[1808], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[1632], &pA[1824], &pB[1248], &pC[1632], &pC[1824], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[1648], &pA[1840], &pB[1456], &pC[1648], &pC[1840], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pB[0], &pC[1920], &pC[2112], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1936], &pA[2128], &pB[208], &pC[1936], &pC[2128], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1952], &pA[2144], &pB[416], &pC[1952], &pC[2144], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1968], &pA[2160], &pB[624], &pC[1968], &pC[2160], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1984], &pA[2176], &pB[832], &pC[1984], &pC[2176], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[2000], &pA[2192], &pB[1040], &pC[2000], &pC[2192], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[2016], &pA[2208], &pB[1248], &pC[2016], &pC[2208], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[2032], &pA[2224], &pB[1456], &pC[2032], &pC[2224], 4);
	pC[2112] += pB[120];
	pC[2116] += pB[121];
	pC[2120] += pB[122];
	pC[2124] += pB[123];
	pC[2128] += pB[312];
	pC[2132] += pB[313];
	pC[2136] += pB[314];
	pC[2140] += pB[315];
	pC[2144] += pB[504];
	pC[2148] += pB[505];
	pC[2152] += pB[506];
	pC[2156] += pB[507];
	pC[2160] += pB[696];
	pC[2164] += pB[697];
	pC[2168] += pB[698];
	pC[2172] += pB[699];
	pC[2176] += pB[888];
	pC[2180] += pB[889];
	pC[2184] += pB[890];
	pC[2188] += pB[891];
	pC[2192] += pB[1080];
	pC[2196] += pB[1081];
	pC[2200] += pB[1082];
	pC[2204] += pB[1083];
	pC[2208] += pB[1272];
	pC[2212] += pB[1273];
	pC[2216] += pB[1274];
	pC[2220] += pB[1275];
	pC[2224] += pB[1464];
	pC[2228] += pB[1465];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[0], &pA[192], &pA[0], &pC[0], &pC[192], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(30, &pA[192], &pA[192], &pC[208], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[0], &pC[384], &pC[576], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[192], &pC[400], &pC[592], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[384], &pC[416], &pC[608], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(30, &pA[576], &pA[576], &pC[624], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[0], &pC[768], &pC[960], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[192], &pC[784], &pC[976], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[384], &pC[800], &pC[992], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[576], &pC[816], &pC[1008], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[768], &pC[832], &pC[1024], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(30, &pA[960], &pA[960], &pC[1040], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[0], &pC[1152], &pC[1344], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[192], &pC[1168], &pC[1360], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[384], &pC[1184], &pC[1376], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[576], &pC[1200], &pC[1392], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[768], &pC[1216], &pC[1408], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[960], &pC[1232], &pC[1424], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[1152], &pC[1248], &pC[1440], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(30, &pA[1344], &pA[1344], &pC[1456], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[0], &pC[1536], &pC[1728], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[192], &pC[1552], &pC[1744], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[384], &pC[1568], &pC[1760], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[576], &pC[1584], &pC[1776], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[768], &pC[1600], &pC[1792], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[960], &pC[1616], &pC[1808], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[1152], &pC[1632], &pC[1824], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[1344], &pC[1648], &pC[1840], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[1536], &pC[1664], &pC[1856], 4, 1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(30, &pA[1728], &pA[1728], &pC[1872], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[0], &pC[1920], &pC[2112], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[192], &pC[1936], &pC[2128], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[384], &pC[1952], &pC[2144], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[576], &pC[1968], &pC[2160], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[768], &pC[1984], &pC[2176], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[960], &pC[2000], &pC[2192], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[1152], &pC[2016], &pC[2208], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[1344], &pC[2032], &pC[2224], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[1536], &pC[2048], &pC[2240], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[1728], &pC[2064], &pC[2256], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[1920], &pC[2080], &pC[2272], 4, 1);
	kernel_dgemm_pp_nt_4x1_avx_lib4(30, &pA[2112], &pA[2112], &pC[2288], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dpotrf_dtrsv_4x4_sse_lib4(41, &pC[0], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[192], &pC[384], &pC[192], &pC[208], &pC[400], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[576], &pC[768], &pC[192], &pC[592], &pC[784], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[960], &pC[1152], &pC[192], &pC[976], &pC[1168], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1344], &pC[1536], &pC[192], &pC[1360], &pC[1552], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1728], &pC[1920], &pC[192], &pC[1744], &pC[1936], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[2112], &pC[192], &pC[2128], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(37, &pC[208], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[384], &pC[576], &pC[384], &pC[416], &pC[608], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[768], &pC[960], &pC[384], &pC[800], &pC[992], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1152], &pC[1344], &pC[384], &pC[1184], &pC[1376], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1536], &pC[1728], &pC[384], &pC[1568], &pC[1760], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1920], &pC[2112], &pC[384], &pC[1952], &pC[2144], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(33, &pC[416], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[576], &pC[768], &pC[576], &pC[624], &pC[816], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[960], &pC[1152], &pC[576], &pC[1008], &pC[1200], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[1344], &pC[1536], &pC[576], &pC[1392], &pC[1584], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(12, &pC[1728], &pC[1920], &pC[576], &pC[1776], &pC[1968], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(12, &pC[2112], &pC[576], &pC[2160], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(29, &pC[624], 48, 2, &pL[0], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[768], &pC[960], &pC[768], &pC[832], &pC[1024], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1152], &pC[1344], &pC[768], &pC[1216], &pC[1408], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1536], &pC[1728], &pC[768], &pC[1600], &pC[1792], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(16, &pC[1920], &pC[2112], &pC[768], &pC[1984], &pC[2176], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(25, &pC[832], 48, 2, &pL[208], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[960], &pC[1152], &pC[960], &pC[1040], &pC[1232], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[1344], &pC[1536], &pC[960], &pC[1424], &pC[1616], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(20, &pC[1728], &pC[1920], &pC[960], &pC[1808], &pC[2000], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(20, &pC[2112], &pC[960], &pC[2192], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(21, &pC[1040], 48, 2, &pL[416], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1152], &pC[1344], &pC[1152], &pC[1248], &pC[1440], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1536], &pC[1728], &pC[1152], &pC[1632], &pC[1824], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(24, &pC[1920], &pC[2112], &pC[1152], &pC[2016], &pC[2208], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(17, &pC[1248], 48, 2, &pL[624], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(28, &pC[1344], &pC[1536], &pC[1344], &pC[1456], &pC[1648], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(28, &pC[1728], &pC[1920], &pC[1344], &pC[1840], &pC[2032], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(28, &pC[2112], &pC[1344], &pC[2224], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(13, &pC[1456], 48, 2, &pL[832], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(32, &pC[1536], &pC[1728], &pC[1536], &pC[1664], &pC[1856], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(32, &pC[1920], &pC[2112], &pC[1536], &pC[2048], &pC[2240], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(9, &pC[1664], 48, 2, &pL[1040], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(36, &pC[1728], &pC[1920], &pC[1728], &pC[1872], &pC[2064], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(36, &pC[2112], &pC[1728], &pC[2256], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(5, &pC[1872], 48, 2, &pL[1248], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(40, &pC[1920], &pC[2112], &pC[1920], &pC[2080], &pC[2272], 4, -1);
	kernel_dpotrf_dtrsv_dcopy_4x4_sse_lib4(1, &pC[2080], 48, 2, &pL[1456], 48);
	kernel_dgemm_pp_nt_4x1_avx_lib4(44, &pC[2112], &pC[2112], &pC[2288], 4, -1);
	corner_dpotrf_dtrsv_dcopy_1x1_sse_lib4(&pC[2288], 48, 2, &pL[1664], 48);
		
		}
	
	/* initial stage */
	
		/* dtrmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+208;
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[0], &pA[192], &pB[0], &pC[0], &pC[192], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[16], &pA[208], &pB[208], &pC[16], &pC[208], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[32], &pA[224], &pB[416], &pC[32], &pC[224], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[48], &pA[240], &pB[624], &pC[48], &pC[240], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[64], &pA[256], &pB[832], &pC[64], &pC[256], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[80], &pA[272], &pB[1040], &pC[80], &pC[272], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[96], &pA[288], &pB[1248], &pC[96], &pC[288], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[112], &pA[304], &pB[1456], &pC[112], &pC[304], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pB[0], &pC[384], &pC[576], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[400], &pA[592], &pB[208], &pC[400], &pC[592], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[416], &pA[608], &pB[416], &pC[416], &pC[608], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[432], &pA[624], &pB[624], &pC[432], &pC[624], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[448], &pA[640], &pB[832], &pC[448], &pC[640], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[464], &pA[656], &pB[1040], &pC[464], &pC[656], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[480], &pA[672], &pB[1248], &pC[480], &pC[672], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[496], &pA[688], &pB[1456], &pC[496], &pC[688], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pB[0], &pC[768], &pC[960], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[784], &pA[976], &pB[208], &pC[784], &pC[976], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[800], &pA[992], &pB[416], &pC[800], &pC[992], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[816], &pA[1008], &pB[624], &pC[816], &pC[1008], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[832], &pA[1024], &pB[832], &pC[832], &pC[1024], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[848], &pA[1040], &pB[1040], &pC[848], &pC[1040], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[864], &pA[1056], &pB[1248], &pC[864], &pC[1056], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[880], &pA[1072], &pB[1456], &pC[880], &pC[1072], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pB[0], &pC[1152], &pC[1344], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1168], &pA[1360], &pB[208], &pC[1168], &pC[1360], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1184], &pA[1376], &pB[416], &pC[1184], &pC[1376], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1200], &pA[1392], &pB[624], &pC[1200], &pC[1392], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1216], &pA[1408], &pB[832], &pC[1216], &pC[1408], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[1232], &pA[1424], &pB[1040], &pC[1232], &pC[1424], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[1248], &pA[1440], &pB[1248], &pC[1248], &pC[1440], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[1264], &pA[1456], &pB[1456], &pC[1264], &pC[1456], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pB[0], &pC[1536], &pC[1728], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1552], &pA[1744], &pB[208], &pC[1552], &pC[1744], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1568], &pA[1760], &pB[416], &pC[1568], &pC[1760], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1584], &pA[1776], &pB[624], &pC[1584], &pC[1776], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1600], &pA[1792], &pB[832], &pC[1600], &pC[1792], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[1616], &pA[1808], &pB[1040], &pC[1616], &pC[1808], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[1632], &pA[1824], &pB[1248], &pC[1632], &pC[1824], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[1648], &pA[1840], &pB[1456], &pC[1648], &pC[1840], 4);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pB[0], &pC[1920], &pC[2112], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(26, &pA[1936], &pA[2128], &pB[208], &pC[1936], &pC[2128], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(22, &pA[1952], &pA[2144], &pB[416], &pC[1952], &pC[2144], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(18, &pA[1968], &pA[2160], &pB[624], &pC[1968], &pC[2160], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(14, &pA[1984], &pA[2176], &pB[832], &pC[1984], &pC[2176], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(10, &pA[2000], &pA[2192], &pB[1040], &pC[2000], &pC[2192], 4, 0);
	kernel_dgemm_pp_nt_8x4_avx_lib4(6, &pA[2016], &pA[2208], &pB[1248], &pC[2016], &pC[2208], 4, 0);
	corner_dtrmm_pp_nt_8x2_avx_lib4(&pA[2032], &pA[2224], &pB[1456], &pC[2032], &pC[2224], 4);
	pC[2112] += pB[120];
	pC[2116] += pB[121];
	pC[2120] += pB[122];
	pC[2124] += pB[123];
	pC[2128] += pB[312];
	pC[2132] += pB[313];
	pC[2136] += pB[314];
	pC[2140] += pB[315];
	pC[2144] += pB[504];
	pC[2148] += pB[505];
	pC[2152] += pB[506];
	pC[2156] += pB[507];
	pC[2160] += pB[696];
	pC[2164] += pB[697];
	pC[2168] += pB[698];
	pC[2172] += pB[699];
	pC[2176] += pB[888];
	pC[2180] += pB[889];
	pC[2184] += pB[890];
	pC[2188] += pB[891];
	pC[2192] += pB[1080];
	pC[2196] += pB[1081];
	pC[2200] += pB[1082];
	pC[2204] += pB[1083];
	pC[2208] += pB[1272];
	pC[2212] += pB[1273];
	pC[2216] += pB[1274];
	pC[2220] += pB[1275];
	pC[2224] += pB[1464];
	pC[2228] += pB[1465];
		
		/* dsyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[0], &pA[192], &pA[0], &pC[0], &pC[192], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[0], &pA[192], &pA[192], &pC[16], &pC[208], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[0], &pC[384], &pC[576], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[192], &pC[400], &pC[592], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[384], &pA[576], &pA[384], &pC[416], &pC[608], 4, 1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(30, &pA[384], &pA[576], &pA[576], &pC[432], &pC[624], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[0], &pC[768], &pC[960], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[192], &pC[784], &pC[976], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[768], &pA[960], &pA[384], &pC[800], &pC[992], 4, 1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(30, &pA[768], &pA[960], &pA[576], &pC[816], &pC[1008], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[0], &pC[1152], &pC[1344], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[192], &pC[1168], &pC[1360], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1152], &pA[1344], &pA[384], &pC[1184], &pC[1376], 4, 1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(30, &pA[1152], &pA[1344], &pA[576], &pC[1200], &pC[1392], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[0], &pC[1536], &pC[1728], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[192], &pC[1552], &pC[1744], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1536], &pA[1728], &pA[384], &pC[1568], &pC[1760], 4, 1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(30, &pA[1536], &pA[1728], &pA[576], &pC[1584], &pC[1776], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[0], &pC[1920], &pC[2112], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[192], &pC[1936], &pC[2128], 4, 1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(30, &pA[1920], &pA[2112], &pA[384], &pC[1952], &pC[2144], 4, 1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(30, &pA[1920], &pA[2112], &pA[576], &pC[1968], &pC[2160], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[0], &pC[192], &pC[0], &pC[0], &pC[192], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[384], &pC[576], &pC[0], &pC[384], &pC[576], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[768], &pC[960], &pC[0], &pC[768], &pC[960], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[1152], &pC[1344], &pC[0], &pC[1152], &pC[1344], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[1536], &pC[1728], &pC[0], &pC[1536], &pC[1728], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(0, &pC[1920], &pC[2112], &pC[0], &pC[1920], &pC[2112], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(41, &pC[0], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[192], &pC[384], &pC[192], &pC[208], &pC[400], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[576], &pC[768], &pC[192], &pC[592], &pC[784], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[960], &pC[1152], &pC[192], &pC[976], &pC[1168], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1344], &pC[1536], &pC[192], &pC[1360], &pC[1552], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(4, &pC[1728], &pC[1920], &pC[192], &pC[1744], &pC[1936], 4, -1);
	kernel_dgemm_pp_nt_4x4_avx_lib4(4, &pC[2112], &pC[192], &pC[2128], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(37, &pC[208], 48);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[384], &pC[576], &pC[384], &pC[416], &pC[608], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[768], &pC[960], &pC[384], &pC[800], &pC[992], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1152], &pC[1344], &pC[384], &pC[1184], &pC[1376], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1536], &pC[1728], &pC[384], &pC[1568], &pC[1760], 4, -1);
	kernel_dgemm_pp_nt_8x4_avx_lib4(8, &pC[1920], &pC[2112], &pC[384], &pC[1952], &pC[2144], 4, -1);
	kernel_dpotrf_dtrsv_4x4_sse_lib4(33, &pC[416], 48);
	kernel_dgemm_pp_nt_8x2_avx_lib4(12, &pC[576], &pC[768], &pC[576], &pC[624], &pC[816], 4, -1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(12, &pC[960], &pC[1152], &pC[576], &pC[1008], &pC[1200], 4, -1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(12, &pC[1344], &pC[1536], &pC[576], &pC[1392], &pC[1584], 4, -1);
	kernel_dgemm_pp_nt_8x2_avx_lib4(12, &pC[1728], &pC[1920], &pC[576], &pC[1776], &pC[1968], 4, -1);
	kernel_dgemm_pp_nt_4x2_avx_lib4(12, &pC[2112], &pC[576], &pC[2160], 4, -1);
	kernel_dpotrf_dtrsv_2x2_sse_lib4(31, &pC[624], 48);
	
	
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<11; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][2112+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][2112+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][2112+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][2112+4*(jj+3)];
			}
		hux[ii][12] = hpQ[ii][2160];
		hux[ii][13] = hpQ[ii][2164];
		
		/* dgemv */
		pA = &hpQ[ii][578];
		x = &hux[ii][14];
		y = &hux[ii][0];
	kernel_dgemv_t_8_avx_lib4(30, 2, pA+0, 48, x, y+0, 1);
	kernel_dgemv_t_4_avx_lib4(30, 2, pA+32, 48, x, y+8, 1);
	kernel_dgemv_t_2_avx_lib4(30, 2, pA+48, 48, x, y+12, 1);
		
		/* dtrsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 624;
	ptrx = x + 12;
	kernel_dgemv_t_2_avx_lib4(0, 0, &ptrA[2], 48, &ptrx[2], &ptrx[0], -1);
	ptrx[1] = (ptrx[1]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[1]*ptrx[1]) / ptrA[0];
	ptrA = pA + 208;
	ptrx = x  + 4;
	kernel_dgemv_t_8_avx_lib4(2, 0, ptrA+384, sda, ptrx+8, ptrx, -1);
	ptrA = pA + 416;
	ptrx = x  + 8;
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
	ptrA = pA + 208;
	ptrx = x  + 4;
	kernel_dgemv_t_4_avx_lib4(4, 0, ptrA+192, sda, ptrx+4, ptrx, -1);
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
	ptrA = pA + 0;
	ptrx = x  + 0;
	kernel_dgemv_t_4_avx_lib4(10, 0, ptrA+192, sda, ptrx+4, ptrx, -1);
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
		
		/* */
		for(jj=0; jj<11; jj+=4)
			{
			hux[ii][jj+0] = - hux[ii][jj+0];
			hux[ii][jj+1] = - hux[ii][jj+1];
			hux[ii][jj+2] = - hux[ii][jj+2];
			hux[ii][jj+3] = - hux[ii][jj+3];
			}
		hux[ii][12] = - hux[ii][12];
		hux[ii][13] = - hux[ii][13];
		
		/* */
		for(jj=0; jj<27; jj+=4)
			{
			hux[ii+1][jj+14] = hpBAbt[ii][2112+4*(jj+0)];
			hux[ii+1][jj+15] = hpBAbt[ii][2112+4*(jj+1)];
			hux[ii+1][jj+16] = hpBAbt[ii][2112+4*(jj+2)];
			hux[ii+1][jj+17] = hpBAbt[ii][2112+4*(jj+3)];
			}
		hux[ii+1][42] = hpBAbt[ii][2224];
		hux[ii+1][43] = hpBAbt[ii][2228];
		
		/* dgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][14];
	kernel_dgemv_t_8_avx_lib4(44, 0, pA+0, 48, x, y+0, 1);
	kernel_dgemv_t_8_avx_lib4(44, 0, pA+32, 48, x, y+8, 1);
	kernel_dgemv_t_8_avx_lib4(44, 0, pA+64, 48, x, y+16, 1);
	kernel_dgemv_t_4_avx_lib4(44, 0, pA+96, 48, x, y+24, 1);
	kernel_dgemv_t_2_avx_lib4(44, 0, pA+112, 48, x, y+28, 1);
		
		
		}
	
	}
	
