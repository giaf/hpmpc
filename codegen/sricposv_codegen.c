#include <stdlib.h>
#include <stdio.h>

#include "../include/kernel_d_avx.h"

void sricposv_mpc(int nx, int nu, int N, int sda, float **hpBAbt, float **hpQ, float **hux, float *pL, float *pBAbtL)
	{
	if(!(nx==22 && nu==10 && N==10))
		{
		printf("\nError: solver not generated for that problem size\n\n");
		exit(1);
		}
	
	float *pA, *pB, *pC, *x, *y, *ptrA, *ptrx;
	
	int i, j, k, ii, jj, kk;
	
	/* factorization and backward substitution */
	
	/* final stage */
	
	/*spotrf */
	pC = hpQ[10];
	kernel_spotrf_strsv_4x4_c99_lib4(29, &pC[0], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[144], &pC[144], &pC[160], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[288], &pC[144], &pC[304], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[432], &pC[144], &pC[448], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[576], &pC[144], &pC[592], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[720], &pC[144], &pC[736], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[864], &pC[144], &pC[880], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1008], &pC[144], &pC[1024], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1152], &pC[144], &pC[1168], 4, -1);
	kernel_spotrf_strsv_4x4_c99_lib4(25, &pC[160], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[288], &pC[288], &pC[320], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[432], &pC[288], &pC[464], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[576], &pC[288], &pC[608], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[720], &pC[288], &pC[752], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[864], &pC[288], &pC[896], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[1008], &pC[288], &pC[1040], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[1152], &pC[288], &pC[1184], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(21, &pC[320], 36, 2, &pL[0], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[432], &pC[432], &pC[480], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[576], &pC[432], &pC[624], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[720], &pC[432], &pC[768], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[864], &pC[432], &pC[912], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[1008], &pC[432], &pC[1056], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[1152], &pC[432], &pC[1200], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(17, &pC[480], 36, 2, &pL[160], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[576], &pC[576], &pC[640], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[720], &pC[576], &pC[784], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[864], &pC[576], &pC[928], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[1008], &pC[576], &pC[1072], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[1152], &pC[576], &pC[1216], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(13, &pC[640], 36, 2, &pL[320], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[720], &pC[720], &pC[800], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[864], &pC[720], &pC[944], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[1008], &pC[720], &pC[1088], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[1152], &pC[720], &pC[1232], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(9, &pC[800], 36, 2, &pL[480], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[864], &pC[864], &pC[960], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[1008], &pC[864], &pC[1104], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[1152], &pC[864], &pC[1248], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(5, &pC[960], 36, 2, &pL[640], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(28, &pC[1008], &pC[1008], &pC[1120], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(28, &pC[1152], &pC[1008], &pC[1264], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(1, &pC[1120], 36, 2, &pL[800], 36);
	kernel_sgemm_pp_nt_4x1_c99_lib4(32, &pC[1152], &pC[1152], &pC[1280], 4, -1);
	corner_spotrf_strsv_scopy_1x1_c99_lib4(&pC[1280], 36, 2, &pL[960], 36);
	
	/* middle stages */
	for(ii=0; ii<N-1; ii++)
		{
		
		/* strmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+160;
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[16], &pB[160], &pC[16], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[32], &pB[320], &pC[32], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[48], &pB[480], &pC[48], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[64], &pB[640], &pC[64], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[80], &pB[800], &pC[80], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pB[0], &pC[144], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[160], &pB[160], &pC[160], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[176], &pB[320], &pC[176], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[192], &pB[480], &pC[192], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[208], &pB[640], &pC[208], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[224], &pB[800], &pC[224], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pB[0], &pC[288], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[304], &pB[160], &pC[304], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[320], &pB[320], &pC[320], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[336], &pB[480], &pC[336], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[352], &pB[640], &pC[352], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[368], &pB[800], &pC[368], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pB[0], &pC[432], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[448], &pB[160], &pC[448], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[464], &pB[320], &pC[464], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[480], &pB[480], &pC[480], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[496], &pB[640], &pC[496], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[512], &pB[800], &pC[512], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pB[0], &pC[576], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[592], &pB[160], &pC[592], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[608], &pB[320], &pC[608], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[624], &pB[480], &pC[624], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[640], &pB[640], &pC[640], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[656], &pB[800], &pC[656], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pB[0], &pC[720], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[736], &pB[160], &pC[736], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[752], &pB[320], &pC[752], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[768], &pB[480], &pC[768], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[784], &pB[640], &pC[784], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[800], &pB[800], &pC[800], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pB[0], &pC[864], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[880], &pB[160], &pC[880], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[896], &pB[320], &pC[896], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[912], &pB[480], &pC[912], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[928], &pB[640], &pC[928], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[944], &pB[800], &pC[944], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pB[0], &pC[1008], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[1024], &pB[160], &pC[1024], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[1040], &pB[320], &pC[1040], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[1056], &pB[480], &pC[1056], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[1072], &pB[640], &pC[1072], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[1088], &pB[800], &pC[1088], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pB[0], &pC[1152], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[1168], &pB[160], &pC[1168], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[1184], &pB[320], &pC[1184], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[1200], &pB[480], &pC[1200], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[1216], &pB[640], &pC[1216], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[1232], &pB[800], &pC[1232], 4);
	pC[1152] += pB[88];
	pC[1156] += pB[89];
	pC[1160] += pB[90];
	pC[1164] += pB[91];
	pC[1168] += pB[232];
	pC[1172] += pB[233];
	pC[1176] += pB[234];
	pC[1180] += pB[235];
	pC[1184] += pB[376];
	pC[1188] += pB[377];
	pC[1192] += pB[378];
	pC[1196] += pB[379];
	pC[1200] += pB[520];
	pC[1204] += pB[521];
	pC[1208] += pB[522];
	pC[1212] += pB[523];
	pC[1216] += pB[664];
	pC[1220] += pB[665];
	pC[1224] += pB[666];
	pC[1228] += pB[667];
	pC[1232] += pB[808];
	pC[1236] += pB[809];
		
		/* ssyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pA[0], &pC[144], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pA[144], &pC[160], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pA[0], &pC[288], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pA[144], &pC[304], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pA[288], &pC[320], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[0], &pC[432], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[144], &pC[448], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[288], &pC[464], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[432], &pC[480], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[0], &pC[576], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[144], &pC[592], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[288], &pC[608], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[432], &pC[624], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[576], &pC[640], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[0], &pC[720], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[144], &pC[736], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[288], &pC[752], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[432], &pC[768], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[576], &pC[784], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[720], &pC[800], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[0], &pC[864], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[144], &pC[880], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[288], &pC[896], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[432], &pC[912], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[576], &pC[928], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[720], &pC[944], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[864], &pC[960], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[0], &pC[1008], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[144], &pC[1024], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[288], &pC[1040], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[432], &pC[1056], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[576], &pC[1072], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[720], &pC[1088], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[864], &pC[1104], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[1008], &pC[1120], 4, 1);
	kernel_sgemm_pp_nt_4x1_c99_lib4(22, &pA[1008], &pA[1152], &pC[1136], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[0], &pC[1152], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[144], &pC[1168], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[288], &pC[1184], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[432], &pC[1200], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[576], &pC[1216], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[720], &pC[1232], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[864], &pC[1248], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[1008], &pC[1264], 4, 1);
	kernel_sgemm_pp_nt_4x1_c99_lib4(22, &pA[1152], &pA[1152], &pC[1280], 4, 1);
		
		/* spotrf */
		pC = hpQ[9-ii];
	kernel_spotrf_strsv_4x4_c99_lib4(29, &pC[0], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[144], &pC[144], &pC[160], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[288], &pC[144], &pC[304], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[432], &pC[144], &pC[448], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[576], &pC[144], &pC[592], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[720], &pC[144], &pC[736], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[864], &pC[144], &pC[880], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1008], &pC[144], &pC[1024], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1152], &pC[144], &pC[1168], 4, -1);
	kernel_spotrf_strsv_4x4_c99_lib4(25, &pC[160], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[288], &pC[288], &pC[320], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[432], &pC[288], &pC[464], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[576], &pC[288], &pC[608], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[720], &pC[288], &pC[752], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[864], &pC[288], &pC[896], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[1008], &pC[288], &pC[1040], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(8, &pC[1152], &pC[288], &pC[1184], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(21, &pC[320], 36, 2, &pL[0], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[432], &pC[432], &pC[480], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[576], &pC[432], &pC[624], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[720], &pC[432], &pC[768], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[864], &pC[432], &pC[912], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[1008], &pC[432], &pC[1056], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(12, &pC[1152], &pC[432], &pC[1200], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(17, &pC[480], 36, 2, &pL[160], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[576], &pC[576], &pC[640], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[720], &pC[576], &pC[784], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[864], &pC[576], &pC[928], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[1008], &pC[576], &pC[1072], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(16, &pC[1152], &pC[576], &pC[1216], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(13, &pC[640], 36, 2, &pL[320], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[720], &pC[720], &pC[800], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[864], &pC[720], &pC[944], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[1008], &pC[720], &pC[1088], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(20, &pC[1152], &pC[720], &pC[1232], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(9, &pC[800], 36, 2, &pL[480], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[864], &pC[864], &pC[960], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[1008], &pC[864], &pC[1104], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(24, &pC[1152], &pC[864], &pC[1248], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(5, &pC[960], 36, 2, &pL[640], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(28, &pC[1008], &pC[1008], &pC[1120], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(28, &pC[1152], &pC[1008], &pC[1264], 4, -1);
	kernel_spotrf_strsv_scopy_4x4_c99_lib4(1, &pC[1120], 36, 2, &pL[800], 36);
	kernel_sgemm_pp_nt_4x1_c99_lib4(32, &pC[1152], &pC[1152], &pC[1280], 4, -1);
	corner_spotrf_strsv_scopy_1x1_c99_lib4(&pC[1280], 36, 2, &pL[960], 36);
		
		}
	
	/* initial stage */
	
		/* strmm */
		pA = hpBAbt[9-ii];
		pB = pL;
		pC = pBAbtL;
	pB = pB+160;
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[0], &pB[0], &pC[0], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[16], &pB[160], &pC[16], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[32], &pB[320], &pC[32], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[48], &pB[480], &pC[48], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[64], &pB[640], &pC[64], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[80], &pB[800], &pC[80], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pB[0], &pC[144], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[160], &pB[160], &pC[160], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[176], &pB[320], &pC[176], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[192], &pB[480], &pC[192], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[208], &pB[640], &pC[208], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[224], &pB[800], &pC[224], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pB[0], &pC[288], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[304], &pB[160], &pC[304], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[320], &pB[320], &pC[320], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[336], &pB[480], &pC[336], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[352], &pB[640], &pC[352], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[368], &pB[800], &pC[368], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pB[0], &pC[432], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[448], &pB[160], &pC[448], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[464], &pB[320], &pC[464], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[480], &pB[480], &pC[480], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[496], &pB[640], &pC[496], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[512], &pB[800], &pC[512], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pB[0], &pC[576], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[592], &pB[160], &pC[592], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[608], &pB[320], &pC[608], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[624], &pB[480], &pC[624], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[640], &pB[640], &pC[640], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[656], &pB[800], &pC[656], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pB[0], &pC[720], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[736], &pB[160], &pC[736], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[752], &pB[320], &pC[752], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[768], &pB[480], &pC[768], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[784], &pB[640], &pC[784], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[800], &pB[800], &pC[800], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pB[0], &pC[864], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[880], &pB[160], &pC[880], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[896], &pB[320], &pC[896], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[912], &pB[480], &pC[912], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[928], &pB[640], &pC[928], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[944], &pB[800], &pC[944], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pB[0], &pC[1008], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[1024], &pB[160], &pC[1024], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[1040], &pB[320], &pC[1040], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[1056], &pB[480], &pC[1056], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[1072], &pB[640], &pC[1072], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[1088], &pB[800], &pC[1088], 4);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pB[0], &pC[1152], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(18, &pA[1168], &pB[160], &pC[1168], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(14, &pA[1184], &pB[320], &pC[1184], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(10, &pA[1200], &pB[480], &pC[1200], 4, 0);
	kernel_sgemm_pp_nt_4x4_atom_lib4(6, &pA[1216], &pB[640], &pC[1216], 4, 0);
	corner_strmm_pp_nt_4x2_c99_lib4(&pA[1232], &pB[800], &pC[1232], 4);
	pC[1152] += pB[88];
	pC[1156] += pB[89];
	pC[1160] += pB[90];
	pC[1164] += pB[91];
	pC[1168] += pB[232];
	pC[1172] += pB[233];
	pC[1176] += pB[234];
	pC[1180] += pB[235];
	pC[1184] += pB[376];
	pC[1188] += pB[377];
	pC[1192] += pB[378];
	pC[1196] += pB[379];
	pC[1200] += pB[520];
	pC[1204] += pB[521];
	pC[1208] += pB[522];
	pC[1212] += pB[523];
	pC[1216] += pB[664];
	pC[1220] += pB[665];
	pC[1224] += pB[666];
	pC[1228] += pB[667];
	pC[1232] += pB[808];
	pC[1236] += pB[809];
		
		/* ssyrk */
		pA = pBAbtL;
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[0], &pA[0], &pC[0], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pA[0], &pC[144], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[144], &pA[144], &pC[160], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[144], &pA[288], &pC[176], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pA[0], &pC[288], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[288], &pA[144], &pC[304], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[288], &pA[288], &pC[320], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[0], &pC[432], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[432], &pA[144], &pC[448], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[432], &pA[288], &pC[464], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[0], &pC[576], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[576], &pA[144], &pC[592], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[576], &pA[288], &pC[608], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[0], &pC[720], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[720], &pA[144], &pC[736], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[720], &pA[288], &pC[752], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[0], &pC[864], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[864], &pA[144], &pC[880], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[864], &pA[288], &pC[896], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[0], &pC[1008], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1008], &pA[144], &pC[1024], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[1008], &pA[288], &pC[1040], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[0], &pC[1152], 4, 1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(22, &pA[1152], &pA[144], &pC[1168], 4, 1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(22, &pA[1152], &pA[288], &pC[1184], 4, 1);
		
		/* dpotrf */
		pC = hpQ[9-ii];
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[0], &pC[0], &pC[0], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[144], &pC[0], &pC[144], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[288], &pC[0], &pC[288], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[432], &pC[0], &pC[432], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[576], &pC[0], &pC[576], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[720], &pC[0], &pC[720], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[864], &pC[0], &pC[864], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[1008], &pC[0], &pC[1008], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(0, &pC[1152], &pC[0], &pC[1152], 4, -1);
	kernel_spotrf_strsv_4x4_c99_lib4(29, &pC[0], 36);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[144], &pC[144], &pC[160], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[288], &pC[144], &pC[304], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[432], &pC[144], &pC[448], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[576], &pC[144], &pC[592], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[720], &pC[144], &pC[736], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[864], &pC[144], &pC[880], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1008], &pC[144], &pC[1024], 4, -1);
	kernel_sgemm_pp_nt_4x4_atom_lib4(4, &pC[1152], &pC[144], &pC[1168], 4, -1);
	kernel_spotrf_strsv_4x4_c99_lib4(25, &pC[160], 36);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[288], &pC[288], &pC[320], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[432], &pC[288], &pC[464], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[576], &pC[288], &pC[608], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[720], &pC[288], &pC[752], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[864], &pC[288], &pC[896], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[1008], &pC[288], &pC[1040], 4, -1);
	kernel_sgemm_pp_nt_4x2_c99_lib4(8, &pC[1152], &pC[288], &pC[1184], 4, -1);
	kernel_spotrf_strsv_2x2_c99_lib4(23, &pC[320], 36);
	
	
	
	/* forward substitution */
	for(ii=0; ii<N; ii++)
		{
		
		/* */
		for(jj=0; jj<7; jj+=4)
			{
			hux[ii][jj+0] = hpQ[ii][1152+4*(jj+0)];
			hux[ii][jj+1] = hpQ[ii][1152+4*(jj+1)];
			hux[ii][jj+2] = hpQ[ii][1152+4*(jj+2)];
			hux[ii][jj+3] = hpQ[ii][1152+4*(jj+3)];
			}
		hux[ii][8] = hpQ[ii][1184];
		hux[ii][9] = hpQ[ii][1188];
		
		/* sgemv */
		pA = &hpQ[ii][290];
		x = &hux[ii][10];
		y = &hux[ii][0];
	kernel_sgemv_t_4_atom_lib4(22, 2, pA+0, 36, x, y+0, 1);
	kernel_sgemv_t_4_atom_lib4(22, 2, pA+16, 36, x, y+4, 1);
	kernel_sgemv_t_2_sse_lib4(22, 2, pA+32, 36, x, y+8, 1);
		
		/* strsv */
		pA = hpQ[ii];
		x = &hux[ii][0];
	ptrA = pA + 320;
	ptrx = x + 8;
	kernel_sgemv_t_2_sse_lib4(0, 0, &ptrA[2], 36, &ptrx[2], &ptrx[0], -1);
	ptrx[1] = (ptrx[1]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[1]*ptrx[1]) / ptrA[0];
	ptrA = pA + 160;
	ptrx = x  + 4;
	kernel_sgemv_t_4_atom_lib4(2, 0, ptrA+144, sda, ptrx+4, ptrx, -1);
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
	ptrA = pA + 0;
	ptrx = x  + 0;
	kernel_sgemv_t_4_atom_lib4(6, 0, ptrA+144, sda, ptrx+4, ptrx, -1);
	ptrx[3] = (ptrx[3]) / ptrA[15];
	ptrx[2] = (ptrx[2] - ptrA[11]*ptrx[3]) / ptrA[10];
	ptrx[1] = (ptrx[1] - ptrA[7]*ptrx[3] - ptrA[6]*ptrx[2]) / ptrA[5];
	ptrx[0] = (ptrx[0] - ptrA[3]*ptrx[3] - ptrA[2]*ptrx[2] - ptrA[1]*ptrx[1]) / ptrA[0];
		
		/* */
		for(jj=0; jj<7; jj+=4)
			{
			hux[ii][jj+0] = - hux[ii][jj+0];
			hux[ii][jj+1] = - hux[ii][jj+1];
			hux[ii][jj+2] = - hux[ii][jj+2];
			hux[ii][jj+3] = - hux[ii][jj+3];
			}
		hux[ii][8] = - hux[ii][8];
		hux[ii][9] = - hux[ii][9];
		
		/* */
		for(jj=0; jj<19; jj+=4)
			{
			hux[ii+1][jj+10] = hpBAbt[ii][1152+4*(jj+0)];
			hux[ii+1][jj+11] = hpBAbt[ii][1152+4*(jj+1)];
			hux[ii+1][jj+12] = hpBAbt[ii][1152+4*(jj+2)];
			hux[ii+1][jj+13] = hpBAbt[ii][1152+4*(jj+3)];
			}
		hux[ii+1][30] = hpBAbt[ii][1232];
		hux[ii+1][31] = hpBAbt[ii][1236];
		
		/* sgemv */
		pA = hpBAbt[ii];
		x = &hux[ii][0];
		y = &hux[ii+1][10];
	kernel_sgemv_t_4_atom_lib4(32, 0, pA+0, 36, x, y+0, 1);
	kernel_sgemv_t_4_atom_lib4(32, 0, pA+16, 36, x, y+4, 1);
	kernel_sgemv_t_4_atom_lib4(32, 0, pA+32, 36, x, y+8, 1);
	kernel_sgemv_t_4_atom_lib4(32, 0, pA+48, 36, x, y+12, 1);
	kernel_sgemv_t_4_atom_lib4(32, 0, pA+64, 36, x, y+16, 1);
	kernel_sgemv_t_2_sse_lib4(32, 0, pA+80, 36, x, y+20, 1);
		
		
		}
	
	}
	
