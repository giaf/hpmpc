	.file	"kernel_dtrinv_avx2_lib4.c"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB5:
	.text
.LHOTB5:
	.p2align 4,,15
	.globl	corner_dtrtri_12x4_lib4
	.type	corner_dtrtri_12x4_lib4, @function
corner_dtrtri_12x4_lib4:
.LFB4586:
	.cfi_startproc
	vxorpd	%xmm0, %xmm0, %xmm0
	pushq	%r10
	.cfi_def_cfa_offset 16
	.cfi_offset 10, -16
	sall	$2, %esi
	movq	24(%rsp), %r11
	sall	$2, %r8d
	vbroadcastsd	32(%rdx), %ymm5
	movslq	%esi, %rsi
	vblendpd	$1, (%rdi), %ymm0, %ymm1
	vblendpd	$3, 32(%rdi), %ymm0, %ymm4
	leaq	(%rdi,%rsi,8), %rsi
	vbroadcastsd	(%rdx), %ymm2
	leaq	16(%rsp), %r10
	vbroadcastsd	8(%rdx), %ymm9
	vblendpd	$3, 160(%rsi), %ymm0, %ymm10
	vfnmadd132pd	%ymm1, %ymm0, %ymm2
	vfnmadd231pd	%ymm5, %ymm4, %ymm2
	vbroadcastsd	40(%rdx), %ymm5
	vfnmadd132pd	%ymm1, %ymm0, %ymm9
	vbroadcastsd	16(%rdx), %ymm8
	movslq	%r8d, %r8
	vfnmadd231pd	%ymm5, %ymm4, %ymm9
	vbroadcastsd	48(%rdx), %ymm5
	salq	$3, %r8
	vfnmadd132pd	%ymm1, %ymm0, %ymm8
	vbroadcastsd	24(%rdx), %ymm3
	leaq	(%rcx,%r8), %rax
	vfnmadd231pd	%ymm5, %ymm4, %ymm8
	vbroadcastsd	56(%rdx), %ymm5
	vfnmadd132pd	%ymm1, %ymm0, %ymm3
	vmovapd	96(%rdi), %ymm1
	addq	%rax, %r8
	vfnmadd132pd	%ymm5, %ymm3, %ymm4
	vblendpd	$7, 64(%rdi), %ymm0, %ymm3
	vbroadcastsd	64(%rdx), %ymm5
	vbroadcastsd	120(%rdx), %ymm6
	vfnmadd231pd	%ymm5, %ymm3, %ymm2
	vbroadcastsd	72(%rdx), %ymm5
	vbroadcastsd	152(%rdx), %ymm12
	vfnmadd231pd	%ymm5, %ymm3, %ymm9
	vbroadcastsd	80(%rdx), %ymm5
	vbroadcastsd	136(%rdx), %ymm14
	vmovapd	%ymm5, %ymm7
	vbroadcastsd	88(%rdx), %ymm5
	vfnmadd132pd	%ymm3, %ymm8, %ymm7
	vfnmadd132pd	%ymm5, %ymm4, %ymm3
	vbroadcastsd	96(%rdx), %ymm5
	vbroadcastsd	112(%rdx), %ymm8
	vfnmadd231pd	%ymm6, %ymm1, %ymm3
	vfnmadd231pd	%ymm5, %ymm1, %ymm2
	vbroadcastsd	104(%rdx), %ymm5
	vmovapd	128(%rdi), %ymm6
	vfnmadd132pd	%ymm1, %ymm7, %ymm8
	vfnmadd231pd	%ymm5, %ymm1, %ymm9
	vbroadcastsd	144(%rdx), %ymm7
	vblendpd	$1, 128(%rsi), %ymm0, %ymm1
	vbroadcastsd	128(%rdx), %ymm5
	vfnmadd231pd	%ymm14, %ymm6, %ymm9
	vbroadcastsd	160(%rdx), %ymm4
	vfnmadd231pd	%ymm7, %ymm6, %ymm8
	vfnmadd132pd	%ymm1, %ymm0, %ymm14
	vfnmadd231pd	%ymm5, %ymm6, %ymm2
	vfnmadd132pd	%ymm12, %ymm3, %ymm6
	vmovapd	160(%rdi), %ymm3
	vfnmadd132pd	%ymm1, %ymm0, %ymm5
	vfnmadd132pd	%ymm1, %ymm0, %ymm7
	vfnmadd231pd	%ymm4, %ymm10, %ymm5
	vfnmadd132pd	%ymm12, %ymm0, %ymm1
	vfnmadd231pd	%ymm4, %ymm3, %ymm2
	vbroadcastsd	168(%rdx), %ymm4
	vbroadcastsd	176(%rdx), %ymm11
	vbroadcastsd	184(%rdx), %ymm12
	vfnmadd231pd	%ymm4, %ymm3, %ymm9
	vfnmadd231pd	%ymm4, %ymm10, %ymm14
	vfnmadd231pd	%ymm11, %ymm10, %ymm7
	vbroadcastsd	192(%rdx), %ymm4
	vfnmadd231pd	%ymm11, %ymm3, %ymm8
	vfnmadd132pd	%ymm12, %ymm6, %ymm3
	vfnmadd132pd	%ymm10, %ymm1, %ymm12
	vmovapd	192(%rdi), %ymm6
	vblendpd	$7, 192(%rsi), %ymm0, %ymm10
	vbroadcastsd	208(%rdx), %ymm11
	vfnmadd231pd	%ymm4, %ymm6, %ymm2
	vmovapd	%ymm2, %ymm15
	vmovapd	224(%rdi), %ymm2
	vfnmadd231pd	%ymm4, %ymm10, %ymm5
	vbroadcastsd	200(%rdx), %ymm4
	vfnmadd231pd	%ymm11, %ymm6, %ymm8
	vfnmadd132pd	%ymm10, %ymm7, %ymm11
	vmovapd	%ymm8, %ymm0
	vmovapd	%ymm2, %ymm7
	vfnmadd231pd	%ymm4, %ymm6, %ymm9
	vfnmadd231pd	%ymm4, %ymm10, %ymm14
	vbroadcastsd	216(%rdx), %ymm4
	vmovapd	%ymm9, %ymm1
	vfnmadd132pd	%ymm4, %ymm3, %ymm6
	vfnmadd132pd	%ymm4, %ymm12, %ymm10
	vmovapd	224(%rsi), %ymm12
	vbroadcastsd	224(%rdx), %ymm4
	vbroadcastsd	240(%rdx), %ymm3
	vfnmadd231pd	%ymm4, %ymm2, %ymm15
	vfnmadd231pd	%ymm4, %ymm12, %ymm5
	vbroadcastsd	232(%rdx), %ymm4
	vfnmadd231pd	%ymm3, %ymm2, %ymm0
	vfnmadd231pd	%ymm3, %ymm12, %ymm11
	vmovapd	%ymm12, %ymm3
	vfnmadd231pd	%ymm4, %ymm2, %ymm1
	vfnmadd231pd	%ymm4, %ymm12, %ymm14
	vbroadcastsd	248(%rdx), %ymm4
	movl	(%r10), %edx
	vfnmadd132pd	%ymm4, %ymm6, %ymm7
	vfnmadd132pd	%ymm4, %ymm10, %ymm3
	testl	%edx, %edx
	jne	.L6
	vmovapd	.LC0(%rip), %ymm2
	vbroadcastsd	(%r9), %ymm6
	vbroadcastsd	40(%r9), %ymm12
	vbroadcastsd	80(%r9), %ymm9
	vbroadcastsd	120(%r9), %ymm8
	vdivpd	%ymm6, %ymm2, %ymm13
	vdivpd	%ymm12, %ymm2, %ymm12
	vdivpd	%ymm9, %ymm2, %ymm10
	vdivpd	%ymm8, %ymm2, %ymm8
.L3:
	vmulpd	%ymm15, %ymm13, %ymm2
	vmulpd	%ymm5, %ymm13, %ymm5
	vmulpd	.LC1(%rip), %ymm13, %ymm6
	vmovapd	%ymm2, (%rcx)
	vmovapd	%ymm5, (%rax)
	vmovapd	(%r8), %ymm4
	vblendpd	$1, %ymm6, %ymm4, %ymm4
	vmovapd	%ymm4, (%r8)
	vmovapd	%ymm14, %ymm4
	vbroadcastsd	8(%r9), %ymm9
	vfnmadd231pd	%ymm9, %ymm5, %ymm4
	vfnmadd231pd	%ymm9, %ymm2, %ymm1
	vmulpd	%ymm4, %ymm12, %ymm14
	vfnmadd213pd	.LC2(%rip), %ymm6, %ymm9
	vmulpd	%ymm1, %ymm12, %ymm1
	vmulpd	%ymm9, %ymm12, %ymm12
	vmovapd	%ymm1, 32(%rcx)
	vmovapd	%ymm14, 32(%rax)
	vmovapd	32(%r8), %ymm4
	vblendpd	$3, %ymm12, %ymm4, %ymm4
	vmovapd	%ymm4, 32(%r8)
	vbroadcastsd	48(%r9), %ymm13
	vbroadcastsd	16(%r9), %ymm9
	vmovapd	%ymm13, %ymm4
	vfnmadd231pd	%ymm9, %ymm2, %ymm0
	vfnmadd231pd	%ymm9, %ymm5, %ymm11
	vfnmadd231pd	%ymm13, %ymm1, %ymm0
	vfnmadd231pd	%ymm13, %ymm14, %ymm11
	vmulpd	%ymm0, %ymm10, %ymm0
	vfnmadd213pd	.LC3(%rip), %ymm6, %ymm9
	vfnmadd132pd	%ymm12, %ymm9, %ymm4
	vmulpd	%ymm11, %ymm10, %ymm11
	vmulpd	%ymm4, %ymm10, %ymm9
	vmovapd	%ymm0, 64(%rcx)
	vmovapd	%ymm11, 64(%rax)
	vmovapd	64(%r8), %ymm4
	vblendpd	$7, %ymm9, %ymm4, %ymm4
	vmovapd	%ymm4, 64(%r8)
	vbroadcastsd	24(%r9), %ymm15
	vbroadcastsd	56(%r9), %ymm13
	vbroadcastsd	88(%r9), %ymm10
	vfnmadd132pd	%ymm15, %ymm7, %ymm2
	vfnmadd132pd	%ymm15, %ymm3, %ymm5
	vfnmadd132pd	%ymm13, %ymm2, %ymm1
	vmovapd	%ymm5, %ymm4
	vmovapd	%ymm12, %ymm5
	vfnmadd231pd	%ymm10, %ymm0, %ymm1
	vmulpd	%ymm1, %ymm8, %ymm0
	vfnmadd213pd	.LC4(%rip), %ymm15, %ymm6
	vmovapd	%ymm11, %ymm3
	vfnmadd132pd	%ymm13, %ymm6, %ymm5
	vfnmadd231pd	%ymm13, %ymm14, %ymm4
	vmovapd	%ymm9, %ymm6
	vfnmadd132pd	%ymm10, %ymm4, %ymm3
	vfnmadd132pd	%ymm10, %ymm5, %ymm6
	vmulpd	%ymm6, %ymm8, %ymm6
	vmovapd	%ymm0, 96(%rcx)
	vmulpd	%ymm3, %ymm8, %ymm0
	vmovapd	%ymm0, 96(%rax)
	vmovapd	%ymm6, 96(%r8)
	vzeroupper
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L6:
	.cfi_restore_state
	vbroadcastsd	(%r11), %ymm13
	vbroadcastsd	8(%r11), %ymm12
	vbroadcastsd	16(%r11), %ymm10
	vbroadcastsd	24(%r11), %ymm8
	jmp	.L3
	.cfi_endproc
.LFE4586:
	.size	corner_dtrtri_12x4_lib4, .-corner_dtrtri_12x4_lib4
	.section	.text.unlikely
.LCOLDE5:
	.text
.LHOTE5:
	.section	.text.unlikely
.LCOLDB6:
	.text
.LHOTB6:
	.p2align 4,,15
	.globl	corner_dtrtri_8x8_lib4
	.type	corner_dtrtri_8x8_lib4, @function
corner_dtrtri_8x8_lib4:
.LFB4587:
	.cfi_startproc
	sall	$2, %esi
	sall	$2, %r9d
	testl	%edx, %edx
	movslq	%esi, %rsi
	movslq	%r9d, %r9
	leaq	(%rdi,%rsi,8), %rax
	leaq	(%r8,%r9,8), %rsi
	je	.L8
	vbroadcastsd	(%rcx), %ymm0
	vbroadcastsd	8(%rcx), %ymm2
	vbroadcastsd	16(%rcx), %ymm4
	vbroadcastsd	24(%rcx), %ymm5
.L9:
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovapd	(%r8), %ymm7
	testl	%edx, %edx
	vmovapd	.LC3(%rip), %ymm11
	vmovapd	.LC4(%rip), %ymm10
	vblendpd	$1, %ymm0, %ymm6, %ymm0
	vblendpd	$1, %ymm0, %ymm7, %ymm7
	vmovapd	%ymm7, (%r8)
	vmovapd	.LC2(%rip), %ymm7
	vbroadcastsd	8(%rdi), %ymm8
	vfnmadd132pd	%ymm0, %ymm7, %ymm8
	vmulpd	%ymm8, %ymm2, %ymm2
	vmovapd	32(%r8), %ymm8
	vblendpd	$3, %ymm2, %ymm8, %ymm8
	vmovapd	%ymm8, 32(%r8)
	vbroadcastsd	48(%rdi), %ymm8
	vbroadcastsd	16(%rdi), %ymm1
	vfnmadd132pd	%ymm0, %ymm11, %ymm1
	vfnmadd231pd	%ymm8, %ymm2, %ymm1
	vmulpd	%ymm1, %ymm4, %ymm4
	vmovapd	64(%r8), %ymm1
	vblendpd	$7, %ymm4, %ymm1, %ymm1
	vmovapd	%ymm1, 64(%r8)
	vbroadcastsd	56(%rdi), %ymm8
	vbroadcastsd	88(%rdi), %ymm1
	vbroadcastsd	24(%rdi), %ymm3
	vfnmadd132pd	%ymm0, %ymm10, %ymm3
	vfnmadd231pd	%ymm8, %ymm2, %ymm3
	vfnmadd231pd	%ymm1, %ymm4, %ymm3
	vmulpd	%ymm3, %ymm5, %ymm3
	vmovapd	%ymm3, 96(%r8)
	vbroadcastsd	24(%rax), %ymm1
	vbroadcastsd	(%rax), %ymm12
	vbroadcastsd	8(%rax), %ymm9
	vbroadcastsd	16(%rax), %ymm5
	vfnmadd132pd	%ymm0, %ymm6, %ymm12
	vfnmadd132pd	%ymm0, %ymm6, %ymm9
	vbroadcastsd	56(%rax), %ymm8
	vfnmadd132pd	%ymm0, %ymm6, %ymm5
	vfnmadd132pd	%ymm1, %ymm6, %ymm0
	vbroadcastsd	32(%rax), %ymm1
	vbroadcastsd	40(%rax), %ymm6
	vfnmadd231pd	%ymm1, %ymm2, %ymm12
	vbroadcastsd	48(%rax), %ymm1
	vfnmadd231pd	%ymm6, %ymm2, %ymm9
	vbroadcastsd	104(%rax), %ymm6
	vfnmadd231pd	%ymm1, %ymm2, %ymm5
	vfnmadd132pd	%ymm8, %ymm0, %ymm2
	vbroadcastsd	64(%rax), %ymm8
	vbroadcastsd	96(%rax), %ymm1
	vfnmadd231pd	%ymm8, %ymm4, %ymm12
	vbroadcastsd	72(%rax), %ymm8
	vfnmadd231pd	%ymm1, %ymm3, %ymm12
	vbroadcastsd	112(%rax), %ymm0
	vfnmadd231pd	%ymm8, %ymm4, %ymm9
	vbroadcastsd	80(%rax), %ymm8
	vfnmadd132pd	%ymm3, %ymm9, %ymm6
	vbroadcastsd	120(%rax), %ymm1
	vfnmadd231pd	%ymm8, %ymm4, %ymm5
	vbroadcastsd	88(%rax), %ymm8
	vfnmadd231pd	%ymm0, %ymm3, %ymm5
	vfnmadd132pd	%ymm8, %ymm2, %ymm4
	vfnmadd132pd	%ymm1, %ymm4, %ymm3
	jne	.L12
	vmovapd	.LC0(%rip), %ymm0
	vbroadcastsd	128(%rax), %ymm2
	vbroadcastsd	168(%rax), %ymm9
	vbroadcastsd	208(%rax), %ymm4
	vbroadcastsd	248(%rax), %ymm8
	vdivpd	%ymm2, %ymm0, %ymm2
	vdivpd	%ymm9, %ymm0, %ymm9
	vdivpd	%ymm4, %ymm0, %ymm4
	vdivpd	%ymm8, %ymm0, %ymm8
.L11:
	vmulpd	%ymm12, %ymm2, %ymm1
	vmulpd	.LC1(%rip), %ymm2, %ymm13
	vmovapd	%ymm1, 128(%r8)
	vmovapd	128(%rsi), %ymm0
	vblendpd	$1, %ymm13, %ymm0, %ymm0
	vmovapd	%ymm0, 128(%rsi)
	vbroadcastsd	136(%rax), %ymm2
	vfnmadd231pd	%ymm2, %ymm1, %ymm6
	vmulpd	%ymm6, %ymm9, %ymm6
	vfnmadd132pd	%ymm13, %ymm7, %ymm2
	vmulpd	%ymm2, %ymm9, %ymm7
	vmovapd	%ymm6, 160(%r8)
	vmovapd	160(%rsi), %ymm0
	vblendpd	$3, %ymm7, %ymm0, %ymm0
	vmovapd	%ymm0, 160(%rsi)
	vbroadcastsd	144(%rax), %ymm2
	vbroadcastsd	176(%rax), %ymm9
	vfnmadd231pd	%ymm2, %ymm1, %ymm5
	vfnmadd132pd	%ymm13, %ymm11, %ymm2
	vfnmadd231pd	%ymm9, %ymm6, %ymm5
	vmulpd	%ymm5, %ymm4, %ymm0
	vfnmadd132pd	%ymm7, %ymm2, %ymm9
	vmulpd	%ymm9, %ymm4, %ymm5
	vmovapd	%ymm0, 192(%r8)
	vmovapd	192(%rsi), %ymm2
	vblendpd	$7, %ymm5, %ymm2, %ymm2
	vmovapd	%ymm2, 192(%rsi)
	vbroadcastsd	152(%rax), %ymm12
	vbroadcastsd	184(%rax), %ymm11
	vbroadcastsd	216(%rax), %ymm9
	vfnmadd132pd	%ymm12, %ymm3, %ymm1
	vfnmadd231pd	%ymm12, %ymm13, %ymm10
	vfnmadd231pd	%ymm11, %ymm6, %ymm1
	vfnmadd132pd	%ymm11, %ymm10, %ymm7
	vfnmadd132pd	%ymm9, %ymm1, %ymm0
	vfnmadd132pd	%ymm9, %ymm7, %ymm5
	vmulpd	%ymm0, %ymm8, %ymm0
	vmulpd	%ymm5, %ymm8, %ymm8
	vmovapd	%ymm0, 224(%r8)
	vmovapd	%ymm8, 224(%rsi)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L8:
	vmovapd	.LC0(%rip), %ymm1
	vbroadcastsd	(%rdi), %ymm0
	vbroadcastsd	40(%rdi), %ymm2
	vbroadcastsd	80(%rdi), %ymm4
	vbroadcastsd	120(%rdi), %ymm3
	vdivpd	%ymm0, %ymm1, %ymm0
	vdivpd	%ymm2, %ymm1, %ymm2
	vdivpd	%ymm4, %ymm1, %ymm4
	vdivpd	%ymm3, %ymm1, %ymm5
	jmp	.L9
	.p2align 4,,10
	.p2align 3
.L12:
	vbroadcastsd	32(%rcx), %ymm2
	vbroadcastsd	40(%rcx), %ymm9
	vbroadcastsd	48(%rcx), %ymm4
	vbroadcastsd	56(%rcx), %ymm8
	jmp	.L11
	.cfi_endproc
.LFE4587:
	.size	corner_dtrtri_8x8_lib4, .-corner_dtrtri_8x8_lib4
	.section	.text.unlikely
.LCOLDE6:
	.text
.LHOTE6:
	.section	.text.unlikely
.LCOLDB7:
	.text
.LHOTB7:
	.p2align 4,,15
	.globl	corner_dtrtri_7x7_lib4
	.type	corner_dtrtri_7x7_lib4, @function
corner_dtrtri_7x7_lib4:
.LFB4588:
	.cfi_startproc
	sall	$2, %esi
	sall	$2, %r9d
	testl	%edx, %edx
	movslq	%esi, %rsi
	movslq	%r9d, %r9
	leaq	(%rdi,%rsi,8), %rax
	leaq	(%r8,%r9,8), %rsi
	je	.L14
	vbroadcastsd	(%rcx), %ymm2
	vbroadcastsd	8(%rcx), %ymm4
	vbroadcastsd	16(%rcx), %ymm6
	vbroadcastsd	24(%rcx), %ymm8
.L15:
	vxorpd	%xmm7, %xmm7, %xmm7
	vmovapd	(%r8), %ymm1
	testl	%edx, %edx
	vmovapd	.LC2(%rip), %ymm9
	vmovapd	.LC3(%rip), %ymm11
	vblendpd	$1, %ymm2, %ymm7, %ymm2
	vblendpd	$1, %ymm2, %ymm1, %ymm1
	vmovapd	%ymm1, (%r8)
	vbroadcastsd	8(%rdi), %ymm1
	vfnmadd132pd	%ymm2, %ymm9, %ymm1
	vmulpd	%ymm1, %ymm4, %ymm4
	vmovapd	32(%r8), %ymm1
	vblendpd	$3, %ymm4, %ymm1, %ymm1
	vmovapd	%ymm1, 32(%r8)
	vbroadcastsd	48(%rdi), %ymm1
	vbroadcastsd	16(%rdi), %ymm0
	vfnmadd132pd	%ymm2, %ymm11, %ymm0
	vfnmadd231pd	%ymm1, %ymm4, %ymm0
	vmulpd	%ymm0, %ymm6, %ymm6
	vmovapd	64(%r8), %ymm0
	vblendpd	$7, %ymm6, %ymm0, %ymm0
	vmovapd	%ymm0, 64(%r8)
	vbroadcastsd	56(%rdi), %ymm1
	vbroadcastsd	88(%rdi), %ymm0
	vbroadcastsd	24(%rdi), %ymm5
	vfnmadd213pd	.LC4(%rip), %ymm2, %ymm5
	vfnmadd231pd	%ymm1, %ymm4, %ymm5
	vfnmadd231pd	%ymm0, %ymm6, %ymm5
	vmulpd	%ymm5, %ymm8, %ymm5
	vmovapd	%ymm5, 96(%r8)
	vbroadcastsd	16(%rax), %ymm8
	vbroadcastsd	(%rax), %ymm10
	vbroadcastsd	8(%rax), %ymm1
	vfnmadd132pd	%ymm2, %ymm7, %ymm10
	vbroadcastsd	96(%rax), %ymm3
	vfnmadd132pd	%ymm2, %ymm7, %ymm1
	vfnmadd132pd	%ymm8, %ymm7, %ymm2
	vbroadcastsd	32(%rax), %ymm8
	vbroadcastsd	104(%rax), %ymm0
	vfnmadd231pd	%ymm8, %ymm4, %ymm10
	vbroadcastsd	40(%rax), %ymm8
	vfnmadd231pd	%ymm8, %ymm4, %ymm1
	vbroadcastsd	48(%rax), %ymm8
	vfnmadd132pd	%ymm8, %ymm2, %ymm4
	vbroadcastsd	64(%rax), %ymm8
	vfnmadd231pd	%ymm8, %ymm6, %ymm10
	vbroadcastsd	72(%rax), %ymm8
	vfnmadd231pd	%ymm3, %ymm5, %ymm10
	vbroadcastsd	112(%rax), %ymm3
	vfnmadd231pd	%ymm8, %ymm6, %ymm1
	vbroadcastsd	80(%rax), %ymm8
	vfnmadd231pd	%ymm0, %ymm5, %ymm1
	vfnmadd132pd	%ymm8, %ymm4, %ymm6
	vfnmadd132pd	%ymm3, %ymm6, %ymm5
	jne	.L18
	vmovapd	.LC0(%rip), %ymm0
	vbroadcastsd	128(%rax), %ymm12
	vbroadcastsd	168(%rax), %ymm7
	vbroadcastsd	208(%rax), %ymm8
	vdivpd	%ymm12, %ymm0, %ymm12
	vdivpd	%ymm7, %ymm0, %ymm7
	vdivpd	%ymm8, %ymm0, %ymm8
.L17:
	vmulpd	%ymm10, %ymm12, %ymm3
	vmulpd	.LC1(%rip), %ymm12, %ymm13
	vmovapd	%ymm3, 128(%r8)
	vmovapd	128(%rsi), %ymm0
	vblendpd	$1, %ymm13, %ymm0, %ymm0
	vmovapd	%ymm0, 128(%rsi)
	vbroadcastsd	136(%rax), %ymm10
	vfnmadd231pd	%ymm10, %ymm3, %ymm1
	vmulpd	%ymm1, %ymm7, %ymm0
	vfnmadd132pd	%ymm13, %ymm9, %ymm10
	vmulpd	%ymm10, %ymm7, %ymm1
	vmovapd	%ymm0, 160(%r8)
	vmovapd	160(%rsi), %ymm2
	vblendpd	$3, %ymm1, %ymm2, %ymm2
	vmovapd	%ymm2, 160(%rsi)
	vbroadcastsd	144(%rax), %ymm9
	vbroadcastsd	176(%rax), %ymm7
	vfnmadd132pd	%ymm9, %ymm5, %ymm3
	vfnmadd132pd	%ymm13, %ymm11, %ymm9
	vfnmadd132pd	%ymm7, %ymm3, %ymm0
	vmulpd	%ymm0, %ymm8, %ymm0
	vfnmadd132pd	%ymm7, %ymm9, %ymm1
	vmulpd	%ymm1, %ymm8, %ymm8
	vmovapd	%ymm0, 192(%r8)
	vmovapd	192(%rsi), %ymm0
	vblendpd	$7, %ymm8, %ymm0, %ymm8
	vmovapd	%ymm8, 192(%rsi)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L14:
	vmovapd	.LC0(%rip), %ymm0
	vbroadcastsd	(%rdi), %ymm2
	vbroadcastsd	40(%rdi), %ymm4
	vbroadcastsd	80(%rdi), %ymm6
	vbroadcastsd	120(%rdi), %ymm5
	vdivpd	%ymm2, %ymm0, %ymm2
	vdivpd	%ymm4, %ymm0, %ymm4
	vdivpd	%ymm6, %ymm0, %ymm6
	vdivpd	%ymm5, %ymm0, %ymm8
	jmp	.L15
	.p2align 4,,10
	.p2align 3
.L18:
	vbroadcastsd	32(%rcx), %ymm12
	vbroadcastsd	40(%rcx), %ymm7
	vbroadcastsd	48(%rcx), %ymm8
	jmp	.L17
	.cfi_endproc
.LFE4588:
	.size	corner_dtrtri_7x7_lib4, .-corner_dtrtri_7x7_lib4
	.section	.text.unlikely
.LCOLDE7:
	.text
.LHOTE7:
	.section	.text.unlikely
.LCOLDB8:
	.text
.LHOTB8:
	.p2align 4,,15
	.globl	corner_dtrtri_6x6_lib4
	.type	corner_dtrtri_6x6_lib4, @function
corner_dtrtri_6x6_lib4:
.LFB4589:
	.cfi_startproc
	sall	$2, %esi
	sall	$2, %r9d
	testl	%edx, %edx
	movslq	%esi, %rsi
	movslq	%r9d, %r9
	leaq	(%rdi,%rsi,8), %rax
	leaq	(%r8,%r9,8), %rsi
	je	.L20
	vbroadcastsd	(%rcx), %ymm1
	vbroadcastsd	8(%rcx), %ymm2
	vbroadcastsd	16(%rcx), %ymm4
	vbroadcastsd	24(%rcx), %ymm7
.L21:
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovapd	(%r8), %ymm5
	testl	%edx, %edx
	vmovapd	.LC2(%rip), %ymm8
	vblendpd	$1, %ymm1, %ymm6, %ymm1
	vblendpd	$1, %ymm1, %ymm5, %ymm5
	vmovapd	%ymm5, (%r8)
	vbroadcastsd	8(%rdi), %ymm5
	vfnmadd132pd	%ymm1, %ymm8, %ymm5
	vmulpd	%ymm5, %ymm2, %ymm2
	vmovapd	32(%r8), %ymm5
	vblendpd	$3, %ymm2, %ymm5, %ymm5
	vmovapd	%ymm5, 32(%r8)
	vbroadcastsd	48(%rdi), %ymm5
	vbroadcastsd	16(%rdi), %ymm0
	vfnmadd213pd	.LC3(%rip), %ymm1, %ymm0
	vfnmadd231pd	%ymm5, %ymm2, %ymm0
	vmulpd	%ymm0, %ymm4, %ymm4
	vmovapd	64(%r8), %ymm0
	vblendpd	$7, %ymm4, %ymm0, %ymm0
	vmovapd	%ymm0, 64(%r8)
	vbroadcastsd	56(%rdi), %ymm5
	vbroadcastsd	24(%rdi), %ymm3
	vbroadcastsd	88(%rdi), %ymm0
	vfnmadd213pd	.LC4(%rip), %ymm1, %ymm3
	vfnmadd231pd	%ymm5, %ymm2, %ymm3
	vfnmadd231pd	%ymm0, %ymm4, %ymm3
	vmulpd	%ymm3, %ymm7, %ymm3
	vmovapd	%ymm3, 96(%r8)
	vbroadcastsd	8(%rax), %ymm5
	vbroadcastsd	(%rax), %ymm9
	vbroadcastsd	104(%rax), %ymm7
	vfnmadd132pd	%ymm1, %ymm6, %ymm9
	vfnmadd132pd	%ymm5, %ymm6, %ymm1
	vbroadcastsd	32(%rax), %ymm5
	vfnmadd231pd	%ymm5, %ymm2, %ymm9
	vbroadcastsd	40(%rax), %ymm5
	vfnmadd132pd	%ymm5, %ymm1, %ymm2
	vbroadcastsd	64(%rax), %ymm5
	vfnmadd231pd	%ymm5, %ymm4, %ymm9
	vbroadcastsd	72(%rax), %ymm5
	vfnmadd132pd	%ymm5, %ymm2, %ymm4
	vbroadcastsd	96(%rax), %ymm5
	vfnmadd231pd	%ymm5, %ymm3, %ymm9
	vfnmadd132pd	%ymm7, %ymm4, %ymm3
	jne	.L24
	vmovapd	.LC0(%rip), %ymm6
	vbroadcastsd	128(%rax), %ymm1
	vbroadcastsd	168(%rax), %ymm5
	vdivpd	%ymm1, %ymm6, %ymm1
	vdivpd	%ymm5, %ymm6, %ymm5
.L23:
	vmulpd	%ymm9, %ymm1, %ymm0
	vmulpd	.LC1(%rip), %ymm1, %ymm2
	vmovapd	%ymm0, 128(%r8)
	vmovapd	128(%rsi), %ymm1
	vblendpd	$1, %ymm2, %ymm1, %ymm1
	vmovapd	%ymm1, 128(%rsi)
	vbroadcastsd	136(%rax), %ymm6
	vfnmadd132pd	%ymm6, %ymm3, %ymm0
	vmulpd	%ymm0, %ymm5, %ymm0
	vfnmadd132pd	%ymm2, %ymm8, %ymm6
	vmulpd	%ymm6, %ymm5, %ymm5
	vmovapd	%ymm0, 160(%r8)
	vmovapd	160(%rsi), %ymm0
	vblendpd	$3, %ymm5, %ymm0, %ymm5
	vmovapd	%ymm5, 160(%rsi)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L20:
	vmovapd	.LC0(%rip), %ymm5
	vbroadcastsd	(%rdi), %ymm1
	vbroadcastsd	40(%rdi), %ymm2
	vbroadcastsd	80(%rdi), %ymm4
	vbroadcastsd	120(%rdi), %ymm3
	vdivpd	%ymm1, %ymm5, %ymm1
	vdivpd	%ymm2, %ymm5, %ymm2
	vdivpd	%ymm4, %ymm5, %ymm4
	vdivpd	%ymm3, %ymm5, %ymm7
	jmp	.L21
	.p2align 4,,10
	.p2align 3
.L24:
	vbroadcastsd	32(%rcx), %ymm1
	vbroadcastsd	40(%rcx), %ymm5
	jmp	.L23
	.cfi_endproc
.LFE4589:
	.size	corner_dtrtri_6x6_lib4, .-corner_dtrtri_6x6_lib4
	.section	.text.unlikely
.LCOLDE8:
	.text
.LHOTE8:
	.section	.text.unlikely
.LCOLDB9:
	.text
.LHOTB9:
	.p2align 4,,15
	.globl	corner_dtrtri_5x5_lib4
	.type	corner_dtrtri_5x5_lib4, @function
corner_dtrtri_5x5_lib4:
.LFB4590:
	.cfi_startproc
	sall	$2, %esi
	sall	$2, %r9d
	testl	%edx, %edx
	movslq	%esi, %rsi
	movslq	%r9d, %r9
	leaq	(%rdi,%rsi,8), %rax
	leaq	(%r8,%r9,8), %rsi
	je	.L26
	vbroadcastsd	(%rcx), %ymm0
	vbroadcastsd	8(%rcx), %ymm1
	vbroadcastsd	16(%rcx), %ymm4
	vbroadcastsd	24(%rcx), %ymm2
.L27:
	vxorpd	%xmm7, %xmm7, %xmm7
	vmovapd	(%r8), %ymm3
	testl	%edx, %edx
	vblendpd	$1, %ymm0, %ymm7, %ymm0
	vblendpd	$1, %ymm0, %ymm3, %ymm3
	vmovapd	%ymm3, (%r8)
	vbroadcastsd	8(%rdi), %ymm3
	vfnmadd213pd	.LC2(%rip), %ymm0, %ymm3
	vmulpd	%ymm3, %ymm1, %ymm1
	vmovapd	32(%r8), %ymm3
	vblendpd	$3, %ymm1, %ymm3, %ymm3
	vmovapd	%ymm3, 32(%r8)
	vbroadcastsd	48(%rdi), %ymm5
	vbroadcastsd	16(%rdi), %ymm3
	vfnmadd213pd	.LC3(%rip), %ymm0, %ymm3
	vfnmadd231pd	%ymm5, %ymm1, %ymm3
	vmulpd	%ymm3, %ymm4, %ymm4
	vmovapd	64(%r8), %ymm3
	vblendpd	$7, %ymm4, %ymm3, %ymm3
	vmovapd	%ymm3, 64(%r8)
	vbroadcastsd	24(%rdi), %ymm3
	vbroadcastsd	56(%rdi), %ymm6
	vbroadcastsd	88(%rdi), %ymm5
	vfnmadd213pd	.LC4(%rip), %ymm0, %ymm3
	vfnmadd231pd	%ymm6, %ymm1, %ymm3
	vfnmadd231pd	%ymm5, %ymm4, %ymm3
	vmulpd	%ymm3, %ymm2, %ymm3
	vmovapd	%ymm3, 96(%r8)
	vbroadcastsd	(%rax), %ymm2
	vfnmadd132pd	%ymm2, %ymm7, %ymm0
	vbroadcastsd	32(%rax), %ymm2
	vfnmadd132pd	%ymm2, %ymm0, %ymm1
	vbroadcastsd	64(%rax), %ymm2
	vfnmadd231pd	%ymm2, %ymm4, %ymm1
	vbroadcastsd	96(%rax), %ymm2
	vfnmadd231pd	%ymm2, %ymm3, %ymm1
	jne	.L30
	vbroadcastsd	128(%rax), %ymm0
	vmovapd	.LC0(%rip), %ymm2
	vdivpd	%ymm0, %ymm2, %ymm2
.L29:
	vmulpd	%ymm1, %ymm2, %ymm1
	vmulpd	.LC1(%rip), %ymm2, %ymm2
	vmovapd	%ymm1, 128(%r8)
	vmovapd	128(%rsi), %ymm0
	vblendpd	$1, %ymm2, %ymm0, %ymm2
	vmovapd	%ymm2, 128(%rsi)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L26:
	vmovapd	.LC0(%rip), %ymm2
	vbroadcastsd	(%rdi), %ymm0
	vbroadcastsd	40(%rdi), %ymm1
	vbroadcastsd	80(%rdi), %ymm4
	vbroadcastsd	120(%rdi), %ymm3
	vdivpd	%ymm0, %ymm2, %ymm0
	vdivpd	%ymm1, %ymm2, %ymm1
	vdivpd	%ymm4, %ymm2, %ymm4
	vdivpd	%ymm3, %ymm2, %ymm2
	jmp	.L27
	.p2align 4,,10
	.p2align 3
.L30:
	vbroadcastsd	32(%rcx), %ymm2
	jmp	.L29
	.cfi_endproc
.LFE4590:
	.size	corner_dtrtri_5x5_lib4, .-corner_dtrtri_5x5_lib4
	.section	.text.unlikely
.LCOLDE9:
	.text
.LHOTE9:
	.section	.text.unlikely
.LCOLDB10:
	.text
.LHOTB10:
	.p2align 4,,15
	.globl	corner_dtrtri_4x4_lib4
	.type	corner_dtrtri_4x4_lib4, @function
corner_dtrtri_4x4_lib4:
.LFB4591:
	.cfi_startproc
	testl	%esi, %esi
	jne	.L34
	vmovapd	.LC0(%rip), %ymm4
	vbroadcastsd	(%rdi), %ymm1
	vbroadcastsd	40(%rdi), %ymm0
	vbroadcastsd	80(%rdi), %ymm3
	vbroadcastsd	120(%rdi), %ymm2
	vdivpd	%ymm1, %ymm4, %ymm1
	vdivpd	%ymm0, %ymm4, %ymm5
	vdivpd	%ymm3, %ymm4, %ymm3
	vdivpd	%ymm2, %ymm4, %ymm4
.L33:
	vmovapd	(%rcx), %ymm2
	vxorpd	%xmm0, %xmm0, %xmm0
	vblendpd	$1, %ymm1, %ymm0, %ymm1
	vblendpd	$1, %ymm1, %ymm2, %ymm2
	vmovapd	%ymm2, (%rcx)
	vmovapd	32(%rcx), %ymm2
	vbroadcastsd	8(%rdi), %ymm0
	vfnmadd213pd	.LC2(%rip), %ymm1, %ymm0
	vmulpd	%ymm0, %ymm5, %ymm0
	vblendpd	$3, %ymm0, %ymm2, %ymm2
	vmovapd	%ymm2, 32(%rcx)
	vbroadcastsd	48(%rdi), %ymm5
	vbroadcastsd	16(%rdi), %ymm2
	vfnmadd213pd	.LC3(%rip), %ymm1, %ymm2
	vfnmadd231pd	%ymm5, %ymm0, %ymm2
	vmulpd	%ymm2, %ymm3, %ymm2
	vmovapd	64(%rcx), %ymm3
	vblendpd	$7, %ymm2, %ymm3, %ymm3
	vmovapd	%ymm3, 64(%rcx)
	vbroadcastsd	24(%rdi), %ymm6
	vbroadcastsd	56(%rdi), %ymm5
	vbroadcastsd	88(%rdi), %ymm3
	vfnmadd213pd	.LC4(%rip), %ymm6, %ymm1
	vfnmadd132pd	%ymm5, %ymm1, %ymm0
	vfnmadd132pd	%ymm3, %ymm0, %ymm2
	vmulpd	%ymm2, %ymm4, %ymm2
	vmovapd	%ymm2, 96(%rcx)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L34:
	vbroadcastsd	(%rdx), %ymm1
	vbroadcastsd	8(%rdx), %ymm5
	vbroadcastsd	16(%rdx), %ymm3
	vbroadcastsd	24(%rdx), %ymm4
	jmp	.L33
	.cfi_endproc
.LFE4591:
	.size	corner_dtrtri_4x4_lib4, .-corner_dtrtri_4x4_lib4
	.section	.text.unlikely
.LCOLDE10:
	.text
.LHOTE10:
	.section	.text.unlikely
.LCOLDB11:
	.text
.LHOTB11:
	.p2align 4,,15
	.globl	corner_dtrtri_3x3_lib4
	.type	corner_dtrtri_3x3_lib4, @function
corner_dtrtri_3x3_lib4:
.LFB4592:
	.cfi_startproc
	testl	%esi, %esi
	jne	.L38
	vmovapd	.LC0(%rip), %ymm3
	vbroadcastsd	(%rdi), %ymm2
	vbroadcastsd	40(%rdi), %ymm0
	vbroadcastsd	80(%rdi), %ymm4
	vdivpd	%ymm2, %ymm3, %ymm2
	vdivpd	%ymm0, %ymm3, %ymm1
	vdivpd	%ymm4, %ymm3, %ymm4
.L37:
	vmovapd	(%rcx), %ymm0
	vxorpd	%xmm3, %xmm3, %xmm3
	vblendpd	$1, %ymm2, %ymm3, %ymm2
	vblendpd	$1, %ymm2, %ymm0, %ymm0
	vmovapd	%ymm0, (%rcx)
	vbroadcastsd	8(%rdi), %ymm0
	vfnmadd213pd	.LC2(%rip), %ymm2, %ymm0
	vmulpd	%ymm0, %ymm1, %ymm0
	vmovapd	32(%rcx), %ymm1
	vblendpd	$3, %ymm0, %ymm1, %ymm1
	vmovapd	%ymm1, 32(%rcx)
	vbroadcastsd	48(%rdi), %ymm1
	vbroadcastsd	16(%rdi), %ymm3
	vfnmadd213pd	.LC3(%rip), %ymm3, %ymm2
	vfnmadd132pd	%ymm1, %ymm2, %ymm0
	vmulpd	%ymm0, %ymm4, %ymm0
	vmovapd	64(%rcx), %ymm1
	vblendpd	$7, %ymm0, %ymm1, %ymm0
	vmovapd	%ymm0, 64(%rcx)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L38:
	vbroadcastsd	(%rdx), %ymm2
	vbroadcastsd	8(%rdx), %ymm1
	vbroadcastsd	16(%rdx), %ymm4
	jmp	.L37
	.cfi_endproc
.LFE4592:
	.size	corner_dtrtri_3x3_lib4, .-corner_dtrtri_3x3_lib4
	.section	.text.unlikely
.LCOLDE11:
	.text
.LHOTE11:
	.section	.text.unlikely
.LCOLDB14:
	.text
.LHOTB14:
	.p2align 4,,15
	.globl	corner_dtrtri_2x2_lib4
	.type	corner_dtrtri_2x2_lib4, @function
corner_dtrtri_2x2_lib4:
.LFB4593:
	.cfi_startproc
	testl	%esi, %esi
	jne	.L42
	vmovsd	.LC12(%rip), %xmm0
	vdivsd	(%rdi), %xmm0, %xmm1
	vdivsd	40(%rdi), %xmm0, %xmm0
.L41:
	vmovsd	%xmm0, 40(%rcx)
	vmovsd	%xmm1, (%rcx)
	vmovsd	.LC13(%rip), %xmm2
	vxorpd	%xmm2, %xmm0, %xmm0
	vmulsd	8(%rdi), %xmm0, %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 32(%rcx)
	ret
	.p2align 4,,10
	.p2align 3
.L42:
	vmovsd	(%rdx), %xmm1
	vmovsd	8(%rdx), %xmm0
	jmp	.L41
	.cfi_endproc
.LFE4593:
	.size	corner_dtrtri_2x2_lib4, .-corner_dtrtri_2x2_lib4
	.section	.text.unlikely
.LCOLDE14:
	.text
.LHOTE14:
	.section	.text.unlikely
.LCOLDB15:
	.text
.LHOTB15:
	.p2align 4,,15
	.globl	corner_dtrtri_1x1_lib4
	.type	corner_dtrtri_1x1_lib4, @function
corner_dtrtri_1x1_lib4:
.LFB4594:
	.cfi_startproc
	testl	%esi, %esi
	jne	.L46
	vmovsd	.LC12(%rip), %xmm0
	vdivsd	(%rdi), %xmm0, %xmm0
	vmovsd	%xmm0, (%rcx)
	ret
	.p2align 4,,10
	.p2align 3
.L46:
	vmovsd	(%rdx), %xmm0
	vmovsd	%xmm0, (%rcx)
	ret
	.cfi_endproc
.LFE4594:
	.size	corner_dtrtri_1x1_lib4, .-corner_dtrtri_1x1_lib4
	.section	.text.unlikely
.LCOLDE15:
	.text
.LHOTE15:
	.section	.text.unlikely
.LCOLDB16:
	.text
.LHOTB16:
	.p2align 4,,15
	.globl	kernel_dtrtri_12x4_lib4
	.type	kernel_dtrtri_12x4_lib4, @function
kernel_dtrtri_12x4_lib4:
.LFB4595:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	leal	0(,%rdx,4), %eax
	vxorpd	%xmm15, %xmm15, %xmm15
	sall	$2, %r9d
	pushq	-8(%r10)
	pushq	%rbp
	cltq
	salq	$3, %rax
	movslq	%r9d, %r9
	addq	$384, %rcx
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x70,0x6
	.cfi_escape 0x10,0xc,0x2,0x76,0x78
	pushq	%rbx
	leaq	(%rsi,%rax), %rdx
	salq	$3, %r9
	leaq	(%r8,%r9), %r12
	addq	$384, %rsi
	.cfi_escape 0x10,0x3,0x2,0x76,0x68
	addq	%rdx, %rax
	movq	(%r10), %r11
	movl	8(%r10), %ebx
	vblendpd	$1, -384(%rsi), %ymm15, %ymm1
	vmovapd	-384(%rcx), %ymm2
	vblendpd	$3, -352(%rsi), %ymm15, %ymm0
	vblendpd	$7, -320(%rsi), %ymm15, %ymm7
	vmovapd	-256(%rcx), %ymm10
	vblendpd	$3, 160(%rdx), %ymm15, %ymm6
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm5
	vmovapd	-224(%rsi), %ymm11
	vmovapd	%ymm1, %ymm4
	vmovapd	224(%rdx), %ymm14
	movq	16(%r10), %r10
	vfnmadd132pd	%ymm2, %ymm15, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	-128(%rsi), %ymm13
	addq	%r12, %r9
	addq	$384, %rdx
	addq	$384, %rax
	vfnmadd132pd	%ymm2, %ymm15, %ymm5
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfnmadd132pd	%ymm2, %ymm15, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfnmadd132pd	%ymm1, %ymm15, %ymm2
	vmovapd	-352(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm0, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm0, %ymm5
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm0, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd132pd	%ymm0, %ymm2, %ymm1
	vmovapd	-320(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm3, %ymm2
	vfnmadd231pd	%ymm0, %ymm7, %ymm5
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd132pd	%ymm7, %ymm1, %ymm0
	vmovapd	-288(%rsi), %ymm7
	vmovapd	-288(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm2
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm5
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm0
	vblendpd	$1, -256(%rdx), %ymm15, %ymm1
	vmovapd	-256(%rsi), %ymm7
	vmovapd	%ymm1, %ymm3
	vfnmadd231pd	%ymm10, %ymm7, %ymm2
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vfnmadd132pd	%ymm10, %ymm15, %ymm3
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfnmadd231pd	%ymm10, %ymm7, %ymm5
	vfnmadd132pd	%ymm10, %ymm15, %ymm9
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfnmadd231pd	%ymm10, %ymm7, %ymm4
	vfnmadd132pd	%ymm10, %ymm15, %ymm8
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfnmadd132pd	%ymm10, %ymm0, %ymm7
	vfnmadd132pd	%ymm1, %ymm15, %ymm10
	vmovapd	-224(%rcx), %ymm1
	vmovapd	-192(%rcx), %ymm0
	vfnmadd231pd	%ymm1, %ymm11, %ymm2
	vfnmadd231pd	%ymm1, %ymm6, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm11, %ymm5
	vfnmadd231pd	%ymm1, %ymm6, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm11, %ymm4
	vfnmadd231pd	%ymm1, %ymm6, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm11, %ymm7
	vmovapd	-192(%rsi), %ymm11
	vfnmadd132pd	%ymm6, %ymm10, %ymm1
	vblendpd	$7, -192(%rdx), %ymm15, %ymm6
	vmovapd	-160(%rsi), %ymm10
	vfnmadd231pd	%ymm0, %ymm11, %ymm2
	vfnmadd231pd	%ymm0, %ymm6, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm11, %ymm5
	vfnmadd231pd	%ymm0, %ymm6, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm11, %ymm4
	vfnmadd231pd	%ymm0, %ymm6, %ymm8
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm11, %ymm7
	vfnmadd132pd	%ymm0, %ymm1, %ymm6
	vmovapd	-160(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm10, %ymm2
	vfnmadd231pd	%ymm0, %ymm14, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm10, %ymm5
	vfnmadd231pd	%ymm0, %ymm14, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm10, %ymm4
	vfnmadd231pd	%ymm0, %ymm14, %ymm8
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm14, %ymm6
	vmovapd	-128(%rdx), %ymm14
	vblendpd	$1, -128(%rax), %ymm15, %ymm1
	vfnmadd231pd	%ymm0, %ymm10, %ymm7
	vmovapd	-128(%rcx), %ymm0
	vmovapd	%ymm1, %ymm10
	vmovapd	%ymm1, %ymm12
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm0, %ymm14, %ymm3
	vmovapd	%ymm1, %ymm11
	vfnmadd132pd	%ymm0, %ymm15, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm5
	vfnmadd231pd	%ymm0, %ymm14, %ymm9
	vfnmadd132pd	%ymm0, %ymm15, %ymm12
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm4
	vfnmadd231pd	%ymm0, %ymm14, %ymm8
	vfnmadd132pd	%ymm0, %ymm15, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm14, %ymm6
	vfnmadd132pd	%ymm0, %ymm15, %ymm1
	vmovapd	-96(%rdx), %ymm14
	vfnmadd231pd	%ymm0, %ymm13, %ymm7
	vmovapd	%ymm1, -80(%rbp)
	vblendpd	$3, -96(%rax), %ymm15, %ymm1
	vmovapd	-96(%rsi), %ymm13
	vblendpd	$7, -64(%rax), %ymm15, %ymm15
	vmovapd	-96(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm0, %ymm14, %ymm3
	vfnmadd231pd	%ymm0, %ymm1, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm5
	vfnmadd231pd	%ymm0, %ymm14, %ymm9
	vfnmadd231pd	%ymm0, %ymm1, %ymm12
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm4
	vfnmadd231pd	%ymm0, %ymm14, %ymm8
	vfnmadd231pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm7
	vfnmadd132pd	%ymm0, %ymm6, %ymm14
	vmovapd	-64(%rsi), %ymm13
	vmovapd	-64(%rdx), %ymm6
	vfnmadd213pd	-80(%rbp), %ymm1, %ymm0
	vmovapd	-64(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm1, %ymm6, %ymm3
	vfnmadd231pd	%ymm1, %ymm15, %ymm10
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm13, %ymm5
	vfnmadd231pd	%ymm1, %ymm6, %ymm9
	vfnmadd231pd	%ymm1, %ymm15, %ymm12
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm13, %ymm4
	vfnmadd231pd	%ymm1, %ymm6, %ymm8
	vfnmadd231pd	%ymm1, %ymm15, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm6, %ymm14
	vfnmadd231pd	%ymm1, %ymm13, %ymm7
	vmovapd	-32(%rdx), %ymm6
	vmovapd	-32(%rsi), %ymm13
	vfnmadd132pd	%ymm15, %ymm0, %ymm1
	vmovapd	-32(%rax), %ymm15
	vmovapd	-32(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm2
	vfnmadd231pd	%ymm0, %ymm6, %ymm3
	vfnmadd231pd	%ymm0, %ymm15, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm5
	vfnmadd231pd	%ymm0, %ymm6, %ymm9
	vfnmadd231pd	%ymm0, %ymm15, %ymm12
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm4
	vfnmadd231pd	%ymm0, %ymm6, %ymm8
	vfnmadd231pd	%ymm0, %ymm15, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm13, %ymm7
	vfnmadd132pd	%ymm0, %ymm14, %ymm6
	vmovapd	(%rsi), %ymm13
	vmovapd	(%rdx), %ymm14
	vfnmadd132pd	%ymm15, %ymm1, %ymm0
	vmovapd	(%rcx), %ymm1
	vmovapd	(%rax), %ymm15
	cmpl	$15, %edi
	jle	.L48
	subl	$16, %edi
	shrl	$2, %edi
	addq	$1, %rdi
	salq	$7, %rdi
	addq	%rsi, %rdi
	.p2align 4,,10
	.p2align 3
.L49:
	vfnmadd231pd	%ymm1, %ymm13, %ymm2
	vfnmadd231pd	%ymm1, %ymm14, %ymm3
	vfnmadd231pd	%ymm1, %ymm15, %ymm10
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	subq	$-128, %rsi
	subq	$-128, %rdx
	subq	$-128, %rax
	subq	$-128, %rcx
	vfnmadd231pd	%ymm1, %ymm13, %ymm5
	vfnmadd231pd	%ymm1, %ymm14, %ymm9
	vfnmadd231pd	%ymm1, %ymm15, %ymm12
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm13, %ymm4
	vfnmadd231pd	%ymm1, %ymm14, %ymm8
	vfnmadd231pd	%ymm1, %ymm15, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm15, %ymm0
	vfnmadd132pd	%ymm1, %ymm7, %ymm13
	vmovapd	-96(%rax), %ymm15
	vmovapd	-96(%rsi), %ymm7
	vfnmadd132pd	%ymm1, %ymm6, %ymm14
	vmovapd	-96(%rdx), %ymm6
	vmovapd	-96(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm2
	vfnmadd231pd	%ymm1, %ymm6, %ymm3
	vfnmadd231pd	%ymm1, %ymm15, %ymm10
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm5
	vfnmadd231pd	%ymm1, %ymm6, %ymm9
	vfnmadd231pd	%ymm1, %ymm15, %ymm12
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm4
	vfnmadd231pd	%ymm1, %ymm6, %ymm8
	vfnmadd231pd	%ymm1, %ymm15, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm7, %ymm13
	vfnmadd231pd	%ymm1, %ymm6, %ymm14
	vmovapd	-64(%rsi), %ymm7
	vmovapd	-64(%rdx), %ymm6
	vfnmadd132pd	%ymm1, %ymm0, %ymm15
	vmovapd	-64(%rax), %ymm1
	vmovapd	-64(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm2
	vfnmadd231pd	%ymm0, %ymm6, %ymm3
	vfnmadd231pd	%ymm0, %ymm1, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm5
	vfnmadd231pd	%ymm0, %ymm6, %ymm9
	vfnmadd231pd	%ymm0, %ymm1, %ymm12
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm4
	vfnmadd231pd	%ymm0, %ymm6, %ymm8
	vfnmadd231pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm13
	vfnmadd231pd	%ymm0, %ymm6, %ymm14
	vmovapd	-32(%rsi), %ymm7
	vmovapd	-32(%rdx), %ymm6
	vfnmadd231pd	%ymm0, %ymm1, %ymm15
	vmovapd	-32(%rax), %ymm1
	vmovapd	-32(%rcx), %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm2
	vfnmadd231pd	%ymm0, %ymm6, %ymm3
	vfnmadd231pd	%ymm0, %ymm1, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm5
	vfnmadd231pd	%ymm0, %ymm6, %ymm9
	vfnmadd231pd	%ymm0, %ymm1, %ymm12
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm7, %ymm4
	vfnmadd231pd	%ymm0, %ymm6, %ymm8
	vfnmadd231pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd132pd	%ymm0, %ymm13, %ymm7
	vmovapd	(%rsi), %ymm13
	vfnmadd132pd	%ymm0, %ymm14, %ymm6
	vmovapd	(%rdx), %ymm14
	vfnmadd132pd	%ymm1, %ymm15, %ymm0
	vmovapd	(%rcx), %ymm1
	vmovapd	(%rax), %ymm15
	cmpq	%rdi, %rsi
	jne	.L49
.L48:
	vblendpd	$5, %ymm5, %ymm2, %ymm1
	vblendpd	$10, %ymm5, %ymm2, %ymm13
	testl	%ebx, %ebx
	vblendpd	$10, %ymm4, %ymm7, %ymm5
	vblendpd	$5, %ymm4, %ymm7, %ymm4
	vblendpd	$5, %ymm12, %ymm10, %ymm14
	vblendpd	$12, %ymm5, %ymm13, %ymm2
	vblendpd	$3, %ymm5, %ymm13, %ymm13
	vblendpd	$12, %ymm4, %ymm1, %ymm5
	vblendpd	$3, %ymm4, %ymm1, %ymm4
	vblendpd	$10, %ymm9, %ymm3, %ymm1
	vblendpd	$5, %ymm9, %ymm3, %ymm3
	vblendpd	$10, %ymm8, %ymm6, %ymm9
	vblendpd	$5, %ymm8, %ymm6, %ymm6
	vblendpd	$10, %ymm11, %ymm0, %ymm8
	vblendpd	$5, %ymm11, %ymm0, %ymm0
	vblendpd	$12, %ymm6, %ymm3, %ymm15
	vblendpd	$3, %ymm6, %ymm3, %ymm3
	vblendpd	$10, %ymm12, %ymm10, %ymm6
	vblendpd	$12, %ymm9, %ymm1, %ymm7
	vblendpd	$12, %ymm0, %ymm14, %ymm12
	vblendpd	$3, %ymm9, %ymm1, %ymm1
	vblendpd	$12, %ymm8, %ymm6, %ymm10
	vblendpd	$3, %ymm0, %ymm14, %ymm0
	vblendpd	$3, %ymm8, %ymm6, %ymm6
	jne	.L54
	vbroadcastsd	120(%r11), %ymm8
	vbroadcastsd	(%r11), %ymm14
	vbroadcastsd	40(%r11), %ymm11
	vbroadcastsd	80(%r11), %ymm9
	vmovapd	%ymm8, -80(%rbp)
	vmovapd	.LC0(%rip), %ymm8
	vdivpd	%ymm14, %ymm8, %ymm14
	vdivpd	%ymm11, %ymm8, %ymm11
	vdivpd	%ymm9, %ymm8, %ymm9
	vdivpd	-80(%rbp), %ymm8, %ymm8
.L51:
	vmulpd	%ymm2, %ymm14, %ymm2
	vmulpd	%ymm7, %ymm14, %ymm7
	vmulpd	%ymm10, %ymm14, %ymm14
	vmovapd	%ymm2, (%r8)
	vmovapd	%ymm7, (%r12)
	vmovapd	%ymm14, (%r9)
	vbroadcastsd	8(%r11), %ymm10
	vfnmadd231pd	%ymm10, %ymm2, %ymm5
	vfnmadd231pd	%ymm10, %ymm7, %ymm15
	vmulpd	%ymm5, %ymm11, %ymm5
	vfnmadd231pd	%ymm10, %ymm14, %ymm12
	vmulpd	%ymm15, %ymm11, %ymm15
	vmulpd	%ymm12, %ymm11, %ymm12
	vmovapd	%ymm5, 32(%r8)
	vmovapd	%ymm15, 32(%r12)
	vmovapd	%ymm12, 32(%r9)
	vbroadcastsd	16(%r11), %ymm11
	vbroadcastsd	48(%r11), %ymm10
	vfnmadd231pd	%ymm11, %ymm2, %ymm13
	vfnmadd231pd	%ymm11, %ymm7, %ymm1
	vfnmadd231pd	%ymm11, %ymm14, %ymm6
	vfnmadd231pd	%ymm10, %ymm5, %ymm13
	vfnmadd231pd	%ymm10, %ymm12, %ymm6
	vfnmadd231pd	%ymm10, %ymm15, %ymm1
	vmulpd	%ymm13, %ymm9, %ymm13
	vmulpd	%ymm1, %ymm9, %ymm1
	vmulpd	%ymm6, %ymm9, %ymm9
	vmovapd	%ymm13, 64(%r8)
	vmovapd	%ymm1, 64(%r12)
	vmovapd	%ymm9, 64(%r9)
	vbroadcastsd	24(%r11), %ymm10
	vbroadcastsd	56(%r11), %ymm11
	vbroadcastsd	88(%r11), %ymm6
	vfnmadd132pd	%ymm10, %ymm4, %ymm2
	vfnmadd132pd	%ymm10, %ymm3, %ymm7
	vfnmadd132pd	%ymm11, %ymm2, %ymm5
	vfnmadd132pd	%ymm14, %ymm0, %ymm10
	vfnmadd132pd	%ymm11, %ymm7, %ymm15
	vfnmadd132pd	%ymm11, %ymm10, %ymm12
	vfnmadd132pd	%ymm6, %ymm5, %ymm13
	vfnmadd132pd	%ymm6, %ymm15, %ymm1
	vmulpd	%ymm13, %ymm8, %ymm13
	vfnmadd132pd	%ymm6, %ymm12, %ymm9
	vmulpd	%ymm1, %ymm8, %ymm1
	vmulpd	%ymm9, %ymm8, %ymm8
	vmovapd	%ymm13, 96(%r8)
	vmovapd	%ymm1, 96(%r12)
	vmovapd	%ymm8, 96(%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L54:
	.cfi_restore_state
	vbroadcastsd	(%r10), %ymm14
	vbroadcastsd	8(%r10), %ymm11
	vbroadcastsd	16(%r10), %ymm9
	vbroadcastsd	24(%r10), %ymm8
	jmp	.L51
	.cfi_endproc
.LFE4595:
	.size	kernel_dtrtri_12x4_lib4, .-kernel_dtrtri_12x4_lib4
	.section	.text.unlikely
.LCOLDE16:
	.text
.LHOTE16:
	.section	.text.unlikely
.LCOLDB17:
	.text
.LHOTB17:
	.p2align 4,,15
	.globl	kernel_dtrtri_8x4_lib4
	.type	kernel_dtrtri_8x4_lib4, @function
kernel_dtrtri_8x4_lib4:
.LFB4596:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	sall	$2, %edx
	vxorpd	%xmm3, %xmm3, %xmm3
	movslq	%edx, %rdx
	pushq	-8(%r10)
	pushq	%rbp
	leaq	(%rsi,%rdx,8), %rax
	sall	$2, %r9d
	addq	$256, %rcx
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x70,0x6
	.cfi_escape 0x10,0xc,0x2,0x76,0x78
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x68
	movslq	%r9d, %r9
	addq	$256, %rax
	movq	(%r10), %r11
	movl	8(%r10), %ebx
	leaq	(%r8,%r9,8), %r9
	vblendpd	$1, (%rsi), %ymm3, %ymm0
	vblendpd	$3, 32(%rsi), %ymm3, %ymm4
	vmovapd	-256(%rcx), %ymm2
	vmovapd	-224(%rcx), %ymm1
	vblendpd	$1, -128(%rax), %ymm3, %ymm9
	movq	16(%r10), %r10
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm7
	vmovapd	-128(%rcx), %ymm12
	vmovapd	%ymm0, %ymm11
	vmovapd	128(%rsi), %ymm15
	vmovapd	%ymm9, %ymm14
	vfnmadd132pd	%ymm2, %ymm3, %ymm6
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfnmadd231pd	%ymm1, %ymm4, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vshufpd	$5, %ymm12, %ymm12, %ymm5
	vfnmadd132pd	%ymm2, %ymm3, %ymm7
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfnmadd231pd	%ymm1, %ymm4, %ymm7
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vperm2f128	$1, %ymm5, %ymm5, %ymm8
	vfnmadd132pd	%ymm2, %ymm3, %ymm11
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfnmadd231pd	%ymm1, %ymm4, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vshufpd	$5, %ymm8, %ymm8, %ymm13
	vfnmadd132pd	%ymm0, %ymm3, %ymm2
	vmovapd	-192(%rcx), %ymm0
	vfnmadd132pd	%ymm4, %ymm2, %ymm1
	vblendpd	$7, 64(%rsi), %ymm3, %ymm4
	vmovapd	-160(%rcx), %ymm2
	vfnmadd132pd	%ymm13, %ymm3, %ymm14
	vfnmadd231pd	%ymm0, %ymm4, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm4, %ymm7
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfnmadd231pd	%ymm0, %ymm4, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfnmadd132pd	%ymm4, %ymm1, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm1
	vmovapd	96(%rsi), %ymm4
	vfnmadd231pd	%ymm2, %ymm4, %ymm6
	vmovapd	%ymm12, %ymm2
	vfnmadd231pd	%ymm12, %ymm15, %ymm6
	vfnmadd231pd	%ymm1, %ymm4, %ymm7
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vmovapd	-96(%rcx), %ymm12
	vfnmadd132pd	%ymm9, %ymm3, %ymm2
	vfnmadd231pd	%ymm5, %ymm15, %ymm7
	vfnmadd132pd	%ymm9, %ymm3, %ymm5
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vfnmadd231pd	%ymm1, %ymm4, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfnmadd231pd	%ymm1, %ymm4, %ymm0
	vmovapd	%ymm11, %ymm1
	vmovapd	%ymm8, %ymm4
	vmovapd	160(%rsi), %ymm11
	vfnmadd231pd	%ymm13, %ymm15, %ymm0
	vfnmadd231pd	%ymm8, %ymm15, %ymm1
	vperm2f128	$1, %ymm10, %ymm10, %ymm8
	vfnmadd132pd	%ymm9, %ymm3, %ymm4
	vblendpd	$3, -96(%rax), %ymm3, %ymm9
	vfnmadd231pd	%ymm12, %ymm11, %ymm6
	vblendpd	$7, -64(%rax), %ymm3, %ymm3
	vfnmadd231pd	%ymm10, %ymm11, %ymm7
	vshufpd	$5, %ymm8, %ymm8, %ymm13
	vfnmadd132pd	%ymm9, %ymm2, %ymm12
	vmovapd	-64(%rcx), %ymm2
	vfnmadd231pd	%ymm10, %ymm9, %ymm5
	vfnmadd231pd	%ymm8, %ymm11, %ymm1
	vfnmadd132pd	%ymm9, %ymm4, %ymm8
	vfnmadd231pd	%ymm2, %ymm3, %ymm12
	vfnmadd231pd	%ymm13, %ymm11, %ymm0
	vfnmadd132pd	%ymm9, %ymm14, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm9
	vmovapd	192(%rsi), %ymm11
	vperm2f128	$1, %ymm9, %ymm9, %ymm4
	vfnmadd231pd	%ymm2, %ymm11, %ymm6
	vmovapd	-32(%rcx), %ymm2
	vfnmadd231pd	%ymm9, %ymm3, %ymm5
	vfnmadd231pd	%ymm9, %ymm11, %ymm7
	vshufpd	$5, %ymm4, %ymm4, %ymm10
	vfnmadd231pd	%ymm4, %ymm3, %ymm8
	vfnmadd231pd	%ymm4, %ymm11, %ymm1
	vfnmadd132pd	%ymm10, %ymm13, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm13
	vfnmadd231pd	%ymm10, %ymm11, %ymm0
	vmovapd	-32(%rax), %ymm10
	vmovapd	%ymm3, %ymm9
	vmovapd	224(%rsi), %ymm11
	vperm2f128	$1, %ymm13, %ymm13, %ymm4
	vmovapd	(%rcx), %ymm3
	vfnmadd231pd	%ymm13, %ymm10, %ymm5
	vfnmadd231pd	%ymm2, %ymm11, %ymm6
	vfnmadd132pd	%ymm10, %ymm12, %ymm2
	vfnmadd231pd	%ymm13, %ymm11, %ymm7
	vshufpd	$5, %ymm4, %ymm4, %ymm12
	vfnmadd231pd	%ymm4, %ymm11, %ymm1
	vfnmadd132pd	%ymm10, %ymm8, %ymm4
	vfnmadd231pd	%ymm12, %ymm11, %ymm0
	vfnmadd132pd	%ymm12, %ymm9, %ymm10
	vmovapd	(%rax), %ymm12
	cmpl	$11, %edi
	vmovapd	256(%rsi), %ymm11
	jle	.L56
	leal	-12(%rdi), %edx
	leaq	256(%rsi), %r12
	addq	$288, %rsi
	shrl	$2, %edx
	salq	$7, %rdx
	leaq	160(%r12,%rdx), %rdx
	.p2align 4,,10
	.p2align 3
.L57:
	vshufpd	$5, %ymm3, %ymm3, %ymm9
	vfnmadd231pd	%ymm3, %ymm11, %ymm6
	vfnmadd231pd	%ymm3, %ymm12, %ymm2
	vmovapd	32(%rcx), %ymm3
	subq	$-128, %rsi
	subq	$-128, %rax
	vmovapd	-128(%rsi), %ymm13
	subq	$-128, %rcx
	vperm2f128	$1, %ymm9, %ymm9, %ymm8
	vfnmadd231pd	%ymm9, %ymm11, %ymm7
	vfnmadd132pd	%ymm12, %ymm5, %ymm9
	vmovapd	-96(%rax), %ymm15
	vfnmadd231pd	%ymm3, %ymm13, %ymm6
	vshufpd	$5, %ymm8, %ymm8, %ymm5
	vfnmadd231pd	%ymm3, %ymm15, %ymm2
	vfnmadd231pd	%ymm8, %ymm11, %ymm1
	vfnmadd132pd	%ymm12, %ymm4, %ymm8
	vfnmadd231pd	%ymm5, %ymm11, %ymm0
	vfnmadd231pd	%ymm5, %ymm12, %ymm10
	vshufpd	$5, %ymm3, %ymm3, %ymm5
	vmovapd	-64(%rcx), %ymm3
	vmovapd	(%rax), %ymm12
	vperm2f128	$1, %ymm5, %ymm5, %ymm4
	vfnmadd231pd	%ymm5, %ymm13, %ymm7
	vfnmadd132pd	%ymm15, %ymm9, %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vmovapd	-96(%rsi), %ymm9
	vshufpd	$5, %ymm4, %ymm4, %ymm14
	vfnmadd231pd	%ymm4, %ymm13, %ymm1
	vfnmadd132pd	%ymm15, %ymm8, %ymm4
	vperm2f128	$1, %ymm11, %ymm11, %ymm8
	vfnmadd231pd	%ymm3, %ymm9, %ymm6
	vfnmadd231pd	%ymm11, %ymm9, %ymm7
	vfnmadd231pd	%ymm14, %ymm15, %ymm10
	vmovapd	-64(%rax), %ymm15
	vfnmadd231pd	%ymm14, %ymm13, %ymm0
	vshufpd	$5, %ymm8, %ymm8, %ymm14
	vmovapd	%ymm9, %ymm13
	vfnmadd231pd	%ymm8, %ymm9, %ymm1
	vfnmadd132pd	%ymm15, %ymm2, %ymm3
	vmovapd	-32(%rcx), %ymm2
	vfnmadd231pd	%ymm11, %ymm15, %ymm5
	vfnmadd231pd	%ymm8, %ymm15, %ymm4
	vmovapd	-32(%rax), %ymm8
	vfnmadd132pd	%ymm14, %ymm10, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfnmadd132pd	%ymm14, %ymm0, %ymm13
	vmovapd	-64(%rsi), %ymm0
	vmovapd	-32(%rsi), %ymm11
	vperm2f128	$1, %ymm10, %ymm10, %ymm9
	vfnmadd231pd	%ymm10, %ymm0, %ymm7
	vfnmadd231pd	%ymm10, %ymm8, %ymm5
	vfnmadd231pd	%ymm2, %ymm0, %ymm6
	vfnmadd132pd	%ymm8, %ymm3, %ymm2
	vmovapd	(%rcx), %ymm3
	cmpq	%rsi, %rdx
	vshufpd	$5, %ymm9, %ymm9, %ymm10
	vfnmadd231pd	%ymm9, %ymm0, %ymm1
	vfnmadd231pd	%ymm9, %ymm8, %ymm4
	vfnmadd132pd	%ymm10, %ymm13, %ymm0
	vfnmadd132pd	%ymm8, %ymm15, %ymm10
	jne	.L57
.L56:
	vblendpd	$10, %ymm5, %ymm2, %ymm13
	vblendpd	$10, %ymm1, %ymm0, %ymm8
	testl	%ebx, %ebx
	vblendpd	$10, %ymm7, %ymm6, %ymm3
	vblendpd	$5, %ymm5, %ymm2, %ymm5
	vblendpd	$5, %ymm7, %ymm6, %ymm6
	vblendpd	$10, %ymm4, %ymm10, %ymm2
	vblendpd	$5, %ymm1, %ymm0, %ymm0
	vblendpd	$5, %ymm4, %ymm10, %ymm4
	vblendpd	$12, %ymm8, %ymm3, %ymm12
	vblendpd	$12, %ymm2, %ymm13, %ymm11
	vblendpd	$12, %ymm0, %ymm6, %ymm9
	vblendpd	$3, %ymm0, %ymm6, %ymm1
	vblendpd	$3, %ymm8, %ymm3, %ymm8
	vblendpd	$3, %ymm2, %ymm13, %ymm6
	vblendpd	$12, %ymm4, %ymm5, %ymm13
	vblendpd	$3, %ymm4, %ymm5, %ymm4
	jne	.L62
	vmovapd	.LC0(%rip), %ymm14
	vbroadcastsd	(%r11), %ymm10
	vbroadcastsd	40(%r11), %ymm7
	vbroadcastsd	80(%r11), %ymm5
	vbroadcastsd	120(%r11), %ymm0
	vdivpd	%ymm10, %ymm14, %ymm10
	vdivpd	%ymm7, %ymm14, %ymm7
	vdivpd	%ymm5, %ymm14, %ymm5
	vdivpd	%ymm0, %ymm14, %ymm14
.L59:
	vmovapd	%ymm9, %ymm0
	vmulpd	%ymm12, %ymm10, %ymm12
	vmulpd	%ymm11, %ymm10, %ymm11
	vmovapd	%ymm12, (%r8)
	vmovapd	%ymm11, (%r9)
	vbroadcastsd	8(%r11), %ymm2
	vfnmadd231pd	%ymm2, %ymm12, %ymm0
	vfnmadd132pd	%ymm11, %ymm13, %ymm2
	vmulpd	%ymm0, %ymm7, %ymm9
	vmulpd	%ymm2, %ymm7, %ymm13
	vmovapd	%ymm9, 32(%r8)
	vmovapd	%ymm13, 32(%r9)
	vbroadcastsd	16(%r11), %ymm2
	vbroadcastsd	48(%r11), %ymm0
	vfnmadd231pd	%ymm2, %ymm12, %ymm8
	vfnmadd132pd	%ymm11, %ymm6, %ymm2
	vfnmadd231pd	%ymm0, %ymm9, %ymm8
	vfnmadd132pd	%ymm13, %ymm2, %ymm0
	vmulpd	%ymm8, %ymm5, %ymm3
	vmulpd	%ymm0, %ymm5, %ymm8
	vmovapd	%ymm12, %ymm0
	vmovapd	%ymm3, 64(%r8)
	vmovapd	%ymm8, 64(%r9)
	vbroadcastsd	24(%r11), %ymm2
	vbroadcastsd	56(%r11), %ymm5
	vbroadcastsd	88(%r11), %ymm15
	vfnmadd132pd	%ymm2, %ymm1, %ymm0
	vfnmadd132pd	%ymm2, %ymm4, %ymm11
	vfnmadd231pd	%ymm5, %ymm9, %ymm0
	vfnmadd132pd	%ymm5, %ymm11, %ymm13
	vfnmadd132pd	%ymm15, %ymm0, %ymm3
	vfnmadd132pd	%ymm15, %ymm13, %ymm8
	vmulpd	%ymm3, %ymm14, %ymm3
	vmulpd	%ymm8, %ymm14, %ymm8
	vmovapd	%ymm3, 96(%r8)
	vmovapd	%ymm8, 96(%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L62:
	.cfi_restore_state
	vbroadcastsd	(%r10), %ymm10
	vbroadcastsd	8(%r10), %ymm7
	vbroadcastsd	16(%r10), %ymm5
	vbroadcastsd	24(%r10), %ymm14
	jmp	.L59
	.cfi_endproc
.LFE4596:
	.size	kernel_dtrtri_8x4_lib4, .-kernel_dtrtri_8x4_lib4
	.section	.text.unlikely
.LCOLDE17:
	.text
.LHOTE17:
	.section	.text.unlikely
.LCOLDB18:
	.text
.LHOTB18:
	.p2align 4,,15
	.globl	kernel_dtrtri_8x3_lib4
	.type	kernel_dtrtri_8x3_lib4, @function
kernel_dtrtri_8x3_lib4:
.LFB4597:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	sall	$2, %edx
	vxorpd	%xmm10, %xmm10, %xmm10
	movslq	%edx, %rdx
	pushq	-8(%r10)
	pushq	%rbp
	leaq	(%rsi,%rdx,8), %rax
	sall	$2, %r9d
	addq	$256, %rsi
	addq	$256, %rcx
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x70
	vmovapd	%ymm10, %ymm2
	movq	(%r10), %r11
	movl	8(%r10), %ebx
	vblendpd	$1, -256(%rsi), %ymm10, %ymm1
	vbroadcastsd	-256(%rcx), %ymm9
	vbroadcastsd	-248(%rcx), %ymm6
	vblendpd	$3, -224(%rsi), %ymm10, %ymm3
	vbroadcastsd	-240(%rcx), %ymm0
	vfnmadd132pd	%ymm1, %ymm10, %ymm9
	movq	16(%r10), %r10
	vfnmadd132pd	%ymm1, %ymm10, %ymm6
	vmovapd	-128(%rsi), %ymm4
	movslq	%r9d, %r9
	vfnmadd132pd	%ymm1, %ymm10, %ymm0
	vbroadcastsd	-224(%rcx), %ymm1
	addq	$256, %rax
	vbroadcastsd	-128(%rcx), %ymm7
	leaq	(%r8,%r9,8), %r9
	vfnmadd231pd	%ymm1, %ymm3, %ymm9
	vbroadcastsd	-216(%rcx), %ymm1
	vbroadcastsd	-120(%rcx), %ymm5
	vfnmadd231pd	%ymm1, %ymm3, %ymm6
	vbroadcastsd	-208(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm3, %ymm0
	vblendpd	$7, -192(%rsi), %ymm10, %ymm3
	vbroadcastsd	-192(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm3, %ymm9
	vbroadcastsd	-184(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm3, %ymm6
	vbroadcastsd	-176(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm3, %ymm0
	vmovapd	-160(%rsi), %ymm1
	vbroadcastsd	-160(%rcx), %ymm3
	vfnmadd231pd	%ymm3, %ymm1, %ymm9
	vbroadcastsd	-152(%rcx), %ymm3
	vfnmadd231pd	%ymm7, %ymm4, %ymm9
	vfnmadd231pd	%ymm3, %ymm1, %ymm6
	vbroadcastsd	-144(%rcx), %ymm3
	vfnmadd231pd	%ymm5, %ymm4, %ymm6
	vfnmadd231pd	%ymm3, %ymm1, %ymm0
	vblendpd	$1, -128(%rax), %ymm10, %ymm1
	vbroadcastsd	-112(%rcx), %ymm3
	vfnmadd132pd	%ymm1, %ymm10, %ymm7
	vfnmadd231pd	%ymm3, %ymm1, %ymm2
	vfnmadd132pd	%ymm1, %ymm10, %ymm5
	vfnmadd132pd	%ymm3, %ymm0, %ymm4
	vblendpd	$3, -96(%rax), %ymm10, %ymm1
	vmovapd	-96(%rsi), %ymm0
	vbroadcastsd	-96(%rcx), %ymm3
	vbroadcastsd	-88(%rcx), %ymm8
	vblendpd	$7, -64(%rax), %ymm10, %ymm10
	vfnmadd231pd	%ymm3, %ymm0, %ymm9
	vfnmadd231pd	%ymm3, %ymm1, %ymm7
	vbroadcastsd	-80(%rcx), %ymm3
	vfnmadd231pd	%ymm8, %ymm0, %ymm6
	vfnmadd231pd	%ymm8, %ymm1, %ymm5
	vbroadcastsd	-56(%rcx), %ymm8
	vfnmadd132pd	%ymm3, %ymm2, %ymm1
	vfnmadd231pd	%ymm3, %ymm0, %ymm4
	vbroadcastsd	-48(%rcx), %ymm2
	vmovapd	-64(%rsi), %ymm0
	vfnmadd231pd	%ymm8, %ymm10, %ymm5
	vbroadcastsd	-64(%rcx), %ymm3
	vfnmadd231pd	%ymm8, %ymm0, %ymm6
	vfnmadd231pd	%ymm2, %ymm0, %ymm4
	vbroadcastsd	-24(%rcx), %ymm8
	vfnmadd231pd	%ymm3, %ymm0, %ymm9
	vfnmadd231pd	%ymm3, %ymm10, %ymm7
	vmovapd	-32(%rsi), %ymm0
	vbroadcastsd	-32(%rcx), %ymm3
	vfnmadd132pd	%ymm2, %ymm1, %ymm10
	vmovapd	-32(%rax), %ymm2
	vfnmadd231pd	%ymm8, %ymm0, %ymm6
	vfnmadd231pd	%ymm3, %ymm0, %ymm9
	vfnmadd231pd	%ymm3, %ymm2, %ymm7
	vbroadcastsd	-16(%rcx), %ymm3
	vmovapd	%ymm2, %ymm1
	vfnmadd132pd	%ymm2, %ymm5, %ymm8
	vmovapd	(%rax), %ymm2
	vfnmadd231pd	%ymm3, %ymm0, %ymm4
	vmovapd	(%rsi), %ymm0
	cmpl	$11, %edi
	vfnmadd132pd	%ymm3, %ymm10, %ymm1
	jle	.L64
	leal	-12(%rdi), %edx
	shrl	$2, %edx
	addq	$1, %rdx
	salq	$7, %rdx
	addq	%rsi, %rdx
	.p2align 4,,10
	.p2align 3
.L65:
	vbroadcastsd	(%rcx), %ymm3
	subq	$-128, %rsi
	subq	$-128, %rax
	vbroadcastsd	8(%rcx), %ymm5
	subq	$-128, %rcx
	vfnmadd231pd	%ymm3, %ymm0, %ymm9
	vfnmadd231pd	%ymm3, %ymm2, %ymm7
	vbroadcastsd	-112(%rcx), %ymm3
	vfnmadd231pd	%ymm5, %ymm0, %ymm6
	vfnmadd231pd	%ymm5, %ymm2, %ymm8
	vbroadcastsd	-88(%rcx), %ymm5
	vfnmadd132pd	%ymm3, %ymm4, %ymm0
	vmovapd	-96(%rsi), %ymm4
	vfnmadd132pd	%ymm2, %ymm1, %ymm3
	vmovapd	-96(%rax), %ymm1
	vbroadcastsd	-96(%rcx), %ymm2
	vfnmadd231pd	%ymm5, %ymm4, %ymm6
	vfnmadd231pd	%ymm5, %ymm1, %ymm8
	vbroadcastsd	-56(%rcx), %ymm5
	vfnmadd231pd	%ymm2, %ymm4, %ymm9
	vfnmadd231pd	%ymm2, %ymm1, %ymm7
	vbroadcastsd	-80(%rcx), %ymm2
	vfnmadd231pd	%ymm2, %ymm4, %ymm0
	vmovapd	-64(%rsi), %ymm4
	vfnmadd132pd	%ymm1, %ymm3, %ymm2
	vmovapd	-64(%rax), %ymm3
	vbroadcastsd	-64(%rcx), %ymm1
	vfnmadd231pd	%ymm5, %ymm4, %ymm6
	vfnmadd132pd	%ymm3, %ymm8, %ymm5
	vbroadcastsd	-24(%rcx), %ymm8
	vfnmadd231pd	%ymm1, %ymm4, %ymm9
	vfnmadd231pd	%ymm1, %ymm3, %ymm7
	vbroadcastsd	-48(%rcx), %ymm1
	vfnmadd231pd	%ymm1, %ymm4, %ymm0
	vmovapd	-32(%rsi), %ymm4
	vfnmadd132pd	%ymm3, %ymm2, %ymm1
	vmovapd	-32(%rax), %ymm2
	vbroadcastsd	-32(%rcx), %ymm3
	vfnmadd231pd	%ymm8, %ymm4, %ymm6
	vfnmadd132pd	%ymm2, %ymm5, %ymm8
	vfnmadd231pd	%ymm3, %ymm4, %ymm9
	vfnmadd231pd	%ymm3, %ymm2, %ymm7
	vbroadcastsd	-16(%rcx), %ymm3
	vfnmadd132pd	%ymm3, %ymm0, %ymm4
	vmovapd	(%rsi), %ymm0
	vfnmadd231pd	%ymm3, %ymm2, %ymm1
	vmovapd	(%rax), %ymm2
	cmpq	%rdx, %rsi
	jne	.L65
.L64:
	testl	%ebx, %ebx
	jne	.L70
	vmovapd	.LC0(%rip), %ymm0
	vbroadcastsd	(%r11), %ymm3
	vbroadcastsd	40(%r11), %ymm2
	vbroadcastsd	80(%r11), %ymm10
	vdivpd	%ymm3, %ymm0, %ymm3
	vdivpd	%ymm2, %ymm0, %ymm2
	vdivpd	%ymm10, %ymm0, %ymm10
.L67:
	vmulpd	%ymm9, %ymm3, %ymm9
	vmulpd	%ymm7, %ymm3, %ymm7
	vmovapd	%ymm9, (%r8)
	vmovapd	%ymm7, (%r9)
	vbroadcastsd	8(%r11), %ymm0
	vfnmadd231pd	%ymm0, %ymm9, %ymm6
	vfnmadd231pd	%ymm0, %ymm7, %ymm8
	vmulpd	%ymm6, %ymm2, %ymm5
	vmulpd	%ymm8, %ymm2, %ymm8
	vmovapd	%ymm5, 32(%r8)
	vmovapd	%ymm8, 32(%r9)
	vbroadcastsd	16(%r11), %ymm3
	vbroadcastsd	48(%r11), %ymm0
	vfnmadd231pd	%ymm3, %ymm9, %ymm4
	vfnmadd132pd	%ymm7, %ymm1, %ymm3
	vfnmadd132pd	%ymm0, %ymm4, %ymm5
	vfnmadd132pd	%ymm0, %ymm3, %ymm8
	vmulpd	%ymm5, %ymm10, %ymm5
	vmulpd	%ymm8, %ymm10, %ymm8
	vmovapd	%ymm5, 64(%r8)
	vmovapd	%ymm8, 64(%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L70:
	.cfi_restore_state
	vbroadcastsd	(%r10), %ymm3
	vbroadcastsd	8(%r10), %ymm2
	vbroadcastsd	16(%r10), %ymm10
	jmp	.L67
	.cfi_endproc
.LFE4597:
	.size	kernel_dtrtri_8x3_lib4, .-kernel_dtrtri_8x3_lib4
	.section	.text.unlikely
.LCOLDE18:
	.text
.LHOTE18:
	.section	.text.unlikely
.LCOLDB19:
	.text
.LHOTB19:
	.p2align 4,,15
	.globl	kernel_dtrtri_8x2_lib4
	.type	kernel_dtrtri_8x2_lib4, @function
kernel_dtrtri_8x2_lib4:
.LFB4598:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	sall	$2, %edx
	vxorpd	%xmm1, %xmm1, %xmm1
	movslq	%edx, %rdx
	pushq	-8(%r10)
	pushq	%rbp
	leaq	(%rsi,%rdx,8), %rdx
	sall	$2, %r9d
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x70
	movq	(%r10), %r11
	movl	8(%r10), %ebx
	movslq	%r9d, %r9
	vblendpd	$1, (%rsi), %ymm1, %ymm3
	vblendpd	$3, 32(%rsi), %ymm1, %ymm5
	vbroadcastf128	(%rcx), %ymm2
	vbroadcastf128	32(%rcx), %ymm4
	vblendpd	$1, 128(%rdx), %ymm1, %ymm8
	vblendpd	$3, 160(%rdx), %ymm1, %ymm9
	vmovapd	%ymm3, %ymm0
	vmovapd	128(%rsi), %ymm6
	movq	16(%r10), %r10
	addq	$256, %rdx
	leaq	(%r8,%r9,8), %r9
	vfnmadd132pd	%ymm2, %ymm1, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfnmadd231pd	%ymm4, %ymm5, %ymm0
	vshufpd	$5, %ymm4, %ymm4, %ymm4
	vfnmadd132pd	%ymm3, %ymm1, %ymm2
	vbroadcastf128	64(%rcx), %ymm3
	vfnmadd132pd	%ymm5, %ymm2, %ymm4
	vblendpd	$7, 64(%rsi), %ymm1, %ymm5
	vbroadcastf128	96(%rcx), %ymm2
	vfnmadd231pd	%ymm3, %ymm5, %ymm0
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfnmadd132pd	%ymm5, %ymm4, %ymm3
	vmovapd	96(%rsi), %ymm5
	vbroadcastf128	128(%rcx), %ymm4
	vfnmadd231pd	%ymm2, %ymm5, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfnmadd231pd	%ymm4, %ymm6, %ymm0
	vfnmadd132pd	%ymm5, %ymm3, %ymm2
	vshufpd	$5, %ymm4, %ymm4, %ymm5
	vbroadcastf128	160(%rcx), %ymm3
	vfnmadd132pd	%ymm8, %ymm1, %ymm4
	vfnmadd132pd	%ymm5, %ymm2, %ymm6
	vfnmadd132pd	%ymm5, %ymm1, %ymm8
	vshufpd	$5, %ymm3, %ymm3, %ymm5
	vmovapd	%ymm6, %ymm7
	vmovapd	160(%rsi), %ymm6
	vblendpd	$7, -64(%rdx), %ymm1, %ymm1
	vfnmadd231pd	%ymm5, %ymm6, %ymm7
	vfnmadd231pd	%ymm3, %ymm6, %ymm0
	vfnmadd132pd	%ymm9, %ymm4, %ymm3
	vmovapd	192(%rsi), %ymm6
	vmovapd	%ymm3, %ymm2
	vmovapd	%ymm9, %ymm4
	vbroadcastf128	192(%rcx), %ymm3
	vfnmadd132pd	%ymm5, %ymm8, %ymm4
	vfnmadd231pd	%ymm3, %ymm6, %ymm0
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vfnmadd132pd	%ymm1, %ymm2, %ymm3
	vbroadcastf128	224(%rcx), %ymm2
	vshufpd	$5, %ymm2, %ymm2, %ymm5
	vfnmadd132pd	%ymm8, %ymm7, %ymm6
	vfnmadd132pd	%ymm1, %ymm4, %ymm8
	vmovapd	224(%rsi), %ymm7
	vmovapd	-32(%rdx), %ymm4
	vfnmadd231pd	%ymm2, %ymm7, %ymm0
	vbroadcastf128	256(%rcx), %ymm1
	vfnmadd132pd	%ymm5, %ymm6, %ymm7
	vfnmadd132pd	%ymm4, %ymm3, %ymm2
	vfnmadd132pd	%ymm5, %ymm8, %ymm4
	vmovapd	(%rdx), %ymm8
	cmpl	$11, %edi
	vmovapd	256(%rsi), %ymm6
	jle	.L72
	subl	$12, %edi
	leaq	256(%rcx), %rax
	leaq	256(%rsi), %rcx
	shrl	$2, %edi
	addq	$288, %rsi
	salq	$7, %rdi
	leaq	160(%rcx,%rdi), %rcx
	.p2align 4,,10
	.p2align 3
.L73:
	vshufpd	$5, %ymm1, %ymm1, %ymm5
	vbroadcastf128	32(%rax), %ymm3
	vfnmadd231pd	%ymm1, %ymm8, %ymm2
	vmovapd	(%rsi), %ymm9
	vfnmadd231pd	%ymm1, %ymm6, %ymm0
	subq	$-128, %rsi
	subq	$-128, %rdx
	vfnmadd231pd	%ymm5, %ymm6, %ymm7
	vfnmadd231pd	%ymm5, %ymm8, %ymm4
	vmovapd	-96(%rdx), %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vfnmadd231pd	%ymm3, %ymm9, %ymm0
	vmovapd	-32(%rsi), %ymm6
	vfnmadd132pd	%ymm5, %ymm2, %ymm3
	vbroadcastf128	64(%rax), %ymm2
	vfnmadd132pd	%ymm8, %ymm7, %ymm9
	vmovapd	-96(%rsi), %ymm7
	vfnmadd132pd	%ymm5, %ymm4, %ymm8
	vmovapd	-64(%rdx), %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm5
	vfnmadd231pd	%ymm2, %ymm7, %ymm0
	vfnmadd132pd	%ymm4, %ymm3, %ymm2
	vbroadcastf128	96(%rax), %ymm3
	subq	$-128, %rax
	vfnmadd132pd	%ymm5, %ymm9, %ymm7
	vfnmadd132pd	%ymm4, %ymm8, %ymm5
	vmovapd	-64(%rsi), %ymm9
	vshufpd	$5, %ymm3, %ymm3, %ymm4
	vmovapd	-32(%rdx), %ymm8
	vfnmadd231pd	%ymm3, %ymm9, %ymm0
	vbroadcastf128	(%rax), %ymm1
	vfnmadd231pd	%ymm3, %ymm8, %ymm2
	vfnmadd231pd	%ymm4, %ymm9, %ymm7
	vfnmadd132pd	%ymm8, %ymm5, %ymm4
	vmovapd	(%rdx), %ymm8
	cmpq	%rcx, %rsi
	jne	.L73
.L72:
	testl	%ebx, %ebx
	vblendpd	$10, %ymm7, %ymm0, %ymm9
	vblendpd	$10, %ymm4, %ymm2, %ymm1
	vblendpd	$5, %ymm7, %ymm0, %ymm0
	vblendpd	$5, %ymm4, %ymm2, %ymm4
	jne	.L78
	vmovapd	.LC0(%rip), %ymm3
	vbroadcastsd	(%r11), %ymm2
	vbroadcastsd	40(%r11), %ymm5
	vdivpd	%ymm2, %ymm3, %ymm2
	vdivpd	%ymm5, %ymm3, %ymm5
.L75:
	vmulpd	%ymm9, %ymm2, %ymm9
	vmulpd	%ymm1, %ymm2, %ymm1
	vmovapd	%ymm9, (%r8)
	vmovapd	%ymm1, (%r9)
	vbroadcastsd	8(%r11), %ymm3
	vfnmadd231pd	%ymm3, %ymm9, %ymm0
	vfnmadd231pd	%ymm3, %ymm1, %ymm4
	vmulpd	%ymm0, %ymm5, %ymm0
	vmulpd	%ymm4, %ymm5, %ymm5
	vmovapd	%ymm0, 32(%r8)
	vmovapd	%ymm5, 32(%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L78:
	.cfi_restore_state
	vbroadcastsd	(%r10), %ymm2
	vbroadcastsd	8(%r10), %ymm5
	jmp	.L75
	.cfi_endproc
.LFE4598:
	.size	kernel_dtrtri_8x2_lib4, .-kernel_dtrtri_8x2_lib4
	.section	.text.unlikely
.LCOLDE19:
	.text
.LHOTE19:
	.section	.text.unlikely
.LCOLDB20:
	.text
.LHOTB20:
	.p2align 4,,15
	.globl	kernel_dtrtri_8x1_lib4
	.type	kernel_dtrtri_8x1_lib4, @function
kernel_dtrtri_8x1_lib4:
.LFB4599:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	sall	$2, %edx
	vxorpd	%xmm2, %xmm2, %xmm2
	movslq	%edx, %rdx
	pushq	-8(%r10)
	pushq	%rbp
	leaq	(%rsi,%rdx,8), %rax
	sall	$2, %r9d
	addq	$256, %rcx
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x70
	movslq	%r9d, %r9
	addq	$256, %rax
	movq	(%r10), %rbx
	vblendpd	$1, (%rsi), %ymm2, %ymm1
	vbroadcastsd	-256(%rcx), %ymm0
	vbroadcastsd	-224(%rcx), %ymm3
	vblendpd	$3, -96(%rax), %ymm2, %ymm4
	vfnmadd132pd	%ymm1, %ymm2, %ymm0
	vblendpd	$3, 32(%rsi), %ymm2, %ymm1
	movl	8(%r10), %r11d
	leaq	(%r8,%r9,8), %r9
	movq	16(%r10), %r10
	vfnmadd231pd	%ymm3, %ymm1, %ymm0
	vbroadcastsd	-192(%rcx), %ymm3
	vblendpd	$7, 64(%rsi), %ymm2, %ymm1
	vfnmadd132pd	%ymm3, %ymm0, %ymm1
	vbroadcastsd	-160(%rcx), %ymm0
	vblendpd	$1, -128(%rax), %ymm2, %ymm3
	vfnmadd132pd	96(%rsi), %ymm1, %ymm0
	vbroadcastsd	-128(%rcx), %ymm1
	vfnmadd231pd	128(%rsi), %ymm1, %ymm0
	vfnmadd132pd	%ymm3, %ymm2, %ymm1
	vbroadcastsd	-96(%rcx), %ymm3
	vblendpd	$7, -64(%rax), %ymm2, %ymm2
	vfnmadd231pd	160(%rsi), %ymm3, %ymm0
	vfnmadd132pd	%ymm4, %ymm1, %ymm3
	vbroadcastsd	-64(%rcx), %ymm1
	vmovapd	(%rax), %ymm4
	vfnmadd231pd	192(%rsi), %ymm1, %ymm0
	vfnmadd132pd	%ymm1, %ymm3, %ymm2
	vbroadcastsd	-32(%rcx), %ymm1
	vmovapd	%ymm0, %ymm3
	vmovapd	256(%rsi), %ymm0
	vfnmadd231pd	224(%rsi), %ymm1, %ymm3
	vfnmadd132pd	-32(%rax), %ymm2, %ymm1
	cmpl	$11, %edi
	jle	.L80
	subl	$12, %edi
	leaq	256(%rsi), %rdx
	addq	$288, %rsi
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	160(%rdx,%rdi), %rdx
	.p2align 4,,10
	.p2align 3
.L81:
	vbroadcastsd	(%rcx), %ymm2
	subq	$-128, %rsi
	subq	$-128, %rax
	subq	$-128, %rcx
	vfnmadd132pd	%ymm2, %ymm3, %ymm0
	vfnmadd231pd	%ymm2, %ymm4, %ymm1
	vbroadcastsd	-96(%rcx), %ymm2
	vmovapd	(%rax), %ymm4
	vfnmadd231pd	-96(%rax), %ymm2, %ymm1
	vfnmadd231pd	-128(%rsi), %ymm2, %ymm0
	vbroadcastsd	-64(%rcx), %ymm2
	vfnmadd231pd	-96(%rsi), %ymm2, %ymm0
	vfnmadd132pd	-64(%rax), %ymm1, %ymm2
	vmovapd	%ymm0, %ymm3
	vbroadcastsd	-32(%rcx), %ymm1
	vmovapd	-32(%rsi), %ymm0
	vfnmadd231pd	-64(%rsi), %ymm1, %ymm3
	vfnmadd132pd	-32(%rax), %ymm2, %ymm1
	cmpq	%rdx, %rsi
	jne	.L81
.L80:
	testl	%r11d, %r11d
	jne	.L86
	vbroadcastsd	(%rbx), %ymm2
	vmovapd	.LC0(%rip), %ymm0
	vdivpd	%ymm2, %ymm0, %ymm2
.L83:
	vmulpd	%ymm3, %ymm2, %ymm0
	vmulpd	%ymm1, %ymm2, %ymm1
	vmovapd	%ymm0, (%r8)
	vmovapd	%ymm1, (%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L86:
	.cfi_restore_state
	vbroadcastsd	(%r10), %ymm2
	jmp	.L83
	.cfi_endproc
.LFE4599:
	.size	kernel_dtrtri_8x1_lib4, .-kernel_dtrtri_8x1_lib4
	.section	.text.unlikely
.LCOLDE20:
	.text
.LHOTE20:
	.section	.text.unlikely
.LCOLDB21:
	.text
.LHOTB21:
	.p2align 4,,15
	.globl	corner_dtrinv_4x4_lib4
	.type	corner_dtrinv_4x4_lib4, @function
corner_dtrinv_4x4_lib4:
.LFB4600:
	.cfi_startproc
	vmovsd	16(%rdi), %xmm2
	vmovsd	72(%rdi), %xmm7
	vmovsd	.LC13(%rip), %xmm5
	vmovapd	%xmm2, %xmm1
	vmovapd	%xmm7, %xmm3
	vmovsd	(%rdi), %xmm0
	vmovsd	40(%rdi), %xmm6
	vxorpd	%xmm5, %xmm1, %xmm1
	vmovsd	%xmm0, (%rsi)
	vxorpd	%xmm5, %xmm3, %xmm3
	vmovsd	%xmm2, 40(%rsi)
	vmovsd	%xmm6, 80(%rsi)
	vmovsd	%xmm7, 120(%rsi)
	vmulsd	8(%rdi), %xmm1, %xmm1
	vmulsd	64(%rdi), %xmm3, %xmm3
	vmulsd	%xmm1, %xmm0, %xmm1
	vmulsd	%xmm3, %xmm6, %xmm3
	vmovsd	%xmm1, 32(%rsi)
	vmovsd	%xmm3, 112(%rsi)
	vmovsd	32(%rdi), %xmm10
	vmovsd	56(%rdi), %xmm8
	vmulsd	%xmm10, %xmm1, %xmm9
	vmulsd	%xmm8, %xmm1, %xmm4
	vmulsd	24(%rdi), %xmm0, %xmm1
	vmulsd	%xmm10, %xmm2, %xmm10
	vaddsd	%xmm9, %xmm1, %xmm9
	vmulsd	48(%rdi), %xmm0, %xmm1
	vmulsd	%xmm9, %xmm3, %xmm11
	vmulsd	%xmm10, %xmm3, %xmm3
	vaddsd	%xmm4, %xmm1, %xmm0
	vmulsd	%xmm9, %xmm6, %xmm9
	vmulsd	%xmm10, %xmm6, %xmm6
	vmulsd	%xmm7, %xmm0, %xmm0
	vxorpd	%xmm5, %xmm9, %xmm9
	vxorpd	%xmm5, %xmm6, %xmm6
	vaddsd	%xmm11, %xmm0, %xmm0
	vmovsd	%xmm9, 64(%rsi)
	vmovsd	%xmm6, 72(%rsi)
	vxorpd	%xmm5, %xmm0, %xmm0
	vmovsd	%xmm0, 96(%rsi)
	vmulsd	%xmm8, %xmm2, %xmm0
	vmulsd	%xmm7, %xmm0, %xmm0
	vaddsd	%xmm3, %xmm0, %xmm2
	vxorpd	%xmm5, %xmm2, %xmm2
	vmovsd	%xmm2, 104(%rsi)
	ret
	.cfi_endproc
.LFE4600:
	.size	corner_dtrinv_4x4_lib4, .-corner_dtrinv_4x4_lib4
	.section	.text.unlikely
.LCOLDE21:
	.text
.LHOTE21:
	.section	.text.unlikely
.LCOLDB22:
	.text
.LHOTB22:
	.p2align 4,,15
	.globl	corner_dtrinv_2x2_lib4
	.type	corner_dtrinv_2x2_lib4, @function
corner_dtrinv_2x2_lib4:
.LFB4601:
	.cfi_startproc
	vmovsd	16(%rdi), %xmm0
	vmovsd	.LC13(%rip), %xmm2
	vmovsd	(%rdi), %xmm1
	vmovsd	%xmm0, 40(%rsi)
	vxorpd	%xmm2, %xmm0, %xmm0
	vmovsd	%xmm1, (%rsi)
	vmulsd	8(%rdi), %xmm0, %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 32(%rsi)
	ret
	.cfi_endproc
.LFE4601:
	.size	corner_dtrinv_2x2_lib4, .-corner_dtrinv_2x2_lib4
	.section	.text.unlikely
.LCOLDE22:
	.text
.LHOTE22:
	.section	.text.unlikely
.LCOLDB23:
	.text
.LHOTB23:
	.p2align 4,,15
	.globl	kernel_dtrinv_8x4_lib4
	.type	kernel_dtrinv_8x4_lib4, @function
kernel_dtrinv_8x4_lib4:
.LFB4602:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	sall	$2, %edx
	vxorpd	%xmm0, %xmm0, %xmm0
	movslq	%edx, %rdx
	pushq	-8(%r10)
	pushq	%rbp
	leaq	(%rsi,%rdx,8), %rax
	sall	$2, %r9d
	addq	$256, %rcx
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	movslq	%r9d, %r9
	addq	$256, %rax
	leaq	(%r8,%r9,8), %r9
	subq	$560, %rsp
	movq	(%r10), %r10
	vmovapd	-256(%rcx), %ymm12
	vblendpd	$1, -128(%rax), %ymm0, %ymm3
	vblendpd	$1, (%rsi), %ymm0, %ymm1
	vmovapd	-224(%rcx), %ymm10
	vshufpd	$5, %ymm12, %ymm12, %ymm14
	vmovapd	-192(%rcx), %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm15
	vmovapd	-160(%rcx), %ymm7
	vshufpd	$5, %ymm9, %ymm9, %ymm8
	vmovapd	-128(%rcx), %ymm6
	vperm2f128	$1, %ymm14, %ymm14, %ymm5
	vmovapd	%ymm3, -112(%rbp)
	vshufpd	$5, %ymm6, %ymm6, %ymm3
	vmovapd	%ymm1, -80(%rbp)
	vshufpd	$5, %ymm5, %ymm5, %ymm2
	vmovapd	%ymm5, -272(%rbp)
	vblendpd	$3, 32(%rsi), %ymm0, %ymm5
	vmovapd	%ymm2, -304(%rbp)
	vperm2f128	$1, %ymm15, %ymm15, %ymm2
	vmovapd	%ymm5, -144(%rbp)
	vmovapd	-96(%rcx), %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm4
	vmovapd	%ymm2, -336(%rbp)
	vblendpd	$7, 64(%rsi), %ymm0, %ymm2
	vmovapd	%ymm2, -176(%rbp)
	vshufpd	$5, %ymm5, %ymm5, %ymm2
	vmovapd	%ymm4, -368(%rbp)
	vperm2f128	$1, %ymm8, %ymm8, %ymm4
	vshufpd	$5, %ymm4, %ymm4, %ymm11
	vmovapd	%ymm4, -400(%rbp)
	vperm2f128	$1, %ymm3, %ymm3, %ymm4
	vmovapd	%ymm11, -432(%rbp)
	vshufpd	$5, %ymm7, %ymm7, %ymm11
	vmovapd	%ymm4, -528(%rbp)
	vperm2f128	$1, %ymm11, %ymm11, %ymm13
	vshufpd	$5, %ymm13, %ymm13, %ymm1
	vmovapd	%ymm13, -464(%rbp)
	vperm2f128	$1, %ymm2, %ymm2, %ymm13
	vmovapd	%ymm1, -496(%rbp)
	vshufpd	$5, %ymm4, %ymm4, %ymm1
	vmovapd	%ymm1, -560(%rbp)
	vblendpd	$3, -96(%rax), %ymm0, %ymm4
	vmulpd	-80(%rbp), %ymm12, %ymm12
	vmovapd	%ymm13, -592(%rbp)
	vmovapd	%ymm4, -208(%rbp)
	vmovapd	-64(%rcx), %ymm4
	vshufpd	$5, %ymm13, %ymm13, %ymm1
	vblendpd	$7, -64(%rax), %ymm0, %ymm13
	vmulpd	-144(%rbp), %ymm10, %ymm10
	vmovapd	%ymm13, -48(%rbp)
	vmulpd	-176(%rbp), %ymm9, %ymm9
	vmovapd	%ymm1, -624(%rbp)
	vsubpd	%ymm12, %ymm0, %ymm12
	vshufpd	$5, %ymm4, %ymm4, %ymm1
	vmulpd	96(%rsi), %ymm7, %ymm7
	vperm2f128	$1, %ymm1, %ymm1, %ymm13
	vsubpd	%ymm10, %ymm12, %ymm10
	vmulpd	160(%rsi), %ymm5, %ymm12
	vmulpd	-208(%rbp), %ymm5, %ymm5
	vmovapd	%ymm13, -240(%rbp)
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vsubpd	%ymm9, %ymm10, %ymm10
	vmulpd	128(%rsi), %ymm6, %ymm9
	vmovapd	%ymm13, -656(%rbp)
	vmovapd	-32(%rcx), %ymm13
	vsubpd	%ymm7, %ymm10, %ymm10
	vmulpd	192(%rsi), %ymm4, %ymm7
	vmulpd	-48(%rbp), %ymm4, %ymm4
	vsubpd	%ymm9, %ymm10, %ymm10
	vmovapd	-112(%rbp), %ymm9
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vmulpd	%ymm9, %ymm6, %ymm6
	vsubpd	%ymm12, %ymm10, %ymm10
	vmovapd	-32(%rcx), %ymm12
	vsubpd	%ymm6, %ymm0, %ymm6
	vmulpd	224(%rsi), %ymm12, %ymm12
	vsubpd	%ymm7, %ymm10, %ymm10
	vmovapd	-144(%rbp), %ymm7
	vsubpd	%ymm5, %ymm6, %ymm5
	vmovapd	-80(%rbp), %ymm6
	vmulpd	%ymm7, %ymm15, %ymm15
	vmulpd	%ymm6, %ymm14, %ymm14
	vsubpd	%ymm12, %ymm10, %ymm12
	vmovapd	(%rcx), %ymm10
	vmovapd	%ymm10, -688(%rbp)
	vsubpd	%ymm4, %ymm5, %ymm5
	vmovapd	-32(%rcx), %ymm10
	vperm2f128	$1, %ymm13, %ymm13, %ymm4
	vsubpd	%ymm14, %ymm0, %ymm14
	vmulpd	-32(%rax), %ymm10, %ymm10
	vsubpd	%ymm15, %ymm14, %ymm15
	vmovapd	-176(%rbp), %ymm14
	vmulpd	%ymm14, %ymm8, %ymm8
	vsubpd	%ymm10, %ymm5, %ymm10
	vsubpd	%ymm8, %ymm15, %ymm15
	vmulpd	96(%rsi), %ymm11, %ymm8
	vmovapd	%ymm10, -112(%rbp)
	vmovapd	%ymm9, %ymm10
	vsubpd	%ymm8, %ymm15, %ymm15
	vmulpd	128(%rsi), %ymm3, %ymm8
	vmulpd	160(%rsi), %ymm2, %ymm5
	vmulpd	224(%rsi), %ymm13, %ymm11
	vmulpd	%ymm9, %ymm3, %ymm3
	vsubpd	%ymm8, %ymm15, %ymm15
	vmulpd	-32(%rax), %ymm13, %ymm9
	vmovapd	%ymm7, %ymm13
	vsubpd	%ymm3, %ymm0, %ymm3
	vsubpd	%ymm5, %ymm15, %ymm15
	vmulpd	192(%rsi), %ymm1, %ymm5
	vmulpd	-48(%rbp), %ymm1, %ymm1
	vsubpd	%ymm5, %ymm15, %ymm15
	vsubpd	%ymm11, %ymm15, %ymm8
	vmovapd	-208(%rbp), %ymm15
	vmulpd	224(%rsi), %ymm4, %ymm11
	vmulpd	%ymm15, %ymm2, %ymm2
	vmovapd	%ymm8, -144(%rbp)
	vsubpd	%ymm2, %ymm3, %ymm2
	vmulpd	-336(%rbp), %ymm7, %ymm3
	vmovapd	-528(%rbp), %ymm7
	vsubpd	%ymm1, %ymm2, %ymm2
	vshufpd	$5, %ymm4, %ymm4, %ymm1
	vsubpd	%ymm9, %ymm2, %ymm8
	vmulpd	-272(%rbp), %ymm6, %ymm2
	vmovapd	%ymm8, -176(%rbp)
	vmovapd	%ymm14, %ymm8
	vsubpd	%ymm2, %ymm0, %ymm2
	vsubpd	%ymm3, %ymm2, %ymm2
	vmulpd	-400(%rbp), %ymm14, %ymm3
	vmovapd	-592(%rbp), %ymm14
	vmulpd	160(%rsi), %ymm14, %ymm5
	vsubpd	%ymm3, %ymm2, %ymm2
	vmovapd	-464(%rbp), %ymm3
	vmulpd	96(%rsi), %ymm3, %ymm3
	vsubpd	%ymm3, %ymm2, %ymm3
	vmulpd	128(%rsi), %ymm7, %ymm2
	vsubpd	%ymm2, %ymm3, %ymm3
	vmovapd	-240(%rbp), %ymm2
	vmulpd	192(%rsi), %ymm2, %ymm2
	vsubpd	%ymm5, %ymm3, %ymm3
	vsubpd	%ymm2, %ymm3, %ymm3
	vmulpd	%ymm10, %ymm7, %ymm2
	vmulpd	%ymm15, %ymm14, %ymm7
	vsubpd	%ymm11, %ymm3, %ymm9
	vmovapd	-240(%rbp), %ymm3
	vmovapd	-48(%rbp), %ymm11
	vsubpd	%ymm2, %ymm0, %ymm2
	vmulpd	-32(%rax), %ymm4, %ymm14
	vmulpd	-368(%rbp), %ymm13, %ymm4
	vmovapd	%ymm9, -208(%rbp)
	vsubpd	%ymm7, %ymm2, %ymm2
	vmulpd	%ymm11, %ymm3, %ymm7
	vmovapd	256(%rsi), %ymm3
	vsubpd	%ymm7, %ymm2, %ymm2
	vsubpd	%ymm14, %ymm2, %ymm5
	vmulpd	-304(%rbp), %ymm6, %ymm2
	vmovapd	-560(%rbp), %ymm6
	vmovapd	%ymm5, -240(%rbp)
	vmulpd	-432(%rbp), %ymm8, %ymm5
	vsubpd	%ymm2, %ymm0, %ymm2
	vsubpd	%ymm4, %ymm2, %ymm2
	vsubpd	%ymm5, %ymm2, %ymm5
	vmovapd	-496(%rbp), %ymm2
	vmulpd	96(%rsi), %ymm2, %ymm4
	vsubpd	%ymm4, %ymm5, %ymm5
	vmulpd	128(%rsi), %ymm6, %ymm4
	vmovapd	-624(%rbp), %ymm7
	vmovapd	-656(%rbp), %ymm13
	vmulpd	160(%rsi), %ymm7, %ymm2
	vsubpd	%ymm4, %ymm5, %ymm5
	vmulpd	%ymm10, %ymm6, %ymm4
	vsubpd	%ymm2, %ymm5, %ymm2
	vmulpd	192(%rsi), %ymm13, %ymm5
	vsubpd	%ymm4, %ymm0, %ymm0
	vmulpd	%ymm15, %ymm7, %ymm4
	vsubpd	%ymm4, %ymm0, %ymm0
	vmulpd	%ymm11, %ymm13, %ymm4
	vsubpd	%ymm5, %ymm2, %ymm2
	vmulpd	224(%rsi), %ymm1, %ymm5
	vmulpd	-32(%rax), %ymm1, %ymm1
	vsubpd	%ymm4, %ymm0, %ymm0
	vsubpd	%ymm5, %ymm2, %ymm5
	vmovapd	(%rax), %ymm2
	cmpl	$11, %edi
	vsubpd	%ymm1, %ymm0, %ymm1
	vmovapd	%ymm5, -272(%rbp)
	vmovapd	%ymm1, -304(%rbp)
	jle	.L90
	leal	-12(%rdi), %edx
	leaq	256(%rsi), %r11
	vmovapd	-688(%rbp), %ymm4
	addq	$288, %rsi
	shrl	$2, %edx
	salq	$7, %rdx
	leaq	160(%r11,%rdx), %rdx
	.p2align 4,,10
	.p2align 3
.L91:
	vmulpd	%ymm4, %ymm3, %ymm15
	vmovapd	32(%rcx), %ymm7
	subq	$-128, %rsi
	vmovapd	-96(%rsi), %ymm10
	subq	$-128, %rax
	subq	$-128, %rcx
	vmovapd	-64(%rcx), %ymm13
	vshufpd	$5, %ymm4, %ymm4, %ymm6
	vmulpd	%ymm4, %ymm2, %ymm4
	vmovapd	-96(%rax), %ymm14
	vmovapd	-64(%rax), %ymm9
	vsubpd	%ymm15, %ymm12, %ymm12
	vmulpd	-128(%rsi), %ymm7, %ymm15
	vshufpd	$5, %ymm7, %ymm7, %ymm1
	vmulpd	%ymm14, %ymm7, %ymm7
	vshufpd	$5, %ymm13, %ymm13, %ymm11
	vperm2f128	$1, %ymm6, %ymm6, %ymm5
	vsubpd	%ymm15, %ymm12, %ymm12
	vmulpd	%ymm10, %ymm13, %ymm15
	vshufpd	$5, %ymm5, %ymm5, %ymm0
	vmulpd	%ymm9, %ymm13, %ymm13
	vsubpd	%ymm15, %ymm12, %ymm12
	vmovapd	-112(%rbp), %ymm15
	vmovapd	%ymm0, -48(%rbp)
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vsubpd	%ymm4, %ymm15, %ymm4
	vmulpd	%ymm3, %ymm6, %ymm15
	vmulpd	%ymm2, %ymm6, %ymm6
	vsubpd	%ymm7, %ymm4, %ymm4
	vmovapd	-144(%rbp), %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm8
	vsubpd	%ymm15, %ymm7, %ymm15
	vmulpd	-128(%rsi), %ymm1, %ymm7
	vmulpd	%ymm1, %ymm14, %ymm1
	vmovapd	%ymm8, -80(%rbp)
	vsubpd	%ymm13, %ymm4, %ymm13
	vmovapd	-32(%rcx), %ymm8
	vperm2f128	$1, %ymm11, %ymm11, %ymm4
	vsubpd	%ymm7, %ymm15, %ymm7
	vmulpd	%ymm11, %ymm10, %ymm15
	vmulpd	%ymm11, %ymm9, %ymm11
	vsubpd	%ymm15, %ymm7, %ymm15
	vmovapd	-176(%rbp), %ymm7
	vsubpd	%ymm6, %ymm7, %ymm6
	vshufpd	$5, %ymm4, %ymm4, %ymm7
	vsubpd	%ymm1, %ymm6, %ymm6
	vmulpd	%ymm3, %ymm5, %ymm1
	vmulpd	%ymm2, %ymm5, %ymm5
	vsubpd	%ymm11, %ymm6, %ymm11
	vmovapd	-208(%rbp), %ymm6
	vmulpd	-48(%rbp), %ymm3, %ymm3
	vsubpd	%ymm1, %ymm6, %ymm1
	vmulpd	-128(%rsi), %ymm0, %ymm6
	vmulpd	%ymm0, %ymm14, %ymm0
	vmulpd	-48(%rbp), %ymm2, %ymm2
	vsubpd	%ymm6, %ymm1, %ymm1
	vmulpd	%ymm4, %ymm10, %ymm6
	vmulpd	%ymm4, %ymm9, %ymm4
	vmulpd	%ymm7, %ymm10, %ymm10
	vsubpd	%ymm6, %ymm1, %ymm1
	vmovapd	-240(%rbp), %ymm6
	vmulpd	%ymm7, %ymm9, %ymm9
	vsubpd	%ymm5, %ymm6, %ymm5
	vmovapd	-64(%rsi), %ymm6
	vsubpd	%ymm0, %ymm5, %ymm0
	vmovapd	-32(%rax), %ymm5
	vsubpd	%ymm4, %ymm0, %ymm0
	vmovapd	-272(%rbp), %ymm4
	vsubpd	%ymm3, %ymm4, %ymm3
	vmovapd	-80(%rbp), %ymm4
	vmulpd	-128(%rsi), %ymm4, %ymm4
	vsubpd	%ymm4, %ymm3, %ymm3
	vmovapd	(%rcx), %ymm4
	vsubpd	%ymm10, %ymm3, %ymm10
	vmovapd	-304(%rbp), %ymm3
	vsubpd	%ymm2, %ymm3, %ymm2
	vmulpd	-80(%rbp), %ymm14, %ymm3
	vsubpd	%ymm3, %ymm2, %ymm2
	vmulpd	%ymm6, %ymm8, %ymm3
	vsubpd	%ymm9, %ymm2, %ymm9
	vshufpd	$5, %ymm8, %ymm8, %ymm2
	vmulpd	%ymm5, %ymm8, %ymm8
	vsubpd	%ymm3, %ymm12, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm3
	vsubpd	%ymm8, %ymm13, %ymm7
	vmovapd	%ymm7, -112(%rbp)
	vmulpd	%ymm2, %ymm6, %ymm7
	vmulpd	%ymm2, %ymm5, %ymm2
	vsubpd	%ymm7, %ymm15, %ymm7
	vsubpd	%ymm2, %ymm11, %ymm2
	vmovapd	%ymm7, -144(%rbp)
	vshufpd	$5, %ymm3, %ymm3, %ymm7
	vmovapd	%ymm2, -176(%rbp)
	vmulpd	%ymm3, %ymm6, %ymm2
	vmulpd	%ymm3, %ymm5, %ymm3
	vmulpd	%ymm7, %ymm6, %ymm6
	vsubpd	%ymm2, %ymm1, %ymm1
	vmulpd	%ymm7, %ymm5, %ymm5
	vsubpd	%ymm3, %ymm0, %ymm0
	vmovapd	-32(%rsi), %ymm3
	vmovapd	%ymm1, -208(%rbp)
	vsubpd	%ymm5, %ymm9, %ymm5
	vmovapd	%ymm0, -240(%rbp)
	vsubpd	%ymm6, %ymm10, %ymm0
	vmovapd	(%rax), %ymm2
	cmpq	%rsi, %rdx
	vmovapd	%ymm5, -304(%rbp)
	vmovapd	%ymm0, -272(%rbp)
	jne	.L91
.L90:
	vmovapd	-144(%rbp), %ymm0
	vmovapd	-112(%rbp), %ymm2
	vblendpd	$10, %ymm0, %ymm12, %ymm1
	vmovapd	-176(%rbp), %ymm3
	vblendpd	$5, %ymm0, %ymm12, %ymm12
	vmovapd	-208(%rbp), %ymm5
	vmovapd	-272(%rbp), %ymm0
	vblendpd	$5, %ymm3, %ymm2, %ymm9
	vmovapd	-240(%rbp), %ymm6
	vblendpd	$10, %ymm5, %ymm0, %ymm15
	vblendpd	$5, %ymm5, %ymm0, %ymm5
	vblendpd	$10, %ymm3, %ymm2, %ymm0
	vmovapd	-304(%rbp), %ymm2
	vblendpd	$12, %ymm5, %ymm12, %ymm3
	vblendpd	$3, %ymm5, %ymm12, %ymm5
	vblendpd	$10, %ymm6, %ymm2, %ymm8
	vblendpd	$5, %ymm6, %ymm2, %ymm14
	vblendpd	$12, %ymm15, %ymm1, %ymm6
	vblendpd	$3, %ymm15, %ymm1, %ymm15
	vblendpd	$12, %ymm8, %ymm0, %ymm4
	vblendpd	$3, %ymm8, %ymm0, %ymm8
	vbroadcastsd	(%r10), %ymm0
	vblendpd	$12, %ymm14, %ymm9, %ymm2
	vblendpd	$3, %ymm14, %ymm9, %ymm10
	vmulpd	%ymm0, %ymm4, %ymm4
	vmulpd	%ymm0, %ymm6, %ymm6
	vmovapd	%ymm6, (%r8)
	vmovapd	%ymm4, (%r9)
	vbroadcastsd	8(%r10), %ymm7
	vbroadcastsd	16(%r10), %ymm14
	vmulpd	%ymm7, %ymm6, %ymm0
	vmulpd	%ymm7, %ymm4, %ymm7
	vsubpd	%ymm0, %ymm3, %ymm3
	vsubpd	%ymm7, %ymm2, %ymm7
	vmulpd	%ymm14, %ymm3, %ymm3
	vmulpd	%ymm14, %ymm7, %ymm2
	vmovapd	%ymm3, 32(%r8)
	vmovapd	%ymm2, 32(%r9)
	vbroadcastsd	24(%r10), %ymm0
	vbroadcastsd	32(%r10), %ymm9
	vmulpd	%ymm0, %ymm6, %ymm1
	vmulpd	%ymm9, %ymm3, %ymm7
	vmulpd	%ymm0, %ymm4, %ymm0
	vsubpd	%ymm1, %ymm15, %ymm1
	vbroadcastsd	40(%r10), %ymm11
	vsubpd	%ymm7, %ymm1, %ymm1
	vsubpd	%ymm0, %ymm8, %ymm7
	vmulpd	%ymm9, %ymm2, %ymm0
	vmulpd	%ymm11, %ymm1, %ymm1
	vsubpd	%ymm0, %ymm7, %ymm0
	vmovapd	%ymm1, 64(%r8)
	vmulpd	%ymm11, %ymm0, %ymm0
	vmovapd	%ymm0, 64(%r9)
	vbroadcastsd	48(%r10), %ymm14
	vbroadcastsd	56(%r10), %ymm11
	vmulpd	%ymm14, %ymm6, %ymm6
	vmulpd	%ymm14, %ymm4, %ymm14
	vmulpd	%ymm11, %ymm3, %ymm3
	vsubpd	%ymm6, %ymm5, %ymm5
	vbroadcastsd	64(%r10), %ymm9
	vsubpd	%ymm14, %ymm10, %ymm7
	vmulpd	%ymm11, %ymm2, %ymm14
	vmulpd	%ymm9, %ymm1, %ymm1
	vsubpd	%ymm3, %ymm5, %ymm5
	vmulpd	%ymm9, %ymm0, %ymm0
	vsubpd	%ymm14, %ymm7, %ymm14
	vbroadcastsd	72(%r10), %ymm8
	vsubpd	%ymm1, %ymm5, %ymm1
	vsubpd	%ymm0, %ymm14, %ymm0
	vmulpd	%ymm8, %ymm1, %ymm1
	vmulpd	%ymm8, %ymm0, %ymm0
	vmovapd	%ymm1, 96(%r8)
	vmovapd	%ymm0, 96(%r9)
	vzeroupper
	addq	$560, %rsp
	popq	%r10
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4602:
	.size	kernel_dtrinv_8x4_lib4, .-kernel_dtrinv_8x4_lib4
	.section	.text.unlikely
.LCOLDE23:
	.text
.LHOTE23:
	.section	.text.unlikely
.LCOLDB24:
	.text
.LHOTB24:
	.p2align 4,,15
	.globl	kernel_dtrinv_4x4_lib4
	.type	kernel_dtrinv_4x4_lib4, @function
kernel_dtrinv_4x4_lib4:
.LFB4603:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	subq	$-128, %rsi
	vxorpd	%xmm0, %xmm0, %xmm0
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	subq	$16, %rsp
	vblendpd	$1, -128(%rsi), %ymm0, %ymm12
	vmovapd	(%rdx), %ymm2
	vblendpd	$3, -96(%rsi), %ymm0, %ymm11
	vmovapd	32(%rdx), %ymm7
	vblendpd	$7, -64(%rsi), %ymm0, %ymm10
	vshufpd	$5, %ymm2, %ymm2, %ymm5
	vmovapd	64(%rdx), %ymm6
	vmulpd	%ymm12, %ymm2, %ymm2
	vmovapd	96(%rdx), %ymm8
	vperm2f128	$1, %ymm5, %ymm5, %ymm1
	vshufpd	$5, %ymm7, %ymm7, %ymm4
	vmulpd	%ymm5, %ymm12, %ymm5
	vsubpd	%ymm2, %ymm0, %ymm2
	vmulpd	%ymm11, %ymm7, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm3
	vperm2f128	$1, %ymm4, %ymm4, %ymm15
	vsubpd	%ymm5, %ymm0, %ymm5
	vmulpd	%ymm1, %ymm12, %ymm1
	vmovapd	%ymm3, -80(%rbp)
	vsubpd	%ymm7, %ymm2, %ymm7
	vshufpd	$5, %ymm15, %ymm15, %ymm14
	vshufpd	$5, %ymm6, %ymm6, %ymm3
	vmulpd	%ymm4, %ymm11, %ymm4
	vsubpd	%ymm1, %ymm0, %ymm1
	vmovapd	%ymm14, -112(%rbp)
	vmulpd	%ymm10, %ymm6, %ymm6
	vmulpd	%ymm15, %ymm11, %ymm15
	vsubpd	%ymm4, %ymm5, %ymm4
	vmulpd	-80(%rbp), %ymm12, %ymm12
	vsubpd	%ymm6, %ymm7, %ymm7
	vshufpd	$5, %ymm8, %ymm8, %ymm14
	vmulpd	-32(%rsi), %ymm8, %ymm8
	vsubpd	%ymm15, %ymm1, %ymm15
	vperm2f128	$1, %ymm3, %ymm3, %ymm9
	vmulpd	%ymm3, %ymm10, %ymm3
	vsubpd	%ymm12, %ymm0, %ymm0
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vperm2f128	$1, %ymm14, %ymm14, %ymm6
	vmulpd	%ymm9, %ymm10, %ymm2
	vsubpd	%ymm8, %ymm7, %ymm7
	vmovapd	%ymm13, -144(%rbp)
	vmulpd	-112(%rbp), %ymm11, %ymm11
	vsubpd	%ymm3, %ymm4, %ymm4
	vmovapd	128(%rdx), %ymm13
	vmulpd	-32(%rsi), %ymm14, %ymm8
	vsubpd	%ymm2, %ymm15, %ymm2
	vmovapd	(%rsi), %ymm15
	vshufpd	$5, %ymm6, %ymm6, %ymm3
	vmovapd	%ymm7, -48(%rbp)
	vmulpd	-32(%rsi), %ymm6, %ymm6
	vmulpd	-144(%rbp), %ymm10, %ymm10
	vsubpd	%ymm11, %ymm0, %ymm11
	vsubpd	%ymm8, %ymm4, %ymm8
	vsubpd	%ymm6, %ymm2, %ymm4
	vmulpd	-32(%rsi), %ymm3, %ymm2
	cmpl	$7, %edi
	vsubpd	%ymm10, %ymm11, %ymm10
	vsubpd	%ymm2, %ymm10, %ymm2
	jle	.L95
	subl	$8, %edi
	leaq	160(%rdx), %rax
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	288(%rdx,%rdi), %rdx
	.p2align 4,,10
	.p2align 3
.L96:
	vshufpd	$5, %ymm13, %ymm13, %ymm1
	vmovapd	(%rax), %ymm3
	subq	$-128, %rax
	vmulpd	%ymm13, %ymm15, %ymm13
	vmovapd	32(%rsi), %ymm6
	vmovapd	%ymm3, -80(%rbp)
	subq	$-128, %rsi
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vmovapd	-96(%rax), %ymm12
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vmovapd	-48(%rbp), %ymm3
	vmulpd	%ymm15, %ymm1, %ymm1
	vmovapd	-64(%rsi), %ymm5
	vsubpd	%ymm13, %ymm3, %ymm13
	vshufpd	$5, %ymm0, %ymm0, %ymm7
	vmulpd	-80(%rbp), %ymm6, %ymm3
	vmulpd	%ymm15, %ymm0, %ymm0
	vmovapd	%ymm7, -112(%rbp)
	vsubpd	%ymm1, %ymm8, %ymm8
	vmovapd	-64(%rax), %ymm7
	vperm2f128	$1, %ymm11, %ymm11, %ymm9
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vmulpd	%ymm11, %ymm6, %ymm11
	vsubpd	%ymm0, %ymm4, %ymm4
	vmulpd	-112(%rbp), %ymm15, %ymm15
	vsubpd	%ymm3, %ymm13, %ymm13
	vshufpd	$5, %ymm9, %ymm9, %ymm14
	vmulpd	%ymm9, %ymm6, %ymm9
	vsubpd	%ymm11, %ymm8, %ymm11
	vmulpd	%ymm5, %ymm12, %ymm3
	vperm2f128	$1, %ymm10, %ymm10, %ymm12
	vmulpd	%ymm10, %ymm5, %ymm10
	vsubpd	%ymm9, %ymm4, %ymm4
	vmulpd	%ymm12, %ymm5, %ymm9
	vsubpd	%ymm15, %ymm2, %ymm2
	vmovapd	(%rsi), %ymm15
	vmulpd	%ymm14, %ymm6, %ymm6
	vsubpd	%ymm10, %ymm11, %ymm10
	vmovapd	-32(%rsi), %ymm11
	vshufpd	$5, %ymm7, %ymm7, %ymm8
	vshufpd	$5, %ymm12, %ymm12, %ymm1
	vsubpd	%ymm9, %ymm4, %ymm9
	vmulpd	%ymm11, %ymm7, %ymm7
	vsubpd	%ymm3, %ymm13, %ymm3
	vmovapd	-32(%rax), %ymm13
	cmpq	%rax, %rdx
	vperm2f128	$1, %ymm8, %ymm8, %ymm4
	vsubpd	%ymm6, %ymm2, %ymm6
	vmulpd	%ymm1, %ymm5, %ymm5
	vshufpd	$5, %ymm4, %ymm4, %ymm2
	vsubpd	%ymm7, %ymm3, %ymm7
	vmulpd	%ymm8, %ymm11, %ymm8
	vsubpd	%ymm5, %ymm6, %ymm5
	vmulpd	%ymm4, %ymm11, %ymm4
	vmulpd	%ymm2, %ymm11, %ymm2
	vmovapd	%ymm7, -48(%rbp)
	vsubpd	%ymm8, %ymm10, %ymm8
	vsubpd	%ymm4, %ymm9, %ymm4
	vsubpd	%ymm2, %ymm5, %ymm2
	jne	.L96
.L95:
	vmovapd	-48(%rbp), %ymm1
	vblendpd	$10, %ymm8, %ymm1, %ymm0
	vblendpd	$5, %ymm8, %ymm1, %ymm13
	vblendpd	$10, %ymm4, %ymm2, %ymm1
	vblendpd	$5, %ymm4, %ymm2, %ymm2
	vblendpd	$12, %ymm1, %ymm0, %ymm5
	vblendpd	$3, %ymm1, %ymm0, %ymm0
	vbroadcastsd	(%r8), %ymm1
	vblendpd	$12, %ymm2, %ymm13, %ymm4
	vblendpd	$3, %ymm2, %ymm13, %ymm8
	vmulpd	%ymm1, %ymm5, %ymm5
	vmovapd	%ymm5, (%rcx)
	vbroadcastsd	8(%r8), %ymm3
	vbroadcastsd	16(%r8), %ymm2
	vmulpd	%ymm3, %ymm5, %ymm3
	vsubpd	%ymm3, %ymm4, %ymm3
	vmulpd	%ymm2, %ymm3, %ymm4
	vmovapd	%ymm4, 32(%rcx)
	vbroadcastsd	24(%r8), %ymm1
	vbroadcastsd	32(%r8), %ymm3
	vmulpd	%ymm1, %ymm5, %ymm1
	vbroadcastsd	40(%r8), %ymm2
	vsubpd	%ymm1, %ymm0, %ymm1
	vmulpd	%ymm3, %ymm4, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmulpd	%ymm2, %ymm0, %ymm0
	vmovapd	%ymm0, 64(%rcx)
	vbroadcastsd	48(%r8), %ymm13
	vbroadcastsd	56(%r8), %ymm2
	vmulpd	%ymm13, %ymm5, %ymm13
	vmulpd	%ymm2, %ymm4, %ymm2
	vbroadcastsd	64(%r8), %ymm3
	vsubpd	%ymm13, %ymm8, %ymm13
	vbroadcastsd	72(%r8), %ymm1
	vmulpd	%ymm3, %ymm0, %ymm0
	vsubpd	%ymm2, %ymm13, %ymm2
	vsubpd	%ymm0, %ymm2, %ymm0
	vmulpd	%ymm1, %ymm0, %ymm0
	vmovapd	%ymm0, 96(%rcx)
	vzeroupper
	addq	$16, %rsp
	popq	%r10
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4603:
	.size	kernel_dtrinv_4x4_lib4, .-kernel_dtrinv_4x4_lib4
	.section	.text.unlikely
.LCOLDE24:
	.text
.LHOTE24:
	.section	.text.unlikely
.LCOLDB26:
	.text
.LHOTB26:
	.p2align 4,,15
	.globl	kernel_dtrinv_4x2_lib4
	.type	kernel_dtrinv_4x2_lib4, @function
kernel_dtrinv_4x2_lib4:
.LFB4604:
	.cfi_startproc
	vmovsd	(%rsi), %xmm0
	subq	$-128, %rdx
	vxorpd	%xmm12, %xmm12, %xmm12
	leaq	128(%rsi), %rax
	vmulsd	-128(%rdx), %xmm0, %xmm3
	vmulsd	-120(%rdx), %xmm0, %xmm0
	vmovsd	-96(%rdx), %xmm5
	vmovsd	-88(%rdx), %xmm11
	vmovsd	32(%rsi), %xmm7
	vmovsd	-64(%rdx), %xmm6
	vmulsd	%xmm5, %xmm7, %xmm8
	vsubsd	%xmm3, %xmm12, %xmm2
	vmulsd	%xmm11, %xmm7, %xmm7
	vsubsd	%xmm0, %xmm12, %xmm0
	vmovsd	-56(%rdx), %xmm10
	vmovsd	40(%rsi), %xmm1
	vmovsd	64(%rsi), %xmm4
	vsubsd	%xmm8, %xmm2, %xmm3
	vsubsd	%xmm7, %xmm0, %xmm0
	vmulsd	%xmm11, %xmm1, %xmm11
	vmulsd	%xmm6, %xmm4, %xmm8
	vmulsd	%xmm10, %xmm4, %xmm4
	vmulsd	%xmm5, %xmm1, %xmm5
	vmovsd	72(%rsi), %xmm1
	vsubsd	%xmm11, %xmm12, %xmm11
	vmovsd	80(%rsi), %xmm2
	vsubsd	%xmm8, %xmm3, %xmm8
	vsubsd	%xmm4, %xmm0, %xmm7
	vmulsd	%xmm6, %xmm1, %xmm3
	vmovsd	-32(%rdx), %xmm0
	vsubsd	%xmm5, %xmm12, %xmm5
	vmulsd	%xmm10, %xmm1, %xmm1
	vmovsd	96(%rsi), %xmm4
	vmulsd	%xmm10, %xmm2, %xmm10
	vmulsd	%xmm6, %xmm2, %xmm6
	vsubsd	%xmm3, %xmm5, %xmm5
	vmulsd	%xmm0, %xmm4, %xmm2
	vsubsd	%xmm1, %xmm11, %xmm13
	vmovsd	104(%rsi), %xmm11
	vmovsd	-24(%rdx), %xmm9
	vsubsd	%xmm10, %xmm12, %xmm14
	cmpl	$7, %edi
	vmovsd	112(%rsi), %xmm10
	vsubsd	%xmm6, %xmm12, %xmm6
	vsubsd	%xmm2, %xmm8, %xmm8
	vmulsd	%xmm0, %xmm11, %xmm2
	vmovsd	120(%rsi), %xmm1
	vmulsd	%xmm9, %xmm4, %xmm4
	vmulsd	%xmm9, %xmm11, %xmm11
	vsubsd	%xmm2, %xmm5, %xmm5
	vmulsd	%xmm0, %xmm10, %xmm2
	vmulsd	%xmm0, %xmm1, %xmm0
	vmulsd	%xmm9, %xmm10, %xmm10
	vsubsd	%xmm4, %xmm7, %xmm4
	vmulsd	%xmm9, %xmm1, %xmm1
	vsubsd	%xmm11, %xmm13, %xmm11
	vsubsd	%xmm2, %xmm6, %xmm6
	vsubsd	%xmm0, %xmm12, %xmm0
	vsubsd	%xmm10, %xmm14, %xmm10
	vsubsd	%xmm1, %xmm12, %xmm9
	jle	.L100
	subl	$8, %edi
	shrl	$2, %edi
	addq	$2, %rdi
	salq	$7, %rdi
	addq	%rdi, %rsi
	.p2align 4,,10
	.p2align 3
.L101:
	vmovsd	(%rdx), %xmm7
	subq	$-128, %rax
	subq	$-128, %rdx
	vmovsd	-128(%rax), %xmm13
	vmovsd	-120(%rax), %xmm3
	vmulsd	%xmm7, %xmm13, %xmm14
	vmovsd	-112(%rax), %xmm2
	vmovsd	-104(%rax), %xmm1
	vmovsd	-120(%rdx), %xmm12
	vsubsd	%xmm14, %xmm8, %xmm8
	vmulsd	%xmm7, %xmm3, %xmm14
	vmulsd	%xmm12, %xmm13, %xmm13
	vmulsd	%xmm12, %xmm3, %xmm3
	vsubsd	%xmm14, %xmm5, %xmm5
	vmulsd	%xmm7, %xmm2, %xmm14
	vmulsd	%xmm7, %xmm1, %xmm7
	vsubsd	%xmm13, %xmm4, %xmm4
	vmulsd	%xmm12, %xmm2, %xmm2
	vsubsd	%xmm3, %xmm11, %xmm3
	vmulsd	%xmm12, %xmm1, %xmm1
	vmovsd	-96(%rax), %xmm12
	vsubsd	%xmm14, %xmm6, %xmm6
	vsubsd	%xmm7, %xmm0, %xmm0
	vmovsd	-96(%rdx), %xmm7
	vmovsd	-88(%rax), %xmm11
	vsubsd	%xmm2, %xmm10, %xmm2
	vmulsd	%xmm7, %xmm12, %xmm14
	vsubsd	%xmm1, %xmm9, %xmm1
	vmovsd	-88(%rdx), %xmm13
	vmovsd	-80(%rax), %xmm10
	vmovsd	-72(%rax), %xmm9
	vsubsd	%xmm14, %xmm8, %xmm8
	vmulsd	%xmm7, %xmm11, %xmm14
	vmulsd	%xmm13, %xmm12, %xmm12
	vmulsd	%xmm13, %xmm11, %xmm11
	vsubsd	%xmm14, %xmm5, %xmm5
	vmulsd	%xmm7, %xmm10, %xmm14
	vmulsd	%xmm7, %xmm9, %xmm7
	vsubsd	%xmm12, %xmm4, %xmm12
	vmovsd	-64(%rax), %xmm4
	vsubsd	%xmm11, %xmm3, %xmm11
	vmulsd	%xmm13, %xmm10, %xmm10
	vsubsd	%xmm14, %xmm6, %xmm6
	vmovsd	-56(%rax), %xmm3
	vsubsd	%xmm7, %xmm0, %xmm7
	vmovsd	-64(%rdx), %xmm0
	vmulsd	%xmm13, %xmm9, %xmm9
	vmulsd	%xmm0, %xmm4, %xmm14
	vsubsd	%xmm10, %xmm2, %xmm10
	vmovsd	-56(%rdx), %xmm13
	vmovsd	-48(%rax), %xmm2
	vsubsd	%xmm9, %xmm1, %xmm9
	vmovsd	-40(%rax), %xmm1
	vsubsd	%xmm14, %xmm8, %xmm8
	vmulsd	%xmm0, %xmm3, %xmm14
	vmulsd	%xmm13, %xmm4, %xmm4
	vmulsd	%xmm13, %xmm3, %xmm3
	vsubsd	%xmm14, %xmm5, %xmm5
	vmulsd	%xmm0, %xmm2, %xmm14
	vmulsd	%xmm0, %xmm1, %xmm0
	vsubsd	%xmm4, %xmm12, %xmm4
	vmovsd	-32(%rdx), %xmm12
	vsubsd	%xmm3, %xmm11, %xmm3
	vmulsd	%xmm13, %xmm2, %xmm2
	vsubsd	%xmm14, %xmm6, %xmm6
	vmovsd	-24(%rax), %xmm11
	vsubsd	%xmm0, %xmm7, %xmm0
	vmovsd	-32(%rax), %xmm7
	vmulsd	%xmm13, %xmm1, %xmm1
	vmulsd	%xmm12, %xmm7, %xmm14
	vsubsd	%xmm2, %xmm10, %xmm2
	vmovsd	-8(%rax), %xmm13
	vmovsd	-16(%rax), %xmm10
	vsubsd	%xmm1, %xmm9, %xmm1
	vmovsd	-24(%rdx), %xmm9
	vsubsd	%xmm14, %xmm8, %xmm8
	vmulsd	%xmm12, %xmm11, %xmm14
	cmpq	%rsi, %rax
	vmulsd	%xmm9, %xmm7, %xmm7
	vmulsd	%xmm9, %xmm11, %xmm11
	vsubsd	%xmm14, %xmm5, %xmm5
	vmulsd	%xmm12, %xmm10, %xmm14
	vmulsd	%xmm9, %xmm10, %xmm10
	vsubsd	%xmm7, %xmm4, %xmm4
	vmulsd	%xmm12, %xmm13, %xmm12
	vsubsd	%xmm11, %xmm3, %xmm11
	vmulsd	%xmm9, %xmm13, %xmm9
	vsubsd	%xmm14, %xmm6, %xmm6
	vsubsd	%xmm10, %xmm2, %xmm10
	vsubsd	%xmm12, %xmm0, %xmm0
	vsubsd	%xmm9, %xmm1, %xmm9
	jne	.L101
.L100:
	vmovsd	(%r8), %xmm1
	vmulsd	%xmm8, %xmm1, %xmm8
	vmulsd	%xmm5, %xmm1, %xmm5
	vmulsd	%xmm6, %xmm1, %xmm6
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm8, (%rcx)
	vmovsd	%xmm5, 8(%rcx)
	vmovsd	%xmm6, 16(%rcx)
	vmovsd	%xmm0, 24(%rcx)
	vmovsd	8(%r8), %xmm1
	vmovsd	16(%r8), %xmm2
	vmulsd	%xmm1, %xmm8, %xmm8
	vmulsd	%xmm1, %xmm5, %xmm5
	vmulsd	%xmm1, %xmm6, %xmm6
	vmulsd	%xmm1, %xmm0, %xmm1
	vsubsd	%xmm8, %xmm4, %xmm8
	vsubsd	%xmm5, %xmm11, %xmm5
	vsubsd	%xmm6, %xmm10, %xmm6
	vsubsd	%xmm1, %xmm9, %xmm1
	vmulsd	%xmm8, %xmm2, %xmm8
	vmulsd	%xmm5, %xmm2, %xmm5
	vmulsd	%xmm6, %xmm2, %xmm6
	vmulsd	%xmm1, %xmm2, %xmm1
	vmovsd	%xmm8, 32(%rcx)
	vmovsd	%xmm5, 40(%rcx)
	vmovsd	%xmm6, 48(%rcx)
	vmovsd	%xmm1, 56(%rcx)
	ret
	.cfi_endproc
.LFE4604:
	.size	kernel_dtrinv_4x2_lib4, .-kernel_dtrinv_4x2_lib4
	.section	.text.unlikely
.LCOLDE26:
	.text
.LHOTE26:
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC0:
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.align 32
.LC1:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.align 32
.LC2:
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.align 32
.LC3:
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.align 32
.LC4:
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC12:
	.long	0
	.long	1072693248
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC13:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
	.ident	"GCC: (GNU) 5.2.0"
	.section	.note.GNU-stack,"",@progbits
