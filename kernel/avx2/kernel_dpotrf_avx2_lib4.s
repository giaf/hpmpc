	.file	"kernel_dpotrf_avx2_lib4.c"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB5:
	.text
.LHOTB5:
	.p2align 4,,15
	.globl	kernel_dpotrf_nt_12x4_lib4_new
	.type	kernel_dpotrf_nt_12x4_lib4_new, @function
kernel_dpotrf_nt_12x4_lib4_new:
.LFB4589:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x60,0x6
	.cfi_escape 0x10,0xe,0x2,0x76,0x78
	.cfi_escape 0x10,0xd,0x2,0x76,0x70
	.cfi_escape 0x10,0xc,0x2,0x76,0x68
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x58
	movl	16(%r10), %r11d
	movq	8(%r10), %rbx
	movl	(%r10), %r12d
	movq	24(%r10), %r14
	sall	$2, %r11d
	movslq	%r11d, %r11
	salq	$3, %r11
	leaq	(%rbx,%r11), %r13
	addq	%r13, %r11
	testl	%edi, %edi
	jle	.L10
	leal	0(,%rdx,4), %eax
	vmovapd	(%rsi), %ymm13
	vmovapd	(%rcx), %ymm1
	cltq
	salq	$3, %rax
	leaq	(%rsi,%rax), %r10
	addq	%r10, %rax
	cmpl	$3, %edi
	vmovapd	(%r10), %ymm14
	vmovapd	(%rax), %ymm15
	jle	.L10
	vxorpd	%xmm0, %xmm0, %xmm0
	subl	$4, %edi
	leaq	32(%rsi), %rdx
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	160(%rsi,%rdi), %rsi
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm3
	.p2align 4,,10
	.p2align 3
.L3:
	vfmadd231pd	%ymm1, %ymm13, %ymm3
	vfmadd231pd	%ymm1, %ymm14, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	subq	$-128, %rdx
	subq	$-128, %r10
	subq	$-128, %rax
	subq	$-128, %rcx
	vfmadd231pd	%ymm1, %ymm13, %ymm12
	vfmadd231pd	%ymm1, %ymm14, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm11
	vfmadd231pd	%ymm1, %ymm14, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm5
	vfmadd231pd	%ymm1, %ymm14, %ymm4
	vmovapd	-128(%rdx), %ymm13
	vmovapd	-96(%r10), %ymm14
	vfmadd231pd	%ymm1, %ymm15, %ymm0
	vmovapd	-96(%rcx), %ymm1
	vmovapd	-96(%rax), %ymm15
	vfmadd231pd	%ymm1, %ymm13, %ymm3
	vfmadd231pd	%ymm1, %ymm14, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm12
	vfmadd231pd	%ymm1, %ymm14, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm11
	vfmadd231pd	%ymm1, %ymm14, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm5, %ymm13
	vfmadd132pd	%ymm1, %ymm4, %ymm14
	vmovapd	-96(%rdx), %ymm5
	vmovapd	-64(%r10), %ymm4
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vmovapd	-64(%rax), %ymm1
	vmovapd	-64(%rcx), %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm13
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vmovapd	-64(%rdx), %ymm5
	vmovapd	-32(%r10), %ymm4
	vfmadd231pd	%ymm0, %ymm1, %ymm15
	vmovapd	-32(%rax), %ymm1
	vmovapd	-32(%rcx), %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm13, %ymm5
	vmovapd	-32(%rdx), %ymm13
	vfmadd132pd	%ymm0, %ymm14, %ymm4
	vmovapd	(%r10), %ymm14
	vfmadd132pd	%ymm1, %ymm15, %ymm0
	vmovapd	(%rcx), %ymm1
	vmovapd	(%rax), %ymm15
	cmpq	%rdx, %rsi
	jne	.L3
.L2:
	vblendpd	$10, %ymm12, %ymm3, %ymm1
	vblendpd	$10, %ymm11, %ymm5, %ymm13
	testl	%r8d, %r8d
	vblendpd	$5, %ymm12, %ymm3, %ymm3
	vblendpd	$5, %ymm11, %ymm5, %ymm5
	vblendpd	$10, %ymm9, %ymm4, %ymm15
	vblendpd	$12, %ymm13, %ymm1, %ymm11
	vblendpd	$5, %ymm9, %ymm4, %ymm4
	vblendpd	$3, %ymm13, %ymm1, %ymm13
	vblendpd	$12, %ymm5, %ymm3, %ymm1
	vblendpd	$3, %ymm5, %ymm3, %ymm5
	vblendpd	$10, %ymm10, %ymm2, %ymm3
	vblendpd	$5, %ymm10, %ymm2, %ymm2
	vblendpd	$10, %ymm7, %ymm0, %ymm10
	vblendpd	$5, %ymm7, %ymm0, %ymm0
	vblendpd	$12, %ymm15, %ymm3, %ymm12
	vblendpd	$3, %ymm15, %ymm3, %ymm15
	vblendpd	$12, %ymm4, %ymm2, %ymm3
	vblendpd	$3, %ymm4, %ymm2, %ymm4
	vblendpd	$10, %ymm8, %ymm6, %ymm2
	vblendpd	$5, %ymm8, %ymm6, %ymm6
	vblendpd	$12, %ymm10, %ymm2, %ymm7
	vblendpd	$12, %ymm0, %ymm6, %ymm14
	vblendpd	$3, %ymm10, %ymm2, %ymm10
	vblendpd	$3, %ymm0, %ymm6, %ymm0
	je	.L4
	leal	0(,%r12,4), %eax
	vaddpd	(%r9), %ymm11, %ymm11
	cltq
	vaddpd	32(%r9), %ymm1, %ymm1
	salq	$3, %rax
	leaq	(%r9,%rax), %rdx
	vaddpd	64(%r9), %ymm13, %ymm13
	addq	%rdx, %rax
	vaddpd	96(%r9), %ymm5, %ymm5
	vaddpd	(%rdx), %ymm12, %ymm12
	vaddpd	32(%rdx), %ymm3, %ymm3
	vaddpd	64(%rdx), %ymm15, %ymm15
	vaddpd	96(%rdx), %ymm4, %ymm4
	vaddpd	(%rax), %ymm7, %ymm7
	vaddpd	32(%rax), %ymm14, %ymm14
	vaddpd	64(%rax), %ymm10, %ymm10
	vaddpd	96(%rax), %ymm0, %ymm0
.L4:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovsd	.LC0(%rip), %xmm9
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovsd	%xmm11, %xmm8, %xmm6
	vucomisd	%xmm9, %xmm6
	jbe	.L5
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm2
.L5:
	vmulpd	%ymm2, %ymm11, %ymm11
	vmovlpd	%xmm2, (%r14)
	vmulpd	%ymm2, %ymm12, %ymm12
	vmulpd	%ymm2, %ymm7, %ymm2
	vmovapd	%ymm11, (%rbx)
	vpermpd	$85, %ymm11, %ymm6
	vmovapd	%ymm12, 0(%r13)
	vxorpd	%xmm7, %xmm7, %xmm7
	vfmadd231pd	%ymm6, %ymm11, %ymm1
	vfmadd231pd	%ymm6, %ymm12, %ymm3
	vfmadd231pd	%ymm6, %ymm2, %ymm14
	vpermilpd	$3, %xmm1, %xmm6
	vmovapd	%ymm2, (%r11)
	vucomisd	%xmm9, %xmm6
	jbe	.L6
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm7, %xmm6
	vbroadcastsd	%xmm6, %ymm7
.L6:
	vmulpd	%ymm1, %ymm7, %ymm6
	vmovlpd	%xmm7, 8(%r14)
	vmovdqa	.LC2(%rip), %ymm1
	vmulpd	%ymm3, %ymm7, %ymm3
	vmulpd	%ymm14, %ymm7, %ymm14
	vpermpd	$170, %ymm11, %ymm7
	vmaskmovpd	%ymm6, %ymm1, 32(%rbx)
	vmovapd	%ymm3, 32(%r13)
	vfmadd231pd	%ymm7, %ymm11, %ymm13
	vfmadd231pd	%ymm7, %ymm12, %ymm15
	vfmadd231pd	%ymm7, %ymm2, %ymm10
	vpermpd	$170, %ymm6, %ymm7
	vmovapd	%ymm13, %ymm1
	vmovapd	%ymm14, 32(%r11)
	vfmadd231pd	%ymm7, %ymm6, %ymm1
	vextractf128	$0x1, %ymm1, %xmm8
	vfmadd231pd	%ymm7, %ymm3, %ymm15
	vfmadd231pd	%ymm7, %ymm14, %ymm10
	vxorpd	%xmm7, %xmm7, %xmm7
	vucomisd	%xmm9, %xmm8
	jbe	.L7
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm8, %xmm8, %xmm8
	vdivsd	%xmm8, %xmm7, %xmm7
	vbroadcastsd	%xmm7, %ymm7
.L7:
	vmulpd	%ymm1, %ymm7, %ymm1
	vmovlpd	%xmm7, 16(%r14)
	vmovdqa	.LC3(%rip), %ymm8
	vmulpd	%ymm15, %ymm7, %ymm13
	vmulpd	%ymm10, %ymm7, %ymm7
	vmaskmovpd	%ymm1, %ymm8, 64(%rbx)
	vpermpd	$255, %ymm11, %ymm8
	vmovapd	%ymm13, 64(%r13)
	vmovapd	%ymm7, 64(%r11)
	vfmadd231pd	%ymm8, %ymm11, %ymm5
	vpermpd	$255, %ymm6, %ymm11
	vfmadd231pd	%ymm8, %ymm2, %ymm0
	vfmadd231pd	%ymm8, %ymm12, %ymm4
	vfmadd231pd	%ymm11, %ymm6, %ymm5
	vpermpd	$255, %ymm1, %ymm6
	vfmadd231pd	%ymm11, %ymm14, %ymm0
	vfmadd231pd	%ymm11, %ymm3, %ymm4
	vfmadd132pd	%ymm6, %ymm5, %ymm1
	vextractf128	$0x1, %ymm1, %xmm2
	vfmadd132pd	%ymm6, %ymm0, %ymm7
	vfmadd132pd	%ymm6, %ymm4, %ymm13
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermilpd	$3, %xmm2, %xmm2
	vucomisd	%xmm9, %xmm2
	jbe	.L8
	vsqrtsd	%xmm2, %xmm2, %xmm3
	vmovsd	.LC1(%rip), %xmm2
	vdivsd	%xmm3, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm0
.L8:
	vmovlpd	%xmm0, 24(%r14)
	vmulpd	%ymm1, %ymm0, %ymm1
	vmovdqa	.LC4(%rip), %ymm2
	vmulpd	%ymm13, %ymm0, %ymm13
	vmulpd	%ymm7, %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm2, 96(%rbx)
	vmovapd	%ymm13, 96(%r13)
	vmovapd	%ymm0, 96(%r11)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L10:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm3
	jmp	.L2
	.cfi_endproc
.LFE4589:
	.size	kernel_dpotrf_nt_12x4_lib4_new, .-kernel_dpotrf_nt_12x4_lib4_new
	.section	.text.unlikely
.LCOLDE5:
	.text
.LHOTE5:
	.section	.text.unlikely
.LCOLDB6:
	.text
.LHOTB6:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_12x4_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_12x4_lib4_new, @function
kernel_dsyrk_dpotrf_nt_12x4_lib4_new:
.LFB4590:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	(%r10), %eax
	movl	48(%r10), %r11d
	movq	40(%r10), %rbx
	movl	%r8d, -84(%rbp)
	movq	56(%r10), %r14
	movl	%eax, -100(%rbp)
	movq	8(%r10), %rax
	sall	$2, %r11d
	movslq	%r11d, %r11
	salq	$3, %r11
	movq	%rax, -112(%rbp)
	movl	16(%r10), %eax
	leaq	(%rbx,%r11), %r13
	addq	%r13, %r11
	testl	%edi, %edi
	movl	%eax, -88(%rbp)
	movq	24(%r10), %rax
	movq	%rax, -120(%rbp)
	movl	32(%r10), %eax
	movl	%eax, -104(%rbp)
	jle	.L37
	sall	$2, %edx
	vmovapd	(%rsi), %ymm14
	movslq	%edx, %rdx
	vmovapd	(%rcx), %ymm1
	salq	$3, %rdx
	cmpl	$3, %edi
	leaq	(%rsi,%rdx), %r15
	leaq	(%r15,%rdx), %rax
	vmovapd	(%r15), %ymm13
	movq	%rax, %r10
	movq	%rax, -96(%rbp)
	vmovapd	(%rax), %ymm15
	jle	.L38
	vxorpd	%xmm0, %xmm0, %xmm0
	leal	-4(%rdi), %edx
	leaq	32(%rcx), %rax
	movq	%r15, %r8
	shrl	$2, %edx
	movl	%edx, -132(%rbp)
	movq	%rdx, -128(%rbp)
	vmovapd	%ymm0, %ymm7
	salq	$7, %rdx
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm9
	leaq	160(%rcx,%rdx), %r12
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm2
	movq	%rsi, %rdx
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm3
	.p2align 4,,10
	.p2align 3
.L28:
	vfmadd231pd	%ymm1, %ymm14, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	subq	$-128, %rax
	subq	$-128, %rdx
	subq	$-128, %r8
	subq	$-128, %r10
	vfmadd231pd	%ymm1, %ymm14, %ymm12
	vfmadd231pd	%ymm1, %ymm13, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm11
	vfmadd231pd	%ymm1, %ymm13, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm5
	vfmadd231pd	%ymm1, %ymm13, %ymm4
	vmovapd	-96(%rdx), %ymm14
	vmovapd	-96(%r8), %ymm13
	vfmadd231pd	%ymm1, %ymm15, %ymm0
	vmovapd	-128(%rax), %ymm1
	vmovapd	-96(%r10), %ymm15
	vfmadd231pd	%ymm1, %ymm14, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm12
	vfmadd231pd	%ymm1, %ymm13, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm11
	vfmadd231pd	%ymm1, %ymm13, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vfmadd132pd	%ymm1, %ymm5, %ymm14
	vmovapd	-96(%rax), %ymm0
	vmovapd	-64(%rdx), %ymm5
	vfmadd132pd	%ymm1, %ymm4, %ymm13
	vmovapd	-64(%r8), %ymm4
	vmovapd	-64(%r10), %ymm1
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm14
	vfmadd231pd	%ymm0, %ymm4, %ymm13
	vmovapd	-32(%rdx), %ymm5
	vmovapd	-32(%r8), %ymm4
	vfmadd132pd	%ymm0, %ymm15, %ymm1
	vmovapd	-32(%r10), %ymm15
	vmovapd	-64(%rax), %ymm0
	vmovapd	%ymm1, -80(%rbp)
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm15, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm15, %ymm7
	vmovapd	-80(%rbp), %ymm0
	vfmadd132pd	%ymm1, %ymm14, %ymm5
	vfmadd132pd	%ymm1, %ymm13, %ymm4
	vfmadd231pd	%ymm1, %ymm15, %ymm0
	vmovapd	-32(%rax), %ymm1
	vmovapd	(%rdx), %ymm14
	vmovapd	(%r8), %ymm13
	vmovapd	(%r10), %ymm15
	cmpq	%r12, %rax
	jne	.L28
	movq	-128(%rbp), %rax
	addq	$1, %rax
	salq	$7, %rax
	addq	%rax, -96(%rbp)
	addq	%rax, %rsi
	addq	%rax, %r15
	addq	%rax, %rcx
	movl	-132(%rbp), %eax
	leal	4(,%rax,4), %eax
.L27:
	leal	-1(%rdi), %edx
	cmpl	%eax, %edx
	jle	.L29
	vfmadd231pd	%ymm1, %ymm14, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	movq	-96(%rbp), %rdx
	addl	$2, %eax
	vfmadd231pd	%ymm1, %ymm14, %ymm12
	vfmadd231pd	%ymm1, %ymm13, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm11
	vfmadd231pd	%ymm1, %ymm13, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm5, %ymm14
	vfmadd132pd	%ymm1, %ymm4, %ymm13
	vmovapd	32(%rsi), %ymm5
	vmovapd	32(%r15), %ymm4
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vmovapd	32(%rdx), %ymm1
	vmovapd	32(%rcx), %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm14, %ymm5
	vfmadd132pd	%ymm0, %ymm13, %ymm4
	vmovapd	64(%rsi), %ymm14
	vfmadd132pd	%ymm1, %ymm15, %ymm0
	vmovapd	64(%r15), %ymm13
	vmovapd	64(%rcx), %ymm1
	vmovapd	64(%rdx), %ymm15
.L29:
	cmpl	%edi, %eax
	jl	.L53
.L26:
	movl	-84(%rbp), %edi
	testl	%edi, %edi
	jle	.L30
	movl	-100(%rbp), %eax
	movq	-112(%rbp), %r15
	vmovapd	(%r9), %ymm13
	sall	$2, %eax
	vmovapd	(%r15), %ymm1
	cltq
	salq	$3, %rax
	leaq	(%r9,%rax), %rcx
	addq	%rcx, %rax
	cmpl	$3, %edi
	vmovapd	(%rcx), %ymm14
	vmovapd	(%rax), %ymm15
	jle	.L30
	movl	%edi, %esi
	leaq	32(%r15), %rdx
	subl	$4, %esi
	shrl	$2, %esi
	salq	$7, %rsi
	leaq	160(%r15,%rsi), %rsi
	.p2align 4,,10
	.p2align 3
.L31:
	vfmadd231pd	%ymm1, %ymm13, %ymm3
	vfmadd231pd	%ymm1, %ymm14, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	subq	$-128, %rdx
	subq	$-128, %r9
	subq	$-128, %rcx
	subq	$-128, %rax
	vfmadd231pd	%ymm1, %ymm13, %ymm12
	vfmadd231pd	%ymm1, %ymm14, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm11
	vfmadd231pd	%ymm1, %ymm14, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm5
	vfmadd231pd	%ymm1, %ymm14, %ymm4
	vmovapd	-96(%r9), %ymm13
	vmovapd	-96(%rcx), %ymm14
	vfmadd231pd	%ymm1, %ymm15, %ymm0
	vmovapd	-128(%rdx), %ymm1
	vmovapd	-96(%rax), %ymm15
	vfmadd231pd	%ymm1, %ymm13, %ymm3
	vfmadd231pd	%ymm1, %ymm14, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm12
	vfmadd231pd	%ymm1, %ymm14, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm13, %ymm11
	vfmadd231pd	%ymm1, %ymm14, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm5, %ymm13
	vfmadd132pd	%ymm1, %ymm4, %ymm14
	vmovapd	-64(%r9), %ymm5
	vmovapd	-64(%rcx), %ymm4
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vmovapd	-64(%rax), %ymm1
	vmovapd	-96(%rdx), %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm13
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vmovapd	-32(%r9), %ymm5
	vmovapd	-32(%rcx), %ymm4
	vfmadd231pd	%ymm0, %ymm1, %ymm15
	vmovapd	-32(%rax), %ymm1
	vmovapd	-64(%rdx), %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm3
	vfmadd231pd	%ymm0, %ymm4, %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm4, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm13, %ymm5
	vfmadd132pd	%ymm0, %ymm14, %ymm4
	vfmadd132pd	%ymm1, %ymm15, %ymm0
	vmovapd	-32(%rdx), %ymm1
	vmovapd	(%r9), %ymm13
	vmovapd	(%rcx), %ymm14
	vmovapd	(%rax), %ymm15
	cmpq	%rdx, %rsi
	jne	.L31
.L30:
	vblendpd	$10, %ymm12, %ymm3, %ymm1
	vblendpd	$10, %ymm11, %ymm5, %ymm13
	movl	-88(%rbp), %eax
	vblendpd	$5, %ymm12, %ymm3, %ymm3
	vblendpd	$5, %ymm11, %ymm5, %ymm5
	vblendpd	$10, %ymm9, %ymm4, %ymm15
	vblendpd	$12, %ymm13, %ymm1, %ymm11
	vblendpd	$5, %ymm9, %ymm4, %ymm4
	vblendpd	$3, %ymm13, %ymm1, %ymm13
	testl	%eax, %eax
	vblendpd	$12, %ymm5, %ymm3, %ymm1
	vblendpd	$3, %ymm5, %ymm3, %ymm5
	vblendpd	$10, %ymm10, %ymm2, %ymm3
	vblendpd	$5, %ymm10, %ymm2, %ymm2
	vblendpd	$10, %ymm7, %ymm0, %ymm10
	vblendpd	$5, %ymm7, %ymm0, %ymm0
	vblendpd	$12, %ymm15, %ymm3, %ymm12
	vblendpd	$3, %ymm15, %ymm3, %ymm15
	vblendpd	$12, %ymm4, %ymm2, %ymm3
	vblendpd	$3, %ymm4, %ymm2, %ymm4
	vblendpd	$10, %ymm8, %ymm6, %ymm2
	vblendpd	$5, %ymm8, %ymm6, %ymm6
	vblendpd	$12, %ymm10, %ymm2, %ymm7
	vblendpd	$12, %ymm0, %ymm6, %ymm14
	vblendpd	$3, %ymm10, %ymm2, %ymm10
	vblendpd	$3, %ymm0, %ymm6, %ymm0
	je	.L32
	movl	-104(%rbp), %eax
	movq	-120(%rbp), %rdi
	sall	$2, %eax
	vaddpd	(%rdi), %ymm11, %ymm11
	cltq
	salq	$3, %rax
	vaddpd	32(%rdi), %ymm1, %ymm1
	leaq	(%rdi,%rax), %rdx
	vaddpd	64(%rdi), %ymm13, %ymm13
	addq	%rdx, %rax
	vaddpd	96(%rdi), %ymm5, %ymm5
	vaddpd	(%rdx), %ymm12, %ymm12
	vaddpd	32(%rdx), %ymm3, %ymm3
	vaddpd	64(%rdx), %ymm15, %ymm15
	vaddpd	96(%rdx), %ymm4, %ymm4
	vaddpd	(%rax), %ymm7, %ymm7
	vaddpd	32(%rax), %ymm14, %ymm14
	vaddpd	64(%rax), %ymm10, %ymm10
	vaddpd	96(%rax), %ymm0, %ymm0
.L32:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovsd	.LC0(%rip), %xmm9
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovsd	%xmm11, %xmm8, %xmm6
	vucomisd	%xmm9, %xmm6
	jbe	.L33
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm2
.L33:
	vmulpd	%ymm2, %ymm11, %ymm11
	vmovlpd	%xmm2, (%r14)
	vmulpd	%ymm2, %ymm12, %ymm12
	vmulpd	%ymm2, %ymm7, %ymm2
	vmovapd	%ymm11, (%rbx)
	vpermpd	$85, %ymm11, %ymm6
	vmovapd	%ymm12, 0(%r13)
	vxorpd	%xmm7, %xmm7, %xmm7
	vfmadd231pd	%ymm6, %ymm11, %ymm1
	vfmadd231pd	%ymm6, %ymm12, %ymm3
	vfmadd231pd	%ymm6, %ymm2, %ymm14
	vpermilpd	$3, %xmm1, %xmm6
	vmovapd	%ymm2, (%r11)
	vucomisd	%xmm9, %xmm6
	jbe	.L34
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm7, %xmm6
	vbroadcastsd	%xmm6, %ymm7
.L34:
	vmulpd	%ymm1, %ymm7, %ymm6
	vmovlpd	%xmm7, 8(%r14)
	vmovdqa	.LC2(%rip), %ymm1
	vmulpd	%ymm3, %ymm7, %ymm3
	vmulpd	%ymm14, %ymm7, %ymm14
	vpermpd	$170, %ymm11, %ymm7
	vmaskmovpd	%ymm6, %ymm1, 32(%rbx)
	vmovapd	%ymm3, 32(%r13)
	vfmadd231pd	%ymm7, %ymm11, %ymm13
	vfmadd231pd	%ymm7, %ymm12, %ymm15
	vfmadd231pd	%ymm7, %ymm2, %ymm10
	vpermpd	$170, %ymm6, %ymm7
	vmovapd	%ymm13, %ymm1
	vmovapd	%ymm14, 32(%r11)
	vfmadd231pd	%ymm7, %ymm6, %ymm1
	vextractf128	$0x1, %ymm1, %xmm8
	vfmadd231pd	%ymm7, %ymm3, %ymm15
	vfmadd231pd	%ymm7, %ymm14, %ymm10
	vxorpd	%xmm7, %xmm7, %xmm7
	vucomisd	%xmm9, %xmm8
	jbe	.L35
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm8, %xmm8, %xmm8
	vdivsd	%xmm8, %xmm7, %xmm7
	vbroadcastsd	%xmm7, %ymm7
.L35:
	vmulpd	%ymm1, %ymm7, %ymm1
	vmovlpd	%xmm7, 16(%r14)
	vmovdqa	.LC3(%rip), %ymm8
	vmulpd	%ymm15, %ymm7, %ymm13
	vmulpd	%ymm10, %ymm7, %ymm7
	vmaskmovpd	%ymm1, %ymm8, 64(%rbx)
	vpermpd	$255, %ymm11, %ymm8
	vmovapd	%ymm13, 64(%r13)
	vmovapd	%ymm7, 64(%r11)
	vfmadd231pd	%ymm8, %ymm11, %ymm5
	vpermpd	$255, %ymm6, %ymm11
	vfmadd231pd	%ymm8, %ymm2, %ymm0
	vfmadd231pd	%ymm8, %ymm12, %ymm4
	vfmadd231pd	%ymm11, %ymm6, %ymm5
	vpermpd	$255, %ymm1, %ymm6
	vfmadd231pd	%ymm11, %ymm14, %ymm0
	vfmadd231pd	%ymm11, %ymm3, %ymm4
	vfmadd132pd	%ymm6, %ymm5, %ymm1
	vextractf128	$0x1, %ymm1, %xmm2
	vfmadd132pd	%ymm6, %ymm0, %ymm7
	vfmadd132pd	%ymm6, %ymm4, %ymm13
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermilpd	$3, %xmm2, %xmm2
	vucomisd	%xmm9, %xmm2
	jbe	.L36
	vsqrtsd	%xmm2, %xmm2, %xmm3
	vmovsd	.LC1(%rip), %xmm2
	vdivsd	%xmm3, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm0
.L36:
	vmovlpd	%xmm0, 24(%r14)
	vmulpd	%ymm1, %ymm0, %ymm1
	vmovdqa	.LC4(%rip), %ymm2
	vmulpd	%ymm13, %ymm0, %ymm13
	vmulpd	%ymm7, %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm2, 96(%rbx)
	vmovapd	%ymm13, 96(%r13)
	vmovapd	%ymm0, 96(%r11)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L53:
	.cfi_restore_state
	vfmadd231pd	%ymm1, %ymm14, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm2
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm12
	vfmadd231pd	%ymm1, %ymm13, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm11
	vfmadd231pd	%ymm1, %ymm13, %ymm9
	vfmadd231pd	%ymm1, %ymm15, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm14, %ymm5
	vfmadd231pd	%ymm1, %ymm13, %ymm4
	vfmadd231pd	%ymm1, %ymm15, %ymm0
	jmp	.L26
	.p2align 4,,10
	.p2align 3
.L37:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm3
	jmp	.L26
.L38:
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm8
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm3
	jmp	.L27
	.cfi_endproc
.LFE4590:
	.size	kernel_dsyrk_dpotrf_nt_12x4_lib4_new, .-kernel_dsyrk_dpotrf_nt_12x4_lib4_new
	.section	.text.unlikely
.LCOLDE6:
	.text
.LHOTE6:
	.section	.text.unlikely
.LCOLDB8:
	.text
.LHOTB8:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_12x4_vs_lib4_new:
.LFB4591:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$8, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	8(%r10), %ebx
	movl	72(%r10), %r12d
	movq	80(%r10), %r14
	movl	%esi, -148(%rbp)
	movq	(%r10), %rax
	movq	16(%r10), %rsi
	movl	%ebx, -152(%rbp)
	movl	24(%r10), %ebx
	movq	32(%r10), %r11
	movq	%r14, -80(%rbp)
	movl	%ebx, -160(%rbp)
	movl	40(%r10), %ebx
	movl	%ebx, -156(%rbp)
	movq	48(%r10), %rbx
	movq	%rbx, -168(%rbp)
	movl	56(%r10), %ebx
	movl	%ebx, -172(%rbp)
	movq	64(%r10), %rbx
	leal	0(,%r12,4), %r10d
	movslq	%r10d, %r10
	salq	$3, %r10
	cmpl	$11, %edi
	leaq	(%rbx,%r10), %r15
	leaq	(%r15,%r10), %r14
	movq	%r14, -72(%rbp)
	jle	.L55
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%edx, %edx
	vmovdqu	%ymm0, mask_bkp.27203(%rip)
	jle	.L74
.L102:
	sall	$2, %r9d
	vmovapd	(%r8), %ymm5
	movslq	%r9d, %r9
	vmovapd	(%rax), %ymm1
	salq	$3, %r9
	cmpl	$1, %ecx
	leaq	(%r8,%r9), %r13
	leaq	0(%r13,%r9), %rdi
	vmovapd	0(%r13), %ymm3
	movq	%rdi, -144(%rbp)
	vmovapd	(%rdi), %ymm15
	je	.L99
	vxorpd	%xmm6, %xmm6, %xmm6
	xorl	%r14d, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vmovapd	%ymm6, %ymm2
	vmovapd	%ymm6, %ymm7
	vmovapd	%ymm6, %ymm10
	vmovapd	%ymm6, %ymm13
	vmovapd	%ymm6, %ymm4
	vmovapd	%ymm6, %ymm8
	vmovapd	%ymm6, %ymm11
	vmovapd	%ymm6, %ymm14
.L58:
	leal	-3(%rdx), %ecx
	movq	%rax, %r10
	cmpl	%ecx, %r14d
	jge	.L63
	leal	-4(%rdx), %ecx
	movq	-144(%rbp), %r9
	vmovapd	-112(%rbp), %ymm0
	movq	%r13, %rdi
	subl	%r14d, %ecx
	shrl	$2, %ecx
	movl	%ecx, %r12d
	movl	%ecx, -176(%rbp)
	movq	%r8, %rcx
	addq	$1, %r12
	salq	$7, %r12
	leaq	(%rax,%r12), %r10
	.p2align 4,,10
	.p2align 3
.L64:
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vfmadd231pd	%ymm1, %ymm15, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	subq	$-128, %rax
	subq	$-128, %rcx
	subq	$-128, %rdi
	subq	$-128, %r9
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vfmadd132pd	%ymm1, %ymm4, %ymm5
	vmovapd	-96(%r9), %ymm0
	vmovapd	-96(%rcx), %ymm4
	vfmadd132pd	%ymm1, %ymm2, %ymm3
	vmovapd	-96(%rdi), %ymm2
	vmovapd	-96(%rax), %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vfmadd231pd	%ymm1, %ymm2, %ymm13
	vfmadd231pd	%ymm1, %ymm0, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm2, %ymm10
	vfmadd231pd	%ymm1, %ymm0, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm8
	vfmadd231pd	%ymm1, %ymm2, %ymm7
	vfmadd231pd	%ymm1, %ymm0, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm15, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm5
	vmovapd	-64(%rax), %ymm15
	vmovapd	-64(%rcx), %ymm4
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vmovapd	-64(%rdi), %ymm2
	vmovapd	-64(%r9), %ymm1
	vfmadd231pd	%ymm15, %ymm4, %ymm14
	vfmadd231pd	%ymm15, %ymm2, %ymm13
	vfmadd231pd	%ymm15, %ymm1, %ymm12
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm11
	vfmadd231pd	%ymm15, %ymm2, %ymm10
	vfmadd231pd	%ymm15, %ymm1, %ymm9
	vperm2f128	$1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm8
	vfmadd231pd	%ymm15, %ymm2, %ymm7
	vfmadd231pd	%ymm15, %ymm1, %ymm6
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm5
	vfmadd231pd	%ymm15, %ymm2, %ymm3
	vmovapd	-32(%rcx), %ymm4
	vmovapd	-32(%rdi), %ymm2
	vfmadd132pd	%ymm15, %ymm0, %ymm1
	vmovapd	-32(%r9), %ymm15
	vmovapd	-32(%rax), %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vfmadd231pd	%ymm0, %ymm15, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm10
	vfmadd231pd	%ymm0, %ymm15, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vfmadd231pd	%ymm0, %ymm2, %ymm7
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm5, %ymm4
	vfmadd132pd	%ymm0, %ymm3, %ymm2
	vfmadd132pd	%ymm15, %ymm1, %ymm0
	vmovapd	(%rax), %ymm1
	vmovapd	(%rcx), %ymm5
	vmovapd	(%rdi), %ymm3
	vmovapd	(%r9), %ymm15
	cmpq	%r10, %rax
	jne	.L64
	movl	-176(%rbp), %eax
	addq	%r12, -144(%rbp)
	vmovapd	%ymm0, -112(%rbp)
	addq	%r12, %r8
	addq	%r12, %r13
	leal	4(%r14,%rax,4), %r14d
.L63:
	leal	-1(%rdx), %eax
	cmpl	%r14d, %eax
	jle	.L65
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vfmadd231pd	%ymm1, %ymm15, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	movq	-144(%rbp), %rax
	vmovapd	32(%r10), %ymm0
	addl	$2, %r14d
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd213pd	-112(%rbp), %ymm1, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	32(%r8), %ymm5
	vmovapd	32(%r13), %ymm3
	vmovapd	32(%rax), %ymm1
	vfmadd231pd	%ymm0, %ymm5, %ymm14
	vfmadd231pd	%ymm0, %ymm3, %ymm13
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm3, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm8
	vfmadd231pd	%ymm0, %ymm3, %ymm7
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm15, %ymm1
	vfmadd231pd	%ymm0, %ymm5, %ymm4
	vfmadd231pd	%ymm0, %ymm3, %ymm2
	vmovapd	%ymm1, -112(%rbp)
	vmovapd	64(%r8), %ymm5
	vmovapd	64(%r13), %ymm3
	vmovapd	64(%r10), %ymm1
	vmovapd	64(%rax), %ymm15
.L65:
	cmpl	%edx, %r14d
	jl	.L100
.L57:
	movl	-152(%rbp), %edi
	testl	%edi, %edi
	jle	.L66
	movl	-160(%rbp), %ecx
	vmovapd	(%rsi), %ymm5
	vmovapd	(%r11), %ymm0
	sall	$2, %ecx
	movslq	%ecx, %rcx
	salq	$3, %rcx
	leaq	(%rsi,%rcx), %rdx
	addq	%rdx, %rcx
	cmpl	$3, %edi
	vmovapd	(%rdx), %ymm3
	vmovapd	(%rcx), %ymm15
	jle	.L66
	subl	$4, %edi
	leaq	32(%rcx), %rax
	vmovapd	-112(%rbp), %ymm1
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	160(%rcx,%rdi), %rcx
	.p2align 4,,10
	.p2align 3
.L67:
	vfmadd231pd	%ymm0, %ymm5, %ymm14
	vfmadd231pd	%ymm0, %ymm3, %ymm13
	vfmadd231pd	%ymm0, %ymm15, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	subq	$-128, %rax
	subq	$-128, %rsi
	subq	$-128, %rdx
	subq	$-128, %r11
	vfmadd231pd	%ymm0, %ymm5, %ymm11
	vfmadd231pd	%ymm0, %ymm3, %ymm10
	vfmadd231pd	%ymm0, %ymm15, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm8
	vfmadd231pd	%ymm0, %ymm3, %ymm7
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm1, %ymm15
	vfmadd132pd	%ymm0, %ymm4, %ymm5
	vmovapd	-96(%r11), %ymm1
	vmovapd	-96(%rsi), %ymm4
	vfmadd132pd	%ymm0, %ymm2, %ymm3
	vmovapd	-96(%rdx), %ymm2
	vmovapd	-128(%rax), %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vfmadd231pd	%ymm1, %ymm2, %ymm13
	vfmadd231pd	%ymm1, %ymm0, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm2, %ymm10
	vfmadd231pd	%ymm1, %ymm0, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm8
	vfmadd231pd	%ymm1, %ymm2, %ymm7
	vfmadd231pd	%ymm1, %ymm0, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm15, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm5
	vmovapd	-64(%r11), %ymm15
	vmovapd	-64(%rsi), %ymm4
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vmovapd	-64(%rdx), %ymm2
	vmovapd	-96(%rax), %ymm1
	vfmadd231pd	%ymm15, %ymm4, %ymm14
	vfmadd231pd	%ymm15, %ymm2, %ymm13
	vfmadd231pd	%ymm15, %ymm1, %ymm12
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm11
	vfmadd231pd	%ymm15, %ymm2, %ymm10
	vfmadd231pd	%ymm15, %ymm1, %ymm9
	vperm2f128	$1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm8
	vfmadd231pd	%ymm15, %ymm2, %ymm7
	vfmadd231pd	%ymm15, %ymm1, %ymm6
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm5
	vfmadd231pd	%ymm15, %ymm2, %ymm3
	vmovapd	-32(%rsi), %ymm4
	vmovapd	-32(%rdx), %ymm2
	vfmadd132pd	%ymm15, %ymm0, %ymm1
	vmovapd	-64(%rax), %ymm15
	vmovapd	-32(%r11), %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vfmadd231pd	%ymm0, %ymm15, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm10
	vfmadd231pd	%ymm0, %ymm15, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vfmadd231pd	%ymm0, %ymm2, %ymm7
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm15, %ymm1
	vmovapd	-32(%rax), %ymm15
	vfmadd132pd	%ymm0, %ymm5, %ymm4
	vfmadd132pd	%ymm0, %ymm3, %ymm2
	vmovapd	(%rsi), %ymm5
	vmovapd	(%rdx), %ymm3
	vmovapd	(%r11), %ymm0
	cmpq	%rax, %rcx
	jne	.L67
	vmovapd	%ymm1, -112(%rbp)
.L66:
	vblendpd	$10, %ymm8, %ymm4, %ymm1
	vblendpd	$10, %ymm11, %ymm14, %ymm0
	vmovapd	-112(%rbp), %ymm5
	vblendpd	$5, %ymm11, %ymm14, %ymm14
	movl	-156(%rbp), %eax
	vblendpd	$10, %ymm7, %ymm2, %ymm11
	vblendpd	$12, %ymm1, %ymm0, %ymm3
	vblendpd	$3, %ymm1, %ymm0, %ymm0
	vblendpd	$10, %ymm10, %ymm13, %ymm1
	vblendpd	$5, %ymm8, %ymm4, %ymm8
	vblendpd	$5, %ymm10, %ymm13, %ymm13
	vblendpd	$5, %ymm7, %ymm2, %ymm7
	testl	%eax, %eax
	vblendpd	$10, %ymm6, %ymm5, %ymm10
	vblendpd	$12, %ymm11, %ymm1, %ymm4
	vblendpd	$5, %ymm6, %ymm5, %ymm6
	vblendpd	$3, %ymm11, %ymm1, %ymm11
	vblendpd	$10, %ymm9, %ymm12, %ymm1
	vblendpd	$5, %ymm9, %ymm12, %ymm12
	vblendpd	$12, %ymm8, %ymm14, %ymm15
	vblendpd	$3, %ymm8, %ymm14, %ymm8
	vblendpd	$12, %ymm10, %ymm1, %ymm5
	vblendpd	$12, %ymm7, %ymm13, %ymm14
	vblendpd	$12, %ymm6, %ymm12, %ymm2
	vblendpd	$3, %ymm7, %ymm13, %ymm7
	vblendpd	$3, %ymm10, %ymm1, %ymm10
	vblendpd	$3, %ymm6, %ymm12, %ymm6
	je	.L68
	movl	-172(%rbp), %eax
	movq	-168(%rbp), %rsi
	sall	$2, %eax
	vaddpd	(%rsi), %ymm3, %ymm3
	cltq
	salq	$3, %rax
	vaddpd	32(%rsi), %ymm15, %ymm15
	leaq	(%rsi,%rax), %rdx
	vaddpd	64(%rsi), %ymm0, %ymm0
	addq	%rdx, %rax
	vaddpd	96(%rsi), %ymm8, %ymm8
	vaddpd	(%rdx), %ymm4, %ymm4
	vaddpd	32(%rdx), %ymm14, %ymm14
	vaddpd	64(%rdx), %ymm11, %ymm11
	vaddpd	96(%rdx), %ymm7, %ymm7
	vaddpd	(%rax), %ymm5, %ymm5
	vaddpd	32(%rax), %ymm2, %ymm2
	vaddpd	64(%rax), %ymm10, %ymm10
	vaddpd	96(%rax), %ymm6, %ymm6
.L68:
	vxorpd	%xmm12, %xmm12, %xmm12
	vmovsd	.LC0(%rip), %xmm13
	vxorpd	%xmm9, %xmm9, %xmm9
	vmovsd	%xmm3, %xmm12, %xmm1
	vucomisd	%xmm13, %xmm1
	jbe	.L69
	vmovsd	.LC1(%rip), %xmm9
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm9, %xmm1
	vbroadcastsd	%xmm1, %ymm9
.L69:
	movq	-80(%rbp), %rax
	vmulpd	%ymm9, %ymm3, %ymm3
	vmulpd	%ymm9, %ymm4, %ymm4
	vmovlpd	%xmm9, (%rax)
	movq	-72(%rbp), %rax
	vmulpd	%ymm9, %ymm5, %ymm9
	vmovapd	%ymm15, %ymm5
	vmovdqu	mask_bkp.27203(%rip), %ymm1
	vmovapd	%ymm3, (%rbx)
	vmovapd	%ymm4, (%r15)
	vmaskmovpd	%ymm9, %ymm1, (%rax)
	vpermpd	$85, %ymm3, %ymm1
	vfmadd231pd	%ymm1, %ymm3, %ymm5
	vpermilpd	$3, %xmm5, %xmm15
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vfmadd231pd	%ymm1, %ymm9, %ymm2
	vxorpd	%xmm1, %xmm1, %xmm1
	vucomisd	%xmm13, %xmm15
	jbe	.L70
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm15, %xmm15, %xmm15
	vdivsd	%xmm15, %xmm1, %xmm15
	vbroadcastsd	%xmm15, %ymm1
.L70:
	movq	-80(%rbp), %rax
	vmulpd	%ymm5, %ymm1, %ymm5
	vmulpd	%ymm2, %ymm1, %ymm15
	vmovlpd	%xmm1, 8(%rax)
	vmulpd	%ymm14, %ymm1, %ymm14
	vmovdqa	.LC2(%rip), %ymm1
	vmovdqu	mask_bkp.27203(%rip), %ymm12
	vmaskmovpd	%ymm5, %ymm1, 32(%rbx)
	movq	-72(%rbp), %rax
	vpermpd	$170, %ymm3, %ymm1
	vmovapd	%ymm14, 32(%r15)
	vfmadd231pd	%ymm1, %ymm3, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm9, %ymm10
	vpermpd	$170, %ymm5, %ymm1
	vmaskmovpd	%ymm15, %ymm12, 32(%rax)
	vfmadd231pd	%ymm1, %ymm5, %ymm0
	vextractf128	$0x1, %ymm0, %xmm2
	vfmadd231pd	%ymm1, %ymm14, %ymm11
	vfmadd231pd	%ymm1, %ymm15, %ymm10
	vxorpd	%xmm1, %xmm1, %xmm1
	vucomisd	%xmm13, %xmm2
	jbe	.L71
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm2, %xmm2, %xmm2
	vdivsd	%xmm2, %xmm1, %xmm1
	vbroadcastsd	%xmm1, %ymm1
.L71:
	movq	-80(%rbp), %rax
	vmulpd	%ymm0, %ymm1, %ymm0
	vmulpd	%ymm11, %ymm1, %ymm11
	vmovlpd	%xmm1, 16(%rax)
	vmulpd	%ymm10, %ymm1, %ymm1
	vmovdqa	.LC3(%rip), %ymm10
	vmovdqu	mask_bkp.27203(%rip), %ymm2
	vmaskmovpd	%ymm0, %ymm10, 64(%rbx)
	movq	-72(%rbp), %rax
	vmovapd	%ymm11, 64(%r15)
	vmaskmovpd	%ymm1, %ymm2, 64(%rax)
	cmpl	$3, -148(%rbp)
	jle	.L97
	vpermpd	$255, %ymm3, %ymm2
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm4, %ymm7
	vfmadd231pd	%ymm2, %ymm9, %ymm6
	vpermpd	$255, %ymm5, %ymm2
	vpermpd	$255, %ymm0, %ymm3
	vfmadd231pd	%ymm2, %ymm5, %ymm8
	vmovapd	%ymm0, %ymm5
	vfmadd231pd	%ymm2, %ymm14, %ymm7
	vfmadd132pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm3, %ymm11, %ymm7
	vfmadd132pd	%ymm3, %ymm15, %ymm1
	vfmadd132pd	%ymm3, %ymm8, %ymm5
	vextractf128	$0x1, %ymm5, %xmm2
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermilpd	$3, %xmm2, %xmm2
	vucomisd	%xmm13, %xmm2
	ja	.L101
.L73:
	movq	-80(%rbp), %rax
	vmulpd	%ymm5, %ymm0, %ymm5
	vmovdqa	.LC4(%rip), %ymm3
	vmulpd	%ymm7, %ymm0, %ymm7
	vmovlpd	%xmm0, 24(%rax)
	vmulpd	%ymm1, %ymm0, %ymm0
	vmovdqu	mask_bkp.27203(%rip), %ymm2
	vmaskmovpd	%ymm5, %ymm3, 96(%rbx)
	movq	-72(%rbp), %rax
	vmovapd	%ymm7, 96(%r15)
	vmaskmovpd	%ymm0, %ymm2, 96(%rax)
.L97:
	vzeroupper
	addq	$8, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L55:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27204(%rip), %ymm1
	testl	%edx, %edx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC7(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.27203(%rip)
	jg	.L102
.L74:
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vmovapd	%ymm6, %ymm2
	vmovapd	%ymm6, %ymm7
	vmovapd	%ymm6, %ymm10
	vmovapd	%ymm6, %ymm13
	vmovapd	%ymm6, %ymm4
	vmovapd	%ymm6, %ymm8
	vmovapd	%ymm6, %ymm11
	vmovapd	%ymm6, %ymm14
	jmp	.L57
	.p2align 4,,10
	.p2align 3
.L100:
	vfmadd231pd	%ymm1, %ymm15, %ymm12
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm15, %ymm9
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd213pd	-112(%rbp), %ymm1, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	%ymm15, -112(%rbp)
	jmp	.L57
	.p2align 4,,10
	.p2align 3
.L101:
	vsqrtsd	%xmm2, %xmm2, %xmm0
	vmovsd	.LC1(%rip), %xmm2
	vdivsd	%xmm0, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm0
	jmp	.L73
	.p2align 4,,10
	.p2align 3
.L99:
	cmpl	$3, %edx
	jle	.L59
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovapd	32(%rax), %ymm2
	movq	-144(%rbp), %r14
	vmovapd	128(%r13), %ymm3
	leaq	128(%r8), %r10
	leaq	128(%r13), %rdi
	leaq	128(%rax), %r9
	vblendpd	$1, %ymm5, %ymm6, %ymm5
	vblendpd	$3, 32(%r8), %ymm6, %ymm11
	movq	%r14, %rcx
	vblendpd	$7, 64(%r8), %ymm6, %ymm4
	subq	$-128, %rcx
	cmpl	$7, %edx
	vfmadd132pd	%ymm1, %ymm6, %ymm5
	vmovapd	%ymm5, %ymm14
	vblendpd	$7, 64(%rax), %ymm6, %ymm1
	vmovapd	96(%r8), %ymm5
	vmovapd	%ymm4, %ymm8
	vfmadd231pd	%ymm2, %ymm11, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm2, %ymm6, %ymm11
	vmovapd	96(%rax), %ymm2
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm2, %ymm5, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm1, %ymm6, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm2, %ymm5, %ymm11
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm4, %ymm6, %ymm1
	vfmadd231pd	%ymm2, %ymm5, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm5
	vmovapd	128(%rax), %ymm1
	vmovapd	%ymm5, %ymm4
	vmovapd	128(%r8), %ymm5
	jle	.L60
	vblendpd	$1, %ymm3, %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	movq	%r14, %rcx
	addq	$256, %rcx
	cmpl	$11, %edx
	vmovapd	256(%r14), %ymm15
	vmovapd	%ymm3, %ymm13
	vmovapd	%ymm3, %ymm10
	leaq	256(%r8), %r10
	vmovapd	%ymm3, %ymm7
	leaq	256(%r13), %rdi
	leaq	256(%rax), %r9
	vfmadd132pd	%ymm1, %ymm6, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd132pd	%ymm1, %ymm6, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd132pd	%ymm1, %ymm6, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	160(%r8), %ymm6
	vfmadd132pd	%ymm1, %ymm4, %ymm5
	vxorpd	%xmm4, %xmm4, %xmm4
	vblendpd	$3, 160(%r13), %ymm4, %ymm2
	vfmadd132pd	%ymm1, %ymm4, %ymm3
	vmovapd	160(%rax), %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm14
	vfmadd231pd	%ymm1, %ymm2, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm11
	vfmadd231pd	%ymm1, %ymm2, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm8
	vfmadd231pd	%ymm1, %ymm2, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm5, %ymm6
	vmovapd	%ymm6, %ymm4
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovapd	192(%r8), %ymm5
	vfmadd132pd	%ymm1, %ymm3, %ymm2
	vmovapd	192(%rax), %ymm1
	vblendpd	$7, 192(%r13), %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	224(%r8), %ymm5
	vmovapd	224(%r13), %ymm3
	vmovapd	224(%rax), %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	256(%r8), %ymm5
	vmovapd	256(%r13), %ymm3
	vmovapd	256(%rax), %ymm1
	jg	.L103
	cmpl	$8, %edx
	je	.L76
	vxorpd	%xmm6, %xmm6, %xmm6
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	-144(%rbp), %r14
	cmpl	$9, %edx
	vblendpd	$1, %ymm15, %ymm6, %ymm15
	vmovapd	%ymm15, %ymm12
	vmovapd	%ymm15, %ymm9
	vfmadd132pd	%ymm1, %ymm6, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd132pd	%ymm1, %ymm6, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm15, %ymm6
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd132pd	%ymm1, %ymm0, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	%ymm15, -112(%rbp)
	vmovapd	288(%r8), %ymm5
	vmovapd	288(%r13), %ymm3
	vmovapd	288(%rax), %ymm1
	vmovapd	288(%r14), %ymm15
	je	.L77
	vblendpd	$3, %ymm15, %ymm0, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	cmpl	$11, %edx
	vfmadd231pd	%ymm1, %ymm15, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm15, %ymm0
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	320(%r14), %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vfmadd213pd	-112(%rbp), %ymm1, %ymm0
	vmovapd	320(%r8), %ymm5
	vmovapd	320(%r13), %ymm3
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	320(%rax), %ymm1
	jne	.L78
	vxorpd	%xmm0, %xmm0, %xmm0
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	movq	%rcx, -144(%rbp)
	vblendpd	$7, %ymm15, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm0, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm0, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm0, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm0, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd213pd	-112(%rbp), %ymm1, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	352(%r8), %ymm5
	movq	%r10, %r8
	vmovapd	352(%r13), %ymm3
	movq	%rdi, %r13
	vmovapd	%ymm15, -112(%rbp)
	vmovapd	352(%rax), %ymm1
	movq	%r9, %rax
	vmovapd	352(%r14), %ymm15
	movl	$11, %r14d
	jmp	.L58
	.p2align 4,,10
	.p2align 3
.L59:
	vxorpd	%xmm0, %xmm0, %xmm0
	cmpl	$1, %edx
	vblendpd	$1, %ymm5, %ymm0, %ymm14
	vmovapd	32(%r8), %ymm5
	vfmadd132pd	%ymm1, %ymm0, %ymm14
	vmovapd	32(%rax), %ymm1
	je	.L82
	vshufpd	$5, %ymm1, %ymm1, %ymm11
	vblendpd	$3, %ymm5, %ymm0, %ymm5
	cmpl	$3, %edx
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vmovapd	64(%rax), %ymm1
	vfmadd132pd	%ymm5, %ymm0, %ymm11
	vmovapd	64(%r8), %ymm5
	jne	.L83
	vblendpd	$7, %ymm5, %ymm0, %ymm5
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm12
	movl	$3, %r14d
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm5, %ymm8
	vmovapd	%ymm0, %ymm2
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm13
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm5
	vmovapd	96(%rax), %ymm1
	vmovapd	%ymm5, %ymm4
	vmovapd	96(%r8), %ymm5
	jmp	.L58
.L60:
	cmpl	$4, %edx
	je	.L79
	vxorpd	%xmm7, %xmm7, %xmm7
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	cmpl	$5, %edx
	vblendpd	$1, %ymm3, %ymm7, %ymm3
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm3, %ymm13
	vmovapd	%ymm3, %ymm10
	vfmadd132pd	%ymm1, %ymm7, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd132pd	%ymm1, %ymm7, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd132pd	%ymm1, %ymm6, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vmovapd	%ymm3, %ymm2
	vmovapd	160(%r8), %ymm5
	vmovapd	160(%r13), %ymm3
	vmovapd	160(%rax), %ymm1
	je	.L80
	vblendpd	$3, %ymm3, %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	cmpl	$7, %edx
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	192(%r8), %ymm5
	vmovapd	192(%r13), %ymm3
	vmovapd	192(%rax), %ymm1
	jne	.L81
	vblendpd	$7, %ymm3, %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	movq	%rcx, -144(%rbp)
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	movl	$7, %r14d
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	224(%r8), %ymm5
	vmovapd	224(%r13), %ymm3
	movq	%r10, %r8
	movq	%rdi, %r13
	vmovapd	224(%rax), %ymm1
	movq	%r9, %rax
	jmp	.L58
.L83:
	vxorpd	%xmm6, %xmm6, %xmm6
	movl	$2, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vmovapd	%ymm6, %ymm2
	vmovapd	%ymm6, %ymm7
	vmovapd	%ymm6, %ymm10
	vmovapd	%ymm6, %ymm13
	vmovapd	%ymm6, %ymm4
	vmovapd	%ymm6, %ymm8
	jmp	.L58
.L79:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$4, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vmovapd	%ymm6, %ymm2
	vmovapd	%ymm6, %ymm7
	vmovapd	%ymm6, %ymm10
	vmovapd	%ymm6, %ymm13
	jmp	.L58
.L103:
	vblendpd	$1, %ymm15, %ymm6, %ymm15
	vxorpd	%xmm0, %xmm0, %xmm0
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	movq	%r14, %rcx
	addq	$384, %r8
	vmovapd	%ymm15, %ymm6
	vmovapd	%ymm15, %ymm12
	addq	$384, %rcx
	vmovapd	%ymm15, %ymm9
	addq	$384, %r13
	addq	$384, %rax
	vfmadd132pd	%ymm1, %ymm0, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd132pd	%ymm1, %ymm0, %ymm12
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd132pd	%ymm1, %ymm0, %ymm9
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm9, %ymm0
	vxorpd	%xmm9, %xmm9, %xmm9
	vfmadd132pd	%ymm1, %ymm4, %ymm5
	vfmadd132pd	%ymm1, %ymm2, %ymm3
	vmovapd	-96(%r8), %ymm4
	vmovapd	-96(%r13), %ymm2
	vfmadd132pd	%ymm1, %ymm9, %ymm15
	vblendpd	$3, 288(%r14), %ymm9, %ymm9
	vmovapd	-96(%rax), %ymm1
	vfmadd231pd	%ymm1, %ymm9, %ymm6
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vfmadd231pd	%ymm1, %ymm2, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm0, %ymm6
	vxorpd	%xmm0, %xmm0, %xmm0
	vfmadd231pd	%ymm1, %ymm9, %ymm12
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm2, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vblendpd	$7, 320(%r14), %ymm0, %ymm0
	vmovapd	%ymm12, -144(%rbp)
	vmovapd	-112(%rbp), %ymm12
	vfmadd231pd	%ymm1, %ymm4, %ymm8
	vfmadd231pd	%ymm1, %ymm2, %ymm7
	vfmadd231pd	%ymm1, %ymm9, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm5
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vmovapd	-64(%r8), %ymm4
	vmovapd	-64(%r13), %ymm2
	vfmadd132pd	%ymm9, %ymm15, %ymm1
	vmovapd	-64(%rax), %ymm15
	vmovapd	-144(%rbp), %ymm9
	movq	%rcx, -144(%rbp)
	vfmadd231pd	%ymm15, %ymm4, %ymm14
	vfmadd231pd	%ymm15, %ymm2, %ymm13
	vfmadd231pd	%ymm15, %ymm0, %ymm12
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm11
	vfmadd231pd	%ymm15, %ymm2, %ymm10
	vfmadd231pd	%ymm15, %ymm0, %ymm9
	vperm2f128	$1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm15, %ymm4, %ymm8
	vfmadd231pd	%ymm15, %ymm2, %ymm7
	vfmadd231pd	%ymm15, %ymm0, %ymm6
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd132pd	%ymm15, %ymm5, %ymm4
	vfmadd132pd	%ymm15, %ymm3, %ymm2
	vmovapd	-32(%r8), %ymm5
	vmovapd	-32(%r13), %ymm3
	vfmadd132pd	%ymm15, %ymm1, %ymm0
	vmovapd	-32(%rax), %ymm1
	vmovapd	352(%r14), %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm14
	vfmadd231pd	%ymm1, %ymm3, %ymm13
	vfmadd231pd	%ymm1, %ymm15, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm15, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm15, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm15
	vfmadd231pd	%ymm1, %ymm5, %ymm4
	vfmadd231pd	%ymm1, %ymm3, %ymm2
	vmovapd	(%r8), %ymm5
	vmovapd	0(%r13), %ymm3
	vmovapd	%ymm15, -112(%rbp)
	vmovapd	(%rax), %ymm1
	vmovapd	384(%r14), %ymm15
	movl	$12, %r14d
	jmp	.L58
.L82:
	vxorpd	%xmm6, %xmm6, %xmm6
	movl	$1, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	vmovapd	%ymm6, %ymm2
	vmovapd	%ymm6, %ymm7
	vmovapd	%ymm6, %ymm10
	vmovapd	%ymm6, %ymm13
	vmovapd	%ymm6, %ymm4
	vmovapd	%ymm6, %ymm8
	vmovapd	%ymm6, %ymm11
	jmp	.L58
.L80:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$5, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	jmp	.L58
.L76:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$8, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	jmp	.L58
.L77:
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$9, %r14d
	jmp	.L58
.L81:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$6, %r14d
	vmovapd	%ymm6, -112(%rbp)
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm12
	jmp	.L58
.L78:
	movq	%rcx, -144(%rbp)
	movq	%rdi, %r13
	movq	%r9, %rax
	movq	%r10, %r8
	movl	$10, %r14d
	jmp	.L58
	.cfi_endproc
.LFE4591:
	.size	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_12x4_vs_lib4_new
	.section	.text.unlikely
.LCOLDE8:
	.text
.LHOTE8:
	.section	.text.unlikely
.LCOLDB9:
	.text
.LHOTB9:
	.p2align 4,,15
	.globl	kernel_dpotrf_nt_8x8_lib4_new
	.type	kernel_dpotrf_nt_8x8_lib4_new, @function
kernel_dpotrf_nt_8x8_lib4_new:
.LFB4592:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x68,0x6
	.cfi_escape 0x10,0xd,0x2,0x76,0x78
	.cfi_escape 0x10,0xc,0x2,0x76,0x70
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x60
	movl	8(%r10), %r12d
	movl	24(%r10), %eax
	movq	(%r10), %r13
	movq	16(%r10), %rbx
	movq	32(%r10), %r11
	leal	0(,%r12,4), %r10d
	sall	$2, %eax
	testl	%edi, %edi
	cltq
	movslq	%r10d, %r10
	leaq	0(%r13,%r10,8), %r12
	leaq	(%rbx,%rax,8), %r10
	jle	.L118
	sall	$2, %edx
	sall	$2, %r8d
	cmpl	$3, %edi
	movslq	%edx, %rdx
	movslq	%r8d, %r8
	vmovapd	(%rsi), %ymm14
	leaq	(%rsi,%rdx,8), %rdx
	leaq	(%rcx,%r8,8), %r8
	vmovapd	(%rcx), %ymm12
	vmovapd	(%rdx), %ymm1
	vmovapd	(%r8), %ymm2
	jle	.L118
	vxorpd	%xmm4, %xmm4, %xmm4
	subl	$4, %edi
	leaq	32(%rsi), %rax
	shrl	$2, %edi
	vmovapd	%ymm12, %ymm3
	salq	$7, %rdi
	leaq	160(%rsi,%rdi), %rsi
	vmovapd	%ymm4, %ymm15
	vmovapd	%ymm4, %ymm13
	vmovapd	%ymm4, %ymm0
	vmovapd	%ymm4, %ymm11
	vmovapd	%ymm4, %ymm8
	vmovapd	%ymm4, %ymm5
	vmovapd	%ymm4, %ymm7
	vmovapd	%ymm4, %ymm9
	vmovapd	%ymm4, %ymm10
	vmovapd	%ymm4, %ymm6
	vmovapd	%ymm4, %ymm12
	.p2align 4,,10
	.p2align 3
.L106:
	vfmadd231pd	%ymm3, %ymm14, %ymm6
	vfmadd231pd	%ymm3, %ymm1, %ymm5
	vfmadd231pd	%ymm2, %ymm1, %ymm13
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	subq	$-128, %rax
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rdx
	subq	$-128, %rcx
	subq	$-128, %r8
	vfmadd231pd	%ymm3, %ymm14, %ymm10
	vfmadd231pd	%ymm3, %ymm1, %ymm8
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm2, %ymm1, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm3, %ymm14, %ymm9
	vfmadd231pd	%ymm3, %ymm1, %ymm11
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm2, %ymm1, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm3, %ymm7, %ymm14
	vmovapd	-128(%rax), %ymm7
	vfmadd132pd	%ymm1, %ymm0, %ymm3
	vfmadd231pd	%ymm2, %ymm1, %ymm4
	vmovapd	-96(%rcx), %ymm0
	vmovapd	-96(%rdx), %ymm1
	vmovapd	-96(%r8), %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm14
	vfmadd231pd	%ymm0, %ymm1, %ymm3
	vmovapd	-96(%rax), %ymm7
	vmovapd	-64(%rcx), %ymm0
	vfmadd132pd	%ymm2, %ymm4, %ymm1
	vmovapd	-64(%rdx), %ymm2
	vmovapd	-64(%r8), %ymm4
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm2, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm4, %ymm2, %ymm13
	vshufpd	$5, %ymm4, %ymm4, %ymm4
	vfmadd231pd	%ymm0, %ymm7, %ymm10
	vfmadd231pd	%ymm0, %ymm2, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm4, %ymm2, %ymm12
	vperm2f128	$1, %ymm4, %ymm4, %ymm4
	vfmadd231pd	%ymm0, %ymm7, %ymm9
	vfmadd231pd	%ymm0, %ymm2, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm4, %ymm2, %ymm15
	vshufpd	$5, %ymm4, %ymm4, %ymm4
	vfmadd231pd	%ymm0, %ymm7, %ymm14
	vfmadd231pd	%ymm0, %ymm2, %ymm3
	vmovapd	-64(%rax), %ymm7
	vmovapd	-32(%rcx), %ymm0
	vfmadd132pd	%ymm4, %ymm1, %ymm2
	vmovapd	-32(%rdx), %ymm4
	vmovapd	-32(%r8), %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm4, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm10
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm12
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm9
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm4, %ymm15
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm0, %ymm14, %ymm7
	vmovapd	-32(%rax), %ymm14
	vfmadd132pd	%ymm4, %ymm3, %ymm0
	vmovapd	(%rcx), %ymm3
	vfmadd132pd	%ymm1, %ymm2, %ymm4
	vmovapd	(%rdx), %ymm1
	vmovapd	(%r8), %ymm2
	cmpq	%rax, %rsi
	jne	.L106
	vmovapd	%ymm12, -112(%rbp)
.L105:
	vblendpd	$10, %ymm10, %ymm6, %ymm1
	vblendpd	$10, %ymm9, %ymm7, %ymm3
	testl	%r9d, %r9d
	vblendpd	$5, %ymm10, %ymm6, %ymm10
	vblendpd	$5, %ymm9, %ymm7, %ymm9
	vblendpd	$10, %ymm8, %ymm5, %ymm6
	vblendpd	$12, %ymm3, %ymm1, %ymm12
	vblendpd	$3, %ymm9, %ymm10, %ymm7
	vblendpd	$5, %ymm8, %ymm5, %ymm5
	vblendpd	$3, %ymm3, %ymm1, %ymm3
	vblendpd	$12, %ymm9, %ymm10, %ymm1
	vblendpd	$10, %ymm11, %ymm0, %ymm10
	vblendpd	$5, %ymm11, %ymm0, %ymm11
	vblendpd	$12, %ymm10, %ymm6, %ymm2
	vblendpd	$12, %ymm11, %ymm5, %ymm9
	vblendpd	$3, %ymm10, %ymm6, %ymm10
	vblendpd	$3, %ymm11, %ymm5, %ymm5
	je	.L107
	vaddpd	0(%r13), %ymm12, %ymm12
	vaddpd	32(%r13), %ymm1, %ymm1
	vaddpd	64(%r13), %ymm3, %ymm3
	vaddpd	96(%r13), %ymm7, %ymm7
	vaddpd	(%r12), %ymm2, %ymm2
	vaddpd	32(%r12), %ymm9, %ymm9
	vaddpd	64(%r12), %ymm10, %ymm10
	vaddpd	96(%r12), %ymm5, %ymm5
.L107:
	vxorpd	%xmm6, %xmm6, %xmm6
	vxorpd	%xmm14, %xmm14, %xmm14
	vmovsd	%xmm12, %xmm6, %xmm6
	vmovapd	%xmm6, %xmm0
	vmovaps	%xmm6, -64(%rbp)
	vmovsd	.LC0(%rip), %xmm6
	vucomisd	%xmm6, %xmm0
	jbe	.L108
	vsqrtsd	%xmm0, %xmm0, %xmm8
	vmovsd	.LC1(%rip), %xmm0
	vdivsd	%xmm8, %xmm0, %xmm0
	vmovaps	%xmm0, -64(%rbp)
	vbroadcastsd	%xmm0, %ymm14
.L108:
	vmulpd	%ymm14, %ymm12, %ymm12
	vmovlpd	%xmm14, (%r11)
	vmulpd	%ymm14, %ymm2, %ymm2
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermpd	$85, %ymm12, %ymm14
	vmovapd	%ymm12, (%rbx)
	vmovapd	%ymm2, (%r10)
	vfmadd231pd	%ymm14, %ymm12, %ymm1
	vpermilpd	$3, %xmm1, %xmm8
	vfmadd132pd	%ymm2, %ymm9, %ymm14
	vucomisd	%xmm6, %xmm8
	jbe	.L109
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm8, %xmm8, %xmm8
	vdivsd	%xmm8, %xmm0, %xmm0
	vbroadcastsd	%xmm0, %ymm0
.L109:
	vmulpd	%ymm1, %ymm0, %ymm1
	vmovlpd	%xmm0, 8(%r11)
	vmovdqa	.LC2(%rip), %ymm9
	vmulpd	%ymm14, %ymm0, %ymm14
	vpermpd	$170, %ymm12, %ymm0
	vmaskmovpd	%ymm1, %ymm9, 32(%rbx)
	vfmadd231pd	%ymm0, %ymm12, %ymm3
	vfmadd231pd	%ymm0, %ymm2, %ymm10
	vpermpd	$170, %ymm1, %ymm0
	vmovapd	%ymm3, %ymm8
	vmovapd	%ymm14, 32(%r10)
	vxorpd	%xmm3, %xmm3, %xmm3
	vfmadd231pd	%ymm0, %ymm1, %ymm8
	vfmadd231pd	%ymm0, %ymm14, %ymm10
	vextractf128	$0x1, %ymm8, %xmm0
	vucomisd	%xmm6, %xmm0
	jbe	.L110
	vmovsd	.LC1(%rip), %xmm3
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vdivsd	%xmm0, %xmm3, %xmm0
	vbroadcastsd	%xmm0, %ymm3
.L110:
	vmulpd	%ymm8, %ymm3, %ymm8
	vmovlpd	%xmm3, 16(%r11)
	vmulpd	%ymm10, %ymm3, %ymm0
	vmovdqa	.LC3(%rip), %ymm10
	vpermpd	$255, %ymm1, %ymm3
	vpermpd	$255, %ymm12, %ymm11
	vmaskmovpd	%ymm8, %ymm10, 64(%rbx)
	vmovapd	%ymm0, 64(%r10)
	vfmadd231pd	%ymm11, %ymm12, %ymm7
	vfmadd132pd	%ymm2, %ymm5, %ymm11
	vfmadd231pd	%ymm3, %ymm1, %ymm7
	vfmadd132pd	%ymm14, %ymm11, %ymm3
	vpermpd	$255, %ymm8, %ymm11
	vfmadd132pd	%ymm11, %ymm7, %ymm8
	vextractf128	$0x1, %ymm8, %xmm1
	vfmadd231pd	%ymm11, %ymm0, %ymm3
	vxorpd	%xmm7, %xmm7, %xmm7
	vpermilpd	$3, %xmm1, %xmm1
	vucomisd	%xmm6, %xmm1
	jbe	.L111
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm7, %xmm1
	vbroadcastsd	%xmm1, %ymm7
.L111:
	vmulpd	%ymm3, %ymm7, %ymm11
	vmovlpd	%xmm7, 24(%r11)
	vfmadd231pd	%ymm2, %ymm2, %ymm13
	vmovdqa	.LC4(%rip), %ymm5
	vfmadd231pd	%ymm14, %ymm14, %ymm13
	vfmadd231pd	%ymm0, %ymm0, %ymm13
	vmulpd	%ymm8, %ymm7, %ymm7
	testl	%r9d, %r9d
	vshufpd	$5, %ymm2, %ymm2, %ymm3
	vshufpd	$5, %ymm14, %ymm14, %ymm8
	vfmadd231pd	%ymm11, %ymm11, %ymm13
	vmaskmovpd	%ymm7, %ymm5, 96(%rbx)
	vmovapd	%ymm11, 96(%r10)
	vshufpd	$5, %ymm0, %ymm0, %ymm7
	vmovapd	-112(%rbp), %ymm12
	vfmadd231pd	%ymm3, %ymm2, %ymm12
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm8, %ymm14, %ymm12
	vperm2f128	$1, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm7, %ymm0, %ymm12
	vperm2f128	$1, %ymm7, %ymm7, %ymm7
	vfmadd231pd	%ymm3, %ymm2, %ymm15
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm8, %ymm14, %ymm15
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm7, %ymm0, %ymm15
	vshufpd	$5, %ymm7, %ymm7, %ymm7
	vfmadd132pd	%ymm3, %ymm4, %ymm2
	vmovapd	%ymm14, %ymm3
	vfmadd132pd	%ymm8, %ymm2, %ymm3
	vfmadd132pd	%ymm7, %ymm3, %ymm0
	vshufpd	$5, %ymm11, %ymm11, %ymm3
	vfmadd231pd	%ymm3, %ymm11, %ymm12
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vblendpd	$10, %ymm12, %ymm13, %ymm1
	vblendpd	$5, %ymm12, %ymm13, %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm15
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm11, %ymm0, %ymm3
	vblendpd	$5, %ymm15, %ymm3, %ymm0
	vblendpd	$10, %ymm15, %ymm3, %ymm4
	vblendpd	$12, %ymm0, %ymm12, %ymm2
	vblendpd	$12, %ymm4, %ymm1, %ymm8
	vblendpd	$3, %ymm4, %ymm1, %ymm3
	vblendpd	$3, %ymm0, %ymm12, %ymm0
	je	.L112
	vaddpd	128(%r12), %ymm8, %ymm8
	vaddpd	160(%r12), %ymm2, %ymm2
	vaddpd	192(%r12), %ymm3, %ymm3
	vaddpd	224(%r12), %ymm0, %ymm0
.L112:
	vmovhps	-56(%rbp), %xmm8, %xmm7
	vxorpd	%xmm1, %xmm1, %xmm1
	vucomisd	%xmm6, %xmm7
	jbe	.L113
	vmovsd	.LC1(%rip), %xmm4
	vsqrtsd	%xmm7, %xmm7, %xmm7
	vdivsd	%xmm7, %xmm4, %xmm7
	vbroadcastsd	%xmm7, %ymm1
.L113:
	vmovlpd	%xmm1, 32(%r11)
	vmulpd	%ymm1, %ymm8, %ymm1
	vpermpd	$85, %ymm1, %ymm8
	vmovapd	%ymm1, 128(%r10)
	vfmadd132pd	%ymm1, %ymm2, %ymm8
	vpermilpd	$3, %xmm8, %xmm4
	vxorpd	%xmm2, %xmm2, %xmm2
	vucomisd	%xmm6, %xmm4
	jbe	.L114
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm4, %xmm7, %xmm4
	vbroadcastsd	%xmm4, %ymm2
.L114:
	vmovlpd	%xmm2, 40(%r11)
	vmulpd	%ymm8, %ymm2, %ymm2
	vpermpd	$170, %ymm1, %ymm4
	vpermpd	$170, %ymm2, %ymm7
	vfmadd231pd	%ymm4, %ymm1, %ymm3
	vmaskmovpd	%ymm2, %ymm9, 160(%r10)
	vxorpd	%xmm4, %xmm4, %xmm4
	vfmadd231pd	%ymm7, %ymm2, %ymm3
	vextractf128	$0x1, %ymm3, %xmm7
	vucomisd	%xmm6, %xmm7
	jbe	.L115
	vsqrtsd	%xmm7, %xmm7, %xmm4
	vmovsd	.LC1(%rip), %xmm7
	vdivsd	%xmm4, %xmm7, %xmm7
	vbroadcastsd	%xmm7, %ymm4
.L115:
	vmovlpd	%xmm4, 48(%r11)
	vmulpd	%ymm3, %ymm4, %ymm4
	vpermpd	$255, %ymm1, %ymm12
	vpermpd	$255, %ymm2, %ymm3
	vfmadd231pd	%ymm12, %ymm1, %ymm0
	vmaskmovpd	%ymm4, %ymm10, 192(%r10)
	vfmadd231pd	%ymm3, %ymm2, %ymm0
	vpermpd	$255, %ymm4, %ymm3
	vfmadd132pd	%ymm3, %ymm0, %ymm4
	vextractf128	$0x1, %ymm4, %xmm3
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermilpd	$3, %xmm3, %xmm3
	vucomisd	%xmm6, %xmm3
	jbe	.L116
	vsqrtsd	%xmm3, %xmm3, %xmm6
	vmovsd	.LC1(%rip), %xmm3
	vdivsd	%xmm6, %xmm3, %xmm3
	vbroadcastsd	%xmm3, %ymm0
.L116:
	vmovlpd	%xmm0, 56(%r11)
	vmulpd	%ymm4, %ymm0, %ymm0
	vmaskmovpd	%ymm0, %ymm5, 224(%r10)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L118:
	.cfi_restore_state
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovapd	%ymm4, %ymm15
	vmovapd	%ymm4, %ymm5
	vmovapd	%ymm4, -112(%rbp)
	vmovapd	%ymm4, %ymm13
	vmovapd	%ymm4, %ymm0
	vmovapd	%ymm4, %ymm11
	vmovapd	%ymm4, %ymm8
	vmovapd	%ymm4, %ymm7
	vmovapd	%ymm4, %ymm9
	vmovapd	%ymm4, %ymm10
	vmovapd	%ymm4, %ymm6
	jmp	.L105
	.cfi_endproc
.LFE4592:
	.size	kernel_dpotrf_nt_8x8_lib4_new, .-kernel_dpotrf_nt_8x8_lib4_new
	.section	.text.unlikely
.LCOLDE9:
	.text
.LHOTE9:
	.section	.text.unlikely
.LCOLDB10:
	.text
.LHOTB10:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_8x8_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_8x8_lib4_new, @function
kernel_dsyrk_dpotrf_nt_8x8_lib4_new:
.LFB4593:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$40, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	(%r10), %rax
	movl	%r9d, -152(%rbp)
	movl	48(%r10), %r9d
	movq	56(%r10), %r12
	movq	16(%r10), %r11
	movq	72(%r10), %rbx
	movq	%rax, -168(%rbp)
	movl	8(%r10), %eax
	sall	$2, %r9d
	movslq	%r9d, %r9
	movl	%eax, -172(%rbp)
	movl	24(%r10), %eax
	movl	%eax, -176(%rbp)
	movl	32(%r10), %eax
	movl	%eax, -148(%rbp)
	movq	40(%r10), %rax
	movq	%rax, %r15
	movq	%rax, -184(%rbp)
	movl	64(%r10), %eax
	leaq	(%r15,%r9,8), %r14
	sall	$2, %eax
	testl	%edi, %edi
	cltq
	leaq	(%r12,%rax,8), %r10
	jle	.L160
	sall	$2, %edx
	sall	$2, %r8d
	cmpl	$3, %edi
	movslq	%edx, %rdx
	movslq	%r8d, %r8
	vmovapd	(%rsi), %ymm10
	leaq	(%rcx,%r8,8), %rax
	leaq	(%rsi,%rdx,8), %r15
	vmovapd	(%rcx), %ymm13
	movq	%rax, %r9
	movq	%rax, -160(%rbp)
	vmovapd	(%r15), %ymm8
	vmovapd	(%rax), %ymm14
	jle	.L161
	vxorpd	%xmm7, %xmm7, %xmm7
	leal	-4(%rdi), %edx
	vmovapd	%ymm13, %ymm9
	leaq	32(%rcx), %rax
	movq	%r15, %r8
	shrl	$2, %edx
	movl	%edx, -196(%rbp)
	movq	%rdx, -192(%rbp)
	vmovapd	%ymm7, -80(%rbp)
	salq	$7, %rdx
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm12
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm3
	leaq	160(%rcx,%rdx), %r13
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, %ymm6
	movq	%rsi, %rdx
	vmovapd	%ymm7, %ymm4
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm13
	.p2align 4,,10
	.p2align 3
.L146:
	vfmadd231pd	%ymm9, %ymm10, %ymm1
	vfmadd231pd	%ymm9, %ymm8, %ymm2
	vshufpd	$5, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm14, %ymm8, %ymm12
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	%ymm10, %ymm15
	subq	$-128, %rax
	subq	$-128, %rdx
	subq	$-128, %r8
	vfmadd231pd	%ymm9, %ymm10, %ymm5
	vfmadd231pd	%ymm9, %ymm8, %ymm3
	vperm2f128	$1, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm14, %ymm8, %ymm11
	vperm2f128	$1, %ymm14, %ymm14, %ymm14
	subq	$-128, %r9
	vfmadd231pd	%ymm9, %ymm10, %ymm4
	vfmadd231pd	%ymm9, %ymm8, %ymm7
	vshufpd	$5, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm14, %ymm8, %ymm13
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	-96(%rdx), %ymm10
	vfmadd132pd	%ymm9, %ymm6, %ymm15
	vmovapd	-96(%r8), %ymm6
	vfmadd132pd	%ymm8, %ymm0, %ymm9
	vmovapd	-128(%rax), %ymm0
	vfmadd213pd	-80(%rbp), %ymm8, %ymm14
	vmovapd	-96(%r9), %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vfmadd231pd	%ymm0, %ymm6, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm12
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vfmadd231pd	%ymm0, %ymm6, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm11
	vperm2f128	$1, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vfmadd231pd	%ymm0, %ymm6, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm13
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm15
	vmovapd	-64(%rdx), %ymm10
	vfmadd132pd	%ymm6, %ymm9, %ymm0
	vmovapd	-64(%r8), %ymm9
	vfmadd231pd	%ymm8, %ymm6, %ymm14
	vmovapd	%ymm0, -80(%rbp)
	vmovapd	-96(%rax), %ymm0
	vmovapd	-64(%r9), %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vfmadd231pd	%ymm0, %ymm9, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm6, %ymm9, %ymm12
	vshufpd	$5, %ymm6, %ymm6, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vfmadd231pd	%ymm0, %ymm9, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm6, %ymm9, %ymm11
	vperm2f128	$1, %ymm6, %ymm6, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vfmadd231pd	%ymm0, %ymm9, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm6, %ymm6, %ymm8
	vfmadd231pd	%ymm6, %ymm9, %ymm13
	vmovapd	-32(%rdx), %ymm6
	vfmadd132pd	%ymm0, %ymm15, %ymm10
	vmovapd	-80(%rbp), %ymm15
	vfmadd231pd	%ymm0, %ymm9, %ymm15
	vmovapd	-64(%rax), %ymm0
	vfmadd132pd	%ymm8, %ymm14, %ymm9
	vmovapd	-32(%r8), %ymm8
	vmovapd	-32(%r9), %ymm14
	vfmadd231pd	%ymm0, %ymm6, %ymm1
	vfmadd231pd	%ymm0, %ymm8, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm9, -80(%rbp)
	vfmadd231pd	%ymm14, %ymm8, %ymm12
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	-32(%rax), %ymm9
	vfmadd231pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm8, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm14, %ymm8, %ymm11
	vperm2f128	$1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm0, %ymm6, %ymm4
	vfmadd231pd	%ymm0, %ymm8, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm14, %ymm8, %ymm13
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vfmadd132pd	%ymm0, %ymm10, %ymm6
	vmovapd	(%rdx), %ymm10
	vfmadd132pd	%ymm8, %ymm15, %ymm0
	vfmadd213pd	-80(%rbp), %ymm14, %ymm8
	vmovapd	(%r9), %ymm14
	vmovapd	%ymm8, -80(%rbp)
	vmovapd	(%r8), %ymm8
	cmpq	%r13, %rax
	jne	.L146
	movq	-192(%rbp), %rax
	vmovapd	%ymm13, -144(%rbp)
	vmovapd	%ymm9, %ymm13
	vmovapd	%ymm12, -112(%rbp)
	addq	$1, %rax
	salq	$7, %rax
	addq	%rax, -160(%rbp)
	addq	%rax, %rsi
	addq	%rax, %r15
	addq	%rax, %rcx
	movl	-196(%rbp), %eax
	leal	4(,%rax,4), %eax
.L145:
	leal	-1(%rdi), %edx
	cmpl	%eax, %edx
	jle	.L147
	vmovapd	-112(%rbp), %ymm9
	vfmadd231pd	%ymm13, %ymm10, %ymm1
	vfmadd231pd	%ymm13, %ymm8, %ymm2
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vmovapd	-144(%rbp), %ymm12
	vmovapd	%ymm10, %ymm15
	vfmadd231pd	%ymm14, %ymm8, %ymm9
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	movq	-160(%rbp), %rdx
	addl	$2, %eax
	vfmadd231pd	%ymm13, %ymm10, %ymm5
	vfmadd231pd	%ymm13, %ymm8, %ymm3
	vperm2f128	$1, %ymm13, %ymm13, %ymm13
	vfmadd231pd	%ymm14, %ymm8, %ymm11
	vperm2f128	$1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm10, %ymm4
	vfmadd231pd	%ymm13, %ymm8, %ymm7
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vfmadd231pd	%ymm14, %ymm8, %ymm12
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	32(%rdx), %ymm10
	vfmadd132pd	%ymm13, %ymm6, %ymm15
	vmovapd	32(%rsi), %ymm6
	vfmadd132pd	%ymm8, %ymm0, %ymm13
	vfmadd213pd	-80(%rbp), %ymm14, %ymm8
	vmovapd	32(%rcx), %ymm0
	vmovapd	%ymm8, -80(%rbp)
	vmovapd	32(%r15), %ymm8
	vfmadd231pd	%ymm0, %ymm6, %ymm1
	vfmadd231pd	%ymm0, %ymm8, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm10, %ymm8, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vmovapd	%ymm9, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm8, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm10, %ymm8, %ymm11
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm0, %ymm6, %ymm4
	vfmadd231pd	%ymm0, %ymm8, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm10, %ymm10, %ymm14
	vfmadd231pd	%ymm10, %ymm8, %ymm12
	vmovapd	64(%rsi), %ymm10
	vmovapd	%ymm12, -144(%rbp)
	vfmadd132pd	%ymm0, %ymm15, %ymm6
	vmovapd	%ymm8, %ymm15
	vfmadd132pd	%ymm8, %ymm13, %ymm0
	vmovapd	64(%rcx), %ymm13
	vfmadd213pd	-80(%rbp), %ymm14, %ymm15
	vmovapd	64(%r15), %ymm8
	vmovapd	%ymm15, -80(%rbp)
	vmovapd	64(%rdx), %ymm14
.L147:
	cmpl	%edi, %eax
	jl	.L187
.L144:
	movl	-152(%rbp), %esi
	testl	%esi, %esi
	jle	.L148
	movl	-172(%rbp), %eax
	movq	-168(%rbp), %rdi
	vmovapd	(%r11), %ymm9
	sall	$2, %eax
	vmovapd	(%rdi), %ymm13
	cltq
	leaq	(%rdi,%rax,8), %rdx
	movl	-176(%rbp), %eax
	vmovapd	(%rdx), %ymm14
	sall	$2, %eax
	cmpl	$3, %esi
	cltq
	leaq	(%r11,%rax,8), %rcx
	vmovapd	(%rcx), %ymm8
	jle	.L148
	subl	$4, %esi
	leaq	32(%rdi), %rax
	vmovapd	-112(%rbp), %ymm12
	shrl	$2, %esi
	vmovapd	-144(%rbp), %ymm15
	salq	$7, %rsi
	leaq	160(%rdi,%rsi), %rsi
	.p2align 4,,10
	.p2align 3
.L149:
	vfmadd231pd	%ymm9, %ymm13, %ymm1
	vfmadd231pd	%ymm9, %ymm14, %ymm2
	vfmadd231pd	%ymm8, %ymm14, %ymm12
	vshufpd	$5, %ymm9, %ymm9, %ymm9
	vmovapd	(%rax), %ymm10
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	subq	$-128, %rax
	subq	$-128, %rdx
	subq	$-128, %r11
	subq	$-128, %rcx
	vfmadd231pd	%ymm9, %ymm13, %ymm5
	vfmadd231pd	%ymm9, %ymm14, %ymm3
	vperm2f128	$1, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm8, %ymm14, %ymm11
	vperm2f128	$1, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm9, %ymm13, %ymm4
	vfmadd231pd	%ymm9, %ymm14, %ymm7
	vshufpd	$5, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm8, %ymm14, %ymm15
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd132pd	%ymm9, %ymm6, %ymm13
	vmovapd	-96(%rdx), %ymm6
	vfmadd132pd	%ymm14, %ymm0, %ymm9
	vfmadd213pd	-80(%rbp), %ymm8, %ymm14
	vmovapd	-96(%r11), %ymm0
	vmovapd	-96(%rcx), %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vfmadd231pd	%ymm0, %ymm6, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm12
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vfmadd231pd	%ymm0, %ymm6, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm11
	vperm2f128	$1, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vfmadd231pd	%ymm0, %ymm6, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm6, %ymm15
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm10, %ymm13
	vfmadd231pd	%ymm0, %ymm6, %ymm9
	vmovapd	-96(%rax), %ymm10
	vmovapd	-64(%r11), %ymm0
	vfmadd132pd	%ymm8, %ymm14, %ymm6
	vmovapd	-64(%rdx), %ymm8
	vmovapd	%ymm6, -80(%rbp)
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vmovapd	-64(%rcx), %ymm6
	vfmadd231pd	%ymm0, %ymm8, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm6, %ymm8, %ymm12
	vshufpd	$5, %ymm6, %ymm6, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vfmadd231pd	%ymm0, %ymm8, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm6, %ymm8, %ymm11
	vperm2f128	$1, %ymm6, %ymm6, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vfmadd231pd	%ymm0, %ymm8, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm6, %ymm6, %ymm14
	vfmadd231pd	%ymm6, %ymm8, %ymm15
	vmovapd	-64(%rax), %ymm6
	vfmadd231pd	%ymm0, %ymm8, %ymm9
	vfmadd132pd	%ymm0, %ymm13, %ymm10
	vmovapd	-32(%r11), %ymm0
	vmovapd	-32(%rax), %ymm13
	vfmadd213pd	-80(%rbp), %ymm14, %ymm8
	vmovapd	-32(%rdx), %ymm14
	vfmadd231pd	%ymm0, %ymm6, %ymm1
	vmovapd	%ymm8, -80(%rbp)
	vmovapd	-32(%rcx), %ymm8
	vfmadd231pd	%ymm0, %ymm14, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm14, %ymm12
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm14, %ymm3
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm14, %ymm11
	vperm2f128	$1, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm6, %ymm4
	vfmadd231pd	%ymm0, %ymm14, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm8, %ymm14, %ymm15
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd132pd	%ymm0, %ymm10, %ymm6
	vfmadd132pd	%ymm14, %ymm9, %ymm0
	vmovapd	(%r11), %ymm9
	vfmadd213pd	-80(%rbp), %ymm8, %ymm14
	vmovapd	(%rcx), %ymm8
	vmovapd	%ymm14, -80(%rbp)
	vmovapd	(%rdx), %ymm14
	cmpq	%rax, %rsi
	jne	.L149
	vmovapd	%ymm12, -112(%rbp)
	vmovapd	%ymm15, -144(%rbp)
.L148:
	vblendpd	$10, %ymm5, %ymm1, %ymm8
	vblendpd	$10, %ymm4, %ymm6, %ymm13
	movl	-148(%rbp), %edx
	vblendpd	$5, %ymm5, %ymm1, %ymm1
	vblendpd	$5, %ymm4, %ymm6, %ymm4
	vblendpd	$10, %ymm7, %ymm0, %ymm5
	vblendpd	$5, %ymm7, %ymm0, %ymm0
	vblendpd	$12, %ymm4, %ymm1, %ymm10
	vblendpd	$3, %ymm4, %ymm1, %ymm1
	testl	%edx, %edx
	vblendpd	$10, %ymm3, %ymm2, %ymm4
	vblendpd	$5, %ymm3, %ymm2, %ymm2
	vblendpd	$12, %ymm13, %ymm8, %ymm9
	vblendpd	$3, %ymm13, %ymm8, %ymm13
	vblendpd	$12, %ymm5, %ymm4, %ymm8
	vblendpd	$3, %ymm5, %ymm4, %ymm4
	vblendpd	$12, %ymm0, %ymm2, %ymm5
	vblendpd	$3, %ymm0, %ymm2, %ymm0
	je	.L150
	movq	-184(%rbp), %rax
	vaddpd	(%r14), %ymm8, %ymm8
	vaddpd	(%rax), %ymm9, %ymm9
	vaddpd	32(%rax), %ymm10, %ymm10
	vaddpd	64(%rax), %ymm13, %ymm13
	vaddpd	96(%rax), %ymm1, %ymm1
	vaddpd	32(%r14), %ymm5, %ymm5
	vaddpd	64(%r14), %ymm4, %ymm4
	vaddpd	96(%r14), %ymm0, %ymm0
.L150:
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovsd	.LC0(%rip), %xmm6
	vxorpd	%xmm15, %xmm15, %xmm15
	vmovsd	%xmm9, %xmm2, %xmm14
	vucomisd	%xmm6, %xmm14
	jbe	.L151
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm14, %xmm14, %xmm14
	vdivsd	%xmm14, %xmm2, %xmm14
	vbroadcastsd	%xmm14, %ymm15
.L151:
	vmulpd	%ymm15, %ymm9, %ymm9
	vmovlpd	%xmm15, (%rbx)
	vmulpd	%ymm15, %ymm8, %ymm8
	vxorpd	%xmm2, %xmm2, %xmm2
	vpermpd	$85, %ymm9, %ymm3
	vmovapd	%ymm9, (%r12)
	vmovapd	%ymm8, (%r10)
	vfmadd231pd	%ymm3, %ymm9, %ymm10
	vfmadd132pd	%ymm8, %ymm5, %ymm3
	vpermilpd	$3, %xmm10, %xmm5
	vucomisd	%xmm6, %xmm5
	jbe	.L152
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm5, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm2
.L152:
	vmulpd	%ymm10, %ymm2, %ymm10
	vmovapd	%ymm13, %ymm12
	vmovlpd	%xmm2, 8(%rbx)
	vpermpd	$170, %ymm9, %ymm5
	vmulpd	%ymm3, %ymm2, %ymm3
	vmovdqa	.LC2(%rip), %ymm2
	vfmadd231pd	%ymm5, %ymm9, %ymm12
	vfmadd231pd	%ymm5, %ymm8, %ymm4
	vpermpd	$170, %ymm10, %ymm5
	vmaskmovpd	%ymm10, %ymm2, 32(%r12)
	vxorpd	%xmm13, %xmm13, %xmm13
	vfmadd231pd	%ymm5, %ymm10, %ymm12
	vfmadd231pd	%ymm5, %ymm3, %ymm4
	vextractf128	$0x1, %ymm12, %xmm5
	vmovapd	%ymm3, 32(%r10)
	vucomisd	%xmm6, %xmm5
	jbe	.L153
	vmovsd	.LC1(%rip), %xmm13
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm5, %xmm13, %xmm5
	vbroadcastsd	%xmm5, %ymm13
.L153:
	vmulpd	%ymm12, %ymm13, %ymm12
	vmovlpd	%xmm13, 16(%rbx)
	vmulpd	%ymm4, %ymm13, %ymm5
	vmovdqa	.LC3(%rip), %ymm4
	vpermpd	$255, %ymm9, %ymm7
	vpermpd	$255, %ymm10, %ymm13
	vmaskmovpd	%ymm12, %ymm4, 64(%r12)
	vfmadd231pd	%ymm7, %ymm8, %ymm0
	vfmadd231pd	%ymm7, %ymm9, %ymm1
	vmovapd	%ymm5, 64(%r10)
	vfmadd231pd	%ymm13, %ymm10, %ymm1
	vfmadd132pd	%ymm3, %ymm0, %ymm13
	vpermpd	$255, %ymm12, %ymm0
	vxorpd	%xmm15, %xmm15, %xmm15
	vfmadd132pd	%ymm0, %ymm1, %ymm12
	vextractf128	$0x1, %ymm12, %xmm1
	vfmadd231pd	%ymm0, %ymm5, %ymm13
	vpermilpd	$3, %xmm1, %xmm1
	vucomisd	%xmm6, %xmm1
	jbe	.L154
	vmovsd	.LC1(%rip), %xmm15
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm15, %xmm1
	vbroadcastsd	%xmm1, %ymm15
.L154:
	vmulpd	%ymm13, %ymm15, %ymm13
	vmovlpd	%xmm15, 24(%rbx)
	vmovdqa	.LC4(%rip), %ymm7
	vmulpd	%ymm12, %ymm15, %ymm15
	vshufpd	$5, %ymm3, %ymm3, %ymm1
	vmaskmovpd	%ymm15, %ymm7, 96(%r12)
	movl	-148(%rbp), %eax
	vshufpd	$5, %ymm8, %ymm8, %ymm15
	vmovapd	%ymm13, 96(%r10)
	vmovapd	-144(%rbp), %ymm12
	testl	%eax, %eax
	vfmadd231pd	%ymm15, %ymm8, %ymm11
	vperm2f128	$1, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm1, %ymm3, %ymm11
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vmovapd	-112(%rbp), %ymm9
	vfmadd231pd	%ymm15, %ymm8, %ymm12
	vmovapd	%ymm12, %ymm0
	vshufpd	$5, %ymm15, %ymm15, %ymm15
	vfmadd231pd	%ymm8, %ymm8, %ymm9
	vfmadd231pd	%ymm3, %ymm3, %ymm9
	vfmadd231pd	%ymm5, %ymm5, %ymm9
	vfmadd231pd	%ymm1, %ymm3, %ymm0
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm13, %ymm13, %ymm9
	vfmadd213pd	-80(%rbp), %ymm15, %ymm8
	vfmadd132pd	%ymm1, %ymm8, %ymm3
	vshufpd	$5, %ymm5, %ymm5, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm11
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm0
	vmovapd	%ymm0, %ymm12
	vshufpd	$5, %ymm13, %ymm13, %ymm0
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm13, %ymm11
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm1, %ymm3, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm13, %ymm12
	vfmadd231pd	%ymm1, %ymm13, %ymm5
	vblendpd	$10, %ymm11, %ymm9, %ymm1
	vblendpd	$10, %ymm12, %ymm5, %ymm3
	vblendpd	$5, %ymm12, %ymm5, %ymm0
	vblendpd	$5, %ymm11, %ymm9, %ymm9
	vblendpd	$12, %ymm3, %ymm1, %ymm10
	vblendpd	$3, %ymm3, %ymm1, %ymm5
	vblendpd	$12, %ymm0, %ymm9, %ymm3
	vblendpd	$3, %ymm0, %ymm9, %ymm0
	je	.L155
	vaddpd	128(%r14), %ymm10, %ymm10
	vaddpd	160(%r14), %ymm3, %ymm3
	vaddpd	192(%r14), %ymm5, %ymm5
	vaddpd	224(%r14), %ymm0, %ymm0
.L155:
	vmovsd	%xmm10, %xmm14, %xmm14
	vxorpd	%xmm1, %xmm1, %xmm1
	vucomisd	%xmm6, %xmm14
	jbe	.L156
	vmovsd	.LC1(%rip), %xmm8
	vsqrtsd	%xmm14, %xmm14, %xmm14
	vdivsd	%xmm14, %xmm8, %xmm14
	vbroadcastsd	%xmm14, %ymm1
.L156:
	vmovlpd	%xmm1, 32(%rbx)
	vmulpd	%ymm1, %ymm10, %ymm1
	vpermpd	$85, %ymm1, %ymm10
	vmovapd	%ymm1, 128(%r10)
	vfmadd132pd	%ymm1, %ymm3, %ymm10
	vpermilpd	$3, %xmm10, %xmm8
	vxorpd	%xmm3, %xmm3, %xmm3
	vucomisd	%xmm6, %xmm8
	jbe	.L157
	vmovsd	.LC1(%rip), %xmm9
	vsqrtsd	%xmm8, %xmm8, %xmm8
	vdivsd	%xmm8, %xmm9, %xmm8
	vbroadcastsd	%xmm8, %ymm3
.L157:
	vmovlpd	%xmm3, 40(%rbx)
	vmulpd	%ymm10, %ymm3, %ymm3
	vpermpd	$170, %ymm3, %ymm8
	vmaskmovpd	%ymm3, %ymm2, 160(%r10)
	vpermpd	$170, %ymm1, %ymm2
	vfmadd231pd	%ymm2, %ymm1, %ymm5
	vfmadd231pd	%ymm8, %ymm3, %ymm5
	vextractf128	$0x1, %ymm5, %xmm8
	vxorpd	%xmm2, %xmm2, %xmm2
	vucomisd	%xmm6, %xmm8
	jbe	.L158
	vsqrtsd	%xmm8, %xmm8, %xmm2
	vmovsd	.LC1(%rip), %xmm8
	vdivsd	%xmm2, %xmm8, %xmm8
	vbroadcastsd	%xmm8, %ymm2
.L158:
	vmovlpd	%xmm2, 48(%rbx)
	vmulpd	%ymm5, %ymm2, %ymm2
	vpermpd	$255, %ymm1, %ymm9
	vmaskmovpd	%ymm2, %ymm4, 192(%r10)
	vfmadd231pd	%ymm9, %ymm1, %ymm0
	vpermpd	$255, %ymm3, %ymm4
	vfmadd231pd	%ymm4, %ymm3, %ymm0
	vpermpd	$255, %ymm2, %ymm4
	vfmadd132pd	%ymm4, %ymm0, %ymm2
	vextractf128	$0x1, %ymm2, %xmm4
	vxorpd	%xmm0, %xmm0, %xmm0
	vpermilpd	$3, %xmm4, %xmm4
	vucomisd	%xmm6, %xmm4
	jbe	.L159
	vsqrtsd	%xmm4, %xmm4, %xmm5
	vmovsd	.LC1(%rip), %xmm4
	vdivsd	%xmm5, %xmm4, %xmm4
	vbroadcastsd	%xmm4, %ymm0
.L159:
	vmovlpd	%xmm0, 56(%rbx)
	vmulpd	%ymm2, %ymm0, %ymm0
	vmaskmovpd	%ymm0, %ymm7, 224(%r10)
	vzeroupper
	addq	$40, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L187:
	.cfi_restore_state
	vmovapd	-112(%rbp), %ymm12
	vfmadd231pd	%ymm13, %ymm8, %ymm2
	vfmadd231pd	%ymm13, %ymm10, %ymm1
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vfmadd231pd	%ymm14, %ymm8, %ymm12
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	%ymm12, -112(%rbp)
	vmovapd	-144(%rbp), %ymm12
	vfmadd231pd	%ymm13, %ymm8, %ymm3
	vfmadd231pd	%ymm13, %ymm10, %ymm5
	vperm2f128	$1, %ymm13, %ymm13, %ymm13
	vfmadd231pd	%ymm14, %ymm8, %ymm11
	vperm2f128	$1, %ymm14, %ymm14, %ymm14
	vfmadd231pd	%ymm13, %ymm8, %ymm7
	vfmadd231pd	%ymm13, %ymm10, %ymm4
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vfmadd231pd	%ymm14, %ymm8, %ymm12
	vshufpd	$5, %ymm14, %ymm14, %ymm14
	vmovapd	%ymm12, -144(%rbp)
	vfmadd231pd	%ymm13, %ymm8, %ymm0
	vfmadd231pd	%ymm13, %ymm10, %ymm6
	vfmadd213pd	-80(%rbp), %ymm14, %ymm8
	vmovapd	%ymm8, -80(%rbp)
	jmp	.L144
	.p2align 4,,10
	.p2align 3
.L160:
	vxorpd	%xmm7, %xmm7, %xmm7
	vmovapd	%ymm7, -80(%rbp)
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm3
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm4
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, %ymm1
	jmp	.L144
.L161:
	vxorpd	%xmm7, %xmm7, %xmm7
	xorl	%eax, %eax
	vmovapd	%ymm7, -80(%rbp)
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm3
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm4
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, %ymm1
	jmp	.L145
	.cfi_endproc
.LFE4593:
	.size	kernel_dsyrk_dpotrf_nt_8x8_lib4_new, .-kernel_dsyrk_dpotrf_nt_8x8_lib4_new
	.section	.text.unlikely
.LCOLDE10:
	.text
.LHOTE10:
	.section	.text.unlikely
.LCOLDB15:
	.text
.LHOTB15:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_8x8_vs_lib4_new:
.LFB4594:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$8, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	%esi, -72(%rbp)
	movl	16(%r10), %esi
	movl	56(%r10), %r14d
	movl	48(%r10), %r11d
	movl	88(%r10), %r13d
	movq	64(%r10), %r15
	movl	%esi, -148(%rbp)
	movq	24(%r10), %rsi
	movl	%r14d, -68(%rbp)
	movl	72(%r10), %r14d
	movl	%r11d, -164(%rbp)
	movq	80(%r10), %r11
	sall	$2, %r13d
	movq	%rsi, -160(%rbp)
	movl	32(%r10), %esi
	movslq	%r13d, %r13
	sall	$2, %r14d
	cmpl	$7, %edi
	movq	(%r10), %rax
	movslq	%r14d, %r14
	movl	8(%r10), %ebx
	movq	%r15, -176(%rbp)
	movl	%esi, -152(%rbp)
	movq	96(%r10), %r12
	leaq	(%r15,%r14,8), %r15
	movq	40(%r10), %rsi
	leaq	(%r11,%r13,8), %r10
	jle	.L189
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%edx, %edx
	vmovdqu	%ymm0, mask_bkp.27500(%rip)
	jle	.L212
.L248:
	sall	$2, %r9d
	sall	$2, %ebx
	cmpl	$1, %ecx
	movslq	%r9d, %r9
	movslq	%ebx, %rbx
	vmovapd	(%r8), %ymm15
	leaq	(%r8,%r9,8), %rdi
	leaq	(%rax,%rbx,8), %rbx
	vmovapd	(%rax), %ymm2
	movq	%rdi, -144(%rbp)
	movq	%rbx, -80(%rbp)
	vmovapd	(%rdi), %ymm14
	vmovapd	(%rbx), %ymm10
	je	.L245
	vxorpd	%xmm1, %xmm1, %xmm1
	xorl	%r14d, %r14d
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm11
	vmovapd	%ymm1, %ymm6
	vmovapd	%ymm1, %ymm4
.L192:
	leal	-3(%rdx), %ecx
	movq	%rax, %rbx
	cmpl	%ecx, %r14d
	jge	.L196
	leal	-4(%rdx), %ecx
	movq	-80(%rbp), %r9
	movq	-144(%rbp), %rdi
	subl	%r14d, %ecx
	shrl	$2, %ecx
	movl	%ecx, %r13d
	movl	%ecx, -168(%rbp)
	movq	%r8, %rcx
	addq	$1, %r13
	salq	$7, %r13
	leaq	(%rax,%r13), %rbx
	.p2align 4,,10
	.p2align 3
.L197:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	subq	$-128, %rcx
	subq	$-128, %rdi
	subq	$-128, %r9
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vmovapd	-96(%rcx), %ymm15
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vmovapd	-96(%rax), %ymm0
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	-96(%rdi), %ymm10
	vmovapd	-96(%r9), %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm4
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd231pd	%ymm0, %ymm10, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm0, %ymm7, %ymm15
	vmovapd	-64(%rcx), %ymm7
	vfmadd132pd	%ymm10, %ymm2, %ymm0
	vmovapd	-64(%rdi), %ymm2
	vfmadd132pd	%ymm1, %ymm14, %ymm10
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	-64(%r9), %ymm1
	vmovapd	-64(%rax), %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	-112(%rbp), %ymm14
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vfmadd231pd	%ymm0, %ymm2, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm9
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm15
	vfmadd231pd	%ymm0, %ymm2, %ymm14
	vmovapd	-32(%rcx), %ymm7
	vmovapd	-32(%rax), %ymm0
	vfmadd132pd	%ymm1, %ymm10, %ymm2
	vmovapd	-32(%rdi), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	-32(%r9), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vmovapd	(%rax), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vmovapd	(%rcx), %ymm15
	vfmadd132pd	%ymm1, %ymm14, %ymm0
	vmovapd	(%rdi), %ymm14
	vfmadd213pd	-112(%rbp), %ymm10, %ymm1
	vmovapd	(%r9), %ymm10
	cmpq	%rbx, %rax
	jne	.L197
	movl	-168(%rbp), %eax
	addq	%r13, -144(%rbp)
	addq	%r13, %r8
	addq	%r13, -80(%rbp)
	leal	4(%r14,%rax,4), %r14d
.L196:
	leal	-1(%rdx), %eax
	cmpl	%r14d, %eax
	jle	.L198
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	movq	-144(%rbp), %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	movq	-80(%rbp), %rdi
	addl	$2, %r14d
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd132pd	%ymm2, %ymm7, %ymm15
	vmovapd	32(%r8), %ymm7
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	32(%rbx), %ymm0
	vmovapd	32(%rax), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	32(%rdi), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vmovapd	64(%rbx), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vfmadd213pd	-112(%rbp), %ymm1, %ymm0
	vmovapd	64(%r8), %ymm15
	vfmadd132pd	%ymm10, %ymm14, %ymm1
	vmovapd	64(%rax), %ymm14
	vmovapd	64(%rdi), %ymm10
.L198:
	cmpl	%edx, %r14d
	jl	.L246
.L191:
	movl	-148(%rbp), %edi
	testl	%edi, %edi
	jle	.L199
	movl	-152(%rbp), %eax
	movq	-160(%rbp), %rbx
	vmovapd	(%rsi), %ymm2
	sall	$2, %eax
	vmovapd	(%rbx), %ymm15
	cltq
	leaq	(%rbx,%rax,8), %rdx
	movl	-164(%rbp), %eax
	vmovapd	(%rdx), %ymm14
	sall	$2, %eax
	cmpl	$3, %edi
	cltq
	leaq	(%rsi,%rax,8), %rcx
	vmovapd	(%rcx), %ymm10
	jle	.L199
	subl	$4, %edi
	leaq	32(%rbx), %rax
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	160(%rbx,%rdi), %rdi
	.p2align 4,,10
	.p2align 3
.L200:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	subq	$-128, %rdx
	subq	$-128, %rsi
	subq	$-128, %rcx
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vmovapd	-128(%rax), %ymm15
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vmovapd	-96(%rsi), %ymm0
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	-96(%rdx), %ymm10
	vmovapd	-96(%rcx), %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm4
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd231pd	%ymm0, %ymm10, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm0, %ymm7, %ymm15
	vmovapd	-96(%rax), %ymm7
	vfmadd132pd	%ymm10, %ymm2, %ymm0
	vmovapd	-64(%rdx), %ymm2
	vfmadd132pd	%ymm1, %ymm14, %ymm10
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	-64(%rcx), %ymm1
	vmovapd	-64(%rsi), %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	-112(%rbp), %ymm14
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vfmadd231pd	%ymm0, %ymm2, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm8
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm9
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm15
	vfmadd231pd	%ymm0, %ymm2, %ymm14
	vmovapd	-64(%rax), %ymm7
	vmovapd	-32(%rsi), %ymm0
	vfmadd132pd	%ymm1, %ymm10, %ymm2
	vmovapd	-32(%rdx), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	-32(%rcx), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vmovapd	(%rsi), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vmovapd	-32(%rax), %ymm15
	vfmadd132pd	%ymm1, %ymm14, %ymm0
	vmovapd	(%rdx), %ymm14
	vfmadd213pd	-112(%rbp), %ymm10, %ymm1
	vmovapd	(%rcx), %ymm10
	cmpq	%rax, %rdi
	jne	.L200
.L199:
	vblendpd	$10, %ymm6, %ymm4, %ymm2
	movl	-68(%rbp), %edx
	vblendpd	$5, %ymm6, %ymm4, %ymm4
	vblendpd	$10, %ymm11, %ymm7, %ymm6
	vblendpd	$5, %ymm11, %ymm7, %ymm7
	vblendpd	$12, %ymm6, %ymm2, %ymm15
	vblendpd	$12, %ymm7, %ymm4, %ymm14
	testl	%edx, %edx
	vblendpd	$3, %ymm6, %ymm2, %ymm2
	vblendpd	$3, %ymm7, %ymm4, %ymm7
	vblendpd	$10, %ymm12, %ymm0, %ymm6
	vblendpd	$10, %ymm13, %ymm5, %ymm4
	vblendpd	$5, %ymm12, %ymm0, %ymm0
	vblendpd	$5, %ymm13, %ymm5, %ymm5
	vblendpd	$12, %ymm6, %ymm4, %ymm12
	vblendpd	$3, %ymm6, %ymm4, %ymm4
	vblendpd	$12, %ymm0, %ymm5, %ymm11
	vblendpd	$3, %ymm0, %ymm5, %ymm0
	je	.L201
	movq	-176(%rbp), %rax
	vaddpd	(%r15), %ymm12, %ymm12
	vaddpd	(%rax), %ymm15, %ymm15
	vaddpd	32(%rax), %ymm14, %ymm14
	vaddpd	64(%rax), %ymm2, %ymm2
	vaddpd	96(%rax), %ymm7, %ymm7
	vaddpd	32(%r15), %ymm11, %ymm11
	vaddpd	64(%r15), %ymm4, %ymm4
	vaddpd	96(%r15), %ymm0, %ymm0
.L201:
	vxorpd	%xmm5, %xmm5, %xmm5
	vmovsd	.LC0(%rip), %xmm13
	vxorpd	%xmm10, %xmm10, %xmm10
	vmovsd	%xmm15, %xmm5, %xmm6
	vucomisd	%xmm13, %xmm6
	jbe	.L202
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm5, %xmm6
	vbroadcastsd	%xmm6, %ymm10
.L202:
	vmulpd	%ymm10, %ymm15, %ymm15
	vmovlpd	%xmm10, (%r12)
	vmulpd	%ymm10, %ymm12, %ymm10
	vmovdqu	mask_bkp.27500(%rip), %ymm5
	vmovapd	%ymm15, (%r11)
	vmaskmovpd	%ymm10, %ymm5, (%r10)
	vpermpd	$85, %ymm15, %ymm5
	vfmadd231pd	%ymm5, %ymm15, %ymm14
	vpermilpd	$3, %xmm14, %xmm12
	vfmadd231pd	%ymm5, %ymm10, %ymm11
	vxorpd	%xmm5, %xmm5, %xmm5
	vucomisd	%xmm13, %xmm12
	jbe	.L203
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm12, %xmm12, %xmm12
	vdivsd	%xmm12, %xmm5, %xmm5
	vbroadcastsd	%xmm5, %ymm5
.L203:
	vmulpd	%ymm14, %ymm5, %ymm14
	vmovlpd	%xmm5, 8(%r12)
	vmulpd	%ymm11, %ymm5, %ymm5
	vmovdqa	.LC2(%rip), %ymm11
	vmovdqu	mask_bkp.27500(%rip), %ymm12
	vmaskmovpd	%ymm14, %ymm11, 32(%r11)
	vpermpd	$170, %ymm15, %ymm11
	vmaskmovpd	%ymm5, %ymm12, 32(%r10)
	vfmadd231pd	%ymm11, %ymm15, %ymm2
	vfmadd231pd	%ymm11, %ymm10, %ymm4
	vpermpd	$170, %ymm14, %ymm11
	vfmadd231pd	%ymm11, %ymm14, %ymm2
	vextractf128	$0x1, %ymm2, %xmm12
	vfmadd231pd	%ymm11, %ymm5, %ymm4
	vxorpd	%xmm11, %xmm11, %xmm11
	vucomisd	%xmm13, %xmm12
	jbe	.L204
	vmovsd	.LC1(%rip), %xmm11
	vsqrtsd	%xmm12, %xmm12, %xmm12
	vdivsd	%xmm12, %xmm11, %xmm11
	vbroadcastsd	%xmm11, %ymm11
.L204:
	vmulpd	%ymm2, %ymm11, %ymm2
	vmovlpd	%xmm11, 16(%r12)
	vmulpd	%ymm4, %ymm11, %ymm4
	vmovdqa	.LC3(%rip), %ymm11
	vmovdqu	mask_bkp.27500(%rip), %ymm12
	vmaskmovpd	%ymm2, %ymm11, 64(%r11)
	vpermpd	$255, %ymm15, %ymm11
	vmaskmovpd	%ymm4, %ymm12, 64(%r10)
	vfmadd132pd	%ymm11, %ymm7, %ymm15
	vpermpd	$255, %ymm14, %ymm7
	vfmadd231pd	%ymm11, %ymm10, %ymm0
	vfmadd231pd	%ymm7, %ymm14, %ymm15
	vpermpd	$255, %ymm2, %ymm14
	vfmadd231pd	%ymm7, %ymm5, %ymm0
	vmovapd	%ymm2, %ymm7
	vxorpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	%ymm14, %ymm15, %ymm7
	vextractf128	$0x1, %ymm7, %xmm11
	vfmadd132pd	%ymm4, %ymm0, %ymm14
	vpermilpd	$3, %xmm11, %xmm11
	vucomisd	%xmm13, %xmm11
	jbe	.L205
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm11, %xmm11, %xmm11
	vdivsd	%xmm11, %xmm2, %xmm2
	vbroadcastsd	%xmm2, %ymm2
.L205:
	vmulpd	%ymm7, %ymm2, %ymm7
	vmovlpd	%xmm2, 24(%r12)
	vfmadd231pd	%ymm10, %ymm10, %ymm3
	vmulpd	%ymm14, %ymm2, %ymm14
	vmovdqa	.LC4(%rip), %ymm2
	vmovdqu	mask_bkp.27500(%rip), %ymm0
	vshufpd	$5, %ymm5, %ymm5, %ymm11
	vmaskmovpd	%ymm7, %ymm2, 96(%r11)
	vmovapd	%ymm3, %ymm2
	vshufpd	$5, %ymm10, %ymm10, %ymm3
	vshufpd	$5, %ymm4, %ymm4, %ymm7
	vfmadd231pd	%ymm5, %ymm5, %ymm2
	vfmadd231pd	%ymm4, %ymm4, %ymm2
	vmaskmovpd	%ymm14, %ymm0, 96(%r10)
	movl	-68(%rbp), %eax
	vfmadd231pd	%ymm14, %ymm14, %ymm2
	vfmadd231pd	%ymm3, %ymm10, %ymm8
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vmovapd	%ymm8, %ymm0
	vfmadd231pd	%ymm11, %ymm5, %ymm0
	vperm2f128	$1, %ymm11, %ymm11, %ymm11
	vfmadd231pd	%ymm7, %ymm4, %ymm0
	vfmadd231pd	%ymm3, %ymm10, %ymm9
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	testl	%eax, %eax
	vperm2f128	$1, %ymm7, %ymm7, %ymm7
	vfmadd132pd	%ymm3, %ymm1, %ymm10
	vshufpd	$5, %ymm14, %ymm14, %ymm3
	vmovapd	%ymm9, %ymm1
	vfmadd231pd	%ymm11, %ymm5, %ymm1
	vshufpd	$5, %ymm11, %ymm11, %ymm11
	vfmadd231pd	%ymm7, %ymm4, %ymm1
	vfmadd231pd	%ymm3, %ymm14, %ymm0
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vshufpd	$5, %ymm7, %ymm7, %ymm7
	vfmadd132pd	%ymm11, %ymm10, %ymm5
	vfmadd231pd	%ymm3, %ymm14, %ymm1
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm7, %ymm5, %ymm4
	vblendpd	$10, %ymm0, %ymm2, %ymm5
	vblendpd	$5, %ymm0, %ymm2, %ymm0
	vfmadd132pd	%ymm14, %ymm4, %ymm3
	vblendpd	$10, %ymm1, %ymm3, %ymm7
	vblendpd	$5, %ymm1, %ymm3, %ymm3
	vblendpd	$12, %ymm7, %ymm5, %ymm4
	vblendpd	$3, %ymm7, %ymm5, %ymm7
	vblendpd	$12, %ymm3, %ymm0, %ymm5
	vblendpd	$3, %ymm3, %ymm0, %ymm0
	je	.L206
	vaddpd	128(%r15), %ymm4, %ymm4
	vaddpd	160(%r15), %ymm5, %ymm5
	vaddpd	192(%r15), %ymm7, %ymm7
	vaddpd	224(%r15), %ymm0, %ymm0
.L206:
	vmovsd	%xmm4, %xmm6, %xmm6
	vxorpd	%xmm1, %xmm1, %xmm1
	vucomisd	%xmm13, %xmm6
	jbe	.L207
	vmovsd	.LC1(%rip), %xmm3
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vdivsd	%xmm6, %xmm3, %xmm6
	vbroadcastsd	%xmm6, %ymm1
.L207:
	vmovlpd	%xmm1, 32(%r12)
	vmulpd	%ymm1, %ymm4, %ymm1
	vmovdqu	mask_bkp.27500(%rip), %ymm2
	vmaskmovpd	%ymm1, %ymm2, 128(%r10)
	vpermpd	$85, %ymm1, %ymm2
	vmovapd	%ymm2, %ymm3
	vxorpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	%ymm1, %ymm5, %ymm3
	vpermilpd	$3, %xmm3, %xmm4
	vucomisd	%xmm13, %xmm4
	jbe	.L208
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm4, %xmm5, %xmm4
	vbroadcastsd	%xmm4, %ymm2
.L208:
	vmovlpd	%xmm2, 40(%r12)
	vmulpd	%ymm3, %ymm2, %ymm2
	vmovdqu	mask_bkp.27500(%rip), %ymm4
	vpermpd	$170, %ymm1, %ymm3
	vandpd	.LC12(%rip), %ymm4, %ymm4
	vfmadd231pd	%ymm3, %ymm1, %ymm7
	vxorpd	%xmm3, %xmm3, %xmm3
	vmaskmovpd	%ymm2, %ymm4, 160(%r10)
	vpermpd	$170, %ymm2, %ymm4
	vfmadd231pd	%ymm4, %ymm2, %ymm7
	vextractf128	$0x1, %ymm7, %xmm4
	vucomisd	%xmm13, %xmm4
	jbe	.L209
	vsqrtsd	%xmm4, %xmm4, %xmm3
	vmovsd	.LC1(%rip), %xmm4
	vdivsd	%xmm3, %xmm4, %xmm4
	vbroadcastsd	%xmm4, %ymm3
.L209:
	vmovlpd	%xmm3, 48(%r12)
	vmulpd	%ymm7, %ymm3, %ymm3
	vmovdqu	mask_bkp.27500(%rip), %ymm4
	vandpd	.LC13(%rip), %ymm4, %ymm4
	vmaskmovpd	%ymm3, %ymm4, 192(%r10)
	cmpl	$7, -72(%rbp)
	jle	.L243
	vpermpd	$255, %ymm1, %ymm4
	vfmadd231pd	%ymm4, %ymm1, %ymm0
	vpermpd	$255, %ymm2, %ymm4
	vfmadd132pd	%ymm4, %ymm0, %ymm2
	vpermpd	$255, %ymm3, %ymm4
	vfmadd132pd	%ymm4, %ymm2, %ymm3
	vextractf128	$0x1, %ymm3, %xmm5
	vxorpd	%xmm4, %xmm4, %xmm4
	vpermilpd	$3, %xmm5, %xmm5
	vucomisd	%xmm13, %xmm5
	ja	.L247
.L211:
	vmovlpd	%xmm4, 56(%r12)
	vmulpd	%ymm3, %ymm4, %ymm4
	vmovdqu	mask_bkp.27500(%rip), %ymm0
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmaskmovpd	%ymm4, %ymm0, 224(%r10)
.L243:
	vzeroupper
	addq	$8, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L189:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27501(%rip), %ymm1
	testl	%edx, %edx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC11(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.27500(%rip)
	jg	.L248
.L212:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm11
	vmovapd	%ymm1, %ymm6
	vmovapd	%ymm1, %ymm4
	jmp	.L191
	.p2align 4,,10
	.p2align 3
.L246:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	jmp	.L191
	.p2align 4,,10
	.p2align 3
.L247:
	vsqrtsd	%xmm5, %xmm5, %xmm6
	vmovsd	.LC1(%rip), %xmm5
	vdivsd	%xmm6, %xmm5, %xmm5
	vbroadcastsd	%xmm5, %ymm4
	jmp	.L211
	.p2align 4,,10
	.p2align 3
.L245:
	cmpl	$3, %edx
	jle	.L193
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	32(%rax), %ymm0
	movq	-80(%rbp), %r13
	vmovapd	128(%rdi), %ymm14
	movq	%rdi, %r14
	subq	$-128, %rdi
	leaq	128(%r8), %rbx
	leaq	128(%rax), %r9
	vblendpd	$1, %ymm15, %ymm1, %ymm4
	vblendpd	$3, 32(%r8), %ymm1, %ymm6
	movq	%r13, %rcx
	vblendpd	$7, 64(%r8), %ymm1, %ymm7
	subq	$-128, %rcx
	cmpl	$7, %edx
	vfmadd132pd	%ymm2, %ymm1, %ymm4
	vfmadd231pd	%ymm0, %ymm6, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm7, %ymm11
	vmovapd	96(%rax), %ymm2
	vmovapd	128(%r8), %ymm15
	vfmadd132pd	%ymm0, %ymm1, %ymm6
	vblendpd	$7, 64(%rax), %ymm1, %ymm0
	vmovapd	128(%r13), %ymm10
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm7, %ymm1, %ymm0
	vmovapd	96(%r8), %ymm7
	vfmadd231pd	%ymm2, %ymm7, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm7, %ymm6
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm7, %ymm11
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm0, %ymm7
	vmovapd	128(%rax), %ymm2
	jle	.L194
	vblendpd	$1, %ymm14, %ymm1, %ymm14
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vmovapd	%ymm10, %ymm8
	vblendpd	$3, 160(%r14), %ymm1, %ymm0
	vmovapd	160(%r8), %ymm3
	movq	%r14, %rbx
	vmovapd	%ymm14, %ymm5
	vmovapd	%ymm14, %ymm10
	vmovapd	160(%rax), %ymm12
	vmovapd	%ymm14, %ymm9
	vfmadd132pd	%ymm14, %ymm1, %ymm8
	vmovapd	192(%rax), %ymm13
	vfmadd132pd	%ymm2, %ymm1, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm12, %ymm3, %ymm4
	vfmadd231pd	%ymm12, %ymm0, %ymm5
	vshufpd	$5, %ymm12, %ymm12, %ymm12
	movq	%r13, %rcx
	addq	$256, %rbx
	addq	$256, %rcx
	addq	$256, %r8
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm12, %ymm3, %ymm6
	vfmadd231pd	%ymm12, %ymm0, %ymm10
	vperm2f128	$1, %ymm12, %ymm12, %ymm12
	addq	$256, %rax
	movq	%rcx, -80(%rbp)
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd132pd	%ymm2, %ymm1, %ymm9
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm12, %ymm3, %ymm11
	vfmadd231pd	%ymm12, %ymm0, %ymm9
	vshufpd	$5, %ymm12, %ymm12, %ymm12
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm14
	vmovapd	160(%r13), %ymm2
	vfmadd132pd	%ymm12, %ymm7, %ymm3
	vmovapd	-64(%r8), %ymm7
	vmovapd	%ymm3, %ymm15
	vfmadd231pd	%ymm2, %ymm0, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm8, %ymm3
	vfmadd231pd	%ymm13, %ymm7, %ymm4
	vfmadd132pd	%ymm0, %ymm1, %ymm2
	vfmadd132pd	%ymm12, %ymm14, %ymm0
	vblendpd	$7, 192(%r13), %ymm1, %ymm12
	vmovapd	%ymm0, -112(%rbp)
	vblendpd	$7, 192(%r14), %ymm1, %ymm0
	vfmadd231pd	%ymm13, %ymm0, %ymm5
	vfmadd231pd	%ymm12, %ymm0, %ymm3
	vshufpd	$5, %ymm13, %ymm13, %ymm13
	vshufpd	$5, %ymm12, %ymm12, %ymm12
	vmovapd	%ymm0, %ymm14
	vperm2f128	$1, %ymm13, %ymm13, %ymm8
	vfmadd231pd	%ymm13, %ymm7, %ymm6
	vfmadd231pd	%ymm13, %ymm0, %ymm10
	vfmadd231pd	%ymm12, %ymm0, %ymm2
	vperm2f128	$1, %ymm12, %ymm12, %ymm12
	vmovapd	%ymm10, %ymm13
	vfmadd231pd	%ymm8, %ymm7, %ymm11
	vfmadd231pd	%ymm8, %ymm0, %ymm9
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd132pd	%ymm12, %ymm1, %ymm14
	vshufpd	$5, %ymm12, %ymm12, %ymm12
	vfmadd231pd	%ymm8, %ymm7, %ymm15
	vmovapd	-32(%r8), %ymm7
	vfmadd213pd	-112(%rbp), %ymm0, %ymm8
	vfmadd132pd	%ymm12, %ymm1, %ymm0
	vmovapd	%ymm8, -112(%rbp)
	vmovapd	224(%r14), %ymm1
	vmovapd	-32(%rax), %ymm8
	vmovapd	%ymm0, -144(%rbp)
	vmovapd	224(%r13), %ymm0
	vfmadd231pd	%ymm8, %ymm7, %ymm4
	vfmadd231pd	%ymm8, %ymm1, %ymm5
	vshufpd	$5, %ymm8, %ymm8, %ymm8
	vfmadd231pd	%ymm0, %ymm1, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vperm2f128	$1, %ymm8, %ymm8, %ymm10
	vfmadd231pd	%ymm8, %ymm7, %ymm6
	vfmadd231pd	%ymm8, %ymm1, %ymm13
	vfmadd231pd	%ymm0, %ymm1, %ymm2
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm2, %ymm8
	vmovapd	(%rax), %ymm2
	vfmadd231pd	%ymm10, %ymm1, %ymm9
	vmovapd	%ymm9, %ymm12
	vmovapd	%ymm14, %ymm9
	vfmadd231pd	%ymm10, %ymm7, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm14
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm9
	vmovapd	-112(%rbp), %ymm0
	vfmadd132pd	%ymm10, %ymm15, %ymm7
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vmovapd	(%r8), %ymm15
	vfmadd213pd	-144(%rbp), %ymm14, %ymm1
	vmovapd	256(%r13), %ymm10
	movq	%rbx, -144(%rbp)
	vmovapd	256(%r14), %ymm14
	movl	$8, %r14d
	jmp	.L192
	.p2align 4,,10
	.p2align 3
.L193:
	vxorpd	%xmm0, %xmm0, %xmm0
	cmpl	$1, %edx
	vblendpd	$1, %ymm15, %ymm0, %ymm4
	vmovapd	32(%r8), %ymm15
	vfmadd132pd	%ymm2, %ymm0, %ymm4
	vmovapd	32, %ymm2
	je	.L217
	vblendpd	$3, %ymm15, %ymm0, %ymm15
	cmpl	$3, %edx
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm15, %ymm6
	vmovapd	64(%r8), %ymm15
	vfmadd132pd	%ymm2, %ymm0, %ymm6
	vmovapd	64, %ymm2
	jne	.L218
	vblendpd	$7, %ymm15, %ymm0, %ymm7
	vblendpd	$7, %ymm2, %ymm0, %ymm1
	vmovapd	96(%r8), %ymm15
	vmovapd	96, %ymm2
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm8
	vfmadd231pd	%ymm1, %ymm7, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm0, %ymm3
	vmovapd	%ymm0, %ymm12
	movl	$3, %r14d
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	vfmadd231pd	%ymm1, %ymm7, %ymm6
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm11
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm7
	vmovapd	%ymm0, %ymm1
	jmp	.L192
.L194:
	cmpl	$4, %edx
	je	.L214
	vblendpd	$1, %ymm14, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	movq	-144(%rbp), %r14
	movq	-80(%rbp), %r13
	vmovapd	%ymm10, %ymm3
	cmpl	$5, %edx
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm12
	vfmadd132pd	%ymm0, %ymm1, %ymm3
	vmovapd	160(%r14), %ymm14
	vfmadd132pd	%ymm2, %ymm1, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	160(%r13), %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd132pd	%ymm2, %ymm1, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm0
	vmovapd	160(%r8), %ymm15
	vmovapd	160(%rax), %ymm2
	je	.L215
	vblendpd	$3, %ymm14, %ymm1, %ymm14
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	cmpl	$7, %edx
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm10, %ymm8
	vmovapd	192(%r13), %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm14, %ymm1, %ymm8
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	192(%r8), %ymm15
	vmovapd	192(%rax), %ymm2
	vmovapd	192(%r14), %ymm14
	jne	.L216
	vblendpd	$7, %ymm14, %ymm1, %ymm14
	vblendpd	$7, %ymm10, %ymm1, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	movq	%rcx, -80(%rbp)
	movq	%rdi, -144(%rbp)
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vmovapd	%ymm14, %ymm9
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm10, %ymm1, %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	224(%r8), %ymm15
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	vmovapd	224(%rax), %ymm2
	movq	%rbx, %r8
	vmovapd	224(%r14), %ymm14
	movq	%r9, %rax
	movl	$7, %r14d
	vmovapd	224(%r13), %ymm10
	jmp	.L192
.L218:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$2, %r14d
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm11
	jmp	.L192
.L214:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%rcx, -80(%rbp)
	movq	%rdi, -144(%rbp)
	movq	%r9, %rax
	movq	%rbx, %r8
	movl	$4, %r14d
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	jmp	.L192
.L217:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$1, %r14d
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm11
	vmovapd	%ymm1, %ymm6
	jmp	.L192
.L215:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%rcx, -80(%rbp)
	movq	%rdi, -144(%rbp)
	movq	%r9, %rax
	movq	%rbx, %r8
	movl	$5, %r14d
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm8
	jmp	.L192
.L216:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%rcx, -80(%rbp)
	movq	%rdi, -144(%rbp)
	movq	%r9, %rax
	movq	%rbx, %r8
	movl	$6, %r14d
	vmovapd	%ymm1, %ymm9
	jmp	.L192
	.cfi_endproc
.LFE4594:
	.size	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_8x8_vs_lib4_new
	.section	.text.unlikely
.LCOLDE15:
	.text
.LHOTE15:
	.section	.text.unlikely
.LCOLDB16:
	.text
.LHOTB16:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_8x4_vs_lib4_new:
.LFB4595:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	8(%r10), %eax
	movq	64(%r10), %rbx
	movl	%esi, -76(%rbp)
	movq	(%r10), %r12
	movq	16(%r10), %rsi
	movq	32(%r10), %r11
	movl	%eax, -80(%rbp)
	movl	24(%r10), %eax
	movq	80(%r10), %r15
	movl	%eax, -88(%rbp)
	movl	40(%r10), %eax
	movl	%eax, -84(%rbp)
	movq	48(%r10), %rax
	movq	%rax, -96(%rbp)
	movl	56(%r10), %eax
	movl	%eax, -100(%rbp)
	movl	72(%r10), %eax
	sall	$2, %eax
	cmpl	$7, %edi
	cltq
	leaq	(%rbx,%rax,8), %rax
	movq	%rax, -72(%rbp)
	jle	.L250
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%edx, %edx
	vmovdqu	%ymm0, mask_bkp.27604(%rip)
	jle	.L271
.L288:
	sall	$2, %r9d
	cmpl	$1, %ecx
	vmovapd	(%r8), %ymm9
	movslq	%r9d, %r9
	vmovapd	(%r12), %ymm3
	leaq	(%r8,%r9,8), %r14
	vmovapd	(%r14), %ymm10
	je	.L285
	vxorpd	%xmm7, %xmm7, %xmm7
	xorl	%r13d, %r13d
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, %ymm4
.L253:
	leal	-3(%rdx), %eax
	cmpl	%eax, %r13d
	jge	.L257
	leal	-4(%rdx), %ecx
	leaq	32(%r12), %rax
	movq	%r14, %rdi
	subl	%r13d, %ecx
	shrl	$2, %ecx
	movl	%ecx, %r10d
	movl	%ecx, -104(%rbp)
	movq	%r10, %rcx
	salq	$7, %rcx
	leaq	160(%r12,%rcx), %r9
	movq	%r8, %rcx
	.p2align 4,,10
	.p2align 3
.L258:
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vfmadd231pd	%ymm3, %ymm10, %ymm2
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vmovapd	(%rax), %ymm3
	subq	$-128, %rax
	subq	$-128, %rcx
	vmovapd	-64(%rcx), %ymm15
	subq	$-128, %rdi
	vperm2f128	$1, %ymm11, %ymm11, %ymm12
	vfmadd231pd	%ymm11, %ymm9, %ymm5
	vfmadd231pd	%ymm11, %ymm10, %ymm1
	vmovapd	-64(%rdi), %ymm14
	vshufpd	$5, %ymm12, %ymm12, %ymm11
	vfmadd231pd	%ymm12, %ymm9, %ymm6
	vfmadd231pd	%ymm12, %ymm10, %ymm0
	vmovapd	-96(%rcx), %ymm12
	vfmadd132pd	%ymm11, %ymm7, %ymm10
	vmovapd	-96(%rdi), %ymm7
	vfmadd231pd	%ymm11, %ymm9, %ymm8
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vfmadd231pd	%ymm3, %ymm12, %ymm4
	vfmadd132pd	%ymm7, %ymm2, %ymm3
	vmovapd	-96(%rax), %ymm2
	vperm2f128	$1, %ymm11, %ymm11, %ymm9
	vfmadd231pd	%ymm11, %ymm7, %ymm1
	vfmadd231pd	%ymm11, %ymm12, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm11
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd132pd	%ymm14, %ymm3, %ymm2
	vmovapd	-64(%rax), %ymm3
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vfmadd231pd	%ymm9, %ymm7, %ymm0
	vfmadd231pd	%ymm9, %ymm12, %ymm6
	vperm2f128	$1, %ymm11, %ymm11, %ymm9
	vfmadd231pd	%ymm11, %ymm15, %ymm5
	vfmadd231pd	%ymm11, %ymm14, %ymm1
	vfmadd132pd	%ymm13, %ymm10, %ymm7
	vfmadd231pd	%ymm13, %ymm12, %ymm8
	vshufpd	$5, %ymm9, %ymm9, %ymm10
	vfmadd231pd	%ymm9, %ymm15, %ymm6
	vfmadd231pd	%ymm9, %ymm14, %ymm0
	vfmadd132pd	%ymm10, %ymm8, %ymm15
	vfmadd132pd	%ymm10, %ymm7, %ymm14
	vshufpd	$5, %ymm3, %ymm3, %ymm10
	vmovapd	-32(%rcx), %ymm8
	vmovapd	-32(%rdi), %ymm7
	vperm2f128	$1, %ymm10, %ymm10, %ymm9
	vfmadd231pd	%ymm10, %ymm8, %ymm5
	vfmadd231pd	%ymm3, %ymm8, %ymm4
	vfmadd231pd	%ymm10, %ymm7, %ymm1
	vfmadd231pd	%ymm3, %ymm7, %ymm2
	vmovapd	-32(%rax), %ymm3
	vshufpd	$5, %ymm9, %ymm9, %ymm10
	vfmadd231pd	%ymm9, %ymm8, %ymm6
	vfmadd231pd	%ymm9, %ymm7, %ymm0
	vmovapd	(%rcx), %ymm9
	vfmadd132pd	%ymm10, %ymm15, %ymm8
	vfmadd132pd	%ymm10, %ymm14, %ymm7
	vmovapd	(%rdi), %ymm10
	cmpq	%r9, %rax
	jne	.L258
	movl	-104(%rbp), %eax
	addq	$1, %r10
	salq	$7, %r10
	addq	%r10, %r8
	addq	%r10, %r14
	addq	%r10, %r12
	leal	4(%r13,%rax,4), %r13d
.L257:
	leal	-1(%rdx), %eax
	cmpl	%r13d, %eax
	jle	.L259
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vfmadd231pd	%ymm3, %ymm10, %ymm2
	vmovapd	32(%r12), %ymm11
	vmovapd	%ymm10, %ymm13
	addl	$2, %r13d
	vperm2f128	$1, %ymm12, %ymm12, %ymm3
	vfmadd231pd	%ymm12, %ymm9, %ymm5
	vfmadd231pd	%ymm12, %ymm10, %ymm1
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm9, %ymm6
	vfmadd231pd	%ymm3, %ymm10, %ymm0
	vmovapd	64(%r12), %ymm3
	vfmadd132pd	%ymm12, %ymm8, %ymm9
	vfmadd132pd	%ymm12, %ymm7, %ymm13
	vshufpd	$5, %ymm11, %ymm11, %ymm12
	vmovapd	32(%r8), %ymm8
	vmovapd	32(%r14), %ymm7
	vperm2f128	$1, %ymm12, %ymm12, %ymm10
	vfmadd231pd	%ymm11, %ymm8, %ymm4
	vfmadd231pd	%ymm12, %ymm8, %ymm5
	vfmadd231pd	%ymm11, %ymm7, %ymm2
	vfmadd231pd	%ymm12, %ymm7, %ymm1
	vshufpd	$5, %ymm10, %ymm10, %ymm11
	vfmadd231pd	%ymm10, %ymm8, %ymm6
	vfmadd231pd	%ymm10, %ymm7, %ymm0
	vmovapd	64(%r14), %ymm10
	vfmadd132pd	%ymm11, %ymm9, %ymm8
	vmovapd	64(%r8), %ymm9
	vfmadd132pd	%ymm11, %ymm13, %ymm7
.L259:
	cmpl	%edx, %r13d
	jl	.L286
.L252:
	movl	-80(%rbp), %edx
	testl	%edx, %edx
	jle	.L260
	movl	-88(%rbp), %eax
	vmovapd	(%rsi), %ymm11
	vmovapd	(%r11), %ymm9
	sall	$2, %eax
	cmpl	$3, %edx
	cltq
	leaq	(%rsi,%rax,8), %rcx
	vmovapd	(%rcx), %ymm10
	jle	.L260
	subl	$4, %edx
	leaq	32(%rcx), %rax
	shrl	$2, %edx
	salq	$7, %rdx
	leaq	160(%rcx,%rdx), %rdx
	.p2align 4,,10
	.p2align 3
.L261:
	vshufpd	$5, %ymm9, %ymm9, %ymm3
	vfmadd231pd	%ymm9, %ymm11, %ymm4
	vfmadd231pd	%ymm9, %ymm10, %ymm2
	vmovapd	32(%r11), %ymm9
	subq	$-128, %rax
	subq	$-128, %rsi
	vmovapd	-96(%rax), %ymm15
	subq	$-128, %r11
	vperm2f128	$1, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm5
	vfmadd231pd	%ymm3, %ymm10, %ymm1
	vshufpd	$5, %ymm12, %ymm12, %ymm3
	vfmadd231pd	%ymm12, %ymm11, %ymm6
	vfmadd231pd	%ymm12, %ymm10, %ymm0
	vmovapd	-96(%rsi), %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm8
	vshufpd	$5, %ymm9, %ymm9, %ymm11
	vfmadd231pd	%ymm3, %ymm10, %ymm7
	vmovapd	-128(%rax), %ymm3
	vfmadd231pd	%ymm9, %ymm12, %ymm4
	vperm2f128	$1, %ymm11, %ymm11, %ymm13
	vfmadd231pd	%ymm9, %ymm3, %ymm2
	vmovapd	-64(%r11), %ymm9
	vfmadd231pd	%ymm11, %ymm12, %ymm5
	vfmadd231pd	%ymm11, %ymm3, %ymm1
	vshufpd	$5, %ymm9, %ymm9, %ymm11
	vshufpd	$5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm13, %ymm12, %ymm6
	vfmadd231pd	%ymm13, %ymm3, %ymm0
	vmovapd	-64(%rsi), %ymm13
	vfmadd231pd	%ymm11, %ymm15, %ymm1
	vfmadd231pd	%ymm14, %ymm12, %ymm8
	vperm2f128	$1, %ymm11, %ymm11, %ymm12
	vfmadd132pd	%ymm14, %ymm7, %ymm3
	vfmadd231pd	%ymm9, %ymm13, %ymm4
	vfmadd132pd	%ymm15, %ymm2, %ymm9
	vmovapd	-32(%r11), %ymm2
	vfmadd231pd	%ymm11, %ymm13, %ymm5
	vmovapd	(%rsi), %ymm11
	vshufpd	$5, %ymm12, %ymm12, %ymm7
	vfmadd231pd	%ymm12, %ymm13, %ymm6
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd132pd	%ymm7, %ymm8, %ymm13
	vfmadd132pd	%ymm7, %ymm3, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm7
	vmovapd	-32(%rsi), %ymm8
	vmovapd	-64(%rax), %ymm3
	vperm2f128	$1, %ymm7, %ymm7, %ymm10
	vfmadd231pd	%ymm7, %ymm8, %ymm5
	vfmadd231pd	%ymm2, %ymm8, %ymm4
	vfmadd231pd	%ymm7, %ymm3, %ymm1
	vfmadd132pd	%ymm3, %ymm9, %ymm2
	vmovapd	(%r11), %ymm9
	vshufpd	$5, %ymm10, %ymm10, %ymm7
	vfmadd231pd	%ymm10, %ymm8, %ymm6
	vfmadd231pd	%ymm10, %ymm3, %ymm0
	vmovapd	-32(%rax), %ymm10
	cmpq	%rax, %rdx
	vfmadd132pd	%ymm7, %ymm13, %ymm8
	vfmadd132pd	%ymm3, %ymm15, %ymm7
	jne	.L261
.L260:
	vblendpd	$10, %ymm6, %ymm8, %ymm9
	vblendpd	$10, %ymm5, %ymm4, %ymm3
	movl	-84(%rbp), %eax
	vblendpd	$5, %ymm6, %ymm8, %ymm6
	vblendpd	$5, %ymm5, %ymm4, %ymm4
	vblendpd	$10, %ymm0, %ymm7, %ymm8
	vblendpd	$12, %ymm9, %ymm3, %ymm5
	vblendpd	$5, %ymm0, %ymm7, %ymm0
	vblendpd	$3, %ymm9, %ymm3, %ymm3
	testl	%eax, %eax
	vblendpd	$12, %ymm6, %ymm4, %ymm9
	vblendpd	$3, %ymm6, %ymm4, %ymm4
	vblendpd	$10, %ymm1, %ymm2, %ymm6
	vblendpd	$5, %ymm1, %ymm2, %ymm2
	vblendpd	$3, %ymm8, %ymm6, %ymm7
	vblendpd	$12, %ymm8, %ymm6, %ymm1
	vblendpd	$12, %ymm0, %ymm2, %ymm6
	vblendpd	$3, %ymm0, %ymm2, %ymm2
	je	.L262
	movl	-100(%rbp), %eax
	movq	-96(%rbp), %rdx
	sall	$2, %eax
	vaddpd	(%rdx), %ymm5, %ymm5
	cltq
	leaq	(%rdx,%rax,8), %rax
	vaddpd	64(%rdx), %ymm3, %ymm3
	vaddpd	32(%rdx), %ymm9, %ymm9
	vaddpd	96(%rdx), %ymm4, %ymm4
	vaddpd	(%rax), %ymm1, %ymm1
	vaddpd	64(%rax), %ymm7, %ymm7
	vaddpd	32(%rax), %ymm6, %ymm6
	vaddpd	96(%rax), %ymm2, %ymm2
.L262:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovsd	.LC0(%rip), %xmm10
	vmovsd	%xmm5, %xmm8, %xmm0
	vucomisd	%xmm10, %xmm0
	jbe	.L263
	vmovsd	.LC1(%rip), %xmm8
	vsqrtsd	%xmm0, %xmm0, %xmm0
	movq	-72(%rbp), %rax
	vdivsd	%xmm0, %xmm8, %xmm0
	vmovlpd	%xmm0, (%r15)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.27604(%rip), %ymm8
	vmulpd	%ymm0, %ymm5, %ymm5
	vmulpd	%ymm0, %ymm1, %ymm1
	vmovapd	%ymm5, (%rbx)
	vmaskmovpd	%ymm1, %ymm8, (%rax)
.L264:
	vpermpd	$85, %ymm5, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm9
	vfmadd231pd	%ymm0, %ymm1, %ymm6
	vpermilpd	$3, %xmm9, %xmm0
	vucomisd	%xmm10, %xmm0
	jbe	.L265
	vmovsd	.LC1(%rip), %xmm8
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC2(%rip), %ymm11
	vdivsd	%xmm0, %xmm8, %xmm0
	vmovlpd	%xmm0, 8(%r15)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.27604(%rip), %ymm8
	vmulpd	%ymm0, %ymm9, %ymm9
	vmulpd	%ymm0, %ymm6, %ymm6
	vmaskmovpd	%ymm9, %ymm11, 32(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm6, %ymm8, 32(%rax)
.L266:
	vpermpd	$170, %ymm9, %ymm0
	vpermpd	$170, %ymm5, %ymm8
	vfmadd231pd	%ymm8, %ymm5, %ymm3
	vfmadd132pd	%ymm1, %ymm7, %ymm8
	vfmadd231pd	%ymm0, %ymm9, %ymm3
	vfmadd231pd	%ymm0, %ymm6, %ymm8
	vextractf128	$0x1, %ymm3, %xmm0
	vucomisd	%xmm10, %xmm0
	jbe	.L267
	vsqrtsd	%xmm0, %xmm0, %xmm7
	vmovsd	.LC1(%rip), %xmm0
	vmovdqa	.LC3(%rip), %ymm11
	vdivsd	%xmm7, %xmm0, %xmm0
	vmovlpd	%xmm0, 16(%r15)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.27604(%rip), %ymm7
	vmulpd	%ymm0, %ymm3, %ymm3
	vmulpd	%ymm0, %ymm8, %ymm8
	vmaskmovpd	%ymm3, %ymm11, 64(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm8, %ymm7, 64(%rax)
.L268:
	cmpl	$3, -76(%rbp)
	jle	.L283
	vpermpd	$255, %ymm5, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm4
	vpermpd	$255, %ymm9, %ymm5
	vfmadd231pd	%ymm0, %ymm1, %ymm2
	vfmadd231pd	%ymm5, %ymm9, %ymm4
	vfmadd132pd	%ymm5, %ymm2, %ymm6
	vpermpd	$255, %ymm3, %ymm5
	vfmadd231pd	%ymm5, %ymm3, %ymm4
	vextractf128	$0x1, %ymm4, %xmm0
	vfmadd231pd	%ymm5, %ymm8, %ymm6
	vpermilpd	$3, %xmm0, %xmm0
	vucomisd	%xmm10, %xmm0
	ja	.L287
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC4(%rip), %ymm2
	vmovdqu	mask_bkp.27604(%rip), %ymm1
	movq	$0, 24(%r15)
	vmaskmovpd	%ymm0, %ymm2, 96(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm0, %ymm1, 96(%rax)
.L283:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L267:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC3(%rip), %ymm11
	vmovdqu	mask_bkp.27604(%rip), %ymm7
	movq	$0, 16(%r15)
	vmaskmovpd	%ymm0, %ymm11, 64(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm0, %ymm7, 64(%rax)
	jmp	.L268
	.p2align 4,,10
	.p2align 3
.L265:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC2(%rip), %ymm11
	vmovdqu	mask_bkp.27604(%rip), %ymm8
	movq	$0, 8(%r15)
	vmaskmovpd	%ymm0, %ymm11, 32(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm0, %ymm8, 32(%rax)
	jmp	.L266
	.p2align 4,,10
	.p2align 3
.L263:
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	-72(%rbp), %rax
	vmovdqu	mask_bkp.27604(%rip), %ymm8
	movq	$0, (%r15)
	vmovapd	%ymm0, (%rbx)
	vmaskmovpd	%ymm0, %ymm8, (%rax)
	jmp	.L264
	.p2align 4,,10
	.p2align 3
.L250:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27605(%rip), %ymm1
	testl	%edx, %edx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC11(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.27604(%rip)
	jg	.L288
.L271:
	vxorpd	%xmm7, %xmm7, %xmm7
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, %ymm4
	jmp	.L252
	.p2align 4,,10
	.p2align 3
.L286:
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vfmadd231pd	%ymm3, %ymm10, %ymm2
	vperm2f128	$1, %ymm11, %ymm11, %ymm3
	vfmadd231pd	%ymm11, %ymm9, %ymm5
	vfmadd231pd	%ymm11, %ymm10, %ymm1
	vshufpd	$5, %ymm3, %ymm3, %ymm11
	vfmadd231pd	%ymm3, %ymm9, %ymm6
	vfmadd231pd	%ymm3, %ymm10, %ymm0
	vfmadd231pd	%ymm11, %ymm9, %ymm8
	vfmadd231pd	%ymm11, %ymm10, %ymm7
	jmp	.L252
	.p2align 4,,10
	.p2align 3
.L287:
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC4(%rip), %ymm2
	vdivsd	%xmm0, %xmm1, %xmm0
	vmovlpd	%xmm0, 24(%r15)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.27604(%rip), %ymm1
	vmulpd	%ymm4, %ymm0, %ymm4
	vmulpd	%ymm6, %ymm0, %ymm6
	vmaskmovpd	%ymm4, %ymm2, 96(%rbx)
	movq	-72(%rbp), %rax
	vmaskmovpd	%ymm6, %ymm1, 96(%rax)
	jmp	.L283
	.p2align 4,,10
	.p2align 3
.L285:
	cmpl	$3, %edx
	jle	.L254
	vxorpd	%xmm11, %xmm11, %xmm11
	vmovapd	32(%r12), %ymm0
	cmpl	$7, %edx
	vmovapd	96(%r12), %ymm1
	leaq	128(%r8), %rdi
	leaq	128(%r14), %rax
	vmovapd	128(%r14), %ymm10
	leaq	128(%r12), %rcx
	vblendpd	$1, %ymm9, %ymm11, %ymm4
	vblendpd	$3, 32(%r8), %ymm11, %ymm5
	vmovapd	128(%r8), %ymm9
	vblendpd	$7, 64(%r8), %ymm11, %ymm8
	vfmadd132pd	%ymm3, %ymm11, %ymm4
	vfmadd231pd	%ymm0, %ymm5, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm8, %ymm6
	vmovapd	128(%r12), %ymm3
	vfmadd132pd	%ymm0, %ymm11, %ymm5
	vblendpd	$7, 64(%r12), %ymm11, %ymm0
	vfmadd231pd	%ymm0, %ymm8, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm8, %ymm5
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm11, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm8, %ymm11, %ymm0
	vmovapd	96(%r8), %ymm8
	vfmadd231pd	%ymm1, %ymm8, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm8, %ymm5
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm8, %ymm6
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm8
	jle	.L255
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vmovapd	160(%r8), %ymm13
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vmovapd	160(%r12), %ymm0
	vblendpd	$1, %ymm10, %ymm11, %ymm10
	vmovapd	%ymm3, %ymm2
	vblendpd	$3, 160(%r14), %ymm11, %ymm14
	addq	$256, %r8
	addq	$256, %r14
	vperm2f128	$1, %ymm12, %ymm12, %ymm7
	vfmadd231pd	%ymm0, %ymm13, %ymm4
	vmovapd	%ymm4, %ymm15
	vshufpd	$5, %ymm0, %ymm0, %ymm4
	vfmadd132pd	%ymm10, %ymm11, %ymm2
	vfmadd231pd	%ymm12, %ymm9, %ymm5
	vfmadd132pd	%ymm10, %ymm11, %ymm12
	vfmadd231pd	%ymm0, %ymm14, %ymm2
	vmovapd	192(%r12), %ymm0
	vshufpd	$5, %ymm7, %ymm7, %ymm1
	vfmadd231pd	%ymm7, %ymm9, %ymm6
	vfmadd132pd	%ymm10, %ymm11, %ymm7
	vfmadd231pd	%ymm4, %ymm13, %ymm5
	vfmadd231pd	%ymm4, %ymm14, %ymm12
	addq	$256, %r12
	movl	$8, %r13d
	vfmadd132pd	%ymm1, %ymm11, %ymm10
	vfmadd231pd	%ymm1, %ymm9, %ymm8
	vperm2f128	$1, %ymm4, %ymm4, %ymm1
	vblendpd	$7, -64(%r14), %ymm11, %ymm11
	vmovapd	%ymm15, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm6
	vfmadd231pd	%ymm1, %ymm14, %ymm7
	vmovapd	-64(%r8), %ymm1
	vfmadd231pd	%ymm3, %ymm13, %ymm8
	vfmadd132pd	%ymm14, %ymm10, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm4
	vfmadd132pd	%ymm11, %ymm2, %ymm0
	vmovapd	-32(%r12), %ymm2
	vperm2f128	$1, %ymm10, %ymm10, %ymm9
	vfmadd231pd	%ymm10, %ymm1, %ymm5
	vfmadd231pd	%ymm10, %ymm11, %ymm12
	vshufpd	$5, %ymm9, %ymm9, %ymm10
	vfmadd231pd	%ymm9, %ymm1, %ymm6
	vfmadd132pd	%ymm11, %ymm7, %ymm9
	vmovapd	-32(%r14), %ymm7
	vfmadd132pd	%ymm10, %ymm8, %ymm1
	vmovapd	%ymm1, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm1
	vmovapd	-32(%r8), %ymm8
	vfmadd132pd	%ymm10, %ymm3, %ymm11
	vmovapd	(%r12), %ymm3
	vfmadd231pd	%ymm2, %ymm8, %ymm4
	vfmadd132pd	%ymm7, %ymm0, %ymm2
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vfmadd231pd	%ymm1, %ymm8, %ymm5
	vfmadd132pd	%ymm7, %ymm12, %ymm1
	vshufpd	$5, %ymm0, %ymm0, %ymm10
	vfmadd231pd	%ymm0, %ymm8, %ymm6
	vfmadd132pd	%ymm7, %ymm9, %ymm0
	vmovapd	(%r8), %ymm9
	vfmadd132pd	%ymm10, %ymm13, %ymm8
	vfmadd132pd	%ymm10, %ymm11, %ymm7
	vmovapd	(%r14), %ymm10
	jmp	.L253
	.p2align 4,,10
	.p2align 3
.L254:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	32(%r12), %ymm3
	cmpl	$1, %edx
	vblendpd	$1, %ymm9, %ymm0, %ymm4
	vmovapd	32(%r8), %ymm9
	vfmadd132pd	%ymm3, %ymm0, %ymm4
	je	.L276
	vshufpd	$5, %ymm3, %ymm3, %ymm5
	vblendpd	$3, %ymm9, %ymm0, %ymm9
	vmovapd	64(%r12), %ymm3
	cmpl	$3, %edx
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vfmadd132pd	%ymm9, %ymm0, %ymm5
	vmovapd	64(%r8), %ymm9
	jne	.L277
	vshufpd	$5, %ymm3, %ymm3, %ymm2
	vblendpd	$7, %ymm9, %ymm0, %ymm6
	vmovapd	96(%r12), %ymm3
	vmovapd	96(%r8), %ymm9
	vmovapd	%ymm0, %ymm7
	movl	$3, %r13d
	vfmadd231pd	%ymm3, %ymm6, %ymm4
	vperm2f128	$1, %ymm2, %ymm2, %ymm1
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm0, %ymm2
	vshufpd	$5, %ymm1, %ymm1, %ymm8
	vfmadd132pd	%ymm1, %ymm0, %ymm6
	vmovapd	%ymm0, %ymm1
	vfmadd132pd	%ymm9, %ymm0, %ymm8
	jmp	.L253
.L255:
	cmpl	$4, %edx
	je	.L273
	vshufpd	$5, %ymm3, %ymm3, %ymm1
	vblendpd	$1, %ymm10, %ymm11, %ymm10
	cmpl	$5, %edx
	vmovapd	%ymm3, %ymm2
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vmovapd	160(%r12), %ymm3
	vperm2f128	$1, %ymm1, %ymm1, %ymm0
	vfmadd231pd	%ymm1, %ymm9, %ymm5
	vfmadd132pd	%ymm10, %ymm11, %ymm2
	vfmadd132pd	%ymm10, %ymm11, %ymm1
	vshufpd	$5, %ymm0, %ymm0, %ymm7
	vfmadd231pd	%ymm0, %ymm9, %ymm6
	vfmadd132pd	%ymm10, %ymm11, %ymm0
	vfmadd231pd	%ymm7, %ymm9, %ymm8
	vfmadd132pd	%ymm10, %ymm11, %ymm7
	vmovapd	160(%r8), %ymm9
	vmovapd	160(%r14), %ymm10
	je	.L274
	vshufpd	$5, %ymm3, %ymm3, %ymm13
	vblendpd	$3, %ymm10, %ymm11, %ymm10
	cmpl	$7, %edx
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	vfmadd231pd	%ymm3, %ymm10, %ymm2
	vmovapd	192(%r12), %ymm3
	vperm2f128	$1, %ymm13, %ymm13, %ymm12
	vfmadd231pd	%ymm13, %ymm9, %ymm5
	vfmadd231pd	%ymm13, %ymm10, %ymm1
	vshufpd	$5, %ymm12, %ymm12, %ymm13
	vfmadd231pd	%ymm12, %ymm9, %ymm6
	vfmadd231pd	%ymm12, %ymm10, %ymm0
	vfmadd231pd	%ymm13, %ymm9, %ymm8
	vfmadd231pd	%ymm13, %ymm10, %ymm7
	vmovapd	192(%r8), %ymm9
	vmovapd	192(%r14), %ymm10
	jne	.L275
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vblendpd	$7, %ymm10, %ymm11, %ymm10
	vfmadd231pd	%ymm3, %ymm9, %ymm4
	movl	$7, %r13d
	vfmadd231pd	%ymm3, %ymm10, %ymm2
	vmovapd	224(%r12), %ymm3
	movq	%rcx, %r12
	vperm2f128	$1, %ymm12, %ymm12, %ymm11
	vfmadd231pd	%ymm12, %ymm9, %ymm5
	vfmadd231pd	%ymm12, %ymm10, %ymm1
	vshufpd	$5, %ymm11, %ymm11, %ymm12
	vfmadd231pd	%ymm11, %ymm9, %ymm6
	vfmadd231pd	%ymm11, %ymm10, %ymm0
	vfmadd231pd	%ymm12, %ymm9, %ymm8
	vfmadd231pd	%ymm12, %ymm10, %ymm7
	vmovapd	224(%r8), %ymm9
	vmovapd	224(%r14), %ymm10
	movq	%rdi, %r8
	movq	%rax, %r14
	jmp	.L253
.L277:
	vxorpd	%xmm7, %xmm7, %xmm7
	movl	$2, %r13d
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm6
	jmp	.L253
.L273:
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rax, %r14
	movq	%rcx, %r12
	movq	%rdi, %r8
	movl	$4, %r13d
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm2
	jmp	.L253
.L276:
	vxorpd	%xmm7, %xmm7, %xmm7
	movl	$1, %r13d
	vmovapd	%ymm7, %ymm0
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm2
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm5
	jmp	.L253
.L274:
	movq	%rax, %r14
	movq	%rcx, %r12
	movq	%rdi, %r8
	movl	$5, %r13d
	jmp	.L253
.L275:
	movq	%rax, %r14
	movq	%rcx, %r12
	movq	%rdi, %r8
	movl	$6, %r13d
	jmp	.L253
	.cfi_endproc
.LFE4595:
	.size	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_8x4_vs_lib4_new
	.section	.text.unlikely
.LCOLDE16:
	.text
.LHOTE16:
	.section	.text.unlikely
.LCOLDB17:
	.text
.LHOTB17:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_4x4_vs_lib4_new:
.LFB4596:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$3, %edi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	24(%r10), %r15d
	movl	(%r10), %ebx
	movq	8(%r10), %r13
	movq	16(%r10), %rax
	movq	32(%r10), %r12
	movq	40(%r10), %r11
	movl	%r15d, -68(%rbp)
	movq	48(%r10), %r10
	jle	.L290
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%edx, %edx
	vmovdqu	%ymm0, mask_bkp.27696(%rip)
	jle	.L310
.L325:
	cmpl	$1, %ecx
	vmovapd	(%r8), %ymm1
	vbroadcastf128	(%r9), %ymm11
	vbroadcastf128	16(%r9), %ymm12
	je	.L322
	vxorpd	%xmm4, %xmm4, %xmm4
	xorl	%r15d, %r15d
	vmovapd	%ymm4, %ymm0
	vmovapd	%ymm4, %ymm7
	vmovapd	%ymm4, %ymm6
.L293:
	leal	-3(%rdx), %ecx
	cmpl	%ecx, %r15d
	jge	.L314
	leal	-4(%rdx), %ecx
	vxorpd	%xmm8, %xmm8, %xmm8
	leaq	32(%r8), %rdi
	subl	%r15d, %ecx
	shrl	$2, %ecx
	vmovapd	%ymm8, %ymm3
	vmovapd	%ymm8, %ymm5
	movl	%ecx, -72(%rbp)
	movq	%rcx, -80(%rbp)
	vmovapd	%ymm8, %ymm2
	salq	$7, %rcx
	leaq	160(%r8,%rcx), %r14
	movq	%r9, %rcx
	.p2align 4,,10
	.p2align 3
.L297:
	vshufpd	$5, %ymm11, %ymm11, %ymm13
	vbroadcastf128	32(%rcx), %ymm10
	vfmadd231pd	%ymm11, %ymm1, %ymm6
	vshufpd	$5, %ymm12, %ymm12, %ymm11
	vmovapd	(%rdi), %ymm9
	vfmadd231pd	%ymm12, %ymm1, %ymm4
	vbroadcastf128	48(%rcx), %ymm12
	subq	$-128, %rdi
	vfmadd231pd	%ymm13, %ymm1, %ymm7
	vshufpd	$5, %ymm10, %ymm10, %ymm13
	vfmadd231pd	%ymm10, %ymm9, %ymm2
	vfmadd132pd	%ymm11, %ymm0, %ymm1
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vbroadcastf128	64(%rcx), %ymm11
	vmovapd	-96(%rdi), %ymm0
	vfmadd231pd	%ymm12, %ymm9, %ymm8
	vbroadcastf128	80(%rcx), %ymm12
	vfmadd231pd	%ymm13, %ymm9, %ymm5
	vshufpd	$5, %ymm11, %ymm11, %ymm13
	vfmadd132pd	%ymm9, %ymm3, %ymm10
	vfmadd231pd	%ymm11, %ymm0, %ymm6
	vmovapd	-64(%rdi), %ymm3
	vbroadcastf128	96(%rcx), %ymm9
	vshufpd	$5, %ymm12, %ymm12, %ymm11
	vfmadd231pd	%ymm12, %ymm0, %ymm4
	vbroadcastf128	112(%rcx), %ymm12
	vfmadd231pd	%ymm13, %ymm0, %ymm7
	subq	$-128, %rcx
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vfmadd231pd	%ymm9, %ymm3, %ymm2
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	vfmadd132pd	%ymm11, %ymm1, %ymm0
	vmovapd	-32(%rdi), %ymm1
	cmpq	%r14, %rdi
	vfmadd231pd	%ymm12, %ymm3, %ymm8
	vbroadcastf128	(%rcx), %ymm11
	vfmadd231pd	%ymm13, %ymm3, %ymm5
	vbroadcastf128	16(%rcx), %ymm12
	vfmadd132pd	%ymm9, %ymm10, %ymm3
	jne	.L297
	movq	-80(%rbp), %rcx
	movl	-72(%rbp), %edi
	addq	$1, %rcx
	leal	4(%r15,%rdi,4), %r15d
	salq	$7, %rcx
	addq	%rcx, %r9
	addq	%rcx, %r8
.L296:
	leal	-1(%rdx), %ecx
	cmpl	%r15d, %ecx
	jg	.L323
.L298:
	cmpl	%edx, %r15d
	jge	.L292
	vshufpd	$5, %ymm11, %ymm11, %ymm9
	vfmadd231pd	%ymm11, %ymm1, %ymm6
	vfmadd231pd	%ymm12, %ymm1, %ymm4
	vfmadd231pd	%ymm9, %ymm1, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	vfmadd231pd	%ymm9, %ymm1, %ymm0
.L292:
	testl	%ebx, %ebx
	jle	.L299
	cmpl	$3, %ebx
	vmovapd	0(%r13), %ymm1
	vbroadcastf128	(%rax), %ymm10
	vbroadcastf128	16(%rax), %ymm12
	jle	.L299
	subl	$4, %ebx
	leaq	32(%r13), %rdx
	shrl	$2, %ebx
	salq	$7, %rbx
	leaq	160(%r13,%rbx), %rcx
	.p2align 4,,10
	.p2align 3
.L300:
	vshufpd	$5, %ymm10, %ymm10, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm6
	vbroadcastf128	32(%rax), %ymm11
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vfmadd231pd	%ymm12, %ymm1, %ymm4
	vmovapd	(%rdx), %ymm9
	vbroadcastf128	64(%rax), %ymm12
	subq	$-128, %rdx
	vfmadd231pd	%ymm13, %ymm1, %ymm7
	vbroadcastf128	48(%rax), %ymm13
	vfmadd231pd	%ymm11, %ymm9, %ymm2
	vfmadd132pd	%ymm10, %ymm0, %ymm1
	vshufpd	$5, %ymm11, %ymm11, %ymm0
	vmovapd	-96(%rdx), %ymm10
	vshufpd	$5, %ymm13, %ymm13, %ymm11
	vfmadd231pd	%ymm13, %ymm9, %ymm8
	vbroadcastf128	80(%rax), %ymm13
	vfmadd231pd	%ymm12, %ymm10, %ymm6
	vfmadd231pd	%ymm0, %ymm9, %ymm5
	vshufpd	$5, %ymm12, %ymm12, %ymm0
	vbroadcastf128	112(%rax), %ymm12
	vfmadd132pd	%ymm9, %ymm3, %ymm11
	vmovapd	-64(%rdx), %ymm3
	vfmadd231pd	%ymm13, %ymm10, %ymm4
	vbroadcastf128	96(%rax), %ymm9
	subq	$-128, %rax
	vfmadd231pd	%ymm0, %ymm10, %ymm7
	vshufpd	$5, %ymm13, %ymm13, %ymm0
	vfmadd231pd	%ymm12, %ymm3, %ymm8
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vfmadd231pd	%ymm9, %ymm3, %ymm2
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	vbroadcastf128	16(%rax), %ymm12
	vfmadd132pd	%ymm10, %ymm1, %ymm0
	vmovapd	-32(%rdx), %ymm1
	cmpq	%rcx, %rdx
	vfmadd231pd	%ymm13, %ymm3, %ymm5
	vbroadcastf128	(%rax), %ymm10
	vfmadd132pd	%ymm9, %ymm11, %ymm3
	jne	.L300
.L299:
	vaddpd	%ymm2, %ymm6, %ymm2
	movl	-68(%rbp), %eax
	vaddpd	%ymm5, %ymm7, %ymm5
	testl	%eax, %eax
	vaddpd	%ymm3, %ymm0, %ymm0
	vaddpd	%ymm8, %ymm4, %ymm4
	vblendpd	$10, %ymm5, %ymm2, %ymm1
	vblendpd	$5, %ymm5, %ymm2, %ymm2
	vblendpd	$10, %ymm0, %ymm4, %ymm9
	vblendpd	$5, %ymm0, %ymm4, %ymm0
	je	.L301
	vaddpd	(%r12), %ymm1, %ymm1
	vaddpd	32(%r12), %ymm2, %ymm2
	vaddpd	64(%r12), %ymm9, %ymm9
	vaddpd	96(%r12), %ymm0, %ymm0
.L301:
	vxorpd	%xmm11, %xmm11, %xmm11
	vmovsd	%xmm1, %xmm11, %xmm10
	vmovsd	.LC0(%rip), %xmm11
	vucomisd	%xmm11, %xmm10
	jbe	.L302
	vmovsd	.LC1(%rip), %xmm3
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vdivsd	%xmm10, %xmm3, %xmm10
	vmovdqu	mask_bkp.27696(%rip), %ymm3
	vmovlpd	%xmm10, (%r10)
	vbroadcastsd	%xmm10, %ymm10
	vmulpd	%ymm10, %ymm1, %ymm1
	vmaskmovpd	%ymm1, %ymm3, (%r11)
.L303:
	vpermpd	$85, %ymm1, %ymm5
	vfmadd231pd	%ymm5, %ymm1, %ymm2
	vpermilpd	$3, %xmm2, %xmm5
	vucomisd	%xmm11, %xmm5
	jbe	.L304
	vmovsd	.LC1(%rip), %xmm4
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vmovdqu	mask_bkp.27696(%rip), %ymm3
	vdivsd	%xmm5, %xmm4, %xmm5
	vandpd	.LC12(%rip), %ymm3, %ymm3
	vmovlpd	%xmm5, 8(%r10)
	vbroadcastsd	%xmm5, %ymm5
	vmulpd	%ymm5, %ymm2, %ymm2
	vmaskmovpd	%ymm2, %ymm3, 32(%r11)
.L305:
	vpermpd	$170, %ymm1, %ymm3
	vfmadd132pd	%ymm1, %ymm9, %ymm3
	vpermpd	$170, %ymm2, %ymm9
	vfmadd132pd	%ymm2, %ymm3, %ymm9
	vextractf128	$0x1, %ymm9, %xmm3
	vucomisd	%xmm11, %xmm3
	jbe	.L306
	vmovsd	.LC1(%rip), %xmm4
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm4, %xmm3
	vmovdqu	mask_bkp.27696(%rip), %ymm4
	vandpd	.LC13(%rip), %ymm4, %ymm4
	vmovlpd	%xmm3, 16(%r10)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm3, %ymm9, %ymm9
	vmaskmovpd	%ymm9, %ymm4, 64(%r11)
.L307:
	cmpl	$3, %esi
	jle	.L320
	vpermpd	$255, %ymm1, %ymm3
	vpermpd	$255, %ymm2, %ymm5
	vfmadd231pd	%ymm3, %ymm1, %ymm0
	vpermpd	$255, %ymm9, %ymm3
	vfmadd132pd	%ymm5, %ymm0, %ymm2
	vfmadd132pd	%ymm3, %ymm2, %ymm9
	vextractf128	$0x1, %ymm9, %xmm3
	vpermilpd	$3, %xmm3, %xmm3
	vucomisd	%xmm11, %xmm3
	ja	.L324
	vmovdqu	mask_bkp.27696(%rip), %ymm0
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 24(%r10)
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm0, 96(%r11)
.L320:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L290:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27697(%rip), %ymm1
	testl	%edx, %edx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.27696(%rip)
	jg	.L325
.L310:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%ymm8, %ymm3
	vmovapd	%ymm8, %ymm5
	vmovapd	%ymm8, %ymm2
	vmovapd	%ymm8, %ymm4
	vmovapd	%ymm8, %ymm0
	vmovapd	%ymm8, %ymm7
	vmovapd	%ymm8, %ymm6
	jmp	.L292
	.p2align 4,,10
	.p2align 3
.L306:
	vmovdqu	mask_bkp.27696(%rip), %ymm3
	vxorpd	%xmm4, %xmm4, %xmm4
	movq	$0, 16(%r10)
	vandpd	.LC13(%rip), %ymm3, %ymm3
	vmaskmovpd	%ymm4, %ymm3, 64(%r11)
	jmp	.L307
	.p2align 4,,10
	.p2align 3
.L304:
	vmovdqu	mask_bkp.27696(%rip), %ymm3
	vxorpd	%xmm4, %xmm4, %xmm4
	movq	$0, 8(%r10)
	vandpd	.LC12(%rip), %ymm3, %ymm3
	vmaskmovpd	%ymm4, %ymm3, 32(%r11)
	jmp	.L305
	.p2align 4,,10
	.p2align 3
.L302:
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovdqu	mask_bkp.27696(%rip), %ymm3
	movq	$0, (%r10)
	vmaskmovpd	%ymm4, %ymm3, (%r11)
	jmp	.L303
	.p2align 4,,10
	.p2align 3
.L324:
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm0, %xmm3
	vmovdqu	mask_bkp.27696(%rip), %ymm0
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmovlpd	%xmm3, 24(%r10)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm9, %ymm3, %ymm9
	vmaskmovpd	%ymm9, %ymm0, 96(%r11)
	jmp	.L320
	.p2align 4,,10
	.p2align 3
.L322:
	cmpl	$3, %edx
	jle	.L294
	vxorpd	%xmm0, %xmm0, %xmm0
	vbroadcastf128	32(%r9), %ymm2
	subq	$-128, %r8
	vbroadcastf128	80(%r9), %ymm4
	movl	$4, %r15d
	vshufpd	$5, %ymm2, %ymm2, %ymm3
	vbroadcastf128	112(%r9), %ymm8
	vblendpd	$1, %ymm1, %ymm0, %ymm1
	vblendpd	$7, -64(%r8), %ymm0, %ymm5
	vbroadcastf128	144(%r9), %ymm12
	vmovapd	%ymm1, %ymm6
	vblendpd	$3, -96(%r8), %ymm0, %ymm1
	vfmadd132pd	%ymm11, %ymm0, %ymm6
	vfmadd231pd	%ymm2, %ymm1, %ymm6
	vbroadcastf128	64(%r9), %ymm2
	vfmadd132pd	%ymm1, %ymm0, %ymm3
	vbroadcastf128	128(%r9), %ymm11
	vblendpd	$7, %ymm2, %ymm0, %ymm1
	vshufpd	$5, %ymm4, %ymm4, %ymm2
	vfmadd132pd	%ymm5, %ymm0, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm7
	vfmadd132pd	%ymm5, %ymm6, %ymm1
	vbroadcastf128	96(%r9), %ymm6
	vfmadd231pd	%ymm2, %ymm5, %ymm0
	vmovapd	-32(%r8), %ymm2
	subq	$-128, %r9
	vfmadd231pd	%ymm7, %ymm5, %ymm3
	vshufpd	$5, %ymm6, %ymm6, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm6
	vshufpd	$5, %ymm8, %ymm8, %ymm1
	vfmadd231pd	%ymm8, %ymm2, %ymm4
	vfmadd132pd	%ymm2, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm2, %ymm0
	vmovapd	(%r8), %ymm1
	jmp	.L293
	.p2align 4,,10
	.p2align 3
.L323:
	vshufpd	$5, %ymm11, %ymm11, %ymm10
	vbroadcastf128	32(%r9), %ymm13
	vfmadd231pd	%ymm12, %ymm1, %ymm4
	vmovapd	32(%r8), %ymm9
	vfmadd231pd	%ymm11, %ymm1, %ymm6
	addl	$2, %r15d
	vbroadcastf128	64(%r9), %ymm11
	vfmadd231pd	%ymm10, %ymm1, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vfmadd231pd	%ymm13, %ymm9, %ymm2
	vshufpd	$5, %ymm13, %ymm13, %ymm12
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vbroadcastf128	48(%r9), %ymm10
	vfmadd231pd	%ymm12, %ymm9, %ymm5
	vmovapd	64(%r8), %ymm1
	vshufpd	$5, %ymm10, %ymm10, %ymm12
	vfmadd231pd	%ymm10, %ymm9, %ymm8
	vfmadd231pd	%ymm12, %ymm9, %ymm3
	vbroadcastf128	80(%r9), %ymm12
	jmp	.L298
	.p2align 4,,10
	.p2align 3
.L294:
	vxorpd	%xmm2, %xmm2, %xmm2
	cmpl	$1, %edx
	vblendpd	$1, %ymm1, %ymm2, %ymm1
	vmovapd	%ymm1, %ymm6
	vmovapd	32(%r8), %ymm1
	vfmadd132pd	%ymm11, %ymm2, %ymm6
	vbroadcastf128	32(%r9), %ymm11
	je	.L312
	vshufpd	$5, %ymm11, %ymm11, %ymm7
	vblendpd	$3, %ymm1, %ymm2, %ymm1
	cmpl	$3, %edx
	vbroadcastf128	80(%r9), %ymm12
	vfmadd231pd	%ymm11, %ymm1, %ymm6
	vbroadcastf128	64(%r9), %ymm11
	vfmadd132pd	%ymm1, %ymm2, %ymm7
	vmovapd	64(%r8), %ymm1
	je	.L326
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$2, %r15d
	vmovapd	%ymm4, %ymm0
	jmp	.L293
.L312:
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$1, %r15d
	vmovapd	%ymm4, %ymm0
	vmovapd	%ymm4, %ymm7
.L314:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%ymm8, %ymm3
	vmovapd	%ymm8, %ymm5
	vmovapd	%ymm8, %ymm2
	jmp	.L296
.L326:
	vshufpd	$5, %ymm11, %ymm11, %ymm0
	vblendpd	$7, %ymm1, %ymm2, %ymm1
	vmovapd	%ymm12, %ymm4
	movl	$3, %r15d
	vfmadd231pd	%ymm11, %ymm1, %ymm6
	vbroadcastf128	96(%r9), %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm0
	vfmadd132pd	%ymm1, %ymm2, %ymm4
	vbroadcastf128	112(%r9), %ymm12
	vfmadd132pd	%ymm1, %ymm2, %ymm0
	vmovapd	96(%r8), %ymm1
	jmp	.L314
	.cfi_endproc
.LFE4596:
	.size	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_4x4_vs_lib4_new
	.section	.text.unlikely
.LCOLDE17:
	.text
.LHOTE17:
	.section	.text.unlikely
.LCOLDB18:
	.text
.LHOTB18:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_4x2_vs_lib4_new:
.LFB4597:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$3, %edi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	32(%r10), %r15
	movl	(%r10), %ebx
	movq	8(%r10), %r11
	movq	16(%r10), %rax
	movl	24(%r10), %r13d
	movq	40(%r10), %r12
	movq	%r15, -72(%rbp)
	movq	48(%r10), %r10
	jle	.L328
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%edx, %edx
	vmovdqu	%ymm0, mask_bkp.27774(%rip)
	jle	.L344
.L359:
	cmpl	$1, %ecx
	vmovapd	(%r8), %ymm10
	vbroadcastf128	(%r9), %ymm0
	je	.L356
	vxorpd	%xmm9, %xmm9, %xmm9
	xorl	%r15d, %r15d
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm9, %ymm6
	vmovapd	%ymm9, %ymm7
.L331:
	leal	-3(%rdx), %ecx
	cmpl	%ecx, %r15d
	jge	.L348
	leal	-4(%rdx), %ecx
	vxorpd	%xmm5, %xmm5, %xmm5
	leaq	32(%r8), %rdi
	subl	%r15d, %ecx
	shrl	$2, %ecx
	vmovapd	%ymm5, %ymm4
	vmovapd	%ymm5, %ymm3
	movl	%ecx, -76(%rbp)
	movq	%rcx, -88(%rbp)
	vmovapd	%ymm5, %ymm2
	salq	$7, %rcx
	leaq	160(%r8,%rcx), %r14
	movq	%r9, %rcx
	.p2align 4,,10
	.p2align 3
.L335:
	vfmadd231pd	%ymm0, %ymm10, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%rcx), %ymm1
	subq	$-128, %rdi
	vfmadd231pd	%ymm0, %ymm10, %ymm6
	vmovapd	-128(%rdi), %ymm10
	vbroadcastf128	64(%rcx), %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vmovapd	-96(%rdi), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vbroadcastf128	96(%rcx), %ymm0
	subq	$-128, %rcx
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vmovapd	-64(%rdi), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vbroadcastf128	(%rcx), %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm5
	vmovapd	-32(%rdi), %ymm10
	cmpq	%r14, %rdi
	jne	.L335
	movq	-88(%rbp), %rcx
	movl	-76(%rbp), %edi
	addq	$1, %rcx
	leal	4(%r15,%rdi,4), %r15d
	salq	$7, %rcx
	addq	%rcx, %r9
	addq	%rcx, %r8
.L334:
	leal	-1(%rdx), %ecx
	cmpl	%r15d, %ecx
	jg	.L357
.L336:
	cmpl	%edx, %r15d
	jge	.L330
	vfmadd231pd	%ymm0, %ymm10, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm10, %ymm6
.L330:
	testl	%ebx, %ebx
	jle	.L337
	cmpl	$3, %ebx
	vmovapd	(%r11), %ymm10
	vbroadcastf128	(%rax), %ymm0
	jle	.L337
	leal	-4(%rbx), %edx
	shrl	$2, %edx
	addq	$1, %rdx
	salq	$7, %rdx
	addq	%rax, %rdx
	.p2align 4,,10
	.p2align 3
.L338:
	vfmadd231pd	%ymm0, %ymm10, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%rax), %ymm1
	subq	$-128, %r11
	vfmadd231pd	%ymm0, %ymm10, %ymm6
	vmovapd	-96(%r11), %ymm10
	vbroadcastf128	64(%rax), %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vmovapd	-64(%r11), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vbroadcastf128	96(%rax), %ymm0
	subq	$-128, %rax
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vmovapd	-32(%r11), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm10, %ymm4
	vbroadcastf128	(%rax), %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm5
	vmovapd	(%r11), %ymm10
	cmpq	%rdx, %rax
	jne	.L338
.L337:
	vaddpd	%ymm2, %ymm7, %ymm2
	testl	%r13d, %r13d
	vaddpd	%ymm4, %ymm8, %ymm4
	vaddpd	%ymm3, %ymm6, %ymm3
	vaddpd	%ymm5, %ymm9, %ymm5
	vaddpd	%ymm4, %ymm2, %ymm2
	vaddpd	%ymm5, %ymm3, %ymm3
	vblendpd	$10, %ymm3, %ymm2, %ymm0
	vblendpd	$5, %ymm3, %ymm2, %ymm2
	je	.L339
	movq	-72(%rbp), %rax
	vaddpd	(%rax), %ymm0, %ymm0
	vaddpd	32(%rax), %ymm2, %ymm2
.L339:
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	.LC0(%rip), %xmm4
	vmovsd	%xmm0, %xmm3, %xmm1
	vucomisd	%xmm4, %xmm1
	jbe	.L340
	vmovsd	.LC1(%rip), %xmm3
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm3, %xmm1
	vmovdqu	mask_bkp.27774(%rip), %ymm3
	vmovlpd	%xmm1, (%r10)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm1, %ymm0, %ymm0
	vmaskmovpd	%ymm0, %ymm3, (%r12)
.L341:
	cmpl	$1, %esi
	jle	.L354
	vpermpd	$85, %ymm0, %ymm1
	vfmadd231pd	%ymm1, %ymm0, %ymm2
	vpermilpd	$3, %xmm2, %xmm1
	vucomisd	%xmm4, %xmm1
	ja	.L358
	vmovdqu	mask_bkp.27774(%rip), %ymm0
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 8(%r10)
	vandpd	.LC12(%rip), %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm0, 32(%r12)
.L354:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L328:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27775(%rip), %ymm1
	testl	%edx, %edx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.27774(%rip)
	jg	.L359
.L344:
	vxorpd	%xmm5, %xmm5, %xmm5
	vmovapd	%ymm5, %ymm4
	vmovapd	%ymm5, %ymm3
	vmovapd	%ymm5, %ymm2
	vmovapd	%ymm5, %ymm9
	vmovapd	%ymm5, %ymm8
	vmovapd	%ymm5, %ymm6
	vmovapd	%ymm5, %ymm7
	jmp	.L330
	.p2align 4,,10
	.p2align 3
.L340:
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovdqu	mask_bkp.27774(%rip), %ymm1
	movq	$0, (%r10)
	vmaskmovpd	%ymm3, %ymm1, (%r12)
	jmp	.L341
	.p2align 4,,10
	.p2align 3
.L358:
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm0, %xmm1
	vmovdqu	mask_bkp.27774(%rip), %ymm0
	vandpd	.LC12(%rip), %ymm0, %ymm0
	vmovlpd	%xmm1, 8(%r10)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm2, %ymm1, %ymm2
	vmaskmovpd	%ymm2, %ymm0, 32(%r12)
	jmp	.L354
	.p2align 4,,10
	.p2align 3
.L356:
	cmpl	$3, %edx
	jle	.L332
	vxorpd	%xmm2, %xmm2, %xmm2
	vbroadcastf128	64(%r9), %ymm1
	subq	$-128, %r8
	vbroadcastf128	32(%r9), %ymm3
	movl	$4, %r15d
	vmovapd	-32(%r8), %ymm9
	vblendpd	$1, %ymm10, %ymm2, %ymm10
	vblendpd	$7, -64(%r8), %ymm2, %ymm6
	vshufpd	$5, %ymm3, %ymm3, %ymm4
	vblendpd	$3, -96(%r8), %ymm2, %ymm5
	vmulpd	%ymm10, %ymm0, %ymm0
	vmovapd	(%r8), %ymm10
	vmulpd	%ymm6, %ymm1, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vaddpd	%ymm2, %ymm0, %ymm0
	vmulpd	%ymm5, %ymm3, %ymm3
	vmulpd	%ymm6, %ymm1, %ymm6
	vmulpd	%ymm5, %ymm4, %ymm4
	vaddpd	%ymm7, %ymm0, %ymm7
	vbroadcastf128	96(%r9), %ymm0
	subq	$-128, %r9
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vmulpd	%ymm9, %ymm0, %ymm8
	vaddpd	%ymm2, %ymm3, %ymm3
	vbroadcastf128	(%r9), %ymm0
	vaddpd	%ymm2, %ymm6, %ymm6
	vmulpd	%ymm9, %ymm1, %ymm1
	vaddpd	%ymm2, %ymm4, %ymm2
	vaddpd	%ymm8, %ymm3, %ymm8
	vaddpd	%ymm1, %ymm2, %ymm9
	jmp	.L331
	.p2align 4,,10
	.p2align 3
.L357:
	vfmadd231pd	%ymm0, %ymm10, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%r9), %ymm1
	addl	$2, %r15d
	vfmadd231pd	%ymm0, %ymm10, %ymm6
	vmovapd	32(%r8), %ymm10
	vbroadcastf128	64(%r9), %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vmovapd	64(%r8), %ymm10
	jmp	.L336
	.p2align 4,,10
	.p2align 3
.L332:
	vxorpd	%xmm1, %xmm1, %xmm1
	cmpl	$1, %edx
	vblendpd	$1, %ymm10, %ymm1, %ymm7
	vmovapd	32(%r8), %ymm10
	vmulpd	%ymm7, %ymm0, %ymm0
	vaddpd	%ymm1, %ymm0, %ymm7
	vbroadcastf128	32(%r9), %ymm0
	je	.L346
	vshufpd	$5, %ymm0, %ymm0, %ymm2
	vblendpd	$3, %ymm10, %ymm1, %ymm9
	cmpl	$3, %edx
	vmovapd	64(%r8), %ymm10
	vmovapd	%ymm1, %ymm6
	movl	$2, %r15d
	vmulpd	%ymm0, %ymm9, %ymm8
	vbroadcastf128	64(%r9), %ymm0
	vmulpd	%ymm9, %ymm2, %ymm9
	vaddpd	%ymm1, %ymm8, %ymm8
	vaddpd	%ymm1, %ymm9, %ymm9
	jne	.L331
	vblendpd	$7, %ymm10, %ymm1, %ymm2
	vmovapd	96(%r8), %ymm10
	movl	$3, %r15d
	vmulpd	%ymm0, %ymm2, %ymm1
	vaddpd	%ymm1, %ymm7, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vbroadcastf128	96(%r9), %ymm0
	vmulpd	%ymm2, %ymm1, %ymm1
	vaddpd	%ymm6, %ymm1, %ymm6
.L348:
	vxorpd	%xmm5, %xmm5, %xmm5
	vmovapd	%ymm5, %ymm4
	vmovapd	%ymm5, %ymm3
	vmovapd	%ymm5, %ymm2
	jmp	.L334
.L346:
	vxorpd	%xmm9, %xmm9, %xmm9
	movl	$1, %r15d
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm9, %ymm6
	jmp	.L348
	.cfi_endproc
.LFE4597:
	.size	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_4x2_vs_lib4_new
	.section	.text.unlikely
.LCOLDE18:
	.text
.LHOTE18:
	.section	.text.unlikely
.LCOLDB20:
	.text
.LHOTB20:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4_new
	.type	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4_new, @function
kernel_dsyrk_dpotrf_nt_2x2_vs_lib4_new:
.LFB4598:
	.cfi_startproc
	testl	%edx, %edx
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	56(%rsp), %r11d
	movq	64(%rsp), %r10
	movq	72(%rsp), %r15
	movq	96(%rsp), %r14
	jle	.L377
	cmpl	$1, %ecx
	vmovapd	(%r8), %xmm3
	vmovapd	(%r9), %xmm0
	je	.L389
	vxorpd	%xmm5, %xmm5, %xmm5
	xorl	%ebp, %ebp
	vmovapd	%xmm5, %xmm6
	vmovapd	%xmm5, %xmm1
.L362:
	leal	-3(%rdx), %eax
	cmpl	%eax, %ebp
	jge	.L390
	leal	-4(%rdx), %r13d
	vxorpd	%xmm8, %xmm8, %xmm8
	leaq	32(%r9), %rax
	subl	%ebp, %r13d
	shrl	$2, %r13d
	vmovapd	%xmm8, %xmm10
	movl	%r13d, %r12d
	vmovapd	%xmm8, %xmm9
	movq	%r12, %rcx
	vmovapd	%xmm8, %xmm11
	salq	$7, %rcx
	vmovapd	%xmm8, %xmm4
	leaq	160(%r9,%rcx), %rbx
	movq	%r8, %rcx
	.p2align 4,,10
	.p2align 3
.L366:
	vfmadd231pd	%xmm0, %xmm3, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	(%rax), %xmm2
	subq	$-128, %rax
	subq	$-128, %rcx
	vfmadd231pd	%xmm0, %xmm3, %xmm4
	vmovapd	-96(%rcx), %xmm3
	vfmadd231pd	%xmm2, %xmm3, %xmm6
	vshufpd	$1, %xmm2, %xmm2, %xmm2
	vmovapd	-96(%rax), %xmm0
	vfmadd231pd	%xmm2, %xmm3, %xmm5
	vmovapd	-64(%rcx), %xmm3
	vshufpd	$1, %xmm0, %xmm0, %xmm2
	vfmadd231pd	%xmm0, %xmm3, %xmm11
	vmovapd	-64(%rax), %xmm0
	vfmadd231pd	%xmm2, %xmm3, %xmm9
	vmovapd	-32(%rcx), %xmm3
	vshufpd	$1, %xmm0, %xmm0, %xmm2
	vfmadd231pd	%xmm0, %xmm3, %xmm10
	vmovapd	-32(%rax), %xmm0
	vfmadd231pd	%xmm2, %xmm3, %xmm8
	vmovapd	(%rcx), %xmm3
	cmpq	%rax, %rbx
	jne	.L366
	addq	$1, %r12
	leal	4(%rbp,%r13,4), %ebp
	salq	$7, %r12
	addq	%r12, %r8
	addq	%r12, %r9
.L365:
	leal	-1(%rdx), %eax
	cmpl	%ebp, %eax
	jle	.L367
	vfmadd231pd	%xmm0, %xmm3, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	32(%r9), %xmm2
	addl	$2, %ebp
	vfmadd231pd	%xmm0, %xmm3, %xmm4
	vmovapd	32(%r8), %xmm3
	vfmadd231pd	%xmm2, %xmm3, %xmm6
	vshufpd	$1, %xmm2, %xmm2, %xmm2
	vmovapd	64(%r9), %xmm0
	vfmadd231pd	%xmm2, %xmm3, %xmm5
	vmovapd	64(%r8), %xmm3
.L367:
	cmpl	%edx, %ebp
	jge	.L361
	vfmadd231pd	%xmm0, %xmm3, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vfmadd231pd	%xmm0, %xmm3, %xmm4
.L361:
	testl	%r11d, %r11d
	jle	.L368
	cmpl	$3, %r11d
	vmovapd	(%r10), %xmm3
	vmovapd	(%r15), %xmm2
	jle	.L368
	subl	$4, %r11d
	leaq	32(%r15), %rax
	shrl	$2, %r11d
	salq	$7, %r11
	leaq	160(%r15,%r11), %rdx
	.p2align 4,,10
	.p2align 3
.L369:
	vfmadd231pd	%xmm2, %xmm3, %xmm1
	vshufpd	$1, %xmm2, %xmm2, %xmm2
	vmovapd	(%rax), %xmm0
	subq	$-128, %rax
	subq	$-128, %r10
	vfmadd231pd	%xmm2, %xmm3, %xmm4
	vmovapd	-96(%r10), %xmm3
	vfmadd231pd	%xmm0, %xmm3, %xmm6
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	-96(%rax), %xmm2
	vfmadd231pd	%xmm0, %xmm3, %xmm5
	vmovapd	-64(%r10), %xmm0
	vshufpd	$1, %xmm2, %xmm2, %xmm3
	vfmadd231pd	%xmm2, %xmm0, %xmm11
	vmovapd	-64(%rax), %xmm2
	vfmadd231pd	%xmm3, %xmm0, %xmm9
	vmovapd	-32(%r10), %xmm0
	vshufpd	$1, %xmm2, %xmm2, %xmm3
	vfmadd231pd	%xmm2, %xmm0, %xmm10
	vmovapd	-32(%rax), %xmm2
	vfmadd231pd	%xmm3, %xmm0, %xmm8
	vmovapd	(%r10), %xmm3
	cmpq	%rax, %rdx
	jne	.L369
.L368:
	vaddpd	%xmm11, %xmm1, %xmm0
	movl	80(%rsp), %eax
	vaddpd	%xmm10, %xmm6, %xmm1
	testl	%eax, %eax
	vaddpd	%xmm9, %xmm4, %xmm4
	vaddpd	%xmm8, %xmm5, %xmm5
	vaddpd	%xmm1, %xmm0, %xmm1
	vaddpd	%xmm5, %xmm4, %xmm4
	vblendpd	$2, %xmm4, %xmm1, %xmm2
	vmovsd	%xmm4, %xmm1, %xmm1
	je	.L370
	movq	88(%rsp), %rax
	vaddpd	(%rax), %xmm2, %xmm2
	vaddpd	32(%rax), %xmm1, %xmm1
.L370:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	.LC0(%rip), %xmm3
	vmovsd	%xmm2, %xmm0, %xmm0
	vucomisd	%xmm3, %xmm0
	jbe	.L371
	vmovsd	.LC1(%rip), %xmm7
	vsqrtsd	%xmm0, %xmm0, %xmm0
	movq	104(%rsp), %rax
	vmovlpd	%xmm0, (%r14)
	vshufpd	$1, %xmm3, %xmm2, %xmm2
	vdivsd	%xmm0, %xmm7, %xmm0
	cmpl	$1, %edi
	vmovlpd	%xmm0, (%rax)
	vmulsd	%xmm0, %xmm2, %xmm7
	je	.L387
	cmpl	$1, %esi
	vmovlpd	%xmm7, 8(%r14)
	jg	.L391
.L387:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L371:
	.cfi_restore_state
	movq	104(%rsp), %rax
	cmpl	$1, %edi
	movq	$0, (%r14)
	movq	$0, (%rax)
	je	.L387
	cmpl	$1, %esi
	movq	$0, 8(%r14)
	jle	.L387
.L391:
	vmulsd	%xmm7, %xmm7, %xmm7
	vshufpd	$1, %xmm3, %xmm1, %xmm1
	vsubsd	%xmm7, %xmm1, %xmm1
	vucomisd	%xmm3, %xmm1
	ja	.L392
	movq	104(%rsp), %rax
	movq	$0, 40(%r14)
	movq	$0, 8(%rax)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L377:
	.cfi_restore_state
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%xmm8, %xmm10
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm5
	vmovapd	%xmm8, %xmm6
	vmovapd	%xmm8, %xmm4
	vmovapd	%xmm8, %xmm1
	jmp	.L361
	.p2align 4,,10
	.p2align 3
.L392:
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm1, %xmm1, %xmm1
	movq	104(%rsp), %rax
	vmovlpd	%xmm1, 40(%r14)
	vdivsd	%xmm1, %xmm0, %xmm1
	vmovlpd	%xmm1, 8(%rax)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L389:
	.cfi_restore_state
	cmpl	$1, %edx
	vmulsd	%xmm0, %xmm3, %xmm0
	je	.L363
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovapd	32(%r8), %xmm5
	addq	$64, %r9
	addq	$64, %r8
	movl	$2, %ebp
	vmovapd	(%r8), %xmm3
	vaddsd	%xmm0, %xmm4, %xmm1
	vmovapd	-32(%r9), %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm2
	vmulpd	%xmm5, %xmm0, %xmm6
	vmovapd	(%r9), %xmm0
	vmulpd	%xmm2, %xmm5, %xmm5
	vaddpd	%xmm4, %xmm6, %xmm6
	vaddpd	%xmm4, %xmm5, %xmm5
	jmp	.L362
.L363:
	vxorpd	%xmm8, %xmm8, %xmm8
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	%xmm8, %xmm10
	vaddsd	%xmm0, %xmm1, %xmm1
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm5
	vmovapd	%xmm8, %xmm6
	vmovapd	%xmm8, %xmm4
	jmp	.L361
.L390:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%xmm8, %xmm10
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm4
	jmp	.L365
	.cfi_endproc
.LFE4598:
	.size	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4_new, .-kernel_dsyrk_dpotrf_nt_2x2_vs_lib4_new
	.section	.text.unlikely
.LCOLDE20:
	.text
.LHOTE20:
	.section	.text.unlikely
.LCOLDB21:
	.text
.LHOTB21:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_12x4_vs_lib4:
.LFB4599:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	subq	$72, %rsp
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	24(%r10), %ebx
	movl	64(%r10), %r13d
	movq	56(%r10), %r12
	movl	80(%r10), %r15d
	movl	%r8d, -76(%rbp)
	movq	8(%r10), %rax
	movl	%ebx, -180(%rbp)
	movq	40(%r10), %rbx
	movl	(%r10), %r8d
	movq	16(%r10), %rsi
	movq	32(%r10), %r11
	movl	%r15d, -80(%rbp)
	movq	%rbx, -192(%rbp)
	movl	48(%r10), %ebx
	movl	%ebx, -184(%rbp)
	movq	72(%r10), %rbx
	leal	0(,%r13,4), %r10d
	movslq	%r10d, %r10
	salq	$3, %r10
	cmpl	$11, %edi
	leaq	(%r12,%r10), %r13
	leaq	0(%r13,%r10), %r14
	movq	%r14, -72(%rbp)
	jle	.L394
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%ecx, %ecx
	vpcmpeqd	%ymm7, %ymm7, %ymm7
	vmovdqu	%ymm0, mask_bkp.27907(%rip)
	vmovdqa	%ymm7, -240(%rbp)
	jle	.L416
.L437:
	sall	$2, %r8d
	vmovapd	(%r9), %ymm6
	movslq	%r8d, %r8
	vmovapd	(%rax), %ymm2
	salq	$3, %r8
	cmpl	$1, %edx
	leaq	(%r9,%r8), %r14
	leaq	(%r14,%r8), %rdi
	vmovapd	(%r14), %ymm3
	movq	%rdi, -176(%rbp)
	vmovapd	(%rdi), %ymm0
	je	.L434
	vxorpd	%xmm7, %xmm7, %xmm7
	xorl	%r15d, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm14
	vmovapd	%ymm7, %ymm9
	vmovapd	%ymm7, %ymm12
	vmovapd	%ymm7, %ymm15
.L397:
	leal	-3(%rcx), %edx
	movq	%rax, %r10
	cmpl	%edx, %r15d
	jge	.L402
	leal	-4(%rcx), %edx
	vmovapd	%ymm7, %ymm1
	movq	-176(%rbp), %r8
	vmovapd	%ymm8, %ymm7
	vmovapd	%ymm6, %ymm4
	vmovapd	-144(%rbp), %ymm6
	subl	%r15d, %edx
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm10, %ymm9
	shrl	$2, %edx
	vmovapd	%ymm11, %ymm10
	vmovapd	%ymm12, %ymm11
	movl	%edx, -204(%rbp)
	addq	$1, %rdx
	vmovapd	%ymm13, %ymm12
	salq	$7, %rdx
	vmovapd	%ymm14, %ymm13
	vmovapd	%ymm15, %ymm14
	vmovapd	%ymm5, %ymm15
	vmovapd	%ymm1, %ymm5
	movq	%rdx, -200(%rbp)
	leaq	(%rax,%rdx), %r10
	movq	%r14, %rdi
	movq	%r9, %rdx
	.p2align 4,,10
	.p2align 3
.L403:
	vshufpd	$5, %ymm2, %ymm2, %ymm1
	vfmadd231pd	%ymm2, %ymm4, %ymm14
	vfmadd231pd	%ymm2, %ymm3, %ymm13
	vfmadd231pd	%ymm2, %ymm0, %ymm12
	vmovapd	32(%r8), %ymm2
	subq	$-128, %rax
	subq	$-128, %rdx
	subq	$-128, %rdi
	subq	$-128, %r8
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm3, %ymm10
	vfmadd231pd	%ymm1, %ymm0, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm4, %ymm8
	vfmadd231pd	%ymm1, %ymm3, %ymm7
	vfmadd231pd	%ymm1, %ymm0, %ymm5
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd213pd	-112(%rbp), %ymm1, %ymm0
	vfmadd132pd	%ymm1, %ymm15, %ymm4
	vfmadd132pd	%ymm1, %ymm6, %ymm3
	vmovapd	-96(%rdx), %ymm15
	vmovapd	-96(%rdi), %ymm6
	vmovapd	-96(%rax), %ymm1
	vfmadd231pd	%ymm1, %ymm15, %ymm14
	vfmadd231pd	%ymm1, %ymm6, %ymm13
	vfmadd231pd	%ymm1, %ymm2, %ymm12
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm15, %ymm11
	vfmadd231pd	%ymm1, %ymm6, %ymm10
	vfmadd231pd	%ymm1, %ymm2, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm15, %ymm8
	vfmadd231pd	%ymm1, %ymm6, %ymm7
	vfmadd231pd	%ymm1, %ymm2, %ymm5
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm3
	vfmadd231pd	%ymm1, %ymm15, %ymm4
	vmovapd	-64(%rdi), %ymm6
	vmovapd	-64(%rdx), %ymm15
	vfmadd132pd	%ymm2, %ymm0, %ymm1
	vmovapd	-64(%r8), %ymm0
	vmovapd	-64(%rax), %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm14
	vfmadd231pd	%ymm2, %ymm6, %ymm13
	vfmadd231pd	%ymm2, %ymm0, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm6, %ymm10
	vfmadd231pd	%ymm2, %ymm0, %ymm9
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm8
	vfmadd231pd	%ymm2, %ymm6, %ymm7
	vfmadd231pd	%ymm2, %ymm0, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd132pd	%ymm2, %ymm3, %ymm6
	vmovapd	-32(%rdx), %ymm15
	vmovapd	-32(%rdi), %ymm3
	vfmadd132pd	%ymm0, %ymm1, %ymm2
	vmovapd	-32(%rax), %ymm0
	vmovapd	-32(%r8), %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm14
	vfmadd231pd	%ymm0, %ymm3, %ymm13
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd231pd	%ymm0, %ymm3, %ymm10
	vfmadd231pd	%ymm0, %ymm1, %ymm9
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm15, %ymm8
	vfmadd231pd	%ymm0, %ymm3, %ymm7
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm2, %ymm1
	vmovapd	(%rax), %ymm2
	vfmadd132pd	%ymm0, %ymm4, %ymm15
	vfmadd231pd	%ymm0, %ymm3, %ymm6
	vmovapd	(%rdx), %ymm4
	vmovapd	(%rdi), %ymm3
	vmovapd	(%r8), %ymm0
	cmpq	%r10, %rax
	vmovapd	%ymm1, -112(%rbp)
	jne	.L403
	movq	-200(%rbp), %rax
	addq	%rax, -176(%rbp)
	vmovapd	%ymm5, %ymm1
	vmovapd	%ymm6, -144(%rbp)
	vmovapd	%ymm15, %ymm5
	vmovapd	%ymm4, %ymm6
	vmovapd	%ymm14, %ymm15
	vmovapd	%ymm13, %ymm14
	addq	%rax, %r9
	addq	%rax, %r14
	vmovapd	%ymm12, %ymm13
	movl	-204(%rbp), %eax
	vmovapd	%ymm11, %ymm12
	vmovapd	%ymm10, %ymm11
	vmovapd	%ymm9, %ymm10
	vmovapd	%ymm8, %ymm9
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm1, %ymm7
	leal	4(%r15,%rax,4), %r15d
.L402:
	leal	-1(%rcx), %eax
	cmpl	%r15d, %eax
	jle	.L404
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vfmadd231pd	%ymm2, %ymm0, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	movq	-176(%rbp), %rax
	vmovapd	-144(%rbp), %ymm4
	addl	$2, %r15d
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vmovapd	32(%rax), %ymm1
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd213pd	-112(%rbp), %ymm2, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vfmadd231pd	%ymm2, %ymm3, %ymm4
	vmovapd	32(%r9), %ymm6
	vmovapd	32(%r14), %ymm3
	vmovapd	32(%r10), %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vfmadd231pd	%ymm2, %ymm1, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm1, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm1, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm4, %ymm3
	vfmadd132pd	%ymm2, %ymm0, %ymm1
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	64(%r9), %ymm6
	vmovapd	64(%r14), %ymm3
	vmovapd	%ymm1, -112(%rbp)
	vmovapd	64(%r10), %ymm2
	vmovapd	64(%rax), %ymm0
.L404:
	cmpl	%ecx, %r15d
	jl	.L435
.L396:
	movl	-76(%rbp), %edi
	testl	%edi, %edi
	jle	.L405
	movl	-180(%rbp), %ecx
	vmovapd	(%rsi), %ymm6
	vmovapd	(%r11), %ymm1
	sall	$2, %ecx
	movslq	%ecx, %rcx
	salq	$3, %rcx
	leaq	(%rsi,%rcx), %rdx
	addq	%rdx, %rcx
	cmpl	$3, %edi
	vmovapd	(%rdx), %ymm4
	vmovapd	(%rcx), %ymm2
	jle	.L405
	subl	$4, %edi
	vmovapd	%ymm6, -176(%rbp)
	vmovapd	%ymm2, %ymm0
	shrl	$2, %edi
	leaq	32(%rcx), %rax
	vmovapd	%ymm5, %ymm6
	salq	$7, %rdi
	vmovapd	%ymm4, %ymm3
	vmovapd	%ymm1, %ymm2
	leaq	160(%rcx,%rdi), %rcx
	vmovapd	-144(%rbp), %ymm4
	vmovapd	%ymm0, %ymm1
	vmovapd	-176(%rbp), %ymm5
	.p2align 4,,10
	.p2align 3
.L406:
	vfmadd231pd	%ymm2, %ymm5, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vfmadd231pd	%ymm2, %ymm1, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rax
	subq	$-128, %rsi
	subq	$-128, %rdx
	subq	$-128, %r11
	vfmadd231pd	%ymm2, %ymm5, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm1, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vshufpd	$5, %ymm2, %ymm2, %ymm0
	vfmadd231pd	%ymm2, %ymm5, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm1, %ymm7
	vmovapd	-112(%rbp), %ymm2
	vfmadd132pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm1, %ymm2
	vmovapd	-96(%rsi), %ymm6
	vmovapd	-128(%rax), %ymm1
	vfmadd132pd	%ymm0, %ymm4, %ymm3
	vmovapd	-96(%rdx), %ymm4
	vmovapd	-96(%r11), %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm15
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm10
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm9
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vfmadd231pd	%ymm0, %ymm1, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm4, %ymm3
	vmovapd	-64(%rsi), %ymm6
	vmovapd	-64(%rdx), %ymm4
	vfmadd132pd	%ymm1, %ymm2, %ymm0
	vmovapd	-96(%rax), %ymm2
	vmovapd	-64(%r11), %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm15
	vfmadd231pd	%ymm1, %ymm4, %ymm14
	vfmadd231pd	%ymm1, %ymm2, %ymm13
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm12
	vfmadd231pd	%ymm1, %ymm4, %ymm11
	vfmadd231pd	%ymm1, %ymm2, %ymm10
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm9
	vfmadd231pd	%ymm1, %ymm4, %ymm8
	vfmadd231pd	%ymm1, %ymm2, %ymm7
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm6, %ymm5
	vfmadd231pd	%ymm1, %ymm4, %ymm3
	vmovapd	-32(%rsi), %ymm6
	vmovapd	-32(%rdx), %ymm4
	vfmadd132pd	%ymm2, %ymm0, %ymm1
	vmovapd	-32(%r11), %ymm2
	vmovapd	-64(%rax), %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm4, %ymm14
	vfmadd231pd	%ymm2, %ymm0, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm4, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm4, %ymm8
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm0
	vmovapd	-32(%rax), %ymm1
	vfmadd132pd	%ymm2, %ymm5, %ymm6
	vfmadd132pd	%ymm2, %ymm3, %ymm4
	vmovapd	(%rsi), %ymm5
	vmovapd	(%rdx), %ymm3
	vmovapd	(%r11), %ymm2
	cmpq	%rax, %rcx
	vmovapd	%ymm0, -112(%rbp)
	jne	.L406
	vmovapd	%ymm6, %ymm5
	vmovapd	%ymm4, -144(%rbp)
.L405:
	vblendpd	$10, %ymm12, %ymm15, %ymm0
	vblendpd	$10, %ymm9, %ymm5, %ymm6
	vmovapd	-144(%rbp), %ymm3
	vblendpd	$5, %ymm9, %ymm5, %ymm9
	movl	-80(%rbp), %eax
	vblendpd	$5, %ymm12, %ymm15, %ymm12
	vblendpd	$10, %ymm8, %ymm3, %ymm1
	vblendpd	$12, %ymm6, %ymm0, %ymm5
	vblendpd	$3, %ymm6, %ymm0, %ymm6
	vblendpd	$10, %ymm11, %ymm14, %ymm0
	vblendpd	$5, %ymm8, %ymm3, %ymm8
	vblendpd	$5, %ymm11, %ymm14, %ymm11
	testl	%eax, %eax
	vblendpd	$12, %ymm1, %ymm0, %ymm4
	vblendpd	$3, %ymm1, %ymm0, %ymm0
	vmovapd	-112(%rbp), %ymm1
	vblendpd	$10, %ymm10, %ymm13, %ymm3
	vblendpd	$5, %ymm10, %ymm13, %ymm10
	vblendpd	$10, %ymm7, %ymm1, %ymm2
	vblendpd	$5, %ymm7, %ymm1, %ymm7
	vblendpd	$12, %ymm9, %ymm12, %ymm15
	vblendpd	$3, %ymm9, %ymm12, %ymm9
	vblendpd	$12, %ymm2, %ymm3, %ymm14
	vblendpd	$12, %ymm8, %ymm11, %ymm12
	vblendpd	$12, %ymm7, %ymm10, %ymm13
	vblendpd	$3, %ymm8, %ymm11, %ymm8
	vblendpd	$3, %ymm2, %ymm3, %ymm2
	vblendpd	$3, %ymm7, %ymm10, %ymm7
	je	.L407
	movl	-184(%rbp), %eax
	movq	-192(%rbp), %rcx
	sall	$2, %eax
	vaddpd	(%rcx), %ymm5, %ymm5
	cltq
	salq	$3, %rax
	vaddpd	32(%rcx), %ymm15, %ymm15
	leaq	(%rcx,%rax), %rdx
	vaddpd	64(%rcx), %ymm6, %ymm6
	addq	%rdx, %rax
	vaddpd	96(%rcx), %ymm9, %ymm9
	vaddpd	(%rdx), %ymm4, %ymm4
	vaddpd	32(%rdx), %ymm12, %ymm12
	vaddpd	64(%rdx), %ymm0, %ymm0
	vaddpd	96(%rdx), %ymm8, %ymm8
	vaddpd	(%rax), %ymm14, %ymm14
	vaddpd	32(%rax), %ymm13, %ymm13
	vaddpd	64(%rax), %ymm2, %ymm2
	vaddpd	96(%rax), %ymm7, %ymm7
.L407:
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	%xmm5, %xmm3, %xmm1
	vmovsd	.LC0(%rip), %xmm3
	vucomisd	%xmm3, %xmm1
	jbe	.L408
	vmovsd	.LC1(%rip), %xmm11
	vsqrtsd	%xmm1, %xmm1, %xmm1
	movq	-72(%rbp), %rax
	vmovdqu	mask_bkp.27907(%rip), %ymm10
	vdivsd	%xmm1, %xmm11, %xmm1
	vmovlpd	%xmm1, (%rbx)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm1, %ymm5, %ymm5
	vmulpd	%ymm1, %ymm4, %ymm4
	vmulpd	%ymm1, %ymm14, %ymm14
	vmovapd	%ymm5, (%r12)
	vmovapd	%ymm4, 0(%r13)
	vmaskmovpd	%ymm14, %ymm10, (%rax)
.L409:
	vpermpd	$85, %ymm5, %ymm1
	vfmadd231pd	%ymm1, %ymm5, %ymm15
	vpermilpd	$3, %xmm15, %xmm10
	vmovlpd	%xmm1, 8(%rbx)
	vfmadd231pd	%ymm1, %ymm4, %ymm12
	vfmadd231pd	%ymm1, %ymm14, %ymm13
	vucomisd	%xmm3, %xmm10
	jbe	.L410
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vdivsd	%xmm10, %xmm1, %xmm10
	vmovdqu	mask_bkp.27907(%rip), %ymm1
	vmovlpd	%xmm10, 16(%rbx)
	vbroadcastsd	%xmm10, %ymm10
	vmulpd	%ymm15, %ymm10, %ymm11
	vmulpd	%ymm10, %ymm12, %ymm12
	vmulpd	%ymm10, %ymm13, %ymm13
.L433:
	vmovdqa	.LC2(%rip), %ymm10
	vmaskmovpd	%ymm11, %ymm10, 32(%r12)
	movq	-72(%rbp), %rax
	vpermpd	$170, %ymm5, %ymm10
	vmovapd	%ymm12, 32(%r13)
	vmaskmovpd	%ymm13, %ymm1, 32(%rax)
	vfmadd231pd	%ymm10, %ymm5, %ymm6
	vfmadd231pd	%ymm10, %ymm4, %ymm0
	vfmadd231pd	%ymm10, %ymm14, %ymm2
	vmovlpd	%xmm10, 24(%rbx)
	vpermpd	$170, %ymm11, %ymm10
	vmovlpd	%xmm10, 32(%rbx)
	vfmadd231pd	%ymm10, %ymm11, %ymm6
	vfmadd231pd	%ymm10, %ymm12, %ymm0
	vfmadd231pd	%ymm10, %ymm13, %ymm2
	vextractf128	$0x1, %ymm6, %xmm10
	vucomisd	%xmm3, %xmm10
	jbe	.L412
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vmovdqu	mask_bkp.27907(%rip), %ymm15
	vdivsd	%xmm10, %xmm1, %xmm10
	vmovlpd	%xmm10, 40(%rbx)
	vbroadcastsd	%xmm10, %ymm10
	vmulpd	%ymm6, %ymm10, %ymm1
	vmovdqa	.LC3(%rip), %ymm6
	vmulpd	%ymm10, %ymm0, %ymm0
	vmulpd	%ymm10, %ymm2, %ymm2
	vmaskmovpd	%ymm1, %ymm6, 64(%r12)
	movq	-72(%rbp), %rax
	vmovapd	%ymm0, 64(%r13)
	vmaskmovpd	%ymm2, %ymm15, 64(%rax)
.L413:
	vpermpd	$255, %ymm5, %ymm6
	vfmadd231pd	%ymm6, %ymm5, %ymm9
	vfmadd231pd	%ymm6, %ymm4, %ymm8
	vfmadd231pd	%ymm6, %ymm14, %ymm7
	vmovlpd	%xmm6, 48(%rbx)
	vpermpd	$255, %ymm11, %ymm6
	vfmadd231pd	%ymm6, %ymm11, %ymm9
	vfmadd231pd	%ymm6, %ymm12, %ymm8
	vfmadd132pd	%ymm6, %ymm7, %ymm13
	vmovlpd	%xmm6, 56(%rbx)
	vpermpd	$255, %ymm1, %ymm6
	vfmadd231pd	%ymm6, %ymm1, %ymm9
	vextractf128	$0x1, %ymm9, %xmm1
	vmovlpd	%xmm6, 64(%rbx)
	vfmadd132pd	%ymm6, %ymm8, %ymm0
	vfmadd231pd	%ymm6, %ymm2, %ymm13
	vpermilpd	$3, %xmm1, %xmm1
	vucomisd	%xmm3, %xmm1
	ja	.L436
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovdqa	.LC4(%rip), %ymm2
	vmovdqu	mask_bkp.27907(%rip), %ymm1
	movq	$0, 72(%rbx)
	vmaskmovpd	%ymm3, %ymm2, 96(%r12)
	movq	-72(%rbp), %rax
	vmovapd	%ymm0, 96(%r13)
	vmaskmovpd	%ymm13, %ymm1, 96(%rax)
.L431:
	vzeroupper
	addq	$72, %rsp
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L394:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.27908(%rip), %ymm1
	testl	%ecx, %ecx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC7(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqa	%ymm0, -240(%rbp)
	vmovdqu	%ymm0, mask_bkp.27907(%rip)
	jg	.L437
.L416:
	vxorpd	%xmm7, %xmm7, %xmm7
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm14
	vmovapd	%ymm7, %ymm9
	vmovapd	%ymm7, %ymm12
	vmovapd	%ymm7, %ymm15
	jmp	.L396
	.p2align 4,,10
	.p2align 3
.L412:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovdqu	mask_bkp.27907(%rip), %ymm10
	movq	$0, 40(%rbx)
	vblendpd	$7, %ymm1, %ymm6, %ymm1
	vmovdqa	.LC3(%rip), %ymm6
	vmaskmovpd	%ymm1, %ymm6, 64(%r12)
	movq	-72(%rbp), %rax
	vmovapd	%ymm0, 64(%r13)
	vmaskmovpd	%ymm2, %ymm10, 64(%rax)
	jmp	.L413
	.p2align 4,,10
	.p2align 3
.L410:
	vxorpd	%xmm11, %xmm11, %xmm11
	vmovdqu	mask_bkp.27907(%rip), %ymm1
	movq	$0, 16(%rbx)
	vblendpd	$3, %ymm11, %ymm15, %ymm11
	jmp	.L433
	.p2align 4,,10
	.p2align 3
.L408:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	-72(%rbp), %rax
	movq	$0, (%rbx)
	vblendpd	$1, %ymm1, %ymm5, %ymm5
	vmovdqa	-240(%rbp), %ymm1
	vmovapd	%ymm5, (%r12)
	vmovapd	%ymm4, 0(%r13)
	vmaskmovpd	%ymm14, %ymm1, (%rax)
	jmp	.L409
	.p2align 4,,10
	.p2align 3
.L435:
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vfmadd231pd	%ymm2, %ymm0, %ymm13
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm10
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd213pd	-144(%rbp), %ymm2, %ymm3
	vfmadd213pd	-112(%rbp), %ymm2, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	%ymm0, -112(%rbp)
	jmp	.L396
	.p2align 4,,10
	.p2align 3
.L436:
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vmovdqa	.LC4(%rip), %ymm3
	vdivsd	%xmm1, %xmm2, %xmm1
	vmovdqu	mask_bkp.27907(%rip), %ymm2
	vmovlpd	%xmm1, 72(%rbx)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm9, %ymm1, %ymm9
	vmulpd	%ymm0, %ymm1, %ymm0
	vmulpd	%ymm13, %ymm1, %ymm13
	vmaskmovpd	%ymm9, %ymm3, 96(%r12)
	movq	-72(%rbp), %rax
	vmovapd	%ymm0, 96(%r13)
	vmaskmovpd	%ymm13, %ymm2, 96(%rax)
	jmp	.L431
	.p2align 4,,10
	.p2align 3
.L434:
	cmpl	$3, %ecx
	vxorpd	%xmm1, %xmm1, %xmm1
	jle	.L398
	vblendpd	$1, %ymm6, %ymm1, %ymm6
	vmovapd	32(%rax), %ymm3
	vblendpd	$3, 32(%r9), %ymm1, %ymm12
	vblendpd	$7, 64(%r9), %ymm1, %ymm5
	movq	-176(%rbp), %r15
	leaq	128(%r9), %r10
	vfmadd132pd	%ymm6, %ymm1, %ymm2
	vfmadd231pd	%ymm3, %ymm12, %ymm2
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vmovapd	%ymm2, %ymm15
	vmovapd	64(%rax), %ymm2
	vmovapd	%ymm5, %ymm9
	vmovapd	96(%r9), %ymm6
	movq	%r15, %rdx
	leaq	128(%r14), %rdi
	vfmadd231pd	%ymm2, %ymm5, %ymm15
	vfmadd132pd	%ymm3, %ymm1, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	96(%rax), %ymm3
	subq	$-128, %rdx
	cmpl	$7, %ecx
	leaq	128(%rax), %r8
	vfmadd231pd	%ymm3, %ymm6, %ymm15
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm2, %ymm5, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm3, %ymm6, %ymm12
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm2, %ymm1, %ymm9
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm3, %ymm6, %ymm9
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm5, %ymm1, %ymm2
	vmovapd	%ymm6, %ymm5
	vmovapd	128(%r9), %ymm6
	vfmadd132pd	%ymm3, %ymm2, %ymm5
	vmovapd	128(%rax), %ymm2
	vmovapd	128(%r14), %ymm3
	jle	.L399
	vblendpd	$1, %ymm3, %ymm1, %ymm3
	vmovapd	160(%r9), %ymm7
	vblendpd	$3, 160(%r14), %ymm1, %ymm4
	vmovapd	160(%rax), %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	movq	%r15, %rdx
	vmovapd	%ymm3, %ymm14
	vmovapd	%ymm3, %ymm11
	addq	$256, %rdx
	vfmadd231pd	%ymm0, %ymm7, %ymm15
	vmovapd	%ymm3, %ymm8
	cmpl	$11, %ecx
	vfmadd132pd	%ymm2, %ymm1, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	leaq	256(%r9), %r10
	leaq	256(%r14), %rdi
	leaq	256(%rax), %r8
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd132pd	%ymm2, %ymm1, %ymm11
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd132pd	%ymm2, %ymm1, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm9
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm2, %ymm5, %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm3
	vmovapd	192(%r9), %ymm5
	vfmadd132pd	%ymm0, %ymm3, %ymm4
	vfmadd132pd	%ymm0, %ymm6, %ymm7
	vblendpd	$7, 192(%r14), %ymm1, %ymm3
	vmovapd	192(%rax), %ymm0
	vmovapd	224(%r9), %ymm6
	vfmadd231pd	%ymm0, %ymm5, %ymm15
	vfmadd231pd	%ymm0, %ymm3, %ymm14
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	224(%r14), %ymm2
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm3, %ymm11
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm5, %ymm9
	vfmadd231pd	%ymm0, %ymm3, %ymm8
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm7, %ymm5
	vfmadd132pd	%ymm3, %ymm4, %ymm0
	vmovapd	224(%rax), %ymm4
	vfmadd231pd	%ymm4, %ymm6, %ymm15
	vfmadd231pd	%ymm4, %ymm2, %ymm14
	vshufpd	$5, %ymm4, %ymm4, %ymm4
	vfmadd231pd	%ymm4, %ymm6, %ymm12
	vfmadd231pd	%ymm4, %ymm2, %ymm11
	vperm2f128	$1, %ymm4, %ymm4, %ymm4
	vshufpd	$5, %ymm4, %ymm4, %ymm3
	vfmadd231pd	%ymm4, %ymm6, %ymm9
	vfmadd231pd	%ymm4, %ymm2, %ymm8
	vfmadd231pd	%ymm3, %ymm6, %ymm5
	vfmadd132pd	%ymm3, %ymm0, %ymm2
	vmovapd	256(%r9), %ymm6
	vmovapd	%ymm2, %ymm13
	vmovapd	%ymm2, -144(%rbp)
	vmovapd	256(%r14), %ymm3
	vmovapd	256(%rax), %ymm2
	vmovapd	256(%r15), %ymm0
	jg	.L438
	cmpl	$8, %ecx
	je	.L418
	vblendpd	$1, %ymm0, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	movq	-176(%rbp), %r15
	cmpl	$9, %ecx
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm10
	vmovapd	%ymm0, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd132pd	%ymm2, %ymm1, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd132pd	%ymm2, %ymm1, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd213pd	-144(%rbp), %ymm2, %ymm3
	vfmadd132pd	%ymm2, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	288(%r9), %ymm6
	vmovapd	288(%r14), %ymm3
	vmovapd	288(%rax), %ymm2
	vmovapd	288(%r15), %ymm0
	je	.L419
	vblendpd	$3, %ymm0, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vmovapd	%ymm3, %ymm4
	cmpl	$11, %ecx
	vfmadd231pd	%ymm2, %ymm0, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	320(%r14), %ymm3
	vfmadd213pd	-112(%rbp), %ymm2, %ymm0
	vfmadd213pd	-144(%rbp), %ymm2, %ymm4
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm4, -144(%rbp)
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	320(%r9), %ymm6
	vmovapd	320(%rax), %ymm2
	vmovapd	320(%r15), %ymm0
	jne	.L420
	vblendpd	$7, %ymm0, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	movq	%rdx, -176(%rbp)
	vfmadd231pd	%ymm2, %ymm0, %ymm13
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm10
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd213pd	-144(%rbp), %ymm2, %ymm3
	vfmadd132pd	%ymm2, %ymm4, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	352(%r9), %ymm6
	movq	%r10, %r9
	vmovapd	352(%r14), %ymm3
	movq	%rdi, %r14
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	352(%rax), %ymm2
	movq	%r8, %rax
	vmovapd	352(%r15), %ymm0
	movl	$11, %r15d
	jmp	.L397
	.p2align 4,,10
	.p2align 3
.L398:
	vblendpd	$1, %ymm6, %ymm1, %ymm15
	cmpl	$1, %ecx
	vmovapd	32(%r9), %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm15
	vmovapd	32(%rax), %ymm2
	je	.L424
	vblendpd	$3, %ymm6, %ymm1, %ymm6
	cmpl	$3, %ecx
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm6
	vmovapd	64(%rax), %ymm2
	vmovapd	%ymm6, %ymm12
	vmovapd	64(%r9), %ymm6
	jne	.L425
	vblendpd	$7, %ymm6, %ymm1, %ymm6
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, -112(%rbp)
	vmovapd	%ymm1, %ymm10
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, -144(%rbp)
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm6, %ymm9
	vmovapd	%ymm6, %ymm5
	vmovapd	%ymm1, %ymm8
	movl	$3, %r15d
	vmovapd	%ymm1, %ymm11
	vmovapd	%ymm1, %ymm14
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vmovapd	96(%r9), %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm9
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm5
	vmovapd	96(%rax), %ymm2
	jmp	.L397
.L399:
	cmpl	$4, %ecx
	je	.L421
	vblendpd	$1, %ymm3, %ymm1, %ymm3
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	cmpl	$5, %ecx
	vmovapd	%ymm3, %ymm14
	vmovapd	%ymm3, %ymm11
	vmovapd	%ymm3, %ymm8
	vfmadd132pd	%ymm2, %ymm1, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd132pd	%ymm2, %ymm1, %ymm11
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd132pd	%ymm2, %ymm1, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm3
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, %ymm7
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	160(%r9), %ymm6
	vmovapd	160(%r14), %ymm3
	vmovapd	160(%rax), %ymm2
	je	.L422
	vblendpd	$3, %ymm3, %ymm1, %ymm3
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	cmpl	$7, %ecx
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm7, %ymm3
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, %ymm7
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	192(%r9), %ymm6
	vmovapd	192(%r14), %ymm3
	vmovapd	192(%rax), %ymm2
	jne	.L423
	vblendpd	$7, %ymm3, %ymm1, %ymm3
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	movq	%rdx, -176(%rbp)
	movl	$7, %r15d
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm7, %ymm3
	vxorpd	%xmm7, %xmm7, %xmm7
	vfmadd231pd	%ymm2, %ymm6, %ymm5
	vmovapd	%ymm3, -144(%rbp)
	vmovapd	224(%r9), %ymm6
	movq	%r10, %r9
	vmovapd	224(%r14), %ymm3
	vmovapd	%ymm7, %ymm10
	movq	%rdi, %r14
	vmovapd	224(%rax), %ymm2
	vmovapd	%ymm7, %ymm13
	movq	%r8, %rax
	vmovapd	%ymm7, -112(%rbp)
	jmp	.L397
.L425:
	vxorpd	%xmm7, %xmm7, %xmm7
	movl	$2, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm14
	vmovapd	%ymm7, %ymm9
	jmp	.L397
.L421:
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$4, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm14
	jmp	.L397
.L438:
	vblendpd	$1, %ymm0, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm6, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	movq	%r15, %rdx
	addq	$384, %r9
	addq	$384, %r14
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm7
	addq	$384, %rdx
	vmovapd	%ymm0, %ymm10
	movq	%rdx, -176(%rbp)
	addq	$384, %rax
	vfmadd132pd	%ymm2, %ymm1, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm6, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd132pd	%ymm2, %ymm1, %ymm7
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm1, %ymm10
	vfmadd231pd	%ymm2, %ymm6, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm13, %ymm3
	vfmadd132pd	%ymm2, %ymm1, %ymm0
	vmovapd	%ymm3, -112(%rbp)
	vmovapd	%ymm0, -144(%rbp)
	vfmadd132pd	%ymm2, %ymm5, %ymm6
	vblendpd	$3, 288(%r15), %ymm1, %ymm0
	vmovapd	-96(%r9), %ymm5
	vmovapd	-96(%r14), %ymm3
	vmovapd	-96(%rax), %ymm2
	vfmadd231pd	%ymm2, %ymm0, %ymm4
	vfmadd231pd	%ymm2, %ymm5, %ymm15
	vfmadd231pd	%ymm2, %ymm3, %ymm14
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm4, %ymm13
	vmovapd	%ymm10, %ymm4
	vfmadd231pd	%ymm2, %ymm5, %ymm12
	vfmadd231pd	%ymm2, %ymm3, %ymm11
	vfmadd231pd	%ymm2, %ymm0, %ymm7
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm0, %ymm4
	vfmadd231pd	%ymm2, %ymm5, %ymm9
	vfmadd231pd	%ymm2, %ymm3, %ymm8
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm5, %ymm6
	vfmadd213pd	-112(%rbp), %ymm2, %ymm3
	vmovapd	-64(%r9), %ymm5
	vfmadd213pd	-144(%rbp), %ymm2, %ymm0
	vmovapd	%ymm3, -112(%rbp)
	vblendpd	$7, 320(%r15), %ymm1, %ymm2
	vmovapd	-64(%r14), %ymm3
	vmovapd	%ymm0, -144(%rbp)
	vmovapd	-64(%rax), %ymm0
	vmovapd	-144(%rbp), %ymm1
	vfmadd231pd	%ymm0, %ymm5, %ymm15
	vfmadd231pd	%ymm0, %ymm3, %ymm14
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm2, %ymm7
	vfmadd231pd	%ymm0, %ymm5, %ymm12
	vfmadd231pd	%ymm0, %ymm3, %ymm11
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm4, %ymm7
	vmovapd	-32(%r14), %ymm4
	vfmadd231pd	%ymm0, %ymm5, %ymm9
	vfmadd231pd	%ymm0, %ymm3, %ymm8
	vfmadd231pd	%ymm0, %ymm2, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd213pd	-112(%rbp), %ymm0, %ymm3
	vfmadd132pd	%ymm0, %ymm6, %ymm5
	vfmadd231pd	%ymm0, %ymm2, %ymm1
	vmovapd	-32(%r9), %ymm6
	vmovapd	-32(%rax), %ymm0
	vmovapd	352(%r15), %ymm2
	vfmadd231pd	%ymm0, %ymm6, %ymm15
	vfmadd231pd	%ymm0, %ymm4, %ymm14
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm12
	vfmadd231pd	%ymm0, %ymm4, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm10
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm6, %ymm9
	vfmadd231pd	%ymm0, %ymm4, %ymm8
	vfmadd231pd	%ymm0, %ymm2, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm3, %ymm4
	vfmadd132pd	%ymm0, %ymm1, %ymm2
	vfmadd231pd	%ymm0, %ymm6, %ymm5
	vmovapd	(%r14), %ymm3
	vmovapd	(%r9), %ymm6
	vmovapd	%ymm2, -112(%rbp)
	vmovapd	384(%r15), %ymm0
	movl	$12, %r15d
	vmovapd	(%rax), %ymm2
	vmovapd	%ymm4, -144(%rbp)
	jmp	.L397
.L424:
	vxorpd	%xmm7, %xmm7, %xmm7
	movl	$1, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	vmovapd	%ymm7, %ymm5
	vmovapd	%ymm7, -144(%rbp)
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm11
	vmovapd	%ymm7, %ymm14
	vmovapd	%ymm7, %ymm9
	vmovapd	%ymm7, %ymm12
	jmp	.L397
.L422:
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$5, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	jmp	.L397
.L418:
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$8, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	jmp	.L397
.L419:
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$9, %r15d
	jmp	.L397
.L423:
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$6, %r15d
	vmovapd	%ymm7, -112(%rbp)
	vmovapd	%ymm7, %ymm10
	vmovapd	%ymm7, %ymm13
	jmp	.L397
.L420:
	movq	%rdx, -176(%rbp)
	movq	%rdi, %r14
	movq	%r8, %rax
	movq	%r10, %r9
	movl	$10, %r15d
	jmp	.L397
	.cfi_endproc
.LFE4599:
	.size	kernel_dsyrk_dpotrf_nt_12x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_12x4_vs_lib4
	.section	.text.unlikely
.LCOLDE21:
	.text
.LHOTE21:
	.section	.text.unlikely
.LCOLDB22:
	.text
.LHOTB22:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_8x8_vs_lib4:
.LFB4600:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	movl	%ecx, %r14d
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movl	32(%r10), %ebx
	movl	64(%r10), %r13d
	movl	80(%r10), %r12d
	movl	%r8d, -124(%rbp)
	movl	(%r10), %eax
	movq	8(%r10), %r15
	movl	%ebx, -128(%rbp)
	movl	48(%r10), %ebx
	sall	$2, %r13d
	movl	16(%r10), %r8d
	movq	24(%r10), %rcx
	movslq	%r13d, %r13
	movq	40(%r10), %rsi
	movq	88(%r10), %r11
	sall	$2, %r12d
	movl	%ebx, -140(%rbp)
	movq	56(%r10), %rbx
	movslq	%r12d, %r12
	cmpl	$7, %edi
	movq	%rbx, -80(%rbp)
	movq	72(%r10), %rbx
	movl	96(%r10), %r10d
	movl	%r10d, -72(%rbp)
	movq	-80(%rbp), %r10
	leaq	(%r10,%r13,8), %r10
	movq	%r10, -136(%rbp)
	leaq	(%rbx,%r12,8), %r10
	jle	.L440
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%r14d, %r14d
	vmovdqu	%ymm0, mask_bkp.28009(%rip)
	jle	.L471
.L495:
	sall	$2, %eax
	sall	$2, %r8d
	cmpl	$1, %edx
	cltq
	movslq	%r8d, %r8
	vmovapd	(%r9), %ymm15
	leaq	(%r15,%r8,8), %rdi
	leaq	(%r9,%rax,8), %rax
	vmovapd	(%r15), %ymm2
	movq	%rdi, -120(%rbp)
	vmovapd	(%rax), %ymm14
	vmovapd	(%rdi), %ymm10
	je	.L491
	vxorpd	%xmm0, %xmm0, %xmm0
	movl	$0, -68(%rbp)
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm4
.L443:
	leal	-3(%r14), %edx
	cmpl	%edx, -68(%rbp)
	jge	.L492
	leal	-4(%r14), %edx
	subl	-68(%rbp), %edx
	movq	-120(%rbp), %r8
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%r15, %rdi
	shrl	$2, %edx
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm9
	movl	%edx, %r13d
	vmovapd	%ymm1, %ymm3
	movl	%edx, -144(%rbp)
	addq	$1, %r13
	movq	%r9, %rdx
	salq	$7, %r13
	leaq	(%rax,%r13), %r12
	.p2align 4,,10
	.p2align 3
.L449:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	subq	$-128, %rdx
	subq	$-128, %rdi
	subq	$-128, %r8
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vmovapd	-96(%rdx), %ymm15
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vmovapd	-96(%rdi), %ymm0
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	-96(%rax), %ymm10
	vmovapd	-96(%r8), %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm4
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd231pd	%ymm0, %ymm10, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm0, %ymm7, %ymm15
	vmovapd	-64(%rdx), %ymm7
	vfmadd132pd	%ymm10, %ymm2, %ymm0
	vmovapd	-64(%rax), %ymm2
	vfmadd132pd	%ymm1, %ymm14, %ymm10
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	-64(%r8), %ymm1
	vmovapd	-64(%rdi), %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	-112(%rbp), %ymm14
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vfmadd231pd	%ymm0, %ymm2, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm15
	vfmadd231pd	%ymm0, %ymm2, %ymm14
	vmovapd	-32(%rdx), %ymm7
	vmovapd	-32(%rdi), %ymm0
	vfmadd132pd	%ymm1, %ymm10, %ymm2
	vmovapd	-32(%rax), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	-32(%r8), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vmovapd	(%rdi), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vfmadd132pd	%ymm1, %ymm14, %ymm0
	vmovapd	(%rax), %ymm14
	vmovapd	(%rdx), %ymm15
	vfmadd213pd	-112(%rbp), %ymm10, %ymm1
	vmovapd	(%r8), %ymm10
	cmpq	%r12, %rax
	jne	.L449
	movl	-68(%rbp), %eax
	movl	-144(%rbp), %edi
	addq	%r13, %r9
	addq	%r13, -120(%rbp)
	addq	%r13, %r15
	leal	4(%rax,%rdi,4), %eax
	movl	%eax, -68(%rbp)
.L448:
	movl	-68(%rbp), %edi
	leal	-1(%r14), %eax
	cmpl	%edi, %eax
	jle	.L450
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	movq	-120(%rbp), %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	addl	$2, %edi
	movl	%edi, -68(%rbp)
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd132pd	%ymm2, %ymm7, %ymm15
	vmovapd	32(%r9), %ymm7
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	32(%r15), %ymm0
	vmovapd	32(%r12), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	32(%rax), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vmovapd	64(%r15), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vfmadd213pd	-112(%rbp), %ymm1, %ymm0
	vmovapd	64(%r9), %ymm15
	vfmadd132pd	%ymm10, %ymm14, %ymm1
	vmovapd	64(%r12), %ymm14
	vmovapd	64(%rax), %ymm10
.L450:
	cmpl	%r14d, -68(%rbp)
	jl	.L493
.L442:
	movl	-124(%rbp), %edi
	testl	%edi, %edi
	jle	.L451
	movl	-128(%rbp), %eax
	vmovapd	(%rcx), %ymm15
	vmovapd	(%rsi), %ymm2
	sall	$2, %eax
	cltq
	leaq	(%rcx,%rax,8), %r9
	movl	-140(%rbp), %eax
	vmovapd	(%r9), %ymm14
	sall	$2, %eax
	cmpl	$3, %edi
	cltq
	leaq	(%rsi,%rax,8), %rdx
	vmovapd	(%rdx), %ymm10
	jle	.L451
	subl	$4, %edi
	leaq	32(%r9), %rax
	shrl	$2, %edi
	salq	$7, %rdi
	leaq	160(%r9,%rdi), %rdi
	.p2align 4,,10
	.p2align 3
.L452:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	subq	$-128, %rax
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	subq	$-128, %rcx
	subq	$-128, %rsi
	subq	$-128, %rdx
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vmovapd	-96(%rcx), %ymm15
	vfmadd132pd	%ymm14, %ymm0, %ymm2
	vmovapd	-96(%rsi), %ymm0
	vfmadd132pd	%ymm10, %ymm1, %ymm14
	vmovapd	-128(%rax), %ymm10
	vmovapd	-96(%rdx), %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm4
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vfmadd231pd	%ymm0, %ymm10, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd231pd	%ymm0, %ymm10, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm10, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm0, %ymm7, %ymm15
	vmovapd	-64(%rcx), %ymm7
	vfmadd132pd	%ymm10, %ymm2, %ymm0
	vmovapd	-96(%rax), %ymm2
	vfmadd132pd	%ymm1, %ymm14, %ymm10
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	-64(%rdx), %ymm1
	vmovapd	-64(%rsi), %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm3
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmovapd	-112(%rbp), %ymm14
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vfmadd231pd	%ymm0, %ymm2, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm9
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm2, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm1, %ymm2, %ymm8
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm2, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm15
	vfmadd231pd	%ymm0, %ymm2, %ymm14
	vmovapd	-32(%rcx), %ymm7
	vmovapd	-32(%rsi), %ymm0
	vfmadd132pd	%ymm1, %ymm10, %ymm2
	vmovapd	-64(%rax), %ymm1
	vmovapd	%ymm2, -112(%rbp)
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vmovapd	-32(%rdx), %ymm2
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm2, %ymm1, %ymm9
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vshufpd	$5, %ymm2, %ymm2, %ymm10
	vfmadd231pd	%ymm2, %ymm1, %ymm8
	vmovapd	(%rsi), %ymm2
	vfmadd132pd	%ymm0, %ymm15, %ymm7
	vfmadd132pd	%ymm1, %ymm14, %ymm0
	vmovapd	-32(%rax), %ymm14
	vmovapd	(%rcx), %ymm15
	vfmadd213pd	-112(%rbp), %ymm10, %ymm1
	vmovapd	(%rdx), %ymm10
	cmpq	%rax, %rdi
	jne	.L452
.L451:
	vblendpd	$10, %ymm6, %ymm4, %ymm2
	movl	-72(%rbp), %edx
	vblendpd	$5, %ymm6, %ymm4, %ymm6
	vblendpd	$10, %ymm11, %ymm7, %ymm4
	vblendpd	$5, %ymm11, %ymm7, %ymm7
	vblendpd	$10, %ymm12, %ymm0, %ymm10
	vblendpd	$5, %ymm12, %ymm0, %ymm0
	vblendpd	$12, %ymm4, %ymm2, %ymm11
	vblendpd	$3, %ymm4, %ymm2, %ymm4
	testl	%edx, %edx
	vblendpd	$12, %ymm7, %ymm6, %ymm2
	vblendpd	$3, %ymm7, %ymm6, %ymm7
	vblendpd	$10, %ymm13, %ymm5, %ymm6
	vblendpd	$5, %ymm13, %ymm5, %ymm5
	vblendpd	$12, %ymm10, %ymm6, %ymm12
	vblendpd	$12, %ymm0, %ymm5, %ymm15
	vblendpd	$3, %ymm10, %ymm6, %ymm6
	vblendpd	$3, %ymm0, %ymm5, %ymm0
	je	.L453
	movq	-80(%rbp), %rax
	vaddpd	(%rax), %ymm11, %ymm11
	vaddpd	32(%rax), %ymm2, %ymm2
	vaddpd	64(%rax), %ymm4, %ymm4
	vaddpd	96(%rax), %ymm7, %ymm7
	movq	-136(%rbp), %rax
	vaddpd	(%rax), %ymm12, %ymm12
	vaddpd	32(%rax), %ymm15, %ymm15
	vaddpd	64(%rax), %ymm6, %ymm6
	vaddpd	96(%rax), %ymm0, %ymm0
.L453:
	vxorpd	%xmm5, %xmm5, %xmm5
	vmovsd	.LC0(%rip), %xmm14
	vmovsd	%xmm11, %xmm5, %xmm10
	vucomisd	%xmm14, %xmm10
	jbe	.L454
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vdivsd	%xmm10, %xmm5, %xmm10
	vbroadcastsd	%xmm10, %ymm5
	vmovlpd	%xmm10, (%r11)
	vmovdqu	mask_bkp.28009(%rip), %ymm13
	vmulpd	%ymm5, %ymm11, %ymm11
	vmulpd	%ymm5, %ymm12, %ymm12
	vmovapd	%ymm11, (%rbx)
	vmaskmovpd	%ymm12, %ymm13, (%r10)
.L455:
	vpermpd	$85, %ymm11, %ymm5
	vmovlpd	%xmm5, 8(%r11)
	vfmadd231pd	%ymm5, %ymm11, %ymm2
	vfmadd231pd	%ymm5, %ymm12, %ymm15
	vpermilpd	$3, %xmm2, %xmm5
	vucomisd	%xmm14, %xmm5
	jbe	.L456
	vmovsd	.LC1(%rip), %xmm13
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm5, %xmm13, %xmm5
	vmovlpd	%xmm5, 16(%r11)
	vbroadcastsd	%xmm5, %ymm5
	vmovdqu	mask_bkp.28009(%rip), %ymm13
	vmulpd	%ymm2, %ymm5, %ymm2
	vmulpd	%ymm5, %ymm15, %ymm15
	vmovdqa	.LC2(%rip), %ymm5
	vmaskmovpd	%ymm2, %ymm5, 32(%rbx)
	vmaskmovpd	%ymm15, %ymm13, 32(%r10)
.L457:
	vpermpd	$170, %ymm11, %ymm5
	vfmadd231pd	%ymm5, %ymm11, %ymm4
	vfmadd231pd	%ymm5, %ymm12, %ymm6
	vmovlpd	%xmm5, 24(%r11)
	vpermpd	$170, %ymm2, %ymm5
	vmovlpd	%xmm5, 32(%r11)
	vfmadd231pd	%ymm5, %ymm2, %ymm4
	vfmadd231pd	%ymm5, %ymm15, %ymm6
	vextractf128	$0x1, %ymm4, %xmm5
	vucomisd	%xmm14, %xmm5
	jbe	.L458
	vmovsd	.LC1(%rip), %xmm13
	vsqrtsd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm5, %xmm13, %xmm5
	vmovlpd	%xmm5, 40(%r11)
	vbroadcastsd	%xmm5, %ymm5
	vmovdqu	mask_bkp.28009(%rip), %ymm13
	vmulpd	%ymm4, %ymm5, %ymm4
	vmulpd	%ymm5, %ymm6, %ymm6
	vmovdqa	.LC3(%rip), %ymm5
	vmaskmovpd	%ymm4, %ymm5, 64(%rbx)
	vmaskmovpd	%ymm6, %ymm13, 64(%r10)
.L459:
	vpermpd	$255, %ymm11, %ymm13
	vpermpd	$255, %ymm4, %ymm5
	vfmadd231pd	%ymm13, %ymm12, %ymm0
	vfmadd231pd	%ymm13, %ymm11, %ymm7
	vmovlpd	%xmm13, 48(%r11)
	vpermpd	$255, %ymm2, %ymm13
	vmovlpd	%xmm5, 64(%r11)
	vfmadd132pd	%ymm13, %ymm7, %ymm2
	vmovlpd	%xmm13, 56(%r11)
	vfmadd231pd	%ymm5, %ymm4, %ymm2
	vfmadd132pd	%ymm15, %ymm0, %ymm13
	vextractf128	$0x1, %ymm2, %xmm0
	vfmadd132pd	%ymm6, %ymm13, %ymm5
	vpermilpd	$3, %xmm0, %xmm0
	vucomisd	%xmm14, %xmm0
	jbe	.L460
	vmovsd	.LC1(%rip), %xmm4
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vdivsd	%xmm0, %xmm4, %xmm0
	vmovlpd	%xmm0, 72(%r11)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.28009(%rip), %ymm4
	vmulpd	%ymm2, %ymm0, %ymm2
	vmulpd	%ymm0, %ymm5, %ymm5
	vmovdqa	.LC4(%rip), %ymm0
	vmaskmovpd	%ymm2, %ymm0, 96(%rbx)
	vmaskmovpd	%ymm5, %ymm4, 96(%r10)
.L461:
	vfmadd231pd	%ymm12, %ymm12, %ymm3
	vmovapd	%ymm3, %ymm2
	vshufpd	$5, %ymm12, %ymm12, %ymm3
	vshufpd	$5, %ymm6, %ymm6, %ymm4
	movl	-72(%rbp), %eax
	vshufpd	$5, %ymm15, %ymm15, %ymm7
	vfmadd231pd	%ymm15, %ymm15, %ymm2
	vfmadd231pd	%ymm6, %ymm6, %ymm2
	vfmadd231pd	%ymm3, %ymm12, %ymm9
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vmovapd	%ymm9, %ymm0
	vfmadd231pd	%ymm5, %ymm5, %ymm2
	testl	%eax, %eax
	vfmadd231pd	%ymm7, %ymm15, %ymm0
	vperm2f128	$1, %ymm7, %ymm7, %ymm7
	vfmadd231pd	%ymm4, %ymm6, %ymm0
	vfmadd231pd	%ymm3, %ymm12, %ymm8
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vperm2f128	$1, %ymm4, %ymm4, %ymm4
	vfmadd132pd	%ymm12, %ymm1, %ymm3
	vmovapd	%ymm8, %ymm1
	vfmadd231pd	%ymm7, %ymm15, %ymm1
	vshufpd	$5, %ymm7, %ymm7, %ymm7
	vfmadd231pd	%ymm4, %ymm6, %ymm1
	vshufpd	$5, %ymm4, %ymm4, %ymm4
	vfmadd231pd	%ymm7, %ymm15, %ymm3
	vfmadd132pd	%ymm4, %ymm3, %ymm6
	vshufpd	$5, %ymm5, %ymm5, %ymm3
	vfmadd231pd	%ymm3, %ymm5, %ymm0
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vshufpd	$5, %ymm3, %ymm3, %ymm4
	vfmadd231pd	%ymm3, %ymm5, %ymm1
	vmovapd	%ymm5, %ymm3
	vblendpd	$10, %ymm0, %ymm2, %ymm5
	vblendpd	$5, %ymm0, %ymm2, %ymm0
	vfmadd132pd	%ymm4, %ymm6, %ymm3
	vblendpd	$10, %ymm1, %ymm3, %ymm2
	vblendpd	$5, %ymm1, %ymm3, %ymm3
	vblendpd	$12, %ymm2, %ymm5, %ymm4
	vblendpd	$12, %ymm3, %ymm0, %ymm1
	vblendpd	$3, %ymm2, %ymm5, %ymm2
	vblendpd	$3, %ymm3, %ymm0, %ymm0
	je	.L462
	movq	-136(%rbp), %rax
	vaddpd	128(%rax), %ymm4, %ymm4
	vaddpd	160(%rax), %ymm1, %ymm1
	vaddpd	192(%rax), %ymm2, %ymm2
	vaddpd	224(%rax), %ymm0, %ymm0
.L462:
	vmovsd	%xmm4, %xmm10, %xmm10
	vucomisd	%xmm14, %xmm10
	jbe	.L463
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vmovdqu	mask_bkp.28009(%rip), %ymm3
	vdivsd	%xmm10, %xmm5, %xmm10
	vmovlpd	%xmm10, 80(%r11)
	vbroadcastsd	%xmm10, %ymm10
	vmulpd	%ymm10, %ymm4, %ymm4
	vmaskmovpd	%ymm4, %ymm3, 128(%r10)
.L464:
	vpermpd	$85, %ymm4, %ymm3
	vmovlpd	%xmm3, 88(%r11)
	vfmadd132pd	%ymm4, %ymm1, %ymm3
	vpermilpd	$3, %xmm3, %xmm1
	vucomisd	%xmm14, %xmm1
	jbe	.L465
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm5, %xmm1
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	vandpd	.LC12(%rip), %ymm5, %ymm5
	vmovlpd	%xmm1, 96(%r11)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm3, %ymm1, %ymm1
	vmaskmovpd	%ymm1, %ymm5, 160(%r10)
.L466:
	vpermpd	$170, %ymm4, %ymm3
	vmovlpd	%xmm3, 104(%r11)
	vfmadd132pd	%ymm4, %ymm2, %ymm3
	vpermpd	$170, %ymm1, %ymm2
	vmovlpd	%xmm2, 112(%r11)
	vfmadd132pd	%ymm1, %ymm3, %ymm2
	vextractf128	$0x1, %ymm2, %xmm3
	vucomisd	%xmm14, %xmm3
	jbe	.L467
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm5, %xmm3
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	vandpd	.LC13(%rip), %ymm5, %ymm5
	vmovlpd	%xmm3, 120(%r11)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm2, %ymm3, %ymm2
	vmaskmovpd	%ymm2, %ymm5, 192(%r10)
.L468:
	vpermpd	$255, %ymm4, %ymm3
	vfmadd231pd	%ymm3, %ymm4, %ymm0
	vmovlpd	%xmm3, 128(%r11)
	vpermpd	$255, %ymm1, %ymm3
	vfmadd132pd	%ymm3, %ymm0, %ymm1
	vmovlpd	%xmm3, 136(%r11)
	vpermpd	$255, %ymm2, %ymm3
	vmovlpd	%xmm3, 144(%r11)
	vfmadd132pd	%ymm3, %ymm1, %ymm2
	vextractf128	$0x1, %ymm2, %xmm3
	vpermilpd	$3, %xmm3, %xmm3
	vucomisd	%xmm14, %xmm3
	ja	.L494
	vmovdqu	mask_bkp.28009(%rip), %ymm0
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 152(%r11)
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm0, 224(%r10)
.L489:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L460:
	.cfi_restore_state
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovdqa	.LC4(%rip), %ymm2
	vmovdqu	mask_bkp.28009(%rip), %ymm0
	movq	$0, 72(%r11)
	vmaskmovpd	%ymm4, %ymm2, 96(%rbx)
	vmaskmovpd	%ymm5, %ymm0, 96(%r10)
	jmp	.L461
	.p2align 4,,10
	.p2align 3
.L467:
	vmovdqu	mask_bkp.28009(%rip), %ymm3
	vxorpd	%xmm5, %xmm5, %xmm5
	movq	$0, 120(%r11)
	vandpd	.LC13(%rip), %ymm3, %ymm3
	vblendpd	$7, %ymm5, %ymm2, %ymm2
	vmaskmovpd	%ymm2, %ymm3, 192(%r10)
	jmp	.L468
	.p2align 4,,10
	.p2align 3
.L465:
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 96(%r11)
	vandpd	.LC12(%rip), %ymm5, %ymm5
	vblendpd	$3, %ymm1, %ymm3, %ymm1
	vmaskmovpd	%ymm1, %ymm5, 160(%r10)
	jmp	.L466
	.p2align 4,,10
	.p2align 3
.L463:
	vxorpd	%xmm5, %xmm5, %xmm5
	vmovdqu	mask_bkp.28009(%rip), %ymm3
	movq	$0, 80(%r11)
	vblendpd	$1, %ymm5, %ymm4, %ymm4
	vmaskmovpd	%ymm4, %ymm3, 128(%r10)
	jmp	.L464
	.p2align 4,,10
	.p2align 3
.L458:
	vxorpd	%xmm13, %xmm13, %xmm13
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	movq	$0, 40(%r11)
	vblendpd	$7, %ymm13, %ymm4, %ymm4
	vmovdqa	.LC3(%rip), %ymm13
	vmaskmovpd	%ymm4, %ymm13, 64(%rbx)
	vmaskmovpd	%ymm6, %ymm5, 64(%r10)
	jmp	.L459
	.p2align 4,,10
	.p2align 3
.L456:
	vxorpd	%xmm13, %xmm13, %xmm13
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	movq	$0, 16(%r11)
	vblendpd	$3, %ymm13, %ymm2, %ymm2
	vmovdqa	.LC2(%rip), %ymm13
	vmaskmovpd	%ymm2, %ymm13, 32(%rbx)
	vmaskmovpd	%ymm15, %ymm5, 32(%r10)
	jmp	.L457
	.p2align 4,,10
	.p2align 3
.L454:
	vxorpd	%xmm13, %xmm13, %xmm13
	vmovdqu	mask_bkp.28009(%rip), %ymm5
	movq	$0, (%r11)
	vblendpd	$1, %ymm13, %ymm11, %ymm11
	vmovapd	%ymm11, (%rbx)
	vmaskmovpd	%ymm12, %ymm5, (%r10)
	jmp	.L455
	.p2align 4,,10
	.p2align 3
.L440:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.28010(%rip), %ymm1
	testl	%r14d, %r14d
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC11(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.28009(%rip)
	jg	.L495
.L471:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm3
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm12
	vmovapd	%ymm1, %ymm13
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm11
	vmovapd	%ymm1, %ymm6
	vmovapd	%ymm1, %ymm4
	jmp	.L442
	.p2align 4,,10
	.p2align 3
.L493:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vfmadd231pd	%ymm10, %ymm14, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm9
	vperm2f128	$1, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm10, %ymm14, %ymm8
	vshufpd	$5, %ymm10, %ymm10, %ymm10
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vfmadd231pd	%ymm10, %ymm14, %ymm1
	jmp	.L442
	.p2align 4,,10
	.p2align 3
.L494:
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm0, %xmm3
	vmovdqu	mask_bkp.28009(%rip), %ymm0
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmovlpd	%xmm3, 152(%r11)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm2, %ymm3, %ymm2
	vmaskmovpd	%ymm2, %ymm0, 224(%r10)
	jmp	.L489
	.p2align 4,,10
	.p2align 3
.L491:
	cmpl	$3, %r14d
	jle	.L444
	vxorpd	%xmm1, %xmm1, %xmm1
	xorl	%edx, %edx
	cmpl	$7, %r14d
	vmovapd	32(%rdx), %ymm0
	leaq	128(%r9), %rdi
	vmovapd	128(%rax), %ymm14
	vblendpd	$1, %ymm15, %ymm1, %ymm4
	vblendpd	$3, 32(%r9), %ymm1, %ymm6
	vmovapd	128(%r9), %ymm15
	vblendpd	$7, 64(%r9), %ymm1, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm4
	vfmadd231pd	%ymm0, %ymm6, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm7, %ymm11
	vmovapd	96(%rdx), %ymm2
	vfmadd132pd	%ymm0, %ymm1, %ymm6
	vmovapd	64(%rdx), %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm1, %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm7, %ymm1, %ymm0
	vmovapd	96(%r9), %ymm7
	vfmadd231pd	%ymm2, %ymm7, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm7, %ymm6
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm7, %ymm11
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm0, %ymm7
	vmovapd	128(%rdx), %ymm2
	leaq	128(%rax), %rdx
	jle	.L445
	vblendpd	$1, %ymm14, %ymm1, %ymm14
	xorl	%edx, %edx
	vblendpd	$3, 160(%rax), %ymm1, %ymm3
	vshufpd	$5, %ymm2, %ymm2, %ymm0
	vmovapd	160(%rdx), %ymm9
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vmovapd	%ymm14, %ymm8
	vmovapd	%ymm14, %ymm5
	cmpl	$11, %r14d
	vmovapd	%ymm14, %ymm12
	leaq	256(%r9), %rdi
	vfmadd132pd	%ymm2, %ymm1, %ymm8
	vmovapd	160(%r9), %ymm2
	vfmadd231pd	%ymm9, %ymm3, %ymm8
	vfmadd231pd	%ymm0, %ymm15, %ymm6
	vfmadd132pd	%ymm0, %ymm1, %ymm5
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm9, %ymm2, %ymm4
	vshufpd	$5, %ymm9, %ymm9, %ymm9
	vfmadd231pd	%ymm0, %ymm15, %ymm11
	vfmadd132pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm9, %ymm3, %ymm5
	vmovapd	%ymm5, %ymm13
	vperm2f128	$1, %ymm9, %ymm9, %ymm5
	vfmadd231pd	%ymm9, %ymm2, %ymm6
	vfmadd231pd	%ymm0, %ymm15, %ymm7
	vfmadd132pd	%ymm14, %ymm1, %ymm0
	vblendpd	$7, 192(%rax), %ymm1, %ymm1
	vfmadd231pd	%ymm5, %ymm2, %ymm11
	vfmadd231pd	%ymm5, %ymm3, %ymm12
	vshufpd	$5, %ymm5, %ymm5, %ymm5
	vmovapd	256(%r9), %ymm15
	vmovapd	256(%rax), %ymm14
	vfmadd231pd	%ymm5, %ymm2, %ymm7
	vfmadd231pd	%ymm5, %ymm3, %ymm0
	vmovapd	192(%r9), %ymm2
	vmovapd	192(%rdx), %ymm3
	vfmadd231pd	%ymm3, %ymm2, %ymm4
	vfmadd231pd	%ymm3, %ymm1, %ymm8
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vmovapd	%ymm8, %ymm5
	vfmadd231pd	%ymm3, %ymm2, %ymm6
	vfmadd231pd	%ymm3, %ymm1, %ymm13
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm2, %ymm11
	vfmadd231pd	%ymm3, %ymm1, %ymm12
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm3, %ymm7, %ymm2
	vfmadd231pd	%ymm3, %ymm1, %ymm0
	vmovapd	224(%r9), %ymm7
	vmovapd	224(%rax), %ymm1
	vmovapd	224(%rdx), %ymm3
	vfmadd231pd	%ymm3, %ymm7, %ymm4
	vfmadd231pd	%ymm3, %ymm1, %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm7, %ymm6
	vfmadd231pd	%ymm3, %ymm1, %ymm13
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm7, %ymm11
	vfmadd231pd	%ymm3, %ymm1, %ymm12
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd132pd	%ymm3, %ymm2, %ymm7
	vfmadd231pd	%ymm3, %ymm1, %ymm0
	vmovapd	256(%rdx), %ymm2
	leaq	256(%rax), %rdx
	jg	.L496
	cmpl	$8, %r14d
	je	.L473
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	cmpl	$9, %r14d
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	288(%r9), %ymm15
	vmovapd	288(%rax), %ymm14
	vmovapd	288, %ymm2
	je	.L474
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	cmpl	$11, %r14d
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	320(%r9), %ymm15
	vmovapd	320(%rax), %ymm14
	vmovapd	320, %ymm2
	jne	.L475
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	movl	$11, -68(%rbp)
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	352(%r9), %ymm15
	vmovapd	352(%rax), %ymm14
	movq	%rdi, %r9
	movq	%rdx, %rax
	vmovapd	352, %ymm2
	jmp	.L443
	.p2align 4,,10
	.p2align 3
.L444:
	vxorpd	%xmm0, %xmm0, %xmm0
	cmpl	$1, %r14d
	vblendpd	$1, %ymm15, %ymm0, %ymm4
	vmovapd	32(%r9), %ymm15
	vfmadd132pd	%ymm2, %ymm0, %ymm4
	vmovapd	32, %ymm2
	je	.L479
	vblendpd	$3, %ymm15, %ymm0, %ymm15
	cmpl	$3, %r14d
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm15, %ymm6
	vmovapd	64(%r9), %ymm15
	vfmadd132pd	%ymm2, %ymm0, %ymm6
	vmovapd	64, %ymm2
	jne	.L480
	vblendpd	$7, %ymm15, %ymm0, %ymm7
	vmovapd	%ymm0, %ymm12
	vmovapd	96(%r9), %ymm15
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	movl	$3, -68(%rbp)
	vfmadd231pd	%ymm2, %ymm7, %ymm4
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vmovapd	%ymm7, %ymm11
	vfmadd231pd	%ymm2, %ymm7, %ymm6
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm0, %ymm11
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd132pd	%ymm2, %ymm0, %ymm7
	vmovapd	96, %ymm2
	jmp	.L443
.L492:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%rax, %r12
	vmovapd	%ymm1, %ymm8
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm3
	jmp	.L448
.L445:
	cmpl	$4, %r14d
	je	.L476
	vblendpd	$1, %ymm14, %ymm1, %ymm0
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	cmpl	$5, %r14d
	vmovapd	160(%rax), %ymm14
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm12
	vfmadd132pd	%ymm2, %ymm1, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd132pd	%ymm2, %ymm1, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd132pd	%ymm2, %ymm1, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd132pd	%ymm2, %ymm1, %ymm0
	vmovapd	160(%r9), %ymm15
	vmovapd	160, %ymm2
	je	.L477
	vblendpd	$3, %ymm14, %ymm1, %ymm14
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	cmpl	$7, %r14d
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vmovapd	192(%r9), %ymm15
	vmovapd	192(%rax), %ymm14
	vmovapd	192, %ymm2
	jne	.L478
	vblendpd	$7, %ymm14, %ymm1, %ymm1
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vmovapd	224(%rax), %ymm14
	movl	$7, -68(%rbp)
	movq	%rdx, %rax
	vfmadd231pd	%ymm2, %ymm1, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm1, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm1, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vfmadd231pd	%ymm2, %ymm1, %ymm0
	vmovapd	224(%r9), %ymm15
	vmovapd	224, %ymm2
	movq	%rdi, %r9
	jmp	.L443
.L480:
	vxorpd	%xmm0, %xmm0, %xmm0
	movl	$2, -68(%rbp)
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm11
	jmp	.L443
.L476:
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$4, -68(%rbp)
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	jmp	.L443
.L496:
	vfmadd231pd	%ymm2, %ymm15, %ymm4
	vfmadd231pd	%ymm2, %ymm14, %ymm5
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	xorl	%edx, %edx
	vmovapd	288(%r9), %ymm1
	addq	$384, %rax
	vmovapd	288(%rdx), %ymm3
	addq	$384, %r9
	movl	$12, -68(%rbp)
	vfmadd231pd	%ymm2, %ymm15, %ymm6
	vfmadd231pd	%ymm2, %ymm14, %ymm13
	vperm2f128	$1, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm3, %ymm1, %ymm4
	vfmadd231pd	%ymm2, %ymm15, %ymm11
	vfmadd231pd	%ymm2, %ymm14, %ymm12
	vshufpd	$5, %ymm2, %ymm2, %ymm2
	vfmadd231pd	%ymm2, %ymm14, %ymm0
	vfmadd231pd	%ymm2, %ymm15, %ymm7
	vmovapd	-96(%rax), %ymm2
	vmovapd	(%r9), %ymm15
	vfmadd231pd	%ymm3, %ymm2, %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vmovapd	(%rax), %ymm14
	vfmadd231pd	%ymm3, %ymm1, %ymm6
	vfmadd231pd	%ymm3, %ymm2, %ymm13
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm1, %ymm11
	vfmadd231pd	%ymm3, %ymm2, %ymm12
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm1, %ymm7
	vfmadd132pd	%ymm3, %ymm0, %ymm2
	vmovapd	-64(%r9), %ymm1
	vmovapd	-64(%rax), %ymm0
	vmovapd	320(%rdx), %ymm3
	vfmadd231pd	%ymm3, %ymm1, %ymm4
	vfmadd231pd	%ymm3, %ymm0, %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vfmadd231pd	%ymm3, %ymm1, %ymm6
	vfmadd231pd	%ymm3, %ymm0, %ymm13
	vperm2f128	$1, %ymm3, %ymm3, %ymm3
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vfmadd231pd	%ymm3, %ymm1, %ymm11
	vfmadd231pd	%ymm3, %ymm0, %ymm12
	vmovapd	%ymm1, %ymm3
	vmovapd	-32(%rax), %ymm1
	vfmadd231pd	%ymm8, %ymm0, %ymm2
	vfmadd132pd	%ymm8, %ymm7, %ymm3
	vmovapd	352(%rdx), %ymm0
	vmovapd	-32(%r9), %ymm7
	vfmadd231pd	%ymm0, %ymm1, %ymm5
	vfmadd231pd	%ymm0, %ymm7, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm13
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm7, %ymm11
	vfmadd231pd	%ymm0, %ymm1, %ymm12
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm3, %ymm7
	vfmadd132pd	%ymm1, %ymm2, %ymm0
	vmovapd	384(%rdx), %ymm2
	jmp	.L443
.L479:
	vxorpd	%xmm0, %xmm0, %xmm0
	movl	$1, -68(%rbp)
	vmovapd	%ymm0, %ymm12
	vmovapd	%ymm0, %ymm13
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm11
	vmovapd	%ymm0, %ymm6
	jmp	.L443
.L477:
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$5, -68(%rbp)
	jmp	.L443
.L473:
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$8, -68(%rbp)
	jmp	.L443
.L474:
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$9, -68(%rbp)
	jmp	.L443
.L478:
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$6, -68(%rbp)
	jmp	.L443
.L475:
	movq	%rdx, %rax
	movq	%rdi, %r9
	movl	$10, -68(%rbp)
	jmp	.L443
	.cfi_endproc
.LFE4600:
	.size	kernel_dsyrk_dpotrf_nt_8x8_vs_lib4, .-kernel_dsyrk_dpotrf_nt_8x8_vs_lib4
	.section	.text.unlikely
.LCOLDE22:
	.text
.LHOTE22:
	.section	.text.unlikely
.LCOLDB23:
	.text
.LHOTB23:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_8x4_vs_lib4:
.LFB4601:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	16(%r10), %rbx
	movl	%r8d, -68(%rbp)
	movl	64(%r10), %r8d
	movl	80(%r10), %r15d
	movl	(%r10), %eax
	movq	8(%r10), %r14
	movq	%rbx, -80(%rbp)
	movl	24(%r10), %ebx
	sall	$2, %r8d
	cmpl	$7, %edi
	movq	32(%r10), %rsi
	movslq	%r8d, %r8
	movq	72(%r10), %r11
	movl	%r15d, -72(%rbp)
	movl	%ebx, -84(%rbp)
	movq	40(%r10), %rbx
	movq	%rbx, -96(%rbp)
	movl	48(%r10), %ebx
	movl	%ebx, -88(%rbp)
	movq	56(%r10), %rbx
	leaq	(%rbx,%r8,8), %r12
	jle	.L498
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%ecx, %ecx
	vmovdqu	%ymm0, mask_bkp.28114(%rip)
	jle	.L522
.L537:
	sall	$2, %eax
	cmpl	$1, %edx
	vmovapd	(%r9), %ymm11
	cltq
	vmovapd	(%r14), %ymm3
	leaq	(%r9,%rax,8), %r8
	vmovapd	(%r8), %ymm6
	je	.L535
	vxorpd	%xmm1, %xmm1, %xmm1
	xorl	%r10d, %r10d
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm2
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm10
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm4
.L501:
	leal	-3(%rcx), %eax
	cmpl	%eax, %r10d
	jge	.L505
	leal	-4(%rcx), %edx
	leaq	32(%r14), %rax
	movq	%r8, %rdi
	subl	%r10d, %edx
	shrl	$2, %edx
	movl	%edx, %r15d
	movl	%edx, -100(%rbp)
	movq	%r15, %rdx
	salq	$7, %rdx
	leaq	160(%r14,%rdx), %r13
	movq	%r9, %rdx
	.p2align 4,,10
	.p2align 3
.L506:
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	vmovapd	(%rax), %ymm3
	subq	$-128, %rax
	subq	$-128, %rdx
	vmovapd	-96(%rdx), %ymm13
	subq	$-128, %rdi
	vperm2f128	$1, %ymm8, %ymm8, %ymm12
	vfmadd231pd	%ymm8, %ymm11, %ymm5
	vfmadd231pd	%ymm8, %ymm6, %ymm7
	vfmadd231pd	%ymm3, %ymm13, %ymm4
	vmovapd	-64(%rdi), %ymm15
	vshufpd	$5, %ymm12, %ymm12, %ymm8
	vfmadd231pd	%ymm12, %ymm11, %ymm10
	vfmadd132pd	%ymm6, %ymm0, %ymm12
	vfmadd231pd	%ymm8, %ymm6, %ymm1
	vfmadd231pd	%ymm8, %ymm11, %ymm9
	vmovapd	-96(%rdi), %ymm6
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vmovapd	(%rdx), %ymm11
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	vmovapd	-96(%rax), %ymm3
	vperm2f128	$1, %ymm8, %ymm8, %ymm0
	vfmadd231pd	%ymm8, %ymm13, %ymm5
	vfmadd231pd	%ymm8, %ymm6, %ymm7
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vshufpd	$5, %ymm0, %ymm0, %ymm14
	vfmadd231pd	%ymm0, %ymm13, %ymm10
	vfmadd231pd	%ymm0, %ymm6, %ymm12
	vperm2f128	$1, %ymm8, %ymm8, %ymm0
	vfmadd231pd	%ymm8, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm13, %ymm9
	vmovapd	-64(%rdx), %ymm13
	vfmadd132pd	%ymm14, %ymm1, %ymm6
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm15, %ymm12
	vfmadd231pd	%ymm3, %ymm13, %ymm4
	vfmadd132pd	%ymm15, %ymm2, %ymm3
	vmovapd	-64(%rax), %ymm2
	vfmadd231pd	%ymm0, %ymm13, %ymm10
	vfmadd231pd	%ymm8, %ymm13, %ymm5
	vfmadd132pd	%ymm1, %ymm6, %ymm15
	vshufpd	$5, %ymm2, %ymm2, %ymm6
	vfmadd132pd	%ymm1, %ymm9, %ymm13
	vmovapd	-32(%rdx), %ymm9
	vmovapd	-32(%rdi), %ymm1
	vperm2f128	$1, %ymm6, %ymm6, %ymm0
	vfmadd231pd	%ymm6, %ymm9, %ymm5
	vfmadd231pd	%ymm2, %ymm9, %ymm4
	vfmadd231pd	%ymm6, %ymm1, %ymm7
	vfmadd132pd	%ymm1, %ymm3, %ymm2
	vmovapd	-32(%rax), %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm6
	vfmadd231pd	%ymm0, %ymm9, %ymm10
	vfmadd132pd	%ymm1, %ymm12, %ymm0
	vfmadd132pd	%ymm6, %ymm13, %ymm9
	vfmadd132pd	%ymm6, %ymm15, %ymm1
	vmovapd	(%rdi), %ymm6
	cmpq	%r13, %rax
	jne	.L506
	movl	-100(%rbp), %eax
	addq	$1, %r15
	salq	$7, %r15
	addq	%r15, %r9
	addq	%r15, %r8
	addq	%r15, %r14
	leal	4(%r10,%rax,4), %r10d
.L505:
	leal	-1(%rcx), %eax
	cmpl	%eax, %r10d
	jge	.L507
	leal	-2(%rcx), %edx
	leaq	32(%r14), %rax
	subl	%r10d, %edx
	shrl	%edx
	movl	%edx, %edi
	salq	$6, %rdi
	leaq	96(%r14,%rdi), %rdi
	.p2align 4,,10
	.p2align 3
.L508:
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	vmovapd	(%rax), %ymm8
	addq	$64, %rax
	addq	$64, %r9
	addq	$64, %r8
	vperm2f128	$1, %ymm12, %ymm12, %ymm3
	vfmadd231pd	%ymm12, %ymm11, %ymm5
	vfmadd231pd	%ymm12, %ymm6, %ymm7
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm10
	vfmadd231pd	%ymm3, %ymm6, %ymm0
	vmovapd	-32(%rax), %ymm3
	vfmadd132pd	%ymm12, %ymm9, %ymm11
	vfmadd132pd	%ymm12, %ymm1, %ymm6
	vshufpd	$5, %ymm8, %ymm8, %ymm12
	vmovapd	%ymm6, %ymm13
	vmovapd	-32(%r9), %ymm9
	vmovapd	-32(%r8), %ymm1
	vperm2f128	$1, %ymm12, %ymm12, %ymm6
	vfmadd231pd	%ymm8, %ymm9, %ymm4
	vfmadd231pd	%ymm12, %ymm9, %ymm5
	vfmadd231pd	%ymm8, %ymm1, %ymm2
	vfmadd231pd	%ymm12, %ymm1, %ymm7
	vshufpd	$5, %ymm6, %ymm6, %ymm8
	vfmadd231pd	%ymm6, %ymm9, %ymm10
	vfmadd231pd	%ymm6, %ymm1, %ymm0
	vmovapd	(%r8), %ymm6
	vfmadd132pd	%ymm8, %ymm11, %ymm9
	vmovapd	(%r9), %ymm11
	cmpq	%rdi, %rax
	vfmadd132pd	%ymm8, %ymm13, %ymm1
	jne	.L508
	leal	2(%r10,%rdx,2), %r10d
.L507:
	cmpl	%r10d, %ecx
	jle	.L500
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vperm2f128	$1, %ymm12, %ymm12, %ymm8
	vshufpd	$5, %ymm8, %ymm8, %ymm13
	.p2align 4,,10
	.p2align 3
.L509:
	addl	$1, %r10d
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	cmpl	%r10d, %ecx
	vfmadd231pd	%ymm12, %ymm11, %ymm5
	vfmadd231pd	%ymm12, %ymm6, %ymm7
	vfmadd231pd	%ymm8, %ymm11, %ymm10
	vfmadd231pd	%ymm8, %ymm6, %ymm0
	vfmadd231pd	%ymm13, %ymm11, %ymm9
	vfmadd231pd	%ymm13, %ymm6, %ymm1
	jne	.L509
.L500:
	movl	-68(%rbp), %ecx
	testl	%ecx, %ecx
	jle	.L510
	movl	-84(%rbp), %eax
	movq	-80(%rbp), %rdi
	vmovapd	(%rsi), %ymm6
	sall	$2, %eax
	cmpl	$3, %ecx
	vmovapd	(%rdi), %ymm11
	cltq
	leaq	(%rdi,%rax,8), %rdx
	vmovapd	(%rdx), %ymm8
	jle	.L510
	subl	$4, %ecx
	leaq	32(%rdi), %rax
	shrl	$2, %ecx
	salq	$7, %rcx
	leaq	160(%rdi,%rcx), %rcx
	.p2align 4,,10
	.p2align 3
.L511:
	vshufpd	$5, %ymm6, %ymm6, %ymm3
	vfmadd231pd	%ymm6, %ymm11, %ymm4
	vfmadd231pd	%ymm6, %ymm8, %ymm2
	vmovapd	32(%rsi), %ymm6
	subq	$-128, %rax
	subq	$-128, %rdx
	vmovapd	-64(%rdx), %ymm15
	subq	$-128, %rsi
	vperm2f128	$1, %ymm3, %ymm3, %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm5
	vfmadd231pd	%ymm3, %ymm8, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm3
	vfmadd231pd	%ymm12, %ymm11, %ymm10
	vfmadd231pd	%ymm12, %ymm8, %ymm0
	vmovapd	-128(%rax), %ymm12
	vfmadd231pd	%ymm3, %ymm11, %ymm9
	vshufpd	$5, %ymm6, %ymm6, %ymm11
	vfmadd231pd	%ymm3, %ymm8, %ymm1
	vmovapd	-96(%rdx), %ymm3
	vfmadd231pd	%ymm6, %ymm12, %ymm4
	vperm2f128	$1, %ymm11, %ymm11, %ymm13
	vfmadd231pd	%ymm6, %ymm3, %ymm2
	vmovapd	-64(%rsi), %ymm6
	vfmadd231pd	%ymm11, %ymm12, %ymm5
	vfmadd231pd	%ymm11, %ymm3, %ymm7
	vshufpd	$5, %ymm6, %ymm6, %ymm11
	vshufpd	$5, %ymm13, %ymm13, %ymm14
	vfmadd231pd	%ymm13, %ymm12, %ymm10
	vfmadd231pd	%ymm13, %ymm3, %ymm0
	vmovapd	-96(%rax), %ymm13
	vfmadd231pd	%ymm11, %ymm15, %ymm7
	vfmadd231pd	%ymm14, %ymm12, %ymm9
	vperm2f128	$1, %ymm11, %ymm11, %ymm12
	vfmadd231pd	%ymm6, %ymm13, %ymm4
	vfmadd132pd	%ymm15, %ymm2, %ymm6
	vmovapd	-32(%rsi), %ymm2
	vfmadd132pd	%ymm14, %ymm1, %ymm3
	vfmadd231pd	%ymm11, %ymm13, %ymm5
	vmovapd	-32(%rax), %ymm11
	vshufpd	$5, %ymm2, %ymm2, %ymm8
	vfmadd231pd	%ymm12, %ymm15, %ymm0
	vfmadd231pd	%ymm12, %ymm13, %ymm10
	vshufpd	$5, %ymm12, %ymm12, %ymm1
	vfmadd132pd	%ymm1, %ymm3, %ymm15
	vperm2f128	$1, %ymm8, %ymm8, %ymm3
	vfmadd132pd	%ymm1, %ymm9, %ymm13
	vmovapd	-64(%rax), %ymm9
	vmovapd	-32(%rdx), %ymm1
	vfmadd231pd	%ymm8, %ymm9, %ymm5
	vfmadd231pd	%ymm2, %ymm9, %ymm4
	vfmadd231pd	%ymm3, %ymm9, %ymm10
	vfmadd231pd	%ymm8, %ymm1, %ymm7
	vshufpd	$5, %ymm3, %ymm3, %ymm8
	vfmadd132pd	%ymm1, %ymm6, %ymm2
	vfmadd231pd	%ymm3, %ymm1, %ymm0
	vmovapd	(%rsi), %ymm6
	vfmadd132pd	%ymm8, %ymm13, %ymm9
	vfmadd132pd	%ymm8, %ymm15, %ymm1
	vmovapd	(%rdx), %ymm8
	cmpq	%rax, %rcx
	jne	.L511
.L510:
	movl	-72(%rbp), %eax
	vblendpd	$10, %ymm5, %ymm4, %ymm3
	vblendpd	$10, %ymm7, %ymm2, %ymm12
	vblendpd	$5, %ymm5, %ymm4, %ymm4
	vblendpd	$10, %ymm10, %ymm9, %ymm6
	vblendpd	$5, %ymm7, %ymm2, %ymm2
	vblendpd	$5, %ymm10, %ymm9, %ymm9
	testl	%eax, %eax
	vblendpd	$10, %ymm0, %ymm1, %ymm7
	vblendpd	$5, %ymm0, %ymm1, %ymm5
	jne	.L512
	vblendpd	$12, %ymm6, %ymm3, %ymm8
	vblendpd	$12, %ymm7, %ymm12, %ymm13
	vblendpd	$3, %ymm6, %ymm3, %ymm3
	vblendpd	$12, %ymm5, %ymm2, %ymm11
	vblendpd	$12, %ymm9, %ymm4, %ymm6
	vblendpd	$3, %ymm7, %ymm12, %ymm12
	vblendpd	$3, %ymm9, %ymm4, %ymm9
	vblendpd	$3, %ymm5, %ymm2, %ymm5
.L513:
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovsd	.LC0(%rip), %xmm14
	vmovsd	%xmm8, %xmm2, %xmm0
	vucomisd	%xmm14, %xmm0
	jbe	.L514
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vdivsd	%xmm0, %xmm1, %xmm0
	vmovlpd	%xmm0, (%r11)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.28114(%rip), %ymm1
	vmulpd	%ymm0, %ymm8, %ymm8
	vmulpd	%ymm0, %ymm13, %ymm13
	vmovapd	%ymm8, (%rbx)
	vmaskmovpd	%ymm13, %ymm1, (%r12)
.L515:
	vpermpd	$85, %ymm8, %ymm0
	vmovapd	%ymm6, %ymm4
	vfmadd231pd	%ymm0, %ymm8, %ymm4
	vpermilpd	$3, %xmm4, %xmm2
	vmovlpd	%xmm0, 8(%r11)
	vfmadd231pd	%ymm0, %ymm13, %ymm11
	vucomisd	%xmm14, %xmm2
	jbe	.L516
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm2, %xmm2, %xmm2
	vmovdqa	.LC2(%rip), %ymm1
	vdivsd	%xmm2, %xmm0, %xmm2
	vmovlpd	%xmm2, 16(%r11)
	vbroadcastsd	%xmm2, %ymm2
	vmovdqu	mask_bkp.28114(%rip), %ymm0
	vmulpd	%ymm4, %ymm2, %ymm6
	vmulpd	%ymm2, %ymm11, %ymm11
	vmaskmovpd	%ymm6, %ymm1, 32(%rbx)
	vmaskmovpd	%ymm11, %ymm0, 32(%r12)
.L517:
	vpermpd	$170, %ymm8, %ymm1
	vfmadd231pd	%ymm1, %ymm8, %ymm3
	vmovlpd	%xmm1, 24(%r11)
	vfmadd132pd	%ymm13, %ymm12, %ymm1
	vpermpd	$170, %ymm6, %ymm12
	vfmadd231pd	%ymm12, %ymm6, %ymm3
	vextractf128	$0x1, %ymm3, %xmm0
	vmovlpd	%xmm12, 32(%r11)
	vfmadd132pd	%ymm11, %ymm1, %ymm12
	vucomisd	%xmm14, %xmm0
	jbe	.L518
	vsqrtsd	%xmm0, %xmm0, %xmm1
	vmovsd	.LC1(%rip), %xmm0
	vmovdqa	.LC3(%rip), %ymm2
	vdivsd	%xmm1, %xmm0, %xmm0
	vmovlpd	%xmm0, 40(%r11)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.28114(%rip), %ymm1
	vmulpd	%ymm3, %ymm0, %ymm3
	vmulpd	%ymm0, %ymm12, %ymm12
	vmaskmovpd	%ymm3, %ymm2, 64(%rbx)
	vmaskmovpd	%ymm12, %ymm1, 64(%r12)
.L519:
	vpermpd	$255, %ymm8, %ymm15
	vmovapd	%ymm8, %ymm4
	vpermpd	$255, %ymm3, %ymm0
	vfmadd132pd	%ymm15, %ymm5, %ymm13
	vpermpd	$255, %ymm6, %ymm5
	vfmadd132pd	%ymm15, %ymm9, %ymm4
	vmovlpd	%xmm0, 64(%r11)
	vmovlpd	%xmm15, 48(%r11)
	vfmadd231pd	%ymm5, %ymm6, %ymm4
	vfmadd132pd	%ymm5, %ymm13, %ymm11
	vfmadd231pd	%ymm0, %ymm3, %ymm4
	vfmadd132pd	%ymm0, %ymm11, %ymm12
	vextractf128	$0x1, %ymm4, %xmm0
	vmovlpd	%xmm5, 56(%r11)
	vpermilpd	$3, %xmm0, %xmm0
	vucomisd	%xmm14, %xmm0
	ja	.L536
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovdqa	.LC4(%rip), %ymm1
	vmovdqu	mask_bkp.28114(%rip), %ymm0
	movq	$0, 72(%r11)
	vmaskmovpd	%ymm2, %ymm1, 96(%rbx)
	vmaskmovpd	%ymm12, %ymm0, 96(%r12)
.L533:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L498:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.28115(%rip), %ymm1
	testl	%ecx, %ecx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	.LC11(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.28114(%rip)
	jg	.L537
.L522:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm2
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm10
	vmovapd	%ymm1, %ymm5
	vmovapd	%ymm1, %ymm4
	jmp	.L500
	.p2align 4,,10
	.p2align 3
.L518:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovdqu	mask_bkp.28114(%rip), %ymm0
	movq	$0, 40(%r11)
	vblendpd	$7, %ymm1, %ymm3, %ymm3
	vmovdqa	.LC3(%rip), %ymm1
	vmaskmovpd	%ymm3, %ymm1, 64(%rbx)
	vmaskmovpd	%ymm12, %ymm0, 64(%r12)
	jmp	.L519
	.p2align 4,,10
	.p2align 3
.L516:
	vxorpd	%xmm6, %xmm6, %xmm6
	vmovdqa	.LC2(%rip), %ymm1
	vmovdqu	mask_bkp.28114(%rip), %ymm0
	movq	$0, 16(%r11)
	vblendpd	$3, %ymm6, %ymm4, %ymm6
	vmaskmovpd	%ymm6, %ymm1, 32(%rbx)
	vmaskmovpd	%ymm11, %ymm0, 32(%r12)
	jmp	.L517
	.p2align 4,,10
	.p2align 3
.L514:
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovdqu	mask_bkp.28114(%rip), %ymm0
	movq	$0, (%r11)
	vblendpd	$1, %ymm4, %ymm8, %ymm8
	vmovapd	%ymm8, (%rbx)
	vmaskmovpd	%ymm13, %ymm0, (%r12)
	jmp	.L515
	.p2align 4,,10
	.p2align 3
.L512:
	movl	-88(%rbp), %eax
	movq	-96(%rbp), %rcx
	vblendpd	$12, %ymm6, %ymm3, %ymm8
	vblendpd	$12, %ymm7, %ymm12, %ymm13
	vblendpd	$3, %ymm6, %ymm3, %ymm3
	vblendpd	$12, %ymm5, %ymm2, %ymm11
	vblendpd	$12, %ymm9, %ymm4, %ymm6
	sall	$2, %eax
	vblendpd	$3, %ymm9, %ymm4, %ymm4
	vblendpd	$3, %ymm7, %ymm12, %ymm12
	cltq
	vblendpd	$3, %ymm5, %ymm2, %ymm5
	leaq	(%rcx,%rax,8), %rax
	vaddpd	(%rcx), %ymm8, %ymm8
	vaddpd	64(%rcx), %ymm3, %ymm3
	vaddpd	32(%rcx), %ymm6, %ymm6
	vaddpd	96(%rcx), %ymm4, %ymm9
	vaddpd	(%rax), %ymm13, %ymm13
	vaddpd	64(%rax), %ymm12, %ymm12
	vaddpd	32(%rax), %ymm11, %ymm11
	vaddpd	96(%rax), %ymm5, %ymm5
	jmp	.L513
	.p2align 4,,10
	.p2align 3
.L536:
	vmovsd	.LC1(%rip), %xmm1
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vmovdqa	.LC4(%rip), %ymm2
	vdivsd	%xmm0, %xmm1, %xmm0
	vmovlpd	%xmm0, 72(%r11)
	vbroadcastsd	%xmm0, %ymm0
	vmovdqu	mask_bkp.28114(%rip), %ymm1
	vmulpd	%ymm4, %ymm0, %ymm4
	vmulpd	%ymm12, %ymm0, %ymm0
	vmaskmovpd	%ymm4, %ymm2, 96(%rbx)
	vmaskmovpd	%ymm0, %ymm1, 96(%r12)
	jmp	.L533
	.p2align 4,,10
	.p2align 3
.L535:
	cmpl	$3, %ecx
	jle	.L502
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	32(%r14), %ymm0
	cmpl	$7, %ecx
	vmovapd	96(%r14), %ymm1
	leaq	128(%r9), %rdi
	leaq	128(%r8), %rax
	vmovapd	128(%r8), %ymm6
	leaq	128(%r14), %rdx
	vblendpd	$1, %ymm11, %ymm8, %ymm4
	vblendpd	$3, 32(%r9), %ymm8, %ymm5
	vmovapd	128(%r9), %ymm11
	vblendpd	$7, 64(%r9), %ymm8, %ymm9
	vfmadd132pd	%ymm3, %ymm8, %ymm4
	vfmadd231pd	%ymm0, %ymm5, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vmovapd	%ymm9, %ymm10
	vmovapd	128(%r14), %ymm3
	vfmadd132pd	%ymm0, %ymm8, %ymm5
	vblendpd	$7, 64(%r14), %ymm8, %ymm0
	vfmadd231pd	%ymm0, %ymm9, %ymm4
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd231pd	%ymm0, %ymm9, %ymm5
	vperm2f128	$1, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm0, %ymm8, %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vfmadd132pd	%ymm9, %ymm8, %ymm0
	vmovapd	96(%r9), %ymm9
	vfmadd231pd	%ymm1, %ymm9, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm9, %ymm5
	vperm2f128	$1, %ymm1, %ymm1, %ymm1
	vfmadd231pd	%ymm1, %ymm9, %ymm10
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vfmadd132pd	%ymm1, %ymm0, %ymm9
	jle	.L503
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vmovapd	160(%r9), %ymm13
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vmovapd	160(%r14), %ymm0
	vblendpd	$1, %ymm6, %ymm8, %ymm6
	vblendpd	$3, 160(%r8), %ymm8, %ymm14
	addq	$256, %r9
	addq	$256, %r8
	addq	$256, %r14
	vperm2f128	$1, %ymm12, %ymm12, %ymm7
	vfmadd231pd	%ymm0, %ymm13, %ymm4
	vmovapd	%ymm4, %ymm15
	vshufpd	$5, %ymm0, %ymm0, %ymm4
	vfmadd132pd	%ymm6, %ymm8, %ymm3
	vfmadd231pd	%ymm12, %ymm11, %ymm5
	vfmadd132pd	%ymm6, %ymm8, %ymm12
	vmovapd	%ymm0, %ymm2
	vmovapd	-64(%r14), %ymm0
	vshufpd	$5, %ymm7, %ymm7, %ymm1
	vfmadd231pd	%ymm7, %ymm11, %ymm10
	vfmadd132pd	%ymm6, %ymm8, %ymm7
	vfmadd132pd	%ymm14, %ymm3, %ymm2
	vfmadd231pd	%ymm4, %ymm13, %ymm5
	vfmadd231pd	%ymm4, %ymm14, %ymm12
	movl	$8, %r10d
	vfmadd132pd	%ymm1, %ymm8, %ymm6
	vfmadd231pd	%ymm1, %ymm11, %ymm9
	vperm2f128	$1, %ymm4, %ymm4, %ymm1
	vshufpd	$5, %ymm0, %ymm0, %ymm11
	vblendpd	$7, -64(%r8), %ymm8, %ymm8
	vmovapd	%ymm15, %ymm4
	vshufpd	$5, %ymm1, %ymm1, %ymm3
	vfmadd231pd	%ymm1, %ymm13, %ymm10
	vfmadd231pd	%ymm1, %ymm14, %ymm7
	vmovapd	-64(%r9), %ymm1
	vfmadd231pd	%ymm11, %ymm8, %ymm12
	vfmadd231pd	%ymm3, %ymm13, %ymm9
	vfmadd132pd	%ymm14, %ymm6, %ymm3
	vperm2f128	$1, %ymm11, %ymm11, %ymm6
	vfmadd231pd	%ymm0, %ymm1, %ymm4
	vfmadd231pd	%ymm11, %ymm1, %ymm5
	vfmadd132pd	%ymm8, %ymm2, %ymm0
	vmovapd	-32(%r14), %ymm2
	vshufpd	$5, %ymm6, %ymm6, %ymm11
	vfmadd231pd	%ymm6, %ymm1, %ymm10
	vfmadd132pd	%ymm8, %ymm7, %ymm6
	vshufpd	$5, %ymm2, %ymm2, %ymm7
	vfmadd132pd	%ymm11, %ymm9, %ymm1
	vmovapd	-32(%r9), %ymm9
	vmovapd	%ymm1, %ymm13
	vmovapd	-32(%r8), %ymm1
	vfmadd132pd	%ymm11, %ymm3, %ymm8
	vfmadd231pd	%ymm2, %ymm9, %ymm4
	vfmadd231pd	%ymm7, %ymm9, %ymm5
	vmovapd	(%r14), %ymm3
	vfmadd132pd	%ymm1, %ymm0, %ymm2
	vperm2f128	$1, %ymm7, %ymm7, %ymm0
	vfmadd132pd	%ymm1, %ymm12, %ymm7
	vmovapd	(%r9), %ymm11
	vshufpd	$5, %ymm0, %ymm0, %ymm12
	vfmadd231pd	%ymm0, %ymm9, %ymm10
	vfmadd132pd	%ymm1, %ymm6, %ymm0
	vmovapd	(%r8), %ymm6
	vfmadd132pd	%ymm12, %ymm13, %ymm9
	vfmadd132pd	%ymm12, %ymm8, %ymm1
	jmp	.L501
	.p2align 4,,10
	.p2align 3
.L502:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	32(%r14), %ymm3
	cmpl	$1, %ecx
	vblendpd	$1, %ymm11, %ymm0, %ymm4
	vmovapd	32(%r9), %ymm11
	vfmadd132pd	%ymm3, %ymm0, %ymm4
	je	.L527
	vshufpd	$5, %ymm3, %ymm3, %ymm5
	vblendpd	$3, %ymm11, %ymm0, %ymm11
	vmovapd	64(%r14), %ymm3
	cmpl	$3, %ecx
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vfmadd132pd	%ymm11, %ymm0, %ymm5
	vmovapd	64(%r9), %ymm11
	jne	.L528
	vshufpd	$5, %ymm3, %ymm3, %ymm2
	vblendpd	$7, %ymm11, %ymm0, %ymm10
	vmovapd	96(%r14), %ymm3
	vmovapd	96(%r9), %ymm11
	vmovapd	%ymm0, %ymm7
	movl	$3, %r10d
	vfmadd231pd	%ymm3, %ymm10, %ymm4
	vperm2f128	$1, %ymm2, %ymm2, %ymm1
	vfmadd231pd	%ymm2, %ymm10, %ymm5
	vmovapd	%ymm0, %ymm2
	vshufpd	$5, %ymm1, %ymm1, %ymm9
	vfmadd132pd	%ymm1, %ymm0, %ymm10
	vmovapd	%ymm0, %ymm1
	vfmadd132pd	%ymm11, %ymm0, %ymm9
	jmp	.L501
.L503:
	cmpl	$4, %ecx
	je	.L524
	vshufpd	$5, %ymm3, %ymm3, %ymm7
	vblendpd	$1, %ymm6, %ymm8, %ymm6
	cmpl	$5, %ecx
	vmovapd	%ymm3, %ymm2
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vmovapd	160(%r14), %ymm3
	vperm2f128	$1, %ymm7, %ymm7, %ymm0
	vfmadd231pd	%ymm7, %ymm11, %ymm5
	vfmadd132pd	%ymm6, %ymm8, %ymm2
	vfmadd132pd	%ymm6, %ymm8, %ymm7
	vshufpd	$5, %ymm0, %ymm0, %ymm1
	vfmadd231pd	%ymm0, %ymm11, %ymm10
	vfmadd132pd	%ymm6, %ymm8, %ymm0
	vfmadd231pd	%ymm1, %ymm11, %ymm9
	vfmadd132pd	%ymm6, %ymm8, %ymm1
	vmovapd	160(%r9), %ymm11
	vmovapd	160(%r8), %ymm6
	je	.L525
	vshufpd	$5, %ymm3, %ymm3, %ymm13
	vblendpd	$3, %ymm6, %ymm8, %ymm6
	cmpl	$7, %ecx
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	vmovapd	192(%r14), %ymm3
	vperm2f128	$1, %ymm13, %ymm13, %ymm12
	vfmadd231pd	%ymm13, %ymm11, %ymm5
	vfmadd231pd	%ymm13, %ymm6, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm13
	vfmadd231pd	%ymm12, %ymm11, %ymm10
	vfmadd231pd	%ymm12, %ymm6, %ymm0
	vfmadd231pd	%ymm13, %ymm11, %ymm9
	vfmadd231pd	%ymm13, %ymm6, %ymm1
	vmovapd	192(%r9), %ymm11
	vmovapd	192(%r8), %ymm6
	jne	.L526
	vshufpd	$5, %ymm3, %ymm3, %ymm12
	vblendpd	$7, %ymm6, %ymm8, %ymm6
	vfmadd231pd	%ymm3, %ymm11, %ymm4
	movl	$7, %r10d
	vfmadd231pd	%ymm3, %ymm6, %ymm2
	vmovapd	224(%r14), %ymm3
	movq	%rdx, %r14
	vperm2f128	$1, %ymm12, %ymm12, %ymm8
	vfmadd231pd	%ymm12, %ymm11, %ymm5
	vfmadd231pd	%ymm12, %ymm6, %ymm7
	vshufpd	$5, %ymm8, %ymm8, %ymm12
	vfmadd231pd	%ymm8, %ymm11, %ymm10
	vfmadd231pd	%ymm8, %ymm6, %ymm0
	vfmadd231pd	%ymm12, %ymm11, %ymm9
	vfmadd231pd	%ymm12, %ymm6, %ymm1
	vmovapd	224(%r9), %ymm11
	vmovapd	224(%r8), %ymm6
	movq	%rdi, %r9
	movq	%rax, %r8
	jmp	.L501
.L528:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$2, %r10d
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm2
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm10
	jmp	.L501
.L524:
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	%rax, %r8
	movq	%rdx, %r14
	movq	%rdi, %r9
	movl	$4, %r10d
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm2
	jmp	.L501
.L527:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$1, %r10d
	vmovapd	%ymm1, %ymm0
	vmovapd	%ymm1, %ymm7
	vmovapd	%ymm1, %ymm2
	vmovapd	%ymm1, %ymm9
	vmovapd	%ymm1, %ymm10
	vmovapd	%ymm1, %ymm5
	jmp	.L501
.L525:
	movq	%rax, %r8
	movq	%rdx, %r14
	movq	%rdi, %r9
	movl	$5, %r10d
	jmp	.L501
.L526:
	movq	%rax, %r8
	movq	%rdx, %r14
	movq	%rdi, %r9
	movl	$6, %r10d
	jmp	.L501
	.cfi_endproc
.LFE4601:
	.size	kernel_dsyrk_dpotrf_nt_8x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_8x4_vs_lib4
	.section	.text.unlikely
.LCOLDE23:
	.text
.LHOTE23:
	.section	.text.unlikely
.LCOLDB24:
	.text
.LHOTB24:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_4x4_vs_lib4:
.LFB4602:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$3, %edi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	(%r10), %rsi
	movq	8(%r10), %r13
	movq	16(%r10), %rax
	movq	24(%r10), %r12
	movq	32(%r10), %rbx
	movq	40(%r10), %r11
	movl	48(%r10), %r10d
	jle	.L539
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%ecx, %ecx
	vmovdqu	%ymm0, mask_bkp.28217(%rip)
	jle	.L561
.L577:
	cmpl	$1, %edx
	vmovapd	(%r9), %ymm1
	vbroadcastf128	(%rsi), %ymm11
	vbroadcastf128	16(%rsi), %ymm12
	je	.L575
	vxorpd	%xmm3, %xmm3, %xmm3
	xorl	%r14d, %r14d
	vmovapd	%ymm3, %ymm0
	vmovapd	%ymm3, %ymm5
	vmovapd	%ymm3, %ymm4
.L542:
	leal	-3(%rcx), %edx
	cmpl	%edx, %r14d
	jge	.L565
	leal	-4(%rcx), %edx
	vxorpd	%xmm8, %xmm8, %xmm8
	leaq	32(%r9), %rdi
	subl	%r14d, %edx
	shrl	$2, %edx
	vmovapd	%ymm8, %ymm6
	vmovapd	%ymm8, %ymm7
	movl	%edx, -68(%rbp)
	movq	%rdx, -80(%rbp)
	vmovapd	%ymm8, %ymm2
	salq	$7, %rdx
	leaq	160(%r9,%rdx), %r15
	movq	%rsi, %rdx
	.p2align 4,,10
	.p2align 3
.L546:
	vshufpd	$5, %ymm11, %ymm11, %ymm13
	vbroadcastf128	32(%rdx), %ymm10
	vfmadd231pd	%ymm11, %ymm1, %ymm4
	vshufpd	$5, %ymm12, %ymm12, %ymm11
	vmovapd	(%rdi), %ymm9
	vfmadd231pd	%ymm12, %ymm1, %ymm3
	vbroadcastf128	48(%rdx), %ymm12
	subq	$-128, %rdi
	vfmadd231pd	%ymm13, %ymm1, %ymm5
	vshufpd	$5, %ymm10, %ymm10, %ymm13
	vfmadd231pd	%ymm10, %ymm9, %ymm2
	vfmadd231pd	%ymm11, %ymm1, %ymm0
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vbroadcastf128	64(%rdx), %ymm11
	vmovapd	-96(%rdi), %ymm1
	vfmadd231pd	%ymm12, %ymm9, %ymm8
	vbroadcastf128	80(%rdx), %ymm12
	vfmadd231pd	%ymm13, %ymm9, %ymm7
	vshufpd	$5, %ymm11, %ymm11, %ymm13
	vfmadd132pd	%ymm9, %ymm6, %ymm10
	vfmadd231pd	%ymm11, %ymm1, %ymm4
	vmovapd	-64(%rdi), %ymm6
	vbroadcastf128	96(%rdx), %ymm9
	vshufpd	$5, %ymm12, %ymm12, %ymm11
	vfmadd231pd	%ymm12, %ymm1, %ymm3
	vbroadcastf128	112(%rdx), %ymm12
	vfmadd231pd	%ymm13, %ymm1, %ymm5
	subq	$-128, %rdx
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vfmadd231pd	%ymm9, %ymm6, %ymm2
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	vfmadd231pd	%ymm11, %ymm1, %ymm0
	vmovapd	-32(%rdi), %ymm1
	cmpq	%r15, %rdi
	vfmadd231pd	%ymm12, %ymm6, %ymm8
	vbroadcastf128	(%rdx), %ymm11
	vfmadd231pd	%ymm13, %ymm6, %ymm7
	vbroadcastf128	16(%rdx), %ymm12
	vfmadd132pd	%ymm9, %ymm10, %ymm6
	jne	.L546
	movq	-80(%rbp), %rdx
	movl	-68(%rbp), %edi
	addq	$1, %rdx
	leal	4(%r14,%rdi,4), %r14d
	salq	$7, %rdx
	addq	%rdx, %rsi
	addq	%rdx, %r9
.L545:
	leal	-1(%rcx), %edx
	cmpl	%edx, %r14d
	jge	.L547
	leal	-2(%rcx), %r15d
	leaq	32(%r9), %rdx
	subl	%r14d, %r15d
	shrl	%r15d
	movl	%r15d, %edi
	salq	$6, %rdi
	leaq	96(%r9,%rdi), %rdi
	.p2align 4,,10
	.p2align 3
.L548:
	vshufpd	$5, %ymm11, %ymm11, %ymm10
	vbroadcastf128	32(%rsi), %ymm13
	vfmadd231pd	%ymm12, %ymm1, %ymm3
	vmovapd	(%rdx), %ymm9
	vfmadd231pd	%ymm11, %ymm1, %ymm4
	addq	$64, %rdx
	vfmadd231pd	%ymm10, %ymm1, %ymm5
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vfmadd231pd	%ymm13, %ymm9, %ymm2
	vshufpd	$5, %ymm13, %ymm13, %ymm12
	vfmadd231pd	%ymm10, %ymm1, %ymm0
	vbroadcastf128	48(%rsi), %ymm10
	addq	$64, %rsi
	vfmadd231pd	%ymm12, %ymm9, %ymm7
	vmovapd	-32(%rdx), %ymm1
	cmpq	%rdi, %rdx
	vshufpd	$5, %ymm10, %ymm10, %ymm12
	vbroadcastf128	(%rsi), %ymm11
	vfmadd231pd	%ymm10, %ymm9, %ymm8
	vfmadd231pd	%ymm12, %ymm9, %ymm6
	vbroadcastf128	16(%rsi), %ymm12
	jne	.L548
	leal	2(%r14,%r15,2), %r14d
.L547:
	cmpl	%r14d, %ecx
	jle	.L541
	vshufpd	$5, %ymm11, %ymm11, %ymm10
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	.p2align 4,,10
	.p2align 3
.L549:
	addl	$1, %r14d
	vfmadd231pd	%ymm11, %ymm1, %ymm4
	vfmadd231pd	%ymm10, %ymm1, %ymm5
	cmpl	%r14d, %ecx
	vfmadd231pd	%ymm12, %ymm1, %ymm3
	vfmadd231pd	%ymm9, %ymm1, %ymm0
	jne	.L549
.L541:
	testl	%r8d, %r8d
	jle	.L550
	cmpl	$3, %r8d
	vmovapd	0(%r13), %ymm1
	vbroadcastf128	(%rax), %ymm10
	vbroadcastf128	16(%rax), %ymm12
	jle	.L550
	subl	$4, %r8d
	leaq	32(%r13), %rdx
	shrl	$2, %r8d
	salq	$7, %r8
	leaq	160(%r13,%r8), %rcx
	.p2align 4,,10
	.p2align 3
.L551:
	vshufpd	$5, %ymm10, %ymm10, %ymm13
	vfmadd231pd	%ymm10, %ymm1, %ymm4
	vbroadcastf128	32(%rax), %ymm11
	vshufpd	$5, %ymm12, %ymm12, %ymm10
	vfmadd231pd	%ymm12, %ymm1, %ymm3
	vmovapd	(%rdx), %ymm9
	vbroadcastf128	64(%rax), %ymm12
	subq	$-128, %rdx
	vfmadd231pd	%ymm13, %ymm1, %ymm5
	vbroadcastf128	48(%rax), %ymm13
	vfmadd231pd	%ymm11, %ymm9, %ymm2
	vfmadd132pd	%ymm10, %ymm0, %ymm1
	vshufpd	$5, %ymm11, %ymm11, %ymm0
	vmovapd	-96(%rdx), %ymm10
	vshufpd	$5, %ymm13, %ymm13, %ymm11
	vfmadd231pd	%ymm13, %ymm9, %ymm8
	vbroadcastf128	80(%rax), %ymm13
	vfmadd231pd	%ymm12, %ymm10, %ymm4
	vfmadd231pd	%ymm0, %ymm9, %ymm7
	vshufpd	$5, %ymm12, %ymm12, %ymm0
	vbroadcastf128	112(%rax), %ymm12
	vfmadd132pd	%ymm9, %ymm6, %ymm11
	vmovapd	-64(%rdx), %ymm6
	vfmadd231pd	%ymm13, %ymm10, %ymm3
	vbroadcastf128	96(%rax), %ymm9
	subq	$-128, %rax
	vfmadd231pd	%ymm0, %ymm10, %ymm5
	vshufpd	$5, %ymm13, %ymm13, %ymm0
	vfmadd231pd	%ymm12, %ymm6, %ymm8
	vshufpd	$5, %ymm9, %ymm9, %ymm13
	vfmadd231pd	%ymm9, %ymm6, %ymm2
	vshufpd	$5, %ymm12, %ymm12, %ymm9
	vbroadcastf128	16(%rax), %ymm12
	vfmadd132pd	%ymm10, %ymm1, %ymm0
	vmovapd	-32(%rdx), %ymm1
	cmpq	%rdx, %rcx
	vfmadd231pd	%ymm13, %ymm6, %ymm7
	vbroadcastf128	(%rax), %ymm10
	vfmadd132pd	%ymm9, %ymm11, %ymm6
	jne	.L551
.L550:
	vaddpd	%ymm7, %ymm5, %ymm9
	testl	%r10d, %r10d
	vaddpd	%ymm2, %ymm4, %ymm2
	vaddpd	%ymm6, %ymm0, %ymm0
	vaddpd	%ymm8, %ymm3, %ymm3
	vblendpd	$10, %ymm9, %ymm2, %ymm11
	vblendpd	$5, %ymm9, %ymm2, %ymm1
	vblendpd	$10, %ymm0, %ymm3, %ymm5
	vblendpd	$5, %ymm0, %ymm3, %ymm0
	je	.L552
	vaddpd	(%r12), %ymm11, %ymm11
	vaddpd	32(%r12), %ymm1, %ymm1
	vaddpd	64(%r12), %ymm5, %ymm5
	vaddpd	96(%r12), %ymm0, %ymm0
.L552:
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovsd	%xmm11, %xmm4, %xmm3
	vmovsd	.LC0(%rip), %xmm4
	vucomisd	%xmm4, %xmm3
	jbe	.L553
	vmovsd	.LC1(%rip), %xmm10
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vmovdqu	mask_bkp.28217(%rip), %ymm2
	vdivsd	%xmm3, %xmm10, %xmm3
	vmovlpd	%xmm3, (%r11)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm3, %ymm11, %ymm10
	vmaskmovpd	%ymm10, %ymm2, (%rbx)
.L554:
	vpermpd	$85, %ymm10, %ymm9
	vmovlpd	%xmm9, 8(%r11)
	vfmadd132pd	%ymm10, %ymm1, %ymm9
	vpermilpd	$3, %xmm9, %xmm1
	vucomisd	%xmm4, %xmm1
	jbe	.L555
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm2, %xmm1
	vmovdqu	mask_bkp.28217(%rip), %ymm2
	vandpd	.LC12(%rip), %ymm2, %ymm2
	vmovlpd	%xmm1, 16(%r11)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm9, %ymm1, %ymm1
	vmaskmovpd	%ymm1, %ymm2, 32(%rbx)
.L556:
	vpermpd	$170, %ymm10, %ymm2
	vmovlpd	%xmm2, 24(%r11)
	vfmadd132pd	%ymm10, %ymm5, %ymm2
	vmovapd	%ymm2, %ymm3
	vpermpd	$170, %ymm1, %ymm2
	vmovlpd	%xmm2, 32(%r11)
	vfmadd132pd	%ymm1, %ymm3, %ymm2
	vextractf128	$0x1, %ymm2, %xmm3
	vucomisd	%xmm4, %xmm3
	jbe	.L557
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm5, %xmm3
	vmovdqu	mask_bkp.28217(%rip), %ymm5
	vandpd	.LC13(%rip), %ymm5, %ymm5
	vmovlpd	%xmm3, 40(%r11)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm2, %ymm3, %ymm2
	vmaskmovpd	%ymm2, %ymm5, 64(%rbx)
.L558:
	vpermpd	$255, %ymm10, %ymm5
	vpermpd	$255, %ymm2, %ymm3
	vfmadd231pd	%ymm5, %ymm10, %ymm0
	vmovlpd	%xmm5, 48(%r11)
	vpermpd	$255, %ymm1, %ymm5
	vmovlpd	%xmm3, 64(%r11)
	vfmadd132pd	%ymm5, %ymm0, %ymm1
	vfmadd132pd	%ymm3, %ymm1, %ymm2
	vextractf128	$0x1, %ymm2, %xmm3
	vmovlpd	%xmm5, 56(%r11)
	vpermilpd	$3, %xmm3, %xmm3
	vucomisd	%xmm4, %xmm3
	ja	.L576
	vmovdqu	mask_bkp.28217(%rip), %ymm0
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 72(%r11)
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmaskmovpd	%ymm1, %ymm0, 96(%rbx)
.L573:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L539:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.28218(%rip), %ymm1
	testl	%ecx, %ecx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.28217(%rip)
	jg	.L577
.L561:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%ymm8, %ymm6
	vmovapd	%ymm8, %ymm7
	vmovapd	%ymm8, %ymm2
	vmovapd	%ymm8, %ymm3
	vmovapd	%ymm8, %ymm0
	vmovapd	%ymm8, %ymm5
	vmovapd	%ymm8, %ymm4
	jmp	.L541
	.p2align 4,,10
	.p2align 3
.L557:
	vmovdqu	mask_bkp.28217(%rip), %ymm3
	vxorpd	%xmm5, %xmm5, %xmm5
	movq	$0, 40(%r11)
	vandpd	.LC13(%rip), %ymm3, %ymm3
	vblendpd	$7, %ymm5, %ymm2, %ymm2
	vmaskmovpd	%ymm2, %ymm3, 64(%rbx)
	jmp	.L558
	.p2align 4,,10
	.p2align 3
.L555:
	vmovdqu	mask_bkp.28217(%rip), %ymm2
	vxorpd	%xmm1, %xmm1, %xmm1
	movq	$0, 16(%r11)
	vandpd	.LC12(%rip), %ymm2, %ymm2
	vblendpd	$3, %ymm1, %ymm9, %ymm1
	vmaskmovpd	%ymm1, %ymm2, 32(%rbx)
	jmp	.L556
	.p2align 4,,10
	.p2align 3
.L553:
	vxorpd	%xmm10, %xmm10, %xmm10
	vmovdqu	mask_bkp.28217(%rip), %ymm2
	movq	$0, (%r11)
	vblendpd	$1, %ymm10, %ymm11, %ymm10
	vmaskmovpd	%ymm10, %ymm2, (%rbx)
	jmp	.L554
	.p2align 4,,10
	.p2align 3
.L576:
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm3, %xmm3, %xmm3
	vdivsd	%xmm3, %xmm0, %xmm3
	vmovdqu	mask_bkp.28217(%rip), %ymm0
	vandpd	.LC14(%rip), %ymm0, %ymm0
	vmovlpd	%xmm3, 72(%r11)
	vbroadcastsd	%xmm3, %ymm3
	vmulpd	%ymm2, %ymm3, %ymm2
	vmaskmovpd	%ymm2, %ymm0, 96(%rbx)
	jmp	.L573
	.p2align 4,,10
	.p2align 3
.L575:
	cmpl	$3, %ecx
	vxorpd	%xmm2, %xmm2, %xmm2
	jle	.L543
	vblendpd	$3, 32(%r9), %ymm2, %ymm4
	vbroadcastf128	32(%rsi), %ymm3
	vblendpd	$1, %ymm1, %ymm2, %ymm1
	vbroadcastf128	64(%rsi), %ymm0
	vblendpd	$7, 64(%r9), %ymm2, %ymm7
	movl	$4, %r14d
	vfmadd132pd	%ymm11, %ymm2, %ymm1
	vbroadcastf128	80(%rsi), %ymm6
	vfmadd231pd	%ymm3, %ymm4, %ymm1
	vblendpd	$7, %ymm0, %ymm2, %ymm0
	vshufpd	$5, %ymm3, %ymm3, %ymm3
	vbroadcastf128	128(%rsi), %ymm11
	subq	$-128, %r9
	vfmadd231pd	%ymm0, %ymm7, %ymm1
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vbroadcastf128	144(%rsi), %ymm12
	vshufpd	$5, %ymm6, %ymm6, %ymm0
	vfmadd132pd	%ymm4, %ymm2, %ymm3
	vbroadcastf128	96(%rsi), %ymm4
	vfmadd132pd	%ymm7, %ymm2, %ymm6
	vfmadd132pd	%ymm7, %ymm3, %ymm5
	vbroadcastf128	112(%rsi), %ymm3
	subq	$-128, %rsi
	vfmadd231pd	%ymm0, %ymm7, %ymm2
	vshufpd	$5, %ymm4, %ymm4, %ymm0
	vmovapd	-32(%r9), %ymm7
	vfmadd132pd	%ymm7, %ymm1, %ymm4
	vmovapd	(%r9), %ymm1
	vfmadd231pd	%ymm0, %ymm7, %ymm5
	vshufpd	$5, %ymm3, %ymm3, %ymm0
	vfmadd132pd	%ymm7, %ymm6, %ymm3
	vfmadd132pd	%ymm7, %ymm2, %ymm0
	jmp	.L542
	.p2align 4,,10
	.p2align 3
.L543:
	vblendpd	$1, %ymm1, %ymm2, %ymm0
	cmpl	$1, %ecx
	vmovapd	32(%r9), %ymm1
	vmovapd	%ymm0, %ymm4
	vfmadd132pd	%ymm11, %ymm2, %ymm4
	vbroadcastf128	32(%rsi), %ymm11
	je	.L563
	vshufpd	$5, %ymm11, %ymm11, %ymm5
	vblendpd	$3, %ymm1, %ymm2, %ymm0
	cmpl	$3, %ecx
	vmovapd	64(%r9), %ymm1
	vfmadd231pd	%ymm11, %ymm0, %ymm4
	vbroadcastf128	80(%rsi), %ymm12
	vbroadcastf128	64(%rsi), %ymm11
	vfmadd132pd	%ymm0, %ymm2, %ymm5
	je	.L578
	vxorpd	%xmm3, %xmm3, %xmm3
	movl	$2, %r14d
	vmovapd	%ymm3, %ymm0
	jmp	.L542
.L563:
	vxorpd	%xmm3, %xmm3, %xmm3
	movl	$1, %r14d
	vmovapd	%ymm3, %ymm0
	vmovapd	%ymm3, %ymm5
.L565:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%ymm8, %ymm6
	vmovapd	%ymm8, %ymm7
	vmovapd	%ymm8, %ymm2
	jmp	.L545
.L578:
	vblendpd	$7, %ymm1, %ymm2, %ymm0
	vshufpd	$5, %ymm11, %ymm11, %ymm1
	vmovapd	%ymm12, %ymm3
	movl	$3, %r14d
	vfmadd231pd	%ymm11, %ymm0, %ymm4
	vbroadcastf128	96(%rsi), %ymm11
	vfmadd231pd	%ymm1, %ymm0, %ymm5
	vshufpd	$5, %ymm12, %ymm12, %ymm1
	vfmadd132pd	%ymm0, %ymm2, %ymm3
	vbroadcastf128	112(%rsi), %ymm12
	vfmadd132pd	%ymm1, %ymm2, %ymm0
	vmovapd	96(%r9), %ymm1
	jmp	.L565
	.cfi_endproc
.LFE4602:
	.size	kernel_dsyrk_dpotrf_nt_4x4_vs_lib4, .-kernel_dsyrk_dpotrf_nt_4x4_vs_lib4
	.section	.text.unlikely
.LCOLDE24:
	.text
.LHOTE24:
	.section	.text.unlikely
.LCOLDB25:
	.text
.LHOTB25:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_4x2_vs_lib4:
.LFB4603:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$3, %edi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	movq	(%r10), %r11
	movq	8(%r10), %r13
	movq	16(%r10), %rax
	movq	24(%r10), %r14
	movq	32(%r10), %rbx
	movq	40(%r10), %r12
	movl	48(%r10), %r10d
	jle	.L580
	vpcmpeqd	%ymm0, %ymm0, %ymm0
	testl	%ecx, %ecx
	vmovdqu	%ymm0, mask_bkp.28302(%rip)
	jle	.L599
.L612:
	cmpl	$1, %edx
	vmovapd	(%r9), %ymm10
	vbroadcastf128	(%r11), %ymm0
	je	.L610
	vxorpd	%xmm4, %xmm4, %xmm4
	xorl	%edi, %edi
	vmovapd	%ymm4, %ymm3
	vmovapd	%ymm4, %ymm1
	vmovapd	%ymm4, %ymm2
.L583:
	leal	-3(%rcx), %edx
	cmpl	%edx, %edi
	jge	.L603
	leal	-4(%rcx), %edx
	vxorpd	%xmm9, %xmm9, %xmm9
	leaq	32(%r9), %rsi
	subl	%edi, %edx
	shrl	$2, %edx
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm9, %ymm7
	movl	%edx, -68(%rbp)
	movq	%rdx, -80(%rbp)
	vmovapd	%ymm9, %ymm6
	salq	$7, %rdx
	leaq	160(%r9,%rdx), %r15
	movq	%r11, %rdx
	.p2align 4,,10
	.p2align 3
.L587:
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%rdx), %ymm5
	subq	$-128, %rsi
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vmovapd	-128(%rsi), %ymm10
	vbroadcastf128	64(%rdx), %ymm0
	vfmadd231pd	%ymm5, %ymm10, %ymm3
	vshufpd	$5, %ymm5, %ymm5, %ymm5
	vfmadd231pd	%ymm5, %ymm10, %ymm4
	vmovapd	-96(%rsi), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vfmadd231pd	%ymm0, %ymm10, %ymm6
	vbroadcastf128	96(%rdx), %ymm0
	subq	$-128, %rdx
	vfmadd231pd	%ymm5, %ymm10, %ymm7
	vmovapd	-64(%rsi), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vfmadd231pd	%ymm0, %ymm10, %ymm8
	vbroadcastf128	(%rdx), %ymm0
	vfmadd231pd	%ymm5, %ymm10, %ymm9
	vmovapd	-32(%rsi), %ymm10
	cmpq	%r15, %rsi
	jne	.L587
	movq	-80(%rbp), %rdx
	movl	-68(%rbp), %esi
	addq	$1, %rdx
	leal	4(%rdi,%rsi,4), %edi
	salq	$7, %rdx
	addq	%rdx, %r11
	addq	%rdx, %r9
.L586:
	leal	-1(%rcx), %edx
	cmpl	%edx, %edi
	jge	.L588
	leal	-2(%rcx), %r15d
	leaq	32(%r9), %rdx
	subl	%edi, %r15d
	shrl	%r15d
	movl	%r15d, %esi
	salq	$6, %rsi
	leaq	96(%r9,%rsi), %rsi
	.p2align 4,,10
	.p2align 3
.L589:
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%r11), %ymm5
	addq	$64, %rdx
	addq	$64, %r11
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vmovapd	-64(%rdx), %ymm10
	vbroadcastf128	(%r11), %ymm0
	vfmadd231pd	%ymm5, %ymm10, %ymm3
	vshufpd	$5, %ymm5, %ymm5, %ymm5
	vfmadd231pd	%ymm5, %ymm10, %ymm4
	vmovapd	-32(%rdx), %ymm10
	cmpq	%rsi, %rdx
	jne	.L589
	leal	2(%rdi,%r15,2), %edi
.L588:
	cmpl	%edi, %ecx
	jle	.L582
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	.p2align 4,,10
	.p2align 3
.L590:
	addl	$1, %edi
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vfmadd231pd	%ymm5, %ymm10, %ymm1
	cmpl	%edi, %ecx
	jne	.L590
.L582:
	testl	%r8d, %r8d
	jle	.L591
	cmpl	$3, %r8d
	vmovapd	0(%r13), %ymm10
	vbroadcastf128	(%rax), %ymm0
	jle	.L591
	subl	$4, %r8d
	leaq	32(%r13), %rdx
	shrl	$2, %r8d
	salq	$7, %r8
	leaq	160(%r13,%r8), %rcx
	.p2align 4,,10
	.p2align 3
.L592:
	vfmadd231pd	%ymm0, %ymm10, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm0
	vbroadcastf128	32(%rax), %ymm5
	subq	$-128, %rdx
	vfmadd231pd	%ymm0, %ymm10, %ymm1
	vmovapd	-128(%rdx), %ymm10
	vbroadcastf128	64(%rax), %ymm0
	vfmadd231pd	%ymm5, %ymm10, %ymm3
	vshufpd	$5, %ymm5, %ymm5, %ymm5
	vfmadd231pd	%ymm5, %ymm10, %ymm4
	vmovapd	-96(%rdx), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vfmadd231pd	%ymm0, %ymm10, %ymm6
	vbroadcastf128	96(%rax), %ymm0
	subq	$-128, %rax
	vfmadd231pd	%ymm5, %ymm10, %ymm7
	vmovapd	-64(%rdx), %ymm10
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vfmadd231pd	%ymm0, %ymm10, %ymm8
	vbroadcastf128	(%rax), %ymm0
	vfmadd231pd	%ymm5, %ymm10, %ymm9
	vmovapd	-32(%rdx), %ymm10
	cmpq	%rcx, %rdx
	jne	.L592
.L591:
	vaddpd	%ymm6, %ymm2, %ymm2
	testl	%r10d, %r10d
	vaddpd	%ymm8, %ymm3, %ymm3
	vaddpd	%ymm7, %ymm1, %ymm1
	vaddpd	%ymm9, %ymm4, %ymm4
	vaddpd	%ymm3, %ymm2, %ymm2
	vaddpd	%ymm4, %ymm1, %ymm0
	jne	.L593
	vblendpd	$10, %ymm0, %ymm2, %ymm12
	vblendpd	$5, %ymm0, %ymm2, %ymm2
.L594:
	vxorpd	%xmm11, %xmm11, %xmm11
	vmovsd	.LC0(%rip), %xmm3
	vmovsd	%xmm12, %xmm11, %xmm10
	vucomisd	%xmm3, %xmm10
	jbe	.L595
	vmovsd	.LC1(%rip), %xmm5
	vsqrtsd	%xmm10, %xmm10, %xmm10
	vmovdqu	mask_bkp.28302(%rip), %ymm0
	vdivsd	%xmm10, %xmm5, %xmm10
	vmovlpd	%xmm10, (%r12)
	vbroadcastsd	%xmm10, %ymm10
	vmulpd	%ymm10, %ymm12, %ymm5
	vmaskmovpd	%ymm5, %ymm0, (%rbx)
.L596:
	vpermpd	$85, %ymm5, %ymm0
	vmovlpd	%xmm0, 8(%r12)
	vfmadd132pd	%ymm5, %ymm2, %ymm0
	vpermilpd	$3, %xmm0, %xmm1
	vucomisd	%xmm3, %xmm1
	ja	.L611
	vmovdqu	mask_bkp.28302(%rip), %ymm1
	vxorpd	%xmm2, %xmm2, %xmm2
	movq	$0, 16(%r12)
	vandpd	.LC12(%rip), %ymm1, %ymm1
	vblendpd	$3, %ymm2, %ymm0, %ymm0
	vmaskmovpd	%ymm0, %ymm1, 32(%rbx)
.L608:
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.p2align 4,,10
	.p2align 3
.L580:
	.cfi_restore_state
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovupd	d_mask.28303(%rip), %ymm1
	testl	%ecx, %ecx
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vmovsd	%xmm0, -56(%rbp)
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm1, %ymm0
	vmovdqu	%ymm0, mask_bkp.28302(%rip)
	jg	.L612
.L599:
	vxorpd	%xmm9, %xmm9, %xmm9
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm9, %ymm7
	vmovapd	%ymm9, %ymm6
	vmovapd	%ymm9, %ymm4
	vmovapd	%ymm9, %ymm3
	vmovapd	%ymm9, %ymm1
	vmovapd	%ymm9, %ymm2
	jmp	.L582
	.p2align 4,,10
	.p2align 3
.L595:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovdqu	mask_bkp.28302(%rip), %ymm1
	movq	$0, (%r12)
	vblendpd	$1, %ymm0, %ymm12, %ymm5
	vmaskmovpd	%ymm5, %ymm1, (%rbx)
	jmp	.L596
	.p2align 4,,10
	.p2align 3
.L593:
	vblendpd	$10, %ymm0, %ymm2, %ymm5
	vblendpd	$5, %ymm0, %ymm2, %ymm2
	vaddpd	(%r14), %ymm5, %ymm12
	vaddpd	32(%r14), %ymm2, %ymm2
	jmp	.L594
	.p2align 4,,10
	.p2align 3
.L611:
	vmovsd	.LC1(%rip), %xmm2
	vsqrtsd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm1, %xmm2, %xmm1
	vmovdqu	mask_bkp.28302(%rip), %ymm2
	vandpd	.LC12(%rip), %ymm2, %ymm2
	vmovlpd	%xmm1, 16(%r12)
	vbroadcastsd	%xmm1, %ymm1
	vmulpd	%ymm0, %ymm1, %ymm0
	vmaskmovpd	%ymm0, %ymm2, 32(%rbx)
	jmp	.L608
	.p2align 4,,10
	.p2align 3
.L610:
	cmpl	$3, %ecx
	jle	.L584
	vxorpd	%xmm6, %xmm6, %xmm6
	vbroadcastf128	64(%r11), %ymm1
	subq	$-128, %r9
	vbroadcastf128	32(%r11), %ymm3
	movl	$4, %edi
	vmovapd	-32(%r9), %ymm4
	vblendpd	$1, %ymm10, %ymm6, %ymm10
	vblendpd	$7, -64(%r9), %ymm6, %ymm5
	vblendpd	$3, -96(%r9), %ymm6, %ymm8
	vshufpd	$5, %ymm3, %ymm3, %ymm7
	vmulpd	%ymm10, %ymm0, %ymm0
	vmovapd	(%r9), %ymm10
	vmulpd	%ymm5, %ymm1, %ymm2
	vmulpd	%ymm8, %ymm3, %ymm3
	vaddpd	%ymm6, %ymm0, %ymm0
	vshufpd	$5, %ymm1, %ymm1, %ymm1
	vmulpd	%ymm8, %ymm7, %ymm7
	vmulpd	%ymm5, %ymm1, %ymm1
	vaddpd	%ymm2, %ymm0, %ymm2
	vbroadcastf128	96(%r11), %ymm0
	subq	$-128, %r11
	vmulpd	%ymm4, %ymm0, %ymm5
	vaddpd	%ymm6, %ymm3, %ymm3
	vaddpd	%ymm6, %ymm1, %ymm1
	vaddpd	%ymm6, %ymm7, %ymm6
	vaddpd	%ymm5, %ymm3, %ymm3
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vbroadcastf128	(%r11), %ymm0
	vmulpd	%ymm4, %ymm5, %ymm4
	vaddpd	%ymm4, %ymm6, %ymm4
	jmp	.L583
	.p2align 4,,10
	.p2align 3
.L584:
	vxorpd	%xmm1, %xmm1, %xmm1
	cmpl	$1, %ecx
	vblendpd	$1, %ymm10, %ymm1, %ymm2
	vmovapd	32(%r9), %ymm10
	vmulpd	%ymm2, %ymm0, %ymm0
	vaddpd	%ymm1, %ymm0, %ymm2
	vbroadcastf128	32(%r11), %ymm0
	je	.L601
	vshufpd	$5, %ymm0, %ymm0, %ymm4
	vblendpd	$3, %ymm10, %ymm1, %ymm5
	cmpl	$3, %ecx
	vmovapd	64(%r9), %ymm10
	movl	$2, %edi
	vmulpd	%ymm0, %ymm5, %ymm3
	vbroadcastf128	64(%r11), %ymm0
	vmulpd	%ymm5, %ymm4, %ymm4
	vaddpd	%ymm1, %ymm3, %ymm3
	vaddpd	%ymm1, %ymm4, %ymm4
	jne	.L583
	vblendpd	$7, %ymm10, %ymm1, %ymm6
	vmovapd	96(%r9), %ymm10
	movl	$3, %edi
	vmulpd	%ymm0, %ymm6, %ymm5
	vaddpd	%ymm5, %ymm2, %ymm2
	vshufpd	$5, %ymm0, %ymm0, %ymm5
	vbroadcastf128	96(%r11), %ymm0
	vmulpd	%ymm6, %ymm5, %ymm5
	vaddpd	%ymm1, %ymm5, %ymm1
.L603:
	vxorpd	%xmm9, %xmm9, %xmm9
	vmovapd	%ymm9, %ymm8
	vmovapd	%ymm9, %ymm7
	vmovapd	%ymm9, %ymm6
	jmp	.L586
.L601:
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$1, %edi
	vmovapd	%ymm4, %ymm3
	vmovapd	%ymm4, %ymm1
	jmp	.L603
	.cfi_endproc
.LFE4603:
	.size	kernel_dsyrk_dpotrf_nt_4x2_vs_lib4, .-kernel_dsyrk_dpotrf_nt_4x2_vs_lib4
	.section	.text.unlikely
.LCOLDE25:
	.text
.LHOTE25:
	.section	.text.unlikely
.LCOLDB27:
	.text
.LHOTB27:
	.p2align 4,,15
	.globl	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4
	.type	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4, @function
kernel_dsyrk_dpotrf_nt_2x2_vs_lib4:
.LFB4604:
	.cfi_startproc
	testl	%ecx, %ecx
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	56(%rsp), %rbp
	movq	72(%rsp), %r10
	movq	88(%rsp), %r14
	movq	96(%rsp), %r15
	jle	.L637
	cmpl	$1, %edx
	vmovapd	(%r9), %xmm7
	vmovapd	0(%rbp), %xmm0
	je	.L645
	vxorpd	%xmm3, %xmm3, %xmm3
	xorl	%r11d, %r11d
	vmovapd	%xmm3, %xmm4
	vmovapd	%xmm3, %xmm1
.L615:
	leal	-3(%rcx), %eax
	cmpl	%eax, %r11d
	jge	.L646
	leal	-4(%rcx), %r13d
	vxorpd	%xmm8, %xmm8, %xmm8
	leaq	32(%rbp), %rax
	subl	%r11d, %r13d
	shrl	$2, %r13d
	vmovapd	%xmm8, %xmm10
	movl	%r13d, %r12d
	vmovapd	%xmm8, %xmm9
	movq	%r12, %rdx
	vmovapd	%xmm8, %xmm11
	salq	$7, %rdx
	vmovapd	%xmm8, %xmm2
	leaq	160(%rbp,%rdx), %rbx
	movq	%r9, %rdx
	.p2align 4,,10
	.p2align 3
.L619:
	vfmadd231pd	%xmm0, %xmm7, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	(%rax), %xmm5
	subq	$-128, %rax
	subq	$-128, %rdx
	vfmadd231pd	%xmm0, %xmm7, %xmm2
	vmovapd	-96(%rdx), %xmm7
	vfmadd231pd	%xmm5, %xmm7, %xmm4
	vshufpd	$1, %xmm5, %xmm5, %xmm5
	vmovapd	-96(%rax), %xmm0
	vfmadd231pd	%xmm5, %xmm7, %xmm3
	vmovapd	-64(%rdx), %xmm7
	vshufpd	$1, %xmm0, %xmm0, %xmm5
	vfmadd231pd	%xmm0, %xmm7, %xmm11
	vmovapd	-64(%rax), %xmm0
	vfmadd231pd	%xmm5, %xmm7, %xmm9
	vmovapd	-32(%rdx), %xmm7
	vshufpd	$1, %xmm0, %xmm0, %xmm5
	vfmadd231pd	%xmm0, %xmm7, %xmm10
	vmovapd	-32(%rax), %xmm0
	vfmadd231pd	%xmm5, %xmm7, %xmm8
	vmovapd	(%rdx), %xmm7
	cmpq	%rbx, %rax
	jne	.L619
	addq	$1, %r12
	leal	4(%r11,%r13,4), %r11d
	salq	$7, %r12
	addq	%r12, %r9
	addq	%r12, %rbp
.L618:
	leal	-1(%rcx), %eax
	cmpl	%eax, %r11d
	jge	.L620
	leal	-2(%rcx), %ebx
	leaq	32(%rbp), %rdx
	subl	%r11d, %ebx
	shrl	%ebx
	movl	%ebx, %eax
	salq	$6, %rax
	leaq	96(%rbp,%rax), %rax
	.p2align 4,,10
	.p2align 3
.L621:
	vfmadd231pd	%xmm0, %xmm7, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	(%rdx), %xmm5
	addq	$64, %rdx
	addq	$64, %r9
	vfmadd231pd	%xmm0, %xmm7, %xmm2
	vmovapd	-32(%r9), %xmm7
	vfmadd231pd	%xmm5, %xmm7, %xmm4
	vshufpd	$1, %xmm5, %xmm5, %xmm5
	vmovapd	-32(%rdx), %xmm0
	vfmadd231pd	%xmm5, %xmm7, %xmm3
	vmovapd	(%r9), %xmm7
	cmpq	%rdx, %rax
	jne	.L621
	leal	2(%r11,%rbx,2), %r11d
.L620:
	cmpl	%r11d, %ecx
	jle	.L614
	vshufpd	$1, %xmm0, %xmm0, %xmm5
	.p2align 4,,10
	.p2align 3
.L622:
	addl	$1, %r11d
	vfmadd231pd	%xmm0, %xmm7, %xmm1
	vfmadd231pd	%xmm5, %xmm7, %xmm2
	cmpl	%r11d, %ecx
	jne	.L622
.L614:
	testl	%r8d, %r8d
	jle	.L623
	movq	64(%rsp), %rax
	cmpl	$3, %r8d
	vmovapd	(%r10), %xmm5
	vmovapd	(%rax), %xmm7
	jle	.L623
	subl	$4, %r8d
	movq	64(%rsp), %rbx
	addq	$32, %rax
	shrl	$2, %r8d
	salq	$7, %r8
	leaq	160(%rbx,%r8), %rdx
	.p2align 4,,10
	.p2align 3
.L624:
	vfmadd231pd	%xmm5, %xmm7, %xmm1
	vshufpd	$1, %xmm5, %xmm5, %xmm5
	vmovapd	32(%r10), %xmm0
	subq	$-128, %rax
	subq	$-128, %r10
	vfmadd231pd	%xmm5, %xmm7, %xmm2
	vmovapd	-128(%rax), %xmm7
	vfmadd231pd	%xmm0, %xmm7, %xmm4
	vshufpd	$1, %xmm0, %xmm0, %xmm0
	vmovapd	-64(%r10), %xmm5
	vfmadd231pd	%xmm0, %xmm7, %xmm3
	vmovapd	-96(%rax), %xmm0
	vshufpd	$1, %xmm5, %xmm5, %xmm7
	vfmadd231pd	%xmm5, %xmm0, %xmm11
	vmovapd	-32(%r10), %xmm5
	vfmadd231pd	%xmm7, %xmm0, %xmm9
	vshufpd	$1, %xmm5, %xmm5, %xmm7
	vmovapd	-64(%rax), %xmm0
	vfmadd231pd	%xmm7, %xmm0, %xmm8
	vmovapd	-32(%rax), %xmm7
	vfmadd231pd	%xmm5, %xmm0, %xmm10
	vmovapd	(%r10), %xmm5
	cmpq	%rax, %rdx
	jne	.L624
.L623:
	vaddpd	%xmm8, %xmm3, %xmm3
	movl	104(%rsp), %ecx
	vaddpd	%xmm11, %xmm1, %xmm1
	testl	%ecx, %ecx
	vaddpd	%xmm10, %xmm4, %xmm4
	vaddpd	%xmm9, %xmm2, %xmm2
	vaddpd	%xmm4, %xmm1, %xmm1
	vaddpd	%xmm3, %xmm2, %xmm2
	vblendpd	$2, %xmm2, %xmm1, %xmm3
	jne	.L625
	vmovsd	%xmm2, %xmm1, %xmm1
.L626:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovsd	.LC0(%rip), %xmm2
	vmovsd	%xmm3, %xmm0, %xmm0
	vucomisd	%xmm2, %xmm0
	jbe	.L627
	movl	112(%rsp), %edx
	testl	%edx, %edx
	jle	.L628
	vxorps	%xmm4, %xmm4, %xmm4
	vshufpd	$1, %xmm2, %xmm3, %xmm3
	vcvtsd2ss	%xmm0, %xmm4, %xmm4
	vsqrtss	%xmm4, %xmm4, %xmm4
	vcvtss2sd	%xmm4, %xmm0, %xmm5
	vmovlpd	%xmm5, (%r14)
	vmovss	.LC26(%rip), %xmm5
	vdivss	%xmm4, %xmm5, %xmm4
	vcvtss2sd	%xmm4, %xmm0, %xmm0
.L629:
	cmpl	$1, %edi
	vmovlpd	%xmm0, (%r15)
	vmulsd	%xmm0, %xmm3, %xmm6
	jne	.L647
.L643:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L625:
	.cfi_restore_state
	movq	80(%rsp), %rax
	vmovsd	%xmm2, %xmm1, %xmm1
	vaddpd	(%rax), %xmm3, %xmm3
	vaddpd	32(%rax), %xmm1, %xmm1
	jmp	.L626
	.p2align 4,,10
	.p2align 3
.L627:
	cmpl	$1, %edi
	movq	$0, (%r14)
	movq	$0, (%r15)
	je	.L643
.L647:
	cmpl	$1, %esi
	vmovsd	%xmm6, 8(%r14)
	jle	.L643
	vmovsd	%xmm6, 8(%r15)
	vmulsd	%xmm6, %xmm6, %xmm6
	vshufpd	$1, %xmm2, %xmm1, %xmm1
	vsubsd	%xmm6, %xmm1, %xmm6
	vucomisd	%xmm2, %xmm6
	jbe	.L634
	movl	112(%rsp), %eax
	testl	%eax, %eax
	jle	.L635
	vmovsd	.LC1(%rip), %xmm0
	vsqrtsd	%xmm6, %xmm6, %xmm6
	vmovlpd	%xmm6, 40(%r14)
	vdivsd	%xmm6, %xmm0, %xmm6
.L636:
	vmovlpd	%xmm6, 16(%r15)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L634:
	.cfi_restore_state
	movq	$0, 40(%r14)
	movq	$0, 16(%r15)
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L637:
	.cfi_restore_state
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%xmm8, %xmm10
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm3
	vmovapd	%xmm8, %xmm4
	vmovapd	%xmm8, %xmm2
	vmovapd	%xmm8, %xmm1
	jmp	.L614
	.p2align 4,,10
	.p2align 3
.L645:
	cmpl	$1, %ecx
	vmulsd	%xmm0, %xmm7, %xmm0
	je	.L616
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovapd	32(%r9), %xmm3
	addq	$64, %rbp
	addq	$64, %r9
	movl	$2, %r11d
	vmovapd	(%r9), %xmm7
	vaddsd	%xmm0, %xmm2, %xmm1
	vmovapd	-32(%rbp), %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm5
	vmulpd	%xmm3, %xmm0, %xmm4
	vmovapd	0(%rbp), %xmm0
	vmulpd	%xmm5, %xmm3, %xmm3
	vaddpd	%xmm2, %xmm4, %xmm4
	vaddpd	%xmm2, %xmm3, %xmm3
	jmp	.L615
	.p2align 4,,10
	.p2align 3
.L628:
	vmovsd	.LC1(%rip), %xmm6
	vsqrtsd	%xmm0, %xmm0, %xmm0
	vshufpd	$1, %xmm2, %xmm3, %xmm3
	vmovlpd	%xmm0, (%r14)
	vdivsd	%xmm0, %xmm6, %xmm0
	jmp	.L629
.L616:
	vxorpd	%xmm8, %xmm8, %xmm8
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovapd	%xmm8, %xmm10
	vaddsd	%xmm0, %xmm1, %xmm1
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm3
	vmovapd	%xmm8, %xmm4
	vmovapd	%xmm8, %xmm2
	jmp	.L614
	.p2align 4,,10
	.p2align 3
.L635:
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsd2ss	%xmm6, %xmm0, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm1
	vcvtss2sd	%xmm1, %xmm6, %xmm0
	vmovlpd	%xmm0, 40(%r14)
	vmovss	.LC26(%rip), %xmm0
	vdivss	%xmm1, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm6, %xmm6
	jmp	.L636
.L646:
	vxorpd	%xmm8, %xmm8, %xmm8
	vmovapd	%xmm8, %xmm10
	vmovapd	%xmm8, %xmm9
	vmovapd	%xmm8, %xmm11
	vmovapd	%xmm8, %xmm2
	jmp	.L618
	.cfi_endproc
.LFE4604:
	.size	kernel_dsyrk_dpotrf_nt_2x2_vs_lib4, .-kernel_dsyrk_dpotrf_nt_2x2_vs_lib4
	.section	.text.unlikely
.LCOLDE27:
	.text
.LHOTE27:
	.data
	.align 32
	.type	d_mask.28303, @object
	.size	d_mask.28303, 32
d_mask.28303:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.28302
	.comm	mask_bkp.28302,128,32
	.align 32
	.type	d_mask.28218, @object
	.size	d_mask.28218, 32
d_mask.28218:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.28217
	.comm	mask_bkp.28217,128,32
	.align 32
	.type	d_mask.28115, @object
	.size	d_mask.28115, 32
d_mask.28115:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.28114
	.comm	mask_bkp.28114,128,32
	.align 32
	.type	d_mask.28010, @object
	.size	d_mask.28010, 32
d_mask.28010:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.28009
	.comm	mask_bkp.28009,128,32
	.align 32
	.type	d_mask.27908, @object
	.size	d_mask.27908, 32
d_mask.27908:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27907
	.comm	mask_bkp.27907,128,32
	.align 32
	.type	d_mask.27775, @object
	.size	d_mask.27775, 32
d_mask.27775:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27774
	.comm	mask_bkp.27774,128,32
	.align 32
	.type	d_mask.27697, @object
	.size	d_mask.27697, 32
d_mask.27697:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27696
	.comm	mask_bkp.27696,128,32
	.align 32
	.type	d_mask.27605, @object
	.size	d_mask.27605, 32
d_mask.27605:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27604
	.comm	mask_bkp.27604,128,32
	.align 32
	.type	d_mask.27501, @object
	.size	d_mask.27501, 32
d_mask.27501:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27500
	.comm	mask_bkp.27500,128,32
	.align 32
	.type	d_mask.27204, @object
	.size	d_mask.27204, 32
d_mask.27204:
	.long	0
	.long	1071644672
	.long	0
	.long	1073217536
	.long	0
	.long	1074003968
	.long	0
	.long	1074528256
	.local	mask_bkp.27203
	.comm	mask_bkp.27203,128,32
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.long	2665960982
	.long	1020396463
	.long	0
	.long	0
	.align 16
.LC1:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC2:
	.quad	1
	.quad	-1
	.quad	-1
	.quad	-1
	.align 32
.LC3:
	.quad	1
	.quad	1
	.quad	-1
	.quad	-1
	.align 32
.LC4:
	.quad	1
	.quad	1
	.quad	1
	.quad	-1
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC7:
	.long	0
	.long	1075838976
	.align 8
.LC11:
	.long	0
	.long	1074790400
	.section	.rodata.cst32
	.align 32
.LC12:
	.long	1
	.long	0
	.long	4294967295
	.long	-1
	.long	4294967295
	.long	-1
	.long	4294967295
	.long	-1
	.align 32
.LC13:
	.long	1
	.long	0
	.long	1
	.long	0
	.long	4294967295
	.long	-1
	.long	4294967295
	.long	-1
	.align 32
.LC14:
	.long	1
	.long	0
	.long	1
	.long	0
	.long	1
	.long	0
	.long	4294967295
	.long	-1
	.section	.rodata.cst16
	.align 16
.LC26:
	.long	1065353216
	.long	0
	.long	0
	.long	0
	.ident	"GCC: (GNU) 5.2.0"
	.section	.note.GNU-stack,"",@progbits
