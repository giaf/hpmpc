	.file	"kernel_dtrsv_avx2_lib4.c"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB1:
	.text
.LHOTB1:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_8_lib4_new
	.type	kernel_dtrsv_n_8_lib4_new, @function
kernel_dtrsv_n_8_lib4_new:
.LFB4586:
	.cfi_startproc
	pushq	%r10
	.cfi_def_cfa_offset 16
	.cfi_offset 10, -16
	sall	$2, %edx
	cmpl	$7, %edi
	movslq	%edx, %rdx
	movq	16(%rsp), %r11
	leaq	16(%rsp), %r10
	leaq	(%rsi,%rdx,8), %rax
	jle	.L6
	leal	-8(%rdi), %r10d
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%rsi, %rdx
	shrl	$3, %r10d
	addq	$1, %r10
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm1
	salq	$8, %r10
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm8
	leaq	(%rax,%r10), %rdi
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm6
	.p2align 4,,10
	.p2align 3
.L3:
	vbroadcastf128	(%r9), %ymm3
	addq	$256, %rax
	addq	$256, %rdx
	vunpcklpd	%ymm3, %ymm3, %ymm2
	vunpckhpd	%ymm3, %ymm3, %ymm3
	vfmadd231pd	-256(%rdx), %ymm2, %ymm6
	vfmadd231pd	-256(%rax), %ymm2, %ymm4
	vbroadcastf128	16(%r9), %ymm2
	vfmadd231pd	-224(%rdx), %ymm3, %ymm7
	vfmadd132pd	-224(%rax), %ymm1, %ymm3
	vunpcklpd	%ymm2, %ymm2, %ymm1
	vunpckhpd	%ymm2, %ymm2, %ymm2
	vfmadd231pd	-192(%rdx), %ymm1, %ymm9
	vfmadd231pd	-192(%rax), %ymm1, %ymm5
	vbroadcastf128	32(%r9), %ymm1
	vfmadd231pd	-160(%rdx), %ymm2, %ymm8
	vfmadd132pd	-160(%rax), %ymm0, %ymm2
	vunpcklpd	%ymm1, %ymm1, %ymm0
	vunpckhpd	%ymm1, %ymm1, %ymm1
	vfmadd231pd	-128(%rdx), %ymm0, %ymm6
	vfmadd231pd	-128(%rax), %ymm0, %ymm4
	vbroadcastf128	48(%r9), %ymm0
	vfmadd231pd	-96(%rdx), %ymm1, %ymm7
	vfmadd132pd	-96(%rax), %ymm3, %ymm1
	addq	$64, %r9
	vunpcklpd	%ymm0, %ymm0, %ymm3
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rax), %ymm3, %ymm5
	vfmadd231pd	-64(%rdx), %ymm3, %ymm9
	vfmadd231pd	-32(%rdx), %ymm0, %ymm8
	vfmadd132pd	-32(%rax), %ymm2, %ymm0
	cmpq	%rdi, %rax
	jne	.L3
	vaddpd	%ymm0, %ymm1, %ymm0
	addq	%r10, %rsi
	vaddpd	%ymm5, %ymm4, %ymm4
	vaddpd	%ymm9, %ymm6, %ymm6
	vaddpd	%ymm8, %ymm7, %ymm7
	vaddpd	%ymm4, %ymm0, %ymm0
	vaddpd	%ymm7, %ymm6, %ymm6
.L2:
	vmovupd	(%r11), %ymm9
	testl	%ecx, %ecx
	vmovupd	32(%r11), %ymm5
	vsubpd	%ymm6, %ymm9, %ymm6
	vsubpd	%ymm0, %ymm5, %ymm0
	jne	.L11
	vmovsd	(%rsi), %xmm3
	vmovapd	.LC0(%rip), %xmm2
	vmovapd	%xmm6, %xmm1
	vextractf128	$0x1, %ymm6, %xmm5
	vdivsd	%xmm3, %xmm2, %xmm3
	vmovapd	48(%rsi), %xmm9
	vmovsd	40(%rsi), %xmm6
	vxorpd	%xmm7, %xmm7, %xmm7
	vdivsd	%xmm6, %xmm2, %xmm8
	vunpcklpd	%xmm7, %xmm1, %xmm4
	vshufpd	$1, %xmm7, %xmm1, %xmm6
	vmovsd	8(%rsi), %xmm1
	vmulsd	%xmm4, %xmm3, %xmm3
	vmovapd	16(%rsi), %xmm4
	vfmadd231pd	%xmm3, %xmm1, %xmm6
	vmulsd	%xmm6, %xmm8, %xmm6
	vmovlpd	%xmm3, (%r11)
	vunpcklpd	%xmm3, %xmm3, %xmm3
	vmovlpd	%xmm6, 8(%r11)
	vunpcklpd	%xmm6, %xmm6, %xmm6
	vfmadd132pd	%xmm3, %xmm5, %xmm4
	vmovapd	%xmm9, %xmm5
	vmovsd	120(%rsi), %xmm1
	vfmadd132pd	%xmm6, %xmm4, %xmm5
	vmovsd	80(%rsi), %xmm4
	vdivsd	%xmm1, %xmm2, %xmm1
	vdivsd	%xmm4, %xmm2, %xmm4
	vperm2f128	$0, %ymm3, %ymm3, %ymm3
	vperm2f128	$0, %ymm6, %ymm6, %ymm6
	vfmadd231pd	(%rdi), %ymm3, %ymm0
	vfmadd132pd	32(%rdi), %ymm0, %ymm6
	vunpcklpd	%xmm7, %xmm5, %xmm0
	vshufpd	$1, %xmm7, %xmm5, %xmm5
	vmulsd	%xmm0, %xmm4, %xmm4
	vmovsd	88(%rsi), %xmm0
	vfmadd231pd	%xmm4, %xmm0, %xmm5
	vmulsd	%xmm5, %xmm1, %xmm0
	vmovlpd	%xmm4, 16(%r11)
	vunpcklpd	%xmm4, %xmm4, %xmm4
	vmovlpd	%xmm0, 24(%r11)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vperm2f128	$0, %ymm4, %ymm4, %ymm4
	vmovsd	128(%rdi), %xmm3
	vmovsd	168(%rdi), %xmm1
	vdivsd	%xmm3, %xmm2, %xmm3
	vfmadd132pd	64(%rdi), %ymm6, %ymm4
	vmovapd	144(%rdi), %xmm8
	vmovapd	176(%rdi), %xmm5
	vdivsd	%xmm1, %xmm2, %xmm1
	vperm2f128	$0, %ymm0, %ymm0, %ymm0
	vfmadd132pd	96(%rdi), %ymm4, %ymm0
	vunpcklpd	%xmm7, %xmm0, %xmm4
	vextractf128	$0x1, %ymm0, %xmm6
	vshufpd	$1, %xmm7, %xmm0, %xmm0
	vmulsd	%xmm4, %xmm3, %xmm3
	vmovsd	136(%rdi), %xmm4
	vfmadd132pd	%xmm3, %xmm0, %xmm4
	vmulsd	%xmm4, %xmm1, %xmm0
	vmovlpd	%xmm3, 32(%r11)
	vunpcklpd	%xmm3, %xmm3, %xmm3
	vmovlpd	%xmm0, 40(%r11)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm8, %xmm6, %xmm3
	vmovsd	208(%rdi), %xmm1
	vfmadd132pd	%xmm5, %xmm3, %xmm0
	vmovsd	248(%rdi), %xmm3
	vdivsd	%xmm1, %xmm2, %xmm1
	vdivsd	%xmm3, %xmm2, %xmm2
	vunpcklpd	%xmm7, %xmm0, %xmm3
	vshufpd	$1, %xmm7, %xmm0, %xmm0
	vmovsd	216(%rdi), %xmm7
	vmulsd	%xmm3, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm7, %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm1, 48(%r11)
	vmovlpd	%xmm2, 56(%r11)
	vzeroupper
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L11:
	.cfi_restore_state
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	(%r8), %xmm1
	vextractf128	$0x1, %ymm6, %xmm8
	vmovapd	16(%rsi), %xmm5
	vunpcklpd	%xmm3, %xmm6, %xmm4
	vshufpd	$1, %xmm3, %xmm6, %xmm6
	vmovsd	8(%r8), %xmm2
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovapd	48(%rsi), %xmm4
	vmovsd	8(%rsi), %xmm7
	vfmadd231pd	%xmm1, %xmm7, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm6
	vmovlpd	%xmm1, (%r11)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vmovlpd	%xmm6, 8(%r11)
	vfmadd132pd	%xmm1, %xmm8, %xmm5
	vunpcklpd	%xmm6, %xmm6, %xmm6
	vperm2f128	$0, %ymm1, %ymm1, %ymm1
	vmovsd	24(%r8), %xmm2
	vfmadd231pd	%xmm6, %xmm4, %xmm5
	vmovsd	16(%r8), %xmm4
	vfmadd231pd	(%rdi), %ymm1, %ymm0
	vunpcklpd	%xmm3, %xmm5, %xmm1
	vshufpd	$1, %xmm3, %xmm5, %xmm5
	vperm2f128	$0, %ymm6, %ymm6, %ymm6
	vmulsd	%xmm1, %xmm4, %xmm4
	vmovsd	88(%rsi), %xmm1
	vfmadd231pd	32(%rdi), %ymm6, %ymm0
	vfmadd231pd	%xmm4, %xmm1, %xmm5
	vmulsd	%xmm5, %xmm2, %xmm1
	vmovlpd	%xmm4, 16(%r11)
	vunpcklpd	%xmm4, %xmm4, %xmm4
	vmovlpd	%xmm1, 24(%r11)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vperm2f128	$0, %ymm4, %ymm4, %ymm4
	vmovsd	32(%r8), %xmm2
	vperm2f128	$0, %ymm1, %ymm1, %ymm1
	vmovapd	144(%rdi), %xmm7
	vfmadd231pd	64(%rdi), %ymm4, %ymm0
	vmovapd	176(%rdi), %xmm5
	vfmadd231pd	96(%rdi), %ymm1, %ymm0
	vunpcklpd	%xmm3, %xmm0, %xmm4
	vextractf128	$0x1, %ymm0, %xmm6
	vshufpd	$1, %xmm3, %xmm0, %xmm0
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	40(%r8), %xmm1
	vmovsd	136(%rdi), %xmm4
	vfmadd132pd	%xmm2, %xmm0, %xmm4
	vmulsd	%xmm4, %xmm1, %xmm0
	vmovlpd	%xmm2, 32(%r11)
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vmovlpd	%xmm0, 40(%r11)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm7, %xmm6, %xmm2
	vmovsd	48(%r8), %xmm1
	vfmadd132pd	%xmm5, %xmm2, %xmm0
	vunpcklpd	%xmm3, %xmm0, %xmm4
	vshufpd	$1, %xmm3, %xmm0, %xmm0
	vmovsd	56(%r8), %xmm2
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	216(%rdi), %xmm3
	vfmadd231pd	%xmm1, %xmm3, %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm1, 48(%r11)
	vmovlpd	%xmm2, 56(%r11)
	vzeroupper
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L6:
	.cfi_restore_state
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rax, %rdi
	vmovapd	%ymm6, %ymm0
	jmp	.L2
	.cfi_endproc
.LFE4586:
	.size	kernel_dtrsv_n_8_lib4_new, .-kernel_dtrsv_n_8_lib4_new
	.section	.text.unlikely
.LCOLDE1:
	.text
.LHOTE1:
	.section	.text.unlikely
.LCOLDB2:
	.text
.LHOTB2:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_4_lib4_new
	.type	kernel_dtrsv_n_4_lib4_new, @function
kernel_dtrsv_n_4_lib4_new:
.LFB4587:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$7, %edi
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x70
	jle	.L19
	leal	-8(%rdi), %ebx
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%r8, %rax
	shrl	$3, %ebx
	movl	%ebx, %r11d
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm3
	addq	$1, %r11
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm5
	movq	%r11, %r10
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm4
	salq	$8, %r10
	vmovapd	%ymm7, %ymm2
	addq	%rsi, %r10
	.p2align 4,,10
	.p2align 3
.L14:
	vbroadcastf128	(%rax), %ymm0
	addq	$256, %rsi
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-256(%rsi), %ymm9, %ymm2
	vfmadd231pd	-224(%rsi), %ymm0, %ymm4
	vbroadcastf128	16(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-192(%rsi), %ymm9, %ymm1
	vfmadd231pd	-160(%rsi), %ymm0, %ymm5
	vbroadcastf128	32(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%rsi), %ymm9, %ymm8
	vfmadd231pd	-96(%rsi), %ymm0, %ymm3
	vbroadcastf128	48(%rax), %ymm0
	addq	$64, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rsi), %ymm9, %ymm6
	vfmadd231pd	-32(%rsi), %ymm0, %ymm7
	cmpq	%r10, %rsi
	jne	.L14
	vaddpd	%ymm6, %ymm1, %ymm6
	salq	$6, %r11
	leal	8(,%rbx,8), %esi
	addq	%r11, %r8
	vaddpd	%ymm8, %ymm2, %ymm1
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm7, %ymm5, %ymm2
.L13:
	leal	-3(%rdi), %eax
	cmpl	%esi, %eax
	jle	.L20
	leal	-4(%rdi), %eax
	subl	%esi, %eax
	shrl	$2, %eax
	addq	$1, %rax
	salq	$7, %rax
	addq	%r10, %rax
	.p2align 4,,10
	.p2align 3
.L16:
	vbroadcastf128	(%r8), %ymm0
	subq	$-128, %r10
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%r10), %ymm4, %ymm1
	vfmadd231pd	-96(%r10), %ymm0, %ymm3
	vbroadcastf128	16(%r8), %ymm0
	addq	$32, %r8
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%r10), %ymm4, %ymm6
	vfmadd231pd	-32(%r10), %ymm0, %ymm2
	cmpq	%rax, %r10
	jne	.L16
.L15:
	vaddpd	%ymm2, %ymm3, %ymm3
	vmovupd	(%r9), %ymm0
	testl	%edx, %edx
	vaddpd	%ymm6, %ymm1, %ymm1
	vaddpd	%ymm3, %ymm1, %ymm1
	vsubpd	%ymm1, %ymm0, %ymm1
	vextractf128	$0x1, %ymm1, %xmm3
	jne	.L25
	vmovsd	(%rax), %xmm0
	vmovapd	.LC0(%rip), %xmm2
	vxorpd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm0, %xmm2, %xmm0
	vmovapd	16(%rax), %xmm7
	vmovapd	48(%rax), %xmm6
	vunpcklpd	%xmm4, %xmm1, %xmm5
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmulsd	%xmm5, %xmm0, %xmm0
	vmovsd	8(%rax), %xmm5
	vmovlpd	%xmm0, (%r9)
	vfmadd132pd	%xmm0, %xmm1, %xmm5
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd231pd	%xmm0, %xmm7, %xmm3
	vmovsd	40(%rax), %xmm0
	vdivsd	%xmm0, %xmm2, %xmm0
	vmulsd	%xmm5, %xmm0, %xmm1
	vunpcklpd	%xmm1, %xmm1, %xmm0
	vmovlpd	%xmm1, 8(%r9)
	vfmadd132pd	%xmm0, %xmm3, %xmm6
	vmovsd	80(%rax), %xmm0
	vunpcklpd	%xmm4, %xmm6, %xmm3
	vdivsd	%xmm0, %xmm2, %xmm0
	vshufpd	$1, %xmm4, %xmm6, %xmm1
	vmulsd	%xmm3, %xmm0, %xmm0
	vmovsd	88(%rax), %xmm3
	vmovlpd	%xmm0, 16(%r9)
	vfmadd231pd	%xmm0, %xmm3, %xmm1
	vmovsd	120(%rax), %xmm0
	vdivsd	%xmm0, %xmm2, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm2
	vmovlpd	%xmm2, 24(%r9)
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
.L25:
	.cfi_restore_state
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovsd	(%rcx), %xmm2
	vmovapd	16(%rax), %xmm6
	vunpcklpd	%xmm4, %xmm1, %xmm0
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmovapd	48(%rax), %xmm5
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovsd	8(%rax), %xmm0
	vmovlpd	%xmm2, (%r9)
	vfmadd231pd	%xmm2, %xmm0, %xmm1
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vmovsd	8(%rcx), %xmm0
	vfmadd132pd	%xmm6, %xmm3, %xmm2
	vmulsd	%xmm1, %xmm0, %xmm1
	vunpcklpd	%xmm1, %xmm1, %xmm0
	vmovlpd	%xmm1, 8(%r9)
	vfmadd132pd	%xmm0, %xmm2, %xmm5
	vunpcklpd	%xmm4, %xmm5, %xmm2
	vmovsd	16(%rcx), %xmm0
	vshufpd	$1, %xmm4, %xmm5, %xmm1
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovsd	88(%rax), %xmm2
	vmovlpd	%xmm0, 16(%r9)
	vfmadd231pd	%xmm0, %xmm2, %xmm1
	vmovsd	24(%rcx), %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovlpd	%xmm0, 24(%r9)
	vzeroupper
	popq	%rbx
	popq	%r10
	.cfi_remember_state
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
.L20:
	.cfi_restore_state
	movq	%r10, %rax
	jmp	.L15
.L19:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rsi, %r10
	xorl	%esi, %esi
	vmovapd	%ymm6, %ymm3
	vmovapd	%ymm6, %ymm1
	vmovapd	%ymm6, %ymm2
	jmp	.L13
	.cfi_endproc
.LFE4587:
	.size	kernel_dtrsv_n_4_lib4_new, .-kernel_dtrsv_n_4_lib4_new
	.section	.text.unlikely
.LCOLDE2:
	.text
.LHOTE2:
	.section	.text.unlikely
.LCOLDB3:
	.text
.LHOTB3:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_4_vs_lib4_new
	.type	kernel_dtrsv_n_4_vs_lib4_new, @function
kernel_dtrsv_n_4_vs_lib4_new:
.LFB4588:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$7, %edx
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
	movq	(%r10), %rax
	movq	8(%r10), %rbx
	jle	.L47
	leal	-8(%rdx), %r13d
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rcx, %r10
	shrl	$3, %r13d
	movl	%r13d, %r12d
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm3
	addq	$1, %r12
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm5
	movq	%r12, %r11
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm4
	salq	$6, %r11
	vmovapd	%ymm7, %ymm2
	addq	%rax, %r11
	.p2align 4,,10
	.p2align 3
.L28:
	vbroadcastf128	(%rax), %ymm0
	addq	$256, %r10
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-256(%r10), %ymm9, %ymm2
	vfmadd231pd	-224(%r10), %ymm0, %ymm4
	vbroadcastf128	16(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-192(%r10), %ymm9, %ymm1
	vfmadd231pd	-160(%r10), %ymm0, %ymm5
	vbroadcastf128	32(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%r10), %ymm9, %ymm8
	vfmadd231pd	-96(%r10), %ymm0, %ymm3
	vbroadcastf128	48(%rax), %ymm0
	addq	$64, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%r10), %ymm9, %ymm6
	vfmadd231pd	-32(%r10), %ymm0, %ymm7
	cmpq	%r11, %rax
	jne	.L28
	vaddpd	%ymm6, %ymm1, %ymm6
	salq	$8, %r12
	leal	8(,%r13,8), %eax
	addq	%r12, %rcx
	vaddpd	%ymm8, %ymm2, %ymm1
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm7, %ymm5, %ymm2
.L27:
	leal	-3(%rdx), %r10d
	cmpl	%eax, %r10d
	jle	.L29
	subl	$4, %edx
	subl	%eax, %edx
	movq	%rcx, %rax
	shrl	$2, %edx
	addq	$1, %rdx
	movq	%rdx, %r10
	salq	$5, %r10
	addq	%r11, %r10
	.p2align 4,,10
	.p2align 3
.L30:
	vbroadcastf128	(%r11), %ymm0
	subq	$-128, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%rax), %ymm4, %ymm1
	vfmadd231pd	-96(%rax), %ymm0, %ymm3
	vbroadcastf128	16(%r11), %ymm0
	addq	$32, %r11
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rax), %ymm4, %ymm6
	vfmadd231pd	-32(%rax), %ymm0, %ymm2
	cmpq	%r10, %r11
	jne	.L30
	salq	$7, %rdx
	addq	%rdx, %rcx
.L29:
	vaddpd	%ymm6, %ymm1, %ymm1
	vmovupd	(%rbx), %ymm0
	testl	%r8d, %r8d
	vaddpd	%ymm2, %ymm3, %ymm3
	vaddpd	%ymm3, %ymm1, %ymm1
	vsubpd	%ymm1, %ymm0, %ymm1
	vextractf128	$0x1, %ymm1, %xmm5
	je	.L31
	vxorpd	%xmm3, %xmm3, %xmm3
	cmpl	$1, %esi
	vmovsd	(%r9), %xmm2
	vmovsd	8(%rcx), %xmm4
	vunpcklpd	%xmm3, %xmm1, %xmm0
	vmovapd	48(%rcx), %xmm6
	vshufpd	$1, %xmm3, %xmm1, %xmm1
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovapd	16(%rcx), %xmm0
	vfmadd231pd	%xmm2, %xmm4, %xmm1
	vmovlpd	%xmm2, (%rbx)
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	%xmm0, %xmm5, %xmm2
	je	.L71
	vmovsd	8(%r9), %xmm0
	cmpl	$2, %esi
	vmulsd	%xmm1, %xmm0, %xmm1
	vmovlpd	%xmm1, 8(%rbx)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm6, %xmm2
	je	.L72
	vunpcklpd	%xmm3, %xmm2, %xmm1
	cmpl	$3, %esi
	vmovsd	16(%r9), %xmm0
	vshufpd	$1, %xmm3, %xmm2, %xmm2
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovsd	88(%rcx), %xmm1
	vmovlpd	%xmm0, 16(%rbx)
	vfmadd231pd	%xmm0, %xmm1, %xmm2
	je	.L73
	vmovsd	24(%r9), %xmm0
	vmulsd	%xmm2, %xmm0, %xmm2
.L69:
	vmovlpd	%xmm2, 24(%rbx)
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
.L31:
	.cfi_restore_state
	vmovsd	(%rcx), %xmm0
	cmpl	$1, %esi
	vmovapd	.LC0(%rip), %xmm3
	vxorpd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm0, %xmm3, %xmm0
	vmovapd	16(%rcx), %xmm6
	vmovapd	48(%rcx), %xmm7
	vunpcklpd	%xmm4, %xmm1, %xmm2
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovsd	8(%rcx), %xmm2
	vmovlpd	%xmm0, (%rbx)
	vfmadd231pd	%xmm0, %xmm2, %xmm1
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm6, %xmm5, %xmm0
	je	.L74
	vmovsd	40(%rcx), %xmm2
	cmpl	$2, %esi
	vdivsd	%xmm2, %xmm3, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm1
	vmovlpd	%xmm1, 8(%rbx)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm7, %xmm0
	je	.L75
	vmovsd	80(%rcx), %xmm1
	cmpl	$3, %esi
	vunpcklpd	%xmm4, %xmm0, %xmm2
	vdivsd	%xmm1, %xmm3, %xmm1
	vshufpd	$1, %xmm4, %xmm0, %xmm0
	vmulsd	%xmm2, %xmm1, %xmm1
	vmovsd	88(%rcx), %xmm2
	vmovlpd	%xmm1, 16(%rbx)
	vfmadd231pd	%xmm1, %xmm2, %xmm0
	je	.L76
	vmovsd	120(%rcx), %xmm1
	vdivsd	%xmm1, %xmm3, %xmm3
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, 24(%rbx)
.L65:
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
.L74:
	.cfi_restore_state
	cmpl	$1, %edi
	je	.L65
	cmpl	$2, %edi
	jle	.L67
	cmpl	$3, %edi
	vmovlpd	%xmm1, 8(%rbx)
	je	.L70
.L45:
	vmovaps	%xmm0, 16(%rbx)
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
.L71:
	.cfi_restore_state
	cmpl	$1, %edi
	je	.L65
	cmpl	$2, %edi
	jle	.L67
	cmpl	$3, %edi
	vmovlpd	%xmm1, 8(%rbx)
	je	.L68
.L39:
	vmovaps	%xmm2, 16(%rbx)
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
.L73:
	.cfi_restore_state
	cmpl	$3, %edi
	jne	.L69
	jmp	.L65
.L76:
	cmpl	$3, %edi
	je	.L65
	vmovlpd	%xmm0, 24(%rbx)
	jmp	.L65
.L72:
	cmpl	$2, %edi
	je	.L65
	cmpl	$3, %edi
	jne	.L39
.L68:
	vmovlpd	%xmm2, 16(%rbx)
	jmp	.L65
.L75:
	cmpl	$2, %edi
	je	.L65
	cmpl	$3, %edi
	jne	.L45
.L70:
	vmovlpd	%xmm0, 16(%rbx)
	jmp	.L65
.L47:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rax, %r11
	xorl	%eax, %eax
	vmovapd	%ymm6, %ymm3
	vmovapd	%ymm6, %ymm1
	vmovapd	%ymm6, %ymm2
	jmp	.L27
.L67:
	vmovlpd	%xmm1, 8(%rbx)
	jmp	.L65
	.cfi_endproc
.LFE4588:
	.size	kernel_dtrsv_n_4_vs_lib4_new, .-kernel_dtrsv_n_4_vs_lib4_new
	.section	.text.unlikely
.LCOLDE3:
	.text
.LHOTE3:
	.section	.text.unlikely
.LCOLDB10:
	.text
.LHOTB10:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_4_lib4_new
	.type	kernel_dtrsv_t_4_lib4_new, @function
kernel_dtrsv_t_4_lib4_new:
.LFB4589:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L96
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm6
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -80(%rbp)
	vmovsd	.LC6(%rip), %xmm4
	vmovsd	.LC7(%rip), %xmm6
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	%xmm4, -64(%rbp)
	vmovsd	%xmm6, -56(%rbp)
	vmovupd	-80(%rbp), %ymm9
	jg	.L97
	vxorpd	%xmm1, %xmm1, %xmm1
	testl	%ecx, %ecx
	vmovapd	%xmm1, %xmm5
	vmovapd	%xmm1, %xmm6
	vmovapd	%xmm1, %xmm2
	vhaddpd	%xmm1, %xmm1, %xmm1
	je	.L85
.L99:
	vmovsd	24(%r9), %xmm3
	vmovsd	16(%r9), %xmm0
	vsubsd	%xmm1, %xmm3, %xmm1
	vmovsd	24(%r8), %xmm3
	vmulsd	%xmm3, %xmm1, %xmm3
	vmovlpd	%xmm3, 24(%r9)
	vmovsd	88(%rsi), %xmm1
	vmovapd	%xmm1, %xmm4
	vfnmadd132sd	%xmm3, %xmm0, %xmm4
	vhaddpd	%xmm5, %xmm5, %xmm0
	vsubsd	%xmm0, %xmm4, %xmm1
	vmovsd	16(%r8), %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	(%r9), %xmm1
	vmovlpd	%xmm0, 16(%r9)
	vunpcklpd	%xmm3, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm3
	vfmadd231pd	16(%rsi), %xmm0, %xmm2
	vmovsd	8(%r9), %xmm0
	vhaddpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	48(%rsi), %xmm6, %xmm3
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm0, %xmm3
	vmovsd	8(%r8), %xmm0
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm3, %xmm1, %xmm0
	vsubsd	%xmm2, %xmm0, %xmm2
	vmovsd	(%r8), %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm2, (%r9)
	vzeroupper
.L93:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L96:
	rep ret
	.p2align 4,,10
	.p2align 3
.L97:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	leal	0(,%rdx,4), %ebx
	cmpl	$11, %edi
	leaq	32(%r9), %rdx
	movslq	%ebx, %rbx
	leaq	0(,%rbx,8), %r12
	leaq	(%rsi,%r12), %r14
	jle	.L89
	leal	-12(%rdi), %eax
	vxorpd	%xmm4, %xmm4, %xmm4
	leaq	(%r14,%r12), %r10
	salq	$4, %rbx
	movq	%rdx, %r11
	shrl	$3, %eax
	movl	%eax, %r15d
	movl	%eax, -100(%rbp)
	vmovapd	%ymm4, %ymm5
	movq	%r15, %rax
	vmovapd	%ymm4, %ymm3
	vmovapd	%ymm4, %ymm2
	salq	$6, %rax
	vmovapd	%ymm4, %ymm1
	vmovapd	%ymm4, %ymm8
	vmovapd	%ymm4, %ymm7
	vmovapd	%ymm4, %ymm6
	leaq	96(%r9,%rax), %r13
	movq	%r14, %rax
	.p2align 4,,10
	.p2align 3
.L81:
	vmovupd	(%r11), %ymm0
	addq	$64, %r11
	vfmadd231pd	(%rax), %ymm0, %ymm6
	vfmadd231pd	32(%rax), %ymm0, %ymm7
	vfmadd231pd	64(%rax), %ymm0, %ymm8
	vfmadd231pd	96(%rax), %ymm0, %ymm1
	vmovupd	-32(%r11), %ymm0
	addq	%rbx, %rax
	vfmadd231pd	(%r10), %ymm0, %ymm2
	vfmadd231pd	32(%r10), %ymm0, %ymm3
	vfmadd231pd	64(%r10), %ymm0, %ymm5
	vfmadd231pd	96(%r10), %ymm0, %ymm4
	addq	%rbx, %r10
	cmpq	%r11, %r13
	jne	.L81
	addq	$1, %r15
	movl	-100(%rbp), %eax
	imulq	%r15, %rbx
	vaddpd	%ymm5, %ymm8, %ymm5
	salq	$6, %r15
	vaddpd	%ymm3, %ymm7, %ymm3
	leal	12(,%rax,8), %r13d
	addq	%r15, %rdx
	vaddpd	%ymm2, %ymm6, %ymm2
	addq	%rbx, %r14
	vaddpd	%ymm1, %ymm4, %ymm1
.L80:
	leal	-3(%rdi), %eax
	cmpl	%r13d, %eax
	jle	.L90
	leal	-4(%rdi), %r11d
	movq	%r14, %rax
	subl	%r13d, %r11d
	shrl	$2, %r11d
	movl	%r11d, %ebx
	addq	$1, %rbx
	movq	%rbx, %r10
	salq	$5, %r10
	addq	%rdx, %r10
	.p2align 4,,10
	.p2align 3
.L83:
	vmovupd	(%rdx), %ymm0
	addq	$32, %rdx
	vfmadd231pd	(%rax), %ymm0, %ymm2
	vfmadd231pd	32(%rax), %ymm0, %ymm3
	vfmadd231pd	64(%rax), %ymm0, %ymm5
	vfmadd231pd	96(%rax), %ymm0, %ymm1
	addq	%r12, %rax
	cmpq	%r10, %rdx
	jne	.L83
	imulq	%rbx, %r12
	leal	4(%r13,%r11,4), %r13d
	addq	%r12, %r14
.L82:
	cmpl	%r13d, %edi
	jg	.L98
.L84:
	vextractf128	$0x1, %ymm1, %xmm4
	testl	%ecx, %ecx
	vextractf128	$0x1, %ymm2, %xmm0
	vextractf128	$0x1, %ymm3, %xmm6
	vaddpd	%xmm1, %xmm4, %xmm1
	vextractf128	$0x1, %ymm5, %xmm7
	vaddpd	%xmm2, %xmm0, %xmm2
	vaddpd	%xmm3, %xmm6, %xmm6
	vaddpd	%xmm5, %xmm7, %xmm5
	vhaddpd	%xmm1, %xmm1, %xmm1
	jne	.L99
.L85:
	vmovsd	120(%rsi), %xmm3
	vmovapd	.LC0(%rip), %xmm7
	vhaddpd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm3, %xmm7, %xmm3
	vmovsd	24(%r9), %xmm4
	vmovsd	16(%r9), %xmm0
	vsubsd	%xmm1, %xmm4, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm3
	vmovlpd	%xmm3, 24(%r9)
	vmovsd	88(%rsi), %xmm1
	vmovapd	%xmm1, %xmm4
	vfnmadd132sd	%xmm3, %xmm0, %xmm4
	vmovsd	80(%rsi), %xmm0
	vsubsd	%xmm5, %xmm4, %xmm1
	vdivsd	%xmm0, %xmm7, %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovlpd	%xmm0, 16(%r9)
	vunpcklpd	%xmm3, %xmm0, %xmm0
	vmovsd	40(%rsi), %xmm1
	vmovapd	%xmm0, %xmm3
	vfmadd231pd	16(%rsi), %xmm0, %xmm2
	vmovsd	8(%r9), %xmm0
	vdivsd	%xmm1, %xmm7, %xmm1
	vhaddpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	48(%rsi), %xmm6, %xmm3
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm0, %xmm3
	vmulsd	%xmm1, %xmm3, %xmm3
	vmovsd	(%r9), %xmm1
	vmovlpd	%xmm3, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm3, %xmm1, %xmm0
	vmovsd	(%rsi), %xmm1
	vsubsd	%xmm2, %xmm0, %xmm2
	vdivsd	%xmm1, %xmm7, %xmm7
	vmulsd	%xmm7, %xmm2, %xmm2
	vmovlpd	%xmm2, (%r9)
	vzeroupper
	jmp	.L93
	.p2align 4,,10
	.p2align 3
.L98:
	vxorpd	%xmm0, %xmm0, %xmm0
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm4
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	%xmm0, %xmm4, %xmm0
	vmovupd	(%r10), %ymm4
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm9, %ymm9
	vxorpd	%xmm0, %xmm0, %xmm0
	vblendvpd	%ymm9, %ymm0, %ymm4, %ymm9
	vfmadd231pd	(%r14), %ymm9, %ymm2
	vfmadd231pd	32(%r14), %ymm9, %ymm3
	vfmadd231pd	64(%r14), %ymm9, %ymm5
	vfmadd231pd	96(%r14), %ymm9, %ymm1
	jmp	.L84
.L90:
	movq	%rdx, %r10
	jmp	.L82
.L89:
	vxorpd	%xmm5, %xmm5, %xmm5
	movl	$4, %r13d
	vmovapd	%ymm5, %ymm3
	vmovapd	%ymm5, %ymm2
	vmovapd	%ymm5, %ymm1
	jmp	.L80
	.cfi_endproc
.LFE4589:
	.size	kernel_dtrsv_t_4_lib4_new, .-kernel_dtrsv_t_4_lib4_new
	.section	.text.unlikely
.LCOLDE10:
	.text
.LHOTE10:
	.section	.text.unlikely
.LCOLDB11:
	.text
.LHOTB11:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_3_lib4_new
	.type	kernel_dtrsv_t_3_lib4_new, @function
kernel_dtrsv_t_3_lib4_new:
.LFB4590:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L120
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm7
	vmovsd	.LC6(%rip), %xmm6
	vmovsd	%xmm7, -80(%rbp)
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -64(%rbp)
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	.LC7(%rip), %xmm7
	vmovsd	%xmm7, -56(%rbp)
	jg	.L121
	jne	.L114
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	24(%r9), %xmm4
	vmovsd	24(%rsi), %xmm3
	vmovsd	56(%rsi), %xmm2
	vmovsd	88(%rsi), %xmm0
	vfmadd132sd	%xmm4, %xmm1, %xmm3
	vfmadd132sd	%xmm4, %xmm1, %xmm2
	vfmadd132sd	%xmm4, %xmm1, %xmm0
.L108:
	testl	%ecx, %ecx
	vhaddpd	%xmm0, %xmm0, %xmm0
	jne	.L122
	vmovsd	80(%rsi), %xmm5
	vmovapd	.LC0(%rip), %xmm4
	vdivsd	%xmm5, %xmm4, %xmm5
	vmovsd	16(%r9), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmulsd	%xmm5, %xmm0, %xmm0
	vmovlpd	%xmm0, 16(%r9)
	vmovsd	16(%rsi), %xmm1
	vmulsd	%xmm0, %xmm1, %xmm5
	vmovsd	48(%rsi), %xmm1
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	40(%rsi), %xmm1
	vaddsd	%xmm5, %xmm3, %xmm3
	vdivsd	%xmm1, %xmm4, %xmm1
	vaddsd	%xmm0, %xmm2, %xmm2
	vhaddpd	%xmm3, %xmm3, %xmm3
	vhaddpd	%xmm2, %xmm2, %xmm2
	vmovsd	8(%r9), %xmm0
	vsubsd	%xmm2, %xmm0, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm2
	vmovsd	(%r9), %xmm1
	vmovlpd	%xmm2, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm2, %xmm1, %xmm0
	vmovsd	(%rsi), %xmm1
	vsubsd	%xmm3, %xmm0, %xmm3
	vdivsd	%xmm1, %xmm4, %xmm4
	vmulsd	%xmm4, %xmm3, %xmm3
	vmovlpd	%xmm3, (%r9)
.L117:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L120:
	rep ret
	.p2align 4,,10
	.p2align 3
.L122:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	vmovsd	16(%r9), %xmm4
	vsubsd	%xmm0, %xmm4, %xmm0
	vmovsd	16(%r8), %xmm4
	vmulsd	%xmm4, %xmm0, %xmm0
	vmovlpd	%xmm0, 16(%r9)
	vmovsd	48(%rsi), %xmm4
	vmovsd	16(%rsi), %xmm1
	vmulsd	%xmm0, %xmm4, %xmm4
	vmulsd	%xmm0, %xmm1, %xmm1
	vmovsd	8(%r9), %xmm0
	vaddsd	%xmm4, %xmm2, %xmm2
	vaddsd	%xmm1, %xmm3, %xmm3
	vmovsd	(%r9), %xmm1
	vhaddpd	%xmm2, %xmm2, %xmm2
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm2, %xmm0, %xmm2
	vmovsd	8(%r8), %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm2, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm2, %xmm1, %xmm0
	vsubsd	%xmm3, %xmm0, %xmm3
	vmovsd	(%r8), %xmm0
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, (%r9)
	jmp	.L117
	.p2align 4,,10
	.p2align 3
.L121:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovupd	(%r9), %ymm0
	leal	0(,%rdx,4), %ebx
	cmpl	$11, %edi
	vmovupd	-80(%rbp), %ymm7
	leaq	32(%r9), %r11
	vblendpd	$7, %ymm1, %ymm0, %ymm0
	movslq	%ebx, %rbx
	leaq	0(,%rbx,8), %r12
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm5
	vfmadd132pd	64(%rsi), %ymm1, %ymm0
	leaq	(%rsi,%r12), %r14
	vfmadd132pd	(%rsi), %ymm1, %ymm6
	vfmadd132pd	32(%rsi), %ymm1, %ymm5
	jle	.L112
	leal	-12(%rdi), %eax
	leaq	(%r14,%r12), %r10
	vmovapd	%ymm1, %ymm4
	vmovapd	%ymm1, %ymm2
	vmovapd	%ymm1, %ymm3
	salq	$4, %rbx
	shrl	$3, %eax
	movq	%r11, %rdx
	movl	%eax, %r15d
	movl	%eax, -100(%rbp)
	movq	%r15, %rax
	salq	$6, %rax
	leaq	96(%r9,%rax), %r13
	movq	%r14, %rax
	.p2align 4,,10
	.p2align 3
.L104:
	vmovupd	(%rdx), %ymm1
	addq	$64, %rdx
	vfmadd231pd	(%rax), %ymm1, %ymm6
	vfmadd231pd	32(%rax), %ymm1, %ymm5
	vfmadd231pd	64(%rax), %ymm1, %ymm0
	vmovupd	-32(%rdx), %ymm1
	addq	%rbx, %rax
	vfmadd231pd	(%r10), %ymm1, %ymm3
	vfmadd231pd	32(%r10), %ymm1, %ymm2
	vfmadd231pd	64(%r10), %ymm1, %ymm4
	addq	%rbx, %r10
	cmpq	%rdx, %r13
	jne	.L104
	addq	$1, %r15
	movl	-100(%rbp), %eax
	imulq	%r15, %rbx
	salq	$6, %r15
	addq	%r15, %r11
	leal	12(,%rax,8), %r13d
	addq	%rbx, %r14
.L103:
	leal	-3(%rdi), %eax
	vaddpd	%ymm3, %ymm6, %ymm3
	cmpl	%r13d, %eax
	vaddpd	%ymm2, %ymm5, %ymm2
	vaddpd	%ymm4, %ymm0, %ymm0
	jle	.L113
	leal	-4(%rdi), %r10d
	movq	%r14, %rax
	subl	%r13d, %r10d
	shrl	$2, %r10d
	movl	%r10d, %ebx
	addq	$1, %rbx
	movq	%rbx, %rdx
	salq	$5, %rdx
	addq	%r11, %rdx
	.p2align 4,,10
	.p2align 3
.L106:
	vmovupd	(%r11), %ymm1
	addq	$32, %r11
	vfmadd231pd	(%rax), %ymm1, %ymm3
	vfmadd231pd	32(%rax), %ymm1, %ymm2
	vfmadd231pd	64(%rax), %ymm1, %ymm0
	addq	%r12, %rax
	cmpq	%rdx, %r11
	jne	.L106
	imulq	%rbx, %r12
	leal	4(%r13,%r10,4), %r13d
	addq	%r12, %r14
.L105:
	cmpl	%r13d, %edi
	jg	.L123
.L107:
	vextractf128	$0x1, %ymm3, %xmm5
	vextractf128	$0x1, %ymm2, %xmm4
	vextractf128	$0x1, %ymm0, %xmm1
	vaddpd	%xmm3, %xmm5, %xmm3
	vaddpd	%xmm2, %xmm4, %xmm2
	vaddpd	%xmm0, %xmm1, %xmm0
	vzeroupper
	jmp	.L108
	.p2align 4,,10
	.p2align 3
.L123:
	vxorpd	%xmm1, %xmm1, %xmm1
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm4
	vcvtsi2sd	%edi, %xmm1, %xmm1
	vsubsd	%xmm1, %xmm4, %xmm1
	vmovupd	(%rdx), %ymm4
	vbroadcastsd	%xmm1, %ymm1
	vsubpd	%ymm1, %ymm7, %ymm7
	vxorpd	%xmm1, %xmm1, %xmm1
	vblendvpd	%ymm7, %ymm1, %ymm4, %ymm7
	vfmadd231pd	(%r14), %ymm7, %ymm3
	vfmadd231pd	32(%r14), %ymm7, %ymm2
	vfmadd231pd	64(%r14), %ymm7, %ymm0
	jmp	.L107
	.p2align 4,,10
	.p2align 3
.L114:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	jmp	.L108
.L112:
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$4, %r13d
	vmovapd	%ymm4, %ymm2
	vmovapd	%ymm4, %ymm3
	jmp	.L103
.L113:
	movq	%r11, %rdx
	jmp	.L105
	.cfi_endproc
.LFE4590:
	.size	kernel_dtrsv_t_3_lib4_new, .-kernel_dtrsv_t_3_lib4_new
	.section	.text.unlikely
.LCOLDE11:
	.text
.LHOTE11:
	.section	.text.unlikely
.LCOLDB12:
	.text
.LHOTB12:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_2_lib4_new
	.type	kernel_dtrsv_t_2_lib4_new, @function
kernel_dtrsv_t_2_lib4_new:
.LFB4591:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L145
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm6
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -80(%rbp)
	vmovsd	.LC6(%rip), %xmm5
	vmovsd	.LC7(%rip), %xmm6
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	%xmm5, -64(%rbp)
	vmovsd	%xmm6, -56(%rbp)
	jg	.L146
	cmpl	$2, %edi
	jle	.L139
	vxorpd	%xmm2, %xmm2, %xmm2
	cmpl	$4, %edi
	vmovsd	16(%r9), %xmm3
	vmovsd	16(%rsi), %xmm1
	vmovsd	48(%rsi), %xmm0
	vfmadd132sd	%xmm3, %xmm2, %xmm1
	vfmadd132sd	%xmm3, %xmm2, %xmm0
	jne	.L132
	vmovsd	24(%r9), %xmm3
	vmovsd	24(%rsi), %xmm2
	vfmadd132sd	%xmm3, %xmm1, %xmm2
	vmovapd	%xmm2, %xmm1
	vmovsd	56(%rsi), %xmm2
	vfmadd132sd	%xmm3, %xmm0, %xmm2
	vmovapd	%xmm2, %xmm0
.L132:
	testl	%ecx, %ecx
	vhaddpd	%xmm0, %xmm0, %xmm0
	je	.L134
	vmovsd	8(%r9), %xmm2
	vhaddpd	%xmm1, %xmm1, %xmm1
	vmovsd	(%r9), %xmm3
	vsubsd	%xmm0, %xmm2, %xmm0
	vmovsd	8(%r8), %xmm2
	vmulsd	%xmm2, %xmm0, %xmm2
	vmovlpd	%xmm2, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm2, %xmm3, %xmm0
	vsubsd	%xmm1, %xmm0, %xmm1
	vmovsd	(%r8), %xmm0
	vmulsd	%xmm0, %xmm1, %xmm1
	vmovlpd	%xmm1, (%r9)
.L142:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L145:
	rep ret
	.p2align 4,,10
	.p2align 3
.L134:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	vmovsd	40(%rsi), %xmm2
	vmovapd	.LC0(%rip), %xmm3
	vhaddpd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm2, %xmm3, %xmm4
	vmovsd	8(%r9), %xmm2
	vsubsd	%xmm0, %xmm2, %xmm0
	vmulsd	%xmm4, %xmm0, %xmm2
	vmovsd	(%r9), %xmm4
	vmovlpd	%xmm2, 8(%r9)
	vmovsd	8(%rsi), %xmm0
	vfnmadd132sd	%xmm2, %xmm4, %xmm0
	vmovsd	(%rsi), %xmm2
	vsubsd	%xmm1, %xmm0, %xmm1
	vdivsd	%xmm2, %xmm3, %xmm3
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovlpd	%xmm1, (%r9)
	jmp	.L142
	.p2align 4,,10
	.p2align 3
.L146:
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovupd	(%r9), %ymm0
	leal	0(,%rdx,4), %ebx
	cmpl	$11, %edi
	vmovupd	-80(%rbp), %ymm5
	leaq	32(%r9), %r10
	vblendpd	$3, %ymm2, %ymm0, %ymm0
	movslq	%ebx, %rbx
	leaq	0(,%rbx,8), %r11
	vmovapd	%ymm0, %ymm1
	vfmadd132pd	32(%rsi), %ymm2, %ymm0
	leaq	(%rsi,%r11), %r14
	vfmadd132pd	(%rsi), %ymm2, %ymm1
	jle	.L137
	leal	-12(%rdi), %r15d
	vmovapd	%ymm2, %ymm3
	vmovapd	%ymm2, %ymm4
	salq	$4, %rbx
	movq	%r10, %rdx
	shrl	$3, %r15d
	movl	%r15d, %r13d
	movq	%r13, %rax
	salq	$6, %rax
	leaq	96(%r9,%rax), %r12
	movq	%r14, %rax
	.p2align 4,,10
	.p2align 3
.L128:
	vmovupd	(%rdx), %ymm2
	addq	$64, %rdx
	vfmadd231pd	(%rax), %ymm2, %ymm1
	vfmadd231pd	32(%rax), %ymm2, %ymm0
	vmovupd	-32(%rdx), %ymm2
	vfmadd231pd	(%rax,%r11), %ymm2, %ymm4
	vfmadd231pd	32(%rax,%r11), %ymm2, %ymm3
	addq	%rbx, %rax
	cmpq	%rdx, %r12
	jne	.L128
	leaq	(%r11,%r11), %rax
	addq	$1, %r13
	imulq	%r13, %rax
	salq	$6, %r13
	addq	%r13, %r10
	leal	12(,%r15,8), %r13d
	addq	%rax, %r14
.L127:
	leal	-3(%rdi), %eax
	vaddpd	%ymm4, %ymm1, %ymm1
	cmpl	%r13d, %eax
	vaddpd	%ymm3, %ymm0, %ymm0
	jle	.L138
	leal	-4(%rdi), %ebx
	movq	%r14, %rax
	subl	%r13d, %ebx
	shrl	$2, %ebx
	movl	%ebx, %r12d
	addq	$1, %r12
	movq	%r12, %rdx
	salq	$5, %rdx
	addq	%r10, %rdx
	.p2align 4,,10
	.p2align 3
.L130:
	vmovupd	(%r10), %ymm2
	addq	$32, %r10
	vfmadd231pd	(%rax), %ymm2, %ymm1
	vfmadd231pd	32(%rax), %ymm2, %ymm0
	addq	%r11, %rax
	cmpq	%rdx, %r10
	jne	.L130
	imulq	%r12, %r11
	leal	4(%r13,%rbx,4), %r13d
	addq	%r11, %r14
.L129:
	cmpl	%r13d, %edi
	jg	.L147
.L131:
	vextractf128	$0x1, %ymm1, %xmm3
	vextractf128	$0x1, %ymm0, %xmm2
	vaddpd	%xmm1, %xmm3, %xmm1
	vaddpd	%xmm0, %xmm2, %xmm0
	vzeroupper
	jmp	.L132
	.p2align 4,,10
	.p2align 3
.L147:
	vxorpd	%xmm3, %xmm3, %xmm3
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm2
	vxorpd	%xmm4, %xmm4, %xmm4
	vcvtsi2sd	%edi, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm2, %xmm2
	vmovupd	(%rdx), %ymm3
	vbroadcastsd	%xmm2, %ymm2
	vsubpd	%ymm2, %ymm5, %ymm2
	vblendvpd	%ymm2, %ymm4, %ymm3, %ymm2
	vfmadd231pd	(%r14), %ymm2, %ymm1
	vfmadd231pd	32(%r14), %ymm2, %ymm0
	jmp	.L131
	.p2align 4,,10
	.p2align 3
.L139:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	jmp	.L132
.L137:
	vxorpd	%xmm3, %xmm3, %xmm3
	movl	$4, %r13d
	vmovapd	%ymm3, %ymm4
	jmp	.L127
.L138:
	movq	%r10, %rdx
	jmp	.L129
	.cfi_endproc
.LFE4591:
	.size	kernel_dtrsv_t_2_lib4_new, .-kernel_dtrsv_t_2_lib4_new
	.section	.text.unlikely
.LCOLDE12:
	.text
.LHOTE12:
	.section	.text.unlikely
.LCOLDB13:
	.text
.LHOTB13:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_1_lib4_new
	.type	kernel_dtrsv_t_1_lib4_new, @function
kernel_dtrsv_t_1_lib4_new:
.LFB4592:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L170
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm7
	vmovsd	.LC5(%rip), %xmm3
	vmovsd	%xmm7, -80(%rbp)
	vmovsd	%xmm3, -72(%rbp)
	vmovsd	.LC6(%rip), %xmm7
	vmovsd	.LC7(%rip), %xmm3
	vmovsd	%xmm7, -64(%rbp)
	vmovsd	%xmm3, -56(%rbp)
	jg	.L171
	cmpl	$1, %edi
	je	.L163
	leal	-2(%rdi), %edx
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	addq	$1, %rdx
	.p2align 4,,10
	.p2align 3
.L157:
	vmovsd	8(%rsi,%rax,8), %xmm1
	vmovsd	8(%r9,%rax,8), %xmm2
	addq	$1, %rax
	cmpq	%rdx, %rax
	vfmadd132sd	%xmm2, %xmm0, %xmm1
	vmovapd	%xmm1, %xmm0
	jne	.L157
.L156:
	testl	%ecx, %ecx
	vhaddpd	%xmm0, %xmm0, %xmm0
	je	.L158
	vmovsd	(%r9), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmovsd	(%r8), %xmm1
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovlpd	%xmm0, (%r9)
.L167:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L170:
	rep ret
	.p2align 4,,10
	.p2align 3
.L158:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	vmovsd	(%rsi), %xmm2
	vmovapd	.LC0(%rip), %xmm1
	vdivsd	%xmm2, %xmm1, %xmm2
	vmovsd	(%r9), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovlpd	%xmm0, (%r9)
	jmp	.L167
	.p2align 4,,10
	.p2align 3
.L171:
	vxorpd	%xmm1, %xmm1, %xmm1
	leal	0(,%rdx,4), %ebx
	vmovupd	(%r9), %ymm0
	cmpl	$11, %edi
	leaq	32(%r9), %rdx
	movslq	%ebx, %rbx
	vmovupd	-80(%rbp), %ymm2
	vblendpd	$1, %ymm1, %ymm0, %ymm0
	leaq	0(,%rbx,8), %r11
	vfmadd132pd	(%rsi), %ymm1, %ymm0
	leaq	(%rsi,%r11), %r14
	jle	.L161
	leal	-12(%rdi), %r15d
	salq	$4, %rbx
	movq	%r14, %r10
	shrl	$3, %r15d
	movl	%r15d, %r13d
	movq	%r13, %rax
	salq	$6, %rax
	leaq	96(%r9,%rax), %r12
	movq	%rdx, %rax
	.p2align 4,,10
	.p2align 3
.L152:
	vmovapd	(%r10), %ymm4
	addq	$64, %rax
	vmovapd	(%r10,%r11), %ymm5
	addq	%rbx, %r10
	vfmadd231pd	-64(%rax), %ymm4, %ymm0
	vfmadd231pd	-32(%rax), %ymm5, %ymm1
	cmpq	%rax, %r12
	jne	.L152
	leaq	(%r11,%r11), %rax
	addq	$1, %r13
	imulq	%r13, %rax
	salq	$6, %r13
	addq	%r13, %rdx
	leal	12(,%r15,8), %r13d
	addq	%rax, %r14
.L151:
	leal	-3(%rdi), %eax
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%r13d, %eax
	jle	.L162
	leal	-4(%rdi), %ebx
	movq	%r14, %rax
	subl	%r13d, %ebx
	shrl	$2, %ebx
	movl	%ebx, %r12d
	addq	$1, %r12
	movq	%r12, %r10
	salq	$5, %r10
	addq	%rdx, %r10
	.p2align 4,,10
	.p2align 3
.L154:
	vmovapd	(%rax), %ymm6
	addq	$32, %rdx
	addq	%r11, %rax
	vfmadd231pd	-32(%rdx), %ymm6, %ymm0
	cmpq	%r10, %rdx
	jne	.L154
	imulq	%r12, %r11
	leal	4(%r13,%rbx,4), %r13d
	addq	%r11, %r14
.L153:
	cmpl	%r13d, %edi
	jg	.L172
.L155:
	vextractf128	$0x1, %ymm0, %xmm1
	vaddpd	%xmm0, %xmm1, %xmm0
	vzeroupper
	jmp	.L156
	.p2align 4,,10
	.p2align 3
.L172:
	vxorpd	%xmm3, %xmm3, %xmm3
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm1
	vcvtsi2sd	%edi, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm1, %xmm1
	vmovupd	(%r10), %ymm3
	vbroadcastsd	%xmm1, %ymm1
	vsubpd	%ymm1, %ymm2, %ymm2
	vxorpd	%xmm1, %xmm1, %xmm1
	vblendvpd	%ymm2, %ymm1, %ymm3, %ymm2
	vfmadd231pd	(%r14), %ymm2, %ymm0
	jmp	.L155
.L163:
	vxorpd	%xmm0, %xmm0, %xmm0
	jmp	.L156
.L161:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$4, %r13d
	jmp	.L151
.L162:
	movq	%rdx, %r10
	jmp	.L153
	.cfi_endproc
.LFE4592:
	.size	kernel_dtrsv_t_1_lib4_new, .-kernel_dtrsv_t_1_lib4_new
	.section	.text.unlikely
.LCOLDE13:
	.text
.LHOTE13:
	.section	.text.unlikely
.LCOLDB14:
	.text
.LHOTB14:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_8_lib4
	.type	kernel_dtrsv_n_8_lib4, @function
kernel_dtrsv_n_8_lib4:
.LFB4593:
	.cfi_startproc
	sall	$2, %ecx
	cmpl	$7, %edi
	movslq	%ecx, %rcx
	leaq	(%rdx,%rcx,8), %rax
	jle	.L178
	leal	-8(%rdi), %r10d
	vxorpd	%xmm0, %xmm0, %xmm0
	movq	%rdx, %rcx
	shrl	$3, %r10d
	addq	$1, %r10
	vmovapd	%ymm0, %ymm5
	vmovapd	%ymm0, %ymm1
	salq	$8, %r10
	vmovapd	%ymm0, %ymm4
	vmovapd	%ymm0, %ymm8
	leaq	(%rax,%r10), %rdi
	vmovapd	%ymm0, %ymm9
	vmovapd	%ymm0, %ymm7
	vmovapd	%ymm0, %ymm6
	.p2align 4,,10
	.p2align 3
.L175:
	vbroadcastf128	(%r8), %ymm3
	addq	$256, %rax
	addq	$256, %rcx
	vunpcklpd	%ymm3, %ymm3, %ymm2
	vunpckhpd	%ymm3, %ymm3, %ymm3
	vfmadd231pd	-256(%rcx), %ymm2, %ymm6
	vfmadd231pd	-256(%rax), %ymm2, %ymm4
	vbroadcastf128	16(%r8), %ymm2
	vfmadd231pd	-224(%rcx), %ymm3, %ymm7
	vfmadd132pd	-224(%rax), %ymm1, %ymm3
	vunpcklpd	%ymm2, %ymm2, %ymm1
	vunpckhpd	%ymm2, %ymm2, %ymm2
	vfmadd231pd	-192(%rcx), %ymm1, %ymm9
	vfmadd231pd	-192(%rax), %ymm1, %ymm5
	vbroadcastf128	32(%r8), %ymm1
	vfmadd231pd	-160(%rcx), %ymm2, %ymm8
	vfmadd132pd	-160(%rax), %ymm0, %ymm2
	vunpcklpd	%ymm1, %ymm1, %ymm0
	vunpckhpd	%ymm1, %ymm1, %ymm1
	vfmadd231pd	-128(%rcx), %ymm0, %ymm6
	vfmadd231pd	-128(%rax), %ymm0, %ymm4
	vbroadcastf128	48(%r8), %ymm0
	vfmadd231pd	-96(%rcx), %ymm1, %ymm7
	vfmadd132pd	-96(%rax), %ymm3, %ymm1
	addq	$64, %r8
	vunpcklpd	%ymm0, %ymm0, %ymm3
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rax), %ymm3, %ymm5
	vfmadd231pd	-64(%rcx), %ymm3, %ymm9
	vfmadd231pd	-32(%rcx), %ymm0, %ymm8
	vfmadd132pd	-32(%rax), %ymm2, %ymm0
	cmpq	%rdi, %rax
	jne	.L175
	vaddpd	%ymm0, %ymm1, %ymm0
	addq	%r10, %rdx
	vaddpd	%ymm5, %ymm4, %ymm4
	vaddpd	%ymm9, %ymm6, %ymm6
	vaddpd	%ymm8, %ymm7, %ymm7
	vaddpd	%ymm4, %ymm0, %ymm0
	vaddpd	%ymm7, %ymm6, %ymm6
.L174:
	vmovupd	(%r9), %ymm9
	testl	%esi, %esi
	vmovupd	32(%r9), %ymm5
	vsubpd	%ymm6, %ymm9, %ymm6
	vsubpd	%ymm0, %ymm5, %ymm0
	jne	.L181
	vmovsd	(%rdx), %xmm3
	vmovapd	.LC0(%rip), %xmm2
	vmovapd	%xmm6, %xmm1
	vextractf128	$0x1, %ymm6, %xmm5
	vdivsd	%xmm3, %xmm2, %xmm3
	vmovapd	48(%rdx), %xmm9
	vmovsd	40(%rdx), %xmm6
	vxorpd	%xmm7, %xmm7, %xmm7
	vdivsd	%xmm6, %xmm2, %xmm8
	vunpcklpd	%xmm7, %xmm1, %xmm4
	vshufpd	$1, %xmm7, %xmm1, %xmm6
	vmovsd	8(%rdx), %xmm1
	vmulsd	%xmm4, %xmm3, %xmm3
	vmovapd	16(%rdx), %xmm4
	vfmadd231pd	%xmm3, %xmm1, %xmm6
	vmulsd	%xmm6, %xmm8, %xmm6
	vmovlpd	%xmm3, (%r9)
	vunpcklpd	%xmm3, %xmm3, %xmm3
	vmovlpd	%xmm6, 8(%r9)
	vunpcklpd	%xmm6, %xmm6, %xmm6
	vfmadd132pd	%xmm3, %xmm5, %xmm4
	vmovapd	%xmm9, %xmm5
	vmovsd	120(%rdx), %xmm1
	vfmadd132pd	%xmm6, %xmm4, %xmm5
	vmovsd	80(%rdx), %xmm4
	vdivsd	%xmm1, %xmm2, %xmm1
	vdivsd	%xmm4, %xmm2, %xmm4
	vperm2f128	$0, %ymm3, %ymm3, %ymm3
	vperm2f128	$0, %ymm6, %ymm6, %ymm6
	vfmadd231pd	(%rdi), %ymm3, %ymm0
	vfmadd132pd	32(%rdi), %ymm0, %ymm6
	vunpcklpd	%xmm7, %xmm5, %xmm0
	vshufpd	$1, %xmm7, %xmm5, %xmm5
	vmulsd	%xmm0, %xmm4, %xmm4
	vmovsd	88(%rdx), %xmm0
	vfmadd231pd	%xmm4, %xmm0, %xmm5
	vmulsd	%xmm5, %xmm1, %xmm0
	vmovlpd	%xmm4, 16(%r9)
	vunpcklpd	%xmm4, %xmm4, %xmm4
	vmovlpd	%xmm0, 24(%r9)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vperm2f128	$0, %ymm4, %ymm4, %ymm4
	vmovsd	128(%rdi), %xmm3
	vmovsd	168(%rdi), %xmm1
	vdivsd	%xmm3, %xmm2, %xmm3
	vfmadd132pd	64(%rdi), %ymm6, %ymm4
	vmovapd	144(%rdi), %xmm8
	vmovapd	176(%rdi), %xmm5
	vdivsd	%xmm1, %xmm2, %xmm1
	vperm2f128	$0, %ymm0, %ymm0, %ymm0
	vfmadd132pd	96(%rdi), %ymm4, %ymm0
	vunpcklpd	%xmm7, %xmm0, %xmm4
	vextractf128	$0x1, %ymm0, %xmm6
	vshufpd	$1, %xmm7, %xmm0, %xmm0
	vmulsd	%xmm4, %xmm3, %xmm3
	vmovsd	136(%rdi), %xmm4
	vfmadd132pd	%xmm3, %xmm0, %xmm4
	vmulsd	%xmm4, %xmm1, %xmm0
	vmovlpd	%xmm3, 32(%r9)
	vunpcklpd	%xmm3, %xmm3, %xmm3
	vmovlpd	%xmm0, 40(%r9)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm8, %xmm6, %xmm3
	vmovsd	208(%rdi), %xmm1
	vfmadd132pd	%xmm5, %xmm3, %xmm0
	vmovsd	248(%rdi), %xmm3
	vdivsd	%xmm1, %xmm2, %xmm1
	vdivsd	%xmm3, %xmm2, %xmm2
	vunpcklpd	%xmm7, %xmm0, %xmm3
	vshufpd	$1, %xmm7, %xmm0, %xmm0
	vmovsd	216(%rdi), %xmm7
	vmulsd	%xmm3, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm7, %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm1, 48(%r9)
	vmovlpd	%xmm2, 56(%r9)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L181:
	vxorpd	%xmm3, %xmm3, %xmm3
	vmovsd	(%rdx), %xmm1
	vextractf128	$0x1, %ymm6, %xmm8
	vmovapd	16(%rdx), %xmm5
	vunpcklpd	%xmm3, %xmm6, %xmm4
	vshufpd	$1, %xmm3, %xmm6, %xmm6
	vmovsd	40(%rdx), %xmm2
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovapd	48(%rdx), %xmm4
	vmovsd	8(%rdx), %xmm7
	vfmadd231pd	%xmm1, %xmm7, %xmm6
	vmulsd	%xmm6, %xmm2, %xmm6
	vmovlpd	%xmm1, (%r9)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vmovlpd	%xmm6, 8(%r9)
	vfmadd132pd	%xmm1, %xmm8, %xmm5
	vunpcklpd	%xmm6, %xmm6, %xmm6
	vperm2f128	$0, %ymm1, %ymm1, %ymm1
	vmovsd	120(%rdx), %xmm2
	vfmadd231pd	%xmm6, %xmm4, %xmm5
	vmovsd	80(%rdx), %xmm4
	vfmadd231pd	(%rdi), %ymm1, %ymm0
	vunpcklpd	%xmm3, %xmm5, %xmm1
	vshufpd	$1, %xmm3, %xmm5, %xmm5
	vperm2f128	$0, %ymm6, %ymm6, %ymm6
	vmulsd	%xmm1, %xmm4, %xmm4
	vmovsd	88(%rdx), %xmm1
	vfmadd231pd	32(%rdi), %ymm6, %ymm0
	vfmadd231pd	%xmm4, %xmm1, %xmm5
	vmulsd	%xmm5, %xmm2, %xmm1
	vmovlpd	%xmm4, 16(%r9)
	vunpcklpd	%xmm4, %xmm4, %xmm4
	vmovlpd	%xmm1, 24(%r9)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vperm2f128	$0, %ymm4, %ymm4, %ymm4
	vmovsd	128(%rdi), %xmm2
	vperm2f128	$0, %ymm1, %ymm1, %ymm1
	vmovapd	144(%rdi), %xmm7
	vfmadd231pd	64(%rdi), %ymm4, %ymm0
	vmovapd	176(%rdi), %xmm5
	vfmadd231pd	96(%rdi), %ymm1, %ymm0
	vunpcklpd	%xmm3, %xmm0, %xmm4
	vextractf128	$0x1, %ymm0, %xmm6
	vshufpd	$1, %xmm3, %xmm0, %xmm0
	vmulsd	%xmm4, %xmm2, %xmm2
	vmovsd	168(%rdi), %xmm1
	vmovsd	136(%rdi), %xmm4
	vfmadd132pd	%xmm2, %xmm0, %xmm4
	vmulsd	%xmm4, %xmm1, %xmm0
	vmovlpd	%xmm2, 32(%r9)
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vmovlpd	%xmm0, 40(%r9)
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm7, %xmm6, %xmm2
	vmovsd	208(%rdi), %xmm1
	vfmadd132pd	%xmm5, %xmm2, %xmm0
	vunpcklpd	%xmm3, %xmm0, %xmm4
	vshufpd	$1, %xmm3, %xmm0, %xmm0
	vmovsd	248(%rdi), %xmm2
	vmulsd	%xmm4, %xmm1, %xmm1
	vmovsd	216(%rdi), %xmm3
	vfmadd231pd	%xmm1, %xmm3, %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm1, 48(%r9)
	vmovlpd	%xmm2, 56(%r9)
	vzeroupper
	ret
.L178:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rax, %rdi
	vmovapd	%ymm6, %ymm0
	jmp	.L174
	.cfi_endproc
.LFE4593:
	.size	kernel_dtrsv_n_8_lib4, .-kernel_dtrsv_n_8_lib4
	.section	.text.unlikely
.LCOLDE14:
	.text
.LHOTE14:
	.section	.text.unlikely
.LCOLDB15:
	.text
.LHOTB15:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_4_lib4
	.type	kernel_dtrsv_n_4_lib4, @function
kernel_dtrsv_n_4_lib4:
.LFB4594:
	.cfi_startproc
	cmpl	$7, %edi
	jle	.L189
	leal	-8(%rdi), %r11d
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%rcx, %rax
	shrl	$3, %r11d
	movl	%r11d, %r10d
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm3
	addq	$1, %r10
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm5
	movq	%r10, %r9
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm4
	salq	$8, %r9
	vmovapd	%ymm7, %ymm2
	addq	%rdx, %r9
	.p2align 4,,10
	.p2align 3
.L184:
	vbroadcastf128	(%rax), %ymm0
	addq	$256, %rdx
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-256(%rdx), %ymm9, %ymm2
	vfmadd231pd	-224(%rdx), %ymm0, %ymm4
	vbroadcastf128	16(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-192(%rdx), %ymm9, %ymm1
	vfmadd231pd	-160(%rdx), %ymm0, %ymm5
	vbroadcastf128	32(%rax), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%rdx), %ymm9, %ymm8
	vfmadd231pd	-96(%rdx), %ymm0, %ymm3
	vbroadcastf128	48(%rax), %ymm0
	addq	$64, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rdx), %ymm9, %ymm6
	vfmadd231pd	-32(%rdx), %ymm0, %ymm7
	cmpq	%r9, %rdx
	jne	.L184
	vaddpd	%ymm6, %ymm1, %ymm6
	salq	$6, %r10
	leal	8(,%r11,8), %edx
	addq	%r10, %rcx
	vaddpd	%ymm8, %ymm2, %ymm1
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm7, %ymm5, %ymm2
.L183:
	leal	-3(%rdi), %eax
	cmpl	%edx, %eax
	jle	.L190
	leal	-4(%rdi), %eax
	subl	%edx, %eax
	shrl	$2, %eax
	addq	$1, %rax
	salq	$7, %rax
	addq	%r9, %rax
	.p2align 4,,10
	.p2align 3
.L186:
	vbroadcastf128	(%rcx), %ymm0
	subq	$-128, %r9
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%r9), %ymm4, %ymm1
	vfmadd231pd	-96(%r9), %ymm0, %ymm3
	vbroadcastf128	16(%rcx), %ymm0
	addq	$32, %rcx
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%r9), %ymm4, %ymm6
	vfmadd231pd	-32(%r9), %ymm0, %ymm2
	cmpq	%rax, %r9
	jne	.L186
.L185:
	vaddpd	%ymm2, %ymm3, %ymm3
	vmovupd	(%r8), %ymm0
	testl	%esi, %esi
	vaddpd	%ymm6, %ymm1, %ymm1
	vaddpd	%ymm3, %ymm1, %ymm1
	vsubpd	%ymm1, %ymm0, %ymm1
	vextractf128	$0x1, %ymm1, %xmm3
	jne	.L194
	vmovsd	(%rax), %xmm0
	vmovapd	.LC0(%rip), %xmm2
	vxorpd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm0, %xmm2, %xmm0
	vmovapd	16(%rax), %xmm7
	vmovapd	48(%rax), %xmm6
	vunpcklpd	%xmm4, %xmm1, %xmm5
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmulsd	%xmm5, %xmm0, %xmm0
	vmovsd	8(%rax), %xmm5
	vmovlpd	%xmm0, (%r8)
	vfmadd132pd	%xmm0, %xmm1, %xmm5
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd231pd	%xmm0, %xmm7, %xmm3
	vmovsd	40(%rax), %xmm0
	vdivsd	%xmm0, %xmm2, %xmm0
	vmulsd	%xmm5, %xmm0, %xmm1
	vunpcklpd	%xmm1, %xmm1, %xmm0
	vmovlpd	%xmm1, 8(%r8)
	vfmadd132pd	%xmm0, %xmm3, %xmm6
	vmovsd	80(%rax), %xmm0
	vunpcklpd	%xmm4, %xmm6, %xmm3
	vdivsd	%xmm0, %xmm2, %xmm0
	vshufpd	$1, %xmm4, %xmm6, %xmm1
	vmulsd	%xmm3, %xmm0, %xmm0
	vmovsd	88(%rax), %xmm3
	vmovlpd	%xmm0, 16(%r8)
	vfmadd231pd	%xmm0, %xmm3, %xmm1
	vmovsd	120(%rax), %xmm0
	vdivsd	%xmm0, %xmm2, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm2
	vmovlpd	%xmm2, 24(%r8)
	vzeroupper
	ret
	.p2align 4,,10
	.p2align 3
.L194:
	vxorpd	%xmm4, %xmm4, %xmm4
	vmovsd	(%rax), %xmm2
	vmovapd	16(%rax), %xmm6
	vunpcklpd	%xmm4, %xmm1, %xmm0
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmovapd	48(%rax), %xmm5
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovsd	8(%rax), %xmm0
	vmovlpd	%xmm2, (%r8)
	vfmadd231pd	%xmm2, %xmm0, %xmm1
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vmovsd	40(%rax), %xmm0
	vfmadd132pd	%xmm6, %xmm3, %xmm2
	vmulsd	%xmm1, %xmm0, %xmm1
	vunpcklpd	%xmm1, %xmm1, %xmm0
	vmovlpd	%xmm1, 8(%r8)
	vfmadd132pd	%xmm0, %xmm2, %xmm5
	vunpcklpd	%xmm4, %xmm5, %xmm2
	vmovsd	80(%rax), %xmm0
	vshufpd	$1, %xmm4, %xmm5, %xmm1
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovsd	88(%rax), %xmm2
	vmovlpd	%xmm0, 16(%r8)
	vfmadd231pd	%xmm0, %xmm2, %xmm1
	vmovsd	120(%rax), %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovlpd	%xmm0, 24(%r8)
	vzeroupper
	ret
.L190:
	movq	%r9, %rax
	jmp	.L185
.L189:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%rdx, %r9
	xorl	%edx, %edx
	vmovapd	%ymm6, %ymm3
	vmovapd	%ymm6, %ymm1
	vmovapd	%ymm6, %ymm2
	jmp	.L183
	.cfi_endproc
.LFE4594:
	.size	kernel_dtrsv_n_4_lib4, .-kernel_dtrsv_n_4_lib4
	.section	.text.unlikely
.LCOLDE15:
	.text
.LHOTE15:
	.section	.text.unlikely
.LCOLDB16:
	.text
.LHOTB16:
	.p2align 4,,15
	.globl	kernel_dtrsv_n_4_vs_lib4
	.type	kernel_dtrsv_n_4_vs_lib4, @function
kernel_dtrsv_n_4_vs_lib4:
.LFB4595:
	.cfi_startproc
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$7, %edx
	pushq	-8(%r10)
	pushq	%rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	movq	%rsp, %rbp
	pushq	%r12
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x70,0x6
	.cfi_escape 0x10,0xc,0x2,0x76,0x78
	pushq	%rbx
	.cfi_escape 0x10,0x3,0x2,0x76,0x68
	movq	(%r10), %r11
	jle	.L216
	leal	-8(%rdx), %r12d
	vxorpd	%xmm7, %xmm7, %xmm7
	movq	%r8, %rax
	shrl	$3, %r12d
	movl	%r12d, %ebx
	vmovapd	%ymm7, %ymm6
	vmovapd	%ymm7, %ymm3
	addq	$1, %rbx
	vmovapd	%ymm7, %ymm8
	vmovapd	%ymm7, %ymm5
	movq	%rbx, %r10
	vmovapd	%ymm7, %ymm1
	vmovapd	%ymm7, %ymm4
	salq	$6, %r10
	vmovapd	%ymm7, %ymm2
	addq	%r9, %r10
	.p2align 4,,10
	.p2align 3
.L197:
	vbroadcastf128	(%r9), %ymm0
	addq	$256, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-256(%rax), %ymm9, %ymm2
	vfmadd231pd	-224(%rax), %ymm0, %ymm4
	vbroadcastf128	16(%r9), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-192(%rax), %ymm9, %ymm1
	vfmadd231pd	-160(%rax), %ymm0, %ymm5
	vbroadcastf128	32(%r9), %ymm0
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%rax), %ymm9, %ymm8
	vfmadd231pd	-96(%rax), %ymm0, %ymm3
	vbroadcastf128	48(%r9), %ymm0
	addq	$64, %r9
	vunpcklpd	%ymm0, %ymm0, %ymm9
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rax), %ymm9, %ymm6
	vfmadd231pd	-32(%rax), %ymm0, %ymm7
	cmpq	%r10, %r9
	jne	.L197
	vaddpd	%ymm6, %ymm1, %ymm6
	salq	$8, %rbx
	leal	8(,%r12,8), %eax
	addq	%rbx, %r8
	vaddpd	%ymm8, %ymm2, %ymm1
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm7, %ymm5, %ymm2
.L196:
	leal	-3(%rdx), %r9d
	cmpl	%eax, %r9d
	jle	.L198
	subl	$4, %edx
	subl	%eax, %edx
	movq	%r8, %rax
	shrl	$2, %edx
	addq	$1, %rdx
	movq	%rdx, %r9
	salq	$5, %r9
	addq	%r10, %r9
	.p2align 4,,10
	.p2align 3
.L199:
	vbroadcastf128	(%r10), %ymm0
	subq	$-128, %rax
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-128(%rax), %ymm4, %ymm1
	vfmadd231pd	-96(%rax), %ymm0, %ymm3
	vbroadcastf128	16(%r10), %ymm0
	addq	$32, %r10
	vunpcklpd	%ymm0, %ymm0, %ymm4
	vunpckhpd	%ymm0, %ymm0, %ymm0
	vfmadd231pd	-64(%rax), %ymm4, %ymm6
	vfmadd231pd	-32(%rax), %ymm0, %ymm2
	cmpq	%r9, %r10
	jne	.L199
	salq	$7, %rdx
	addq	%rdx, %r8
.L198:
	vaddpd	%ymm6, %ymm1, %ymm1
	vmovupd	(%r11), %ymm0
	testl	%ecx, %ecx
	vaddpd	%ymm2, %ymm3, %ymm3
	vaddpd	%ymm3, %ymm1, %ymm1
	vsubpd	%ymm1, %ymm0, %ymm1
	vextractf128	$0x1, %ymm1, %xmm5
	je	.L200
	vxorpd	%xmm3, %xmm3, %xmm3
	cmpl	$1, %esi
	vmovsd	(%r8), %xmm2
	vmovsd	8(%r8), %xmm4
	vunpcklpd	%xmm3, %xmm1, %xmm0
	vmovapd	48(%r8), %xmm6
	vshufpd	$1, %xmm3, %xmm1, %xmm1
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovapd	16(%r8), %xmm0
	vfmadd231pd	%xmm2, %xmm4, %xmm1
	vmovlpd	%xmm2, (%r11)
	vunpcklpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	%xmm0, %xmm5, %xmm2
	je	.L240
	vmovsd	40(%r8), %xmm0
	cmpl	$2, %esi
	vmulsd	%xmm1, %xmm0, %xmm1
	vmovlpd	%xmm1, 8(%r11)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm6, %xmm2
	je	.L241
	vunpcklpd	%xmm3, %xmm2, %xmm1
	cmpl	$3, %esi
	vmovsd	80(%r8), %xmm0
	vshufpd	$1, %xmm3, %xmm2, %xmm2
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovsd	88(%r8), %xmm1
	vmovlpd	%xmm0, 16(%r11)
	vfmadd231pd	%xmm0, %xmm1, %xmm2
	je	.L242
	vmovsd	120(%r8), %xmm0
	vmulsd	%xmm2, %xmm0, %xmm2
.L238:
	vmovlpd	%xmm2, 24(%r11)
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
.L200:
	.cfi_restore_state
	vmovsd	(%r8), %xmm0
	cmpl	$1, %esi
	vmovapd	.LC0(%rip), %xmm3
	vxorpd	%xmm4, %xmm4, %xmm4
	vdivsd	%xmm0, %xmm3, %xmm0
	vmovapd	16(%r8), %xmm6
	vmovapd	48(%r8), %xmm7
	vunpcklpd	%xmm4, %xmm1, %xmm2
	vshufpd	$1, %xmm4, %xmm1, %xmm1
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovsd	8(%r8), %xmm2
	vmovlpd	%xmm0, (%r11)
	vfmadd231pd	%xmm0, %xmm2, %xmm1
	vunpcklpd	%xmm0, %xmm0, %xmm0
	vfmadd132pd	%xmm6, %xmm5, %xmm0
	je	.L243
	vmovsd	40(%r8), %xmm2
	cmpl	$2, %esi
	vdivsd	%xmm2, %xmm3, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm1
	vmovlpd	%xmm1, 8(%r11)
	vunpcklpd	%xmm1, %xmm1, %xmm1
	vfmadd231pd	%xmm1, %xmm7, %xmm0
	je	.L244
	vmovsd	80(%r8), %xmm1
	cmpl	$3, %esi
	vunpcklpd	%xmm4, %xmm0, %xmm2
	vdivsd	%xmm1, %xmm3, %xmm1
	vshufpd	$1, %xmm4, %xmm0, %xmm0
	vmulsd	%xmm2, %xmm1, %xmm1
	vmovsd	88(%r8), %xmm2
	vmovlpd	%xmm1, 16(%r11)
	vfmadd231pd	%xmm1, %xmm2, %xmm0
	je	.L245
	vmovsd	120(%r8), %xmm1
	vdivsd	%xmm1, %xmm3, %xmm3
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, 24(%r11)
.L234:
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
.L243:
	.cfi_restore_state
	cmpl	$1, %edi
	je	.L234
	cmpl	$2, %edi
	jle	.L236
	cmpl	$3, %edi
	vmovlpd	%xmm1, 8(%r11)
	je	.L239
.L214:
	vmovaps	%xmm0, 16(%r11)
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
.L240:
	.cfi_restore_state
	cmpl	$1, %edi
	je	.L234
	cmpl	$2, %edi
	jle	.L236
	cmpl	$3, %edi
	vmovlpd	%xmm1, 8(%r11)
	je	.L237
.L208:
	vmovaps	%xmm2, 16(%r11)
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
.L242:
	.cfi_restore_state
	cmpl	$3, %edi
	jne	.L238
	jmp	.L234
.L245:
	cmpl	$3, %edi
	je	.L234
	vmovlpd	%xmm0, 24(%r11)
	jmp	.L234
.L241:
	cmpl	$2, %edi
	je	.L234
	cmpl	$3, %edi
	jne	.L208
.L237:
	vmovlpd	%xmm2, 16(%r11)
	jmp	.L234
.L244:
	cmpl	$2, %edi
	je	.L234
	cmpl	$3, %edi
	jne	.L214
.L239:
	vmovlpd	%xmm0, 16(%r11)
	jmp	.L234
.L216:
	vxorpd	%xmm6, %xmm6, %xmm6
	movq	%r9, %r10
	xorl	%eax, %eax
	vmovapd	%ymm6, %ymm3
	vmovapd	%ymm6, %ymm1
	vmovapd	%ymm6, %ymm2
	jmp	.L196
.L236:
	vmovlpd	%xmm1, 8(%r11)
	jmp	.L234
	.cfi_endproc
.LFE4595:
	.size	kernel_dtrsv_n_4_vs_lib4, .-kernel_dtrsv_n_4_vs_lib4
	.section	.text.unlikely
.LCOLDE16:
	.text
.LHOTE16:
	.section	.text.unlikely
.LCOLDB17:
	.text
.LHOTB17:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_4_lib4
	.type	kernel_dtrsv_t_4_lib4, @function
kernel_dtrsv_t_4_lib4:
.LFB4596:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L265
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm6
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -80(%rbp)
	vmovsd	.LC6(%rip), %xmm4
	vmovsd	.LC7(%rip), %xmm6
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	%xmm4, -64(%rbp)
	vmovsd	%xmm6, -56(%rbp)
	vmovupd	-80(%rbp), %ymm9
	jg	.L266
	vxorpd	%xmm1, %xmm1, %xmm1
	testl	%esi, %esi
	vmovapd	%xmm1, %xmm5
	vmovapd	%xmm1, %xmm6
	vmovapd	%xmm1, %xmm2
	vhaddpd	%xmm1, %xmm1, %xmm1
	je	.L254
.L268:
	vmovsd	24(%r8), %xmm3
	vmovsd	16(%r8), %xmm0
	vsubsd	%xmm1, %xmm3, %xmm1
	vmovsd	120(%rdx), %xmm3
	vmulsd	%xmm3, %xmm1, %xmm3
	vmovlpd	%xmm3, 24(%r8)
	vmovsd	88(%rdx), %xmm1
	vmovapd	%xmm1, %xmm4
	vfnmadd132sd	%xmm3, %xmm0, %xmm4
	vhaddpd	%xmm5, %xmm5, %xmm0
	vsubsd	%xmm0, %xmm4, %xmm1
	vmovsd	80(%rdx), %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	(%r8), %xmm1
	vmovlpd	%xmm0, 16(%r8)
	vunpcklpd	%xmm3, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm3
	vfmadd231pd	16(%rdx), %xmm0, %xmm2
	vmovsd	8(%r8), %xmm0
	vhaddpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	48(%rdx), %xmm6, %xmm3
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm0, %xmm3
	vmovsd	40(%rdx), %xmm0
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm3, %xmm1, %xmm0
	vsubsd	%xmm2, %xmm0, %xmm2
	vmovsd	(%rdx), %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm2, (%r8)
	vzeroupper
.L262:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L265:
	rep ret
	.p2align 4,,10
	.p2align 3
.L266:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	sall	$2, %ecx
	cmpl	$11, %edi
	leaq	32(%r8), %r11
	movslq	%ecx, %rax
	leaq	0(,%rax,8), %rbx
	leaq	(%rdx,%rbx), %r12
	jle	.L258
	leal	-12(%rdi), %r14d
	vxorpd	%xmm4, %xmm4, %xmm4
	salq	$4, %rax
	leaq	(%r12,%rbx), %r9
	movq	%r11, %r10
	shrl	$3, %r14d
	movq	%rax, %rcx
	movl	%r14d, %r15d
	vmovapd	%ymm4, %ymm5
	vmovapd	%ymm4, %ymm3
	movq	%r15, %rax
	vmovapd	%ymm4, %ymm2
	vmovapd	%ymm4, %ymm1
	salq	$6, %rax
	vmovapd	%ymm4, %ymm8
	vmovapd	%ymm4, %ymm7
	vmovapd	%ymm4, %ymm6
	leaq	96(%r8,%rax), %r13
	movq	%r12, %rax
	.p2align 4,,10
	.p2align 3
.L250:
	vmovupd	(%r10), %ymm0
	addq	$64, %r10
	vfmadd231pd	(%rax), %ymm0, %ymm6
	vfmadd231pd	32(%rax), %ymm0, %ymm7
	vfmadd231pd	64(%rax), %ymm0, %ymm8
	vfmadd231pd	96(%rax), %ymm0, %ymm1
	vmovupd	-32(%r10), %ymm0
	addq	%rcx, %rax
	vfmadd231pd	(%r9), %ymm0, %ymm2
	vfmadd231pd	32(%r9), %ymm0, %ymm3
	vfmadd231pd	64(%r9), %ymm0, %ymm5
	vfmadd231pd	96(%r9), %ymm0, %ymm4
	addq	%rcx, %r9
	cmpq	%r10, %r13
	jne	.L250
	leaq	1(%r15), %r9
	movq	%rcx, %rax
	leal	12(,%r14,8), %r13d
	vaddpd	%ymm5, %ymm8, %ymm5
	imulq	%r9, %rax
	salq	$6, %r9
	vaddpd	%ymm3, %ymm7, %ymm3
	addq	%r9, %r11
	vaddpd	%ymm2, %ymm6, %ymm2
	addq	%rax, %r12
	vaddpd	%ymm1, %ymm4, %ymm1
.L249:
	leal	-3(%rdi), %eax
	cmpl	%r13d, %eax
	jle	.L259
	leal	-4(%rdi), %r9d
	movq	%r12, %rax
	subl	%r13d, %r9d
	shrl	$2, %r9d
	movl	%r9d, %r10d
	addq	$1, %r10
	movq	%r10, %rcx
	salq	$5, %rcx
	addq	%r11, %rcx
	.p2align 4,,10
	.p2align 3
.L252:
	vmovupd	(%r11), %ymm0
	addq	$32, %r11
	vfmadd231pd	(%rax), %ymm0, %ymm2
	vfmadd231pd	32(%rax), %ymm0, %ymm3
	vfmadd231pd	64(%rax), %ymm0, %ymm5
	vfmadd231pd	96(%rax), %ymm0, %ymm1
	addq	%rbx, %rax
	cmpq	%rcx, %r11
	jne	.L252
	imulq	%r10, %rbx
	leal	4(%r13,%r9,4), %r13d
	addq	%rbx, %r12
.L251:
	cmpl	%r13d, %edi
	jg	.L267
.L253:
	vextractf128	$0x1, %ymm1, %xmm4
	testl	%esi, %esi
	vextractf128	$0x1, %ymm2, %xmm0
	vextractf128	$0x1, %ymm3, %xmm6
	vaddpd	%xmm1, %xmm4, %xmm1
	vextractf128	$0x1, %ymm5, %xmm7
	vaddpd	%xmm2, %xmm0, %xmm2
	vaddpd	%xmm3, %xmm6, %xmm6
	vaddpd	%xmm5, %xmm7, %xmm5
	vhaddpd	%xmm1, %xmm1, %xmm1
	jne	.L268
.L254:
	vmovsd	120(%rdx), %xmm3
	vmovapd	.LC0(%rip), %xmm7
	vhaddpd	%xmm5, %xmm5, %xmm5
	vdivsd	%xmm3, %xmm7, %xmm3
	vmovsd	24(%r8), %xmm4
	vmovsd	16(%r8), %xmm0
	vsubsd	%xmm1, %xmm4, %xmm1
	vmulsd	%xmm3, %xmm1, %xmm3
	vmovlpd	%xmm3, 24(%r8)
	vmovsd	88(%rdx), %xmm1
	vmovapd	%xmm1, %xmm4
	vfnmadd132sd	%xmm3, %xmm0, %xmm4
	vmovsd	80(%rdx), %xmm0
	vsubsd	%xmm5, %xmm4, %xmm1
	vdivsd	%xmm0, %xmm7, %xmm0
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovlpd	%xmm0, 16(%r8)
	vunpcklpd	%xmm3, %xmm0, %xmm0
	vmovsd	40(%rdx), %xmm1
	vmovapd	%xmm0, %xmm3
	vfmadd231pd	16(%rdx), %xmm0, %xmm2
	vmovsd	8(%r8), %xmm0
	vdivsd	%xmm1, %xmm7, %xmm1
	vhaddpd	%xmm2, %xmm2, %xmm2
	vfmadd132pd	48(%rdx), %xmm6, %xmm3
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm0, %xmm3
	vmulsd	%xmm1, %xmm3, %xmm3
	vmovsd	(%r8), %xmm1
	vmovlpd	%xmm3, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm3, %xmm1, %xmm0
	vmovsd	(%rdx), %xmm1
	vsubsd	%xmm2, %xmm0, %xmm2
	vdivsd	%xmm1, %xmm7, %xmm7
	vmulsd	%xmm7, %xmm2, %xmm2
	vmovlpd	%xmm2, (%r8)
	vzeroupper
	jmp	.L262
	.p2align 4,,10
	.p2align 3
.L267:
	vxorpd	%xmm0, %xmm0, %xmm0
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm4
	vcvtsi2sd	%edi, %xmm0, %xmm0
	vsubsd	%xmm0, %xmm4, %xmm0
	vmovupd	(%rcx), %ymm4
	vbroadcastsd	%xmm0, %ymm0
	vsubpd	%ymm0, %ymm9, %ymm9
	vxorpd	%xmm0, %xmm0, %xmm0
	vblendvpd	%ymm9, %ymm0, %ymm4, %ymm9
	vfmadd231pd	(%r12), %ymm9, %ymm2
	vfmadd231pd	32(%r12), %ymm9, %ymm3
	vfmadd231pd	64(%r12), %ymm9, %ymm5
	vfmadd231pd	96(%r12), %ymm9, %ymm1
	jmp	.L253
.L259:
	movq	%r11, %rcx
	jmp	.L251
.L258:
	vxorpd	%xmm5, %xmm5, %xmm5
	movl	$4, %r13d
	vmovapd	%ymm5, %ymm3
	vmovapd	%ymm5, %ymm2
	vmovapd	%ymm5, %ymm1
	jmp	.L249
	.cfi_endproc
.LFE4596:
	.size	kernel_dtrsv_t_4_lib4, .-kernel_dtrsv_t_4_lib4
	.section	.text.unlikely
.LCOLDE17:
	.text
.LHOTE17:
	.section	.text.unlikely
.LCOLDB18:
	.text
.LHOTB18:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_3_lib4
	.type	kernel_dtrsv_t_3_lib4, @function
kernel_dtrsv_t_3_lib4:
.LFB4597:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L289
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm7
	vmovsd	.LC6(%rip), %xmm6
	vmovsd	%xmm7, -80(%rbp)
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -64(%rbp)
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	.LC7(%rip), %xmm7
	vmovsd	%xmm7, -56(%rbp)
	jg	.L290
	jne	.L283
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovsd	24(%r8), %xmm4
	vmovsd	24(%rdx), %xmm3
	vmovsd	56(%rdx), %xmm2
	vmovsd	88(%rdx), %xmm0
	vfmadd132sd	%xmm4, %xmm1, %xmm3
	vfmadd132sd	%xmm4, %xmm1, %xmm2
	vfmadd132sd	%xmm4, %xmm1, %xmm0
.L277:
	testl	%esi, %esi
	vhaddpd	%xmm0, %xmm0, %xmm0
	jne	.L291
	vmovsd	80(%rdx), %xmm5
	vmovapd	.LC0(%rip), %xmm4
	vdivsd	%xmm5, %xmm4, %xmm5
	vmovsd	16(%r8), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmulsd	%xmm5, %xmm0, %xmm0
	vmovlpd	%xmm0, 16(%r8)
	vmovsd	16(%rdx), %xmm1
	vmulsd	%xmm0, %xmm1, %xmm5
	vmovsd	48(%rdx), %xmm1
	vmulsd	%xmm0, %xmm1, %xmm0
	vmovsd	40(%rdx), %xmm1
	vaddsd	%xmm5, %xmm3, %xmm3
	vdivsd	%xmm1, %xmm4, %xmm1
	vaddsd	%xmm0, %xmm2, %xmm2
	vhaddpd	%xmm3, %xmm3, %xmm3
	vhaddpd	%xmm2, %xmm2, %xmm2
	vmovsd	8(%r8), %xmm0
	vsubsd	%xmm2, %xmm0, %xmm2
	vmulsd	%xmm1, %xmm2, %xmm2
	vmovsd	(%r8), %xmm1
	vmovlpd	%xmm2, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm2, %xmm1, %xmm0
	vmovsd	(%rdx), %xmm1
	vsubsd	%xmm3, %xmm0, %xmm3
	vdivsd	%xmm1, %xmm4, %xmm1
	vmulsd	%xmm1, %xmm3, %xmm3
	vmovlpd	%xmm3, (%r8)
.L286:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%r15
	.cfi_restore 15
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L289:
	rep ret
	.p2align 4,,10
	.p2align 3
.L291:
	.cfi_escape 0xf,0x3,0x76,0x58,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x50
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x60
	.cfi_escape 0x10,0xd,0x2,0x76,0x68
	.cfi_escape 0x10,0xe,0x2,0x76,0x70
	.cfi_escape 0x10,0xf,0x2,0x76,0x78
	vmovsd	16(%r8), %xmm4
	vsubsd	%xmm0, %xmm4, %xmm0
	vmovsd	80(%rdx), %xmm4
	vmulsd	%xmm4, %xmm0, %xmm0
	vmovlpd	%xmm0, 16(%r8)
	vmovsd	48(%rdx), %xmm4
	vmovsd	16(%rdx), %xmm1
	vmulsd	%xmm0, %xmm4, %xmm4
	vmulsd	%xmm0, %xmm1, %xmm1
	vmovsd	8(%r8), %xmm0
	vaddsd	%xmm4, %xmm2, %xmm2
	vaddsd	%xmm1, %xmm3, %xmm3
	vmovsd	(%r8), %xmm1
	vhaddpd	%xmm2, %xmm2, %xmm2
	vhaddpd	%xmm3, %xmm3, %xmm3
	vsubsd	%xmm2, %xmm0, %xmm2
	vmovsd	40(%rdx), %xmm0
	vmulsd	%xmm0, %xmm2, %xmm2
	vmovlpd	%xmm2, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm2, %xmm1, %xmm0
	vsubsd	%xmm3, %xmm0, %xmm3
	vmovsd	(%rdx), %xmm0
	vmulsd	%xmm0, %xmm3, %xmm3
	vmovlpd	%xmm3, (%r8)
	jmp	.L286
	.p2align 4,,10
	.p2align 3
.L290:
	vxorpd	%xmm1, %xmm1, %xmm1
	vmovupd	(%r8), %ymm0
	leal	0(,%rcx,4), %eax
	cmpl	$11, %edi
	vmovupd	-80(%rbp), %ymm7
	leaq	32(%r8), %r10
	vblendpd	$7, %ymm1, %ymm0, %ymm0
	cltq
	leaq	0(,%rax,8), %rbx
	vmovapd	%ymm0, %ymm6
	vmovapd	%ymm0, %ymm5
	vfmadd132pd	64(%rdx), %ymm1, %ymm0
	leaq	(%rdx,%rbx), %r12
	vfmadd132pd	(%rdx), %ymm1, %ymm6
	vfmadd132pd	32(%rdx), %ymm1, %ymm5
	jle	.L281
	leal	-12(%rdi), %r14d
	salq	$4, %rax
	leaq	(%r12,%rbx), %r9
	movq	%rax, %r11
	vmovapd	%ymm1, %ymm4
	vmovapd	%ymm1, %ymm2
	shrl	$3, %r14d
	vmovapd	%ymm1, %ymm3
	movq	%r10, %rcx
	movl	%r14d, %r15d
	movq	%r15, %rax
	salq	$6, %rax
	leaq	96(%r8,%rax), %r13
	movq	%r12, %rax
	.p2align 4,,10
	.p2align 3
.L273:
	vmovupd	(%rcx), %ymm1
	addq	$64, %rcx
	vfmadd231pd	(%rax), %ymm1, %ymm6
	vfmadd231pd	32(%rax), %ymm1, %ymm5
	vfmadd231pd	64(%rax), %ymm1, %ymm0
	vmovupd	-32(%rcx), %ymm1
	addq	%r11, %rax
	vfmadd231pd	(%r9), %ymm1, %ymm3
	vfmadd231pd	32(%r9), %ymm1, %ymm2
	vfmadd231pd	64(%r9), %ymm1, %ymm4
	addq	%r11, %r9
	cmpq	%rcx, %r13
	jne	.L273
	leaq	1(%r15), %rcx
	movq	%r11, %rax
	leal	12(,%r14,8), %r13d
	imulq	%rcx, %rax
	salq	$6, %rcx
	addq	%rcx, %r10
	addq	%rax, %r12
.L272:
	leal	-3(%rdi), %eax
	vaddpd	%ymm3, %ymm6, %ymm3
	cmpl	%r13d, %eax
	vaddpd	%ymm2, %ymm5, %ymm2
	vaddpd	%ymm4, %ymm0, %ymm0
	jle	.L282
	leal	-4(%rdi), %r9d
	movq	%r12, %rax
	subl	%r13d, %r9d
	shrl	$2, %r9d
	movl	%r9d, %r11d
	addq	$1, %r11
	movq	%r11, %rcx
	salq	$5, %rcx
	addq	%r10, %rcx
	.p2align 4,,10
	.p2align 3
.L275:
	vmovupd	(%r10), %ymm1
	addq	$32, %r10
	vfmadd231pd	(%rax), %ymm1, %ymm3
	vfmadd231pd	32(%rax), %ymm1, %ymm2
	vfmadd231pd	64(%rax), %ymm1, %ymm0
	addq	%rbx, %rax
	cmpq	%rcx, %r10
	jne	.L275
	imulq	%r11, %rbx
	leal	4(%r13,%r9,4), %r13d
	addq	%rbx, %r12
.L274:
	cmpl	%r13d, %edi
	jg	.L292
.L276:
	vextractf128	$0x1, %ymm3, %xmm5
	vextractf128	$0x1, %ymm2, %xmm4
	vextractf128	$0x1, %ymm0, %xmm1
	vaddpd	%xmm3, %xmm5, %xmm3
	vaddpd	%xmm2, %xmm4, %xmm2
	vaddpd	%xmm0, %xmm1, %xmm0
	vzeroupper
	jmp	.L277
	.p2align 4,,10
	.p2align 3
.L292:
	vxorpd	%xmm1, %xmm1, %xmm1
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm4
	vcvtsi2sd	%edi, %xmm1, %xmm1
	vsubsd	%xmm1, %xmm4, %xmm1
	vmovupd	(%rcx), %ymm4
	vbroadcastsd	%xmm1, %ymm1
	vsubpd	%ymm1, %ymm7, %ymm7
	vxorpd	%xmm1, %xmm1, %xmm1
	vblendvpd	%ymm7, %ymm1, %ymm4, %ymm7
	vfmadd231pd	(%r12), %ymm7, %ymm3
	vfmadd231pd	32(%r12), %ymm7, %ymm2
	vfmadd231pd	64(%r12), %ymm7, %ymm0
	jmp	.L276
	.p2align 4,,10
	.p2align 3
.L283:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm2
	vmovapd	%xmm0, %xmm3
	jmp	.L277
.L281:
	vxorpd	%xmm4, %xmm4, %xmm4
	movl	$4, %r13d
	vmovapd	%ymm4, %ymm2
	vmovapd	%ymm4, %ymm3
	jmp	.L272
.L282:
	movq	%r10, %rcx
	jmp	.L274
	.cfi_endproc
.LFE4597:
	.size	kernel_dtrsv_t_3_lib4, .-kernel_dtrsv_t_3_lib4
	.section	.text.unlikely
.LCOLDE18:
	.text
.LHOTE18:
	.section	.text.unlikely
.LCOLDB19:
	.text
.LHOTB19:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_2_lib4
	.type	kernel_dtrsv_t_2_lib4, @function
kernel_dtrsv_t_2_lib4:
.LFB4598:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L314
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm6
	vmovsd	.LC5(%rip), %xmm7
	vmovsd	%xmm6, -80(%rbp)
	vmovsd	.LC6(%rip), %xmm5
	vmovsd	.LC7(%rip), %xmm6
	vmovsd	%xmm7, -72(%rbp)
	vmovsd	%xmm5, -64(%rbp)
	vmovsd	%xmm6, -56(%rbp)
	jg	.L315
	cmpl	$2, %edi
	jle	.L308
	vxorpd	%xmm2, %xmm2, %xmm2
	cmpl	$4, %edi
	vmovsd	16(%r8), %xmm3
	vmovsd	16(%rdx), %xmm1
	vmovsd	48(%rdx), %xmm0
	vfmadd132sd	%xmm3, %xmm2, %xmm1
	vfmadd132sd	%xmm3, %xmm2, %xmm0
	jne	.L301
	vmovsd	24(%r8), %xmm3
	vmovsd	24(%rdx), %xmm2
	vfmadd132sd	%xmm3, %xmm1, %xmm2
	vmovapd	%xmm2, %xmm1
	vmovsd	56(%rdx), %xmm2
	vfmadd132sd	%xmm3, %xmm0, %xmm2
	vmovapd	%xmm2, %xmm0
.L301:
	testl	%esi, %esi
	vhaddpd	%xmm0, %xmm0, %xmm0
	je	.L303
	vmovsd	8(%r8), %xmm2
	vhaddpd	%xmm1, %xmm1, %xmm1
	vmovsd	(%r8), %xmm3
	vsubsd	%xmm0, %xmm2, %xmm0
	vmovsd	40(%rdx), %xmm2
	vmulsd	%xmm2, %xmm0, %xmm2
	vmovlpd	%xmm2, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm2, %xmm3, %xmm0
	vsubsd	%xmm1, %xmm0, %xmm1
	vmovsd	(%rdx), %xmm0
	vmulsd	%xmm0, %xmm1, %xmm1
	vmovlpd	%xmm1, (%r8)
.L311:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L314:
	rep ret
	.p2align 4,,10
	.p2align 3
.L303:
	.cfi_escape 0xf,0x3,0x76,0x60,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x58
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x68
	.cfi_escape 0x10,0xd,0x2,0x76,0x70
	.cfi_escape 0x10,0xe,0x2,0x76,0x78
	vmovsd	40(%rdx), %xmm2
	vmovapd	.LC0(%rip), %xmm3
	vhaddpd	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm2, %xmm3, %xmm4
	vmovsd	8(%r8), %xmm2
	vsubsd	%xmm0, %xmm2, %xmm0
	vmulsd	%xmm4, %xmm0, %xmm2
	vmovsd	(%r8), %xmm4
	vmovlpd	%xmm2, 8(%r8)
	vmovsd	8(%rdx), %xmm0
	vfnmadd132sd	%xmm2, %xmm4, %xmm0
	vmovsd	(%rdx), %xmm2
	vsubsd	%xmm1, %xmm0, %xmm1
	vdivsd	%xmm2, %xmm3, %xmm3
	vmulsd	%xmm3, %xmm1, %xmm1
	vmovlpd	%xmm1, (%r8)
	jmp	.L311
	.p2align 4,,10
	.p2align 3
.L315:
	vxorpd	%xmm2, %xmm2, %xmm2
	vmovupd	(%r8), %ymm0
	leal	0(,%rcx,4), %r11d
	cmpl	$11, %edi
	vmovupd	-80(%rbp), %ymm5
	leaq	32(%r8), %r9
	vblendpd	$3, %ymm2, %ymm0, %ymm0
	movslq	%r11d, %r11
	leaq	0(,%r11,8), %r10
	vmovapd	%ymm0, %ymm1
	vfmadd132pd	32(%rdx), %ymm2, %ymm0
	leaq	(%rdx,%r10), %rbx
	vfmadd132pd	(%rdx), %ymm2, %ymm1
	jle	.L306
	leal	-12(%rdi), %r13d
	vmovapd	%ymm2, %ymm3
	vmovapd	%ymm2, %ymm4
	salq	$4, %r11
	movq	%r9, %rcx
	shrl	$3, %r13d
	movl	%r13d, %r14d
	movq	%r14, %rax
	salq	$6, %rax
	leaq	96(%r8,%rax), %r12
	movq	%rbx, %rax
	.p2align 4,,10
	.p2align 3
.L297:
	vmovupd	(%rcx), %ymm2
	addq	$64, %rcx
	vfmadd231pd	(%rax), %ymm2, %ymm1
	vfmadd231pd	32(%rax), %ymm2, %ymm0
	vmovupd	-32(%rcx), %ymm2
	vfmadd231pd	(%rax,%r10), %ymm2, %ymm4
	vfmadd231pd	32(%rax,%r10), %ymm2, %ymm3
	addq	%r11, %rax
	cmpq	%rcx, %r12
	jne	.L297
	leaq	1(%r14), %rax
	leaq	(%r10,%r10), %rcx
	leal	12(,%r13,8), %r13d
	imulq	%rax, %rcx
	salq	$6, %rax
	addq	%rax, %r9
	addq	%rcx, %rbx
.L296:
	leal	-3(%rdi), %eax
	vaddpd	%ymm4, %ymm1, %ymm1
	cmpl	%r13d, %eax
	vaddpd	%ymm3, %ymm0, %ymm0
	jle	.L307
	leal	-4(%rdi), %r11d
	movq	%rbx, %rax
	subl	%r13d, %r11d
	shrl	$2, %r11d
	movl	%r11d, %r12d
	addq	$1, %r12
	movq	%r12, %rcx
	salq	$5, %rcx
	addq	%r9, %rcx
	.p2align 4,,10
	.p2align 3
.L299:
	vmovupd	(%r9), %ymm2
	addq	$32, %r9
	vfmadd231pd	(%rax), %ymm2, %ymm1
	vfmadd231pd	32(%rax), %ymm2, %ymm0
	addq	%r10, %rax
	cmpq	%rcx, %r9
	jne	.L299
	imulq	%r12, %r10
	leal	4(%r13,%r11,4), %r13d
	addq	%r10, %rbx
.L298:
	cmpl	%r13d, %edi
	jg	.L316
.L300:
	vextractf128	$0x1, %ymm1, %xmm3
	vextractf128	$0x1, %ymm0, %xmm2
	vaddpd	%xmm1, %xmm3, %xmm1
	vaddpd	%xmm0, %xmm2, %xmm0
	vzeroupper
	jmp	.L301
	.p2align 4,,10
	.p2align 3
.L316:
	vxorpd	%xmm3, %xmm3, %xmm3
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm2
	vcvtsi2sd	%edi, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm2, %xmm2
	vmovupd	(%rcx), %ymm3
	vbroadcastsd	%xmm2, %ymm2
	vsubpd	%ymm2, %ymm5, %ymm5
	vxorpd	%xmm2, %xmm2, %xmm2
	vblendvpd	%ymm5, %ymm2, %ymm3, %ymm5
	vfmadd231pd	(%rbx), %ymm5, %ymm1
	vfmadd231pd	32(%rbx), %ymm5, %ymm0
	jmp	.L300
	.p2align 4,,10
	.p2align 3
.L308:
	vxorpd	%xmm0, %xmm0, %xmm0
	vmovapd	%xmm0, %xmm1
	jmp	.L301
.L306:
	vxorpd	%xmm3, %xmm3, %xmm3
	movl	$4, %r13d
	vmovapd	%ymm3, %ymm4
	jmp	.L296
.L307:
	movq	%r9, %rcx
	jmp	.L298
	.cfi_endproc
.LFE4598:
	.size	kernel_dtrsv_t_2_lib4, .-kernel_dtrsv_t_2_lib4
	.section	.text.unlikely
.LCOLDE19:
	.text
.LHOTE19:
	.section	.text.unlikely
.LCOLDB20:
	.text
.LHOTB20:
	.p2align 4,,15
	.globl	kernel_dtrsv_t_1_lib4
	.type	kernel_dtrsv_t_1_lib4, @function
kernel_dtrsv_t_1_lib4:
.LFB4599:
	.cfi_startproc
	testl	%edi, %edi
	jle	.L339
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	cmpl	$4, %edi
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
	vmovsd	.LC4(%rip), %xmm7
	vmovsd	.LC5(%rip), %xmm3
	vmovsd	%xmm7, -80(%rbp)
	vmovsd	%xmm3, -72(%rbp)
	vmovsd	.LC6(%rip), %xmm7
	vmovsd	.LC7(%rip), %xmm3
	vmovsd	%xmm7, -64(%rbp)
	vmovsd	%xmm3, -56(%rbp)
	jg	.L340
	cmpl	$1, %edi
	je	.L332
	leal	-2(%rdi), %ecx
	vxorpd	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	addq	$1, %rcx
	.p2align 4,,10
	.p2align 3
.L326:
	vmovsd	8(%rdx,%rax,8), %xmm1
	vmovsd	8(%r8,%rax,8), %xmm2
	addq	$1, %rax
	cmpq	%rcx, %rax
	vfmadd132sd	%xmm2, %xmm0, %xmm1
	vmovapd	%xmm1, %xmm0
	jne	.L326
.L325:
	testl	%esi, %esi
	vhaddpd	%xmm0, %xmm0, %xmm0
	je	.L327
	vmovsd	(%r8), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmovsd	(%rdx), %xmm1
	vmulsd	%xmm1, %xmm0, %xmm0
	vmovlpd	%xmm0, (%r8)
.L336:
	popq	%rbx
	.cfi_restore 3
	popq	%r10
	.cfi_restore 10
	.cfi_def_cfa 10, 0
	popq	%r12
	.cfi_restore 12
	popq	%r13
	.cfi_restore 13
	popq	%r14
	.cfi_restore 14
	popq	%rbp
	.cfi_restore 6
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
.L339:
	rep ret
	.p2align 4,,10
	.p2align 3
.L327:
	.cfi_escape 0xf,0x3,0x76,0x60,0x6
	.cfi_escape 0x10,0x3,0x2,0x76,0x58
	.cfi_escape 0x10,0x6,0x2,0x76,0
	.cfi_escape 0x10,0xc,0x2,0x76,0x68
	.cfi_escape 0x10,0xd,0x2,0x76,0x70
	.cfi_escape 0x10,0xe,0x2,0x76,0x78
	vmovsd	(%rdx), %xmm2
	vmovapd	.LC0(%rip), %xmm1
	vdivsd	%xmm2, %xmm1, %xmm2
	vmovsd	(%r8), %xmm1
	vsubsd	%xmm0, %xmm1, %xmm0
	vmulsd	%xmm2, %xmm0, %xmm0
	vmovlpd	%xmm0, (%r8)
	jmp	.L336
	.p2align 4,,10
	.p2align 3
.L340:
	vxorpd	%xmm1, %xmm1, %xmm1
	leal	0(,%rcx,4), %r11d
	vmovupd	(%r8), %ymm0
	cmpl	$11, %edi
	leaq	32(%r8), %rcx
	movslq	%r11d, %r11
	vmovupd	-80(%rbp), %ymm2
	vblendpd	$1, %ymm1, %ymm0, %ymm0
	leaq	0(,%r11,8), %r10
	vfmadd132pd	(%rdx), %ymm1, %ymm0
	leaq	(%rdx,%r10), %rbx
	jle	.L330
	leal	-12(%rdi), %r13d
	salq	$4, %r11
	movq	%rbx, %r9
	shrl	$3, %r13d
	movl	%r13d, %r14d
	movq	%r14, %rax
	salq	$6, %rax
	leaq	96(%r8,%rax), %r12
	movq	%rcx, %rax
	.p2align 4,,10
	.p2align 3
.L321:
	vmovapd	(%r9), %ymm4
	addq	$64, %rax
	vmovapd	(%r9,%r10), %ymm5
	addq	%r11, %r9
	vfmadd231pd	-64(%rax), %ymm4, %ymm0
	vfmadd231pd	-32(%rax), %ymm5, %ymm1
	cmpq	%rax, %r12
	jne	.L321
	leaq	1(%r14), %rax
	leaq	(%r10,%r10), %r9
	leal	12(,%r13,8), %r13d
	imulq	%rax, %r9
	salq	$6, %rax
	addq	%rax, %rcx
	addq	%r9, %rbx
.L320:
	leal	-3(%rdi), %eax
	vaddpd	%ymm1, %ymm0, %ymm0
	cmpl	%r13d, %eax
	jle	.L331
	leal	-4(%rdi), %r11d
	movq	%rbx, %rax
	subl	%r13d, %r11d
	shrl	$2, %r11d
	movl	%r11d, %r12d
	addq	$1, %r12
	movq	%r12, %r9
	salq	$5, %r9
	addq	%rcx, %r9
	.p2align 4,,10
	.p2align 3
.L323:
	vmovapd	(%rax), %ymm6
	addq	$32, %rcx
	addq	%r10, %rax
	vfmadd231pd	-32(%rcx), %ymm6, %ymm0
	cmpq	%r9, %rcx
	jne	.L323
	imulq	%r12, %r10
	leal	4(%r13,%r11,4), %r13d
	addq	%r10, %rbx
.L322:
	cmpl	%r13d, %edi
	jg	.L341
.L324:
	vextractf128	$0x1, %ymm0, %xmm1
	vaddpd	%xmm0, %xmm1, %xmm0
	vzeroupper
	jmp	.L325
	.p2align 4,,10
	.p2align 3
.L341:
	vxorpd	%xmm3, %xmm3, %xmm3
	subl	%r13d, %edi
	vmovsd	.LC8(%rip), %xmm1
	vcvtsi2sd	%edi, %xmm3, %xmm3
	vsubsd	%xmm3, %xmm1, %xmm1
	vmovupd	(%r9), %ymm3
	vbroadcastsd	%xmm1, %ymm1
	vsubpd	%ymm1, %ymm2, %ymm2
	vxorpd	%xmm1, %xmm1, %xmm1
	vblendvpd	%ymm2, %ymm1, %ymm3, %ymm2
	vfmadd231pd	(%rbx), %ymm2, %ymm0
	jmp	.L324
.L332:
	vxorpd	%xmm0, %xmm0, %xmm0
	jmp	.L325
.L330:
	vxorpd	%xmm1, %xmm1, %xmm1
	movl	$4, %r13d
	jmp	.L320
.L331:
	movq	%rcx, %r9
	jmp	.L322
	.cfi_endproc
.LFE4599:
	.size	kernel_dtrsv_t_1_lib4, .-kernel_dtrsv_t_1_lib4
	.section	.text.unlikely
.LCOLDE20:
	.text
.LHOTE20:
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC0:
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC4:
	.long	0
	.long	1074528256
	.align 8
.LC5:
	.long	0
	.long	1074003968
	.align 8
.LC6:
	.long	0
	.long	1073217536
	.align 8
.LC7:
	.long	0
	.long	1071644672
	.align 8
.LC8:
	.long	0
	.long	1074790400
	.ident	"GCC: (GNU) 5.2.0"
	.section	.note.GNU-stack,"",@progbits
