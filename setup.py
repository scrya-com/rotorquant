"""
Build script for TurboQuant with optional CUDA kernels.

Usage:
    pip install -e .                    # PyTorch-only (no CUDA kernels)
    pip install -e . --config-settings="--build-option=--cuda"  # With CUDA kernels
    python setup.py build_ext --inplace # Build CUDA kernels in-place
"""

from setuptools import setup, find_packages
import os
import sys

ext_modules = []
cmdclass = {}

# Check if CUDA build is requested
build_cuda = '--cuda' in sys.argv or os.environ.get('TURBOQUANT_BUILD_CUDA', '0') == '1'
if '--cuda' in sys.argv:
    sys.argv.remove('--cuda')

if build_cuda or 'build_ext' in sys.argv:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # If we detect Blackwell (12.x), force "+ PTX" for compatibility
            if major >= 12:
                arch_list = f"{major}.{minor}+PTX"
            else:
                arch_list = f"{major}.{minor}"
            print(f"Auto-detected GPU architecture: {arch_list}")
            # Force-set the architecture environment variable so PyTorch sees it
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

        def nvcc_flags():
            nvcc_threads = os.getenv("NVCC_THREADS", "8")
            return [
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                f"--threads={nvcc_threads}",
            ]

        csrc_dir = os.path.join(os.path.dirname(__file__), 'turboquant', 'csrc')

        ext_modules = [
            CUDAExtension(
                name='turboquant.cuda_qjl_score',
                sources=[os.path.join(csrc_dir, 'qjl_score_kernel.cu')],
                extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
            ),
            CUDAExtension(
                name='turboquant.cuda_qjl_quant',
                sources=[os.path.join(csrc_dir, 'qjl_quant_kernel.cu')],
                extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
            ),
            CUDAExtension(
                name='turboquant.cuda_qjl_gqa_score',
                sources=[os.path.join(csrc_dir, 'qjl_gqa_score_kernel.cu')],
                extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()}
            ),
            CUDAExtension(
                name='turboquant.quantization',
                sources=[os.path.join(csrc_dir, 'quantization.cu')],
                extra_compile_args={
                    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
                    "nvcc": nvcc_flags() + [
                        "-DENABLE_BF16",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    ]
                }
            ),
        ]
        cmdclass = {'build_ext': BuildExtension}
        print("CUDA extensions will be built.")
    except ImportError:
        print("WARNING: torch not found, CUDA extensions will not be built.")

setup(
    name='turboquant',
    version='0.2.0',
    description='TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'scipy>=1.10.0',
    ],
    extras_require={
        'validate': ['transformers>=4.40.0', 'accelerate>=0.25.0', 'bitsandbytes>=0.43.0'],
    },
)
