"""
KernelBench Wrapper - 使用 KernelBench 官方评估方法，仅替换 API 调用
"""
from .config import MODELS_TO_TEST, TEST_CONFIG, OUTPUT_DIR
from .custom_inference import create_custom_inference_function
from .test_single_model import test_single_model, main

__all__ = [
    'MODELS_TO_TEST',
    'TEST_CONFIG',
    'OUTPUT_DIR',
    'create_custom_inference_function',
    'test_single_model',
    'main',
]
