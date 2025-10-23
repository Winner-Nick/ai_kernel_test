"""
配置文件 - 使用自定义 API 中转，但使用 KernelBench 的评估方法
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# API 配置（从环境变量读取）
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000/v1/')
API_KEY = os.getenv('API_KEY', 'your-api-key-here')

# 要测试的模型列表
MODELS_TO_TEST = [
    {
        'name': 'DeepSeek Chat',
        'model_id': 'deepseek-chat',
        'server_type': 'openai',  # 使用 OpenAI 兼容格式
    },
    {
        'name': 'GPT-4',
        'model_id': 'gpt-4',
        'server_type': 'openai',
    },
    {
        'name': 'GPT-4 Turbo',
        'model_id': 'gpt-4-turbo-preview',
        'server_type': 'openai',
    },
    {
        'name': 'Claude Sonnet 4',
        'model_id': 'claude-sonnet-4-20250514',
        'server_type': 'openai',
    },
]

# 测试配置
TEST_CONFIG = {
    'level': 1,
    'problem_id': 1,
    'dataset_name': 'ScalingIntelligence/KernelBench',
    'backend': 'cuda',
    'temperature': 0.0,
    'max_tokens': 4096,
    'num_correct_trials': 5,
    'num_perf_trials': 100,
    'gpu_arch': None,
}

# 输出目录
OUTPUT_DIR = './kernelbench_wrapper/results'
