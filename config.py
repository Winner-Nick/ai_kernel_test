"""
配置文件 - API Keys 和模型配置
参考 agent_learn/main.py 的配置方式
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

# 配置自定义 OpenAI 兼容客户端（从环境变量读取）
OPENAI_CLIENT = OpenAI(
    base_url=os.getenv('API_BASE_URL', 'http://localhost:8000/v1/'),
    api_key=os.getenv('API_KEY', 'your-api-key-here')
)

# 要测试的模型列表
MODELS_TO_TEST = [
    {
        'name': 'DeepSeek Chat',
        'model_id': 'deepseek-chat',
        'client': OPENAI_CLIENT,
    },
    {
        'name': 'GPT-4',
        'model_id': 'gpt-4',
        'client': OPENAI_CLIENT,
    },
    {
        'name': 'GPT-4 Turbo',
        'model_id': 'gpt-4-turbo-preview',
        'client': OPENAI_CLIENT,
    },
    {
        'name': 'Claude Sonnet 4',
        'model_id': 'claude-sonnet-4-20250514',
        'client': OPENAI_CLIENT,
    },
]

# 测试配置
TEST_CONFIG = {
    'level': 1,
    'problem_id': 1,
    'dataset_src': 'huggingface',
    'dataset_name': 'ScalingIntelligence/KernelBench',
    'backend': 'cuda',  # 可选: cuda, triton
    'temperature': 0,  # 0 = 最确定的输出
    'max_tokens': 4000,
}

# 输出目录
OUTPUT_DIR = './results'
