"""
配置 HuggingFace 镜像站，解决连接问题
"""
import os

# 设置 HuggingFace 镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或者使用 ModelScope 镜像
# os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'

print("✅ HuggingFace 镜像已配置")
print(f"   使用镜像: {os.environ.get('HF_ENDPOINT')}")
