"""
手动下载 KernelBench 数据集

如果无法连接 HuggingFace，可以：
1. 使用此脚本通过镜像下载
2. 或在能连接的机器上下载后转移
"""
import os
from datasets import load_dataset

# 配置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("🔄 正在下载 KernelBench 数据集...")
print(f"   使用镜像: {os.environ['HF_ENDPOINT']}")

try:
    # 下载数据集
    dataset = load_dataset('ScalingIntelligence/KernelBench')

    # 显示数据集信息
    print("\n✅ 数据集下载成功!")
    print(f"   包含的 split: {list(dataset.keys())}")

    for split_name in dataset.keys():
        print(f"   {split_name}: {len(dataset[split_name])} 个问题")

    # 获取缓存路径
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
    print(f"\n💾 数据集已缓存到: {cache_dir}")
    print("   下次运行会直接使用本地缓存")

except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n可选方案:")
    print("1. 检查网络连接")
    print("2. 尝试使用代理")
    print("3. 在其他机器下载后复制缓存文件夹")
