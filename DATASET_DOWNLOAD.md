# KernelBench 数据集下载指南

## 📍 数据集存储位置

数据集下载后会缓存在：

**Windows**:
```
C:\Users\你的用户名\.cache\huggingface\datasets\ScalingIntelligence___kernel_bench
```

**Linux/Mac**:
```
~/.cache/huggingface/datasets/ScalingIntelligence___kernel_bench
```

## 🌐 方法 1: 使用镜像站（推荐）

### 自动配置（已集成）

代码已配置使用 HF-Mirror 镜像站，无需额外操作。

### 手动设置环境变量

如果需要更换镜像或临时使用：

**Windows (PowerShell)**:
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python run_kernelbench_wrapper.py
```

**Linux/Mac**:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
python run_kernelbench_wrapper.py
```

### 可用镜像站

1. **HF-Mirror** (推荐):
   ```
   https://hf-mirror.com
   ```

2. **ModelScope**:
   ```
   https://www.modelscope.cn
   ```

## 📥 方法 2: 预先下载数据集

运行下载脚本：

```bash
python download_dataset.py
```

这会：
- 使用镜像站下载数据集
- 缓存到本地
- 下次运行直接使用缓存

## 🔄 方法 3: 从其他机器转移

如果某台机器能访问 HuggingFace：

### 1. 在能连接的机器上下载

```bash
python download_dataset.py
```

### 2. 找到缓存目录

**Windows**:
```
C:\Users\你的用户名\.cache\huggingface\datasets
```

**Linux/Mac**:
```
~/.cache/huggingface/datasets
```

### 3. 复制整个 datasets 文件夹

将整个文件夹复制到目标机器的相同位置。

### 4. 验证

```bash
python -c "from datasets import load_dataset; ds = load_dataset('ScalingIntelligence/KernelBench'); print('✅ 数据集加载成功')"
```

## 🌟 方法 4: 使用代理

如果有代理服务器：

**Windows (PowerShell)**:
```powershell
$env:HTTP_PROXY="http://proxy-server:port"
$env:HTTPS_PROXY="http://proxy-server:port"
python run_kernelbench_wrapper.py
```

**Linux/Mac**:
```bash
export HTTP_PROXY="http://proxy-server:port"
export HTTPS_PROXY="http://proxy-server:port"
python run_kernelbench_wrapper.py
```

## ✅ 验证数据集

检查数据集是否正确加载：

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset('ScalingIntelligence/KernelBench')

# 查看包含的级别
print("数据集包含:", list(dataset.keys()))

# 查看 Level 1 的问题数量
print(f"Level 1 有 {len(dataset['level_1'])} 个问题")

# 查看第一个问题
problem = dataset['level_1'][0]
print(f"问题名称: {problem['name']}")
```

## 📊 数据集大小

- 总大小: ~100MB
- Level 1: 15个问题
- Level 2: 20个问题
- Level 3: 30个问题

## ❓ 常见问题

### Q: 为什么下载很慢？

A: 使用镜像站可以大幅提升速度。已在代码中默认配置。

### Q: 如何清空缓存重新下载？

A:
```bash
# Windows
rm -r C:\Users\你的用户名\.cache\huggingface\datasets

# Linux/Mac
rm -rf ~/.cache/huggingface/datasets
```

### Q: 是否支持离线使用？

A: 是的！数据集下载后会缓存，之后可以离线使用。

### Q: 服务器无法连接 HuggingFace 怎么办？

A:
1. 使用方法 1（镜像站）
2. 或使用方法 3（从其他机器转移）
3. 或配置代理（方法 4）
