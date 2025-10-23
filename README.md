# Nick's KernelBench Test

AI模型 GPU 核函数生成能力测试框架，基于 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 进行评估。

## 🎯 项目简介

本项目用于测试和评估不同 AI 模型（GPT-4、Claude、DeepSeek 等）生成高性能 CUDA kernel 代码的能力。使用 KernelBench 官方的评估方法，确保结果准确可靠。

**核心特点**：
- ✅ 使用 KernelBench 官方 prompt 和评估方法
- ✅ 支持自定义 API 中转（OpenAI 兼容格式）
- ✅ 完整的正确性和性能测试
- ✅ 支持多种模型并行测试

## 📁 项目结构

```
nick_kernelbench_test/
├── kernelbench_wrapper/        # 主要测试框架
│   ├── config.py               # 配置文件（模型、测试参数）
│   ├── custom_inference.py     # 自定义 API 调用
│   ├── test_single_model.py    # 测试逻辑
│   └── results/                # 测试结果输出
├── test_models.py              # 简单测试脚本（无评估）
├── test_models_with_eval.py    # 带评估的测试脚本
├── clean_results.py            # 清理结果目录
├── run_kernelbench_wrapper.py  # 主运行入口
├── .env.example                # 环境变量模板
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA-capable GPU (NVIDIA)
- PyTorch with CUDA support
- Visual Studio 2019/2022 (Windows, 用于编译 CUDA 扩展)

### 2. 安装依赖

```bash
# 创建 conda 环境
conda create -n kernel-bench python=3.10 -y
conda activate kernel-bench

# 安装 PyTorch (根据你的 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install datasets openai python-dotenv transformers setuptools
```

### 3. 配置 API

复制环境变量模板并填入你的 API 信息：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
API_BASE_URL=http://your-api-server:port/v1/
API_KEY=your-api-key-here
```

### 4. 配置测试参数

编辑 `kernelbench_wrapper/config.py`：

```python
# 选择要测试的模型
MODELS_TO_TEST = [
    {'name': 'GPT-4', 'model_id': 'gpt-4', 'server_type': 'openai'},
    {'name': 'Claude', 'model_id': 'claude-sonnet-4', 'server_type': 'openai'},
    # ...
]

# 测试配置
TEST_CONFIG = {
    'level': 1,              # KernelBench 难度级别
    'problem_id': 1,         # 问题 ID
    'backend': 'cuda',       # cuda/triton
    'num_correct_trials': 5, # 正确性测试次数
    'num_perf_trials': 100,  # 性能测试次数
}
```

### 5. 运行测试

```bash
python run_kernelbench_wrapper.py
```

## 📊 测试结果

测试完成后，结果保存在 `kernelbench_wrapper/results/` 目录：

- `reference_level{X}_problem{Y}.py` - 参考实现
- `{model_id}_level{X}_problem{Y}.py` - 各模型生成的代码
- `test_results_{timestamp}.json` - 完整测试结果（JSON格式）

结果包含：
- ✅ 代码生成成功率
- ✅ 编译成功率
- ✅ 正确性通过率
- ✅ 性能加速比（相对于 PyTorch 参考实现）

## 🔧 常见问题

### 1. CUDA 编译器找不到 (Windows)

**问题**: `Error checking compiler version for cl`

**解决方案**:
- 安装 Visual Studio 2019/2022（包含 C++ 开发工具）
- 或使用 "Developer Command Prompt for VS" 运行脚本

### 2. GPU 架构不兼容

**问题**: `CUDA capability sm_120 is not compatible`

**解决方案**:
```python
# 在 config.py 中设置后备架构
TEST_CONFIG = {
    'gpu_arch': ['Hopper', 'Ada'],  # 使用最新的支持架构
}
```

### 3. distutils 模块错误

**问题**: `module 'distutils' has no attribute '_msvccompiler'`

**解决方案**:
```bash
pip install setuptools==69.5.1
```

## 📚 参考资源

- [KernelBench 官方仓库](https://github.com/ScalingIntelligence/KernelBench)
- [KernelBench 论文](https://arxiv.org/abs/2410.23552)
- [PyTorch C++ 扩展文档](https://pytorch.org/tutorials/advanced/cpp_extension.html)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 🙏 致谢

本项目基于 [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 构建，感谢 Scaling Intelligence 团队的优秀工作！
