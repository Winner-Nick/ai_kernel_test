# KernelBench Wrapper

完全使用 KernelBench 官方的评估方法，仅替换 API 调用为自定义中转。

## 特点

✅ **使用 KernelBench 官方 prompt** - 确保生成正确格式的代码
✅ **使用 KernelBench 官方评估** - 保证评估结果准确性
✅ **自定义 API 调用** - 使用我们自己的 API 中转
✅ **完整兼容** - 所有 KernelBench 功能都可用

## 目录结构

```
kernelbench_wrapper/
├── config.py              # 配置文件（API、模型、测试参数）
├── custom_inference.py    # 自定义推理函数（替换 API 调用）
├── test_single_model.py   # 测试逻辑（使用 KernelBench 方法）
├── __init__.py           # 模块初始化
└── results/              # 测试结果输出目录
```

## 使用方法

### 1. 配置模型和参数

编辑 `config.py`:

```python
# API 配置
API_BASE_URL = "http://your-api-server/v1/"
API_KEY = "your-api-key"

# 要测试的模型
MODELS_TO_TEST = [
    {'name': 'GPT-4', 'model_id': 'gpt-4', 'server_type': 'openai'},
    # ...
]

# 测试配置
TEST_CONFIG = {
    'level': 1,
    'problem_id': 1,
    'backend': 'cuda',
    'temperature': 0.0,
    'max_tokens': 4096,
}
```

### 2. 运行测试

```bash
# 从 ai_kernel_test 目录运行
python run_kernelbench_wrapper.py
```

### 3. 查看结果

结果保存在 `kernelbench_wrapper/results/` 目录：
- `reference_level1_problem1.py` - 参考代码
- `{model_id}_level1_problem1.py` - 各模型生成的代码
- `test_results_{timestamp}.json` - 完整测试结果

## 与原版的区别

| 组件 | 原版 test_models_with_eval.py | KernelBench Wrapper |
|------|------------------------------|---------------------|
| **Prompt** | 自定义简单 prompt | ✅ KernelBench 官方 prompt（含示例） |
| **API 调用** | 直接使用 OpenAI SDK | ✅ 自定义中转 API |
| **代码提取** | 使用 KernelBench 的 extract_first_code | ✅ 使用 KernelBench 的 extract_first_code |
| **评估方法** | 使用 KernelBench 的 eval_kernel_against_ref | ✅ 使用 KernelBench 的 eval_kernel_against_ref |

## 优势

1. **正确的代码格式** - 使用官方 prompt 确保生成 Python + inline CUDA 格式
2. **准确的评估** - 使用官方评估方法保证结果可靠
3. **灵活的 API** - 可以轻松替换为任何 OpenAI 兼容的 API
4. **完整的功能** - 支持所有 KernelBench 特性（多后端、few-shot 等）

## 注意事项

- 需要 CUDA GPU 才能运行评估
- PyTorch 版本需要支持你的 GPU 架构
- 首次运行会编译 CUDA 代码，可能较慢
