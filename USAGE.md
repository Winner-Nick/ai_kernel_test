# 使用说明

## 完全照搬 KernelBench 的工作流程

我们的实现**完全遵循** KernelBench 官方的评估流程，只替换了 API 调用部分。

---

## 📋 两步流程

### 第一步：生成 Baseline 时间

**必须先运行这个！** 它会测量 PyTorch 参考实现的性能。

```bash
python generate_baseline.py
```

这个脚本会：
- 读取 `config.py` 里定义的 level 和 problem_id
- 加载对应的参考实现
- 使用 KernelBench 的 `time_execution_with_cuda_event` 测量性能
- 保存结果到 `kernelbench_wrapper/results/baseline_time.json`

**输出示例：**
```
================================================================================
⏱️  生成 Baseline 时间 (参考实现性能)
   照搬 KernelBench 的 generate_baseline_time.py
================================================================================
✅ GPU: Tesla V100-PCIE-32GB

📝 配置:
  - Level: 1
  - Problem ID: 1
  - Backend: cuda
  - 性能测试次数: 100

📥 从 HuggingFace 加载数据集...
✅ 问题: 1_Square_matrix_multiplication_

⏱️  开始测量参考实现性能...
   (预热 3 次，测试 100 次)
[Profiling] Using device: 0 Tesla V100-PCIE-32GB, warm up 3, trials 100

✅ Baseline 测量完成!
   平均时间: 0.1234 ms
   标准差: 0.0056 ms
💾 结果已保存到: kernelbench_wrapper/results/baseline_time.json

🎉 现在可以运行 python run_kernelbench_wrapper.py 来测试模型了
```

---

### 第二步：测试 AI 模型

```bash
python run_kernelbench_wrapper.py
```

这个脚本会：
1. 调用 AI 模型生成 CUDA 内核代码
2. 使用 KernelBench 的 `eval_kernel_against_ref` 评估：
   - 编译是否成功
   - 正确性检查
   - 生成代码的运行时间
3. 读取第一步生成的 baseline 时间
4. 计算加速比：`speedup = baseline_time / custom_time`
5. 计算 `fast_1` (是否比 PyTorch 快) 和 `fast_2` (是否快 2 倍以上)

---

## 📁 文件结构

```
ai_kernel_test/
├── config.py                    # 配置文件
├── generate_baseline.py         # 第一步：生成 baseline
├── run_kernelbench_wrapper.py   # 第二步：测试模型
├── kernelbench_wrapper/
│   ├── __init__.py
│   ├── config.py
│   ├── custom_inference.py      # 自定义 API 调用
│   ├── test_single_model.py     # 测试逻辑
│   └── results/
│       ├── baseline_time.json   # Baseline 时间（第一步生成）
│       └── *.py                 # 生成的代码
```

---

## ⚙️ 配置

编辑 `kernelbench_wrapper/config.py`：

```python
# 要测试的模型
MODELS_TO_TEST = [
    {
        'name': 'DeepSeek Chat',
        'model_id': 'deepseek-chat',
        'server_type': 'openai',
    },
    # ... 更多模型
]

# 测试配置
TEST_CONFIG = {
    'level': 1,              # 问题等级
    'problem_id': 1,         # 问题 ID
    'backend': 'cuda',       # 后端 (cuda/triton/cute)
    'num_perf_trials': 100,  # 性能测试次数
    # ...
}
```

---

## 🔄 与 KernelBench 官方的对应关系

| KernelBench 官方 | 我们的实现 |
|-----------------|-----------|
| `scripts/generate_baseline_time.py` | `generate_baseline.py` |
| `scripts/generate_and_eval_single_sample.py` | `run_kernelbench_wrapper.py` |
| `src.utils.create_inference_server_from_presets()` | `custom_inference.create_custom_inference_function()` |
| 读取 `results/timing/baseline_time_torch.json` | 读取 `kernelbench_wrapper/results/baseline_time.json` |

**核心区别：** 我们只是把 KernelBench 的推理函数（调用 DeepSeek API）替换成了我们自己的 API 中转，其他评估逻辑完全一样！

---

## 📊 评估指标

完全照搬 KernelBench 的指标：

- **compiled**: 是否编译成功
- **correctness**: 是否正确（通过随机输入测试）
- **speedup**: 加速比 = PyTorch时间 / 生成代码时间
- **fast_1**: correctness ✅ **且** speedup > 1.0
- **fast_2**: correctness ✅ **且** speedup > 2.0

---

## ❓ 常见问题

### Q: 为什么要先运行 generate_baseline.py？

A: 因为 KernelBench 官方就是这样设计的！参考实现的性能需要预先测量并保存，这样可以：
- 避免重复测量（节省时间）
- 确保公平对比（所有模型使用相同的 baseline）
- 支持离线分析

### Q: 如果我换了问题（改了 level 或 problem_id）怎么办？

A: 需要重新运行 `python generate_baseline.py` 来生成新问题的 baseline。

### Q: 能不能实时测量 baseline？

A: 可以，但那不是 KernelBench 的官方做法。我们选择完全照搬官方流程。

---

## 🎉 总结

1. **第一步**：`python generate_baseline.py` - 测量参考实现
2. **第二步**：`python run_kernelbench_wrapper.py` - 测试 AI 模型
3. **查看结果**：在 `kernelbench_wrapper/results/` 目录

完全照搬 KernelBench，保证结果可靠！
