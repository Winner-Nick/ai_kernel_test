"""
测试不同 AI 模型对 Level 1 第一个问题的 CUDA 核函数生成能力
使用 KernelBench 的完整评估方法（正确性 + 性能）
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加 KernelBench 到路径
kernelbench_path = os.path.join(os.path.dirname(__file__), '../KernelBench')
sys.path.insert(0, kernelbench_path)

from datasets import load_dataset
from config import MODELS_TO_TEST, TEST_CONFIG, OUTPUT_DIR

# 从 KernelBench 导入评估函数
from src.eval import eval_kernel_against_ref
from src.utils import extract_first_code, set_gpu_arch


def get_problem_from_dataset(level, problem_id):
    """从 HuggingFace 获取问题"""
    print(f"\n📥 正在从 HuggingFace 加载数据集...")
    dataset = load_dataset(TEST_CONFIG['dataset_name'])
    curr_level_dataset = dataset[f"level_{level}"]

    # 过滤出指定问题
    curr_problem = curr_level_dataset.filter(
        lambda x: x["problem_id"] == problem_id
    )

    if len(curr_problem) == 0:
        raise ValueError(f"找不到 Level {level} Problem {problem_id}")

    ref_code = curr_problem["code"][0]
    problem_name = curr_problem["name"][0]

    print(f"✅ 成功加载问题: {problem_name}")
    return ref_code, problem_name


def generate_prompt(ref_code, backend='cuda'):
    """生成提示词"""
    if backend == 'cuda':
        prompt = f"""You are an expert CUDA programmer. Given the following PyTorch reference implementation, generate an optimized version using custom CUDA kernels.

Reference PyTorch code:
```python
{ref_code}
```

Requirements:
1. Output must be VALID PYTHON CODE that can be directly imported and executed
2. Use torch.utils.cpp_extension.load_inline() to compile CUDA kernels inline
3. Put CUDA kernel code in a Python string variable (e.g., cuda_source = \"\"\"...\"\"\")
4. Create a Model class (or ModelNew) that uses the compiled kernel
5. Include get_inputs() function from the reference code
6. Optimize for performance using shared memory, tiling, and memory coalescing

Example structure:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
__global__ void my_kernel(...) {{
    // Your CUDA kernel code
}}
torch::Tensor my_function(...) {{
    // C++ wrapper that launches kernel
}}
\"\"\"

cpp_source = \"\"\"
torch::Tensor my_function(...);
\"\"\"

custom_module = load_inline(
    name='custom_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['my_function'],
    verbose=True
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_module = custom_module

    def forward(self, ...):
        return self.custom_module.my_function(...)

def get_inputs():
    # Same as reference implementation
    return [...]
```

Generate the complete Python code with inline CUDA kernel."""

    elif backend == 'triton':
        prompt = f"""You are an expert Triton programmer. Given the following PyTorch reference implementation, generate an efficient Triton kernel implementation.

Reference PyTorch code:
```python
{ref_code}
```

Requirements:
1. Generate a complete Triton implementation
2. Optimize for performance using Triton's features
3. Match the exact interface and behavior of the reference implementation

Generate the complete Triton implementation code."""

    else:
        prompt = f"""Given the following PyTorch reference implementation, generate an efficient {backend} implementation.

Reference PyTorch code:
```python
{ref_code}
```

Generate the optimized {backend} implementation code that matches the reference behavior."""

    return prompt


def test_single_model(model_config, ref_code, problem_name, backend='cuda', verbose=False):
    """测试单个模型（包含完整评估）"""
    model_name = model_config['name']
    model_id = model_config['model_id']
    client = model_config['client']

    print(f"\n{'='*70}")
    print(f"🤖 测试模型: {model_name} ({model_id})")
    print(f"📝 问题: {problem_name}")
    print(f"{'='*70}")

    # 生成提示词
    prompt = generate_prompt(ref_code, backend)

    # 调用模型生成代码
    try:
        print(f"⏳ 正在生成代码...")
        start_time = datetime.now()

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an expert GPU programmer specializing in high-performance kernel optimization."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEST_CONFIG['temperature'],
            max_tokens=TEST_CONFIG['max_tokens'],
        )

        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        # 提取生成的代码
        generated_text = response.choices[0].message.content
        generated_code = extract_first_code(generated_text, ["python", "cpp", "cuda"])

        if not generated_code:
            raise ValueError("未能从响应中提取代码")

        print(f"✅ 代码生成成功！耗时: {generation_time:.2f} 秒")
        print(f"📊 生成代码长度: {len(generated_code)} 字符")

    except Exception as e:
        print(f"❌ 代码生成失败: {str(e)}")
        return {
            'model_name': model_name,
            'model_id': model_id,
            'success': False,
            'generation_time': None,
            'generated_code': None,
            'error': str(e),
            'eval_result': None,
        }

    # 使用 KernelBench 的评估函数进行完整评估
    print(f"\n🔍 开始评估代码（正确性 + 性能）...")
    try:
        eval_result = eval_kernel_against_ref(
            ref_code,
            generated_code,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=5,  # 正确性测试次数
            num_perf_trials=100,   # 性能测试次数
            backend=backend,
        )

        # 解析评估结果
        is_correct = eval_result.get('is_correct', False)
        speedup = eval_result.get('speedup', 0.0)
        ref_time = eval_result.get('ref_time_ms', 0.0)
        custom_time = eval_result.get('custom_time_ms', 0.0)

        print(f"\n📈 评估结果:")
        print(f"  ✅ 正确性: {'通过 ✓' if is_correct else '失败 ✗'}")
        if is_correct:
            print(f"  ⏱️  PyTorch 时间: {ref_time:.4f} ms")
            print(f"  ⏱️  生成代码时间: {custom_time:.4f} ms")
            print(f"  🚀 加速比: {speedup:.2f}x")
            if speedup > 1.0:
                print(f"  🎉 比 PyTorch 快 {(speedup-1)*100:.1f}%!")
            elif speedup < 1.0:
                print(f"  🐌 比 PyTorch 慢 {(1-speedup)*100:.1f}%")
            else:
                print(f"  ⚖️  与 PyTorch 速度相当")

        return {
            'model_name': model_name,
            'model_id': model_id,
            'success': True,
            'generation_time': generation_time,
            'generated_code': generated_code,
            'full_response': generated_text,
            'error': None,
            'eval_result': {
                'is_correct': is_correct,
                'speedup': speedup,
                'ref_time_ms': ref_time,
                'custom_time_ms': custom_time,
                'fast_1': is_correct and speedup > 1.0,  # 正确且快于PyTorch
                'fast_2': is_correct and speedup > 2.0,  # 正确且快2倍以上
            }
        }

    except Exception as e:
        print(f"❌ 评估失败: {str(e)}")
        return {
            'model_name': model_name,
            'model_id': model_id,
            'success': True,  # 生成成功，但评估失败
            'generation_time': generation_time,
            'generated_code': generated_code,
            'full_response': generated_text,
            'error': f"Evaluation error: {str(e)}",
            'eval_result': None,
        }


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 AI 模型 GPU 核函数生成能力完整测试")
    print("   包含 KernelBench 官方评估（正确性 + 性能）")
    print("=" * 80)

    # 检查 GPU
    import torch
    if not torch.cuda.is_available():
        print("⚠️  警告: 未检测到 CUDA GPU！")
        print("   评估需要 GPU 才能运行。")
        response = input("是否继续（只生成代码，跳过评估）？(y/n): ")
        if response.lower() != 'y':
            print("❌ 测试取消")
            return
        skip_eval = True
    else:
        skip_eval = False
        print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
        # 设置 GPU 架构
        if TEST_CONFIG.get('gpu_arch'):
            set_gpu_arch(TEST_CONFIG['gpu_arch'])

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # 获取问题
    level = TEST_CONFIG['level']
    problem_id = TEST_CONFIG['problem_id']
    backend = TEST_CONFIG['backend']

    try:
        ref_code, problem_name = get_problem_from_dataset(level, problem_id)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    # 保存参考代码
    ref_code_path = output_path / f"reference_level{level}_problem{problem_id}.py"
    with open(ref_code_path, 'w', encoding='utf-8') as f:
        f.write(ref_code)
    print(f"\n💾 参考代码已保存到: {ref_code_path}")

    # 测试所有模型
    results = []
    for i, model_config in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'#'*80}")
        print(f"# 测试进度: {i}/{len(MODELS_TO_TEST)}")
        print(f"{'#'*80}")

        if skip_eval:
            print("⚠️  跳过评估（无 GPU）")
            # TODO: 实现只生成不评估的版本
            continue

        result = test_single_model(
            model_config,
            ref_code,
            problem_name,
            backend=backend,
            verbose=False
        )
        results.append(result)

        # 保存生成的代码
        if result['success'] and result['generated_code']:
            code_filename = f"{model_config['model_id'].replace('/', '_')}_level{level}_problem{problem_id}.py"
            code_path = output_path / code_filename
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(result['generated_code'])
            print(f"💾 生成的代码已保存到: {code_path}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_with_eval_{timestamp}.json"
    results_path = output_path / results_filename

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_config': TEST_CONFIG,
            'problem_name': problem_name,
            'results': results,
            'timestamp': timestamp,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 完整结果已保存到: {results_path}")

    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    print(f"问题: Level {level} Problem {problem_id} - {problem_name}")
    print(f"Backend: {backend}")
    print(f"\n模型测试结果:")

    success_count = sum(1 for r in results if r['success'])
    correct_count = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('is_correct'))
    fast_1_count = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_1'))
    fast_2_count = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_2'))

    print(f"\n总体统计:")
    print(f"  📝 测试模型数: {len(results)}")
    print(f"  ✅ 成功生成: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  ✓  正确性通过: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"  🚀 fast_1 (快于PyTorch): {fast_1_count}/{len(results)} ({fast_1_count/len(results)*100:.1f}%)")
    print(f"  🏆 fast_2 (快2倍以上): {fast_2_count}/{len(results)} ({fast_2_count/len(results)*100:.1f}%)")

    print(f"\n各模型详情:")
    print(f"{'模型':<35} {'生成':<8} {'正确':<8} {'加速比':<10} {'fast_1':<8}")
    print("-" * 80)
    for result in results:
        name = result['model_name'][:33]
        gen_status = "✅" if result['success'] else "❌"

        if result.get('eval_result'):
            eval_res = result['eval_result']
            correct = "✅" if eval_res.get('is_correct') else "❌"
            speedup = f"{eval_res.get('speedup', 0):.2f}x"
            fast1 = "✅" if eval_res.get('fast_1') else "❌"
        else:
            correct = "N/A"
            speedup = "N/A"
            fast1 = "N/A"

        print(f"{name:<35} {gen_status:<8} {correct:<8} {speedup:<10} {fast1:<8}")

    print("\n🎉 测试完成！")
    print(f"💡 提示: 运行 'python analyze_results_with_eval.py' 查看详细分析")


if __name__ == "__main__":
    main()
