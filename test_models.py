"""
测试不同 AI 模型对 Level 1 第一个问题的 CUDA 核函数生成能力
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加 KernelBench 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../KernelBench'))

from datasets import load_dataset
from config import MODELS_TO_TEST, TEST_CONFIG, OUTPUT_DIR

def extract_first_code(text, languages=None):
    """从 LLM 输出中提取代码块"""
    if languages is None:
        languages = ["python", "cpp", "cuda"]

    # 尝试提取代码块
    for lang in languages:
        if f"```{lang}" in text:
            start = text.find(f"```{lang}") + len(f"```{lang}")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

    # 尝试提取通用代码块
    if "```" in text:
        start = text.find("```") + 3
        # 跳过语言标识符
        newline = text.find("\n", start)
        if newline != -1:
            start = newline + 1
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    return text


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
    """生成提示词（简化版）"""
    if backend == 'cuda':
        prompt = f"""You are an expert CUDA programmer. Given the following PyTorch reference implementation, generate an efficient CUDA kernel implementation.

Reference PyTorch code:
```python
{ref_code}
```

Please provide:
1. A complete CUDA implementation with kernel function and wrapper
2. Optimized for performance (use shared memory, coalescing, etc.)
3. The code should be directly compilable and callable from Python using PyTorch

Generate only the CUDA implementation code."""
    else:
        prompt = f"""Given the following PyTorch reference implementation, generate an efficient {backend} implementation.

Reference PyTorch code:
```python
{ref_code}
```

Generate the optimized {backend} implementation code."""

    return prompt


def test_single_model(model_config, ref_code, problem_name):
    """测试单个模型"""
    model_name = model_config['name']
    model_id = model_config['model_id']
    client = model_config['client']

    print(f"\n🤖 测试模型: {model_name} ({model_id})")
    print(f"📝 问题: {problem_name}")

    # 生成提示词
    prompt = generate_prompt(ref_code, TEST_CONFIG['backend'])

    # 调用模型
    try:
        print(f"⏳ 正在生成代码...")
        start_time = datetime.now()

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an expert GPU programmer specializing in high-performance CUDA kernels."},
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

        print(f"✅ 代码生成成功！耗时: {generation_time:.2f} 秒")
        print(f"📊 生成代码长度: {len(generated_code)} 字符")

        result = {
            'model_name': model_name,
            'model_id': model_id,
            'success': True,
            'generation_time': generation_time,
            'code_length': len(generated_code),
            'generated_code': generated_code,
            'full_response': generated_text,
            'error': None,
        }

    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        result = {
            'model_name': model_name,
            'model_id': model_id,
            'success': False,
            'generation_time': None,
            'code_length': 0,
            'generated_code': None,
            'full_response': None,
            'error': str(e),
        }

    return result


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 AI 模型 GPU 核函数生成能力测试")
    print("=" * 60)

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # 获取问题
    level = TEST_CONFIG['level']
    problem_id = TEST_CONFIG['problem_id']

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
    for model_config in MODELS_TO_TEST:
        result = test_single_model(model_config, ref_code, problem_name)
        results.append(result)

        # 保存生成的代码
        if result['success']:
            code_filename = f"{model_config['model_id'].replace('/', '_')}_level{level}_problem{problem_id}.py"
            code_path = output_path / code_filename
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(result['generated_code'])
            print(f"💾 生成的代码已保存到: {code_path}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp}.json"
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
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"问题: Level {level} Problem {problem_id} - {problem_name}")
    print(f"测试模型数: {len(MODELS_TO_TEST)}")
    print(f"成功生成: {sum(1 for r in results if r['success'])}/{len(results)}")
    print("\n各模型详情:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        time_info = f"{result['generation_time']:.2f}s" if result['success'] else "N/A"
        print(f"  {status} {result['model_name']:30s} - {time_info}")

    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()
