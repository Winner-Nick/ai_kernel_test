"""
测试单个模型 - 完全使用 KernelBench 的方法
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 配置 HuggingFace 镜像（解决国内连接问题）
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

# 添加 KernelBench 到路径
kernelbench_path = os.path.join(os.path.dirname(__file__), '../../KernelBench')
sys.path.insert(0, kernelbench_path)

from datasets import load_dataset
from src.eval import eval_kernel_against_ref
from src.utils import extract_first_code, set_gpu_arch
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template

from .config import MODELS_TO_TEST, TEST_CONFIG, OUTPUT_DIR
from .custom_inference import create_custom_inference_function


def test_single_model(model_config, ref_code, problem_name, backend='cuda', verbose=False):
    """
    测试单个模型

    Args:
        model_config: 模型配置字典
        ref_code: 参考代码
        problem_name: 问题名称
        backend: 后端类型 (cuda/triton)
        verbose: 是否显示详细信息

    Returns:
        dict: 测试结果
    """
    model_name = model_config['name']
    model_id = model_config['model_id']

    print(f"\n{'='*70}")
    print(f"🤖 测试模型: {model_name} ({model_id})")
    print(f"📝 问题: {problem_name}")
    print(f"{'='*70}")

    # 1. 使用 KernelBench 的官方 prompt
    print("⏳ 生成 prompt...")
    if backend == 'cuda':
        prompt = prompt_generate_custom_cuda_from_prompt_template(ref_code)
    else:
        from src.prompt_constructor_multilang import get_prompt_for_backend
        prompt = get_prompt_for_backend(ref_code, backend)

    # 2. 使用自定义 API 调用生成代码
    print("⏳ 调用模型生成代码...")
    inference_fn = create_custom_inference_function(
        model_id=model_id,
        temperature=TEST_CONFIG['temperature'],
        max_tokens=TEST_CONFIG['max_tokens'],
        verbose=verbose
    )

    try:
        start_time = datetime.now()
        generated_text = inference_fn(prompt)
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        # 3. 使用 KernelBench 的代码提取方法
        generated_code = extract_first_code(generated_text, ["python", "cpp"])

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

    # 4. 使用 KernelBench 的官方评估方法
    print(f"\n🔍 开始评估代码（正确性 + 性能）...")
    try:
        eval_result = eval_kernel_against_ref(
            ref_code,
            generated_code,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=TEST_CONFIG['num_correct_trials'],
            num_perf_trials=TEST_CONFIG['num_perf_trials'],
            backend=backend,
        )

        # KernelExecResult 是一个 Pydantic model，需要直接访问属性
        is_compiled = eval_result.compiled if eval_result else False
        is_correct = eval_result.correctness if eval_result else False
        custom_time_ms = eval_result.runtime / 1000.0 if (eval_result and eval_result.runtime > 0) else 0.0  # 转换为 ms

        # 如果编译失败或正确性测试失败
        if not is_compiled or not is_correct:
            print(f"\n📈 评估结果:")
            print(f"  ❌ 编译: {'通过 ✓' if is_compiled else '失败 ✗'}")
            print(f"  ❌ 正确性: {'通过 ✓' if is_correct else '失败 ✗'}")
            if eval_result and eval_result.metadata:
                print(f"  ℹ️  错误信息: {eval_result.metadata}")

            return {
                'model_name': model_name,
                'model_id': model_id,
                'success': True,
                'generation_time': generation_time,
                'generated_code': generated_code,
                'full_response': generated_text,
                'error': None,
                'eval_result': {
                    'compiled': is_compiled,
                    'is_correct': is_correct,
                    'speedup': 0.0,
                    'ref_time_ms': 0.0,
                    'custom_time_ms': custom_time_ms,
                    'fast_1': False,
                    'fast_2': False,
                    'metadata': eval_result.metadata if eval_result else {}
                }
            }

        # 读取预先生成的 baseline 时间（照搬 KernelBench 的方式）
        print("  ⏱️  读取参考实现 baseline 时间...")
        baseline_path = Path(OUTPUT_DIR) / 'baseline_time.json'

        if not baseline_path.exists():
            print(f"  ⚠️  找不到 baseline 文件: {baseline_path}")
            print(f"  ⚠️  请先运行: python generate_baseline.py")
            print(f"  ⚠️  将使用生成代码时间作为基准 (加速比 = 1.0x)")
            ref_time_ms = custom_time_ms
        else:
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                ref_time_ms = baseline_data['runtime_mean_ms']
                print(f"  ✅ 读取到 baseline: {ref_time_ms:.4f} ms")
            except Exception as e:
                print(f"  ⚠️  读取 baseline 失败: {str(e)}")
                print(f"  ⚠️  将使用生成代码时间作为基准 (加速比 = 1.0x)")
                ref_time_ms = custom_time_ms

        # 计算加速比
        speedup = ref_time_ms / custom_time_ms if custom_time_ms > 0 else 0.0

        print(f"\n📈 评估结果:")
        print(f"  ✅ 编译: 通过 ✓")
        print(f"  ✅ 正确性: 通过 ✓")
        print(f"  ⏱️  PyTorch 时间: {ref_time_ms:.4f} ms")
        print(f"  ⏱️  生成代码时间: {custom_time_ms:.4f} ms")
        print(f"  🚀 加速比: {speedup:.2f}x")
        if speedup > 1.0:
            print(f"  🎉 比 PyTorch 快 {(speedup-1)*100:.1f}%!")
        elif speedup < 1.0:
            print(f"  🐌 比 PyTorch 慢 {(1-speedup)*100:.1f}%")

        return {
            'model_name': model_name,
            'model_id': model_id,
            'success': True,
            'generation_time': generation_time,
            'generated_code': generated_code,
            'full_response': generated_text,
            'error': None,
            'eval_result': {
                'compiled': is_compiled,
                'is_correct': is_correct,
                'speedup': speedup,
                'ref_time_ms': ref_time_ms,
                'custom_time_ms': custom_time_ms,
                'fast_1': is_correct and speedup > 1.0,
                'fast_2': is_correct and speedup > 2.0,
                'metadata': eval_result.metadata if eval_result else {}
            }
        }

    except Exception as e:
        import traceback
        print(f"❌ 评估失败: {str(e)}")
        if verbose:
            traceback.print_exc()
        return {
            'model_name': model_name,
            'model_id': model_id,
            'success': True,
            'generation_time': generation_time,
            'generated_code': generated_code,
            'full_response': generated_text,
            'error': f"Evaluation error: {str(e)}",
            'eval_result': None,
        }


def check_msvc_compiler():
    """检查 Windows 上是否有 MSVC 编译器"""
    import subprocess
    import platform

    if platform.system() != 'Windows':
        return True

    try:
        result = subprocess.run(['where', 'cl'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except:
        pass

    return False


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 使用 KernelBench 官方方法测试 AI 模型")
    print("   仅替换 API 调用为自定义中转")
    print("=" * 80)

    # 检查 GPU
    import torch
    if not torch.cuda.is_available():
        print("⚠️  警告: 未检测到 CUDA GPU！")
        print("   评估需要 GPU 才能运行。")
        return

    print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")

    # 检查编译器
    if not check_msvc_compiler():
        print("\n⚠️  警告: 未检测到 MSVC 编译器 (cl.exe)！")
        print("   Windows 上编译 CUDA 扩展需要 Visual Studio。")
        print("   请安装 Visual Studio 2019/2022 (包含 C++ 开发工具)。")
        print("\n   或者使用 'Developer Command Prompt for VS' 运行此脚本。")
        response = input("\n是否继续？(y/n): ")
        if response.lower() != 'y':
            print("❌ 测试取消")
            return

    # 设置 GPU 架构
    if TEST_CONFIG.get('gpu_arch'):
        set_gpu_arch(TEST_CONFIG['gpu_arch'])

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取问题
    level = TEST_CONFIG['level']
    problem_id = TEST_CONFIG['problem_id']
    backend = TEST_CONFIG['backend']

    print(f"\n📥 正在从 HuggingFace 加载数据集...")
    dataset = load_dataset(TEST_CONFIG['dataset_name'])
    curr_level_dataset = dataset[f"level_{level}"]

    curr_problem = curr_level_dataset.filter(
        lambda x: x["problem_id"] == problem_id
    )

    if len(curr_problem) == 0:
        print(f"❌ 找不到 Level {level} Problem {problem_id}")
        return

    ref_code = curr_problem["code"][0]
    problem_name = curr_problem["name"][0]
    print(f"✅ 成功加载问题: {problem_name}")

    # 保存参考代码
    ref_code_path = output_path / f"reference_level{level}_problem{problem_id}.py"
    with open(ref_code_path, 'w', encoding='utf-8') as f:
        f.write(ref_code)
    print(f"💾 参考代码已保存到: {ref_code_path}")

    # 测试所有模型
    results = []
    for i, model_config in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'#'*80}")
        print(f"# 测试进度: {i}/{len(MODELS_TO_TEST)}")
        print(f"{'#'*80}")

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
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)
    print(f"问题: Level {level} Problem {problem_id} - {problem_name}")
    print(f"Backend: {backend}")

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


if __name__ == "__main__":
    main()
