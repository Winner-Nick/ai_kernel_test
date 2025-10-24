"""
生成 baseline 时间 - 只测量 config 里定义的问题
完全照搬 KernelBench 的 generate_baseline_time.py
"""
import sys
import os
import json
import torch

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

# 添加 KernelBench 到路径
kernelbench_path = os.path.join(os.path.dirname(__file__), '../KernelBench')
sys.path.insert(0, kernelbench_path)

from datasets import load_dataset
from src.eval import load_original_model_and_inputs, time_execution_with_cuda_event, get_timing_stats, set_seed
from src.utils import set_gpu_arch

from kernelbench_wrapper.config import TEST_CONFIG, OUTPUT_DIR


def measure_reference_time(ref_code, num_trials=100, device=None, backend='cuda', verbose=False):
    """
    测量参考实现的性能 - 照搬 KernelBench 的 measure_program_time

    Args:
        ref_code: 参考代码
        num_trials: 测试次数
        device: CUDA 设备
        backend: 后端类型
        verbose: 是否显示详细信息

    Returns:
        dict: 运行时统计信息
    """
    if device is None:
        device = torch.cuda.current_device()

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(ref_code, context)

    if Model is None or get_init_inputs is None or get_inputs is None:
        raise ValueError("参考代码无法加载必需的函数")

    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()

            # 将输入移到 GPU
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # 初始化模型（PyTorch Eager 模式）
            model = Model(*init_inputs)
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)

            # 使用 CUDA Event 计时
            elapsed_times = time_execution_with_cuda_event(
                model, *inputs, num_trials=num_trials, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"参考实现时间统计: {runtime_stats}")

            return runtime_stats

    except Exception as e:
        print(f"❌ 测量参考实现性能失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数 - 生成 baseline 时间"""
    print("=" * 80)
    print("⏱️  生成 Baseline 时间 (参考实现性能)")
    print("   照搬 KernelBench 的 generate_baseline_time.py")
    print("=" * 80)

    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA GPU！")
        return

    device = torch.cuda.current_device()
    print(f"✅ GPU: {torch.cuda.get_device_name(device)}")

    # 设置 GPU 架构
    if TEST_CONFIG.get('gpu_arch'):
        set_gpu_arch(TEST_CONFIG['gpu_arch'])

    # 获取问题
    level = TEST_CONFIG['level']
    problem_id = TEST_CONFIG['problem_id']
    backend = TEST_CONFIG['backend']
    num_perf_trials = TEST_CONFIG['num_perf_trials']

    print(f"\n📝 配置:")
    print(f"  - Level: {level}")
    print(f"  - Problem ID: {problem_id}")
    print(f"  - Backend: {backend}")
    print(f"  - 性能测试次数: {num_perf_trials}")

    print(f"\n📥 从 HuggingFace 加载数据集...")
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
    print(f"✅ 问题: {problem_name}")

    # 测量参考实现性能
    print(f"\n⏱️  开始测量参考实现性能...")
    print(f"   (预热 3 次，测试 {num_perf_trials} 次)")

    runtime_stats = measure_reference_time(
        ref_code,
        num_trials=num_perf_trials,
        device=device,
        backend=backend,
        verbose=False
    )

    if runtime_stats is None:
        print("❌ 测量失败")
        return

    # 保存结果
    baseline_data = {
        'level': level,
        'problem_id': problem_id,
        'problem_name': problem_name,
        'backend': backend,
        'device': str(torch.cuda.get_device_name(device)),
        'num_trials': num_perf_trials,
        'runtime_stats': runtime_stats,
        'runtime_mean_us': runtime_stats['mean'],  # 微秒
        'runtime_mean_ms': runtime_stats['mean'] / 1000.0,  # 毫秒
    }

    # 保存到文件
    output_path = os.path.join(OUTPUT_DIR, 'baseline_time.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Baseline 测量完成!")
    print(f"   平均时间: {baseline_data['runtime_mean_ms']:.4f} ms")
    print(f"   标准差: {runtime_stats.get('std', 0) / 1000.0:.4f} ms")
    print(f"💾 结果已保存到: {output_path}")

    print("\n🎉 现在可以运行 python run_kernelbench_wrapper.py 来测试模型了")


if __name__ == "__main__":
    main()
