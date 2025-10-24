"""
诊断脚本 - 检查 eval_kernel_against_ref 为什么返回 None
"""
import sys
import os

# 添加 KernelBench 到路径
kernelbench_path = os.path.join(os.path.dirname(__file__), '../KernelBench')
sys.path.insert(0, kernelbench_path)

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

from datasets import load_dataset
from src.eval import eval_kernel_against_ref
import torch

print("=" * 80)
print("🔍 诊断 eval_kernel_against_ref 返回 None 的问题")
print("=" * 80)

# 检查 GPU
if not torch.cuda.is_available():
    print("❌ 没有检测到 CUDA GPU！")
    sys.exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# 加载数据集
print("\n📥 加载数据集...")
dataset = load_dataset('ScalingIntelligence/KernelBench')
curr_problem = dataset["level_1"].filter(lambda x: x["problem_id"] == 1)

if len(curr_problem) == 0:
    print("❌ 找不到问题")
    sys.exit(1)

ref_code = curr_problem["code"][0]
problem_name = curr_problem["name"][0]
print(f"✅ 问题: {problem_name}")

# 检查参考代码
print("\n📝 检查参考代码...")
print(f"代码长度: {len(ref_code)} 字符")
print("\n前 500 字符:")
print("-" * 80)
print(ref_code[:500])
print("-" * 80)

# 尝试编译参考代码
print("\n🔧 测试 1: 编译参考代码...")
try:
    compile(ref_code, "<string>", "exec")
    print("✅ 编译成功")
except SyntaxError as e:
    print(f"❌ 语法错误: {e}")
    sys.exit(1)

# 尝试执行参考代码
print("\n🔧 测试 2: 执行参考代码...")
context = {}
try:
    exec(ref_code, context)
    print("✅ 执行成功")
except Exception as e:
    print(f"❌ 执行错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查必需的函数
print("\n🔧 测试 3: 检查必需的函数...")
required_items = ['Model', 'get_init_inputs', 'get_inputs']
for item in required_items:
    if item in context:
        print(f"✅ 找到 {item}: {type(context[item])}")
    else:
        print(f"❌ 缺少 {item}")

# 测试 get_init_inputs
print("\n🔧 测试 4: 调用 get_init_inputs...")
try:
    get_init_inputs = context.get('get_init_inputs')
    if get_init_inputs is None:
        print("❌ get_init_inputs 是 None")
    else:
        init_inputs = get_init_inputs()
        print(f"✅ get_init_inputs() 成功，返回 {len(init_inputs) if hasattr(init_inputs, '__len__') else '?'} 个参数")
except Exception as e:
    print(f"❌ 调用 get_init_inputs 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 get_inputs
print("\n🔧 测试 5: 调用 get_inputs...")
try:
    get_inputs = context.get('get_inputs')
    if get_inputs is None:
        print("❌ get_inputs 是 None")
    else:
        inputs = get_inputs()
        print(f"✅ get_inputs() 成功，返回 {len(inputs) if hasattr(inputs, '__len__') else '?'} 个参数")
except Exception as e:
    print(f"❌ 调用 get_inputs 失败: {e}")
    import traceback
    traceback.print_exc()

# 尝试完整的评估（只测试正确性，不测性能）
print("\n🔧 测试 6: 完整评估（不测性能）...")
try:
    result = eval_kernel_against_ref(
        ref_code,
        ref_code,
        verbose=True,
        measure_performance=False,
        num_correct_trials=1,
        backend='cuda',
    )
    if result is None:
        print("❌ 评估返回 None")
    else:
        print(f"✅ 评估成功")
        print(f"   - compiled: {result.compiled}")
        print(f"   - correctness: {result.correctness}")
        print(f"   - runtime: {result.runtime}")
except Exception as e:
    print(f"❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

# 尝试完整的评估（测试性能）
print("\n🔧 测试 7: 完整评估（测试性能）...")
try:
    result = eval_kernel_against_ref(
        ref_code,
        ref_code,
        verbose=True,
        measure_performance=True,
        num_correct_trials=1,
        num_perf_trials=10,
        backend='cuda',
    )
    if result is None:
        print("❌ 评估返回 None")
    else:
        print(f"✅ 评估成功")
        print(f"   - compiled: {result.compiled}")
        print(f"   - correctness: {result.correctness}")
        print(f"   - runtime: {result.runtime}")
except Exception as e:
    print(f"❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("🎉 诊断完成")
print("=" * 80)
