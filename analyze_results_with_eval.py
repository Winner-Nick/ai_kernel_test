"""
分析包含完整评估的测试结果
"""

import json
import os
from pathlib import Path
from datetime import datetime


def load_latest_results(results_dir='./results', pattern='test_results_with_eval_*.json'):
    """加载最新的测试结果"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return None

    # 找到最新的结果文件
    result_files = list(results_path.glob(pattern))
    if not result_files:
        print(f"❌ 未找到测试结果文件: {pattern}")
        return None

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 加载结果文件: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_performance(results_data):
    """详细分析性能数据"""
    print("\n" + "=" * 80)
    print("📊 KernelBench 完整评估结果分析")
    print("=" * 80)

    problem_name = results_data.get('problem_name', 'Unknown')
    test_config = results_data.get('test_config', {})
    results = results_data.get('results', [])

    print(f"\n📝 测试问题: {problem_name}")
    print(f"🎯 Level: {test_config.get('level')}, Problem ID: {test_config.get('problem_id')}")
    print(f"🔧 Backend: {test_config.get('backend')}")
    print(f"📅 测试时间: {results_data.get('timestamp')}")

    # 整体统计
    total = len(results)
    generated = sum(1 for r in results if r['success'])
    evaluated = sum(1 for r in results if r.get('eval_result') is not None)
    correct = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('is_correct'))
    fast_1 = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_1'))
    fast_2 = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_2'))

    print(f"\n📈 整体统计:")
    print(f"  测试模型数: {total}")
    print(f"  成功生成代码: {generated}/{total} ({generated/total*100:.1f}%)")
    print(f"  完成评估: {evaluated}/{total} ({evaluated/total*100:.1f}%)")
    print(f"  ✅ fast_0 (正确性): {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  🚀 fast_1 (正确 且 快于PyTorch): {fast_1}/{total} ({fast_1/total*100:.1f}%)")
    print(f"  🏆 fast_2 (正确 且 快2倍以上): {fast_2}/{total} ({fast_2/total*100:.1f}%)")

    # 详细表格
    print("\n" + "-" * 100)
    print(f"{'模型':<30} {'生成':<6} {'正确':<6} {'加速比':<10} {'PyTorch(ms)':<12} {'生成(ms)':<12} {'fast_1':<6}")
    print("-" * 100)

    for result in results:
        name = result['model_name'][:28]
        gen = "✅" if result['success'] else "❌"

        if result.get('eval_result'):
            ev = result['eval_result']
            correct_mark = "✅" if ev.get('is_correct') else "❌"
            speedup = f"{ev.get('speedup', 0):.2f}x"
            ref_time = f"{ev.get('ref_time_ms', 0):.4f}"
            custom_time = f"{ev.get('custom_time_ms', 0):.4f}"
            fast1 = "✅" if ev.get('fast_1') else "❌"
        else:
            correct_mark = "N/A"
            speedup = "N/A"
            ref_time = "N/A"
            custom_time = "N/A"
            fast1 = "N/A"

        print(f"{name:<30} {gen:<6} {correct_mark:<6} {speedup:<10} {ref_time:<12} {custom_time:<12} {fast1:<6}")

    # 性能排名
    correct_results = [r for r in results
                      if r.get('eval_result') and r['eval_result'].get('is_correct')]

    if correct_results:
        print("\n" + "=" * 80)
        print("🏅 性能排名 (仅正确的实现)")
        print("=" * 80)

        # 按加速比排序
        correct_results.sort(key=lambda x: x['eval_result']['speedup'], reverse=True)

        print(f"\n{'排名':<6} {'模型':<35} {'加速比':<10} {'实际时间(ms)':<15}")
        print("-" * 80)
        for i, result in enumerate(correct_results, 1):
            ev = result['eval_result']
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            name = result['model_name'][:33]
            speedup = f"{ev['speedup']:.2f}x"
            time = f"{ev['custom_time_ms']:.4f}"
            print(f"{medal:<6} {name:<35} {speedup:<10} {time:<15}")

        # 加速比可视化
        print("\n加速比可视化 (相对于 PyTorch):")
        max_speedup = max(r['eval_result']['speedup'] for r in correct_results)
        for result in correct_results[:10]:  # 只显示前10
            ev = result['eval_result']
            name = result['model_name'][:25]
            speedup = ev['speedup']
            bar_length = int((speedup / max_speedup) * 50)
            bar = '█' * bar_length
            print(f"  {name:<25} {bar} {speedup:.2f}x")

    # 失败分析
    failed_results = [r for r in results if not r['success']]
    eval_failed = [r for r in results
                  if r['success'] and (not r.get('eval_result') or not r['eval_result'].get('is_correct'))]

    if failed_results:
        print("\n" + "=" * 80)
        print("❌ 代码生成失败分析")
        print("=" * 80)
        for result in failed_results:
            print(f"\n模型: {result['model_name']}")
            print(f"错误: {result['error']}")

    if eval_failed:
        print("\n" + "=" * 80)
        print("⚠️  评估失败或正确性未通过")
        print("=" * 80)
        for result in eval_failed:
            print(f"\n模型: {result['model_name']}")
            if result.get('error'):
                print(f"错误: {result['error']}")
            elif result.get('eval_result') and not result['eval_result'].get('is_correct'):
                print(f"原因: 正确性测试未通过")
            else:
                print(f"原因: 未知评估错误")

    # 生成时间分析
    generated_results = [r for r in results if r['success'] and r.get('generation_time')]
    if generated_results:
        print("\n" + "=" * 80)
        print("⏱️  代码生成速度分析")
        print("=" * 80)

        times = [r['generation_time'] for r in generated_results]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\n平均生成时间: {avg_time:.2f}s")
        print(f"最快: {min_time:.2f}s")
        print(f"最慢: {max_time:.2f}s")

        # 排序
        generated_results.sort(key=lambda x: x['generation_time'])
        print(f"\n生成速度排名:")
        for i, result in enumerate(generated_results[:5], 1):  # 前5名
            print(f"  {i}. {result['model_name']:<30} {result['generation_time']:.2f}s")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


def generate_markdown_report(results_data, output_file='results/report_with_eval.md'):
    """生成 Markdown 格式的详细报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AI 模型 GPU 核函数生成能力完整测试报告\n\n")
        f.write("## 使用 KernelBench 官方评估（正确性 + 性能）\n\n")

        problem_name = results_data.get('problem_name', 'Unknown')
        test_config = results_data.get('test_config', {})
        results = results_data.get('results', [])

        f.write(f"### 测试信息\n\n")
        f.write(f"- **问题**: {problem_name}\n")
        f.write(f"- **Level**: {test_config.get('level')}\n")
        f.write(f"- **Problem ID**: {test_config.get('problem_id')}\n")
        f.write(f"- **Backend**: {test_config.get('backend')}\n")
        f.write(f"- **测试时间**: {results_data.get('timestamp')}\n\n")

        # 统计
        total = len(results)
        correct = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('is_correct'))
        fast_1 = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_1'))
        fast_2 = sum(1 for r in results if r.get('eval_result') and r['eval_result'].get('fast_2'))

        f.write(f"### 整体结果\n\n")
        f.write(f"- 测试模型数: {total}\n")
        f.write(f"- **fast_0** (正确性): {correct}/{total} ({correct/total*100:.1f}%)\n")
        f.write(f"- **fast_1** (快于PyTorch): {fast_1}/{total} ({fast_1/total*100:.1f}%)\n")
        f.write(f"- **fast_2** (快2倍以上): {fast_2}/{total} ({fast_2/total*100:.1f}%)\n\n")

        f.write(f"### 详细结果\n\n")
        f.write("| 模型 | 生成 | 正确 | 加速比 | PyTorch时间(ms) | 生成代码时间(ms) | fast_1 |\n")
        f.write("|------|------|------|--------|----------------|-----------------|--------|\n")

        for result in results:
            name = result['model_name']
            gen = "✅" if result['success'] else "❌"

            if result.get('eval_result'):
                ev = result['eval_result']
                correct_mark = "✅" if ev.get('is_correct') else "❌"
                speedup = f"{ev.get('speedup', 0):.2f}x"
                ref_time = f"{ev.get('ref_time_ms', 0):.4f}"
                custom_time = f"{ev.get('custom_time_ms', 0):.4f}"
                fast1 = "✅" if ev.get('fast_1') else "❌"
            else:
                correct_mark = "N/A"
                speedup = "N/A"
                ref_time = "N/A"
                custom_time = "N/A"
                fast1 = "N/A"

            f.write(f"| {name} | {gen} | {correct_mark} | {speedup} | {ref_time} | {custom_time} | {fast1} |\n")

        # 性能排名
        correct_results = [r for r in results
                          if r.get('eval_result') and r['eval_result'].get('is_correct')]
        if correct_results:
            correct_results.sort(key=lambda x: x['eval_result']['speedup'], reverse=True)

            f.write(f"\n### 性能排名 (仅正确的实现)\n\n")
            f.write("| 排名 | 模型 | 加速比 | 实际时间(ms) |\n")
            f.write("|------|------|--------|-------------|\n")

            for i, result in enumerate(correct_results, 1):
                ev = result['eval_result']
                name = result['model_name']
                speedup = f"{ev['speedup']:.2f}x"
                time = f"{ev['custom_time_ms']:.4f}"
                f.write(f"| {i} | {name} | {speedup} | {time} |\n")

        f.write(f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n📝 Markdown 报告已保存到: {output_file}")


def main():
    """主函数"""
    results_data = load_latest_results()

    if not results_data:
        print("❌ 无法加载测试结果")
        print("💡 请先运行 'python test_models_with_eval.py' 进行测试")
        return

    # 生成控制台分析报告
    analyze_performance(results_data)

    # 生成 Markdown 报告
    generate_markdown_report(results_data)


if __name__ == "__main__":
    main()
