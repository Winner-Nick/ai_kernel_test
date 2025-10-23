"""
分析和对比不同模型的生成结果
"""

import json
import os
from pathlib import Path
from datetime import datetime
import re


def load_latest_results(results_dir='./results'):
    """加载最新的测试结果"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return None

    # 找到最新的结果文件
    result_files = list(results_path.glob('test_results_*.json'))
    if not result_files:
        print(f"❌ 未找到测试结果文件")
        return None

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 加载结果文件: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_code_quality(code):
    """简单分析代码质量指标"""
    if not code:
        return {}

    metrics = {
        'lines': len(code.split('\n')),
        'chars': len(code),
        'has_kernel': '__global__' in code or '@triton.jit' in code,
        'has_shared_mem': '__shared__' in code,
        'has_sync': '__syncthreads()' in code,
        'has_optimization_comment': any(keyword in code.lower() for keyword in
                                       ['optimize', 'coalesce', 'shared memory', 'tiling']),
    }

    return metrics


def compare_models(results_data):
    """对比不同模型的表现"""
    print("\n" + "=" * 80)
    print("📊 模型对比分析报告")
    print("=" * 80)

    problem_name = results_data.get('problem_name', 'Unknown')
    test_config = results_data.get('test_config', {})
    results = results_data.get('results', [])

    print(f"\n📝 测试问题: {problem_name}")
    print(f"🎯 Level: {test_config.get('level')}, Problem ID: {test_config.get('problem_id')}")
    print(f"🔧 Backend: {test_config.get('backend')}")
    print(f"📅 测试时间: {results_data.get('timestamp')}")

    # 统计成功率
    total_models = len(results)
    successful = sum(1 for r in results if r['success'])
    success_rate = (successful / total_models * 100) if total_models > 0 else 0

    print(f"\n✅ 成功率: {successful}/{total_models} ({success_rate:.1f}%)")

    # 详细对比表
    print("\n" + "-" * 80)
    print(f"{'模型名称':<30} {'状态':<8} {'耗时(s)':<10} {'代码行数':<10} {'有核函数':<10}")
    print("-" * 80)

    for result in results:
        model_name = result['model_name'][:28]
        status = "✅ 成功" if result['success'] else "❌ 失败"
        time_str = f"{result['generation_time']:.2f}" if result['success'] else "N/A"

        if result['success']:
            metrics = analyze_code_quality(result['generated_code'])
            lines = metrics['lines']
            has_kernel = "是" if metrics['has_kernel'] else "否"
        else:
            lines = "N/A"
            has_kernel = "N/A"

        print(f"{model_name:<30} {status:<8} {time_str:<10} {lines:<10} {has_kernel:<10}")

    # 代码质量详细分析
    print("\n" + "=" * 80)
    print("🔍 代码质量详细分析")
    print("=" * 80)

    for result in results:
        if not result['success']:
            continue

        print(f"\n{'='*60}")
        print(f"模型: {result['model_name']}")
        print(f"{'='*60}")

        metrics = analyze_code_quality(result['generated_code'])

        print(f"  📏 代码行数: {metrics['lines']}")
        print(f"  📝 字符数: {metrics['chars']}")
        print(f"  🎯 包含核函数标记: {'✅' if metrics['has_kernel'] else '❌'}")
        print(f"  💾 使用共享内存: {'✅' if metrics['has_shared_mem'] else '❌'}")
        print(f"  🔄 包含同步操作: {'✅' if metrics['has_sync'] else '❌'}")
        print(f"  💡 包含优化注释: {'✅' if metrics['has_optimization_comment'] else '❌'}")

        # 显示代码片段（前10行）
        all_code_lines = result['generated_code'].split('\n')
        code_lines = all_code_lines[:10]
        print(f"\n  📄 代码预览（前10行）:")
        for i, line in enumerate(code_lines, 1):
            print(f"     {i:2d} | {line}")
        if len(all_code_lines) > 10:
            remaining_lines = len(all_code_lines) - 10
            print(f"     ... (还有 {remaining_lines} 行)")

    # 生成时间对比
    print("\n" + "=" * 80)
    print("⏱️  生成速度对比")
    print("=" * 80)

    successful_results = [r for r in results if r['success']]
    if successful_results:
        successful_results.sort(key=lambda x: x['generation_time'])

        fastest = successful_results[0]
        slowest = successful_results[-1]
        avg_time = sum(r['generation_time'] for r in successful_results) / len(successful_results)

        print(f"\n🏆 最快: {fastest['model_name']} - {fastest['generation_time']:.2f}s")
        print(f"🐌 最慢: {slowest['model_name']} - {slowest['generation_time']:.2f}s")
        print(f"📊 平均: {avg_time:.2f}s")

        # 绘制简单的条形图
        print("\n速度条形图:")
        max_time = max(r['generation_time'] for r in successful_results)
        for result in successful_results:
            bar_length = int((result['generation_time'] / max_time) * 40)
            bar = '█' * bar_length
            print(f"  {result['model_name'][:25]:<25} {bar} {result['generation_time']:.2f}s")

    # 错误分析
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n" + "=" * 80)
        print("❌ 失败分析")
        print("=" * 80)
        for result in failed_results:
            print(f"\n模型: {result['model_name']}")
            print(f"错误: {result['error']}")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


def generate_markdown_report(results_data, output_file='results/report.md'):
    """生成 Markdown 格式的报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AI 模型 GPU 核函数生成能力测试报告\n\n")

        problem_name = results_data.get('problem_name', 'Unknown')
        test_config = results_data.get('test_config', {})
        results = results_data.get('results', [])

        f.write(f"## 测试信息\n\n")
        f.write(f"- **问题**: {problem_name}\n")
        f.write(f"- **Level**: {test_config.get('level')}\n")
        f.write(f"- **Problem ID**: {test_config.get('problem_id')}\n")
        f.write(f"- **Backend**: {test_config.get('backend')}\n")
        f.write(f"- **测试时间**: {results_data.get('timestamp')}\n\n")

        f.write(f"## 结果汇总\n\n")
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        f.write(f"- 测试模型数: {total}\n")
        f.write(f"- 成功生成: {successful}/{total} ({successful/total*100:.1f}%)\n\n")

        f.write(f"## 详细结果\n\n")
        f.write("| 模型名称 | 状态 | 生成时间(s) | 代码行数 | 包含核函数 |\n")
        f.write("|---------|------|-----------|---------|----------|\n")

        for result in results:
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['generation_time']:.2f}" if result['success'] else "N/A"

            if result['success']:
                metrics = analyze_code_quality(result['generated_code'])
                lines = metrics['lines']
                has_kernel = "是" if metrics['has_kernel'] else "否"
            else:
                lines = "N/A"
                has_kernel = "N/A"

            f.write(f"| {result['model_name']} | {status} | {time_str} | {lines} | {has_kernel} |\n")

        f.write(f"\n---\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"\n📝 Markdown 报告已保存到: {output_file}")


def main():
    """主函数"""
    results_data = load_latest_results()

    if not results_data:
        print("❌ 无法加载测试结果")
        return

    # 生成控制台分析报告
    compare_models(results_data)

    # 生成 Markdown 报告
    generate_markdown_report(results_data)


if __name__ == "__main__":
    main()
