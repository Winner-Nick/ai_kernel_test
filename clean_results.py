"""清理 results 目录"""
import shutil
from pathlib import Path

results_dir = Path("./results")

if results_dir.exists():
    shutil.rmtree(results_dir)
    results_dir.mkdir()
    print("✅ results 目录已清空")
else:
    results_dir.mkdir()
    print("✅ 已创建 results 目录")
