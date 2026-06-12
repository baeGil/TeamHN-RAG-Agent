import json
from pathlib import Path
import sys
# Make sure stdout uses UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from eval.test_data_raw import write_raw_report

def main():
    results_path = Path("../data/vietanh-data/drag_eval/raw_benchmark_results.json").resolve()
    if not results_path.exists():
        results_path = Path("data/vietanh-data/drag_eval/raw_benchmark_results.json").resolve()
    
    print(f"Reading results from {results_path}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    
    report_workspace_path = Path("../data/vietanh-data/drag_eval/raw_benchmark_report.md").resolve()
    print(f"Writing workspace report to {report_workspace_path}")
    write_raw_report(report_workspace_path, results)
    
    report_artifact_path = Path("C:/Users/Admin/.gemini/antigravity-ide/brain/c56aa864-793d-4043-a66b-46079ae6d9a1/data_raw_benchmark_report.md").resolve()
    print(f"Writing IDE artifact report to {report_artifact_path}")
    write_raw_report(report_artifact_path, results)
    
    print("Done!")

if __name__ == "__main__":
    main()
