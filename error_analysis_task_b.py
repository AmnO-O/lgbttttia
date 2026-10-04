"""
Validation Error Analysis for Task B (Hate Speech Classification: no, yes_implicit, yes_explicit).
Partitions validation errors into 6 directional confusion buckets:
  A: no -> implicit   (False Positive Implicit)
  B: no -> explicit   (False Positive Explicit)
  C: implicit -> no   (False Negative Implicit)
  D: implicit -> explicit (Severity Overestimation)
  E: explicit -> no   (False Negative Explicit)
  F: explicit -> implicit (Severity Underestimation)
"""

import os
import argparse
import pandas as pd
import numpy as np

def run_error_analysis(
    csv_path: str = "checkpoints_task_b/task_b_val_predictions.csv",
    output_dir: str = "error_analysis_task_b",
    samples_per_group: int = 30
):
    if not os.path.exists(csv_path):
        print(f"[Error Analysis] Prediction file not found: {csv_path}")
        print("Please train the Task B model first to generate task_b_val_predictions.csv.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} validation samples from {csv_path}")

    # Standardize column names
    true_col = 'hate_speech' if 'hate_speech' in df.columns else 'true_label'
    pred_col = 'pred_hate_speech' if 'pred_hate_speech' in df.columns else 'pred_label'

    # Filter out correct predictions
    df_errors = df[df[true_col] != df[pred_col]].copy()
    total_val = len(df)
    total_errors = len(df_errors)
    acc = (total_val - total_errors) / total_val if total_val > 0 else 0

    print(f"Total Validation Samples: {total_val}")
    print(f"Total Errors: {total_errors} (Accuracy: {acc:.4f}, Error Rate: {total_errors/total_val:.2%})")

    groups = {
        'A_no_to_implicit': (df[true_col] == 'no') & (df[pred_col] == 'yes_implicit'),
        'B_no_to_explicit': (df[true_col] == 'no') & (df[pred_col] == 'yes_explicit'),
        'C_implicit_to_no': (df[true_col] == 'yes_implicit') & (df[pred_col] == 'no'),
        'D_implicit_to_explicit': (df[true_col] == 'yes_implicit') & (df[pred_col] == 'yes_explicit'),
        'E_explicit_to_no': (df[true_col] == 'yes_explicit') & (df[pred_col] == 'no'),
        'F_explicit_to_implicit': (df[true_col] == 'yes_explicit') & (df[pred_col] == 'yes_implicit'),
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_records = []

    print("\n" + "="*80)
    print("TASK B ERROR MATRIX BREAKDOWN (6 DIRECTIONAL ERROR BUCKETS)")
    print("="*80)

    for grp_name, mask in groups.items():
        sub_df = df[mask].copy()
        count = len(sub_df)
        pct_of_errors = (count / total_errors * 100) if total_errors > 0 else 0
        summary_records.append({
            'Group': grp_name,
            'Count': count,
            'Pct_of_Total_Errors': f"{pct_of_errors:.1f}%"
        })
        print(f"  [{grp_name}] Count: {count:4d} ({pct_of_errors:5.1f}% of all errors)")

        # Save individual group CSV
        grp_file = os.path.join(output_dir, f"{grp_name}.csv")
        sub_df.to_csv(grp_file, index=False)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_dir, "error_summary.csv"), index=False)

    # Detailed Inspection of Key Groups
    target_groups = ['A_no_to_implicit', 'C_implicit_to_no', 'D_implicit_to_explicit', 'F_explicit_to_implicit']
    print("\n" + "="*80)
    print(f"INSPECTING UP TO {samples_per_group} SAMPLES FOR KEY GROUPS (A, C, D, F)")
    print("="*80)

    for grp_name in target_groups:
        sub_df = df[groups[grp_name]]
        sample_subset = sub_df.head(samples_per_group)
        print(f"\n>>> GROUP {grp_name} (Total instances: {len(sub_df)}, Showing: {len(sample_subset)})")
        print("-" * 80)
        for i, (_, row) in enumerate(sample_subset.iterrows(), 1):
            comment = row.get('yt_comment', '')
            title = row.get('yt_title', '')
            desc = str(row.get('yt_description', ''))[:120]
            p_no = row.get('prob_no', np.nan)
            p_imp = row.get('prob_implicit', np.nan)
            p_exp = row.get('prob_explicit', np.nan)
            print(f"[{i:02d}] ID: {row.get('StereoQueerEval_id', 'N/A')} | Lang: {row.get('lang', 'N/A')}")
            print(f"     Title:   {title}")
            print(f"     Desc:    {desc}...")
            print(f"     Comment: {comment}")
            print(f"     Probs:   [no: {p_no:.3f}, implicit: {p_imp:.3f}, explicit: {p_exp:.3f}]")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="checkpoints_task_b/task_b_val_predictions.csv")
    parser.add_argument("--output_dir", type=str, default="error_analysis_task_b")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    run_error_analysis(csv_path=args.csv_path, output_dir=args.output_dir, samples_per_group=args.samples)
