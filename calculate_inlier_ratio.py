import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate mean inlier ratio per model from benchmark CSV.")
    parser.add_argument("--input", required=True, help="Path to the per_sample_results.csv file")
    parser.add_argument("--output", default="mean_inlier_ratios.csv", help="Path to save the summary results")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    # Load the benchmark results
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Calculate inlier ratio for each successful sample
    # We define ratio as 0 if no matches were found to avoid NaN
    df['inlier_ratio'] = 0.0
    mask = (df['n_raw_matches'] > 0) & (df['success'] == True)
    df.loc[mask, 'inlier_ratio'] = df.loc[mask, 'n_inliers'] / df.loc[mask, 'n_raw_matches']

    # Group by model and calculate the mean ratio
    # We only aggregate samples where the pipeline successfully finished
    summary = df[df['success'] == True].groupby('model')['inlier_ratio'].mean().reset_index()
    summary.columns = ['model', 'mean_inlier_ratio']

    # Sort for consistent display order
    model_order = {'similarity': 0, 'affine': 1, 'projective': 2}
    summary['sort_order'] = summary['model'].str.lower().map(model_order)
    summary = summary.sort_values('sort_order').drop('sort_order', axis=1)

    # Save to disk
    summary.to_csv(args.output, index=False)
    
    # Print result to console
    print("\n" + "="*40)
    print("MATCHING ROBUSTNESS (Mean Inlier Ratio)")
    print("="*40)
    print(summary.to_string(index=False))
    print("="*40)
    print(f"Full summary saved to: {args.output}")

if __name__ == "__main__":
    main()
