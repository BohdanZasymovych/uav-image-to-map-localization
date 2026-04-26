import argparse
import pandas as pd
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate RMSE per model from benchmark CSV.")
    parser.add_argument("--input", required=True, help="Path to the per_sample_results.csv file")
    parser.add_argument("--output", default="rmse_results.csv", help="Path to save the summary results")
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

    # Filter only successful runs with valid pixel error data
    # Success in this context means technical success (pipeline finished)
    success_df = df[df['success'] == True].copy()
    success_df['pixel_error_px'] = pd.to_numeric(success_df['pixel_error_px'], errors='coerce')
    success_df = success_df.dropna(subset=['pixel_error_px'])

    if success_df.empty:
        print("Error: No successful localization samples found to calculate RMSE.")
        return

    # Calculate RMSE: sqrt(mean(errors^2))
    def calculate_model_rmse(group):
        mse = np.mean(np.square(group))
        return np.sqrt(mse)

    summary = success_df.groupby('model')['pixel_error_px'].apply(calculate_model_rmse).reset_index()
    summary.columns = ['model', 'rmse_px']

    # Sort for consistent display order
    model_order = {'similarity': 0, 'affine': 1, 'projective': 2}
    summary['sort_order'] = summary['model'].str.lower().map(model_order)
    summary = summary.sort_values('sort_order').drop('sort_order', axis=1)

    # Save to disk
    summary.to_csv(args.output, index=False)
    
    # Print result to console
    print("\n" + "="*40)
    print("ACCURACY BENCHMARK (Root Mean Square Error)")
    print("="*40)
    print(summary.to_string(index=False))
    print("="*40)
    print(f"Full RMSE summary saved to: {args.output}")

if __name__ == "__main__":
    main()
