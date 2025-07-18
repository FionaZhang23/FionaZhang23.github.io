import os
import sys
import pandas as pd

def check_quality(input_dir, output_dir="/deac/csc/classes/csc373/zhanx223/assignment_2/output"):
    report = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)
            
            num_duplicated_rows = df.duplicated().sum()
            num_rows_with_missing = df.isnull().any(axis=1).sum()
            num_cols_with_missing = df.isnull().any().sum()
            
            col_info = {}
            for col in df.columns:
                col_data = df[col]
                col_info[col] = {
                    "Data Type": str(col_data.dtype),
                    "Min Value": col_data.min() if pd.api.types.is_numeric_dtype(col_data) else None,
                    "Max Value": col_data.max() if pd.api.types.is_numeric_dtype(col_data) else None,
                    "Unique Values": col_data.nunique(),
                }

            report.append({
                "File Name": filename,
                "Duplicated Rows": num_duplicated_rows,
                "Rows with Missing Values": num_rows_with_missing,
                "Columns with Missing Values": num_cols_with_missing,
                "Column Info": col_info
            })
    report_df = pd.DataFrame(report)
    report_path = os.path.join(output_dir, "data_quality_report.csv")
    report_df.to_csv(report_path, index=False)

def detect_data_leakage(train_path, threshold=0.9):
    train_data = pd.read_csv(train_path)
    numeric_cols = train_data.select_dtypes(include=['number'])
    target_col = numeric_cols.columns[-1]
    correlation = numeric_cols.corr()[target_col].abs().sort_values(ascending=False)
    high_corr_features = correlation[correlation > threshold].drop(target_col)
    result = {feature: {'correlation': round(corr, 3), 'index': train_data.columns.get_loc(feature)} 
              for feature, corr in high_corr_features.items()}
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_quality.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    check_quality(input_dir)

    train_csv = "/deac/csc/classes/csc373/data/housing/train.csv"

    print("Data Leakage Report:", detect_data_leakage(train_csv))