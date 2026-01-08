import pandas as pd


def load_and_process_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df



if __name__ == "__main__":
    df = load_and_process_data('data/bank_data_train.csv')
    print(df.head())
    print(df.columns.to_list())
    
    
    with open('output_columns.txt', 'w') as f:
        df["APP_EDUCATION"].to_string(f)
        # for col in df.columns:
        #     f.write(f"{col}\n")
    # print(df["APP_EDUCATION"])