import pandas as pd

class Preprocessor:
    def __init__(self):
        pass

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        return pd.read_csv(file_path)
    
    def cramers_v(x, y):
        """Measure association between categorical x and binary y (TARGET)"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k-1, r-1))




if __name__ == "__main__":
    # df = load_and_process_data('data/bank_data_train.csv')
    # print(df.head())
    # print(df.columns.to_list())
    
    
    # with open('output_columns.txt', 'w') as f:
    #     df["APP_EDUCATION"].to_string(f)
        # for col in df.columns:
        #     f.write(f"{col}\n")
    # print(df["APP_EDUCATION"])