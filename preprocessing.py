import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self):
        self.TRESHOLD = 0.5  # 70% missing values threshold

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        return pd.read_csv(file_path)
    
    @staticmethod
    def cramers_v(x, y):
        """Measure association between categorical x and binary y (TARGET)"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        return np.sqrt(phi2 / min(k-1, r-1))
    

    def analyze_data(self, df: pd.DataFrame) -> None:
        """
        Analyze the DataFrame to determine which columns to keep or drop.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): The target column name.
        Returns:
            df (pd.DataFrame): The analyzed DataFrame.
        """
        print("\nAnalyzing columns with >70% missing values for correlation with TARGET...")
        print(f"df shape before dropping columns: {df.shape}")
        high_missing = df.columns[df.isnull().sum() / len(df) > self.TRESHOLD]
        print(f"Columns with >{self.TRESHOLD*100}% missing values: {list(high_missing)}")
        for col in high_missing:
            if col == 'TARGET':
                continue
            # coerce to numeric where possible, compute correlation with TARGET
            
            if df[col].dtype == 'object' :
                # categorical column
                corr = self.cramers_v(df[col].dropna(), df.loc[df[col].notnull(), 'TARGET'])
                if corr < 0.1:
                    print(f"Dropping column {col} due to low correlation ({corr:.3f}) with TARGET")
                    df = df.drop(columns=[col])

            else:
                # numerical column
                series_num = pd.to_numeric(df[col], errors='coerce')
                corr = series_num.corr(df['TARGET'])
                if pd.notnull(corr) and abs(corr) < 0.05:
                    print(f"Dropping column {col} due to low correlation ({corr:.3f}) with TARGET")
                    df = df.drop(columns=[col])
        print(f"df shape after dropping columns: {df.shape}")
        
        return df
    
    def normalize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize categorical columns by stripping whitespace and converting to uppercase.

        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            pd.DataFrame: DataFrame with normalized categorical columns.
        """
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in cat_cols:
            # Convert to string, lowercase, and strip whitespace
            df[col] = df[col].astype(str).str.upper()
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('NAN', np.nan)  # Replace 'NA' strings with NaN
            df[col] = df[col].replace('', np.nan)  # Replace empty strings with NaN
        
        return df
    
    def fit_encoders(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Fit encoders on the training data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
        Returns:
            dict: Dictionary of fitted encoders.
        """
        obj_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        
        # Try numeric conversion
        numeric_conversions = {}
        for col in obj_cols:
            converted = pd.to_numeric(X_train[col], errors='coerce')
            if converted.notna().mean() > 0.9:
                numeric_conversions[col] = True
        print("numeric conversion", numeric_conversions)
        # Update object columns
        obj_cols = [col for col in obj_cols if col not in numeric_conversions]
        
        # Categorize by cardinality
        low_card_cols = []
        high_card_cols = []
        
        for col in obj_cols:
            if X_train[col].isnull().all():
                continue
            
            unique_count = X_train[col].nunique(dropna=True)
            
            if unique_count < 10:
                low_card_cols.append(col)
            else:
                high_card_cols.append(col)
        
        # Fit target encoder on training data
        target_encoder = None
        if high_card_cols:
            target_encoder = ce.TargetEncoder(cols=high_card_cols, smoothing=1.0)
            valid_mask = X_train[high_card_cols].notna().all(axis=1) & y_train['TARGET'].notna()
            target_encoder.fit(X_train.loc[valid_mask, high_card_cols], y_train.loc[valid_mask, 'TARGET'])
        
        return {
            'numeric_conversions': numeric_conversions,
            'low_card_cols': low_card_cols,
            'high_card_cols': high_card_cols,
            'target_encoder': target_encoder,
            'global_mean': y_train['TARGET'].mean()
    }

    

    def transform_with_encoders(self, X: pd.DataFrame, encoders_dict: dict) -> pd.DataFrame:
        """
        Transform data using pre-fitted encoders.
        """
        X = X.copy()
        
        # Apply numeric conversions
        for col in encoders_dict['numeric_conversions']:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Drop all-null columns
        obj_cols = X.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if X[col].isnull().all():
                X = X.drop(columns=[col])
        
        # Target encode high-cardinality columns
        if encoders_dict['high_card_cols'] and encoders_dict['target_encoder']:
            high_card_cols = encoders_dict['high_card_cols']
            print(f"\n\nTarget encoding: {high_card_cols}\n\n")
            for col in high_card_cols:
                if col in X.columns:
                    # Transform using fitted encoder
                    print(f"Encoding column: {col}")
                    print()
                    try:
                        # Transform using the encoder fitted on all high-cardinality cols,
                        # then take the encoded series for the current column
                        encoded = encoders_dict['target_encoder'].transform(X[encoders_dict['high_card_cols']])[col]
                        X[col] = encoded
                        # Fill unseen / missing encodings with global mean
                        X[col].fillna(encoders_dict['global_mean'], inplace=True)
                    except Exception:
                        # If transform fails, fall back to global mean
                        X[col] = encoders_dict['global_mean']
        
        # One-hot encode low-cardinality columns
        if encoders_dict['low_card_cols']:
            print(f"\nOne-hot encoding: {encoders_dict['low_card_cols']}")

            X = pd.get_dummies(X, columns=encoders_dict['low_card_cols'], drop_first=True)
        
        return X

    def split_data(self, df: pd.DataFrame, target_col: str) -> tuple:
        """
        Split the DataFrame into training and testing sets.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): The target column name.
        Returns:
            tuple: X_train, X_test, y_train, y_test DataFrames.
        """
        X = df.drop(columns=[target_col])
        y = df[[target_col]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def fill_missing_with_mean(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """
        Fill missing values in training and testing data using statistics computed from the training set.

        Numeric columns: fill with median from training set.
        Categorical/other columns: fill with most frequent value (mode) from training set, or 'MISSING'
        if no mode can be determined.
        """
        # Work on copies to avoid unexpected side-effects
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Identify numeric columns (use numpy number types)
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        # Treat remaining columns as categorical / object-like
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        # Fill numeric columns with training medians
        if num_cols:
            medians = X_train[num_cols].median()
            X_train[num_cols] = X_train[num_cols].fillna(medians)
            # Only fill columns that exist in test
            common_num_cols = [c for c in num_cols if c in X_test.columns]
            if common_num_cols:
                X_test[common_num_cols] = X_test[common_num_cols].fillna(medians[common_num_cols])

        # Fill categorical columns with training mode (most frequent). If no mode, use 'MISSING'
        modes = {}
        for col in cat_cols:
            if col not in X_train.columns:
                continue
            if X_train[col].isnull().all():
                modes[col] = 'MISSING'
            else:
                mode_vals = X_train[col].mode(dropna=True)
                modes[col] = mode_vals.iloc[0] if not mode_vals.empty else 'MISSING'

        for col, fill_val in modes.items():
            X_train[col] = X_train[col].fillna(fill_val)
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(fill_val)

        return X_train, X_test
    
    def adjust_test_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder test columns to match training columns.

        Args:
            X_train (pd.DataFrame): Training feature DataFrame.
            X_test (pd.DataFrame): Testing feature DataFrame.
        Returns:
            pd.DataFrame: Reordered testing feature DataFrame.
        """
        print("\nAligning train/test columns...")
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)

        # Add missing columns to test (fill with 0)
        for col in train_cols - test_cols:
            X_test[col] = 0
            print(f"Added missing column to test: {col}")

        # Remove extra columns from test
        for col in test_cols - train_cols:
            X_test = X_test.drop(columns=[col])
            print(f"Removed extra column from test: {col}")
        X_test = X_test[X_train.columns]
        return X_test
        
    
    def oversample_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Apply SMOTE oversampling to balance the classes in the training data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
        Returns:
            tuple: Oversampled X and y DataFrames.
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled

def process_data(file_path: str) -> pd.DataFrame:
    ''' Process the data from the given file path
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Processed DataFrame (X_train_balanced, X_test_scaled, y_train_balanced, y_test)
    '''
    preprocessor = Preprocessor()
    
    # Load data
    df = preprocessor.load_data(file_path)
    
    # Analyze data and normalize categorical text
    df = preprocessor.analyze_data(df)
    df = preprocessor.normalize_categorical_columns(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df, target_col='TARGET')
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 1) Fill missing values using training statistics (important to do before encoding)
    X_train_filled, X_test_filled = preprocessor.fill_missing_with_mean(X_train, X_test)
    print(f"\nAfter initial fillna\nTrain nulls: {X_train_filled.isnull().sum().sum()}")
    print(f"Test nulls: {X_test_filled.isnull().sum().sum()}")
    
    # 2) Fit encoders on training and transform both train & test (encode categoricals)
    encoders = preprocessor.fit_encoders(X_train_filled, y_train)
    X_train_encoded = preprocessor.transform_with_encoders(X_train_filled, encoders)
    X_test_encoded = preprocessor.transform_with_encoders(X_test_filled, encoders)
    
    # Make sure test has same columns as train (adds missing cols filled with 0 / removes extras)
    X_test_encoded = preprocessor.adjust_test_columns(X_train_encoded, X_test_encoded)
    
    # If any residual nulls remain after encoding (should be minimal), fill them using training medians/modes
    X_train_encoded, X_test_encoded = preprocessor.fill_missing_with_mean(X_train_encoded, X_test_encoded)
    print(f"\nAfter post-encoding fillna\nTrain nulls: {X_train_encoded.isnull().sum().sum()}")
    print(f"Test nulls: {X_test_encoded.isnull().sum().sum()}")
    
    # 3) Scale (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded), columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
    
    # 4) SMOTE oversampling on training set only (after scaling)
    print("Balancing training set using SMOTE...")
    print(f"Before SMOTE: {y_train['TARGET'].value_counts()} | X shape: {X_train_scaled.shape}")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_balanced, y_train_res = smote.fit_resample(X_train_scaled, y_train['TARGET'])
    y_train_balanced = pd.DataFrame(y_train_res, columns=['TARGET'])
    print(f"After SMOTE: {y_train_balanced['TARGET'].value_counts()} | X shape: {X_train_balanced.shape}")
    
    # Ensure returned X_test_scaled has same columns order as balanced X_train (SMOTE may change index but not columns)
    X_test_scaled = X_test_scaled.reindex(columns=X_train_balanced.columns, fill_value=0)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test


def main():
    import os
    if not os.path.exists('data/bank_data_train.csv'):
        print("Data file not found. Please ensure 'data/bank_data_train.csv' exists.")
        return
    processed_df = process_data('data/bank_data_train.csv')
    print("\nData preprocessing completed.")

    print(f"processed data informations:")
    print(f"X_train shape: {processed_df[0].shape}")
    print(f"X_train sample:\n{processed_df[0].head()}")
    print(f"{"*"*40}")
    
    print(f"X_test shape: {processed_df[1].shape}")
    print(f"X_test sample:\n{processed_df[1].head()}")
    print(f"{"*"*40}")
    
    print(f"y_train shape: {processed_df[2].shape}")
    print(f"y_train sample:\n{processed_df[2].head()}")
    print(f"{"*"*40}")
    
    print(f"y_test shape: {processed_df[3].shape}")
    print(f"y_test sample:\n{processed_df[3].head()}")
    print(f"{"*"*40}")
    
    
    
if __name__ == "__main__":
    main()
    
    
    