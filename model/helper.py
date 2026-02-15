#helper.py
#imports
try:
    from datas.load_data import DataLoader
except ImportError:
    from .datas.load_data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class ClassifierHelper:
    def __init__(self):
        self.data = None

    def load_data(self, file_path=None):
        loader = DataLoader()
        if file_path is not None:
            loader.get_file_path(file_path)
        self.data = loader.load_data()
        return self.data
    
    def get_data(self):
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_info(self):
        if self.data is not None:
            data_info = {
                "shape": self.data.shape,
                "columns": self.data.columns.tolist(),
                "missing_values": self.data.isnull().sum().to_dict(),
                "data_types": self.data.dtypes.to_dict(),
                "data_info": self.data.info()
            }
            return data_info
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_unique_values(self, column_name):
        if self.data is not None:
            if column_name in self.data.columns:
                return self.data[column_name].unique()
            else:
                raise ValueError(f"Column '{column_name}' not found in data.")
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def data_cleaning(self):
        self.data["deposit"] = self.data["deposit"].map({"yes": 1, "no": 0})
        self.data["previous_contact"] = self.data["pdays"].apply(lambda x: 0 if x == -1 else 1)
        self.data["pdays"] = self.data["pdays"].replace(-1, 0)
    
    def split_data(self, target_column = "deposit"):
        if self.data is not None:
            if target_column in self.data.columns:
                X = self.data.drop(columns=[target_column], axis=1)
                y = self.data[target_column]
                return X, y
            else:
                raise ValueError(f"Column '{target_column}' not found in data.")
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")

    def get_shape_of_X_y(self, X, y):
        shape_info = {
            "X_shape": X.shape,
            "X_columns": X.columns.tolist(),
            "y_shape": y.shape
        }
        return shape_info
    
    # Identifying categorical and numerical columns
    def get_categorical_and_numerical_columns(self):
        if self.data is not None:
            categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            return categorical_columns, numerical_columns
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_categorical_column_info(self, column_name):
        if self.data is not None:
            if column_name in self.data.columns:
                value_counts = self.data[column_name].value_counts().to_dict()
                return value_counts
            else:
                raise ValueError(f"Column '{column_name}' not found in data.")
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_numerical_column_info(self, column_name):
        if self.data is not None:
            if column_name in self.data.columns:
                column_info = {
                    "mean": self.data[column_name].mean(),
                    "median": self.data[column_name].median(),
                    "std": self.data[column_name].std(),
                    "min": self.data[column_name].min(),
                    "max": self.data[column_name].max()
                }
                return column_info
            else:
                raise ValueError(f"Column '{column_name}' not found in data.")
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_columns_info(self):
        if self.data is not None:
            columns_info = {}
            for column in self.data.columns:
                if self.data[column].dtype == 'object':
                    columns_info[column] = {
                        "type": "categorical",
                        "unique_values": self.get_categorical_column_info(column)
                    }
                else:
                    columns_info[column] = {
                        "type": "numerical",
                        "statistics": self.get_numerical_column_info(column)
                    }
            return columns_info
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def preprocessor_sparse_dense(self, categorical_cols, numerical_cols):
        
        preprocessor_sparse = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )

        # For Naive Bayes, we can use sparse output for OneHotEncoder to save memory
        preprocessor_dense = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ]
        )
        return preprocessor_sparse, preprocessor_dense
    
    def split_train_test(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def get_train_test_info(self, X_train, X_test, y_train, y_test):
        train_test_info = {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "y_train_distribution": y_train.value_counts().to_dict(),
            "y_test_distribution": y_test.value_counts().to_dict()
        }
        return train_test_info
    
    
if __name__ == "__main__":
    helper = ClassifierHelper()
    data = helper.load_data()
    print(data.head())
