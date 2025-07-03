import torch
import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np


class CsvDataset(data.Dataset):
    
    """
        ЗАДАНИЕ 2.2 "Эксперименты с различными датасетами" НАХОДИТСЯ В 
        lesson2/homework/homework_model_modification.py, А СОХРАНЕННЫЕ МОДЕЛИ
        В lesson2/homework/models
    
        Реализация датасета
        В конструкторе указываются:
            - путь до файла
            - названия числовых колонок
            - названия категориальных колонок
            - названия бинарных колонок
            - название таргета
    """
    
    def __init__(self, file_path: str = 'lesson2/homework/data', 
                 num_cols: list=None, cat_cols: list=None, binary_cols: list=None, target: str=None):
        self.data = pd.read_csv(file_path)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.binary_cols = binary_cols
        self.label_col = target
    
    
    # Возвращает tensor по id 
    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        features = row.drop(self.label_col).values.astype('float32')
        label = row[self.label_col]
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32) 
        return features_tensor, label_tensor
        
        
    # Возвращает длину датасета
    def __len__(self) -> int:
        return len(self.data)
    
    # Декодирует категориальные признаки алгоритмом OneHotEncoder
    def decode_categorical(self) -> None:
        if self.cat_cols is None or len(self.cat_cols) == 0:
            raise TypeError('There are no categorical features to encode')
        encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
        cat_data = self.data[self.cat_cols]
        encoded_array = encoder.fit_transform(cat_data)
        encoded_cols = encoder.get_feature_names_out(self.cat_cols)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=self.data.index)
        self.data = self.data.drop(columns=self.cat_cols)
        self.data = pd.concat([self.data, encoded_df], axis=1)
            
    # Декодирует бинарные признаки алгоритмом LabelEncoder
    def decode_binary(self) -> None:
        if self.binary_cols is None or len(self.binary_cols) == 0:
            raise TypeError('There are no binary features to encode')
        for col in self.binary_cols:
            encoder = LabelEncoder()
            self.data[col] = encoder.fit_transform(self.data[col])
        
    # Нормализует признаки алгоритмом StandardScaler
    # Возвращает tensor
    def normalize(self, target: str = None) -> None:
        try:
            scaler = StandardScaler()
            
            if target is not None:
                if target not in self.data.columns:
                    raise ValueError(f"Target column '{target}' not found in data columns")
                cols = self.data.columns.drop(target)
                scaled_array = scaler.fit_transform(self.data[cols])
                scaled_df = pd.DataFrame(scaled_array, columns=cols, index=self.data.index)
                self.data = pd.concat([scaled_df, self.data[[target]]], axis=1)
            else:
                scaled_array = scaler.fit_transform(self.data)
                self.data = pd.DataFrame(scaled_array, columns=self.data.columns, index=self.data.index)
        except Exception as e:
            e.add_note('An unexpected error occurred')
            e.add_note('Try to use CsvDataset.decode_binary() and CsvDataset.decode_categorical()')
            raise
    
    # Удаялет NaN
    def remove_nan(self):
        self.data = self.data.dropna()
    
    # Удаляет выбросы
    def remove_outlier(self) -> None:
        try:
            self.data = self._iqr(self.data)
        except Exception as e:
            e.add_note('An unexpected error occurred')
            e.add_note('Try to use CsvDataset.decode_binary() and CsvDataset.decode_categorical()')
            raise
    
    # Алгоритм межквартильного размаха с последующим удалением выбросов
    def _iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df
            
    # Возвращает датасет в виде pandas DateFrame
    def get_dataframe(self) -> pd.DataFrame:
        try:
            return self.data
        except Exception as e:
            e.add_note('An unexpected error occurred')
            raise
    
    # Возвращает датасет в виде torch Tensor
    def get_tensor(self) -> torch.Tensor:
        try:
            return torch.tensor(self.data.values, dtype=torch.float32)
        except TypeError as e:
            e.add_note('Invalid type')
            e.add_note('Try to use CsvDataset.decode_binary() and CsvDataset.decode_categorical()')
            raise


path = 'lesson2/homework/data/insurance.csv'
num = ['age', 'bmi', 'charges', 'children']
cat = ['region']
binary = ['sex', 'smoker']
targ = ['charges']
dataset = CsvDataset(file_path=path, num_cols=num, cat_cols=cat, binary_cols=binary, target=targ)
dataset.decode_categorical()
dataset.decode_binary()
dataset.remove_outlier()
t = dataset.get_dataframe()
print(t)
