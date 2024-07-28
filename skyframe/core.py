import numpy as np
import csv

class SkyFrame:
    def __init__(self, data=None, view=False, chunk_size=None):
        self.column_names = []
        self.data = np.empty((0, 0), dtype=object)
        self.view = view
        self.chunk_size = chunk_size

        if data is not None:
            if isinstance(data, dict):
                self.column_names = list(data.keys())
                if chunk_size:
                    self.load_dict_in_chunks(data, chunk_size)
                else:
                    self.data = np.array([list(row) for row in zip(*data.values())], dtype=object)
            else:
                raise ValueError("Data should be provided as a dictionary.")
    
    def __len__(self):
        return self.data.shape[0]
    
    @property
    def empty(self):
        return len(self) == 0
    
    def iterrows(self):
        for idx, row in enumerate(self.data):
            yield idx, {col: row[i] for i, col in enumerate(self.column_names)}

    def get_column(self, col):
        if not self.data.size:
            return np.array([], dtype=object)
        
        idx = self.column_names.index(col)
        return self.data[:, idx]

    def get_data(self):
        return {col: self.get_column(col) for col in self.column_names}

    def load_dict_in_chunks(self, data, chunk_size):
        keys = list(data.keys())
        total_rows = len(data[keys[0]])

        for start in range(0, total_rows, chunk_size):
            chunk = {key: data[key][start:start + chunk_size] for key in keys}
            if not self.column_names:
                self.column_names = list(chunk.keys())
            self.load_data_in_chunks([chunk])

    def load_data_in_chunks(self, chunk_loader):
        for chunk in chunk_loader:
            chunk_column_names = list(chunk.keys())
            
            if not set(chunk_column_names).issubset(set(self.column_names)):
                raise ValueError("Chunk contains columns not present in the original DataFrame")

            chunk_data = np.array([list(row) for row in zip(*chunk.values())], dtype=object)
            if self.data.size == 0:
                self.data = chunk_data
            else:
                self.data = np.vstack((self.data, chunk_data))

    def read_csv_in_chunks(self, file_path, chunk_size=1000, delimiter=',', na_values=None, dtype=None, fillna=None):
        chunk_loader = self.csv_chunk_loader(file_path, chunk_size, delimiter, na_values, dtype, fillna)

        for chunk in chunk_loader:
            self.load_data_in_chunks([chunk])

    def csv_chunk_loader(self, file_path, chunk_size, delimiter, na_values, dtype, fillna):
        with open(file_path, 'r') as file:
            header = file.readline().strip().split(delimiter)
            chunk = []

            for line in file:
                values = line.strip().split(delimiter)
                if na_values is not None:
                    values = [val if val not in na_values else fillna for val in values]
                if dtype is not None:
                    values = [dtype[col](val) if col in dtype else val for col, val in zip(header, values)]
                chunk.append(values)

                if len(chunk) == chunk_size:
                    yield {header[i]: [row[i] for row in chunk] for i in range(len(header))}
                    chunk = []

            if chunk:
                yield {header[i]: [row[i] for row in chunk] for i in range(len(header))}

    def add_column(self, column_name, values=None, default_value=None):
        if column_name in self.column_names:
            raise ValueError(f"Failed to Add: Column '{column_name}' already exists in the DataFrame.")
        
        if values is not None:
            if len(values) != self.data.shape[0]:
                raise ValueError(f'Length of values does not match number of rows in the DataFrame. Length of the values: {len(values)} and the length of the rows: {self.data.shape[0]}')
            new_column = np.array(values, dtype=object)
        else:
            new_column = np.full((self.data.shape[0],), default_value, dtype=object)
        
        self.column_names.append(column_name)
        self.data = np.column_stack((self.data, new_column))

    def remove_column(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]

        for column_name in column_names:
            if column_name not in self.column_names:
                raise ValueError(f"Failed to Remove: Column '{column_name}' does not exist in the DataFrame.")
        
        indices = [self.column_names.index(column_name) for column_name in column_names]
        
        self.column_names = [name for name in self.column_names if name not in column_names]
        self.data = np.delete(self.data, indices, axis=1)

    def modify_column(self, column_name, func):
        if column_name not in self.column_names:
            raise ValueError(f"Failed to Modify: Column '{column_name}' does not exist in the DataFrame.")
        
        idx = self.column_names.index(column_name)
        self.data[:, idx] = np.vectorize(func)(self.data[:, idx])

    def sort(self, column_list, ascending):
        if len(column_list) != len(ascending):
            raise ValueError(f"The Length of the columns and ascending list must be the same. Length of columns: {len(column_list)} not equals to Length of ascending: {len(ascending)}")
        
        missing_columns = [col for col in column_list if col not in self.column_names]
        if missing_columns:
            raise ValueError(f"Failed to Sort: Some columns do not exists in the DataFrame: {', '.join(missing_columns)}")
        
        column_indices = [self.column_names.index(col) for col in column_list]

        #idx = self.column_names.index(column_name)
        sort_order = np.array(ascending)
        sort_order = np.where(sort_order, 1, -1)

        sort_keys = tuple(self.data[:, idx] for idx in reversed(column_indices))

        sorted_indices = np.lexsort(sort_keys)
        self.data = self.data[sorted_indices]

        for _, order in zip(reversed(column_indices), sort_order):
            if order == -1:
                self.data = np.flip(self.data, axis=0)
        
        return self

    def groupby(self, column_list, first=True):
        missing_columns = [col for col in column_list if col not in self.column_names]
        if missing_columns:
            raise ValueError(f"Failed to Group By: Some columns do not exists in the DataFrame: {', '.join(missing_columns)}")
        
        column_indices = [self.column_names.index(col) for col in column_list]

        #idx = self.column_names.index(column_name)
        grouped = {}
        for row in self.data:
            key = tuple(row[idx] for idx in column_indices)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(row)
        
        grouped_data = []
        for key, rows in grouped.items():
            if first:
                grouped_data.append(rows[0])
            else:
                grouped_data.append(rows[-1])
        
        self.data = np.array(grouped_data, dtype=object)
        return self
        #unique_values, indices = np.unique(self.data[:, idx], return_inverse=True)
        #grouped_data = {val: self.data[indices == i] for i, val in enumerate(unique_values)}
        
        #return grouped_data
    
    def slice(self, start=None, end=None, keep=True, view=False):
        if start is None:
            start = 0
        
        if end is None:
            end = self.data.shape[0]
        
        if keep is not None:
            col_indices = [self.column_names.index(col) for col in keep]
        else:
            col_indices = list(range(self.data.shape[1]))
        
        sliced_data = self.data[start:end, col_indices]
        new_column_names = [self.column_names[i] for i in col_indices]

        if view:
            return SkyFrameView(new_column_names, sliced_data)
        else:
            return SkyFrame({col: sliced_data[:, i].tolist() for i, col in enumerate(new_column_names)})
    
    def slice_batch(self, batch_size, keep=None, view=False):
        for start in range(0, len(self), batch_size):
            yield self.slice(start, start + batch_size, keep=keep, view=view)
    
    def rename_column(self, old_name, new_name, contains=None):
        if contains:
            self.column_names = [name.replace(old_name, new_name) if contains in name else name for name in self.column_names]
        else:
            if old_name not in self.column_names:
                raise ValueError(f"Failed to Rename: Column '{old_name}' does not exists in the DataFrame.")
            self.column_names[self.column_names.index(old_name)] = new_name
    
    def merge(self, other, on, how='inner', suffix=('_df1', '_df2'), indicator=False):
        if isinstance(on, str):
            on = [on]
        
        # Ensure the columns exist in both dataframes
        missing_columns_self = [col for col in on if col not in self.column_names]
        missing_columns_other = [col for col in on if col not in other.column_names]

        if missing_columns_self or missing_columns_other:
            raise ValueError(f"Failed to Merge: Some columns do not exist in the DataFrame: {', '.join(missing_columns_self + missing_columns_other)}")

        # Get indices of columns to join on
        join_indices_self = [self.column_names.index(col) for col in on]
        join_indices_other = [other.column_names.index(col) for col in on]

        join_dict = {}
        for row in other.data:
            key = tuple(row[idx] for idx in join_indices_other)
            if key not in join_dict:
                join_dict[key] = []
            join_dict[key].append(row)
        
        # List to hold the merged rows
        merged_data = []
        merge_column = []

        for row in self.data:
            key = tuple(row[idx] for idx in join_indices_self)

            if key in join_dict and how == 'inner':
                for matching_row in join_dict[key]:
                    merged_data.append(np.concatenate([row, [matching_row[idx] for idx, col in enumerate(other.column_names) if col not in on]]))
                    merge_column.append("both")
            elif key not in join_dict and how == 'left': #key not in join_dict and how in ('left', 'outer'):
                null_row = [None] * (len(other.column_names) - len(on))
                merged_data.append(np.concatenate([row, null_row]))
                merge_column.append("left_only")
        
        # Create new column names
        new_column_names = []
        for col in self.column_names:
            if col in on or col not in other.column_names:
                new_column_names.append(col)
            else:
                new_column_names.append(col + suffix[0])

        for col in other.column_names:
            if col not in on and col not in self.column_names:
                new_column_names.append(col)
            elif col not in on:
                new_column_names.append(col + suffix[1])
        
        # Add the merge indicator column if requested
        if indicator:
            if isinstance(indicator, bool):
                indicator = '_merge'
            merged_data = [row.tolist() + [merge_col] for row, merge_col in zip(merged_data, merge_column)]
            new_column_names.append(indicator)

        # Return new merged DataFrame
        return SkyFrame({col: [row[idx] for row in merged_data] for idx, col in enumerate(new_column_names)})

    @classmethod
    def read_csv(cls, file, delimiter=',', na_values=None, dtype=None, fillna=None, chunk_size=None):
        data = []
        column_names = []

        if isinstance(file, str):
            file = open(file, 'r')
        elif not hasattr(file, 'read'):
            raise ValueError("The file parameter should be a file path or file-like object.")
        
        with file:
            reader = csv.reader(file, delimiter=delimiter)
            column_names = next(reader)

            for row in reader:
                if na_values:
                    row = [fillna if cell in na_values else cell for cell in row]
                data.append(row)
        
        data = np.array(data, dtype=object)

        if dtype:
            if isinstance(dtype, dict):
                for col_name, col_type in dtype.items():
                    if col_name in column_names:
                        col_idx = column_names.index(col_name)

                        try:
                            data[:, col_idx] = data[:, col_idx].astype(col_type)
                        except ValueError:
                            raise ValueError(f"Cannot convert column '{col_name}' to type {col_type}")
            else:
                try:
                    data = data.astype(dtype)
                except ValueError:
                    raise ValueError(f"Cannot convert data to type {dtype}")
        
        return cls({col: data[:, i] for i, col in enumerate(column_names)}, chunk_size=chunk_size)

class SkyFrameView(SkyFrame):
    def __init__(self, column_names, data ):
        self.column_names = column_names
        self.data = data
        self.view = True
