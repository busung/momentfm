import numpy as np
from sklearn.preprocessing import StandardScaler

from momentfm.utils.data import load_from_tsfile

from tslearn.preprocessing import TimeSeriesResampler




class ClassificationDataset:
    def __init__(self, data_split="train",train_file_path = None,test_file_path = None):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

        self.seq_len = 512
        self.train_file_path_and_name = f"{self.train_file_path}.ts"
        self.test_file_path_and_name = f"{self.test_file_path}.ts"
        self.data_split = data_split  # 'train' or 'test'
        self.cutting = 0

        # Read data
        self._read_data()

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        labels = np.unique(train_labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        self.train_data, self.train_labels = load_from_tsfile(
            self.train_file_path_and_name
        )
        self.test_data, self.test_labels = load_from_tsfile(
            self.test_file_path_and_name
        )

        self.train_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.test_labels
        )

        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        self.num_timeseries = self.data.shape[0]
        self.num_channel = self.data.shape[1]
        self.len_timeseries = self.data.shape[2]
        
        if self.data_split =="train":
            if self.len_timeseries > self.seq_len:
                resampler = TimeSeriesResampler(sz = self.seq_len)
                self.data = np.transpose(self.data,(0,2,1))
                self.data = resampler.fit_transform(self.data)
                self.data = np.transpose(self.data,(0,2,1))
                self.data = self.data.reshape(-1, self.seq_len)
                self.scaler.fit(self.data)
                #self.data = self.scaler.transform(self.data)
                self.data = self.data.reshape(self.num_timeseries*self.num_channel, self.seq_len)
            else:
                self.data = self.data.reshape(-1, self.len_timeseries)
                self.scaler.fit(self.data)
                #self.data = self.scaler.transform(self.data)
                self.data = self.data.reshape(self.num_timeseries*self.num_channel, self.len_timeseries)
                
        elif self.data_split == "test":
            if self.len_timeseries > self.seq_len:
                new_rows = []
                self.data = self.data.reshape(-1, self.len_timeseries)
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)
                self.data = self.data.reshape(self.num_timeseries*self.num_channel, self.len_timeseries)

                for row in self.data:
                    # 각 row를 split_size만큼 잘라서 추가
                    for i in range(0, len(row), self.seq_len):
                        temp_row = row[i:i + self.seq_len]

                        if len(temp_row) < self.seq_len:
                            self.cutting = len(temp_row)
                            temp_row = np.pad(temp_row,(self.seq_len - len(temp_row),0))
                        new_rows.append(temp_row)

                self.data = np.array(new_rows)
 
            else:
                self.data = self.data.reshape(-1, self.len_timeseries)
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)
                self.data = self.data.reshape(self.num_timeseries*self.num_channel, self.len_timeseries)

        self.data = self.data.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index,].astype(int)
        input_mask = np.ones(self.seq_len)
        
        if self.cutting > 0:
            if sum(timeseries[:-self.cutting]) == 0:
                input_mask[:-self.cutting] = 0
        else:
            input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))

        return np.expand_dims(timeseries, axis=0), input_mask, labels
