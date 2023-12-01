import pickle
import numpy as np

class GermanDataLoader:
    def __init__(self, train_file='German_Dataset/train.p', 
                 valid_file='German_Dataset/valid.p', 
                 test_file='German_Dataset/test.p'):
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file

    def load_data(self):
        train_data = self._load_file(self.train_file)
        valid_data = self._load_file(self.valid_file)
        test_data = self._load_file(self.test_file)

        self._check_data_structure(train_data, 'train')
        self._check_data_structure(valid_data, 'valid')
        self._check_data_structure(test_data, 'test')

        return (train_data['features'], train_data['labels']), \
               (valid_data['features'], valid_data['labels']), \
               (test_data['features'], test_data['labels'])

    def _load_file(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def _check_data_structure(self, data, dataset_name):
        required_keys = ['coords', 'labels', 'features', 'sizes']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid data structure in {dataset_name} dataset. '{key}' key is required.")

# Example usage
# data_loader = GermanDataLoader()
# (train_features, train_labels), (valid_features, valid_labels), (test_features, test_labels) = data_loader.load_data()

# print('Train features shape:', train_features.shape)
# print('Train labels shape:', train_labels.shape)
# print('Valid features shape:', valid_features.shape)
# print('Valid labels shape:', valid_labels.shape)
# print('Test features shape:', test_features.shape)
# print('Test labels shape:', test_labels.shape)
