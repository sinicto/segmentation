import cv2
from torch.utils.data import DataLoader, Dataset

class Iterator:
    def __init__(self, data_dict, data_path, batch_size=32, shuffle=False, num_workers=1):
        self.data_dict = data_dict
        self.data_path = data_path
        self.keys = list(data_dict.keys())

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()
        return {k: cv2.imread(self.data_path + self.data_dict[k][index]) for k in self.keys}

    def __len__(self):
        return self.data_dict["img_path"].shape[0]
