import cv2
from torch.utils.data import DataLoader, Dataset

class Iterator(Dataset):
    new_keys = {'img_path': 'img', 'sem_path': 'sem', 'inst_path': 'inst'}

    def __init__(self, data_dict, data_path, batch_size=32, shuffle=False, num_workers=1):
        self.data_dict = data_dict
        self.data_path = data_path
        self.old_keys = list(data_dict.keys())
        self.keys = list(self.new_keys.values())
        self.iter = DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __getitem__(self, index):
        return {self.new_keys[k]: 
                cv2.imread(self.data_path + self.data_dict[k][index])
                .transpose((2, 0, 1))
                for k in self.old_keys}

    def __len__(self):
        return self.data_dict["img_path"].shape[0]
