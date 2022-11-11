from src.commons.constants import DATA_PATH, DIR_SEP

import tarfile, glob, os
from torch import permute, zeros
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


def current_dir():
    return os.getcwd()


def extract_tar(input_path, output_path="."):
    '''
    input_path: path to .tar data
    output_path: path where data will be extracted to
    '''
    with tarfile.open(input_path, "r") as f:
        f.extractall(output_path)

class MVTECTrainDataset(Dataset):
    '''
    main_path - str: path to complete data directory
    category - str: which class you want to load, e.g "capsule"

    The training data contains only normal images.
    There are no labels.
    '''
    def __init__(self, main_path, category):
        self.path = os.path.join(main_path, category)
        
        self.train_data = self.load_data()
       
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]
        
    def load_data(self):
        train_data = []
        path = os.path.join(self.path, "train") 
        for dir in os.listdir(path):
            for name in glob.glob(os.path.join(path, dir) + DIR_SEP + "*.png"):
                image = read_image(name, mode=ImageReadMode.RGB)
                image = permute(image, (1, 2, 0)) # (width, height, channels)
                train_data.append(image)
        return train_data
                

class MVTECTestDataset(Dataset):
    '''
    main_path - str: path to complete data directory
    category - str: which class you want to load, e.g "capsule"

    The test data contains defective and normal images.
    There are labels (ground truth mask) only for the normal images.
    In order to have a label for each test image, we add zero-masks for 
    the normal images.
    '''
    def __init__(self, main_path, category):
        self.path = os.path.join(main_path, category)
        
        self.normal_count = 0
        self.normal_index = 0

        self.test_data, self.ground_truth = self.load_data()
   
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return {"test": self.test_data[idx],
                "ground_truth": self.ground_truth[idx]}
        

    def load_data(self):
        test_data = []
        path = os.path.join(self.path, "test") 

        flag = True
        for dir in os.listdir(path):
            if dir == "good" and flag:
                self.normal_index = len(test_data)    
                flag = False
            for name in glob.glob(os.path.join(current_dir(), path, dir) + DIR_SEP + "*.png"):
                if dir == "good":
                    self.normal_count += 1
                print('good path: ', name)
                image = read_image(name, mode=ImageReadMode.RGB)
                image = permute(image, (1, 2, 0)) # (width, height, channels)
                test_data.append(image)

        temp = []
        path = os.path.join(self.path, "ground_truth")

        for dir in os.listdir(path):
            for name in glob.glob(os.path.join(path, dir) + DIR_SEP + "*.png"):
                image = read_image(name, mode=ImageReadMode.RGB)
                image = permute(image, (1, 2, 0)) # (width, height, channels)
                temp.append(image)

        null_masks = [zeros(test_data[0].shape) for _ in range(self.normal_count)]
        ground_truth = temp[:self.normal_index] + null_masks + temp[self.normal_index:]

        return test_data, ground_truth


if __name__ == "__main__":
    # extract_tar(TARFILE_PATH, DATA_PATH)
    import matplotlib.pyplot as plt
    cat = "capsule"
    dataset = MVTECTestDataset(DATA_PATH, cat)

    print(len(dataset), len(dataset.test_data), len(dataset.ground_truth))
    sample = dataset[0]
    img, mask = sample["test"], sample["ground_truth"]
    plt.imshow(img)
    plt.show()
    plt.imshow(mask, cmap="spring", alpha=0.5, vmax=mask.max()/2)
    plt.show()

    print(img.max(), mask.max())

    sample = dataset[45+5]
    img, mask = sample["test"], sample["ground_truth"]
    plt.imshow(img)
    plt.show()
    plt.imshow(mask, cmap="spring", alpha=0.5, vmax=mask.max()/2)
    plt.show()

    print(dataset.normal_index, dataset.normal_count)