from src.commons.constants import DATA_PATH, DIR_SEP

import logging
import tarfile, glob, os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor


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

        self.test_data, self.ground_truth, self.class_and_id = self.load_data()
   
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return {"test": self.test_data[idx],
                "ground_truth": self.ground_truth[idx],
                "class_and_id": self.class_and_id[idx]}
        
    def load_data(self):
        test_data_img = []
        test_path = os.path.join(self.path, "test") 
        class_and_id = []

        # test images
        for dir in os.listdir(test_path): # for all test categories (crack, poke, scratch, ...), including good
            for name in glob.glob(os.path.join(test_path, dir) + DIR_SEP + "*.png"):
                image = read_image(name, mode=ImageReadMode.RGB)
                test_data_img.append(image)
                class_and_id.append(name.replace(test_path + DIR_SEP, '').replace('.png', ''))

        logging.debug(f"{len(class_and_id)} test images loaded")
        
        # ground truth masks
        mask_path = os.path.join(self.path, "ground_truth")
        logging.debug(f"Loading {len(class_and_id)} masks, from {mask_path} dir")

        ground_truth = []
        for img_class_id in class_and_id:
            mask_name = os.path.join(mask_path, img_class_id) + "_mask.png"
            logging.debug(f'Loading respective masks: {mask_name}')
            if 'good' in mask_name: 
                logging.debug(f'{img_class_id}, good')
                image = torch.zeros(test_data_img[0].shape)
            else: 
                logging.debug(f'{img_class_id}, not good')
                image = read_image(mask_name, mode=ImageReadMode.RGB)
            ground_truth.append(image)

        return test_data_img, ground_truth, class_and_id

class TinyImageNetDataset(Dataset):
    def __init__(self, main_path):
        self.path = os.path.join(main_path, "tiny-imagenet-200")
        self.class_id_and_index = self.id2index()
        self.index_and_name = self.index2name()
        self.inputs, self.targets = self.load_data()
   

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {"inputs": self.inputs[idx],
                "targets": self.targets[idx]}

    def index2name(self):
        with open(os.path.join(self.path, "words.txt"), "r") as f:
            lines = f.readlines()
            lines = [(index, line[10:]) for index, line in enumerate(lines)]
        index_and_name = {index: name for index, name in lines}
        return index_and_name
    
    def id2index(self):
        with open(os.path.join(self.path, "words.txt"), "r") as f:
            lines = f.readlines()                
            lines = [(line[:9], index) for index, line in enumerate(lines)]
        class_id_and_index = {class_id: index for class_id, index in lines}
        return class_id_and_index

    def load_data(self):
        inputs = []
        labels = []
        path = os.path.join(self.path, "train") 
        for dir in os.listdir(path):
            for name in glob.glob(os.path.join(path, dir) + DIR_SEP + "images" + DIR_SEP + "*.jpeg"):
                image = read_image(name, mode=ImageReadMode.RGB)
                inputs.append(image/255) # Convert to 0-1 scale
                labels.append(self.class_id_and_index[dir])
            # logging.debug(f"Finish loading {dir}")
        return torch.stack(inputs), torch.tensor(labels)

# Testing for TinyImageNet
if __name__ == "__main__":

    # Define the logging level
    logging.getLogger().setLevel(logging.DEBUG)

    tiny_dataset = TinyImageNetDataset(DATA_PATH)

    logging.debug(len(tiny_dataset))
    # logging.debug(tiny_dataset.class_id_and_index)
    # logging.debug(tiny_dataset.index_and_name)
    logging.debug(tiny_dataset[0])

    import matplotlib.pyplot as plt
    sample = tiny_dataset[7624]
    img, label = sample["inputs"], sample["targets"]
    plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.title(f"{label}, size: {img.shape}")
    plt.show()

# Testing for other datasaets
if False:
    # extract_tar(TARFILE_PATH, DATA_PATH)
    import matplotlib.pyplot as plt
    cat = "capsule"
    test_dataset = MVTECTestDataset(DATA_PATH, cat)

    print(len(test_dataset), len(test_dataset.test_data), len(test_dataset.ground_truth), len(test_dataset.class_and_id))
    sample = test_dataset[0]
    img, mask, ex_name = sample["test"], sample["ground_truth"], sample['class_and_id']
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    ax1.imshow(torch.permute(img, (1, 2, 0)))
    ax1.set_title('Raw image')
    ax2.imshow(torch.permute(img, (1, 2, 0)))
    ax2.imshow(torch.permute(mask, (1, 2, 0)), cmap="spring", alpha=0.5, vmax=mask.max()/2)
    ax2.set_title('Ground truth mask')
    fig.suptitle(ex_name)
    plt.show()

    print(img.max(), mask.max())

    sample = test_dataset[45+5]
    img, mask, ex_name = sample["test"], sample["ground_truth"], sample['class_and_id']
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    ax1.imshow(torch.permute(img, (1, 2, 0)))
    ax1.set_title('Raw image')
    ax2.imshow(torch.permute(img, (1, 2, 0)))
    ax2.imshow(torch.permute(mask, (1, 2, 0)), cmap="spring", alpha=0.5, vmax=mask.max()/2)
    ax2.set_title('Ground truth mask')
    fig.suptitle(ex_name)
    plt.show()