import sys
sys.path.insert(0, '.')

from src.commons.constants import DATA_PATH, DIR_SEP

import logging
import tarfile, glob, os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms



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


class MVTECViTDataset(Dataset):

    def __init__(self, main_path, category, transforms=None):
        self.path = os.path.join(main_path, category)
        self.transforms = transforms

        self.data, self.ground_truth, self.class_and_id, self.weak_labels = self.load_data()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"inputs": self.data[idx],
                "ground_truth": self.ground_truth[idx],
                "class_and_id": self.class_and_id[idx],
                "labels": self.weak_labels[idx]}

    def load_data(self):

        class_and_id = []

        ## Load data from /train folder
        train_data_img = []
        train_path = os.path.join(self.path, "train") 
        for dir in os.listdir(train_path):
            for name in glob.glob(os.path.join(train_path, dir) + DIR_SEP + "*.png"):
                image = read_image(name, mode=ImageReadMode.RGB)
                if self.transforms is not None:
                    image = self.transforms(image)
                train_data_img.append(image)
                class_and_id.append(name.replace(train_path + DIR_SEP, '').replace('.png', ''))

        ## Load data from /test folder
        test_data_img = []
        test_path = os.path.join(self.path, "test") 

        # test images
        for dir in os.listdir(test_path): # for all test categories (crack, poke, scratch, ...), including good
            for name in glob.glob(os.path.join(test_path, dir) + DIR_SEP + "*.png"):
                image = read_image(name, mode=ImageReadMode.RGB)
                if self.transforms is not None:
                    image = self.transforms(image)
                test_data_img.append(image)
                class_and_id.append(name.replace(test_path + DIR_SEP, '').replace('.png', ''))

        logging.debug(f"{len(class_and_id)} test images loaded")
        
        # ground truth masks
        mask_path = os.path.join(self.path, "ground_truth")
        logging.debug(f"Loading {len(class_and_id)} masks, from {mask_path} dir")

        weak_labels = []
        ground_truth = []
        data_img = train_data_img + test_data_img
        for img_class_id in class_and_id:
            mask_name = os.path.join(mask_path, img_class_id) + "_mask.png"
            logging.debug(f'Loading respective masks: {mask_name}')
            if 'good' in mask_name: 
                logging.debug(f'{img_class_id}, good')
                image = torch.zeros(test_data_img[0].shape)
                weak_labels.append(1)
            else: 
                logging.debug(f'{img_class_id}, not good')
                image = read_image(mask_name, mode=ImageReadMode.RGB)
                if self.transforms is not None:
                    for t in self.transforms.transforms:
                        if isinstance(t, transforms.Resize):
                            image = t(image)

                weak_labels.append(0)
            ground_truth.append(image)

        return data_img, ground_truth, class_and_id, weak_labels



class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        '''
        Parameters:
            root_dir: str, directory where tiny-imagenet-200 is located
            transforms: torchvision.transforms sequence of processing steps to apply at each image during loading
        '''
        self.root_dir = os.path.join(root_dir, "tiny-imagenet-200")
        self.transforms = transforms
        
        self.ids = self.get_ids()
        self.index_to_id = self.get_index_to_id()
        self.index_to_name = self.get_index_to_name()

        self.file_list, self.file_classes = self.get_file_list()

   
    def __len__(self):
        '''
        Return total number of classes
        '''
        return len(self.file_list)

    def __getitem__(self, idx):
        '''
        Randomly samples N images from the available data
        Output:
            dict - "inputs" for the data, "targets" for the class labels
        '''
        path = self.file_list[idx]
        image = read_image(path, mode=ImageReadMode.RGB)
        target = self.file_classes[idx]

        if self.transforms is not None:
            image = self.transforms(image)

        return {"inputs": image,
                "targets": target}

    def get_ids(self):
        path = os.path.join(self.root_dir, "train", "*")
        ids = [x for x in map(os.path.basename, glob.glob(path))]
        return ids

    def get_index_to_name(self):
        index_and_name = {}
        with open(os.path.join(self.root_dir, "words.txt"), "r") as f:
            lines = f.readlines()
            class_ids = [line[:9] for line in lines] # all classes listed in words.txt
            class_names = [line[10:] for line in lines]
            for index, class_id in enumerate(self.ids): # only classes in data/train
                x = class_ids.index(class_id)
                name = class_names[x]
                index_and_name[index] = name
        return index_and_name
    
    def get_index_to_id(self):
        index_and_id = {index: class_id for index, class_id in enumerate(self.ids)}
        return index_and_id

    def get_file_list(self):
        '''
        Output:
            files - list, contains the path for each image
            classes - list, contains the corresponding class index
        '''
        files = []
        classes = []
        for index, class_id in self.index_to_id.items():
            path = os.path.join(self.root_dir, "train", class_id, "images")
            names = glob.glob(path + DIR_SEP + "*.jpeg")
            files += names
            classes += [index]*len(names)
        return files, classes



# Testing for custom MVTec for weakly-supervised training
if __name__ == "__main__":
    # Define the logging level
    logging.getLogger().setLevel(logging.INFO)

    transf = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ColorJitter(brightness=0.1, hue=0.2)
    ]) 

    import matplotlib.pyplot as plt
    cat = "capsule"
    test_dataset = MVTECViTDataset(DATA_PATH, cat, transforms=transf)

    print(len(test_dataset), len(test_dataset.data), len(test_dataset.ground_truth), len(test_dataset.class_and_id))
    sample = test_dataset[219 + 3]
    img, mask, ex_name, labels = sample["test"], sample["ground_truth"], sample['class_and_id'], sample["labels"]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    ax1.imshow(torch.permute(img, (1, 2, 0)))
    ax1.set_title(f'Raw image - label {labels}')
    ax2.imshow(torch.permute(img, (1, 2, 0)))
    ax2.imshow(torch.permute(mask, (1, 2, 0)), cmap="spring", alpha=0.5, vmax=mask.max()/2)
    ax2.set_title('Ground truth mask')
    fig.suptitle(ex_name)
    plt.show()

    print(img.max(), mask.max())

    sample = test_dataset[50]
    img, mask, ex_name, labels = sample["test"], sample["ground_truth"], sample['class_and_id'], sample["labels"]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    ax1.imshow(torch.permute(img, (1, 2, 0)))
    ax1.set_title(f'Raw image - label {labels}')
    ax2.imshow(torch.permute(img, (1, 2, 0)))
    ax2.imshow(torch.permute(mask, (1, 2, 0)), cmap="spring", alpha=0.5, vmax=mask.max()/2)
    ax2.set_title('Ground truth mask')
    fig.suptitle(ex_name)
    plt.show()


# Testing for TinyImageNet
if False:

    # Define the logging level
    logging.getLogger().setLevel(logging.DEBUG)

    tiny_dataset = TinyImageNetDataset(DATA_PATH)
    
    temp = tiny_dataset.index_to_id.items()
    temp = list(temp)
    logging.debug(F"index to id {temp[:3]}")
    logging.debug(len(temp))

    temp = tiny_dataset.index_to_name.items()
    temp = list(temp)
    logging.debug(f"index to name {temp[:3]}")
    logging.debug(len(temp))

    logging.getLogger().setLevel(logging.INFO)
    import matplotlib.pyplot as plt
    sample = tiny_dataset[1000]
    img, label = sample["inputs"], sample["targets"]
    plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.title(f"{tiny_dataset.index_to_name[label]}, size: {img.shape}")
    plt.show()

# Testing for basic MVTec data
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