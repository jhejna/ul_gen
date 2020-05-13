from PIL import Image
import torch
import ul_gen
from torchvision.transforms.functional import pad, resize, to_tensor, normalize, rotate
from torchvision.transforms import Compose, Grayscale, ToTensor, ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import torchvision
import random
import os

class MnistAug(object):

    def __init__(self, output_size=None, resize=None, rotate=None):
        self.output_size = output_size
        self.resize = resize
        self.rotate = rotate
    
    def aug_img(self, img):
        if not self.rotate is None:
            if len(self.rotate) == 2:
                angle = random.uniform(*self.rotate)
            elif len(self.rotate) == 4:
                total_length = self.rotate[1] - self.rotate[0] + self.rotate[3] - self.rotate[2]
                num = random.uniform(0, total_length)
                if num >= self.rotate[1] - self.rotate[0]:
                    angle = self.rotate[2] + num - self.rotate[1] + self.rotate[0]
                else:
                    angle = self.rotate[0] + num
            img = rotate(img, angle, fill=(0,))
        if not self.resize is None:
            w, h = img.size
            # assert w == h, "Image must be square"
            rescale = int(w * random.uniform(*self.resize))
            # img = img.resize((rescale, rescale), Image.BILINEAR)
            img = resize(img, rescale)
        
        w, h = img.size
        left_pad = random.randint(0, self.output_size - w)
        top_pad = random.randint(0, self.output_size - h)
        right_pad = self.output_size - w - left_pad
        bottom_pad = self.output_size - h - top_pad
        img = pad(img, (left_pad, top_pad, right_pad, bottom_pad), fill=0)
        
        return img

    def manual_img_aug(self, img, rescale=None, rotation=None):
        if not rotation is None:
            img = rotate(img, rotation, fill=(0,))
        if not rescale is None:
            w, h = img.size
            # assert w == h, "Image must be square"
            rescale = int(w * rescale)
            img = resize(img, rescale)
        
        w, h = img.size
        left_pad = random.randint(0, self.output_size - w)
        top_pad = random.randint(0, self.output_size - h)
        right_pad = self.output_size - w - left_pad
        bottom_pad = self.output_size - h - top_pad
        img = pad(img, (left_pad, top_pad, right_pad, bottom_pad), fill=0)

        return to_tensor(img)

    def __call__(self, sample):
        aug = sample.copy()
        orig = to_tensor(self.aug_img(sample))
        aug = to_tensor(self.aug_img(aug))

        return {'orig': orig, 'aug': aug}

def get_mnist_aug(**kwargs):
    mnist_data = torchvision.datasets.MNIST('~/.pytorch/mnist', train=True, download=True, transform=MnistAug(**kwargs))
    return mnist_data


class MnistFormat(object):

    def __call__(self, sample):
        orig = to_tensor(sample)
        aug = None
        return {'orig': orig, 'aug': aug}

def get_mnist(**kwargs):
    mnist_data = torchvision.datasets.MNIST('~/.pytorch/mnist', train=True, download=True, transform=MnistFormat())
    return mnist_data

def preprocess_chairs():
    base_dir = os.path.dirname(ul_gen.__file__) + "/aug_vae/data/"
    print("BASE DIR", base_dir)
    dataset_path = os.path.join(base_dir, "rendered_chairs")
    print("dataset_path", dataset_path)
    dest_path = os.path.join(base_dir, "3d_chairs")
    print("dest_path", dest_path)
    os.makedirs(dest_path)
    chair_index = 0
    for dir_name in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, dir_name)):
            class_path = os.path.join(dataset_path, dir_name, "renders")
            os.makedirs(os.path.join(dest_path, "chair" + str(chair_index)))
            for img_file in os.listdir(class_path):
                load_path = os.path.join(class_path, img_file)
                save_path = os.path.join(dest_path, "chair" + str(chair_index), img_file)
                im = Image.open(load_path)
                im = im.crop(box=(60, 60, 540, 540))
                im = im.resize((64, 64), resample=Image.BILINEAR)
                im.save(save_path)
            print("Completed Chair", chair_index)
            chair_index += 1
    print("Completed Dataset Extraction.")

class ChairsDataset(ImageFolder):
    DATASET_PATH = os.path.dirname(ul_gen.__file__) + "/aug_vae/data/3d_chairs"

    def __init__(self, transform=None):
        super(ChairsDataset, self).__init__(self.DATASET_PATH, transform=transform, target_transform=None)
        # Now, add a list so we can easily get paired data points for each class
        self.samples_by_class = dict()
        for _, class_index in self.class_to_idx.items():
            self.samples_by_class[class_index] = list()
        for path, class_index in self.samples:
            self.samples_by_class[class_index].append(path)

    def __getitem__(self, index):
        path, target = self.samples[index]
        aug_path = random.choice(self.samples_by_class[target])
        sample = self.loader(path)
        aug_sample = self.loader(aug_path)
        if self.transform is not None:
            sample = self.transform(sample)
            aug_sample = self.transform(aug_sample)
        return {'orig' : sample, 'aug': aug_sample}, target

def get_chairs(**kwargs):
    return ChairsDataset(transform=Compose([Grayscale(), ToTensor()]), **kwargs)

class ColoredMnist(Dataset):
    def __init__(self, test=False, bias_label=True):
        DATASET_PATH = os.path.dirname(ul_gen.__file__) + "/aug_vae/data/colored_mnist/colored_mnist.npy"
        data_dic = np.load(DATASET_PATH ,encoding='latin1').item()
        if not test:
            self.image = data_dic['train_image']
            self.label = data_dic['train_label']
        else:
            print("USING TEST DATA")
            self.image = data_dic['test_image']
            self.label = data_dic['test_label']
        self.bias_label = bias_label
        self.T = ToTensor()
        self.ToPIL = ToPILImage()

    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]
        image = self.ToPIL(image)
        
        if not self.bias_label:
            return self.T(image), label.astype(np.long)
        
        # Get the color label
        label_image = image.resize((14,14), Image.NEAREST) 
        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()
        return self.T(image), label_image,  label.astype(np.long)

    def __len__(self):
        return self.image.shape[0]

def get_colored_mnist(**kwargs):
    return ColoredMnist(**kwargs)
    
def get_dataset(params):
    dataset_name = params["dataset"]
    dataset_fn = {
        "mnist" : get_mnist,
        "mnist_aug" : get_mnist_aug,
        "chairs" : get_chairs,
        "colored_mnist" : get_colored_mnist
    }[dataset_name]

    return dataset_fn(**params["dataset_args"])
    

if __name__ == "__main__":
    get_colored_mnist()
