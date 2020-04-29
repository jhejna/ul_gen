from PIL import Image
import ul_gen
from torchvision.transforms.functional import pad, resize, to_tensor, normalize, rotate
from torchvision.transforms import Compose, Grayscale, ToTensor
from torchvision.datasets import ImageFolder
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
            angle = random.uniform(*self.rotate)
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

    def __call__(self, sample):
        aug = sample.copy()
        orig = 2*to_tensor(self.aug_img(sample)) - 1
        aug = 2*to_tensor(self.aug_img(aug)) - 1

        return {'orig': orig, 'aug': aug}

def get_mnist(**kwargs):
    mnist_data = torchvision.datasets.MNIST('~/.pytorch/mnist', train=True, download=True, transform=MnistAug(**kwargs))
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
                im = im.crop(box=(160, 160, 460, 460))
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

def get_dataset(params):
    dataset_name = params["dataset"]
    dataset_fn = {
        "mnist" : get_mnist,
        "chairs" : get_chairs,
    }[dataset_name]

    return dataset_fn(**params["dataset_args"])
    