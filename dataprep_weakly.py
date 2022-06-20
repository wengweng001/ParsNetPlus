import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
from PIL import Image
import sys
import os
from scipy import io


def get_labeled_index(labeled_rate):
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    idx = np.arange(2952)    # the dataset has 2952 data
    np.random.seed(6)
    np.random.shuffle(idx)
    train_labeled_idxs.extend(idx[:int(2952*labeled_rate)])
    train_unlabeled_idxs.extend(idx[int(2952*labeled_rate):])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    return train_labeled_idxs, train_unlabeled_idxs

class customDatasetFolder(torchvision.datasets.vision.VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, labeled = True, indexs = None, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(customDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, labeled, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if labeled:
            if indexs is not None:
                self.samples = [self.samples[i] for i in indexs]
                self.targets = [self.targets[i] for i in indexs]
        else:
            if indexs is not None:
                self.classes = ['-1']
                self.samples = [self.samples[i] for i in indexs]
                self.targets = np.array([-1 for s in samples]).tolist()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes = ['0', '1', '2']
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        class_to_idx = {'0_2': 2, '1_0': 0, '2_1': 1, '3_2': 2, '4_1': 1, '5_0': 0, '6_1': 1, '7_2': 2, '8_1': 1, '9_0': 0}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

class data_unlabeled(customDatasetFolder):

    def __init__(self, root, indexs,
                 extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(data_unlabeled, self).__init__(root, indexs,
                 transform=transform, target_transform=target_transform)
        self.targets = np.array([-1 for i in range(len(self.targets))])



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customImageFolder(customDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(customImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

def make_dataset(dir, class_to_idx, labeled, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    if not labeled:
                        item = (path,-1)
                    images.append(item)

    return images

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_image(labeled_proportion, batchsize):
    # load image data

    if os.path.isfile('./label_idx_rate{}.npy'.format(labeled_proportion)) and os.path.isfile('./unlabel_idx_rate{}.npy'.format(labeled_proportion)):
        print("loading idx")
        train_labeled_idxs = np.load('./label_idx_rate{}.npy'.format(labeled_proportion))
        train_unlabeled_idxs = np.load('./unlabel_idx_rate{}.npy'.format(labeled_proportion))
    else:
        train_labeled_idxs, train_unlabeled_idxs = get_labeled_index(labeled_rate=labeled_proportion)
        np.save('./label_idx_rate{}.npy'.format(labeled_proportion),train_labeled_idxs)
        np.save('./unlabel_idx_rate{}.npy'.format(labeled_proportion),train_unlabeled_idxs)

    data_path = './A_AllData/'
    train_labeled_dataset = customDatasetFolder(root=data_path, indexs = train_labeled_idxs,
                                                loader=default_loader, extensions='.jpg',
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                                                           (0.229, 0.224, 0.225))]))

    train_unlabeled_dataset = customDatasetFolder(root=data_path, labeled = False, indexs = train_unlabeled_idxs,
                                                loader=default_loader, extensions='.jpg',
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                                                           (0.229, 0.224, 0.225))]))

    labeled_dataloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        train_unlabeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    print(f"#Labeled image: {len(train_labeled_idxs)} #Unlabeled image: {len(train_unlabeled_idxs)}")

    return labeled_dataloader,unlabeled_dataloader

def sensor():
    # load sensor data
    data1 = io.loadmat(r'injordexp2952.mat')
    data  = data1.get('data')
    data  = torch.from_numpy(data)
    data  = data.float()
    preq_data = data[:,0:-1]

    # indexs1, indexs2 = get_labeled_index(labeled_proportion)
    # labeled = preq_data[indexs1]
    # unlabeled = preq_data[indexs2]
    preq_label        = data[:,-1]
    # targets_label = preq_label[indexs1]
    # targets_unlabel = preq_label[indexs2]

    # preq_label        = preq_label.long()
    # batchSize         = batchSize
    # nData             = preq_data.shape[0]
    # nBatch            = int(nData/batchSize)
    nInputSensor      = preq_data.shape[1]
    return preq_data,preq_label,nInputSensor

class sensordataset(Dataset):
    def __init__(self,labeled = True, indexs = None):
        samples,targets,_ = sensor()
        self.samples = samples
        self.targets = targets
        if labeled:
            if indexs is not None:
                self.samples = samples[indexs]
                self.targets = targets[indexs]
        else:
            if indexs is not None:
                self.samples = samples[indexs]
                self.targets = targets[indexs]

    def __getitem__(self, index):
        return self.samples[index],self.targets[index]

    def __len__(self):
        return self.samples.size(0)

def load_sensor(labeled_proportion, batchsize):
    # load image data
    if os.path.isfile('./label_idx_rate{}.npy'.format(labeled_proportion)) and os.path.isfile('./unlabel_idx_rate{}.npy'.format(labeled_proportion)):
        print("loading idx")
        train_labeled_idxs = np.load('./label_idx_rate{}.npy'.format(labeled_proportion))
        train_unlabeled_idxs = np.load('./unlabel_idx_rate{}.npy'.format(labeled_proportion))
    else:
        train_labeled_idxs, train_unlabeled_idxs = get_labeled_index(labeled_rate=labeled_proportion)
        np.save('./label_idx_rate{}.npy'.format(labeled_proportion),train_labeled_idxs)
        np.save('./unlabel_idx_rate{}.npy'.format(labeled_proportion),train_unlabeled_idxs)
    train_labeled_dataset = sensordataset(labeled=True,indexs=train_labeled_idxs)
    train_unlabeled_dataset = sensordataset(labeled=False,indexs=train_unlabeled_idxs)

    labeled_dataloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    unlabeled_dataloader = torch.utils.data.DataLoader(
        train_unlabeled_dataset,
        batch_size=batchsize,
        num_workers=0,
        shuffle=False,
        drop_last=True
    )
    print(f"#Labeled sensor: {len(train_labeled_idxs)} #Unlabeled sensor: {len(train_unlabeled_idxs)}")

    return labeled_dataloader,unlabeled_dataloader