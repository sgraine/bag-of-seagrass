# Dataloaders for DeepSeagrass dataset

# Import packages
import torch
import numpy as np
import os, random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

import imgaug.augmenters as iaa

### Dataset Classes ###

# Patch dataset: this dataset randomly chooses samples for each batch and performs data augmentation for training
class PatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, num_classes=4, augment=True, crop=False, cropsize=435):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.augment = augment
        self.crop = crop
        self.cropsize = cropsize
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)
        image = np.expand_dims(image, 0)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        # Preprocess and augment data
        if self.augment == True:
            image = self.augmentor(image)

        if self.crop == True:
            image = self.cropper(image, self.cropsize)

        image = np.squeeze(image)
        image_transformed = self.transform_func(image)

        return image_transformed, label

    def transform_func(self, image):
        'Transform into a pytorch Tensor'

        transform_list = []

        transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform_list.append(transforms.Normalize(mean = [0.4598, 0.5328, 0.3893], std = [0.1034, 0.1062, 0.1328])) # deepseagrass
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def cropper(self, images, cropsize):

        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=cropsize, height=cropsize)
        ])

        return seq.augment_images(images)

    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)

        # Original augmentations
        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
            iaa.Fliplr(0.5),
            often(
                iaa.WithChannels(0, iaa.Add((-30, 30))) # RGB = 0,1,2
                ),
            sometimes(
                iaa.LinearContrast((0.5, 2.0))
                ),
            sometimes(
                iaa.AddToBrightness((-30, 30))
                ),
            sometimes(
                iaa.GaussianBlur(sigma=(0,0.5))
                )
        ], random_order=True) # apply augmenters in random order      

        # # Our heavy augmentation for floatyboat
        # seq = iaa.Sequential([           
        #     # Best Augmentation Strategy: Colour Augmentation
        #     iaa.Fliplr(0.5),
        #     sometimes(
        #         iaa.WithChannels(0, iaa.Add((-20, 5))) # RGB = 0,1,2
        #         ),
        #     sometimes(
        #         iaa.WithChannels(1, iaa.Add((-10, 10))) # RGB = 0,1,2
        #         ),
        #     sometimes(
        #         iaa.WithChannels(2, iaa.Add((-10, 30))) # RGB = 0,1,2
        #         ),
        #     often(
        #         iaa.LinearContrast((0.5, 2.0))
        #         ),
        #     often(
        #         iaa.MultiplyAndAddToBrightness(mul=(1.0, 1.5), add=(-30, 30))
        #         ),
        #     often(
        #         iaa.GaussianBlur(sigma=(0,0.5))
        #         ),
        #     often(
        #         iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
        #     ),
        #     often(
        #         iaa.Grayscale(alpha=(0.0, 1.0))
        #     ),
        #     often(
        #         iaa.ScaleX((0.8, 1.2))
        #     ),
        #     often(
        #         iaa.ScaleY((0.8, 1.2))
        #     )
        # ], random_order=True) # apply augmenters in random order
       
        return seq.augment_images(images)

# EfficientNet patch dataset: this dataset formats/pre-processes patches for training with EfficientNet
class EfficientPatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, num_classes=4, augment=True):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.num_classes = num_classes
        self.augment = augment
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)
        image = np.expand_dims(image, 0)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        # Preprocess and augment data
        if self.augment == True:
            image = self.augmentor(image)

        image = np.squeeze(image)
        image = Image.fromarray(image)
        image_transformed = self.transform_func(image)

        return image_transformed, label

    def transform_func(self, image):
        '''Transform into a pytorch Tensor
        EfficientNet-B5 transforms: resize_size=[456] using interpolation=InterpolationMode.BICUBIC, followed by a central crop of crop_size=[456]. 
        Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].'''

        transform_list = []
        transform_list.append(transforms.Resize((456,456), interpolation=transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.CenterCrop(456))
        transform_list.append(transforms.ToTensor()) # also rescales values to between 0 and 1
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def augmentor(self, images):
        ''' EfficientNet augmentation (as per ECU paper): horizontal ﬂip, vertical ﬂip, 
        brightness, colour augmentation, random cropping and zooming transformations.'''
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            sometimes(
                iaa.WithChannels(0, iaa.Add((-30, 30))) # RGB = 0,1,2
                ),
            sometimes(
                iaa.LinearContrast((0.5, 2.0))
                ),
            sometimes(
                iaa.AddToBrightness((-30, 30))
                ),
            sometimes(
                iaa.AddToHueAndSaturation((-30, 30))
                ),
            sometimes(
                iaa.GaussianBlur(sigma=(0,0.5))
                ),
            sometimes(
                iaa.ScaleX((0.7, 2.0))
            ),
            sometimes(
                iaa.ScaleY((0.7, 2.0))
            )
        ], random_order=True) # apply augmenters in random order
        
        return seq.augment_images(images)

# Test patch dataset: no augmentations as for training
class TestPatchDataset(Dataset):
    def __init__(self, image_path_array, labels_array, image_dir, num_classes=4):
        self.image_path_array = image_path_array
        self.labels_array = labels_array
        self.image_dir = image_dir
        self.num_classes = num_classes
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)
        image = np.squeeze(image)
        image_transformed = self.transform_func(image)

        label = torch.from_numpy(np.array([self.labels_array[idx]])).to(torch.int64)
        label = torch.squeeze(label)

        return image_transformed, label

    def transform_func(self, image):
        'Transform into a pytorch Tensor'
        transform_list = []

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        return transform(image).float()

# SeaFeats dataset: this dataset reads in whole images instead of patches
class ImageCosine(Dataset):
    def __init__(self, image_path_array, image_dir,labels_array, num_classes=4, augment=False):
        self.image_path_array = image_path_array
        self.image_dir = image_dir
        self.labels_array = labels_array
        self.num_classes = num_classes
        self.augment = augment
 
    def __len__(self):
        return len(self.image_path_array)

    def __getitem__(self, idx: int):
        filename = self.image_path_array[idx]

        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)
        image_transformed = np.expand_dims(image, 0)

        # Preprocess and augment data
        if self.augment == True:
            image_transformed = self.augmentor(image_transformed)

        image_transformed = np.squeeze(image_transformed)
        image_transformed = Image.fromarray(image_transformed)
        image_transformed = self.transform_func(image_transformed)

        # Create one-hot encoding for image label
        label_np = np.array([self.labels_array[idx]])
        labels_padded = np.zeros(self.num_classes)
        labels_padded[int(label_np)] = 1.0

        label = torch.from_numpy(labels_padded).to(torch.int64)
        label = torch.squeeze(label)

        image_name = str(filename).split('/')[1][:-4]
        return image_transformed, label, image, image_name

    def transform_func(self, image):
        'Transform into a pytorch Tensor'

        transform_list = []

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def augmentor(self, images):
        'Apply data augmentation'
        rarely = lambda aug: iaa.Sometimes(0.2, aug)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)

        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
            iaa.Fliplr(0.5),
            often(
                iaa.WithChannels(0, iaa.Add((-30, 30))) # RGB = 0,1,2
                ),
            sometimes(
                iaa.LinearContrast((0.5, 2.0))
                ),
            sometimes(
                iaa.AddToBrightness((-30, 30))
                ),
            sometimes(
                iaa.GaussianBlur(sigma=(0,0.5))
                ),
            sometimes(
                iaa.ScaleX((1.0, 1.5))
            ),
            sometimes(
                iaa.ScaleY((1.0, 1.5))
            ),
            rarely(
                aug = iaa.Grayscale(alpha=(0.7, 1.0))
            )
        ], random_order=True) # apply augmenters in random order
        
        return seq.augment_images(images)

### Dataloaders ###

# Dataloader for training SeaCLIP on patches already labeled by CLIP
def loadDataValSplit(class_list, train_path, test_path, batch_size=12, efficientnet=False, augment=False, num_workers=4, crop=False, cropsize=435):
    'Loads data into generator object'
    all_images_array = np.array([])
    all_labels_array = np.array([])
    test_images_array = np.array([])
    test_labels_array = np.array([])          

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, class_list[category])) ] 
        
        for i in range(len(img_list)):
            all_images_array = np.append(all_images_array, os.path.join(class_list[category], img_list[i]))
            all_labels_array = np.append(all_labels_array, category)       

    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) ] 
        
        for i in range(len(img_list)):
            test_images_array = np.append(test_images_array, os.path.join(class_list[category], img_list[i]))
            test_labels_array = np.append(test_labels_array, category)         

    all_indexes = list(range(0, np.shape(all_images_array)[0]))
    random.Random(4).shuffle(all_indexes) # Use a seed to ensure the train/val split is always the same

    train_indexes = all_indexes[:(int(1.0*len(all_indexes)))] # 0.8
    val_indexes = all_indexes[(int(0.8*len(all_indexes))):] # 0.8

    train_images_array = all_images_array[train_indexes]
    val_images_array = all_images_array[val_indexes]

    train_labels_array = all_labels_array[train_indexes]
    val_labels_array = all_labels_array[val_indexes]

    target = torch.from_numpy(train_labels_array.astype(np.int32))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    if efficientnet:
        # Now create the dataloaders  
        train_dataset = EfficientPatchDataset(train_images_array, train_labels_array, train_path, num_classes=len(class_list), augment=augment)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        val_dataset = EfficientPatchDataset(val_images_array, val_labels_array, train_path, num_classes=len(class_list), augment=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        test_dataset = EfficientPatchDataset(test_images_array, test_labels_array, test_path, num_classes=len(class_list), augment=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)
    else:
        # Now create the dataloaders  
        train_dataset = PatchDataset(train_images_array, train_labels_array, train_path, num_classes=len(class_list), augment=augment, crop=crop, cropsize=cropsize)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        val_dataset = PatchDataset(val_images_array, val_labels_array, train_path, num_classes=len(class_list), augment=False, crop=crop, cropsize=cropsize)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

        test_dataset = PatchDataset(test_images_array, test_labels_array, test_path, num_classes=len(class_list), augment=False, crop=crop, cropsize=cropsize)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader, val_dataloader, test_dataloader

# Dataloader for reading in the test patch dataset
def loadTestSetOrig(class_list, test_path, batch_size=12, num_workers=4, efficientnet=False):
    'Loads data into generator object'
    test_images_array = np.array([])
    test_labels_array = np.array([])          

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        # img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) if ( re.match(r'^(?![\._]).*$', f) and (f.endswith(".jpg") or f.endswith(".png")) )] # filter out the apple files
        img_list = [f for f in os.listdir(os.path.join(test_path, class_list[category])) if ( (f.endswith(".jpg") or f.endswith(".png")) )] # filter out the apple files
        
        for i in range(len(img_list)):
            test_images_array = np.append(test_images_array, os.path.join(class_list[category], img_list[i]))
            test_labels_array = np.append(test_labels_array, category)         

    if efficientnet:
        test_dataset = EfficientPatchDataset(test_images_array, test_labels_array, test_path, num_classes=len(class_list), augment=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)
    
    else:
        # Now create the dataloaders  
        test_dataset = TestPatchDataset(test_images_array, test_labels_array, test_path, num_classes=len(class_list))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return test_dataloader

# Dataloader for SeaFeats model
def loadImageCosine(class_list, train_path, batch_size=1, num_workers=2):
    'Loads data into generator object'
    images_array = np.array([])
    labels_array = np.array([])

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, class_list[category])) if ( f.endswith(".JPG") )]

        for i in range(len(img_list)):
            images_array = np.append(images_array, os.path.join(class_list[category], img_list[i]))
            labels_array = np.append(labels_array, category) 

    target = torch.from_numpy(labels_array.astype(np.int32))
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])

    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Now create the dataloaders  
    train_dataset = ImageCosine(images_array, train_path, labels_array, num_classes=len(class_list), augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, worker_init_fn=worker_init_fn, drop_last=True, pin_memory=False)

    return train_dataloader

### Helper functions ###

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)