# Import packages
import os
import numpy as np
import torch
import clip, random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import imgaug.augmenters as iaa
from torchvision import transforms

# Dataset class
class DynamicPatchDensity(Dataset):
    def __init__(self, image_path_array, image_dir, labels_array, indexes, num_classes=4, augment=False):
        self.image_path_array = image_path_array
        self.image_dir = image_dir
        self.labels_array = labels_array
        self.num_classes = num_classes
        self.augment = augment
        self.indexes = indexes
 
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx: int):
        split_index = self.indexes[idx]
        filename = self.image_path_array[split_index]
        image_label = self.labels_array[split_index]

        # Read in the full image
        image_path = os.path.join(self.image_dir, filename)
        
        image = Image.open(image_path)
        image = np.array(image)

        width = int(image.shape[1])
        height = int(image.shape[0])

        row = 5
        col = 8

        # Divide the full image into a grid 8x5 of patches
        grid, _, _ = self.img_to_grid(image,row,col)

        all_patches = []
        for patch in grid:
            patch_crop = self.cropper(patch, int(np.floor(width / col)), int(np.floor(height / row)))
            all_patches.append(patch_crop)

        image_name = str(filename).split('/')[1][:-4]
        return all_patches, image_label, image, image_name

    def img_to_grid(self, img, row,col):
        ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
        hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
        grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
        return grid, len(ww), len(hh)

    def transform_func(self, image):
        'Transform into a pytorch Tensor'

        transform_list = []

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # expects the image to be RGB
        transform = transforms.Compose(transform_list)

        return transform(image).float()

    def cropper(self, images, width, height):

        seq = iaa.Sequential([
            iaa.CropToFixedSize(width=width, height=height)
        ])

        return seq.augment_image(images)

    def augmentor(self, images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)

        seq = iaa.Sequential([            
            # Best Augmentation Strategy: Colour Augmentation
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
        
        return seq.augment_images(images)

# Dataloader
def loadDynamicPatchesDensity(class_list, train_path, batch_size=1, num_workers=2):
    'Loads data into generator object'
    images_array = np.array([])
    labels_array = np.array([])

    # Need to obtain the image_paths and labels for the dataset
    for category in range(len(class_list)):
        img_list = [f for f in os.listdir(os.path.join(train_path, class_list[category])) if ( f.endswith(".JPG") )] # filter out the apple files

        for i in range(len(img_list)):
            images_array = np.append(images_array, os.path.join(class_list[category], img_list[i]))
            labels_array = np.append(labels_array, category) 

    all_indexes = list(range(0, np.shape(images_array)[0]))
    random.Random(4).shuffle(all_indexes) # Use a seed to ensure the train/val split is always the same

    train_indexes = all_indexes[:(int(1.0*len(all_indexes)))]
    val_indexes = all_indexes[(int(0.0*len(all_indexes))):]

    # Now create the dataloaders  
    train_dataset = DynamicPatchDensity(images_array, train_path, labels_array, train_indexes, num_classes=len(class_list), augment=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False)

    val_dataset = DynamicPatchDensity(images_array, train_path, labels_array, val_indexes, num_classes=len(class_list), augment=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn, drop_last=False, pin_memory=False)

    return train_dataloader, val_dataloader

# We need this to ensure each worker has a different random seed
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_read_path = "../DeepSeagrass/SeparatedIntoClasses"
    dataset_save_path = "CLipLabels"
    class_list = ["Background","Ferny","Rounded","Strappy"]

    batch_size = 1
    num_images = 1

    train_dataloader, _ = loadDynamicPatchesDensity(class_list, os.path.join(dataset_read_path, "Images"), batch_size)

    # Load CLIP
    feature_extractor, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["a photo of sand", "a photo of water", "a photo of sand or water", "a blurry photo of water", "a blurry photo of sand", "a blurry photo of seagrass", "a photo containing some seagrass", "a photo of underwater plants", "a photo of underwater grass", "a photo of green, grass-like leaves underwater", "a photo of seagrass"]).to(device)
    
    dataloaders = {}
    dataloaders['train'] = train_dataloader

    step = 0
    for inputs, image_label, whole_image, image_name in tqdm(dataloaders['train']):
        i = 0
        for patch in inputs:
            patch = patch.numpy()
            image = preprocess(Image.fromarray(np.squeeze(patch))).unsqueeze(0).to(device)
            logits_per_image, _ = feature_extractor(image, text)

            probs = logits_per_image.softmax(dim=-1)
            preds_torch = torch.argmax(probs, dim=1).int()

            # The first 5 prompts are for background, then all others afterwards are for seagrass
            if preds_torch > 4:
                # Save the image patch in the folder corresponding to image_label:
                bag_int = torch.squeeze(image_label).int()
                save_path = os.path.join(dataset_save_path, class_list[bag_int])+"/"+image_name[0]+"_"+str(i)+".png"
                pil_img = Image.fromarray(np.squeeze(patch)).save(save_path)

            else:
                # Save the image patch in the Background folder:
                save_path = os.path.join(dataset_save_path, class_list[0])+"/"+image_name[0]+"_"+str(i)+".png"
                pil_img = Image.fromarray(np.squeeze(patch)).save(save_path)
            i += 1
