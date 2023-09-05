print("Inference script has started.")

# Import packages
import os
import numpy as np
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import imgaug.augmenters as iaa

print("Packages imported successfully.")

### Data Transformation Functions ###
def transform_func(image):
    'Transform into a pytorch Tensor'
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])) # imagenet
    transform = transforms.Compose(transform_list)

    return transform(image).float()

def transform_func_2(image):
    'Transform into a pytorch Tensor'
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    return transform(image).float()

def transform_func_vis(image):
    'Retain format for visualisation purposes'
    transform_list = []
    transform = transforms.Compose(transform_list)

    return transform(image)

# Function for visualise the model outputs
def single_image_preds(preds, whole_image, image_name):
    # Perform inference on a single image
    fig=plt.figure(dpi=200)
    fig.set_size_inches(20, 10)
    ax=fig.add_subplot()

    plt.axis('off')

    # Plot the original image
    ax.imshow(np.squeeze(whole_image), alpha=1.0) # rows, columns, channels

    scale = np.array([np.shape(whole_image)[1],np.shape(whole_image)[0]])

    sigmoid_vals = preds.detach().cpu().numpy()

    if np.shape(sigmoid_vals)[0] == 45:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 35:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 7)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 40:
        sigmoid_vals = np.reshape(sigmoid_vals, (5, 8)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 54:
        sigmoid_vals = np.reshape(sigmoid_vals, (6, 9)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 77:
        sigmoid_vals = np.reshape(sigmoid_vals, (7, 11)) # rows, columns
    elif np.shape(sigmoid_vals)[0] == 6:
        sigmoid_vals = np.reshape(sigmoid_vals, (2, 3)) # rows, columns
    else:
        print("Not sure how many rows and columns!")

    CMAP = [[255,20,147],[255, 0, 0],[0, 0, 255],[255,165,0]]
    CMAP = np.asarray(CMAP)
    colour_predictions = CMAP[sigmoid_vals]

    # Plot the heatmap as a transparent overlay
    offs = np.array([scale[0]/sigmoid_vals.shape[1], scale[1]/sigmoid_vals.shape[0]])

    # Add the model's predicted class labels to each patch
    for pos, val in np.ndenumerate(sigmoid_vals):
        ax.annotate(val, xy=np.array(pos)[::-1]*offs+offs/2, ha="center", va="center",fontsize=30)

    heatmap = ax.imshow(np.flipud(colour_predictions), alpha=0.5, aspect="auto", extent=(0,scale[0],0,scale[1])) # alpha -> more transparent as value decreases

    ax.invert_yaxis()

    # Save the final figure as a .png
    plt.savefig(image_name+"_pred.png",bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)

### Helper Functions ###
class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def normalize_logits(logits):
    mean = logits.mean()
    std = logits.std()
    normalized_logits = (logits - mean) / std
    return normalized_logits

def img_to_grid(img, row,col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj+1,i:ii+1,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

def cropper(images, width, height):

    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=width, height=height)
    ])

    return seq.augment_image(images)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_path = "SaveTestInferences"
    test_path = "TestImage"

    class_list = ["Background","Ferny","Rounded","Strappy"]

    stride = 16

    # #****************************** SEAFEATS MODEL ***************************************
    seafeats_model_path = os.path.join("Models","SeaFeats.pt")

    seafeats = models.resnet18()
    num_ftrs = seafeats.fc.in_features
    layers = list(seafeats.children())[:-2]
    av_pool = nn.AvgPool2d((16, 16), stride=(stride, stride), padding=0)
    flatten = nn.Flatten(2, -1)
    layers.append(av_pool)
    layers.append(flatten)
    layers.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
    layers.append(nn.Linear(512, 512))
    layers.append(nn.Dropout(0.15))
    layers.append(nn.Linear(512, len(class_list)))
    seafeats = nn.Sequential(*layers)

    seafeats.load_state_dict(torch.load(seafeats_model_path))    

    layers_alter_1 = list(seafeats.children())[:8]
    layers_alter_2 = list(seafeats.children())[11:]

    all_layers = []
    for layer in layers_alter_1:
        all_layers.append(layer)

    all_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    all_layers.append(nn.Flatten(1))

    for layer in layers_alter_2:
        all_layers.append(layer)

    seafeats = nn.Sequential(*all_layers).to('cuda')

    seafeats = seafeats.to(device)
    seafeats.eval()
    print("SeaFeats model has loaded.")

    #****************************** SEACLIP MODEL ***************************************
    clip_model_path = os.path.join("Models","SeaCLIP.pt")

    clip_model_load = models.resnet18()
    clip_model_load.fc = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.15),
                                    nn.Linear(512, 4))

    clip_model_load.load_state_dict(torch.load(clip_model_path))

    all_layers = list(clip_model_load.children())
    clip_model_pool = all_layers[:-2]
    av_pool = nn.AvgPool2d((16, 16), stride=(stride, stride), padding=0)
    flatten = nn.Flatten(2, -1)
    clip_model_pool.append(av_pool)
    clip_model_pool.append(flatten)
    clip_model_pool.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
    clip_model_pool.append(all_layers[-1])
    seaclip = nn.Sequential(*clip_model_pool)

    seaclip = seaclip.to(device)
    seaclip.eval()
    print("SeaCLIP model has loaded.")

    all_images = [i for i in os.listdir(test_path)]

    print("Processing images...")
    for filename in all_images:

        image = Image.open(os.path.join(test_path, filename))
        image_torch = torch.unsqueeze(transform_func_2(image), dim=0).to(device)
        
        vis_image = Image.open(os.path.join(test_path, filename))
        vis_image = np.array(vis_image)

        row = 5
        col = 9

        width = int(vis_image.shape[1])
        height = int(vis_image.shape[0])

        # Divide the full image into a grid of patches
        grid, _, _ = img_to_grid(vis_image,row,col)

        all_patches = []
        for patch in grid:
            patch_crop = cropper(patch, int(np.floor(width / col)), int(np.floor(height / row)))
            all_patches.append(torch.unsqueeze(transform_func(Image.fromarray(patch_crop)), dim=0))

        all_patches_torch = torch.cat(all_patches, dim=0).to(device)

        soft = torch.nn.Softmax(dim=0)

        outputs_list = []
        cos_list = []
        clip_list = []

        for patch_torch in all_patches_torch:
            outputs_cos = seafeats(torch.unsqueeze(patch_torch, dim=0))  # [batch_size, num_classes] 
            cos_soft =  soft(torch.squeeze(outputs_cos))
            preds_cos = torch.argmax(cos_soft, dim=0).int()
            outputs_cos = normalize_logits(outputs_cos)  

            outputs_clip = seaclip(torch.unsqueeze(patch_torch, dim=0))  # [batch_size, num_classes]   
            clip_soft =  soft(torch.squeeze(outputs_clip))
            preds_clip = torch.argmax(clip_soft, dim=0).int()
            outputs_clip = normalize_logits(outputs_clip)

            total_logits = torch.add(outputs_cos, outputs_clip) / 2

            outputs_soft = soft(torch.squeeze(total_logits))
            preds_torch = torch.argmax(outputs_soft, dim=0).int()

            outputs_list.append(torch.unsqueeze(preds_torch, dim=0))
            cos_list.append(torch.unsqueeze(preds_cos, dim=0))
            clip_list.append(torch.unsqueeze(preds_clip, dim=0))

        outputs_torch = torch.cat(outputs_list, dim=0)
        outputs_torch_cos = torch.cat(cos_list, dim=0)
        outputs_torch_clip = torch.cat(clip_list, dim=0)

        single_image_preds(outputs_torch_clip, transform_func_vis(image), os.path.join(save_path, filename[:-4]+"_clip"))
        single_image_preds(outputs_torch_cos, transform_func_vis(image), os.path.join(save_path, filename[:-4]+"_cos"))
        single_image_preds(outputs_torch, transform_func_vis(image), os.path.join(save_path, filename[:-4]+"_tot"))