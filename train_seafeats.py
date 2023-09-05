# Import packages
import os, copy, time
import numpy as np
import torch
import torch.nn as nn
from seagrass_dataloaders import loadImageCosine, loadTestSetOrig
import torch.optim as optim
from tqdm import tqdm
import wandb, torchmetrics
import torchvision.models as models
from sklearn.metrics import confusion_matrix

class CosineLoss(nn.Module):
    def __init__(self, class_list, thresh=0.6, num_patches=45):
        super(CosineLoss, self).__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.num_patches = num_patches
        self.thresh = thresh
        weights = torch.tensor([1.0,1.5,1.2,1.2]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.soft = torch.nn.Softmax(dim=1)
        
    def forward(self, features, average_features, outputs, labels_onehot):
        # INPUTS:
        # 1) features = network feature vectors            (torch.FloatTensor: shape = [B, num_patches, num_features])  # feature vector for each patch
        # 2) average_features = template feature vectors   (torch.FloatTensor: shape = [num_classes, num_features])  # template feature vector for each class
        # 3) outputs = network output logits               (torch.FloatTensor: shape = [B, num_patches, num_classes])  # network output logits for each of the patches
        # 4) labels_onehot                                 (torch.FloatTensor: shape = [B, num_classes])  # one hot encoding for the whole image's label

        # RETURNS:
        # 1) cosine loss                                   (torch.FloatTensor: shape = [1])

        # Define our function for cosine similarity
        cosine = nn.CosineSimilarity(dim=1)
        
        batch_size = features.shape[0]

        # Accumulate the per-patch softmax and pseudo-label vectors over the whole batch
        whole_batch_soft_vals = []
        whole_batch_pseudo_labels = []

        for image_index in range(batch_size):
            label_onehot_one_image = torch.unsqueeze(labels_onehot[image_index, :],0).float()
            label_index_one_image = torch.argmax(label_onehot_one_image).int()
            features_one_image = features[image_index,:,:]
            outputs_one_image = outputs[image_index,:,:]

            # Only perform this operation if whole image is a seagrass image:
            if label_index_one_image > 0:

                similarities_back = cosine(features_one_image, average_features[0])     # shape = [num_patches]
                similarities_sea = cosine(features_one_image, average_features[label_index_one_image])     # shape = [num_patches]

                # Label the patch as background if the similarity is closer to the background template than seagrass
                temp = torch.where(similarities_back > similarities_sea, 0, 1)
                # This ensures the similarity is significant - self.thresh can be tuned, but we use 0.6
                temp2 = torch.where(similarities_back > self.thresh, 0, 1) 
                pseudo = torch.logical_or(temp, temp2).long().to(device)     # shape = [num_patches]

            else:
                pseudo = torch.zeros(self.num_patches).long().to(device)         # shape = [num_patches]

            background_tensor = torch.unsqueeze(torch.tensor([1.0, 0.0, 0.0, 0.0]),0).float().to(device)
            
            pseudo_onehot_one_image = []
            for i in range(pseudo.shape[0]):
                if pseudo[i]==1:
                    pseudo_onehot_one_image.append(label_onehot_one_image)
                else:
                    pseudo_onehot_one_image.append(background_tensor)

            pseudo_onehot_one_image = torch.cat(pseudo_onehot_one_image, dim=0).to(device)

            # Apply the softmax over the per-patch output logits
            outputs_soft_one_image = self.soft(outputs_one_image)

            whole_batch_soft_vals.append(outputs_soft_one_image)
            whole_batch_pseudo_labels.append(pseudo_onehot_one_image)

        whole_batch_soft_vals = torch.cat(whole_batch_soft_vals, dim=0).to(device)
        whole_batch_pseudo_labels = torch.cat(whole_batch_pseudo_labels, dim=0).to(device)

        # Now we can use a basic categorical cross entropy with all samples from the whole batch
        loss = self.criterion(whole_batch_soft_vals, whole_batch_pseudo_labels)

        return loss

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_path = "../DeepSeagrass/SeparatedIntoClasses"
    class_list = ["Background","Ferny","Rounded","Strappy"]

    epochs = 150
    learning_rate = 0.00001
    batch_size = 3
    num_images = 1
    fine_tune = True
    vis_step = [0,500,1000,2000]
    thresh = 0.6

    train_dataloader = loadImageCosine(class_list, os.path.join(dataset_path, "Images"), batch_size=batch_size)
    val_dataloader = loadTestSetOrig(class_list, os.path.join("../DeepSeagrass", "Validate - Refined"), batch_size=12, num_workers=2)

    model = models.resnet18(pretrained=False, num_classes=4)

    checkpoint = torch.load('simclr_feature_extractor.pth.tar')
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    model.fc = torch.nn.Identity()
    model.load_state_dict(state_dict, strict=True)

    layers = list(model.children())[:-2]

    av_pool = nn.AvgPool2d((16, 16), stride=(16, 16), padding=0)
    flatten = nn.Flatten(2, -1)
    layers.append(av_pool)
    layers.append(flatten)
    layers.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
    layers.append(nn.Linear(512, 512))
    layers.append(nn.Dropout(0.15))
    layers.append(nn.Linear(512, len(class_list)))
    model = nn.Sequential(*layers)

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    model.to(device)

    avgpool_layer = model._modules["10"]

    my_embedding = torch.zeros(batch_size, 45, 512) # number of vectors comes from the kernel size and stride of the average pool operation
    def fun(m, i, o): my_embedding.copy_(torch.squeeze(o.data))
    h = avgpool_layer.register_forward_hook(fun)

    criterion = CosineLoss(class_list)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr = 0.000001)

    ##### Accuracy Metrics #####
    acc_metric_train = torchmetrics.Accuracy(num_classes = len(class_list), mdmc_average='global').to(device)
    acc_metric_val = torchmetrics.Accuracy(num_classes = len(class_list), mdmc_average='global').to(device)
    acc_metric_test = torchmetrics.Accuracy(num_classes = len(class_list), mdmc_average='global').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_train = torchmetrics.Precision(num_classes = len(class_list), average='none', mdmc_average='global').to(device)
    per_class_prec_metric_val = torchmetrics.Precision(num_classes = len(class_list), average='none', mdmc_average='global').to(device)
    per_class_prec_metric_test = torchmetrics.Precision(num_classes = len(class_list), average='none', mdmc_average='global').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_train = torchmetrics.Recall(num_classes = len(class_list), average='none', mdmc_average='global').to(device)
    per_class_recall_metric_val = torchmetrics.Recall(num_classes = len(class_list), average='none', mdmc_average='global').to(device)
    per_class_recall_metric_test = torchmetrics.Recall(num_classes = len(class_list), average='none', mdmc_average='global').to(device)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    model_name = f"model-{int(time.time())}"+"_seafeats"
    print("################################# MODEL NAME: ########################################")
    print(model_name)
    os.makedirs(model_name+"_dir")

    soft = torch.nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()

    best_acc_val = -1.0
    prev_loss = 1000.0
    loss = []
    for epoch in range(epochs):

        metrics = {}
        test_metrics = {}

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        ####################### TRAINING PHASE #########################
        phase = 'train'
        print("*** Phase: "+phase+" ***")

        model.train()

        background_feat = []
        ferny_feat = []
        rounded_feat = []
        strappy_feat = []
        
        # Calculate the per-class feature vectors
        print("*** Calculating Template Features ***")
        s = 0
        for inputs, bag_labels, _, _ in tqdm(dataloaders[phase]):
            inputs = inputs.float().to(device)      # [B, 3, H, W]
            outputs = model(inputs)                 # [B, 45, num_classes]

            for image_index in range(inputs.shape[0]):
                # Obtain the class for the whole image
                bag_int = torch.argmax(bag_labels[image_index,:])

                temp_embedding = my_embedding[image_index,:,:].detach()
                temp_embedding_norm = nn.functional.normalize(temp_embedding, dim=1)

                if bag_int == 0:
                    background_feat.append(temp_embedding_norm)  # my_embedding [B, num_patches, num_features]
                elif bag_int == 1:
                    ferny_feat.append(temp_embedding_norm)
                elif bag_int == 2:
                    rounded_feat.append(temp_embedding_norm)
                elif bag_int == 3:
                    strappy_feat.append(temp_embedding_norm)
                else:
                    raise Exception("Something wrong with bag_label - class number does not exist.")
            s += 1

        background_feat_torch = torch.cat(background_feat, dim=0) # [B x num_patches x num_samples, 512]
        ferny_feat_torch = torch.cat(ferny_feat, dim=0)
        rounded_feat_torch = torch.cat(rounded_feat, dim=0)
        strappy_feat_torch =torch.cat(strappy_feat, dim=0)

        # Bundle the feature vectors: typically an element-wise sum or averaging of the vector elements
        background_feat_torch = torch.unsqueeze(torch.mean(background_feat_torch, dim=0), dim=0) # [1, 512]
        ferny_feat_torch = torch.unsqueeze(torch.mean(ferny_feat_torch, dim=0), dim=0)
        rounded_feat_torch = torch.unsqueeze(torch.mean(rounded_feat_torch, dim=0), dim=0)
        strappy_feat_torch = torch.unsqueeze(torch.mean(strappy_feat_torch, dim=0), dim=0)

        # Store the template for each class
        average_features = torch.cat((background_feat_torch, ferny_feat_torch, rounded_feat_torch, strappy_feat_torch), dim=0) # [num_classes, 512] 

        running_loss_train = 0.0
        step = 0
        for inputs, bag_labels, images_orig, filename in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 3, H, W]
            bag_labels = bag_labels.to(device)                  # [B, num_classes]

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                
                outputs = model(inputs)      # [B, num_patches, num_classes]

                # Cross entropy with patch label and class of patch
                batch_loss = criterion(my_embedding, average_features, outputs, bag_labels.float())

                outputs_soft = soft(outputs)  # [B, num_patches, num_classes]
            
                batch_loss.backward()
                optimizer.step()
            
            running_loss_train = running_loss_train + batch_loss.detach()

            step += 1

        train_loss = running_loss_train / len(dataloaders[phase])
        wandb.log({"train_loss":train_loss.item()}, step=epoch)

        scheduler.step(train_loss) 
        if epoch >= 10:
            scheduler.step()       

        # save the model
        if train_loss < prev_loss:
            prev_loss = train_loss
            torch.save(model.state_dict(), str(model_name)+'_CKPT.pt')

        if epoch % 3 == 0:

            ####################### VALIDATION PHASE #########################
            phase = 'val'
            print("*** Phase: "+phase+" ***")
            val_model = copy.deepcopy(model)
            val_model.eval()

            layers_alter_1 = list(val_model.children())[:8]
            layers_alter_2 = list(val_model.children())[11:]

            all_layers = []
            for layer in layers_alter_1:
                all_layers.append(layer)

            all_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            all_layers.append(nn.Flatten(1))

            for layer in layers_alter_2:
                all_layers.append(layer)

            val_model = nn.Sequential(*all_layers)

            # Disable batch normalization for inference
            for m in val_model.modules():
                for child in m.children():
                    if type(child) == nn.BatchNorm2d:
                        child.track_running_stats = False
                        child.running_mean = None
                        child.running_var = None

            soft = torch.nn.Softmax(dim=1)
            step = 0
            preds_list = []
            labels_list = []
            for inputs, labels in tqdm(dataloaders[phase]):

                inputs = inputs.float().to(device)                  # [B, 3, H, W]
                labels = labels.to(device)                            # [B, H, W]

                # forward pass - track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = val_model(inputs) # [batch_size, num_classes]           
                    outputs = soft(outputs) # [batch_size, num_classes]
                    preds_torch = torch.argmax(outputs, dim=1).int()  # [1]
                    labels_torch = labels.int()

                    preds_list.append(preds_torch.cpu().numpy())
                    labels_list.append(labels_torch.cpu().numpy())

                    acc_val = acc_metric_val(preds_torch, labels_torch)
                    per_class_prec_val = per_class_prec_metric_val(preds_torch, labels_torch)
                    per_class_recall_val = per_class_recall_metric_val(preds_torch, labels_torch)

            preds_list = np.stack(preds_list).flatten()
            labels_list = np.stack(labels_list).flatten()

            conf = confusion_matrix(labels_list, preds_list, labels=[0,1,2,3])
            print(conf)     

            acc_val = acc_metric_val.compute()
            per_class_prec_val = per_class_prec_metric_val.compute()
            per_class_recall_val = per_class_recall_metric_val.compute()

            wandb.log({"prec_background_val":per_class_prec_val[0], 
                "prec_ferny_val":per_class_prec_val[1],
                "prec_rounded_val":per_class_prec_val[2],
                "prec_strappy_val":per_class_prec_val[3]}, step=epoch)

            wandb.log({"recall_background_val":per_class_recall_val[0], 
                "recall_ferny_val":per_class_recall_val[1],
                "recall_rounded_val":per_class_recall_val[2],
                "recall_strappy_val":per_class_recall_val[3]}, step=epoch)

            wandb.log({"val_acc":acc_val.item()}, step=epoch)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                torch.save(model.state_dict(), str(model_name)+'_best_acc_CKPT.pt')

            print("val", acc_val.item())
            print(per_class_prec_val, per_class_recall_val)

            acc_metric_val.reset()
            per_class_prec_metric_val.reset()
            per_class_recall_metric_val.reset()

    # Save the final model weights
    torch.save(model.state_dict(), str(model_name)+'_FINAL.pt')