# Import packages
import os
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
from seagrass_dataloaders import loadTestSetOrig
from tqdm import tqdm
import torchmetrics
from sklearn.metrics import confusion_matrix

print("Packages imported successfully.")

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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_list = ["Background","Ferny","Rounded","Strappy"]

    test_dataloader = loadTestSetOrig(class_list, "DeepSeagrass-Test", batch_size=12, num_workers=2, efficientnet=False)

    #****************************** SEAFEATS MODEL ***************************************
    seafeats_model_path = os.path.join("Models","SeaFeats.pt")

    seafeats_model = models.resnet18()
    layers = list(seafeats_model.children())[:-2]
    av_pool = nn.AvgPool2d((16, 16), stride=(16, 16), padding=0)
    flatten = nn.Flatten(2, -1)
    layers.append(av_pool)
    layers.append(flatten)
    layers.append(Lambda(lambda x: torch.transpose(x, 1, 2)))
    layers.append(nn.Linear(512, 512))
    layers.append(nn.Dropout(0.15))
    layers.append(nn.Linear(512, len(class_list)))
    seafeats_model = nn.Sequential(*layers)

    seafeats_model.load_state_dict(torch.load(seafeats_model_path))

    layers_alter_1 = list(seafeats_model.children())[:8]
    layers_alter_2 = list(seafeats_model.children())[11:]

    all_layers = []
    for layer in layers_alter_1:
        all_layers.append(layer)

    all_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    all_layers.append(nn.Flatten(1))

    for layer in layers_alter_2:
        all_layers.append(layer)

    seafeats_model = nn.Sequential(*all_layers).to('cuda')

    seafeats_model.eval()

    print("Loaded the SeaFeats model successfully.")

    #****************************** CLIP MODEL ***************************************
    clip_model_path = os.path.join("Models","SeaCLIP.pt")

    clip_model = models.resnet18() 
    clip_model.fc = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.15),
                                    nn.Linear(512, 4))

    clip_model.load_state_dict(torch.load(clip_model_path))
    clip_model = clip_model.to(device)
    clip_model.eval()

    print("Loaded the SeaCLIP model successfully.")
   
    ##### Accuracy Metrics #####
    acc_metric_test = torchmetrics.Accuracy(num_classes = len(class_list), task='multiclass', multidim_average='global').to(device)

    ##### Precision Metrics #####
    per_class_prec_metric_test = torchmetrics.Precision(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    ##### Recall Metrics #####
    per_class_recall_metric_test = torchmetrics.Recall(num_classes = len(class_list), average='none', task='multiclass', multidim_average='global').to(device)
    
    ##### F1 Metrics #####
    F1_score = torchmetrics.F1Score(num_classes=len(class_list), task='multiclass', multidim_average='global').to(device)
    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_list), average='none', task='multiclass', multidim_average='global').to(device)

    dataloaders = {}
    dataloaders['test'] = test_dataloader

    phase = 'test'
    soft = torch.nn.Softmax(dim=1)

    step = 0
    preds_list = []
    labels_list = []

    print("Iterating through the test dataset...")
    for inputs, labels in tqdm(dataloaders[phase]):
        inputs = inputs.float().to(device)                  # [B, 3, H, W]
        labels = labels.to(device)                            # [B, H, W]

        # forward pass - track history if only in train
        with torch.set_grad_enabled(False):
            # ******** SeaFeats inference *******
            outputs = seafeats_model(torch.squeeze(inputs))  # [batch_size, num_classes]    
            outputs = normalize_logits(outputs)   

            # ******** SeaCLIP inference *******
            outputs_clip = clip_model(torch.squeeze(inputs))  # [batch_size, num_classes]   
            outputs_clip = normalize_logits(outputs_clip)    

            total_logits = torch.add(outputs, outputs_clip) / 2
            outputs_total = soft(total_logits) # [batch_size, num_classes]

            preds_torch = torch.argmax(outputs_total, dim=1).int()  # [1]

            labels_torch = labels.int()

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
            f1_test = F1_score(preds_torch, labels_torch)
            per_class_f1_test = F1_score_per_class(preds_torch, labels_torch)

            preds_list.append(preds_torch.cpu().numpy())
            labels_list.append(labels_torch.cpu().numpy())

    # Calculate metrics
    acc_test = acc_metric_test.compute()
    per_class_prec_test = per_class_prec_metric_test.compute()
    per_class_recall_test = per_class_recall_metric_test.compute()
    f1_test = F1_score.compute()
    per_class_f1_test = F1_score_per_class.compute()

    print("Accuracy", acc_test.item())
    print("Precision and Recall", per_class_prec_test, per_class_recall_test)
    print("F1 Scores", f1_test, per_class_f1_test)

    preds_list = np.stack(preds_list).flatten()
    labels_list = np.stack(labels_list).flatten()

    conf = confusion_matrix(np.squeeze(labels_list), np.squeeze(preds_list), labels=[0,1,2,3])
    print(conf)