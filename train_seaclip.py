# Import packages
import os, time
from torchvision.models import resnet18
import torch
import torch.nn as nn
from seagrass_dataloaders import loadDataValSplit, loadTestSetOrig
import torch.optim as optim
from tqdm import tqdm
import wandb, torchmetrics

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_path = "DeepSeagrass"
    class_list = ["Background","Ferny","Rounded","Strappy"]

    epochs = 10000
    learning_rate = 0.0001
    batch_size = 24
    augment = True

    train_dataloader, val_dataloader, _ = loadDataValSplit(class_list, os.path.join(dataset_path, "ClipLabels"), os.path.join(dataset_path, "Test"), batch_size, efficientnet=False, augment=augment, num_workers=1)
    test_dataloader = loadTestSetOrig(class_list, os.path.join(dataset_path, "Test"), batch_size=1, num_workers=2, efficientnet=False)
    
    ###### Resnet-18 #####
    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(512, len(class_list)))
    model.train()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    weights = torch.tensor([1.0,1.5,1.2,1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr = 0.00003)

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

    F1_score_per_class = torchmetrics.F1Score(num_classes=len(class_list), average='none', mdmc_average='global').to(device)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader
    dataloaders['test'] = test_dataloader

    model_name = f"model-{int(time.time())}"
    print("################################# MODEL NAME: ########################################")
    print(model_name)
    wandb.config.model_name = model_name

    soft = torch.nn.Softmax(dim=1)

    best_acc = -1.0
    min_loss = 1000

    for epoch in range(epochs):

        metrics = {}
        test_metrics = {}

        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        ####################### TRAINING PHASE #########################
        phase = 'train'
        print("*** Phase: "+phase+" ***")

        model.train()
        running_loss_train = 0.0       

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 3, H, W]
            labels = labels.to(device)                            # [B, H, W]

            # forward pass - track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                outputs = soft(outputs)
                preds_torch = torch.argmax(outputs.detach(), dim=1).int()
                labels_torch = labels.int().detach()

                acc_train = acc_metric_train(preds_torch, labels_torch)
                per_class_prec_train = per_class_prec_metric_train(preds_torch, labels_torch)
                per_class_recall_train = per_class_recall_metric_train(preds_torch, labels_torch)

                loss.backward()
                optimizer.step() 
            
            running_loss_train = running_loss_train + loss.detach()

        acc_train = acc_metric_train.compute()
        per_class_prec_train = per_class_prec_metric_train.compute()
        per_class_recall_train = per_class_recall_metric_train.compute()

        wandb.log({"prec_background_train":per_class_prec_train[0], 
            "prec_ferny_train":per_class_prec_train[1],
            "prec_rounded_train":per_class_prec_train[2],
            "prec_strappy_train":per_class_prec_train[3]}, step=epoch)

        wandb.log({"recall_background_train":per_class_recall_train[0], 
            "recall_ferny_train":per_class_recall_train[1],
            "recall_rounded_train":per_class_recall_train[2],
            "recall_strappy_train":per_class_recall_train[3]}, step=epoch)

        train_loss = running_loss_train / len(dataloaders[phase])

        metrics[phase+'_loss'] = train_loss.item()
        metrics[phase+'_pa'] = acc_train.item()

        ####################### VALIDATION PHASE #########################
        phase = 'val'
        print("*** Phase: "+phase+" ***")

        model.eval()
        running_loss_val = 0.0

        for inputs, labels in tqdm(dataloaders[phase]):
            optimizer.zero_grad()  

            inputs = inputs.float().to(device)                  # [B, 3, H, W]
            labels = labels.to(device)                            # [B, H, W]

            # forward pass - track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                outputs = soft(outputs)
                preds_torch = torch.argmax(outputs.detach(), dim=1).int()
                labels_torch = labels.int().detach()

                acc_val = acc_metric_val(preds_torch, labels_torch)
                per_class_prec_val = per_class_prec_metric_val(preds_torch, labels_torch)
                per_class_recall_val = per_class_recall_metric_val(preds_torch, labels_torch)

                per_class_f1_val = F1_score_per_class(preds_torch, labels_torch)
            
            running_loss_val = running_loss_val + loss.detach()

        val_loss = running_loss_val / len(dataloaders[phase])

        acc_val = acc_metric_val.compute()
        per_class_prec_val = per_class_prec_metric_val.compute()
        per_class_recall_val = per_class_recall_metric_val.compute()

        per_class_f1_val = F1_score_per_class.compute()
           

        wandb.log({"prec_background_val":per_class_prec_val[0], 
            "prec_ferny_val":per_class_prec_val[1],
            "prec_rounded_val":per_class_prec_val[2],
            "prec_strappy_val":per_class_prec_val[3]}, step=epoch)

        wandb.log({"recall_background_val":per_class_recall_val[0], 
            "recall_ferny_val":per_class_recall_val[1],
            "recall_rounded_val":per_class_recall_val[2],
            "recall_strappy_val":per_class_recall_val[3]}, step=epoch)

        metrics[phase+'_loss'] = val_loss.item()
        metrics[phase+'_pa'] = acc_val.item()

        scheduler.step(val_loss) 
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
        wandb.log({"lr":curr_lr}, step=epoch)

        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), str(model_name)+'CKPT.pt')

        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), str(model_name)+'_loss_CKPT.pt')

        acc_metric_train.reset()
        per_class_prec_metric_train.reset()
        per_class_recall_metric_train.reset()

        acc_metric_val.reset()
        per_class_prec_metric_val.reset()
        per_class_recall_metric_val.reset()

        F1_score_per_class.reset()

        wandb.log(metrics, step=epoch)

    # Save the final model weights
    torch.save(model.state_dict(), str(model_name)+'_FINAL.pt')

    ####################### TESTING PHASE #########################
    phase = 'test'
    print("*** Phase: "+phase+" ***")
    model.eval()

    for inputs, labels in tqdm(dataloaders[phase]):
        optimizer.zero_grad()  

        inputs = inputs.float().to(device)                  # [B, 3, H, W]
        labels = labels.to(device)                            # [B, H, W]

        # forward pass - track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = soft(outputs)
            preds_torch = torch.argmax(outputs, dim=1).int()
            labels_torch = labels.int()

            acc_test = acc_metric_test(preds_torch, labels_torch)
            per_class_prec_test = per_class_prec_metric_test(preds_torch, labels_torch)
            per_class_recall_test = per_class_recall_metric_test(preds_torch, labels_torch)
        
    acc_test = acc_metric_test.compute()
    per_class_prec_test = per_class_prec_metric_test.compute()
    per_class_recall_test = per_class_recall_metric_test.compute()

    wandb.run.summary["prec_background_test"] = per_class_prec_test[0]
    wandb.run.summary["prec_ferny_test"] = per_class_prec_test[1]
    wandb.run.summary["prec_rounded_test"] = per_class_prec_test[2]
    wandb.run.summary["prec_strappy_test"] = per_class_prec_test[3]

    wandb.run.summary["recall_background_test"] = per_class_recall_test[0]
    wandb.run.summary["recall_ferny_test"] = per_class_recall_test[1]
    wandb.run.summary["recall_rounded_test"] = per_class_recall_test[2]
    wandb.run.summary["recall_strappy_test"] = per_class_recall_test[3]

    wandb.run.summary["acc_test"] = acc_test.item()

    print(per_class_prec_test, per_class_recall_test, acc_test)