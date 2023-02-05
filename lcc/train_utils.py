import mlflow
from mlflow import log_metric, log_param, log_artifacts
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from tqdm import tqdm
from lcc.dataset import N_CLASSES
from lcc import MODEL_DIR

import pdb

EPS_REG_KL_DIV = 1e-8   

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_losses = []
    for i, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader))):
        optimizer.zero_grad()
        image, mask = batch['image'], batch['mask'].to(device)
        #print(image.shape)
        if "Segmenter" in model.name:
            image = image.permute(0, 3, 1, 2)
        image = image.permute(0, 3, 1, 2)
        output = model(image.to(device))
        loss = criterion(output, mask.squeeze())
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    epoch_loss = np.mean(epoch_losses)
    print(f'Mean train epoch loss: {epoch_loss}')
    return epoch_loss

def kl_divergence(p, q):
    return torch.sum((p+EPS_REG_KL_DIV) * torch.log((p + EPS_REG_KL_DIV) / (q + EPS_REG_KL_DIV)))

def test_one_epoch(model, dataloader, criterion, device):
    jac_idx = JaccardIndex(task='multiclass', num_classes=N_CLASSES).to(device)
    epoch_losses = []
    epoch_jac_idx = []
    epoch_kl_div = []
    with torch.no_grad():
        model.eval()
        for _, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader))):
            image, mask = batch['image'], batch['mask'].to(device)
            image = image.permute(0, 3, 1, 2)
            if "Segmenter" in model.name:
                image = image.permute(0, 3, 1, 2)
            output = model(image.to(device))
            loss = criterion(output, mask.squeeze())
            epoch_losses.append(loss.item())

            # Jaccard Index = Intersection over Union
            epoch_jac_idx.append(jac_idx(output, mask.squeeze()).item())

            # KL Divergence
            tmp_out = output.argmax(dim=1)
            tmp_kl_div = []
            for i in range(mask.shape[0]):
                distrib_mask = torch.zeros(N_CLASSES).to(device)
            
                tmp_class, tmp_n = torch.unique(mask[i], return_counts=True)
                distrib_mask[tmp_class] = tmp_n.float()
                distrib_mask = distrib_mask[2:]
                distrib_mask = distrib_mask / distrib_mask.sum()

                distrib_out = torch.zeros(N_CLASSES).to(device)

                tmp_class, tmp_n = torch.unique(tmp_out[i], return_counts=True)
                distrib_out[tmp_class] = tmp_n.float()
                distrib_out = distrib_out[2:]
                if not(distrib_out.sum() == 0):
                    distrib_out = distrib_out / distrib_out.sum()

                tmp_kl_div.append(kl_divergence(distrib_mask, distrib_out).item())
            
            # pdb.set_trace()
            epoch_kl_div.append(np.mean(tmp_kl_div))

    epoch_loss = np.mean(epoch_losses)
    epoch_jac_idx = np.mean(epoch_jac_idx)
    epoch_kl_div = np.mean(epoch_kl_div)
    print(f'Mean test Jaccard Index: {epoch_jac_idx}')
    print(f'Mean test KL Divergence: {epoch_kl_div}')
    print(f'Mean test loss: {epoch_loss}')
    return epoch_loss, epoch_jac_idx, epoch_kl_div


def train(model, train_dataloader, test_dataloader, optimizer, criterion, device, n_epochs, patience=10):    
    with mlflow.start_run():
        log_param('batch_size', train_dataloader.batch_size)
        log_param('lr', optimizer.param_groups[0]['lr'])
        best_loss = 100
        cpt = 0
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
            torch.cuda.empty_cache()
            test_loss, test_jac_idx, test_kl_div = test_one_epoch(model, test_dataloader, criterion, device)
            torch.cuda.empty_cache()
            log_metric('train_loss', train_loss, step=epoch)
            log_metric('test_loss', test_loss, step=epoch)
            log_metric('test_jac_idx', test_jac_idx, step=epoch)
            log_metric('test_kl_div', test_kl_div, step=epoch)
            if test_loss<best_loss:
                best_loss = test_loss
                cpt=0
                torch.save(model.state_dict(), MODEL_DIR.joinpath(f'{model.name}.pth'))
            else:
                cpt+=1
                if cpt>=patience:
                    print(f'Best Test Loss : {best_loss}')
                    return
            