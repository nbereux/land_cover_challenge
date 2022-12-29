import numpy as np
import torch
from tqdm import tqdm

from lcc.dataset import SmallDataset, get_transforms, TRAIN_CLASS_COUNTS, N_CLASSES, IGNORED_CLASSES_IDX
from lcc.models.VAE import ConvVAE
from lcc import OUTPUT_DIR, MODEL_DIR

LR = 1e-3
N_EPOCH = 30
BATCH_SIZE = 8

def train_one_epoch_VAE(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_losses = []
    for i, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader))):
        optimizer.zero_grad()
        image, mask = batch['image'], batch['mask'].to(device)
        image = image.permute(0, 3, 1, 2)
        output, mu, logVar =  model(image.to(device))
        kl_div  = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = criterion(output, mask.squeeze()) + kl_div
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        print(loss.item())
    epoch_loss = np.mean(epoch_losses)
    print(f'Epoch loss TRAIN: {epoch_loss}')
    return epoch_loss

def test_one_epoch_VAE(model, dataloader, criterion, device):
    epoch_losses = []
    with torch.no_grad():
        model.eval()
        for i, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader))):
            image, mask = batch['image'], batch['mask'].to(device)
            image = image.permute(0, 3, 1, 2)
            output, _, _ = model(image.to(device))
            loss = criterion(output, mask.squeeze())
            epoch_losses.append(loss.item())
    epoch_loss = np.mean(epoch_losses)
    print(f'Epoch loss TEST: {epoch_loss}')
    return epoch_loss

def main(n_sample_images: int = 1000):
    dataset = SmallDataset(size=n_sample_images, transform=get_transforms(train=True))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = ConvVAE(
        input_shape=(4,256,256),
        n_classes=10, 
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    class_weight = (1 / TRAIN_CLASS_COUNTS[2:])* TRAIN_CLASS_COUNTS[2:].sum() / (N_CLASSES-2)
    class_weight[IGNORED_CLASSES_IDX] = 0
    criterion = torch.nn.CrossEntropyLoss(class_weight=class_weight)
    all_losses_train = []
    all_losses_test = []
    for epoch in range(N_EPOCH):
        all_losses_train.append(train_one_epoch_VAE(model, train_dataloader, optimizer, criterion, device))
        torch.cuda.empty_cache()
        all_losses_test.append(test_one_epoch_VAE(model, test_dataloader, criterion, device))
        torch.cuda.empty_cache()
    np.save(OUTPUT_DIR / 'train_losses.npy', np.array(all_losses_train))
    np.save(OUTPUT_DIR / 'test_losses.npy', np.array(all_losses_test))
    torch.save(model.state_dict(), MODEL_DIR / 'model_baseline.pth')


if __name__ == "__main__":
    main()