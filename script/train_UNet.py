import torch
from tqdm import tqdm

import segmentation_models_pytorch as smp
from lcc.models.UNet import UNet 
from lcc.dataset import LCCDataset, get_transforms, SmallDataset
from lcc import OUTPUT_DIR
from lcc.train_utils import train

BATCH_SIZE = 8
N_EPOCHS = 30
LR = 5e-5

def main(n_sample_images=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        in_channels=4,
        out_channels=10,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = smp.losses.JaccardLoss(mode='multiclass', classes=10)
    dataset = SmallDataset(size=n_sample_images, transform=get_transforms(train=True))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model, train_dataloader, test_dataloader, optimizer, criterion, device, N_EPOCHS)

    # for epoch in range(N_EPOCHS):
    #     for _, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}', )): 
    #         optimizer.zero_grad()
    #         image, mask = batch['image'], batch['mask'].to(device)
    #         image = image.permute(0, 3, 1, 2)  
    #         # print(image.shape)
    #         output = model(image.to(device))
    #         # print(mask.dtype)
    #         # print(output.dtype)
    #         loss = criterion(output.unsqueeze(-1), mask)
    #         loss.backward()
    #         optimizer.step()
    #         pbar.set_postfix(loss=loss.item())
    #         # print(loss.item())


if __name__=="__main__":
    # dataset = LCCDataset(transform=get_transforms())
    main()