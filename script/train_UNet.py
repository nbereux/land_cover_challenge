import torch
from tqdm import tqdm

import segmentation_models_pytorch as smp
from lcc.models.UNet import UNet 
from lcc.dataset import LCCDataset, get_transforms, SmallDataset, get_transforms_2
from lcc import OUTPUT_DIR
from lcc.train_utils import train

from lcc.dataset import TRAIN_CLASS_COUNTS

CLASS_WEIGHTS = torch.tensor(1/TRAIN_CLASS_COUNTS)
CLASS_WEIGHTS[:2] = 0
print(CLASS_WEIGHTS)

BATCH_SIZE = 8
N_EPOCHS = 1000
LR = 5e-5
PATIENCE = 3

def main(n_sample_images=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        in_channels=4,
        out_channels=10,
    ).to(device)
    model.name = "UNet_cross_entropy_loss_weighted"
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS.to(device).float())
    # criterion = smp.losses.JaccardLoss(mode='multiclass', classes=list(range(2,10)), from_logits=True)
    dataset = SmallDataset(size=n_sample_images, transform=get_transforms())
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model, train_dataloader, test_dataloader, optimizer, criterion, device, N_EPOCHS, patience=PATIENCE)


if __name__=="__main__":
    # dataset = LCCDataset(transform=get_transforms())
    main()