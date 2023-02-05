import numpy as np
import torch

import segmentation_models_pytorch as smp
from lcc.models.segformer import SegFormer
from lcc.dataset import LCCDataset, get_transforms, SmallDataset, TRAIN_CLASS_COUNTS, N_CLASSES, IGNORED_CLASSES_IDX
from lcc import OUTPUT_DIR
from lcc.train_utils import train

N_EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 2


def main(n_sample_images: int = 10000):
    dataset = SmallDataset(size=n_sample_images, transform=get_transforms(train=True))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegFormer(
        in_channels=4,
        widths=[64, 128, 256, 512],
        depths=[3, 4, 6, 3],
        all_num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        overlap_sizes=[4, 2, 2, 2],
        reduction_ratios=[8, 4, 2, 1],
        mlp_expansions=[4, 4, 4, 4],
        decoder_channels=256,
        scale_factors=[8, 4, 2, 1],
        num_classes=10,
    ).to(device)
    model.name = "Segmenter_classic"
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0)
    class_weight = np.ones(N_CLASSES)
    class_weight[2:] = (1 / TRAIN_CLASS_COUNTS[2:])* TRAIN_CLASS_COUNTS[2:].sum() / (N_CLASSES-2)
    class_weight[IGNORED_CLASSES_IDX] = 0
    class_weight = torch.tensor(class_weight).to(device).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    criterion = smp.losses.JaccardLoss(mode='multiclass', classes=10)
    train(model, train_dataloader, test_dataloader, optimizer, criterion, device, N_EPOCHS)

if __name__=="__main__":
    main()