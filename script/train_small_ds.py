import numpy as np
import torch

from lcc.dataset import SmallDataset, get_transforms, TRAIN_CLASS_COUNTS, N_CLASSES, IGNORED_CLASSES_IDX
from lcc.models.baseline import Baseline
from lcc import OUTPUT_DIR, MODEL_DIR
from lcc.train_utils import train

LR = 1e-3
N_EPOCH = 30
BATCH_SIZE = 8

def main(n_sample_images: int = 100):
    dataset = SmallDataset(size=n_sample_images, transform=get_transforms(train=True))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline(
        input_shape=(4,256,256),
        n_classes=10, 
        output_activation='softmax',
        n_layers=2,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    class_weight = np.ones(N_CLASSES)
    class_weight[2:] = (1 / TRAIN_CLASS_COUNTS[2:])* TRAIN_CLASS_COUNTS[2:].sum() / (N_CLASSES-2)
    class_weight[IGNORED_CLASSES_IDX] = 0
    class_weight = torch.tensor(class_weight).to(device).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    all_losses_train = []
    all_losses_test = []
    train(model, train_dataloader, test_dataloader, optimizer, criterion, device, N_EPOCH)
    
    # np.save(OUTPUT_DIR / 'train_losses.npy', np.array(all_losses_train))
    # np.save(OUTPUT_DIR / 'test_losses.npy', np.array(all_losses_test))
    # torch.save(model.state_dict(), MODEL_DIR / 'model_baseline.pth')


if __name__ == "__main__":
    main()
