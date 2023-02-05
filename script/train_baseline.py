import torch
from tqdm import tqdm

from lcc.models.baseline import Baseline 
from lcc.dataset import LCCDataset, get_transforms
from lcc import OUTPUT_DIR

BATCH_SIZE = 8
N_EPOCHS = 90
LR = 1e-3


def main(dataset: LCCDataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline(
        input_shape=(4,256,256),
        n_classes=10, 
        output_activation='softmax',
        n_layers=4,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    all_losses = []
    for epoch in range(N_EPOCHS):
        for _, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}', )): 
            optimizer.zero_grad()
            image, mask = batch['image'], batch['mask'].to(device)
            image = image.permute(0, 3, 1, 2)  
            # print(image.shape)
            output = model(image.to(device))
            # print(mask.dtype)
            # print(output.dtype)
            loss = criterion(output.unsqueeze(-1), mask)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            # print(loss.item())


if __name__=="__main__":
    dataset = LCCDataset(transform=get_transforms())
    main(dataset)