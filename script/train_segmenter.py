from transformers import ViTConfig
import torch
from tqdm import tqdm

from lcc.models.segmenter import Segmenter
from lcc.dataset import LCCDataset, get_transforms
from lcc import OUTPUT_DIR

N_EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 8


def main(dataset: LCCDataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_config = ViTConfig()
    encoder_config.image_size = 256
    encoder_config.num_channels = 4
    segmenter = Segmenter(encoder_config, device).to(device)
    optimizer = torch.optim.SGD(segmenter.parameters(), lr=LR, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    all_losses = []
    for epoch in range(N_EPOCHS):
        for _, batch in (pbar:=tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch}', )): 
            optimizer.zero_grad()
            image, mask = batch['image'], batch['mask'].to(torch.float32).to(device)
            image = image.permute(0, 3, 1, 2)  
            # print(image.shape)
            output = segmenter(image)
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