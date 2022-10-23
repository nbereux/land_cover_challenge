import torch.nn as nn
import torch

CONV2D_PARAMS = {
    'kernel_size': (3,3),
    'stride': (1,1),
    'padding': 'same',
    #'kernel_initializer': 'he_normal'
}

CONV2D_TRANSPOSE_PARAMS = {
    'kernel_size': (3,3),
    'stride': (2,2),
    'padding': (1,1),
    'output_padding': (1,1)
}

MAXPOOL2D_PARAMS = {
    'kernel_size': (2,2),
    'stride': (2,2),
    'padding': 0,
}

FILTERS_DOWN = 64
FILTERS_UP = 96

class Baseline(nn.Module):

    def __init__(
        self,
        input_shape,
        n_classes,
        output_activation,
        n_layers=4):
        super().__init__()
        self.pre_down = []
        self.pre_down.append(nn.Conv2d(input_shape[0], FILTERS_DOWN, **CONV2D_PARAMS))
        self.pre_down.append(ConvBlock(FILTERS_DOWN, FILTERS_DOWN, **CONV2D_PARAMS))
        self.pre_down.append(123)
        self.pre_down.append(ConvBlock(FILTERS_DOWN, FILTERS_DOWN, **CONV2D_PARAMS))
        self.pre_down.append(nn.MaxPool2d(**MAXPOOL2D_PARAMS))
        self.down = []

        for _ in range(n_layers):
            self.down.append(ConvBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_PARAMS))
            self.down.append(ConvBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_PARAMS))
            self.down.append(123)
            self.down.append(ConvBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_PARAMS))
            self.down.append(nn.MaxPool2d(**MAXPOOL2D_PARAMS))

        self.transition=nn.Sequential(
            ConvBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_PARAMS),
            ConvBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_PARAMS),
            ConvTransposeBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_TRANSPOSE_PARAMS)
        )


        self.up = []
        for _ in range(n_layers):
            self.up.append(
                nn.Sequential(
                    ConvBlock(in_channels=2*FILTERS_DOWN, out_channels=FILTERS_UP, **CONV2D_PARAMS),
                    ConvBlock(in_channels=FILTERS_UP, out_channels=FILTERS_DOWN, **CONV2D_PARAMS),
                    ConvTransposeBlock(in_channels=FILTERS_DOWN, out_channels=FILTERS_DOWN, **CONV2D_TRANSPOSE_PARAMS)
                )
            )
        
        self.post_up = nn.Sequential(
            ConvBlock(in_channels=2*FILTERS_DOWN, out_channels=FILTERS_UP, **CONV2D_PARAMS),
            ConvBlock(in_channels=FILTERS_UP, out_channels=FILTERS_DOWN, **CONV2D_PARAMS),
            nn.Conv2d(FILTERS_DOWN, n_classes, kernel_size=(1,1), padding='same'),
            nn.Sigmoid(),
        )


    def forward(self, x):
        save_down_x = []
        for layer in self.pre_down:
            if layer == 123:
                save_down_x.append(x)
            else:
                x = layer(x)
        for layer in self.down:
            if layer==123:
                save_down_x.append(x)
            else:
                x = layer(x)
        save_down_x = list(reversed(save_down_x))
        x = self.transition(x)
        for i, layer in enumerate(self.up):
            x = torch.cat([x, save_down_x[i]], dim=1)
            x = layer(x)
        x = torch.cat([x, save_down_x[-1]], dim=1)
        x = self.post_up(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                nn.ReLU(),
            )
    
        def forward(self, x):
            return self.block(x)

if __name__=="__main__":
    model = Baseline((4,256,256), 10, 'softmax')
    print(model)
    input_batch = torch.normal(0,1,(1, 4, 256, 256))
    output = model(input_batch)
    print(output.shape)