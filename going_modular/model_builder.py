from torch import nn


class TinyVGG(nn.Module):
    def __init__(self, input_size, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, out_channels=hidden_units, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, out_channels=hidden_units, kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, out_channels=hidden_units, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units
                * 13
                * 13,  # get this by printing the shape of the outputs of each layer
                out_features=hidden_units,
            ),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block(self.conv_block(x)))
