import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = self.makeConv(4, 32, 8, 4)
        self.conv2 = self.makeConv(32, 64, 4, 2)
        self.conv3 = self.makeConv(64, 64, 3, 1)

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def makeConv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output