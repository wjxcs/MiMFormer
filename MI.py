import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, in_channels_x, in_channels_y, inter_channels):
        super(MINE, self).__init__()

        self.ma_et = None

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels_x, max(in_channels_x // 2, inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_x // 2, inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(max(in_channels_x // 2, inter_channels), max(in_channels_x // 4, inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_x // 4, inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(max(in_channels_x // 4, inter_channels), inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.AvgPool2d(2, 1),
            nn.Flatten()
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(in_channels_y, max(in_channels_y // 2, inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_y // 2, inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(max(in_channels_y // 2, inter_channels), max(in_channels_y // 4, inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_y // 4, inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(max(in_channels_y // 4, inter_channels), inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.AvgPool2d(2, 1),
            nn.Flatten()
        )
        # Assuming the output size of conv_x and conv_y is the same
        conv_output_size = 4 * inter_channels  # Based on input size 13x13 and conv/pool layers
        # conv_output_size = 4 # 9 11 15
#         conv_output_size = 36 # 17 19
        # conv_output_size = inter_channels  # Based on input size 13x13 and conv/pool layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, 1)
        )

    def forward(self, x, y):
        x_conv = self.conv_x(x)
        y_conv = self.conv_y(y)
#         print(" self.conv_x(x)  {}".format(x_conv.shape))
#         print(" self.conv_y(y)  {}".format(y_conv.shape))

        combined = x_conv + y_conv
#         print(" combined  {}".format(combined.shape))
        return self.fc(combined)

if __name__ == "__main__":
    b = 15
    x = torch.randn(8, 32, b, b)
    y = torch.randn(8, 32, b, b)
    model = MINE(32, 32, 4)
    # optimizer_mine = optim.Adam(MINE.parameters(), lr=1e-4)
    ma_rate = 0.001
    output = model(x, y)
    print(output.shape)
