from torch import nn


class AutoFocus(nn.Module):
    '''
    The focus-branch of the AutoFocus model.

    See https://arxiv.org/pdf/1812.01600.pdf for more details.
    '''

    def __init__(self,
                 in_channels):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=256,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        self.conv_1_relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=(1, 1))
        self.conv_2_relu = nn.ReLU()

        self.conv_3 = nn.Conv2d(in_channels=256,
                                out_channels=2,
                                kernel_size=(1, 1))

    def forward(self, data):
        out = self.conv_1_relu(self.conv_1(data))
        out = self.conv_2_relu(self.conv_2(out))
        out = self.conv_3(out)

        return out
