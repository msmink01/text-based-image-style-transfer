from torch import nn

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=2):
        """
        Applies channel-wise attention to a feature set
        @:param in_channels: number of channels of the input image (of shape [B, C, H, W])
        @:param reduction_ratio: used in the linear layers to squeeze/unsqueeze the features
        """
        super(ChannelAttention, self)._init_()
        self.channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(self.channels, self.channels // self.reduction_ratio, bias=False)
        self.fc2 = nn.Linear(self.channels // self.reduction_ratio, self.channels, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """"
        Applied channel attention to the specified input features.
        @:param x: input tensor of shape [B, C, H, W]
        @:return a tensor of shape [B, C, H, W] representing features to which channel attention was applied
        """
        # input shape: [B, C, H, W]
        pooled = self.pool(x)
        pooled = pooled.view(pooled.size(0), pooled.size(1))

        result = self.fc1(pooled)
        result = self.relu(result)
        result = self.fc2(result)
        result = self.relu(result)
        result = self.sig(result)

        # final shape: [B, C, H, W]
        return x * result.view(result.shape[0], result.shape[1], 1, 1).expand_as(x)