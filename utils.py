import torch.nn as nn
import torch

from torchvision.transforms import v2

emotion_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


class ConvBlock(nn.Module):

  def __init__(self, n_channels, n_filters, kernel_size=3, padding=1, max_pool_kernel_size=2):
    super().__init__()
    self.conv_block = nn.Sequential(
            nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(max_pool_kernel_size)
    )


  def forward(self, x):
      return self.conv_block(x)


class LinearBlock(nn.Module):

  def __init__(self, input_size, output_size, dropout=0.2):
      super().__init__()

      self.linear_block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )


  def forward(self, x):
      return self.linear_block(x)
  


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # 48x48 → 24x24 → 12x12 → 6x6
        self.conv_block0 = ConvBlock(1, 32)
        self.conv_block1 = ConvBlock(32, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)

        self.flatten = nn.Flatten()

        self.fc_block0 = LinearBlock(256*3*3, 512, dropout=0.40)
        self.fc_block1 = LinearBlock(512, 256, dropout=0.20)
        self.fc_block2 = LinearBlock(256, 128, dropout=0.10)
        self.fc_block3 = LinearBlock(128, 64, dropout=0.5)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block0(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc_block0(x)
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.fc_block3(x)
        return self.out(x)
    

def load_checkpoint(checkpoint_path, model, device):
    '''
    Loads checkpoint into model
    
    :param checkpoint_path: path to stored checkpoint
    :param model: model instance 
    :param device: PyTorch device

    Returns: 
    model: Loaded model
    tuple : mean and std used while training 
    '''

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    mean, std = checkpoint['mean_std']
    return model, (mean, std)


def transform(image, mean, std):
    return v2.Compose([
        v2.Resize((48, 48)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std]),
    ])(image)

