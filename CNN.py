from torch import nn

# CNN implementation
# Works on 32 x 32 images
class CNN(nn.Module):
    def __init__(self, output_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 80)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(80, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x