import torch
from torchvision import datasets, transforms
import sys
sys.path.insert(0, '../answers')
from torch import nn, optim
import torch.nn.functional as F

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(
            x.shape[0], -1
            )#flat
        
        x = F.relu(
            self.fc1(x)
            )
        
        x = F.relu(
            self.fc2(x)
            )
        
        x = F.relu(
            self.fc3(x)
            )
        
        x = F.log_softmax(
            self.fc4(x), dim=1)
        
        return x
    
model=Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)