import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn

num_epochs = 2
num_classes = 10
batch_size = 100
learning_rate = 1e-4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                 (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             transform=trans,
                                             train=True,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            transform=trans,
                                            train=False,
                                            download=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = resnet18(pretrained=True)

#for param in model.parameters():
#    param.requires_grad = False
    
model.fc = nn.Linear(model.fc.in_features, 10)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
            
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
    
    print('Accuracy on the CIFAR10 dataset: {:.2f} %'.format(correct/total*100))