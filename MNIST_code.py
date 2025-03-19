"""
deep learning project
classification model using MINIST dataset
step:
0. library
1. get data(prepare and load)
2. build model
3. train model
4. make predition and evaluate model
5. save and load model

"""
# 0 library
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import os


# 1 get data
def get_data(batch_size, num_workers) -> DataLoader:
    # get data
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True)
    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor())
    
    # dataset -> dataloader
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
        # pin_memory=True # GPU
        )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
    
    return train_dataloader, test_dataloader


# build model
# opt1
class cnn(nn.Module):
    def __init__(self, input, hidden_units, output):
        super().__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784,
                      out_features=output))
    
    def forward(self, x):
        x = self.cnn_layer1(x)
        #print(x.shape)
        x = self.cnn_layer2(x)
        #print(x.shape)
        x = self.output(x)
        #print(x.shape)
        return x
    
# 训练函数
def train(model, train_dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        y_label = y_logits.argmax(dim=1)
        train_acc += (y_label == y).sum().item() / len(y_logits)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    return train_loss, train_acc


# 测试函数
def test(model, test_dataloader, loss_fn, device):
    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            test_logits = model(X)

            loss = loss_fn(test_logits, y)
            test_loss += loss.item()
            test_label = test_logits.argmax(dim=1)
            test_acc += (test_label == y).sum().item() / len(test_logits)

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    return test_loss, test_acc

# 训练+测试
def train_test_model(model, train_dataloader, test_dataloader, epochs, loss_fn, optimizer, device):
    results = {
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[]
    }
    train_loss, train_acc, test_loss, test_acc = 0,0,0,0

    for epoch in tqdm(range(epochs)):
        # train
        train_loss, train_acc = train(model, train_dataloader, loss_fn, optimizer, device)

        # test
        test_loss, test_acc = test(model, test_dataloader, loss_fn, device)
        print(f'epoch: {epoch} | train loss: {train_loss:.4f} | train acc: {train_acc*100:.3f}% | test loss: {test_loss:.4f} | test acc: {test_acc*100:.3f}%')
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results

# save
def save_model(model, dir, model_name):
    path = Path(dir)
    path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pt') or model_name.endswith('.pth'), "model name must ends with .pt/.pth"
    model_path = path/model_name

    print('saving model')
    torch.save(obj=model.state_dict(), f=model_path)
    return model_path


# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparams
BATCH_SIZE = 32
NUM_WORKERS = 0 # os.cpu_count()
EPOCHS = 3

# get data
train_dataloader, test_dataloader = get_data(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
print(train_dataloader.dataset)

# 创建model实例
cnnmodel = cnn(1,16, len(train_dataloader.dataset.classes))
data_batch, label_batch = next(iter(train_dataloader))
print(data_batch.shape)
#dummy = torch.rand(1,1,28,28)
#print(cnnmodel(dummy))

# model信息
print(summary(cnnmodel, data_batch.shape))

# loss fn and optimzer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cnnmodel.parameters(),lr=0.001)

# train and test model
print('training')
results = train_test_model(cnnmodel, train_dataloader, test_dataloader,
                           EPOCHS, loss_fn, optimizer, device)
print(results)

# save model
dir = 'models'
model_name = 'MNIST_model0.pth'
model_path = save_model(cnnmodel, dir, model_name)

# load model
loaded_model = cnn(1, 16, len(train_dataloader.dataset.classes))
loaded_model.load_state_dict(torch.load(f=model_path))

# eval model
loaded_loss, loaded_acc = test(loaded_model, test_dataloader, loss_fn, device)
print(f'loaded model loss: {loaded_loss} | loaded model acc: {loaded_acc}')

# check model results
result_same = torch.isclose(torch.tensor(results['test_loss'][EPOCHS-1]),
                            torch.tensor(loaded_loss),
                            atol=1e-2)
print(result_same)

