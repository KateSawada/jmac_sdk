import torch
from torch import optim, nn

class CNN_MLP(nn.Module):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=544):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 8,kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(8)

        self.conv4 = nn.Conv1d(8, 8,kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(8)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(544*8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

        self.loss_module = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        #self.logger.summary.scalar('loss', loss, step=self.global_step)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

    def forward(self,x):
        x = x.float()
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

    def predict(self, x):
        x = torch.unsqueeze(x, dim=0)
        x = self.forward(x)
        x = self.softmax(x)
        return x
    
    def predict_rf(self, x):
        x = torch.unsqueeze(x, dim=0)
        x = self.forward(x)
        return x