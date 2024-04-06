from models import Model
from utils import dataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.nn import BCELoss
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print("Loading data")
ds = dataset("train_balanced.parquet")
dataloader = DataLoader(ds, batch_size=5096, shuffle=True, collate_fn=collate_fn, num_workers=20)
print("Creating model")
print(len(dataloader))
model = Model(4, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = BCELoss()
li_total = []
for epoch in range(10):
    model.train()
    li_loss = []
    for i, (graphe, protein) in enumerate(dataloader):
        print(i)
        protein = protein.to(device)
        graphe.x = graphe.x.to(device)
        graphe.edge_index = graphe.edge_index.to(device)
        graphe.edge_attr = graphe.edge_attr.to(device)
        graphe.y = graphe.y.to(device)
        optimizer.zero_grad()
        y_hat = model(graphe, protein)
        y_hat = y_hat.squeeze(1)
        loss = loss_fn(y_hat, graphe.y.float())
        loss.backward()
        optimizer.step()
        li_loss.append(loss.item())    
    print(f"Epoch: {epoch} Loss: {np.mean(li_loss)}")
    li_total.append(np.mean(li_loss))
    torch.save(model.state_dict(), "model.pth")
with open('test.npy', 'wb') as f:
    np.save(f, np.array(li_total))