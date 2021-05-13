import torch
import numpy
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

import ptn_parser


def result_to_array(result):
    if ["1-0", "R-0", "F-0"].__contains__(result):
        return 1
    elif ["0-1", "0-R", "0-F"].__contains__(result):
        return 0
    elif result == "1/2":
        return -1
    else:
        raise Exception(f'Invalid result {result}')


def main():
    positions = ptn_parser.main("games0_6s_all.ptn")

    inputs = map(lambda a: ptn_parser.extract_features(a[0]), positions)
    targets = map(lambda a: result_to_array(a[1]), positions)

    tensor_x = torch.Tensor(list(inputs))
    tensor_y = torch.Tensor(list(targets))

    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using {} device'.format(device))

    model = NeuralNetwork().to(device)
    print(model)

    learning_rate = 0.001

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loop(data_loader, model, loss_fn, optimizer)

    for t in range(20):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(data_loader, model, loss_fn, optimizer)
        test_loop(data_loader, model, loss_fn)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(36 * 6 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def run():
    X = torch.rand(1, 36 * 6 + 1, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    print(f"Predicted class: {pred_probab}, logits: {logits}")

    print("Model structure: ", model, "\n\n")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
