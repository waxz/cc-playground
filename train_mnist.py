"""MNIST CNN training script, meant to be pushed to Colab with colab_run / colab_job_start.

Usage: python -u train_mnist.py [--epochs 3] [--out /content/out]
Prints one line per epoch and a final `RESULT {...}` JSON line so output is easy to parse.
"""
import argparse, json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.d = nn.Dropout(0.25)
        self.f1 = nn.Linear(9216, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = self.d(torch.flatten(x, 1))
        return self.f2(F.relu(self.f1(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="/content/out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf),
                       batch_size=a.batch_size, shuffle=True)
    test = DataLoader(datasets.MNIST("data", train=False, transform=tf), batch_size=1000)

    model = CNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    history, t0 = [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        model.eval(); ok = 0
        with torch.no_grad():
            for x, y in test:
                ok += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        acc = ok / len(test.dataset)
        history.append(acc)
        print(f"epoch {epoch}: test acc {acc:.4f}", flush=True)

    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    result = {"device": dev, "epochs": a.epochs, "test_acc": history, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(result, f)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
