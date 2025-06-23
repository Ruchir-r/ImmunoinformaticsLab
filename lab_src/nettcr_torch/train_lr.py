import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def compute_auc(true, pred, max_fpr=1.0):
    return roc_auc_score(true, pred, max_fpr=max_fpr)

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: DataLoader,
    device: torch.device,
    valid_loss: float,
) -> dict:
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    true, pred = [], []

    for batch in tqdm(dataloader, disable=True):
        batch = [b.to(device, non_blocking=True) for b in batch]
        optimizer.zero_grad(set_to_none=True)

        output = model(*batch[:-2])
        loss = torch.mean(criterion(output, batch[-2]) * batch[-1])

        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
        true.extend(batch[-2].detach().cpu())
        pred.extend(output.detach().cpu())

    scheduler.step()
    return true, pred, total_loss.item() / len(dataloader)


def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    true, pred = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=True):
            batch = [b.to(device, non_blocking=True) for b in batch]
            output = model(*batch[:-2])
            loss = torch.mean(criterion(output, batch[-2]))

            total_loss += loss.detach()
            true.extend(batch[-2].detach().cpu())
            pred.extend(output.detach().cpu())

    return true, pred, total_loss.item() / len(dataloader)


def infer(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=True):
            batch = [b.to(device, non_blocking=True) for b in batch]
            output = model(*batch)

            pred.extend(output.detach().cpu())

    return pred


def train_one_fold(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    patience: int
) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    best_auc_01 = 0.0
    epochs_without_improvement = 0
    valid_loss = 0.5

    for e in range(epochs):
        _, _, train_loss = train(
            model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, dataloader=train_loader, device=device, valid_loss=valid_loss,
        )
        valid_true, valid_pred, valid_loss = evaluate(
            model=model, criterion=criterion, dataloader=valid_loader, device=device,
        )

        valid_auc = compute_auc(valid_true, valid_pred)
        valid_auc_01 = compute_auc(valid_true, valid_pred, max_fpr=0.1)

        # Check for early stopping
        if valid_auc_01 > best_auc_01:
            best_auc_01 = valid_auc_01
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {e} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}, Valid AUC 0.1: {valid_auc_01:.4f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {e+1} epochs!")
            break

    # Load best model state
    model.load_state_dict(best_model_state)

    valid_true, valid_pred, valid_loss = evaluate(
            model=model, criterion=criterion, dataloader=valid_loader, device=device,
    )

    test_true, test_pred, _ = evaluate(
        model=model, criterion=criterion, dataloader=test_loader, device=device
    )

    return model, test_true, test_pred, valid_true, valid_pred
