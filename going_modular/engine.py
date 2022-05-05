import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Send data to GPU
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Calculate metrics
        train_loss += loss.item()
        y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
        # print(f"y: \n{y}\ny_pred_class:{y_pred_class}")
        # print(f"y argmax: {y_pred.argmax(dim=1)}")
        # print(f"Equal: {(y_pred_class == y)}")
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        # print(f"batch: {batch} train_acc: {train_acc}")

    # Adjust returned metrics
    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model, dataloader, loss_fn, device):
    model.eval()  # put model in eval mode
    test_loss_total, test_acc = 0, 0
    # Turn on inference context manager
    for batch, (X, y) in enumerate(dataloader):
        # Send data to GPU
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.inference_mode():
            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss
            test_loss = loss_fn(test_pred, y)

            # Calculate metrics
            test_loss_total += test_loss.item()
            test_acc += torch.eq(test_pred.argmax(dim=1), y).sum().item() / len(
                test_pred
            )

    # Adjust metrics
    test_loss_total /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss_total, test_acc


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
