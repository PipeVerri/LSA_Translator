import torch.nn as nn
import torch

def training_loop(model, train_loader, test_loader, filename, lr=1e-3, weight_decay=0.0, EPOCH=100, label_idx=None,
                  criterion=nn.CrossEntropyLoss()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Detect loss type
    is_bce = isinstance(criterion, nn.BCEWithLogitsLoss)

    # ==========================================
    # TRACK BEST PERFORMANCES
    # ==========================================
    best_train_acc = 0
    best_train_acc_test_acc = 0

    best_test_acc = 0
    best_test_acc_train_acc = 0

    # ==========================================

    def process_y(y):
        if label_idx is not None:
            return (y[:, label_idx]).unsqueeze(1)
        return y

    def compute_accuracy(y_pred, y):
        """Compute accuracy based on loss type"""
        if is_bce:
            # For binary classification with BCEWithLogitsLoss
            predicted = (y_pred > 0).float()
            correct = (predicted == y).sum().item()
        else:
            # For multi-class classification with CrossEntropyLoss
            _, predicted = y_pred.max(1)
            correct = (predicted == y).sum().item()
        return correct

    for epoch in range(EPOCH):
        # ======================
        # TRAIN
        # ======================
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            y = process_y(y)

            # Ensure correct dtype for BCE
            if is_bce:
                y = y.float()

            y = y.to(device)

            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ======================
        # TRAIN ERROR (subset)
        # ======================
        model.eval()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_batches = 5

        with torch.no_grad():
            for i, (x, lengths, y) in enumerate(train_loader):
                if i >= train_batches:
                    break

                x = x.to(device)
                y = process_y(y)

                # Ensure correct dtype for BCE
                if is_bce:
                    y = y.float()

                y = y.to(device)

                y_pred = model(x, lengths)
                loss = criterion(y_pred, y)
                train_loss += loss.item()

                train_correct += compute_accuracy(y_pred, y)
                train_total += y.size(0)

        avg_train_loss = train_loss / train_batches
        train_acc = 100 * train_correct / train_total

        # ======================
        # TEST ERROR
        # ======================
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, lengths, y in test_loader:
                x = x.to(device)
                y = process_y(y)

                # Ensure correct dtype for BCE
                if is_bce:
                    y = y.float()

                y = y.to(device)

                y_pred = model(x, lengths)
                loss = criterion(y_pred, y)
                test_loss += loss.item()

                test_correct += compute_accuracy(y_pred, y)
                test_total += y.size(0)

        test_acc = 100 * test_correct / test_total

        # ==========================================
        # UPDATE BEST TRAIN & TEST RESULTS
        # ==========================================
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_acc_test_acc = test_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_acc_train_acc = train_acc
            if epoch > 70:  # Asi no guardo en todas las iteraciones iniciales
                torch.save(model.state_dict(), filename)
        # ==========================================

        # ======================
        # PRINT SUMMARY
        # ======================
        print(f"Epoch {epoch + 1}/{EPOCH} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    print("\n================ FINAL RESULTS ================")
    print(f"Highest Train Acc: {best_train_acc:.2f}% (Test Acc at that time: {best_train_acc_test_acc:.2f}%)")
    print(f"Highest Test Acc:  {best_test_acc:.2f}% (Train Acc at that time: {best_test_acc_train_acc:.2f}%)")