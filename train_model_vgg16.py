import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sns
import pandas as pd

# Interactive mode off (better for saving figures)
plt.ioff()

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Dataset paths
# ----------------------------
data_dir = "data_split"  # Must contain 'train', 'val', 'test' folders

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ----------------------------
# Training function
# ----------------------------
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=15):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    tloss, vloss, tacc, vacc = [], [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                tloss.append(epoch_loss)
                tacc.append(epoch_acc.cpu())
            else:
                vloss.append(epoch_loss)
                vacc.append(epoch_acc.cpu())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, tloss, vloss, tacc, vacc


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=0)
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print("Classes found:", class_names)
    print("Dataset sizes:", dataset_sizes)

    # ----------------------------
    # Load pretrained VGG16
    # ----------------------------
    model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze all layers
    for param in model_ft.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # ----------------------------
    # Train the model
    # ----------------------------
    model_ft, tloss, vloss, tacc, vacc = train_model(
        model_ft, criterion, optimizer_ft, scheduler,
        dataloaders, dataset_sizes, num_epochs=15
    )

    # ----------------------------
    # Save Accuracy Plot
    # ----------------------------
    plt.figure()
    plt.plot([x.item() for x in tacc], '-o')
    plt.plot([x.item() for x in vacc], '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Validation'])
    plt.title("Accuracy Curve")
    plt.savefig("accuracy_vgg16.png")
    plt.close()

    # ----------------------------
    # Save Loss Plot
    # ----------------------------
    plt.figure()
    plt.plot(tloss, '-o')
    plt.plot(vloss, '-o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['Train', 'Validation'])
    plt.title("Loss Curve")
    plt.savefig("loss_vgg16.png")
    plt.close()

    # ----------------------------
    # Save trained model
    # ----------------------------
    torch.save(model_ft.state_dict(), "vgg16_veggies.pth")
    print("Trained model saved as vgg16_veggies.pth")

    # ----------------------------
    # Test Set Evaluation
    # ----------------------------
    test_loader = torch.utils.data.DataLoader(image_datasets['test'],
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=0)

    correct_count = 0
    total_count = 0
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    model_ft.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            correct_count += torch.sum(preds == labels.data).item()
            total_count += labels.size(0)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    test_accuracy = correct_count / total_count
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Number of Test Images: {total_count}")

    # ----------------------------
    # Plot Confusion Matrix
    # ----------------------------
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(12,10))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_vgg16.png")
    plt.close()

    print("Confusion matrix saved as confusion_matrix_vgg16.png")
