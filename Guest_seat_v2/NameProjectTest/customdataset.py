import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


model = NeuralNetwork().to(device)


X = torch.rand(1, 28, 28, device=device)

logits = model(X)
print(logits)
pred_probab = nn.Softmax(dim=1)(logits)
print(pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
    
# imagedir = "D:\\repositories\\test ai assitance\\lighmcheni test\\img"
# annotations_file = "D:\\repositories\\test ai assitance\\lighmcheni test\\annotations_file.txt"

# training_data = CustomImageDataset(annotations_file,imagedir, transform=None, target_transform=None)
# test_data = CustomImageDataset(annotations_file,imagedir, transform=None, target_transform=None)

# train_dataloader = DataLoader(training_data, batch_size=600, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# # Display image and label.

# for train_features, train_labels in train_dataloader:
#     # train_features, train_labels = next(iter(train_dataloader))
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
#     # print(train_features)
#     # print(train_labels)

#     img = train_features.squeeze()
#     label = train_labels
#     print(img)
#     # plt.imshow(img, cmap="gray")
#     # plt.axis("off")
#     # plt.show()
#     print(f"Label: {label}")