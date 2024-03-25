print('Started')
from transformers import SamProcessor, SamModel, SamConfig
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import SamModel
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

print('Modules loaded')

def generate_input_points(array_size=1024, grid_size=5):
    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)
    xv, yv = np.meshgrid(x, y)
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv.tolist(), yv.tolist())]
    input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)  # Reshape as needed
    return input_points

input_points=generate_input_points()

class XRayDataset(Dataset):
    def __init__(self, items_df, processor, dataset_path):
        self.items_df = items_df
        self.processor = processor
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.items_df)

    def __getitem__(self, idx):
        row = self.items_df.iloc[idx]
        image_id, bbox_id = row['image_id'], row['bbox_id']
        # print(type(),bbox)
        # print([bbox])

        image_path = os.path.join(self.dataset_path, image_id + '.png')
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)

        mask_path = os.path.join(self.dataset_path, f"{image_id}_{bbox_id}_mask.pt")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = torch.load(mask_path)

        # Assuming bbox is stored or calculated beforehand if needed for `input_boxes`
        inputs = self.processor(images=image,input_points=input_points, return_tensors="pt")
        inputs["ground_truth_mask"] = mask
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
print('Dataset Class Defined')
# Define paths
dataset_path = 'C:\\Users\\theju\\Documents\\MSc AI\\MLP CW3\\preprocessed_train\\preprocessed_train'
# csv_path = 'C:\\Users\\theju\\Documents\\MSc AI\\MLP CW3\\flattened_df.csv'

flattened_df = pd.read_csv('C:\\Users\\theju\\Documents\\MSc AI\\MLP CW3\\flattened_df_normalized.csv')
train,test=train_test_split(flattened_df, test_size=0.1, random_state=42)
train_items, val_items = train_test_split(train, test_size=0.1, random_state=42)
test.to_csv('C:\\Users\\theju\\Documents\\MSc AI\\MLP CW3\\test.csv',index=False)

print('Dataset Paths Defined')
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print('Processor Initialized')

# Initialize datasets

train_dataset = XRayDataset(train_items, processor, dataset_path)
val_dataset = XRayDataset(val_items, processor, dataset_path)
print('Datasets Initialized')
# Prepare DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
print('Dataset Loaders Ready')

# Load the model

model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is:",device)

model = SamModel(config=model_config)

checkpoint_path = "C:\\Users\\theju\\Documents\\MSc AI\\MLP CW3\\epochs_0_model_checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
print('Model Loaded')
# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')



# Assuming seg_loss is defined (as your segmentation loss function)
# Assuming optimizer and model are already defined

# Training and validation loop
num_epochs = 2

# device = "mps" if torch else "cpu"
# device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print('Starting training on {}'.format(device))

for epoch in [1,2,3,4]:
    # Training phase
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc="Training"):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=batch["input_points"].to(device),
                        multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_loss_mean = mean(train_losses)
    filename="epochs_{}_model_checkpoint.pth".format(epoch)
    print('Saved the model for the epoch:',filename)
    torch.save(model.state_dict(),filename)
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_points=batch["input_points"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            val_losses.append(loss.item())

    val_loss_mean = mean(val_losses)

    print(f'EPOCH: {epoch}')
    print(f'Training mean loss: {train_loss_mean}')
    print(f'Validation mean loss: {val_loss_mean}')
print('Training over- Saving model')
torch.save(model.state_dict(), "final_model_checkpoint.pth")
print('Model saved')