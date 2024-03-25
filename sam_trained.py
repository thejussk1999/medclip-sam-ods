from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw
import os
import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import to_tensor, to_pil_image

# Define paths
dataset_path = '/home/thejussk/selective-search-for-object-recognition/archive/train'
csv_path = '/home/thejussk/selective-search-for-object-recognition/train.csv'

# Load the bounding box data
bbox_df = pd.read_csv(csv_path)
bbox_df = bbox_df[bbox_df['class_name'] != 'No finding']  # Filter out 'No finding'
image_ids = bbox_df['image_id'].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Flatten the dataset
items = []
for _, row in bbox_df.iterrows():
    items.append((row['image_id'], [row['x_min'], row['y_min'], row['x_max'], row['y_max']], row['class_name']))

# Convert to a DataFrame for easy handling
flattened_df = pd.DataFrame(items, columns=['image_id', 'bbox', 'class_name'])

# Now split this DataFrame
train_items, val_items = train_test_split(flattened_df, test_size=0.2, random_state=42)



def create_mask_from_bbox(image, bboxes, resize_dim=(256, 256)):
    # Assuming `image` is a PIL Image, get its dimensions
    width, height = image.size
    # Create a single-channel (grayscale) mask with the same dimensions
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        # Normalize and resize bbox coordinates as needed here
        draw.rectangle(bbox, outline=1, fill=1)
    # Resize mask if needed to match model input dimensions
    mask = mask.resize(resize_dim, Image.NEAREST)
    return torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # Add channel dimension


# # 3. Create Text Prompts
# def create_text_prompt(label):
#     return f"Outline the area in this chest x-ray with {label}"



def load_and_preprocess_image(image_path):
    image = Image.open(image_path) # Load grayscale image
    image = to_tensor(image)  # Convert to PyTorch tensor
    image = image.repeat(3, 1, 1)  # Repeat the single channel to get 3 channels
    image = to_pil_image(image)  # Convert back to PIL Image
    return image

class XRayDataset(Dataset):
    def __init__(self, items_df, processor):
        self.items_df = items_df
        self.processor = processor

    def __len__(self):
        return len(self.items_df)


    def __len__(self):
        return len(self.items_df)

    def __getitem__(self, idx):
        row = self.items_df.iloc[idx]
        image_id, bbox, label = row['image_id'], row['bbox'], row['class_name']
        image_path = os.path.join(dataset_path, image_id + '.png')
        image = load_and_preprocess_image(image_path)
        width, height = image.size

        # Create mask for the single bounding box
        mask = create_mask_from_bbox(image, [bbox])

        # Adjust the bounding box for model input

        normalized_bbox = [[bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]]

        inputs = self.processor(images=image, input_boxes=[normalized_bbox], return_tensors="pt")
        inputs["ground_truth_mask"] = mask
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs


# Initialize datasets
train_dataset = XRayDataset(train_items, processor)
val_dataset = XRayDataset(val_items, processor)

# Prepare DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

from torch.optim import Adam
import monai
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

# Assuming seg_loss is defined (as your segmentation loss function)
# Assuming optimizer and model are already defined

# Training and validation loop
num_epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_losses = []
    for batch in tqdm(train_loader, desc="Training"):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_loss_mean = mean(train_losses)

    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            val_losses.append(loss.item())

    val_loss_mean = mean(val_losses)

    print(f'EPOCH: {epoch}')
    print(f'Training mean loss: {train_loss_mean}')
    print(f'Validation mean loss: {val_loss_mean}')

torch.save(model.state_dict(), "first_model_checkpoint.pth")