import torch
from torch import nn
from ultralytics import YOLO

# Define a custom detection head
class CustomDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        A custom detection head for YOLOv8.
        
        Args:
        in_channels (int): Number of input channels from the backbone.
        num_classes (int): Number of output classes for detection.
        
        """
        super(CustomDetectionHead, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)  # Fully connected layer for class predictions

    def forward(self, x):
        return self.fc(x)


# Define a segmentation head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        A segmentation head for YOLOv8 to refine object localization at the pixel level
        
        Args:
        in_channels (int): Number of input channels from the backbone.
        num_classes (int): Number of output classes for segmentation.
        """
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # Convolutional layer for segmentation output

    def forward(self, x):
        return self.conv(x)


# Define a transformer-based detection head
class TransformerDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        A transformer-based detection head for YOLOv8 to enhance feature extraction with attention.
        
        Args:
        in_channels (int): Number of input channels from the backbone.
        num_classes (int): Number of output classes for detection.
        """
        super(TransformerDetectionHead, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8)  # Multi-head attention mechanism
        self.fc = nn.Linear(in_channels, num_classes)  # Fully connected layer for class predictions

    def forward(self, x):
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        return self.fc(attn_output)  # Pass the attention output through a fully connected layer


model = YOLO('yolov8n.pt')  # Use YOLOv8 nano weights as the base model

# Replace the detection head with a custom detection head
model.model.head = CustomDetectionHead(in_channels=1024, num_classes=20)  

# Add a segmentation head on top of the YOLO model
model.model.segmentation_head = SegmentationHead(in_channels=1024, num_classes=20)  

# Add a transformer-based detection head
model.model.head = TransformerDetectionHead(in_channels=1024, num_classes=20)  

# Training the YOLO model
model.train(
    data='drive/MyDrive/archive (1)/VehiclesDetectionDataset/dataset.yaml',  # Path to the dataset YAML file
    epochs=70,  # Number of training epochs
    batch=16,  # Batch size
    imgsz=640  # Input image size
)
