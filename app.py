import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# Define your model architecture (must match the one used for training)
class OralCancerModel(nn.Module):
    def __init__(self):
        super(OralCancerModel, self).__init__()
        # Example: replace this with your actual model architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*64*64, 6)  # 6 classes
        )

    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OralCancerModel()
model.load_state_dict(torch.load("vit_tiny_dental_updated.pth", map_location=device))
model.eval()
model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Match the size used during training
    transforms.ToTensor(),
])

# Classes
classes = ['hypodontia', 'Tooth Discoloration', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Calculus']

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item()
    
    return {classes[class_idx]: confidence}

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Oral Health Classifier",
    description="Upload an oral image and predict its class."
)

iface.launch()
