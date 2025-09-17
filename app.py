import torch
import torch.nn as nn
import timm
import gradio as gr
from PIL import Image
from torchvision import transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Number of classes
NUM_CLASSES = 6

# Load the ViT model architecture
model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

# Load the saved model weights
model.load_state_dict(torch.load("vit_tiny_dental.pth", map_location=device))
model.to(device)
model.eval()

# Class names in the correct order based on folder structure
classes = ['Calculus', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia']

# Preprocessing using ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
    return f"{classes[predicted.item()]} ({conf.item() * 100:.2f}%)"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Oral Health Condition Classifier",
    description="Upload an image to classify the oral health condition."
)

if __name__ == "__main__":
    interface.launch()
