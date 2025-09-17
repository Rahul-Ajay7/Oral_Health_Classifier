import torch
import timm
import gradio as gr
from PIL import Image
from torchvision import transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the ViT model architecture
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=6)

# Load the saved model weights
model.load_state_dict(torch.load("vit_tiny_dental_updated.pth", map_location=device))
model.to(device)
model.eval()

# Class names
classes = ['hypodontia', 'Tooth Discoloration', 'Data caries', 'Gingivitis', 'Mouth Ulcer', 'Calculus']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Oral Cancer Detection",
    description="Upload an image of oral health condition to classify."
)

if __name__ == "__main__":
    interface.launch()
