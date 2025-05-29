import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 classes
    model.load_state_dict(torch.load('model/vgg16_cifar10.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Define transform (must match training)
transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert from numpy (OpenCV) to PIL Image
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title("CIFAR-10 Image Classifier with VGG16 ")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read file bytes to numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image (OpenCV loads as BGR)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Show image using Streamlit
    st.image(cv_img, caption='Uploaded Image (OpenCV)', use_column_width=True)

    # Preprocess using transform pipeline
    input_tensor = transform(cv_img).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    st.write(f"The image you uploaded is most likely a  **{predicted_class}**")
