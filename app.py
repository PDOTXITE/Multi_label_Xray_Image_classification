import streamlit as st
import os
from torchvision import models
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
# รายการคลาส
all_classes = ["Cardiomegaly", "Hernia", "Infiltration", "Nodule", "Emphysema", "Effusion", "Atelectasis",
               "Pleural_Thickening", "Pneumothorax", "Mass", "Fibrosis", "Consolidation",
               "Edema", "Pneumonia"]

sample_images = {
        "None" : "../",
        "Emphysema": "sample/sample1.png",
        "Atelectasis": "sample/sample2.png",
        "No Finding": "sample/sample3.png"
}

# ฟังก์ชันสำหรับโหลดโมเดล
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(all_classes))
    model.load_state_dict(torch.load('models/resnet50finalmodel.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

# ฟังก์ชันสำหรับแปลงภาพ
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ฟังก์ชันสำหรับทำนายผลลัพธ์
def predict_image(image, model):
    with torch.no_grad():
        output = model(image)
        threshold = 0.1
        predicted_labels = (output > threshold).squeeze().int()
        return predicted_labels

# ฟังก์ชันหลัก
def main():
    model = load_model()
    image = None

    st.title("Multi-Label X-Ray Image Classification")
    st.write("Upload an image")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

    with st.expander("Or choose from sample here..."):
        sample = st.selectbox(label="Select here", options=list(sample_images.keys()), label_visibility="hidden")
        col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.image(sample_images["Emphysema"], caption="Emphysema", use_column_width=True)
    # with col2:
    #     st.image(sample_images["Atelectasis"], caption="Atelectasis", use_column_width=True)
    # with col3:
    #     st.image(sample_images["No Finding"], caption="No Finding", use_column_width=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
    elif sample != "None":
            image = Image.open(sample_images[sample])
            st.image(image, caption=f'Selected Sample: {sample}', use_column_width=True)
            st.write("")
    else:
        image = None

    if image is not None:
        image_tensor = preprocess_image(image)
        predicted_labels = predict_image(image_tensor, model)

        st.subheader("Predicted labels:")
        if predicted_labels.sum() == 0:
            st.write("No Finding")
        else:
            for idx, label in enumerate(all_classes):
                if predicted_labels[idx] == 1:
                    st.write(f"- {label}")

    st.subheader("Credits")
    st.write("By : Phetdau Dueramae | AI-Builders")
    st.markdown("Source : [Github]('https://github.com/PDOTXITE/Multi_label_Xray_image_classification')")


if __name__ == "__main__":
    main()