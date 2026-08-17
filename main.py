import streamlit as st
from PIL import Image
import chromadb
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# 1. إعداد الصفحة
st.set_page_config(page_title="Jewelry Visual Search", layout="centered")
st.title("💎 Jewelry Search Engine")
st.write("Upload a jewelry photo to find similar items.")


# 2. تحميل النموذج
@st.cache_resource
def load_model():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor


model = load_model()


# 3. اتصال بقاعدة البيانات (باستخدام get_or_create_collection لتجنب الأخطاء)
@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./data")
    return client.get_or_create_collection(name="jewelry_collection")


collection = load_db()


# 4. دالة استخراج المتجه للصورة
def get_embedding(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
        embedding = torch.mean(features, dim=[2, 3]).squeeze(0).numpy()
    return embedding.tolist()


# 5. واجهة رفع الصورة والبحث
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", width=300)

    if st.button("Search"):
        with st.spinner("Searching..."):
            query_emb = get_embedding(image)
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=25,
                include=["uris", "distances"]
            )

            match_uris = results['uris'][0]

            st.subheader("Similar Items:")
            cols = st.columns(5)
            for i, uri in enumerate(match_uris):
                with cols[i % 5]:
                    st.image(uri, use_column_width=True)