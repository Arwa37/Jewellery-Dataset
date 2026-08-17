import os
import streamlit as st
from PIL import Image
import chromadb
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

# 1. إعداد الصفحة
st.set_page_config(page_title="Jewelry Visual Search", layout="centered")
st.title("💎 Jewelry Search Engine")
st.write("Upload a jewelry photo to find the top 5 similar items.")


# 2. تحميل النموذج
@st.cache_resource
def load_model():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    feature_extractor.eval()
    return feature_extractor


model = load_model()

# تحويلات الصور
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# دالة ذكية للبحث عن مسار الصورة
def find_image_path(filename, base_dir):
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


# 3. إعداد قاعدة البيانات
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./data")
    try:
        client.delete_collection(name="jewelry_collection")
    except Exception:
        pass

    collection = client.create_collection(name="jewelry_collection")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_dir, 'product_metadata_500.pkl')

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            payload = pickle.load(f)
        features = payload.get('features', [])
        paths = payload.get('paths', [])

        for idx, (feat, path) in enumerate(zip(features, paths)):
            filename = os.path.basename(path)
            collection.add(
                embeddings=[feat if isinstance(feat, list) else feat.tolist()],
                uris=[filename],
                ids=[str(idx)]
            )
    return collection


collection = init_db()


def get_embedding(img):
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
        embedding = torch.mean(features, dim=[2, 3]).squeeze(0).numpy()
    return embedding.tolist()


# 4. واجهة البحث
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", width=300)

    if st.button("Search"):
        with st.spinner("Searching..."):
            query_emb = get_embedding(image)
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=5,
                include=["uris", "distances"]
            )
            base_dir = os.path.dirname(os.path.abspath(__file__))

            if results and 'uris' in results and len(results['uris']) > 0 and len(results['uris'][0]) > 0:
                match_filenames = results['uris'][0]
                match_distances = results['distances'][0]

                st.success(f"Found top {len(match_filenames)} similar items!")
                st.subheader("Top 5 Similar Items:")
                cols = st.columns(5)

                for i, (filename, dist) in enumerate(zip(match_filenames, match_distances)):
                    with cols[i]:
                        real_path = find_image_path(filename, base_dir)
                        if real_path and os.path.exists(real_path):
                            st.image(real_path, use_column_width=True)
                        else:
                            st.info(f"Item: {filename}")

                        # معادلة النسبة المئوية (تم زيادة القاسم إلى 10 لضمان ظهور أرقام)
                        # جربي تغيير الرقم 10.0 إلى رقم أكبر إذا استمرت النسبة 0
                        similarity = max(0.0, (1.0 - (dist / 10.0)) * 100)

                        st.caption(f"Similarity: {similarity:.2f}%")
            else:
                st.warning("No results found.")