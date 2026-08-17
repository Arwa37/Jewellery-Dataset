import streamlit as st
from PIL import Image
import chromadb
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import os

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


# 3. إعداد قاعدة البيانات وتعبئتها من ملف الـ pkl مباشرة إذا كانت فارغة
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./data")
    collection = client.get_or_create_collection(name="jewelry_collection")

    # إذا كانت القاعدة فارغة، نقوم بقراءة ملف البيكل وتعبئة القاعدة
    if collection.count() == 0 and os.path.exists('product_metadata_500.pkl'):
        with st.spinner("Loading data into ChromaDB... Please wait."):
            with open('product_metadata_500.pkl', 'rb') as f:
                payload = pickle.load(f)

            features = payload.get('features', [])
            paths = payload.get('paths', [])

            # إضافة البيانات إلى قاعدة البيانات
            for idx, (feat, path) in enumerate(zip(features, paths)):
                collection.add(
                    embeddings=[feat if isinstance(feat, list) else feat.tolist()],
                    uris=[path],
                    ids=[str(idx)]
                )
            st.success(f"Successfully loaded {len(paths)} items into database!")

    return collection


collection = init_db()


# دالة استخراج المتجه للصورة المرفوعة
def get_embedding(img):
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
        embedding = torch.mean(features, dim=[2, 3]).squeeze(0).numpy()
    return embedding.tolist()


# 4. واجهة رفع الصورة والبحث
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Image", width=300)

    if st.button("Search"):
        with st.spinner("Searching..."):
            query_emb = get_embedding(image)

            # جلب أكبر 5 نتائج فقط بدقة
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=5,
                include=["uris", "distances"]
            )

            if results and 'uris' in results and len(results['uris']) > 0 and len(results['uris'][0]) > 0:
                match_uris = results['uris'][0]
                match_distances = results['distances'][0]

                st.success(f"Found top {len(match_uris)} similar items!")

                st.subheader("Top 5 Similar Items:")
                cols = st.columns(5)

                for i, (uri, dist) in enumerate(zip(match_uris, match_distances)):
                    with cols[i]:
                        try:
                            st.image(uri, use_column_width=True)
                        except Exception:
                            st.warning(f"Could not load image: {uri}")

                        # حساب نسبة التشابه وعرضها تحت الصورة
                        similarity = max(0, (1 - dist / 2) * 100)
                        st.caption(f"Similarity: {similarity:.1f}%")
            else:
                st.warning("No results found in the database.")