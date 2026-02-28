import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import imagehash

# =======================
# Image preprocessing
# =======================
def preprocess(image: Image.Image) -> Image.Image:
    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (256, 256))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 80, 180)

    return Image.fromarray(edges)

# =======================
# Hashes
# =======================
def get_hashes(image: Image.Image):
    return {
        "phash": imagehash.phash(image),
        "dhash": imagehash.dhash(image)
    }

def hash_similarity(h1, h2) -> float:
    return 1 - (h1 - h2) / len(h1.hash.flatten())

# =======================
# Combined similarity
# =======================
def combined_similarity(img1: Image.Image, img2: Image.Image) -> float:
    img1_p = preprocess(img1)
    img2_p = preprocess(img2)

    h1 = get_hashes(img1_p)
    h2 = get_hashes(img2_p)

    sims = [
        hash_similarity(h1["phash"], h2["phash"]),
        hash_similarity(h1["dhash"], h2["dhash"])
    ]

    return float(np.mean(sims))

# =======================
# Originality score
# =======================
def originality_score(uploaded_image, dataset_path):
    similarities = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path).convert("RGB")
                sim = combined_similarity(uploaded_image, img)
                similarities.append(sim)
            except:
                continue

    max_sim = max(similarities) if similarities else 0
    originality = (1 - max_sim ** 0.7) * 100

    return round(originality, 2), round(max_sim * 100, 2)

# =======================
# Streamlit UI
# =======================
st.set_page_config(
    page_title="Logo Originality Checker",
    layout="centered"
)

st.title("🎨 Проверка оригинальности логотипа")
st.write(
    "Загрузите логотип в формате PNG или JPG "
    "(SVG необходимо предварительно экспортировать)."
)

uploaded_file = st.file_uploader(
    "Загрузите логотип",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Загруженный логотип",
        use_container_width=True
    )

    with st.spinner("Анализируем логотип..."):
        originality, similarity = originality_score(image, "dataset")

    st.subheader("📊 Результат")

    st.metric("Оригинальность", f"{originality}%")
    st.metric("Максимальное сходство", f"{similarity}%")

    if originality > 80:
        st.success("🔥 Высокая оригинальность")
    elif originality > 60:
        st.warning("⚠️ Средняя оригинальность")
    else:
        st.error("❌ Низкая оригинальность — логотип похож на существующие")

st.caption(
    "Процент отражает визуальное сходство и не является юридической экспертизой."
)
