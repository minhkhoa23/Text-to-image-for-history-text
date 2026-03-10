import os
import time
import numpy as np
from typing import List

import streamlit as st
import torch
from diffusers import DiffusionPipeline
from openai import OpenAI
import clip
from PIL import Image

# ================================
# 1. Cấu hình chung & CSS
# ================================
st.set_page_config(
    page_title="Text-to-Image SDXL + LoRA + CLIP",
    page_icon="🖼️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .score-badge {
        background-color: #262730;
        color: white;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        display: inline-block;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🖼️ Text-to-Image với SDXL + LoRA + CLIP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Nhập prompt lịch sử → GPT tăng chất lượng prompt → SDXL + LoRA sinh ảnh → '
    'CLIP chấm điểm & chọn ảnh đẹp nhất.</div>',
    unsafe_allow_html=True,
)

# ================================
# 2. Thiết bị & cache model
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

@st.cache_resource(show_spinner="Đang load SDXL + LoRA, vui lòng chờ...")
def load_sdxl_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to(DEVICE)
    # Load LoRA đã dùng trong notebook
    pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-oldbookillustrations-lora")
    return pipe

@st.cache_resource(show_spinner="Đang load CLIP model...")
def load_clip_model():
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return clip_model, preprocess

# ================================
# 3. Hàm GPT enhance prompt (GitHub Models)
# ================================
SYSTEM_PROMPT = (
    "You are a Prompt Enhancer for text-to-image models. "
    "You refine historical prompts for diffusion models with high clarity, detail, and historical accuracy."
)

def get_openai_client(api_key: str, endpoint: str):
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    return client

def enhance_prompt(client: OpenAI, user_prompt: str, model_name: str) -> str:
    """
    Gọi GPT (qua GitHub Models) để tăng chất lượng prompt.
    Bạn có thể chỉnh lại TEMPLATE theo đúng template trong notebook gốc của mình.
    """
    template = f"""
Bạn là một trợ lý viết prompt cho mô hình text-to-image. 
Nhiệm vụ: chuyển prompt lịch sử sau thành prompt tiếng Anh chi tiết, rõ ràng, phù hợp cho Stable Diffusion XL.

Yêu cầu:
- Mô tả bối cảnh lịch sử chính xác (thời gian, địa điểm nếu có).
- Miêu tả chi tiết về nhân vật, trang phục, biểu cảm.
- Miêu tả môi trường xung quanh (ánh sáng, không khí, background).
- Phù hợp phong cách minh họa cổ điển / old illustration.

Prompt gốc (tiếng Việt):
\"\"\"{user_prompt}\"\"\"


Hãy trả về:
- Một prompt tiếng Anh hoàn chỉnh, chỉ một đoạn, không giải thích thêm.
- Không ghi lại yêu cầu, không ghi thêm text thừa.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

# ================================
# 4. Hàm sinh ảnh & chấm điểm CLIP
# ================================
def generate_images(
    pipe: DiffusionPipeline,
    prompt: str,
    num_images: int,
    num_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
) -> List[Image.Image]:
    images = []
    for _ in range(num_images):
        out = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )
        images.append(out.images[0])
    return images

def score_with_clip(
    clip_model,
    preprocess,
    images: List[Image.Image],
    prompt: str,
) -> List[float]:
    text_token = clip.tokenize([prompt], truncate=True).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_token)

    scores = []
    for img in images:
        image_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
        similarity = (image_features @ text_features.T).item()
        scores.append(float(similarity))
    return scores

# ================================
# 5. Sidebar: cấu hình
# ================================
st.sidebar.header("⚙️ Cài đặt")

st.sidebar.subheader("🔑 API key & Model (GitHub Models)")
default_api_from_env = os.getenv("GITHUB_MODELS_TOKEN", "")
github_api_key = st.sidebar.text_input(
    "GitHub Models Token",
    value="",  # add your token here
    type="password",
    help="Token dùng cho https://models.github.ai/inference. Không commit key này vào code.",
)

endpoint = st.sidebar.text_input(
    "Endpoint",
    value="https://models.github.ai/inference",
    help="Mặc định dùng endpoint của GitHub Models.",
)

model_name = st.sidebar.text_input(
    "GPT model",
    value="openai/gpt-4.1",
)

st.sidebar.subheader("🖼️ Cấu hình sinh ảnh")
num_images = st.sidebar.slider("Số lượng ảnh", min_value=1, max_value=6, value=3)
num_steps = st.sidebar.slider("Số bước suy luận (steps)", 10, 80, 40, step=5)
guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 15.0, 7.0, step=0.5)
height = st.sidebar.number_input("Chiều cao (px)", min_value=256, max_value=1024, value=576, step=64)
width = st.sidebar.number_input("Chiều rộng (px)", min_value=256, max_value=1280, value=1024, step=64)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Thiết bị:** `{DEVICE}`")
st.sidebar.markdown(
    "💡 Gợi ý: Nếu dùng CPU, quá trình sinh ảnh sẽ khá chậm. Nên chạy trên GPU (Colab / Kaggle / local có CUDA)."
)

# ================================
# 6. Main: Form nhập prompt & chạy pipeline
# ================================
with st.form("prompt_form"):
    user_prompt = st.text_area(
        "Nhập prompt (tiếng Việt, nội dung lịch sử muốn vẽ):",
        value="",
        height=120,
    )
    run_button = st.form_submit_button("✨ Generate")

if run_button:
    if not user_prompt.strip():
        st.warning("Vui lòng nhập prompt trước khi chạy.")
    elif not github_api_key.strip():
        st.error("Bạn cần nhập GitHub Models Token ở sidebar.")
    else:
        # 1) Tạo client GPT
        client = get_openai_client(github_api_key, endpoint)

        # 2) Enhance prompt
        with st.spinner("🔍 Đang tăng chất lượng prompt bằng GPT..."):
            try:
                enhanced_prompt = enhance_prompt(client, user_prompt, model_name)
            except Exception as e:
                st.error(f"Lỗi khi gọi GPT: {e}")
                st.stop()

        st.success("Hoàn tất enhance prompt!")
        st.markdown("**Prompt gốc (tiếng Việt):**")
        st.write(user_prompt)

        st.markdown("**Prompt đã được enhance (tiếng Anh, dùng cho SDXL):**")
        st.code(enhanced_prompt, language="markdown")

        # 3) Load model SDXL + LoRA & CLIP
        pipe = load_sdxl_pipeline()
        clip_model, preprocess = load_clip_model()

        # 4) Sinh ảnh
        with st.spinner("🎨 Đang sinh ảnh từ SDXL + LoRA..."):
            start_time = time.time()
            images = generate_images(
                pipe,
                prompt=enhanced_prompt,
                num_images=num_images,
                num_steps=num_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            )
            gen_time = time.time() - start_time

        st.info(f"Đã sinh {len(images)} ảnh trong khoảng {gen_time:.1f} giây.")

        # 5) Chấm điểm CLIP
        with st.spinner("🧠 Đang chấm điểm CLIP để chọn ảnh phù hợp nhất với prompt..."):
            scores = score_with_clip(clip_model, preprocess, images, enhanced_prompt)

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_image = images[best_idx]

        # ================================
        # 7. Hiển thị kết quả
        # ================================
        st.markdown("## 🏆 Ảnh tốt nhất theo CLIP")

        col_best_img, col_best_info = st.columns([2, 1])
        with col_best_img:
            st.image(best_image, caption=f"Best image (score={best_score:.3f})", use_container_width=True)
        with col_best_info:
            st.markdown("**Thông tin lựa chọn:**")
            st.write(f"- Chỉ số CLIP cao nhất: `{best_score:.4f}`")
            st.write(f"- Vị trí ảnh trong tập: `{best_idx + 1}` / `{len(images)}`")
            st.markdown('<span class="score-badge">CLIP Score</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📚 Toàn bộ ảnh sinh ra & điểm CLIP")

        cols = st.columns(3)
        for i, (img, score) in enumerate(zip(images, scores)):
            with cols[i % 3]:
                st.image(img, caption=f"#{i+1} | CLIP={score:.3f}", use_container_width=True)
