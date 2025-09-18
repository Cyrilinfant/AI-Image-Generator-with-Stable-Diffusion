import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

@st.cache_resource
def load_model(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    return pipe.to("cpu")

# ---- Streamlit UI ----
st.title("🎨 Multi-Model Stable Diffusion (CPU — Streamlit)")

model_choice = st.selectbox(
    "Choose a model:",
    [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "dreamlike-art/dreamlike-diffusion-1.0",
    ],
)

prompt = st.text_area("Prompt", "A beautiful fantasy castle, cinematic, highly detailed")
negative_prompt = st.text_area("Negative prompt (optional)", "lowres, watermark, deformed")

steps = st.slider("Steps", 10, 80, 25)
guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
width = st.selectbox("Width", [512, 640, 768], index=0)
height = st.selectbox("Height", [512, 640, 768], index=0)
seed = st.text_input("Seed (optional)", "")

if st.button("🚀 Generate Image"):
    with st.spinner(f"Loading {model_choice}... (may take time on CPU ⏳)"):
        pipe = load_model(model_choice)

    generator = None
    if seed.strip():
        try:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        except:
            st.warning("Invalid seed, using random seed instead.")

    with st.spinner("Generating image... (CPU is very slow ⏳)"):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )
        image = output.images[0]
        st.image(image, caption=f"Generated with {model_choice}", use_container_width=True)
