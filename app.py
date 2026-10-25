import streamlit as st
from src.image_processor import process_passport_photo
from src.clothing_inpaint import suit_up_clothing
from src.background_generator import generate_studio_background
from src.pdf_generator import generate_print_sheet
from src.config import GROQ_API_KEY, REPLICATE_API_TOKEN
import io
from PIL import Image

st.set_page_config(page_title="AI Passport Studio", page_icon="📸", layout="centered")

st.title("📸 AI Passport Studio")
st.write(
    "Upload a photo to automatically remove the background, resize to passport dimensions, and apply Gen AI upgrades!"
)

# File uploader
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Define image_bytes immediately after upload
    image_bytes = uploaded_file.getvalue()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Photo")
        st.image(uploaded_file, use_container_width=True)

    # UI for Upgrades
    st.divider()
    st.subheader("✨ Gen AI Upgrades")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        apply_suit = st.checkbox("👔 Apply AI Suit-Up (Formal Attire)")
        if apply_suit and not REPLICATE_API_TOKEN:
            st.warning("⚠️ REPLICATE_API_TOKEN missing in .env. Suit-up will fail.")

    with col_opt2:
        apply_bg = st.checkbox("🎨 Generate AI Studio Background")

    bg_prompt = "blurred modern office, professional lighting, corporate studio"
    if apply_bg:
        bg_prompt = st.text_input("Background Prompt", bg_prompt)

    st.divider()

    # Add a button to start processing
    if st.button("Generate Passport Photo", type="primary"):
        # --- STEP 1: BIOMETRIC CROP & REMBG ---
        with st.spinner("Removing background and formatting biometrics..."):
            try:
                processed_image = process_passport_photo(image_bytes)
            except Exception as e:
                st.error(f"Error during biometric processing: {e}")
                st.stop()

        # --- STEP 2: SUIT UP (Gen AI) ---
        if apply_suit:
            with st.spinner(
                "Generating professional suit via AI (This may take a few seconds)..."
            ):
                try:
                    processed_image = suit_up_clothing(
                        processed_image, REPLICATE_API_TOKEN
                    )
                except Exception as e:
                    st.error(f"Error during suit-up: {e}")

        # --- STEP 3: GENERATIVE BACKGROUND (Gen AI) ---
        if apply_bg:
            with st.spinner("Generating AI Studio Background..."):
                try:
                    background = generate_studio_background(
                        bg_prompt, size=processed_image.size
                    )
                    # Composite the foreground onto the new background
                    background.paste(processed_image, (0, 0), processed_image)
                    processed_image = background
                except Exception as e:
                    st.error(f"Error generating background: {e}")

        # --- STEP 4: DISPLAY AND DOWNLOAD ---
        with col2:
            st.subheader("Passport Ready")
            st.image(processed_image, use_container_width=True)

        st.success("Passport photo generated successfully!")

        # Download as JPEG
        buf = io.BytesIO()
        processed_image.convert("RGB").save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Passport JPEG",
            data=byte_im,
            file_name="passport_photo.jpg",
            mime="image/jpeg",
            type="primary",
        )

        # --- STEP 5: PRINT-READY PDF ---
        with st.spinner("Generating 4x6 Print Sheet PDF..."):
            try:
                pdf_buffer = generate_print_sheet(processed_image)
                st.download_button(
                    label="📄 Download 4x6 Print Sheet (PDF)",
                    data=pdf_buffer,
                    file_name="passport_print_sheet.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
