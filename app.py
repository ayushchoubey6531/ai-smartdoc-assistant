import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AI SmartDoc & Vision Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI SmartDoc & Vision Assistant")
st.caption("AI Manthan 2.0 Showcase Project — High-Speed Turbo Edition")

# Retrieve Key safely from Secrets
api_key = st.secrets.get("GEMINI_API_KEY", "")

# Sidebar
with st.sidebar:
    st.header("⚡ System Status")
    if api_key:
        st.success("API Key Active & Ready")
    else:
        st.error("API Key Missing in Secrets")
    st.markdown("---")
    st.markdown("### 🚀 Speed Features:")
    st.markdown("- ⚡ **Real-time Word Streaming**")
    st.markdown("- 🏎️ **Persistent Client Connection**")
    st.markdown("- 📄 **Optimized Document Context**")

tab1, tab2, tab3 = st.tabs(["💬 Fast Chat", "📄 PDF Turbo Analyzer", "🖼️ Fast Vision Reasoning"])

# Persistent Cached Client (Faster API Handshake)
@st.cache_resource
def get_cached_client(key):
    if not key:
        return None
    return genai.Client(api_key=key)

client = get_cached_client(api_key)

# Fast Streaming Runner with Quick Fallback
def stream_ai_response(client_instance, contents_payload):
    target_models = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]
    
    # Disable heavy thinking traces for maximum output speed
    speed_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    
    for model_name in target_models:
        try:
            stream = client_instance.models.generate_content_stream(
                model=model_name,
                contents=contents_payload,
                config=speed_config
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except Exception:
            continue
    yield "⚠️ Server is under heavy traffic. Please try once again."

# ----------------- TAB 1: Fast Chat -----------------
with tab1:
    st.subheader("Instant Chat with AI")
    user_query = st.text_area("Enter your prompt / question:", placeholder="e.g., Explain Pointer arithmetic in C...", key="tab1_prompt")
    
    if st.button("Generate Fast Answer", key="tab1_btn"):
        if not client:
            st.error("⚠️ API Key is missing in Secrets.")
        elif user_query.strip():
            st.markdown("### 📝 Response:")
            st.write_stream(stream_ai_response(client, user_query))

# ----------------- TAB 2: PDF Turbo Analyzer -----------------
with tab2:
    st.subheader("Upload PDF for Fast Parsing")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_pdf is not None:
        pdf_reader = PdfReader(uploaded_pdf)
        extracted_chunks = []
        for i, page in enumerate(pdf_reader.pages[:25]):  # Process up to 25 pages rapidly
            txt = page.extract_text()
            if txt:
                extracted_chunks.append(txt)
        
        extracted_text = "\n".join(extracted_chunks)
        st.info(f"📄 Extracted {len(extracted_chunks)} pages in memory.")
        
        pdf_task = st.radio("Choose Action:", ["Instant Summary & Key Takeaways", "Ask Contextual Question"], horizontal=True)
        
        if pdf_task == "Instant Summary & Key Takeaways":
            if st.button("Summarize Now", key="sum_btn"):
                if client:
                    st.markdown("### 📑 Summary:")
                    prompt = f"Provide a rapid, bulleted executive summary of this document:\n\n{extracted_text[:18000]}"
                    st.write_stream(stream_ai_response(client, prompt))
                            
        elif pdf_task == "Ask Contextual Question":
            pdf_question = st.text_input("Enter question based on document:")
            if st.button("Get Quick Answer", key="qa_btn"):
                if client and pdf_question.strip():
                    st.markdown("### 🔍 Answer:")
                    prompt = f"Answer strictly using this context:\n\nContext:\n{extracted_text[:18000]}\n\nQuestion: {pdf_question}"
                    st.write_stream(stream_ai_response(client, prompt))

# ----------------- TAB 3: Fast Vision Reasoning -----------------
with tab3:
    st.subheader("Instant Visual Inspection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        # Resize thumbnail internally if image is massive (reduces network upload latency)
        img.thumbnail((1024, 1024))
        st.image(img, caption="Uploaded Image Preview", width=320)
        
        img_prompt = st.text_input("Question about this image:", value="Quickly summarize what you see.")
        
        if st.button("Analyze Now", key="img_btn"):
            if client:
                st.markdown("### 🔍 AI Observation:")
                st.write_stream(stream_ai_response(client, [img, img_prompt]))
