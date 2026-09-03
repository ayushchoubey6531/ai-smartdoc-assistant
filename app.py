import streamlit as st
import time
from google import genai
from pypdf import PdfReader
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AI SmartDoc & Vision Assistant",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI SmartDoc & Vision Assistant")
st.caption("AI Manthan 2.0 Showcase Project — High-Speed Edition")

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
    st.markdown("### 📌 Features:")
    st.markdown("- 💬 **Direct AI Chat**")
    st.markdown("- 📄 **PDF Document Q&A & Summary**")
    st.markdown("- 🖼️ **Image Analysis & Reasoning**")

tab1, tab2, tab3 = st.tabs(["💬 Fast Chat", "📄 PDF Analyzer", "🖼️ Vision Reasoning"])

# Persistent Cached Client
@st.cache_resource
def get_cached_client(key):
    if not key:
        return None
    return genai.Client(api_key=key)

client = get_cached_client(api_key)

# Clean Streaming with Auto-Retry on 503 Spikes (Using ONLY active model: gemini-3.6-flash)
def stream_ai_response(client_instance, contents_payload):
    max_retries = 3
    last_err = None
    
    for attempt in range(max_retries):
        try:
            response_stream = client_instance.models.generate_content_stream(
                model="gemini-3.6-flash",
                contents=contents_payload
            )
            yielded_any = False
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    yielded_any = True
            if yielded_any:
                return
        except Exception as e:
            last_err = e
            time.sleep(2)  # Wait for Google demand spike to clear
            continue
            
    yield f"⚠️ Server busy, please click once again. (Details: {str(last_err)})"

# ----------------- TAB 1: Fast Chat -----------------
with tab1:
    st.subheader("Instant Chat with AI")
    user_query = st.text_area("Enter your prompt / question:", placeholder="e.g., Explain Pointer arithmetic in C...", key="tab1_prompt")
    
    if st.button("Generate Answer", key="tab1_btn"):
        if not client:
            st.error("⚠️ API Key is missing in Streamlit Secrets.")
        elif user_query.strip():
            st.markdown("### 📝 Response:")
            st.write_stream(stream_ai_response(client, user_query))

# ----------------- TAB 2: PDF Analyzer -----------------
with tab2:
    st.subheader("Upload PDF for Fast Parsing")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_pdf is not None:
        pdf_reader = PdfReader(uploaded_pdf)
        extracted_chunks = []
        for page in pdf_reader.pages[:20]:
            txt = page.extract_text()
            if txt:
                extracted_chunks.append(txt)
        
        extracted_text = "\n".join(extracted_chunks)
        st.info(f"📄 Extracted {len(extracted_chunks)} pages in memory.")
        
        pdf_task = st.radio("Choose Action:", ["Summary & Key Takeaways", "Ask Contextual Question"], horizontal=True)
        
        if pdf_task == "Summary & Key Takeaways":
            if st.button("Summarize Now", key="sum_btn"):
                if client:
                    st.markdown("### 📑 Summary:")
                    prompt = f"Provide a structured summary with bullet points of this document:\n\n{extracted_text[:20000]}"
                    st.write_stream(stream_ai_response(client, prompt))
                            
        elif pdf_task == "Ask Contextual Question":
            pdf_question = st.text_input("Enter question based on document:")
            if st.button("Get Answer", key="qa_btn"):
                if client and pdf_question.strip():
                    st.markdown("### 🔍 Answer:")
                    prompt = f"Answer strictly using this context:\n\nContext:\n{extracted_text[:20000]}\n\nQuestion: {pdf_question}"
                    st.write_stream(stream_ai_response(client, prompt))

# ----------------- TAB 3: Vision Reasoning -----------------
with tab3:
    st.subheader("Visual Inspection")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image Preview", width=320)
        
        img_prompt = st.text_input("Question about this image:", value="Describe what you see in this image in detail.")
        
        if st.button("Analyze Now", key="img_btn"):
            if client:
                st.markdown("### 🔍 AI Observation:")
                st.write_stream(stream_ai_response(client, [img, img_prompt]))
