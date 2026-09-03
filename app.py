import streamlit as st
import time
from google import genai
from pypdf import PdfReader
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="AI SmartDoc & Vision Assistant",
    page_icon="🤖",
    layout="wide"
)

# App Title & Subtitle
st.title("🤖 AI SmartDoc & Vision Assistant")
st.caption("AI Manthan 2.0 Showcase Project — Powered by GenAI")

# Retrieve Key safely from Streamlit Secrets
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

# Mode Selection Tabs
tab1, tab2, tab3 = st.tabs(["💬 General Chat", "📄 PDF Analyzer", "🖼️ Image Reasoning"])

# Function to initialize Gemini Client
def get_gemini_client():
    if not api_key:
        st.error("⚠️ Please configure GEMINI_API_KEY in Streamlit Secrets.")
        return None
    return genai.Client(api_key=api_key)

# Fallback & Retry execution across active models
def generate_ai_response(client, contents_payload):
    target_models = ["gemini-3.6-flash", "gemini-3.1-pro-preview"]
    last_err = None

    for model_name in target_models:
        for attempt in range(2):  # Temporary 503 spike ke liye 1 retry
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload
                )
                return response.text
            except Exception as err:
                last_err = err
                time.sleep(1.5)
                continue
    raise last_err

# ----------------- TAB 1: General Chat -----------------
with tab1:
    st.subheader("Ask Anything to AI")
    user_query = st.text_area("Enter your prompt / question:", placeholder="e.g., Explain Quantum Computing in simple terms...", key="tab1_prompt")
    
    if st.button("Generate Answer", key="tab1_btn"):
        client = get_gemini_client()
        if client and user_query.strip():
            with st.spinner("AI is thinking..."):
                try:
                    text_output = generate_ai_response(client, user_query)
                    st.success("✅ Done!")
                    st.markdown("### 📝 Response:")
                    st.write(text_output)
                except Exception as e:
                    st.error(f"Error: {e}")

# ----------------- TAB 2: PDF Analyzer -----------------
with tab2:
    st.subheader("Upload PDF for Q&A and Summarization")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_pdf is not None:
        pdf_reader = PdfReader(uploaded_pdf)
        extracted_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        st.info(f"📄 Extracted {len(pdf_reader.pages)} pages successfully.")
        
        pdf_task = st.radio("Choose Action:", ["Generate Summary & Key Takeaways", "Ask a Question from PDF"], horizontal=True)
        
        if pdf_task == "Generate Summary & Key Takeaways":
            if st.button("Summarize PDF", key="sum_btn"):
                client = get_gemini_client()
                if client:
                    with st.spinner("Analyzing and summarizing document..."):
                        try:
                            prompt = f"Summarize the following document clearly with key takeaways and bullet points:\n\n{extracted_text[:30000]}"
                            text_output = generate_ai_response(client, prompt)
                            st.success("✅ Summary Generated!")
                            st.write(text_output)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            
        elif pdf_task == "Ask a Question from PDF":
            pdf_question = st.text_input("What do you want to know from this document?")
            if st.button("Get Answer", key="qa_btn"):
                client = get_gemini_client()
                if client and pdf_question.strip():
                    with st.spinner("Searching document context..."):
                        try:
                            prompt = f"Answer the user's question strictly based on the following document context.\n\nContext:\n{extracted_text[:30000]}\n\nQuestion: {pdf_question}"
                            text_output = generate_ai_response(client, prompt)
                            st.write(text_output)
                        except Exception as e:
                            st.error(f"Error: {e}")

# ----------------- TAB 3: Image Reasoning -----------------
with tab3:
    st.subheader("Upload an Image for Visual AI Analysis")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image Preview", width=350)
        
        img_prompt = st.text_input("Ask a question about this image:", value="Describe what you see in this image in detail.")
        
        if st.button("Analyze Image", key="img_btn"):
            client = get_gemini_client()
            if client:
                with st.spinner("Analyzing image..."):
                    try:
                        text_output = generate_ai_response(client, [img, img_prompt])
                        st.success("✅ Analysis Complete!")
                        st.markdown("### 🔍 AI Observation:")
                        st.write(text_output)
                    except Exception as e:
                        st.error(f"Error: {e}")
