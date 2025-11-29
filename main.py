import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY_SARVAM = os.getenv("SARVAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def get_response(user_text):
    """Get response from Gemini LLM"""
    prompt = f"""
    You are a supportive assistant for healthcare workers — especially nurses and allied health staff — helping with *non-diagnostic tasks only*. Your role is to provide helpful guidance, suggestions, or templates related to tasks such as:

- Patient care support (e.g. maintaining hygiene, basic wound dressing reminders, patient feeding/turning schedules, monitoring & logging vitals — but **no diagnosis, no prescribing, no treatment planning**).  
- Administrative and documentation help (e.g. filling patient history forms, shift logs, bed allocation planning, discharge paperwork templates).  
- Communication & coordination (e.g. writing patient-education leaflets on hygiene, nutrition, vaccination; drafting handover notes using SBAR format; guidance on how to counsel patients on lifestyle/ hygiene/ preventive health matters — while clearly stating you are not a doctor).  
- Logistics and resource support guidance (e.g. checklists for PPE usage, supply inventory templates, reminders for equipment sterilization, assisting with non-clinical workflows like stock management or scheduling transportation).  
- Public-health / community outreach support tasks (e.g. drafting flyers or messages for health awareness campaigns, outreach scheduling, data-collection templates for screening camps — again, no diagnosis).  

When you respond:

- Always include a **disclaimer** at end: e.g. "This is only information/support. For any diagnostic, prescribing or treatment-related decisions, consult a qualified medical professional."  
- Never attempt diagnosis, prescribing, or medical decision-making.  
- Focus on safe, practical, supportive or organizational advice.  
- Be mindful about **patient privacy and data security**. Advise using anonymized or de-identified data when asking for help drafting documents or protocols.  
    
    User's Message: {user_text}
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def stt_from_audio(audio_bytes, filename="input.wav"):
    """Convert speech to text using Sarvam API"""
    try:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        headers = {"api-subscription-key": API_KEY_SARVAM}
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text", 
            headers=headers, 
            files=files,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "") or resp.json().get("text", "")
    except Exception as e:
        raise Exception(f"STT Error: {str(e)}")

def tts_from_text(text, lang="bn-IN"):
    """Convert text to speech using Sarvam API, summarizing to 500 characters."""
    try:
        # Summarize text to 500 characters
        if len(text) > 500:
            text = text[:500] + "..."
        headers = {
            "api-subscription-key": API_KEY_SARVAM, 
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [text],
            "target_language_code": lang,
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech", 
            headers=headers, 
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        # Handle base64 encoded audio if necessary
        response_data = resp.json()
        if "audios" in response_data and len(response_data["audios"]) > 0:
            import base64
            audio_base64 = response_data["audios"][0]
            return base64.b64decode(audio_base64)
        else:
            return resp.content
    except Exception as e:
        raise Exception(f"TTS Error: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="Healthcare Voice Chatbot", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Voice Chatbot")
st.markdown("*Supporting healthcare workers with non-diagnostic administrative and care support*")

# Check for API keys
if not API_KEY_SARVAM or not GOOGLE_API_KEY:
    st.error("⚠️ Please set SARVAM_API_KEY and GOOGLE_API_KEY in your .env file")
    st.stop()

# Language selection
lang_options = {
    "Bengali": "bn-IN",
    "Hindi": "hi-IN",
    "English": "en-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN"
}
selected_lang = st.selectbox("Select output language for audio:", list(lang_options.keys()))
target_lang_code = lang_options[selected_lang]

st.divider()

# Audio recorder
audio_data = mic_recorder(
    start_prompt="🎙️ Press to record your message", 
    stop_prompt="⏹️ Stop recording",
    key="recorder"
)

if audio_data:
    user_audio = audio_data["bytes"]
    
    # Display recorded audio
    st.audio(user_audio, format="audio/wav")
    
    with st.spinner("Transcribing your message..."):
        try:
            # Speech to text
            user_text = stt_from_audio(user_audio)
            
            if user_text:
                st.success(f"**You said:** {user_text}")
                
                # Get response from LLM
                with st.spinner("Generating response..."):
                    reply = get_response(user_text)
                    
                    st.markdown("### 🤖 Assistant Response:")
                    st.info(reply)
                    
                    # Convert reply to speech
                    with st.spinner("Converting to speech..."):
                        try:
                            reply_audio = tts_from_text(reply, lang=target_lang_code)
                            st.audio(reply_audio, format="audio/wav")
                        except Exception as tts_error:
                            st.warning(f"Could not generate audio: {str(tts_error)}")
            else:
                st.warning("No speech detected. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Optional text input fallback
st.divider()
st.markdown("### 💬 Or type your question:")
text_input = st.text_area("Enter your message here:", height=100)

if st.button("Get Response", type="primary"):
    if text_input:
        with st.spinner("Generating response..."):
            try:
                reply = get_response(text_input)
                st.markdown("### 🤖 Assistant Response:")
                st.info(reply)
                
                # Convert to speech
                with st.spinner("Converting to speech..."):
                    try:
                        reply_audio = tts_from_text(reply, lang=target_lang_code)
                        st.audio(reply_audio, format="audio/wav")
                    except Exception as tts_error:
                        st.warning(f"Could not generate audio: {str(tts_error)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please enter a message first.")