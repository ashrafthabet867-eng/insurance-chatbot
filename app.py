import os
import streamlit as st
from gtts import gTTS
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
    page_icon="🏛",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏛 الهيئة القومية للتأمين الاجتماعي</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563;'>البوابة الذكية للرد على استفسارات وشكاوى المواطنين</h3>", unsafe_allow_html=True)
st.write("---")

st.info("أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي إجابتك على كافة الاستفسارات المتعلقة بالمعاشات والخدمات الرسمية صوتياً أو كتابياً.")

# إعداد مفتاح الـ API
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0
)

tools = []
agent_executor = create_react_agent(model, tools)

# إدارة سجل المحادثات
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio_path" in message:
            st.audio(message["audio_path"], format='audio/mp3')

# 2. زر التسجيل الصوتي الأصلي المدمج (يعمل بصلاحيات المتصفح مباشرة دون عزل الـ iframe)
st.markdown("### 🎙 التحدث الصوتي للمساعد الذكي:")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # استخدام مكتبة mic_recorder المباشرة لتسجيل الصوت بدقة
    audio_data = mic_recorder(
        start_prompt="🎙️ اضغط هنا وابدأ التحدث",
        stop_prompt="⏹️ اضغط للإيقاف وإرسال الصوت",
        just_once=False,
        key='mic_recorder'
    )

prompt = None

# إذا تم تسجيل صوت بنجاح، نقوم بتحويله إلى نص عبر مكتبة speech_recognition
if audio_data:
    audio_bytes = audio_data['bytes']
    
    # حفظ الملف الصوتى مؤقتاً لمعالجته
    temp_audio_file = "temp_input_audio.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_bytes)
        
    r = sr.Recognizer()
    try:
        with sr.AudioFile(temp_audio_file) as source:
            audio_content = r.record(source)
            # التعرف على الصوت باللغة العربية
            recognized_text = r.recognize_google(audio_content, language="ar-EG")
            if recognized_text:
                prompt = recognized_text
    except Exception as e:
        st.warning("تعذر التعرف على الكلمات بوضوح، يرجى المحاولة مرة أخرى أو الكتابة في الأسفل.")

# حقل الإدخال النصي التقليدي كخيار بديل أو مكمل
text_prompt = st.chat_input("أو اكتب استفسارك هنا (مثلاً: ما هي شروط المعاش المبكر؟)...")

if text_prompt:
    prompt = text_prompt

# معالجة السؤال وإرساله للوكيل الذكي
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة القوانين واللوائح للإجابة بدقة..."):
            
            system_instruction = (
                "بصفتك المساعد الرسمي للهيئة القومية للتأمين الاجتماعي بمصر، "
                "أجب عن استفسارات المواطنين بخصوص المعاشات والتأمينات بدقة، "
                "رسمية، ومستندة إلى اللوائح والقوانين الرسمية المعمول بها: "
            )
            
            response = agent_executor.invoke({
                "messages": [("user", f"{system_instruction} {prompt}")]
            })
            
            reply = response["messages"][-1].content
            st.markdown(reply)
            
            # تحويل الرد النصي إلى صوت وتشغيله
            try:
                tts = gTTS(text=reply, lang='ar')
                audio_file_path = "response_audio.mp3"
                tts.save(audio_file_path)
                st.audio(audio_file_path, format='audio/mp3', autoplay=True)
            except Exception as e:
                audio_file_path = None

    message_data = {"role": "assistant", "content": reply}
    if 'audio_file_path' in locals() and audio_file_path:
        message_data["audio_path"] = audio_file_path
        
    st.session_state.messages.append(message_data)

# شريط جانبي
with st.sidebar:
    st.header("خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption("جميع الحقوق محفوظة © الهيئة القومية للتأمين الاجتماعي 2026")
