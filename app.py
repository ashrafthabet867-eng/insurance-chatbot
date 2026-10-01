import os
import io
import streamlit as st
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# 1. إعدادات صفحة الويب والتصميم
st.set_page_config(
    page_title="المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
    page_icon="🏛",
    layout="centered"
)

# تخصيص واجهة المستخدم وعنوان الهيئة
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏛 الهيئة القومية للتأمين الاجتماعي</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563;'>البوابة الذكية للرد على استفسارات وشكاوى المواطنين</h3>", unsafe_allow_html=True)
st.write("---")

# ترحيب بالمرتاد وإرشادات الاستخدام
st.info("أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي إجابتك على كافة الاستفسارات المتعلقة بالمعاشات والخدمات الرسمية صوتياً أو كتابياً.")

# استدعاء مفتاح الـ API بأمان تام من أسرار Streamlit
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0
)

# 3. إعداد أدوات البحث والوكيل الذكي
tools = []
agent_executor = create_react_agent(model, tools)

# 4. إدارة سجل المحادثة والرسائل في الواجهة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio_path" in message:
            st.audio(message["audio_path"], format='audio/mp3')

# 5. قسم التسجيل الصوتي المباشر والتفريغ التلقائي
st.markdown("### 🎙️ التحدث الصوتي للمساعد الذكي:")
audio_data = mic_recorder(
    start_prompt="اضغط هنا لبدء التحدث",
    stop_prompt="إيقاف التسجيل",
    just_once=True,
    key='voice_recorder'
)

prompt = st.chat_input("أو اكتب استفسارك هنا (مثلاً: ما هي شروط المعاش المبكر؟)...")

# التقاط الصوت المسجل وتحويله إلى نص عربي تلقائياً
if audio_data:
    try:
        with st.spinner("جاري معالجة الصوت وتفريغه إلى نص..."):
            audio_bytes = audio_data['bytes']
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_content = r.record(source)
                recognized_text = r.recognize_google(audio_content, language="ar-EG")
                if recognized_text:
                    prompt = recognized_text
                    st.success(f"تم استقبال سؤالك الصوتي بنجاح: {prompt}")
    except Exception as e:
        st.warning("تعذر التعرف على الكلمات بوضوح، يرجى إعادة محاولة التحدث أو الكتابة في صندوق الدردشة.")

if prompt:
    # حفظ وعرض السؤال (سواء تم إدخاله صوتی أو كتابةً)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # معالجة الرد من خلال الوكيل الذكي
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
            
            # --- تحويل الرد النصي إلى صوت وتشغيله تلقائياً ---
            try:
                tts = gTTS(text=reply, lang='ar')
                audio_file_path = "response_audio.mp3"
                tts.save(audio_file_path)
                st.audio(audio_file_path, format='audio/mp3', autoplay=True)
            except Exception as e:
                audio_file_path = None

    # حفظ رد المساعد في الذاكرة مع مسار الصوت
    message_data = {"role": "assistant", "content": reply}
    if 'audio_file_path' in locals() and audio_file_path:
        message_data["audio_path"] = audio_file_path
        
    st.session_state.messages.append(message_data)

# شريط جانبي بمعلومات إضافية
with st.sidebar:
    st.header("خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption("جميع الحقوق محفوظة © الهيئة القومية للتأمين الاجتماعي 2026")
