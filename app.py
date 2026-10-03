import os
import streamlit as st
from gtts import gTTS
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
    page_icon="🏛",
    layout="centered"
)

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏛 الهيئة القومية للتأمين الاجتماعي</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563;'>البوابة الذكية للرد على استفسارات وشكاوى المواطنين</h3>", unsafe_allow_html=True)
st.write("---")

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

# شريط جانبي للخدمات وزر محادثة جديدة
with st.sidebar:
    st.header("⚙️ التحكم والخدمات")
    
    # زر بدء محادثة جديدة
    if st.button("🗑️️ بدء محادثة جديدة", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.write("---")
    st.markdown("### خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption("جميع الحقوق محفوظة © الهيئة القومية للتأمين الاجتماعي 2026")

st.info("أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي إجابتك على كافة الاستفسارات وقراءة المستندات والملفات المرفقة.")

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message and message["image"]:
            st.image(message["image"], width=250)
        st.markdown(message["content"])
        if message["role"] == "assistant" and "audio_path" in message:
            st.audio(message["audio_path"], format='audio/mp3')

# أداة رفع الصور والمستندات مباشرة في الواجهة
uploaded_file = st.file_uploader("📂 ارفع صورة أو مستنداً للاستفسار عنه (اختياري):", type=["png", "jpg", "jpeg", "pdf"])

# صندوق إدخال الاستفسارات
prompt = st.chat_input("اكتب استفسارك هنا...")

if prompt or uploaded_file:
    user_content = prompt if prompt else "يرجى قراءة ومراجعة الملف المرفق."
    
    # حفظ ومعاينة الصورة أو الملف المرفق إن وجد
    image_to_display = None
    if uploaded_file:
        try:
            image_to_display = Image.open(uploaded_file)
        except Exception:
            image_to_display = None

    # تخزين وعرض رسالة المستخدم
    message_data = {"role": "user", "content": user_content, "image": image_to_display}
    st.session_state.messages.append(message_data)
    
    with st.chat_message("user"):
        if image_to_display:
            st.image(image_to_display, width=250)
        st.markdown(user_content)

    # معالجة الرد عبر المساعد الذكي
    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة البيانات واللوائح للإجابة بدقة..."):
            
            system_instruction = (
                "بصفتك المساعد الرسمي للهيئة القومية للتأمين الاجتماعي بمصر، "
                "أجب عن استفسارات المواطنين بخصوص المعاشات والتأمينات بدقة، "
                "رسمية، ومستندة إلى اللوائح والقوانين الرسمية المعمول بها: "
            )
            
            response = agent_executor.invoke({
                "messages": [("user", f"{system_instruction} {user_content}")]
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

    assistant_message_data = {"role": "assistant", "content": reply}
    if 'audio_file_path' in locals() and audio_file_path:
        assistant_message_data["audio_path"] = audio_file_path
        
    st.session_state.messages.append(assistant_message_data)
