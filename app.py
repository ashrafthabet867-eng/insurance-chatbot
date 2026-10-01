import os
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

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

# التقاط السؤال الصوتي القادم من الرابط إن وجد
query_params = st.query_params
voice_query = query_params.get("v_query", None)

if voice_query:
    prompt = voice_query
    # مسح البارامتر حتى لا يتكرر عند التحديث
    st.query_params.clear()
else:
    # أو الاستفسار المكتوب بالطريقة التقليدية
    prompt = st.chat_input("اكتب استفسارك هنا أو استخدم زر التحدث الصوتي بالأسفل...")

# 2. زر التسجيل الصوتي المباشر وتحديث الرابط لتمرير السؤال لبايثون
st.markdown("### 🎙 التحدث الصوتي للمساعد الذكي:")

voice_html = """
<div style="text-align: center; padding: 10px;">
    <button id="recordButton" onclick="startRecording()" style="background-color: #1E3A8A; color: white; border: none; padding: 14px 28px; font-size: 16px; border-radius: 8px; cursor: pointer; font-family: Tahoma; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        🎙️ اضغط هنا وابدأ التحدث
    </button>
    <p id="statusText" style="margin-top: 12px; color: #1E3A8A; font-weight: bold; font-family: Tahoma;"></p>
</div>

<script>
function startRecording() {
    const statusText = document.getElementById("statusText");
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        statusText.innerText = "متصفحك لا يدعم التحويل الصوتي المباشر، يرجى استخدام الكتابة.";
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'ar-EG';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    statusText.innerText = "جاري الاستماع الآن... تحدث بوضوح 🎙️";

    recognition.onresult = function(event) {
        const speechResult = event.results[0][0].transcript;
        statusText.innerText = "تم التقاط السؤال بنجاح، جاري المعالجة...";
        
        // إعادة توجيه الصفحة مع تمرير النص الصوتي مباشرة لتلقيه في بايثون
        const baseUrl = window.parent.location.href.split('?')[0];
        window.parent.location.href = baseUrl + "?v_query=" + encodeURIComponent(speechResult);
    };

    recognition.onerror = function(event) {
        statusText.innerText = "تعذر التعرف على الصوت بوضوح، حاول مرة أخرى.";
    };

    recognition.start();
}
</script>
"""
components.html(voice_html, height=130)

# معالجة السؤال (سواء كُتب أو جاء من الصوت) وإرساله للوكيل الذكي
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

# الشريط الجانبي
with st.sidebar:
    st.header("خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption("جميع الحقوق محفوظة © الهيئة القومية للتأمين الاجتماعي 2026")
