from gtts import gTTS
import os
from streamlit_mic_recorder import speech_to_text
import streamlit as st
from groq import Groq

# 1. إعداد الصفحة
st.set_page_config(
    page_title="المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
    layout="centered",
)
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'><a"
    " href='https://www.nosi.gov.eg' target='_blank'"
    " style='text-decoration: none; color: #1E3A8A;'>الهيئة القومية للتأمين"
    " الاجتماعي</a></h1>",
    unsafe_allow_html=True,
)
st.write("---")
st.info(
    "أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي"
    " إجابتك على كافة الاستفسارات المتعلقة بالمعاشات والتأمينات."
)

# 2. إعداد مفتاح الـ API والعميل (Client) المباشر لـ Groq
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = "gsk_mVXibF8ET8Bs3JJ8MG1QGwyddb3FYrtXB7WbDGcxUxh49v3G5u1Id"

client = Groq(api_key=api_key)

# 3. الشريط الجانبي للتسجيل الصوتي
with st.sidebar:
    st.header("🎙️ التحدث الصوتي")
    voice_input = speech_to_text(
        language="ar",
        start_prompt="اضغط للتحدث 🎤",
        stop_prompt="إيقاف التسجيل 🛑",
        key="voice_recorder",
    )
    st.write("---")
    if st.button("🗑️ مسح المحادثة وبدء جديد"):
        st.session_state.messages = []
        st.rerun()

# 4. إدارة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = (
    voice_input
    if voice_input
    else st.chat_input(
        "اكتب استفسارك هنا (مثلاً: ما هي شروط المعاش المبكر؟)..."
    )
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة اللوائح للإجابة..."):
            system_prompt = (
                "بصفتك المساعد الرسمي للهيئة القومية للتأمين الاجتماعي بمصر، "
                "أجب عن استفسارات المواطنين بخصوص المعاشات والتأمينات بدقة، رسمية، "
                "ومستندة للقوانين. وعند ذكر الموقع الإلكتروني استخدم حصرياً:"
                " www.nosi.gov.eg"
            )

            try:
                # استخدام النموذج الحديث والمستقر llama-3.3-70b-versatile
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                )
                reply = chat_completion.choices[0].message.content
            except Exception as e:
                reply = f"حدث خطأ أثناء الاتصال بالخادم: {e}"

            st.markdown(reply)

            # تحويل الرد إلى صوت
            try:
                tts = gTTS(text=reply, lang="ar")
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")
            except Exception:
                pass

    st.session_state.messages.append({"role": "assistant", "content": reply})
