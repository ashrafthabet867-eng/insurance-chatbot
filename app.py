from gtts import gTTS
import os
from streamlit_mic_recorder import speech_to_text
import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent

# 1. إعداد واجهة الصفحة
st.set_page_config(
    page_config={
        "page_title": "المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
        "page_icon": "logo.png",
        "layout": "centered",
    }
)

# تخصيص واجهة المستخدم وعنوان الهيئة مع الرابط الرسمي الصحيح
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'><a"
    " href='https://www.nosi.gov.eg' target='_blank'"
    " style='text-decoration: none; color: #1E3A8A;'>الهيئة القومية للتأمين"
    " الاجتماعي</a></h1>",
    unsafe_allow_html=True,
)
st.write("---")

# ترحيب بالمُرتاد وإرشادات الاستخدام
st.info(
    "أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي"
    " إجابتك على كافة الاستفسارات المتعلقة بالمعاشات، الاشتراكات التأمينية،"
    " والخدمات الرسمية."
)

# 2. إعداد مفتاح الـ API (يُفضل سحبه من Streamlit Secrets للأمان)
if "GROQ_API_KEY" in st.secrets:
  os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
  os.environ["GROQ_API_KEY"] = (
      "gsk_mVXibF8ET8Bs3JJ8MG1QGwyddb3FYrtXB7WbDGcxUxh49v3G5u1Id"
  )

model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0)

# 3. إعداد أدوات البحث والوكيل الذكي
tools = []
agent_executor = create_react_agent(model, tools)

# 4. إضافة زر التسجيل الصوتي في الشريط الجانبي (Sidebar)
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

# 5. إعداد سجل المحادثة في الواجهة
if "messages" not in st.session_state:
  st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# تحديد مصدر الإدخال (صوتي أم كتابي)
prompt = (
    voice_input
    if voice_input
    else st.chat_input(
        "اكتب استفسارك أو شكواك هنا (مثلاً: ما هي شروط المعاش المبكر؟)..."
    )
)

# 6. معالجة الإدخال والرد من النموذج
if prompt:
  # حفظ وعرض سؤال المواطن
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # معالجة الرد من خلال الوكيل الذكي
  with st.chat_message("assistant"):
    with st.spinner("جاري مراجعة القوانين واللوائح للإجابة بدقة..."):
      system_instruction = (
          "بصفتك المساعد الرسمي للهيئة القومية للتأمين الاجتماعي بمصر، "
          "أجب عن استفسارات المواطنين بخصوص المعاشات والتأمينات بدقة، رسمية، "
          "ومستندة إلى اللوائح والقوانين الرسمية. "
          "وعند ذكر الموقع الإلكتروني الرسمي للهيئة، يجب عليك دائماً استخدام"
          " الرابط الصحيح حصرياً: www.nosi.gov.eg"
      )
      response = agent_executor.invoke(
          {"messages": [("user", f"{system_instruction} {prompt}")]}
      )
      reply = response["messages"][-1].content
      st.markdown(reply)

      # تحويل الرد إلى صوت مسموع
      try:
        tts = gTTS(text=reply, lang="ar")
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")
      except Exception:
        pass

  # حفظ رد المساعد في الذاكرة
  st.session_state.messages.append({"role": "assistant", "content": reply})
