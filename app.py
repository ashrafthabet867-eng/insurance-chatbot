import base64
import io
import os
import re

import streamlit as st
from groq import Groq
from gtts import gTTS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pypdf import PdfReader

# ----------------------------------------------------------------------
# الإعدادات (تأكد من أسماء النماذج في قائمة Groq الحالية)
# ----------------------------------------------------------------------
CHAT_MODEL = "qwen/qwen3.8-27b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
STT_MODEL = "whisper-large-v3"
MAX_HISTORY = 10          # عدد آخر الرسائل المرسلة للنموذج
MAX_PDF_PAGES = 10
MAX_DOC_CHARS = 8000
MAX_TTS_CHARS = 1500

SYSTEM_PROMPT = (
    "أنت مساعد ذكي يجيب عن استفسارات المواطنين بخصوص المعاشات والتأمينات "
    "الاجتماعية في مصر. التزم بالقواعد التالية:\n"
    "1. أجب بالعربية بشكل رسمي ومختصر وواضح.\n"
    "2. استند فقط إلى معلومات تعرفها بدقة من اللوائح والقوانين المعمول بها. "
    "لا تخترع أرقامًا أو نسبًا أو شروطًا.\n"
    "3. إذا لم تكن متأكدًا، قل ذلك صراحة وانصح المواطن بمراجعة أقرب مكتب "
    "تأمينات أو الخط الساخن للهيئة.\n"
    "4. للأسئلة المتعلقة بحالة شخصية محددة، وضّح أن القرار النهائي للهيئة.\n"
    "5. إذا أُرفق مستند، اعتمد على محتواه في الإجابة."
)

# ----------------------------------------------------------------------
# إعداد الصفحة
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="المساعد الذكي - التأمين الاجتماعي",
    page_icon="🏛",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🏛 المساعد الذكي للتأمين الاجتماعي</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center; color: #4B5563;'>للرد على استفسارات المواطنين</h3>",
    unsafe_allow_html=True,
)
st.write("---")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def get_clients():
    chat = ChatGroq(model=CHAT_MODEL, temperature=0.0)
    vision = ChatGroq(model=VISION_MODEL, temperature=0.0)
    stt = Groq()
    return chat, vision, stt


chat_model, vision_model, stt_client = get_clients()

# ----------------------------------------------------------------------
# حالة الجلسة
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ التحكم والخدمات")
    if st.button("🗑 بدء محادثة جديدة", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    read_aloud = st.toggle("🔊 قراءة الرد بصوت مسموع", value=True)

    st.write("---")
    st.markdown("### خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption(
        "⚠️ لا تدخل بياناتك الحساسة (رقم قومي، أرقام حسابات) في المحادثة. "
        "المساعد تجريبي، والمرجع النهائي هو الجهة الرسمية."
    )

st.info(
    "أهلاً بك. يمكنك الكتابة، أو الضغط على 🎤 للتحدث، "
    "أو إرفاق صورة أو ملف PDF من مربع المحادثة بالأسفل."
)


# ----------------------------------------------------------------------
# دوال مساعدة
# ----------------------------------------------------------------------
def transcribe(audio_file) -> str:
    """تحويل الصوت إلى نص باستخدام Whisper على Groq."""
    result = stt_client.audio.transcriptions.create(
        file=("voice.wav", audio_file.getvalue()),
        model=STT_MODEL,
        language="ar",
        response_format="text",
    )
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return text.strip()


def read_image(file) -> str:
    """قراءة نص الصورة ووصفها عبر نموذج رؤية."""
    b64 = base64.b64encode(file.getvalue()).decode()
    mime = file.type or "image/jpeg"
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "اقرأ كل النص الموجود في هذه الصورة كما هو بدقة، "
                "ثم لخّص محتواها باختصار.",
            },
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    return vision_model.invoke([msg]).content


def read_pdf(file) -> str:
    """استخراج نص من PDF (لا يعمل مع الملفات الممسوحة ضوئيًا)."""
    reader = PdfReader(io.BytesIO(file.getvalue()))
    pages = reader.pages[:MAX_PDF_PAGES]
    return "\n".join((p.extract_text() or "") for p in pages)[:MAX_DOC_CHARS]


def extract_attachments(files):
    """يرجع (نص المستندات للنموذج، قائمة صور للعرض)."""
    doc_parts, images = [], []
    for f in files:
        try:
            if f.type == "application/pdf":
                text = read_pdf(f)
                label = f"ملف PDF: {f.name}"
            else:
                text = read_image(f)
                label = f"صورة: {f.name}"
                images.append(f.getvalue())
            if text.strip():
                doc_parts.append(f"[{label}]\n{text}")
            else:
                doc_parts.append(f"[{label}]\n(تعذّرت قراءة محتوى هذا الملف)")
        except Exception:
            doc_parts.append(f"[{f.name}]\n(حدث خطأ أثناء قراءة الملف)")
    return "\n\n".join(doc_parts), images


def strip_think(text: str) -> str:
    """إخفاء تفكير نماذج Qwen3 إن ظهر داخل وسوم think."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def make_tts(text: str):
    """تحويل الرد إلى صوت في الذاكرة بدون ملفات على القرص."""
    clean = re.sub(r"[*_#`>|~\-]+", " ", text)[:MAX_TTS_CHARS]
    try:
        buf = io.BytesIO()
        gTTS(text=clean, lang="ar").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def build_llm_messages():
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in st.session_state.messages[-MAX_HISTORY:]:
        content = m.get("llm_content", m["content"])
        msgs.append(HumanMessage(content=content) if m["role"] == "user" else AIMessage(content=content))
    return msgs


def render_message(m, autoplay=False):
    with st.chat_message(m["role"]):
        for img in m.get("images", []):
            st.image(img, width=250)
        st.markdown(m["content"])
        if m.get("audio"):
            st.audio(m["audio"], format="audio/mp3", autoplay=autoplay)


# ----------------------------------------------------------------------
# عرض المحادثات السابقة
# ----------------------------------------------------------------------
for m in st.session_state.messages:
    render_message(m)

# ----------------------------------------------------------------------
# مربع المحادثة الرئيسي: نص + مايك + إرفاق ملفات
# ----------------------------------------------------------------------
chat_value = st.chat_input(
    "اكتب استفسارك أو اضغط على المايك للتحدث...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "pdf"],
    accept_audio=True,
)

if chat_value:
    text = (chat_value.text or "").strip()
    files = list(getattr(chat_value, "files", None) or [])
    audio = getattr(chat_value, "audio", None)

    # 1) تحويل الصوت إلى نص
    if audio:
        with st.spinner("جاري تحويل صوتك إلى نص..."):
            try:
                spoken = transcribe(audio)
            except Exception:
                spoken = ""
                st.error("تعذّر تحويل الصوت إلى نص. حاول مرة أخرى أو اكتب سؤالك.")
        text = f"{text} {spoken}".strip()

    # 2) قراءة المرفقات
    doc_text, images = ("", [])
    if files:
        with st.spinner("جاري قراءة المرفقات..."):
            doc_text, images = extract_attachments(files)

    if not text and not doc_text:
        st.stop()

    display_text = text or "يرجى قراءة ومراجعة الملف المرفق."
    llm_text = display_text
    if doc_text:
        llm_text += f"\n\nالمستندات المرفقة:\n{doc_text}"

    user_msg = {
        "role": "user",
        "content": ("🎤 " if audio else "") + display_text,
        "llm_content": llm_text,
        "images": images,
    }
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # 3) رد المساعد
    with st.chat_message("assistant"):
        with st.spinner("جاري إعداد الإجابة..."):
            try:
                reply = strip_think(chat_model.invoke(build_llm_messages()).content)
            except Exception:
                reply = "عذرًا، حدث خطأ مؤقت. برجاء المحاولة مرة أخرى."
        st.markdown(reply)

        audio_bytes = make_tts(reply) if read_aloud else None
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "audio": audio_bytes}
    )
