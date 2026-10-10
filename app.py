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

from rag import load_index, tokenize

# ----------------------------------------------------------------------
# الإعدادات (تأكد من أسماء النماذج في قائمة Groq الحالية)
# ----------------------------------------------------------------------
CHAT_MODEL = "qwen/qwen3.8-27b"
VISION_MODEL = "qwen/qwen3.8-27b"
STT_MODEL = "whisper-large-v3"
MAX_HISTORY = 10          # عدد آخر الرسائل المرسلة للنموذج
MAX_PDF_PAGES = 10
MAX_DOC_CHARS = 8000
MAX_TTS_CHARS = 1500
TOP_K = 8                 # عدد النصوص المرجعية المرسلة للنموذج
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

SYSTEM_PROMPT = (
    "أنت مساعد ذكي يجيب عن استفسارات المواطنين بخصوص المعاشات والتأمينات "
    "الاجتماعية في مصر. ستصلك مع كل سؤال «نصوص مرجعية» مرقمة [1] [2] ... "
    "التزم بالقواعد التالية:\n"
    "1. أجب بالعربية بشكل رسمي ومختصر وواضح.\n"
    "2. في الأسئلة عن الأحكام والشروط والمبالغ والمواعيد، اعتمد على النصوص "
    "المرجعية فقط، واذكر رقم المصدر بين أقواس مربعة بعد كل معلومة مثل [1]. "
    "لا تخترع أرقامًا أو نسبًا أو شروطًا أو أرقام مواد.\n"
    "3. إذا لم تحتوِ النصوص المرجعية على الإجابة، قل صراحة إنك لم تجد ذلك "
    "في المصادر المتاحة، وانصح المواطن بمراجعة أقرب مكتب تأمينات أو الخط "
    "الساخن للهيئة. لا تجب من ذاكرتك في هذه الحالة.\n"
    "4. للتحيات والأسئلة العامة عن المساعد نفسه، أجب مباشرة دون مصادر.\n"
    "5. للأسئلة المتعلقة بحالة شخصية محددة، وضّح أن القرار النهائي للهيئة.\n"
    "6. إذا أُرفق مستند من المواطن، اعتمد على محتواه إلى جانب النصوص المرجعية.\n"
    "7. إذا ظهر الرمز ⟦؟⟧ مكان رقم في النصوص المرجعية فهذا رقم لم يتم التحقق منه: "
    "لا تذكر أي رقم أو نسبة أو مبلغ أو تاريخ مكانه ولا تخمّنه، وقل صراحة إن الرقم "
    "الدقيق يجب مراجعته في النص الرسمي للقانون أو في مكتب التأمينات.\n"
    "8. النصوص المرجعية التي مصدرها key_facts موثّقة يدويًا من النص الرسمي: "
    "اعتمد عليها في الأرقام والنسب بدل أي نص آخر فيه ⟦؟⟧."
)

ANSWER_MODES = {
    "مختصرة": (
        "نمط الإجابة: مختصر جدًا. أجب في جملتين إلى ثلاث جمل وبحد أقصى 50 كلمة، "
        "بالإجابة المباشرة فقط، واذكر رقم المصدر [n]. ممنوع استخدام القوائم والنقاط أو "
        "سرد الشروط والتفاصيل. وإن كانت هناك شروط أو استثناءات مهمة فاكتفِ بقول إنها "
        "موجودة وإن التفصيل متاح باختيار «مفصلة» من الشريط الجانبي."
    ),
    "مفصلة": (
        "نمط الإجابة: مفصّل. ابدأ بخلاصة من سطر واحد، ثم اشرح الشروط والخطوات والأرقام "
        "والاستثناءات الواردة في النصوص المرجعية بنقاط منظمة، واذكر رقم المادة والمصدر [n] "
        "لكل معلومة. لا تضف أي معلومة غير موجودة في النصوص المرجعية."
    ),
}

# ----------------------------------------------------------------------
# إعداد الصفحة
# ----------------------------------------------------------------------
APP_TITLE = "المساعد الذكي للهيئة القومية للتأمين الاجتماعي"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_logo():
    """الشعار: ارفع صورتك باسم logo.png (أو logo.jpg) بجانب app.py."""
    for name in ("logo.png", "logo.jpg", "logo.jpeg"):
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    return None


def logo_html(height=60):
    path = find_logo()
    if not path:
        return "🏛 "
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f"<img src='data:{mime};base64,{b64}' "
            f"style='height:{height}px; vertical-align:middle; margin-right:12px;'>")


def page_icon():
    path = find_logo()
    if path:
        try:
            from PIL import Image
            return Image.open(path)
        except Exception:
            pass
    return "🏛"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=page_icon(),
    layout="centered",
)

st.markdown(
    f"<h1 style='text-align: center; color: #1E3A8A; font-size: 2rem; line-height: 1.6;'>"
    f"{logo_html()}{APP_TITLE}</h1>",
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


@st.cache_resource
def get_index():
    return load_index(KNOWLEDGE_DIR)


index = get_index()

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
    answer_mode = st.radio("📝 نمط الإجابة", list(ANSWER_MODES), index=0, horizontal=True)

    if index is None:
        st.warning("قاعدة المعرفة فارغة: أضف ملفات القانون (txt/pdf) في مجلد knowledge.")
    else:
        st.caption(f"📚 قاعدة المعرفة: {len(index.chunks)} جزءًا قانونيًا")

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
    text = re.sub(r"\[\d+\]|\u27e6\u061f\u27e7", "", text)  # citations and unverified markers
    clean = re.sub(r"[*_#`>|~\-]+", " ", text)[:MAX_TTS_CHARS]
    try:
        buf = io.BytesIO()
        gTTS(text=clean, lang="ar").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def retrieve():
    """يبحث بالسؤال الحالي؛ ولو كان قصيرًا (سؤال متابعة) يضيف السؤال السابق."""
    if index is None:
        return []
    user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    query = user_msgs[-1]
    if len(tokenize(query)) < 3 and len(user_msgs) > 1:
        query = user_msgs[-2] + " " + query
    return index.search(query, k=TOP_K)


def format_context(hits) -> str:
    if not hits:
        return "(لم يتم العثور على نصوص ذات صلة في قاعدة المعرفة.)"
    return "\n\n".join(
        f"[{i}] ({c['source']} - {c['label']})\n{c['text']}"
        for i, (c, _) in enumerate(hits, 1)
    )


def build_llm_messages(context: str, mode: str):
    msgs = [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + ANSWER_MODES[mode])]
    history = st.session_state.messages[-MAX_HISTORY:]
    for i, m in enumerate(history):
        content = m.get("llm_content", m["content"])
        if m["role"] == "user":
            if i == len(history) - 1:  # السؤال الحالي فقط يأخذ النصوص المرجعية
                content += f"\n\n--- النصوص المرجعية ---\n{context}"
            msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    return msgs


def render_message(m, autoplay=False):
    with st.chat_message(m["role"]):
        for img in m.get("images", []):
            st.image(img, width=250)
        st.markdown(m["content"])
        if m.get("audio"):
            st.audio(m["audio"], format="audio/mp3", autoplay=autoplay)
        if m.get("sources"):
            with st.expander(f"📚 المصادر ({len(m['sources'])})"):
                for i, s in enumerate(m["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']} — {s['label']}** (تشابه: {s['score']:.1f})")
                    st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


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
    with st.spinner("جاري البحث في النصوص القانونية وإعداد الإجابة..."):
        hits = retrieve()
        try:
            reply = strip_think(
                chat_model.invoke(build_llm_messages(format_context(hits), answer_mode)).content
            )
        except Exception:
            reply = "عذرًا، حدث خطأ مؤقت. برجاء المحاولة مرة أخرى."
        audio_bytes = make_tts(reply) if read_aloud else None

    assistant_msg = {
        "role": "assistant",
        "content": reply,
        "audio": audio_bytes,
        "sources": [
            {"source": c["source"], "label": c["label"], "text": c["text"], "score": s}
            for c, s in hits
        ],
    }
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg, autoplay=True)
