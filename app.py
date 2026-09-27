import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

# 1. إعدادات صفحة الويب والتصميم
st.set_page_config(
    page_title="المساعد الذكي - الهيئة القومية للتأمين الاجتماعي",
    page_icon="🏛️",
    layout="centered"
)

# تخصيص واجهة المستخدم وعنوان الهيئة
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏛️ الهيئة القومية للتأمين الاجتماعي</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B5563;'>البوابة الذكية للرد على استفسارات وشكاوى المواطنين</h3>", unsafe_allow_html=True)
st.write("---")

# ترحيب بالمرتاد وإرشادات الاستخدام
st.info("أهلاً بك عزيزي المواطن. أنا المساعد الذكي الرقمي للهيئة، ومهمتي هي إجابتك على كافة الاستفسارات المتعلقة بالمعاشات، الاشتراكات التأمينية، والخدمات الرسمية.")

# 2. إعداد مفتاح الـ API ونموذج التشغيل المعتمد لديك
os.environ["GROQ_API_KEY"] = "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"

model = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0
)

# 3. إعداد أدوات البحث والوكيل الذكي مع توجيه دقيق لدوره
tools = [DuckDuckGoSearchRun(name="Search")]
agent_executor = create_react_agent(model, tools)

# 4. إدارة سجل المحادثة والرسائل في الواجهة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. صندوق إدخال رسائل المواطنين
if prompt := st.chat_input("اكتب استفسارك أو شكواك هنا (مثلاً: ما هي شروط المعاش المبكر؟)..."):
    
    # حفظ وعرض سؤال المواطن
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # معالجة الرد من خلال الوكيل الذكي
    with st.chat_message("assistant"):
        with st.spinner("جاري مراجعة القوانين واللوائح للإجابة بدقة..."):
            
            # توجيه سياقي صارم للوكيل ليلتزم بدور ممثل الهيئة
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
            
    # حفظ رد المساعد في الذاكرة
    st.session_state.messages.append({"role": "assistant", "content": reply})

# شريط جانبي بمعلومات إضافية
with st.sidebar:
    st.header("خدمات سريعة")
    st.markdown("- الاستعلام عن الرقم التأميني")
    st.markdown("- شروط استحقاق المعاش")
    st.markdown("- مواعيد صرف المعاشات")
    st.write("---")
    st.caption("جميع الحقوق محفوظة © الهيئة القومية للتأمين الاجتماعي 2026")