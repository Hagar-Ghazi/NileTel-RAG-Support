"""
NileTel RAG — Streamlit Customer Chat UI
==========================================
Customer-facing chat with ticket info collection form
Run:
    streamlit run app.py
"""

import requests
import streamlit as st

# Config 

API_URL     = "http://localhost:8000"
APP_TITLE   = "NileTel — مساعد الدعم الفني"
AVATAR_BOT  = "💬"
AVATAR_USER = "👤"

GOVERNORATES = [
    "القاهرة", "الجيزة", "الإسكندرية", "الدقهلية", "البحيرة",
    "الغربية", "المنوفية", "القليوبية", "الشرقية", "الفيوم",
    "بني سويف", "المنيا", "أسيوط", "سوهاج", "قنا", "الأقصر",
    "أسوان", "البحر الأحمر", "شمال سيناء", "جنوب سيناء",
    "مطروح", "الوادي الجديد", "السويس", "الإسماعيلية", "بورسعيد",
]

PROBLEM_TYPES = [
    "انقطاع كامل في الإنترنت",
    "بطء شديد في السرعة",
    "عطل في الراوتر / ONT",
    "مشكلة في الفاتورة أو الرسوم",
    "أخرى",
]

SERVICE_TYPES = ["FTTH (فايبر)", "ADSL", "موبايل / 4G"]

# Page setup 
st.set_page_config(page_title=APP_TITLE, page_icon="📡", layout="centered")

# CSS 
st.markdown("""
<style>
    /* RTL */
    .stChatMessage p, .stChatMessage div { direction: rtl; text-align: right; }
    input[type="text"], textarea, select  { direction: rtl; text-align: right; }

    /* Header */
    .nt-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #0f6e56 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .nt-header h1 { color: white; font-size: 1.3rem; margin: 0; font-weight: 600; }
    .nt-header p  { color: rgba(255,255,255,0.75); font-size: 0.85rem; margin: 2px 0 0; }
    .nt-badge {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(255,255,255,0.15); color: white;
        font-size: 0.72rem; padding: 2px 10px; border-radius: 20px; margin-top: 5px;
    }
    .nt-dot { color: #5DCAA5; }

    /* Ticket card */
    .nt-ticket {
        background: #f0fdf4; border: 1.5px solid #5DCAA5;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-top: 0.5rem; direction: rtl;
    }
    .nt-ticket h4 { color: #0f6e56; margin: 0 0 5px 0; font-size: 0.95rem; }
    .nt-ticket p  { color: #333; margin: 0; font-size: 0.85rem; line-height: 1.7; }

    /* Info collection form card */
    .nt-form-card {
        background: #fff8f0; border: 1.5px solid #EF9F27;
        border-radius: 10px; padding: 1.2rem 1.4rem;
        margin-top: 0.5rem; direction: rtl;
    }
    .nt-form-card h4 { color: #854F0B; margin: 0 0 8px 0; font-size: 0.95rem; }

    /* Source pill */
    .nt-src {
        display: inline-block; background: #e8f4f8; color: #185FA5;
        font-size: 0.72rem; padding: 2px 10px; border-radius: 20px;
        margin: 2px 3px; direction: ltr;
    }

    /* Latency */
    .nt-lat { font-size: 0.72rem; color: #999; margin-top: 4px; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Header 

st.markdown("""
<div class="nt-header">
    <div style="font-size:2.2rem;">📡</div>
    <div>
        <h1>NileTel — مساعد الدعم الفني</h1>
        <p>اسألني عن باقاتك فواتيرك أو أي مشكلة تقنية</p>
        <div class="nt-badge"><span class="nt-dot">●</span> متصل الآن</div>
    </div>
</div>
""", unsafe_allow_html=True)



# Health check
@st.cache_data(ttl=30)
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("engine_ready", False)
    except Exception:
        return False

if not check_health():
    st.error("⚠️ الـ API مش شغال — تأكد إن `uvicorn api:app` شغال على port 8000")
    st.stop()




# Session state
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role":    "assistant",
            "content":  "أهلاً بيك في NileTel! 😊 أنا مساعدك الذكي \nاسألني عن الباقات الفواتير أو أي مشكلة تقنية",
            "route":   "greeting",
            "sources": [],
            "latency": None,
            "ticket":  None,
        }]
    if "collecting_info" not in st.session_state:
        st.session_state.collecting_info = False
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""

init_state()

# Helpers 

def add_message(role, content, route=None, sources=None, latency=None, ticket=None):
    st.session_state.messages.append({
        "role":    role,
        "content": content,
        "route":   route,
        "sources": sources or [],
        "latency": latency,
        "ticket":  ticket,
    })

def render_sources(sources):
    if sources:
        html = "".join(f'<span class="nt-src">📄 {s.replace(".md","")}</span>' for s in sources)
        st.markdown(f'<div style="margin-top:6px">{html}</div>', unsafe_allow_html=True)

def render_latency(latency):
    if latency:
        st.markdown(f'<div class="nt-lat">⚡ {latency} ms</div>', unsafe_allow_html=True)

def render_ticket_card(ticket: dict):
    tid = ticket.get("ticket_id", "—")
    st.markdown(f"""
    <div class="nt-ticket">
        <h4>تم رفع التذكرة بنجاح</h4>
        <p>
            رقم التذكرة: <strong>{tid}</strong><br>
            فريق الدعم الفني هيتواصل معاك في أقرب وقت على رقم الموبايل المسجل.<br>
            يمكنك متابعة التذكرة برقم: <strong>{tid}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)



# Render chat history
for msg in st.session_state.messages:
    avatar = AVATAR_BOT if msg["role"] == "assistant" else AVATAR_USER
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("ticket"):
            render_ticket_card(msg["ticket"])
        render_sources(msg.get("sources", []))
        render_latency(msg.get("latency"))



# Ticket Info Collection Form 
if st.session_state.collecting_info:
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        st.markdown("""
        <div class="nt-form-card">
            <h4>📋 محتاج بعض البيانات عشان أرفع التذكرة</h4>
        </div>
        """, unsafe_allow_html=True)

        with st.form("ticket_form", clear_on_submit=True):
            st.markdown("##### 🔴 بيانات أساسية")
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("الاسم بالكامل *", placeholder="مثال: أحمد محمد علي")
            with col2:
                phone = st.text_input("رقم الموبايل *", placeholder="01xxxxxxxxx")

            col3, col4 = st.columns(2)
            with col3:
                account = st.text_input("رقم الحساب / الخط *", placeholder="مثال: NTL-123456")
            with col4:
                gov = st.selectbox("المحافظة *", ["اختر المحافظة"] + GOVERNORATES)

            st.markdown("##### 🟡 تفاصيل المشكلة")
            col5, col6 = st.columns(2)
            with col5:
                problem = st.selectbox("نوع المشكلة *", ["اختر نوع المشكلة"] + PROBLEM_TYPES)
            with col6:
                service = st.selectbox("نوع الخدمة *", ["اختر نوع الخدمة"] + SERVICE_TYPES)

            col7, col8 = st.columns(2)
            with col7:
                since = st.text_input("المشكلة من امتى؟ *", placeholder="مثال: من امبارح الصبح")
            with col8:
                address = st.text_input("العنوان بالتفصيل *", placeholder="مثال: شارع النيل، ش 5، شقة 12")

            submitted = st.form_submit_button("🎫 رفع التذكرة", use_container_width=True)

        if submitted:
            # Validate required fields
            errors = []
            if not name.strip():           errors.append("الاسم")
            if not phone.strip():          errors.append("رقم الموبايل")
            if not account.strip():        errors.append("رقم الحساب")
            if gov == "اختر المحافظة":    errors.append("المحافظة")
            if problem == "اختر نوع المشكلة": errors.append("نوع المشكلة")
            if service == "اختر نوع الخدمة": errors.append("نوع الخدمة")
            if not since.strip():          errors.append("وقت بدء المشكلة")
            if not address.strip():        errors.append("العنوان")

            if errors:
                st.error(f"⚠️ من فضلك اكمل الحقول دي: {', '.join(errors)}")
            else:
                with st.spinner("جاري رفع التذكرة..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/ticket",
                            json={
                                "original_query": st.session_state.pending_query,
                                "name":           name.strip(),
                                "phone":          phone.strip(),
                                "account_number": account.strip(),
                                "governorate":    gov,
                                "problem_type":   problem,
                                "since_when":     since.strip(),
                                "service_type":   service,
                                "address":        address.strip(),
                            },
                            timeout=10,
                        )
                        r.raise_for_status()
                        ticket_data = r.json()
                    except Exception as e:
                        ticket_data = {
                            "ticket_id":      "TKT-LOCAL-" + __import__("datetime").datetime.now().strftime("%H%M%S"),
                            "webhook_status": "failed",
                            "message":        "تم تسجيل التذكرة محلياً.",
                        }

                # Done collecting
                st.session_state.collecting_info = False
                st.session_state.pending_query   = ""

                add_message(
                    role    = "assistant",
                    content = ticket_data.get("message", "تم رفع التذكرة."),
                    route   = "ticket",
                    ticket  = ticket_data,
                )
                st.rerun()



# Chat input 
if not st.session_state.collecting_info:
    if prompt := st.chat_input("اكتب سؤالك هنا..."):

        add_message("user", prompt)
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AVATAR_BOT):
            with st.spinner("بفكر..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={"query": prompt},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data         = resp.json()
                    answer       = data["answer"]
                    route        = data["route"]
                    sources      = data.get("sources", [])
                    latency      = data.get("latency_ms")
                    needs_action = data.get("needs_action", "NO")
                except Exception as e:
                    answer       = f"❌ حصل خطأ: {e}"
                    route        = "error"
                    sources      = []
                    latency      = None
                    needs_action = "NO"

            st.markdown(answer)
            render_sources(sources)
            render_latency(latency)

        add_message("assistant", answer, route=route, sources=sources, latency=latency)

        # Trigger info collection if ticket needed
        if needs_action == "YES":
            st.session_state.collecting_info = True
            st.session_state.pending_query   = prompt
            st.rerun()



# Sidebar 
with st.sidebar:
    st.markdown("### 📊 إحصائيات الجلسة")
    msgs = st.session_state.messages
    c1, c2 = st.columns(2)
    c1.metric("أسئلة",  len([m for m in msgs if m["role"] == "user"]))
    c2.metric("تذاكر",  len([m for m in msgs if m.get("route") == "ticket"]))
    c3, c4 = st.columns(2)
    c3.metric("chat",         len([m for m in msgs if m.get("route") == "chat"]))
    c4.metric("out of scope", len([m for m in msgs if m.get("route") == "out_of_scope"]))

    st.divider()
    st.markdown("### 💡 أمثلة")
    for ex in ["ايه باقات الإنترنت؟", "ايه الـ SLA؟", "انقطع النت عندي", "ابعتلي مهندس"]:
        if st.button(ex, use_container_width=True, disabled=st.session_state.collecting_info):
            st.session_state["_inject"] = ex
            st.rerun()

    st.divider()
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.collecting_info = False
        st.session_state.pending_query   = ""
        init_state()
        st.rerun()




# Sidebar example injection 
if "_inject" in st.session_state and not st.session_state.collecting_info:
    q = st.session_state.pop("_inject")
    add_message("user", q)
    try:
        r    = requests.post(f"{API_URL}/query", json={"query": q}, timeout=30)
        data = r.json()
        add_message("assistant", data["answer"], route=data["route"],
                    sources=data.get("sources", []), latency=data.get("latency_ms"))
        if data.get("needs_action") == "YES":
            st.session_state.collecting_info = True
            st.session_state.pending_query   = q
    except Exception:
        add_message("assistant", "❌ حصل خطأ في الاتصال بالـ API")
    st.rerun()
