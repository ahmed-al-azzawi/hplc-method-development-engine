import base64
import streamlit as st

# Custom SVG Logo String
SVG_LOGO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="64" height="64">
  <defs>
    <linearGradient id="hplcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2563EB" />
      <stop offset="100%" stop-color="#0D9488" />
    </linearGradient>
    <linearGradient id="columnGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38BDF8" />
      <stop offset="50%" stop-color="#818CF8" />
      <stop offset="100%" stop-color="#C084FC" />
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="44" fill="none" stroke="url(#hplcGrad)" stroke-width="4" stroke-dasharray="210 60" stroke-linecap="round" />
  <rect x="36" y="24" width="28" height="52" rx="4" fill="#0F172A" stroke="url(#hplcGrad)" stroke-width="2.5" />
  <rect x="32" y="18" width="36" height="6" rx="2" fill="#94A3B8" />
  <rect x="32" y="76" width="36" height="6" rx="2" fill="#94A3B8" />
  <line x1="50" y1="10" x2="50" y2="18" stroke="#64748B" stroke-width="3" stroke-linecap="round" />
  <line x1="50" y1="82" x2="50" y2="90" stroke="#64748B" stroke-width="3" stroke-linecap="round" />
  <path d="M 40 32 Q 50 28, 60 32" fill="none" stroke="url(#columnGrad)" stroke-width="3" stroke-linecap="round" />
  <path d="M 40 46 Q 50 40, 60 46" fill="none" stroke="url(#columnGrad)" stroke-width="3" stroke-linecap="round" />
  <path d="M 40 60 Q 50 52, 60 60" fill="none" stroke="url(#columnGrad)" stroke-width="3" stroke-linecap="round" />
  <circle cx="43" cy="38" r="1.5" fill="#38BDF8" />
  <circle cx="57" cy="39" r="1.5" fill="#38BDF8" />
  <circle cx="47" cy="53" r="1.5" fill="#818CF8" />
  <circle cx="53" cy="67" r="1.5" fill="#C084FC" />
</svg>
"""

# Convert SVG to Data URI for browser tab favicon
svg_base64 = base64.b64encode(SVG_LOGO.encode("utf-8")).decode("utf-8")
favicon_url = f"data:image/svg+xml;base64,{svg_base64}"

st.set_page_config(
    page_title="HPLC Optimization Suite",
    page_icon=favicon_url,
    layout="centered"
)

# Header Section
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.markdown(SVG_LOGO, unsafe_allow_html=True)

st.title("HPLC Optimization Suite")
st.caption("Select an optimization engine to launch")
st.markdown("---")

# Navigation Options
col1, col2 = st.columns(2)

with col1:
    if st.button("⚡ Isocratic Engine", use_container_width=True, type="primary"):
        st.switch_page("pages/Engine.py")

with col2:
    if st.button("📈 Gradient Engine", use_container_width=True, type="primary"):
        st.switch_page("pages/Gradient Engine.py")
