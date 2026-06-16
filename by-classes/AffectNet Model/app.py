import streamlit as st

from architecture import EngagementMonitoringSystem, SystemConfig


st.set_page_config(page_title="Engagement Level Estimation - AffectNet", layout="wide")
st.title("Engagement Level Estimation - AffectNet")

if "system" not in st.session_state:
    st.session_state.system = EngagementMonitoringSystem(SystemConfig(cameraSource=1))
if "running" not in st.session_state:
    st.session_state.running = False

user_id = st.text_input("User ID", value="User")

col1, col2, col3 = st.columns(3)
if col1.button("Start"):
    st.session_state.system.startSession(user_id)
    st.session_state.running = True

if col2.button("Process frame"):
    record = st.session_state.system.processFrame()
    if record is None:
        st.warning("No face detected.")
    else:
        st.json(record.toJSON())

if col3.button("Stop"):
    st.session_state.system.endSession()
    st.session_state.running = False

st.subheader("Session")
st.json(st.session_state.system.sessionManager.getSessionInfo().__dict__)

st.subheader("Records")
st.write(f"Total records: {len(st.session_state.system.records)}")
