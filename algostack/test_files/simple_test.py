from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simple Test", layout="wide")

st.title("🔧 WSL Connection Test")

# Simple interaction test
if st.button("Click Me"):
    st.success("✅ Button works! WebSocket connection is active.")

# Simple slider test
value = st.slider("Test Slider", 0, 100, 50)
st.write(f"Slider value: {value}")

# Simple selectbox
option = st.selectbox("Test Dropdown", ["Option 1", "Option 2", "Option 3"])
st.write(f"Selected: {option}")

# Display current time
st.write(f"Current time: {datetime.now()}")

# Simple chart
data = pd.DataFrame({"x": range(20), "y": np.random.randn(20).cumsum()})
st.line_chart(data.set_index("x"))

st.info(
    "If you can interact with the controls above, the connection is working properly."
)
