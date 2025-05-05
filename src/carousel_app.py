import os
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Cluster Keyword Trends", layout="wide")
st.title("Cluster and Keyword Trends Over Time")

plot_dir = "./data/plots"
image_files = sorted([
    os.path.join(plot_dir, f)
    for f in os.listdir(plot_dir)
    if f.endswith('.png')
])

if not image_files:
    st.warning("No plots found. Run the script to generate the plots.")
else:
    selected = st.slider("Select Plot", 0, len(image_files) - 1, 0)
    image = Image.open(image_files[selected])
    st.image(image, caption=os.path.basename(
        image_files[selected]), use_column_width=True)
