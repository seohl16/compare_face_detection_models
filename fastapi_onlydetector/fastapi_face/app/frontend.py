import io
import os
from pathlib import Path

import requests
from PIL import Image

import streamlit as st
from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


def main():
    st.title("Face Detection Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        files = [
            ('files', (uploaded_file.name, image_bytes,
                       uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/order", files=files)
        result_str = response.json()["products"][0]["result"]
        st.write(result_str) # Success
        
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(image, use_column_width=True)
        col2.header("detect")
        col2.image(Image.open('annotated/annotated_faces.png'), use_column_width=True)


@cache_on_button_press('Authenticate')
def authenticate(password) -> bool:
    return password == root_password

main()

# password = st.text_input('password', type="password")

# if authenticate(password):
#     st.success('You are authenticated!')
#     main()
# else:
#     st.error('The password is invalid.')
