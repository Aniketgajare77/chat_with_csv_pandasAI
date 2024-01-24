import os
from dotenv import load_dotenv
import streamlit as st
from streamlit import cache
from PIL import Image
import io
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import pandas as pd
from pandasai.llm.openai import OpenAI
import matplotlib
import matplotlib.pyplot 

headers={
    "authorization": st.secrets["OPENAI_API_KEY"]
}
matplotlib.use('Agg')
load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
st.set_page_config("Chat with csv by incrify")


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"], use_container_width=True)
        return
        
    @cache(allow_output_mutation=True)
    def format_plot(self, result):
        resized_image = self.resize_image(result["value"], target_size=(350, 250))
        st.image(resized_image, caption='Resized Image', use_column_width=True)
        #st.experimental_rerun()
        return

    def format_other(self, result):
        st.write(result["value"])
        return

    # Helper function to resize an image
    def resize_image(self, image_bytes, target_size):
        if isinstance(image_bytes, str):  # Check if it's a file path
            with open(image_bytes, 'rb') as file:
                image_bytes = file.read()

        image = Image.open(io.BytesIO(image_bytes))
        resized_image = image.resize(target_size, resample=Image.BICUBIC)  # You can also try Image.BICUBIC
        resized_bytes = io.BytesIO()
        resized_image.save(resized_bytes, format='PNG')  # You can adjust the format if needed
        return resized_bytes.getvalue()


st.title("Data Magic Unleashed: Chat with Your CSV")
st.markdown("<h3 style='text-align: center; color: white;'>Built by <a>Incrify </a></h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    try:
        col1.info("CSV Uploaded Successfully")
        data = pd.read_csv(uploaded_file)
        col1.dataframe(data.head(5), use_container_width=True)


        prompt = col2.text_area("Enter your prompt:")

        llm = OpenAI()

        

        if col2.button("Generate"):
            if prompt:
                with st.spinner("Generating response..."):
                    query_engine = SmartDataframe(
                        data,
                        config={
                            "llm": llm,
                            "response_parser": StreamlitResponse,
                        },
                    )

                    answer = query_engine.chat(prompt)
            else:
                st.warning("Please enter a prompt.")
    except ValueError:
        col1.error("Please ask a relevant question.")
