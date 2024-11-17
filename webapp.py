""" streamlit run app.py """
import os
import os.path as osp

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import streamlit as st

# App title
st.set_page_config(
    page_title="What2Do,ONJ?",
    page_icon="🏥",
)


# load models
@st.cache_resource
def load_model():
    pretrained_model_name_or_path = "../../checkpoint/qwen2_vl_7b_instruct_sft_lora"
    print("load model ...")
    model       = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, torch_dtype="auto", device_map="auto")
    processor   = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
    print("load model [DONE]")
    return model, processor
model, processor = load_model()


def inference(image_path : str, medical_record : str, question : str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {   
                    "type": "text", 
                    "text": f"In this image, the purple area represents the lesion site. Please answer the question based on your medical records.\nMedical Records:{medical_record}\nQuestion:{question}"
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=3072)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# Login
# with st.sidebar:
#     st.title('Login')
    # if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
    #     st.success('HuggingFace Login credentials already provided!', icon='✅')
    #     hf_email = st.secrets['EMAIL']
    #     hf_pass = st.secrets['PASS']
    # else:
    #     hf_email = st.text_input('Enter E-mail:', type='password')
    #     hf_pass = st.text_input('Enter password:', type='password')
    #     if not (hf_email and hf_pass):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')


# Input image
image_path = None
uploaded_file = st.file_uploader('Choose a case', type=['png', 'jpg'], label_visibility='visible')
if uploaded_file:
    temp_root = '.temp'
    if not osp.exists(temp_root): os.makedirs(temp_root)
    image_path = osp.join(temp_root, uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    st.image(image_path)


# Input medical record
medical_record = st.text_input("Medical record")


# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input Question
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = inference(image_path, medical_record, prompt) 
            st.write(response)
    # message = {"role": "assistant", "content": response}
    # st.session_state.messages.append(message)