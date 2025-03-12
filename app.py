import streamlit as st
import openai
import pandas as pd
import random

# -------------------------------------------------------
# 1. PASTE YOUR OPENAI KEY HERE
# -------------------------------------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]


# -------------------------------------------------------
# 2. LOAD YOUR DATASET
# -------------------------------------------------------
try:
    df = pd.read_csv("empathetic_dialogues.csv")
except Exception as e:
    st.error(f"Error loading 'emphatetic_dialogues.csv': {e}")
    st.stop()

# We'll randomly sample a few examples to shape the LLM's style.
NUM_EXAMPLES = 100
sample_examples = df.sample(n=NUM_EXAMPLES, random_state=42).to_dict(orient="records")

# -------------------------------------------------------
# 3. BUILD A FEW-SHOT SYSTEM MESSAGE
# -------------------------------------------------------
few_shot_prompt = ""
for ex in sample_examples:
    example_dialogue = str(ex.get("empathetic_dialogues", "")).strip()
    example_response = str(ex.get("labels", "")).strip()
    few_shot_prompt += (
        "Example Conversation:\n"
        f"{example_dialogue}\n"
        f"Agent's response: {example_response}\n\n"
    )

system_content = (
    "You are a mental health helpline assistant. You respond with empathy and understanding. "
    "Learn from these examples, especially you will be the agent who responds to the client.\n\n"
    f"{few_shot_prompt}"
    "When the user sends a message, reply as an empathetic helpline agent. "
    "Keep responses concise, supportive, and aligned with the style shown in the examples. "
    "Ask clarifying questions when necessary."
)

# -------------------------------------------------------
# 4. SET UP THE STREAMLIT APP
# -------------------------------------------------------
st.set_page_config(page_title="Helpline Conversation Assistant", layout="wide")

# Remove old title and create a new, simpler heading
st.title("Helpline Conversation Assistant")

# -------------------------------------------------------
# 5. SIDEBAR CONTENT
# -------------------------------------------------------
st.sidebar.title("Welcome to Your Helpline")
st.sidebar.markdown(
    """
    <p style="font-size:1.1rem; line-height:1.6;">
    This helpline assistant is here to offer empathy and understanding.
    Please share what's on your mind, and the assistant will do its best
    to provide supportive responses.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# 6. STORE MESSAGES IN SESSION STATE
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_content}
    ]

# -------------------------------------------------------
# 7. CUSTOM CSS FOR A MORE POLISHED, PASTEL LOOK
# -------------------------------------------------------
st.markdown("""
<style>

/* Use a multi-stop pastel gradient for the background with a wavy top effect */
body {
    background: linear-gradient(160deg, #fdf2f9 0%, #e9f9fd 50%, #f2f9f4 100%);
    font-family: "Helvetica Neue", sans-serif;
    color: #333;
}

/* Make the main container more prominent with deeper shadow */
.main > div {
    padding: 30px !important;
    border-radius: 24px;
    background-color: rgba(255, 255, 255, 0.85);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
}

/* Center the content area a bit */
.block-container {
    margin: 0 auto;
    max-width: 900px;
}

/* Style the sidebar to complement the main background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fef6fa 0%, #eefdfd 100%);
    box-shadow: 2px 0 6px rgba(0,0,0,0.1);
}
[data-testid="stSidebar"] > div {
    padding: 1rem 1.2rem;
}

/* Chat container styling */
.chat-container {
    display: flex;
    margin-bottom: 20px;
    padding: 8px;
}

/* User bubble - pastel pink */
.user-bubble {
    background-color: #ffe3ec;
    color: #333;
    padding: 16px 20px;
    border-radius: 20px;
    margin-right: 80px;
    max-width: 70%;
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    font-size: 1.1rem;
    line-height: 1.5;
}

/* Assistant bubble - pastel blue */
.assistant-bubble {
    background-color: #e7f1ff;
    color: #333;
    padding: 16px 20px;
    border-radius: 20px;
    margin-left: 80px;
    max-width: 70%;
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    font-size: 1.1rem;
    line-height: 1.5;
}

/* Thicker input box with pastel background */
textarea, input[type=text] {
    border: 2px solid #ccc !important;
    border-radius: 16px !important;
    padding: 12px !important;
    font-size: 1.1rem !important;
    background-color: #fff;
}

/* Thicker buttons with deeper shadows, pastel hover effect */
.css-1cpxqw2 > button, .stButton button {
    border-radius: 16px;
    padding: 0.8rem 1.4rem;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    background-color: #fff;
    color: #333;
    border: 2px solid #ddd;
    transition: background-color 0.3s ease;
}
.css-1cpxqw2 > button:hover, .stButton button:hover {
    background-color: #f8f8f8;
}

/* Title styling */
h1 {
    color: #333;
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    font-size: 2rem;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# 8. DISPLAY CHAT
# -------------------------------------------------------
def display_chat(messages):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-container">'
                f'<div class="user-bubble"><strong>You:</strong> {msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f'<div class="chat-container" style="justify-content: flex-end;">'
                f'<div class="assistant-bubble"><strong>Assistant:</strong> {msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.header("Conversation")
display_chat(st.session_state.messages)

# -------------------------------------------------------
# 9. FORM FOR USER INPUT (ENTER TO SUBMIT)
# -------------------------------------------------------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:")
    submitted = st.form_submit_button("Send")
    if submitted:
        if user_input.strip():
            # Append user's message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate assistant response
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages,
                    max_tokens=200,
                    temperature=0.7,
                )
                assistant_reply = response.choices[0].message.content
            except Exception as e:
                assistant_reply = f"Error calling OpenAI API: {e}"
            
            # Append assistant's message
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            
            # Rerun to update the conversation display
            st.rerun()
        else:
            st.warning("Please enter a message before sending.")

# -------------------------------------------------------
# 10. CLEAR CONVERSATION BUTTON
# -------------------------------------------------------
if st.button("Clear Conversation"):
    st.session_state.messages = [
        {"role": "system", "content": system_content}
    ]
    st.rerun()
