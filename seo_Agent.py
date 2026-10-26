import os
import streamlit as st
import autogen
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Multi-Agent SEO Content Generator",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Multi-Agent SEO Content Generator (AutoGen)")
st.markdown(
    "This system uses an **AutoGen multi-agent pipeline** to research trending angles, "
    "write engaging articles, optimize SEO elements, and perform a final editorial review."
)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", type="password", value=default_api_key)
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
    
    st.markdown("---")
    st.markdown("### 🤖 Agent Roles")
    st.markdown("1. **Research Agent**: Finds trending subtopics & keywords.")
    st.markdown("2. **Writer Agent**: Drafts the comprehensive article.")
    st.markdown("3. **SEO Agent**: Optimizes meta tags, headings & keywords.")
    st.markdown("4. **Reviewer Agent**: Conducts final quality & tone check.")

# ==========================================
# MAIN INTERFACE
# ==========================================
topic_input = st.text_input("Enter a broad topic or niche:", placeholder="e.g., Artificial Intelligence in Healthcare")
generate_btn = st.button("Generate SEO Content", type="primary")

if generate_btn:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar or set it in your `.env` file.")
    elif not topic_input:
        st.warning("Please enter a topic to begin.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        
        llm_config = {
            "config_list": [{"model": model_name, "api_key": api_key}],
            "temperature": 0.7,
        }

        with st.spinner("Initializing Multi-Agent Pipeline... Please wait."):
            try:
                # 1. User Proxy (Initiator)
                user_proxy = autogen.UserProxyAgent(
                    name="Admin",
                    system_message="A human admin managing the SEO workflow.",
                    code_execution_config=False,
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=1
                )

                # 2. Research Agent
                research_agent = autogen.AssistantAgent(
                    name="Research_Agent",
                    llm_config=llm_config,
                    system_message=(
                        "You are an expert market researcher and trend analyst. "
                        "Given a broad topic, identify current trending subtopics, target audience pain points, "
                        "and high-value primary/secondary keywords that are currently gaining traction."
                    )
                )

                # 3. Writer Agent
                writer_agent = autogen.AssistantAgent(
                    name="Writer_Agent",
                    llm_config=llm_config,
                    system_message=(
                        "You are a professional content writer and copywriter. "
                        "Using the research and trending angles provided by the Research Agent, "
                        "write a comprehensive, engaging, and well-structured blog post/article."
                    )
                )

                # 4. SEO Agent
                seo_agent = autogen.AssistantAgent(
                    name="SEO_Agent",
                    llm_config=llm_config,
                    system_message=(
                        "You are an SEO specialist. Review the drafted content and optimize it by ensuring "
                        "proper keyword placement, adding suggested meta titles, meta descriptions, alt text recommendations, "
                        "and checking heading structure (H1, H2, H3)."
                    )
                )

                # 5. Reviewer Agent
                reviewer_agent = autogen.AssistantAgent(
                    name="Reviewer_Agent",
                    llm_config=llm_config,
                    system_message=(
                        "You are a strict Managing Editor. Review the final SEO-optimized content for overall quality, "
                        "factual coherence, tone consistency, and readability. Provide the final polished output."
                    )
                )

                # Define Group Chat & Manager
                groupchat = autogen.GroupChat(
                    agents=[user_proxy, research_agent, writer_agent, seo_agent, reviewer_agent],
                    messages=[],
                    max_round=6,
                    speaker_selection_method="round_robin"
                )
                
                manager = autogen.GroupChatManager(
                    groupchat=groupchat,
                    llm_config=llm_config
                )

                initial_message = (
                    f"Please execute the SEO content pipeline for the following topic: '{topic_input}'. "
                    "Step 1: Research Agent analyzes trends and keywords. "
                    "Step 2: Writer Agent drafts the article. "
                    "Step 3: SEO Agent optimizes it. "
                    "Step 4: Reviewer Agent reviews and finalizes the output."
                )

                chat_result = user_proxy.initiate_chat(
                    manager,
                    message=initial_message
                )
                
                st.success("Workflow completed successfully!")
                
                if chat_result and chat_result.chat_history:
                    st.markdown("---")
                    st.markdown("## 📄 Final Output & Agent Conversation")
                    
                    for msg in chat_result.chat_history:
                        sender = msg.get("name", "Unknown")
                        content = msg.get("content", "")
                        if sender != "Admin":
                            with st.expander(f"💬 Message from **{sender}**"):
                                st.markdown(content)
                    
                    st.markdown("### ✨ Final Polished Article & SEO Report")
                    st.markdown(chat_result.chat_history[-1]["content"])

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")