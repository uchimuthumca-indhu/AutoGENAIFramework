import os
import autogen
import streamlit as st
from dotenv import load_dotenv

# --- Load Environment Variables from .env file ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Telecom Multi-Agent Support System",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Telecom Customer Support Multi-Agent System")
st.markdown(
    "Powered by **AutoGen** and **Streamlit**. This workflow automates research, content creation, app integration, product mapping, and quality review for telecom support."
)

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Automatically pull from environment variable if available
env_api_key = os.getenv("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key", value=env_api_key, type="password")
model_choice = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar or add it to your `.env` file to proceed.")
    st.stop()

# Set environment variable for AutoGen
os.environ["OPENAI_API_KEY"] = api_key

# --- Define LLM Configuration ---
llm_config = {
    "config_list": [{"model": model_choice, "api_key": api_key}],
    "temperature": 0.4,
}

# --- Initialize Agents ---
@st.cache_resource
def initialize_agents():
    # 1. Research Agent: Finds trending topics in telecom customer support
    research_agent = autogen.AssistantAgent(
        name="Research_Agent",
        system_message="You are a telecom industry research specialist. Your task is to research current trending customer support issues in telecommunications (e.g., 5G rollout bottlenecks, fiber-optic latency, billing disputes, eSIM activation errors).",
        llm_config=llm_config,
    )

    # 2. Content Writer Agent: Writes content based on the researched topic
    content_writer = autogen.AssistantAgent(
        name="Content_Writer",
        system_message="You are a professional technical writer. Based on the research provided by the Research_Agent, write clear, empathetic customer support knowledge base articles, troubleshooting guides, or chatbot response scripts.",
        llm_config=llm_config,
    )

    # 3. Integration Agent: Connects and checks the content with customer support apps/ticketing tools
    integration_agent = autogen.AssistantAgent(
        name="Integration_Agent",
        system_message="You are a CRM and Helpdesk Integration Specialist. Your job is to check and connect the content to customer support applications (like Zendesk, Salesforce Service Cloud, or ServiceNow) via simulated API webhooks and ticketing workflows.",
        llm_config=llm_config,
    )

    # 4. Product Agent: Provides customer support products and service upgrades
    product_agent = autogen.AssistantAgent(
        name="Product_Agent",
        system_message="You are a Telecom Product Manager. Provide relevant telecom products, service upgrades, broadband packages, or add-on support items that address the customer support scenario.",
        llm_config=llm_config,
    )

    # 5. Reviewer Agent: Reviews the content, integration, and products
    reviewer_agent = autogen.AssistantAgent(
        name="Reviewer_Agent",
        system_message="You are the Quality Assurance and Review Lead. Review all generated content, integration steps, and product recommendations for accuracy, brand tone, compliance, and clarity. Provide final approval or request revisions.",
        llm_config=llm_config,
    )

    # User Proxy / Coordinator
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=6,
        code_execution_config=False,
    )

    return [research_agent, content_writer, integration_agent, product_agent, reviewer_agent, user_proxy]

agents = initialize_agents()
research_agent, content_writer, integration_agent, product_agent, reviewer_agent, user_proxy = agents

# --- Group Chat Setup ---
groupchat = autogen.GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    speaker_selection_method="auto"
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Main UI Workflow ---
user_query = st.text_input(
    "Enter the workflow objective or starting prompt:",
    value="Develop a support strategy and troubleshooting workflow for the latest trending issue regarding 5G home internet latency drops."
)

if st.button("Run Multi-Agent Workflow"):
    with st.spinner("Agents are collaborating... Please wait."):
        chat_container = st.container()
        
        with chat_container:
            # Initiate chat
            user_proxy.initiate_chat(
                manager,
                message=user_query
            )
            
        st.success("Workflow completed successfully!")
        
        # Display conversation history
        st.subheader("💬 Agent Conversation Summary")
        for msg in groupchat.messages:
            sender = msg.get("name", "Unknown")
            content = msg.get("content", "")
            
            if sender != "User_Proxy":
                with st.expander(f"🤖 **{sender}**"):
                    st.write(content)
            else:
                with st.expander(f"👤 **{sender} (Initiator)**"):
                    st.write(content)