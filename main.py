"""
Main entry point for the Multi-Document Analysis System.
Run with: streamlit run main.py
"""

import os
import sys
from dotenv import load_dotenv
import streamlit as st

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.agent import MultiDocumentAgent
from config import get_api_key


def main():
    """Main function to run the Streamlit chatbot."""
    # Load environment variables
    load_dotenv()

    # Page configuration
    st.set_page_config(
        page_title="Multi-Document Analysis System",
        page_icon="📄"
    )

    st.title("Multi-Document Analysis System")

    # Verify API key is set
    try:
        get_api_key()
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}")
        st.stop()

    # Initialize agent (cached to avoid reprocessing on every interaction)
    @st.cache_resource
    def initialize_agent():
        """Initialize the agent with caching."""
        try:
            return MultiDocumentAgent()
        except FileNotFoundError as e:
            st.error(f"Document loading error: {str(e)}")
            st.stop()
        except ValueError as e:
            st.error(f"Configuration error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

    agent = initialize_agent()

    # Initialize chat history - store all messages sequentially
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from history
    for msg in st.session_state.messages:
        if msg["type"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["type"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        elif msg["type"] == "tool_call":
            with st.status(f"🔧 Tool: {msg['name']}", state="complete"):
                st.code(msg["args"], language="json")
        elif msg["type"] == "tool_response":
            with st.expander(f"📥 Tool Response: {msg['name']}"):
                if len(msg["content"]) > 500:
                    st.text_area("", msg["content"], height=150, disabled=True, key=msg.get("key"))
                else:
                    st.code(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Display and store user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"type": "user", "content": prompt})

        # Process with agent
        try:
            result = agent.query(prompt, stream=False)
        except ValueError as e:
            error_msg = f"Query error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"type": "assistant", "content": error_msg})
            return
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"type": "assistant", "content": error_msg})
            return

        try:

            # Parse all messages in order
            for msg in result["messages"]:
                # AI message with tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = str(tool_call.get("args", {}))

                        with st.status(f"🔧 Tool: {tool_name}", state="complete"):
                            st.code(tool_args, language="json")

                        st.session_state.messages.append({
                            "type": "tool_call",
                            "name": tool_name,
                            "args": tool_args
                        })

                # Tool response
                elif hasattr(msg, 'name') and msg.name and hasattr(msg, 'content') and msg.content:
                    tool_name = msg.name
                    content = msg.content

                    with st.expander(f"📥 Tool Response: {tool_name}"):
                        if len(content) > 500:
                            st.text_area("", content, height=150, disabled=True, key=f"new_{id(msg)}")
                        else:
                            st.code(content)

                    st.session_state.messages.append({
                        "type": "tool_response",
                        "name": tool_name,
                        "content": content,
                        "key": f"hist_{len(st.session_state.messages)}"
                    })

            # Final AI response
            final_response = result["messages"][-1].content
            with st.chat_message("assistant"):
                st.markdown(final_response)
            st.session_state.messages.append({"type": "assistant", "content": final_response})

        except (KeyError, IndexError) as e:
            error_msg = f"Response parsing error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"type": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"type": "assistant", "content": error_msg})

    # Sidebar
    with st.sidebar:
        st.header("Documents")
        st.text("• Accenture.pdf")
        st.text("• Siemens.pdf")
        st.text("• Infineon.pdf")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
