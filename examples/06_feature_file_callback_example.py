import pandas as pd
import streamlit as st
import streamlit_langgraph as slg


def callback_test(file_path: str) -> str:
    if not file_path.endswith('.csv'):
        return file_path
    
    df = pd.read_csv(file_path)
    num_cols = [col for col in df.columns if col.startswith('num_')]

    if num_cols:
        df_filtered = df[num_cols]
    else:
        df_filtered = pd.DataFrame({'info': ['No columns starting with "num_" found']})
    
    processed_path = file_path.replace('.csv', '_filtered.csv')
    df_filtered.to_csv(processed_path, index=False)
    
    return processed_path


def main():
    assistant = slg.Agent(
        name="Assistant",
        role="A helpful assistant that can analyze CSV files",
        instructions=(
            "You are a helpful assistant that can answer questions and have conversations. "
            "If you do not know the answer, just state that you do not know."
        ),
        model="gpt-4o",
        provider="openai",
        allow_code_interpreter=True,
    )
    config = slg.UIConfig(
        title="File Preprocessing Example",
        file_callback=callback_test,
    )
    
    if "chat" not in st.session_state:
        st.session_state.chat = slg.LangGraphChat(
            agents=[assistant],
            config=config
        )
    st.session_state.chat.run()


if __name__ == "__main__":
    main()

