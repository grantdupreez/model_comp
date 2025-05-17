import streamlit as st
import asyncio
import anthropic
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd

# Set page config
st.set_page_config(page_title="Claude Models Comparison", layout="wide")

# App title
st.title("Claude Models Comparison")

# Sidebar for API key and models selection
with st.sidebar:
    api_key = st.text_input("Anthropic API Key", type="password")
    
    available_models = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20250219"
    ]
    
    selected_models = st.multiselect(
        "Select Claude Models to Compare",
        available_models,
        default=["claude-3-opus-20240229", "claude-3-7-sonnet-20250219"]
    )
    
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

# Input area for prompt
prompt = st.text_area("Enter your prompt", height=150)

# Function to call Claude API
def call_claude(model, prompt, api_key, max_tokens, temperature):
    start_time = time.time()
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="Respond directly to the user's request.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            "model": model,
            "response": response.content[0].text,
            "execution_time": f"{execution_time:.2f}s",
            "success": True,
            "tokens_used": response.usage.output_tokens,
        }
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        return {
            "model": model,
            "response": f"Error: {str(e)}",
            "execution_time": f"{execution_time:.2f}s",
            "success": False,
            "tokens_used": 0,
        }

# Async function to run calls in parallel
async def run_parallel_calls(models, prompt, api_key, max_tokens, temperature):
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                call_claude,
                model,
                prompt,
                api_key,
                max_tokens,
                temperature
            )
            for model in models
        ]
        return await asyncio.gather(*tasks)

# Run button - FIXED: using a single button with one key
if st.button("Run Comparison", key="run_comparison"):
    if prompt and api_key and selected_models:
        with st.spinner("Processing..."):
            # Run the parallel calls
            results = asyncio.run(run_parallel_calls(
                selected_models, prompt, api_key, max_tokens, temperature
            ))
            
            # Display metrics
            metrics_cols = st.columns(len(results))
            for i, result in enumerate(results):
                with metrics_cols[i]:
                    st.metric(
                        label=f"{result['model'].split('-')[-1].title()}", 
                        value=f"{result['tokens_used']} tokens",
                        delta=result['execution_time']
                    )
            
            # Display responses in tabs
            tabs = st.tabs([model.split('-')[-1].title() for model in selected_models])
            
            for i, tab in enumerate(tabs):
                with tab:
                    st.markdown(f"### {results[i]['model']}")
                    st.markdown(results[i]['response'])
            
            # Compare side by side
            st.subheader("Side-by-Side Comparison")
            comparison_data = pd.DataFrame(results)
            
            # Display just model and response columns side by side
            comparison_view = comparison_data[["model", "response"]].copy()
            comparison_view["model"] = comparison_view["model"].apply(lambda x: x.split('-')[-1].title())
            comparison_view = comparison_view.set_index("model").T
            
            st.dataframe(comparison_view, use_container_width=True)
    else:
        st.error("Please provide API key, select models, and enter a prompt")

# Add explanatory text
st.markdown("""
### How It Works
1. Enter your Anthropic API key in the sidebar
2. Select which Claude models you want to compare
3. Enter a prompt and click "Run Comparison"
4. View the results side-by-side

This app uses concurrent processing to call multiple Claude models in parallel, reducing overall response time.
""")
