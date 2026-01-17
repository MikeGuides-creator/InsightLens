# insight_lens.py - FULL PROFESSIONAL VERSION
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import requests
import json
import time
from datetime import datetime

# ======================
# CONFIGURATION
# ======================
st.set_page_config(
    page_title="Insight Lens - AI Teaching Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0EA5E9;
        margin: 1rem 0;
    }
    .teacher-tip {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# SESSION STATE
# ======================
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# ======================
# API FUNCTIONS
# ======================
def test_api_key(api_key):
    """Test if API key is valid"""
    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": ["test"]
            },
            timeout=10
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

def get_embeddings_batch(api_key, texts):
    """Get embeddings using direct HTTP request"""
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": texts
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        total_tokens = result["usage"]["total_tokens"]
        return np.array(embeddings), total_tokens
    else:
        raise Exception(f"API error {response.status_code}: {response.text}")

# ======================
# ANALYSIS FUNCTIONS
# ======================
def analyze_responses(responses_text, api_key, sensitivity=5):
    """Main analysis pipeline"""
    # Clean and prepare responses
    responses = [r.strip() for r in responses_text.split('\n') if r.strip()]
    
    if len(responses) < 3:
        return None, "Please provide at least 3 student responses"
    
    # Limit for reasonable processing
    if len(responses) > 100:
        responses = responses[:100]
        st.info(f"Using first 100 of {len(responses)} responses for performance")
    
    try:
        with st.spinner(f"🔍 Analyzing {len(responses)} responses..."):
            # Get embeddings
            embeddings, total_tokens = get_embeddings_batch(api_key, responses)
            
            # Calculate cost
            cost = (total_tokens / 1000) * 0.00002
            st.caption(f"📊 Analysis cost: ${cost:.6f}")
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings_norm = embeddings / norms
            
            # Calculate distances and cluster
            distance_matrix = pairwise_distances(embeddings_norm, metric='cosine')
            
            # Adjust clustering based on sensitivity (1-10 slider)
            eps_value = 0.15 + (sensitivity / 20)  # Range: 0.2 to 0.65
            
            clustering = DBSCAN(
                eps=eps_value,
                min_samples=2,
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Create results DataFrame
            df = pd.DataFrame({
                'Response': responses,
                'Cluster': cluster_labels,
                'X': embeddings_norm[:, 0],
                'Y': embeddings_norm[:, 1]
            })
            
            # Identify actual clusters (not noise)
            actual_clusters = [c for c in set(cluster_labels) if c != -1]
            
            # Generate cluster summaries
            cluster_summaries = {}
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:
                    cluster_summaries[cluster_id] = "Unique/Individual Responses"
                else:
                    cluster_responses = df[df['Cluster'] == cluster_id]['Response'].tolist()
                    # Simple summary based on common words
                    all_text = ' '.join(cluster_responses).lower()
                    words = all_text.split()
                    common_words = [w for w in set(words) if len(w) > 3 and words.count(w) > len(cluster_responses)/2]
                    if common_words:
                        summary = f"Focus on: {', '.join(common_words[:3])}"
                    else:
                        summary = f"Pattern {cluster_id + 1}"
                    cluster_summaries[cluster_id] = summary
            
            results = {
                'dataframe': df,
                'cluster_summaries': cluster_summaries,
                'num_clusters': len(actual_clusters),
                'num_responses': len(responses),
                'num_unique': sum(cluster_labels == -1),
                'total_tokens': total_tokens,
                'estimated_cost': cost,
                'sensitivity': sensitivity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return results, None
            
    except Exception as e:
        return None, f"Analysis error: {str(e)}"

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("### 🔧 Setup")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenAI API Key:",
        type="password",
        value=st.session_state.api_key,
        help="Get from platform.openai.com"
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
    
    # Test button
    if st.button("Test API Key", use_container_width=True):
        if st.session_state.api_key:
            with st.spinner("Testing..."):
                is_valid, message = test_api_key(st.session_state.api_key)
                if is_valid:
                    st.success("✅ Valid API key!")
                else:
                    st.error(f"❌ Invalid: {message[:100]}")
        else:
            st.warning("Enter API key first")
    
    st.divider()
    
    # Settings
    sensitivity = st.slider(
        "Cluster Sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher = more specific groups"
    )
    
    st.divider()
    
    # Info
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **AI Understanding**: Converts text to numerical vectors
        2. **Pattern Detection**: Groups similar responses  
        3. **Visualization**: Shows conceptual relationships
        4. **Insights**: Provides teaching recommendations
        """)
    
    st.caption(f"Version 2.0 | {datetime.now().strftime('%Y-%m-%d')}")

# ======================
# MAIN CONTENT
# ======================
st.markdown("<h1 class='main-header'>🔍 Insight Lens</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>AI-Powered Misconception Detection for Teachers</h2>", unsafe_allow_html=True)

# Example datasets
example_datasets = {
    "Science - Seasons": """The seasons change because Earth gets closer to the Sun.
When it's summer, we're closer to the Sun.
The tilt of Earth's axis causes seasons.
Earth's elliptical orbit changes seasons.
I think it's about the distance from Sun.
The Earth wobbles on its axis.
Seasons happen because of the moon.
Climate change affects seasons.
Different hemispheres have opposite seasons.
I don't understand why seasons change.""",
    
    "Math - Fractions": """A fraction is part of a whole.
It's like pizza slices.
Numerator on top, denominator bottom.
Half is bigger than quarter.
You can add fractions with same denominator.
Fractions are decimals too.
I get confused with different denominators.
Bigger denominator means smaller pieces.
Fractions are ratios.
I don't like fractions.""",
    
    "History - Revolution": """The American Revolution was about taxes.
Colonists wanted independence from Britain.
It was about representation in government.
The Boston Tea Party was a protest.
They fought for freedom and rights.
It was caused by unfair laws.
The Declaration of Independence was important.
French helped Americans win.
I'm not sure what caused it.
It was about democracy."""
}

# ======================
# INPUT SECTION
# ======================
tab_input, tab_examples = st.tabs(["📝 Input", "📚 Examples"])

with tab_input:
    st.subheader("Student Responses")
    
    # Initialize with empty or example if selected
    initial_value = ""
    if st.session_state.selected_example:
        initial_value = st.session_state.selected_example
        st.session_state.selected_example = None
    
    user_responses = st.text_area(
        "Paste student responses (one per line):",
        value=initial_value,
        height=250,
        placeholder="Student 1: I think...\nStudent 2: In my opinion...\nStudent 3: The reason is...",
        key="user_responses_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analysis_name = st.text_input("Analysis Name (optional):", placeholder="e.g., 'Grade 5 Science - Seasons'")
    with col2:
        st.metric("Responses", len([r for r in user_responses.split('\n') if r.strip()]))

with tab_examples:
    st.subheader("Example Datasets")
    
    selected_example_name = st.selectbox(
        "Choose an example:",
        list(example_datasets.keys())
    )
    
    if selected_example_name:
        example_text = example_datasets[selected_example_name]
        st.text_area("Example responses:", example_text, height=250, key=f"example_{selected_example_name}")
        
        if st.button("📥 Use This Example", key=f"use_{selected_example_name}"):
            st.session_state.selected_example = example_text
            st.rerun()

# ======================
# ANALYSIS SECTION
# ======================
st.divider()

if not st.session_state.api_key:
    st.markdown("""
    <div class="info-box">
    <h4>👈 Get Started</h4>
    <p>1. Enter your OpenAI API key in the sidebar</p>
    <p>2. Click "Test Key" to verify</p>
    <p>3. Paste student responses or use an example</p>
    <p>4. Click "Analyze Responses" below</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Need an API key?** Visit [platform.openai.com](https://platform.openai.com)")
else:
    if st.button("🚀 Analyze Responses", type="primary", use_container_width=True):
        if not user_responses or len([r for r in user_responses.split('\n') if r.strip()]) < 3:
            st.warning("⚠️ Please enter at least 3 student responses")
        else:
            with st.spinner("🔍 AI analysis in progress..."):
                results, error = analyze_responses(user_responses, st.session_state.api_key, sensitivity)
                
                if error:
                    st.error(f"Analysis failed: {error}")
                elif results:
                    # Store in history
                    history_entry = {
                        'name': analysis_name or f"Analysis {len(st.session_state.analysis_history) + 1}",
                        'timestamp': results['timestamp'],
                        'num_responses': results['num_responses'],
                        'num_clusters': results['num_clusters'],
                        'responses': user_responses
                    }
                    st.session_state.analysis_history.append(history_entry)
                    
                    # Store current results
                    st.session_state.current_results = results
                    st.rerun()

# ======================
# RESULTS DISPLAY
# ======================
if st.session_state.current_results:
    results = st.session_state.current_results
    df = results['dataframe']
    
    st.divider()
    st.markdown(f"<div class='success-box'><h3>✅ Analysis Complete!</h3></div>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Responses", results['num_responses'])
    
    with col2:
        st.metric("Pattern Groups", results['num_clusters'])
    
    with col3:
        st.metric("Unique Responses", results['num_unique'])
    
    with col4:
        st.metric("Estimated Cost", f"${results['estimated_cost']:.6f}")
    
    # Tabs for different views
    tab_viz, tab_groups, tab_insights = st.tabs(["📊 Visualization", "🧩 Pattern Groups", "💡 Teaching Insights"])
    
    with tab_viz:
        st.subheader("Conceptual Landscape")
        
        df['Cluster_Label'] = df['Cluster'].apply(
            lambda x: results['cluster_summaries'].get(x, f"Group {x}")
        )
        
        fig = px.scatter(
            df,
            x='X',
            y='Y',
            color='Cluster_Label',
            hover_data=['Response'],
            title="Student Responses Grouped by Similarity",
            labels={'Cluster_Label': 'Pattern'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 *Responses closer together are more conceptually similar*")
    
    with tab_groups:
        st.subheader("Pattern Analysis")
        
        cluster_sizes = df['Cluster'].value_counts()
        
        for cluster_id in cluster_sizes.index:
            cluster_data = df[df['Cluster'] == cluster_id]
            size = len(cluster_data)
            summary = results['cluster_summaries'].get(cluster_id, "No summary")
            
            if cluster_id == -1:
                icon = "🔹"
                title = f"{icon} Unique/Individual Responses"
            else:
                icon = "📚"
                title = f"{icon} Pattern {cluster_id + 1}: {summary}"
            
            with st.expander(f"{title} - {size} student{'s' if size != 1 else ''}"):
                for idx, row in cluster_data.iterrows():
                    st.markdown(f"• **{row['Response']}**")
    
    with tab_insights:
        st.subheader("Teaching Recommendations")
        
        st.markdown("<div class='teacher-tip'><h4>🎯 Actionable Insights</h4></div>", unsafe_allow_html=True)
        
        if results['num_clusters'] == 0:
            st.info("""
            **Observation:** Highly diverse responses with no clear patterns.
            
            **Recommendations:**
            1. **Question Clarity:** Consider if the question was too open-ended
            2. **Prior Knowledge:** Students may have very different background knowledge
            3. **Scaffolding:** Provide more examples or guiding questions
            """)
        
        elif results['num_clusters'] == 1:
            st.info("""
            **Observation:** Strong consensus among students.
            
            **Recommendations:**
            1. **Validate Understanding:** Check if shared understanding is correct
            2. **If Correct:** Celebrate success and move to next topic
            3. **If Misconception:** Whole-class reteach needed
            """)
        
        else:
            st.info(f"""
            **Observation:** {results['num_clusters']} distinct understanding levels.
            
            **Differentiated Instruction Plan:**
            1. **Targeted Instruction:** Focus on largest misconception group
            2. **Peer Teaching:** Pair students from different understanding levels
            3. **Small Group Intervention:** Work with smallest groups individually
            """)
    
    # Export
    st.divider()
    st.subheader("📥 Export Results")
    
    csv_data = df[['Response', 'Cluster']].copy()
    csv_data['Pattern'] = csv_data['Cluster'].apply(
        lambda x: results['cluster_summaries'].get(x, 'Unique')
    )
    
    st.download_button(
        label="📊 Download as CSV",
        data=csv_data.to_csv(index=False),
        file_name=f"insight_lens_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Insight Lens v2.0 | AI Teaching Assistant | For educational use")
