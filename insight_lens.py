# insight_lens.py - The Conceptual Fault Line Detector
# UPDATED for Python 3.12 and OpenAI 1.x API
import streamlit as st
from openai import OpenAI
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import pandas as pd
import plotly.express as px
import os
import time

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Insight Lens - Teacher's Diagnostic Dashboard",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------
# 2. SIDEBAR - SETUP & API KEY
# -------------------------------
with st.sidebar:
    st.title("🔧 Setup")
    st.markdown("""
    **How to use:**
    1. Paste student responses (one per line)
    2. Click 'Analyze Misunderstandings'
    3. Explore the clusters
    
    **Cost:** ~$0.0001 per 100 responses
    """)
    
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
    
    # Optional: Let users set key via environment variable too
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            st.info("Using API key from environment variable")
    
    st.divider()
    st.markdown("**Clustering Settings**")
    sensitivity = st.slider("Cluster Sensitivity", 1, 10, 5, 
                           help="Higher = more specific clusters, Lower = broader groups")
    
    st.divider()
    st.caption("Privacy: All processing happens in real-time. No data is stored.")

# -------------------------------
# 3. MAIN INTERFACE - TITLE
# -------------------------------
st.title("🔍 Insight Lens: Conceptual Fault Line Detector")
st.markdown("""
*Upload student responses to **automatically identify patterns and misconceptions**.*  
*This tool doesn't grade—it helps you **diagnose** so you can **teach better**.*
""")

# -------------------------------
# 4. TEXT INPUT AREA
# -------------------------------
st.subheader("📥 Student Responses")

example_responses = """The seasons change because the Earth gets closer and farther from the Sun.
When it's summer, we're closer to the Sun, and when it's winter, we're farther away.
I think it has to do with the tilt of the Earth, but I'm not sure how.
The Earth spins faster in summer so it gets hotter.
It's because of the Earth's axis being tilted toward or away from the Sun.
Seasons change due to distance from the Sun throughout the year.
The tilt causes different parts of Earth to get more direct sunlight at different times.
I don't really know, maybe it's about the weather cycles?
Summer happens when our hemisphere is tilted toward the Sun.
Winter is when we're tilted away from the Sun.
The Sun is hotter in summer months.
It's all about the Earth's orbit being elliptical.
Seasons are caused by the moon's position.
Climate change affects the seasons now.
The Earth wobbles on its axis.
In winter, the Sun goes behind the moon more often.
The tilt changes throughout the year.
It's because of how the atmosphere traps heat.
Different countries have different seasons at the same time."""

input_mode = st.radio("Input method:", ["Paste text", "Use example (Seasons)"], horizontal=True)

if input_mode == "Paste text":
    student_text = st.text_area(
        "Paste student responses (one per line):",
        height=200,
        placeholder="Paste each student's response on a new line...",
        key="student_text"
    )
else:
    student_text = example_responses
    st.text_area("Example responses (seasons question):", student_text, height=200, disabled=True)

# -------------------------------
# 5. PROCESSING FUNCTION
# -------------------------------
def analyze_responses(responses, api_key, sensitivity=5):
    """Main analysis pipeline: embeddings → clustering → insights"""
    
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None
    
    # Clean and validate input
    responses = [r.strip() for r in responses.split('\n') if r.strip()]
    if len(responses) < 5:
        st.warning("Please enter at least 5 responses for meaningful analysis.")
        return None
    
    if len(responses) > 200:
        st.warning("For this prototype, using first 200 responses to manage costs.")
        responses = responses[:200]
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    try:
        # -------------------------------
        # STEP A: Get embeddings from OpenAI
        # -------------------------------
        status_text.text("Step 1/3: Analyzing response meanings...")
        progress_bar.progress(20)
        
        # Process in batches to handle rate limits
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(responses), batch_size):
            batch = responses[i:i + batch_size]
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min(40, 20 + (i / len(responses)) * 20)
                progress_bar.progress(int(progress))
                
            except Exception as e:
                st.error(f"Error processing batch: {e}")
                return None
        
        embeddings = np.array(all_embeddings)
        
        # Show cost (extremely cheap)
        estimated_tokens = sum(len(r.split()) for r in responses)
        cost = (estimated_tokens / 1000) * 0.00002
        st.caption(f"📊 Estimated cost: ${cost:.6f}")
        
        # -------------------------------
        # STEP B: Cluster with DBSCAN
        # -------------------------------
        status_text.text("Step 2/3: Grouping similar responses...")
        progress_bar.progress(60)
        
        # Normalize embeddings for better clustering
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_norm = embeddings / norms
        
        # Calculate pairwise distances
        distance_matrix = pairwise_distances(embeddings_norm, metric='cosine')
        
        # Adjust epsilon based on sensitivity
        eps_value = 0.2 + (sensitivity / 20)  # Range: 0.25 to 0.7
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=eps_value, 
            min_samples=2, 
            metric='precomputed',
            n_jobs=-1  # Use all CPU cores
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Count clusters (ignore noise labeled as -1)
        unique_labels = set(cluster_labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        
        # If no clusters found, try a different approach
        if n_clusters == 0:
            # Try KMeans as fallback
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(5, len(responses)), random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_norm)
            unique_labels = set(cluster_labels)
            n_clusters = len(unique_labels)
        
        # -------------------------------
        # STEP C: Generate cluster summaries
        # -------------------------------
        status_text.text("Step 3/3: Identifying patterns...")
        progress_bar.progress(80)
        
        # Create DataFrame for results
        df = pd.DataFrame({
            'response': responses,
            'cluster': cluster_labels,
            'embedding_x': embeddings_norm[:, 0],  # For visualization
            'embedding_y': embeddings_norm[:, 1]
        })
        
        # Generate a descriptive name for each cluster
        cluster_summaries = {}
        cluster_examples = {}
        
        # Process clusters in order of size (largest first)
        cluster_sizes = df['cluster'].value_counts()
        
        for cluster_id in cluster_sizes.index:
            cluster_responses = df[df['cluster'] == cluster_id]['response'].tolist()
            examples = cluster_responses[:min(3, len(cluster_responses))]
            
            if cluster_id == -1:
                cluster_name = "Uncategorized / Unique Responses"
            else:
                # Use AI to summarize the common theme
                try:
                    response_sample = "\n".join(cluster_responses[:min(5, len(cluster_responses))])
                    
                    prompt = f"""
                    Analyze these student answers. They all share a similar pattern or misunderstanding.
                    
                    STUDENT ANSWERS:
                    {response_sample}
                    
                    Identify the COMMON THEME or MISCONCEPTION. 
                    Respond with ONLY a short, clear phrase (max 6 words).
                    
                    If they're correct, say "Correct understanding of [concept]"
                    If they're partially correct, say "Partial understanding: [what they get right]"
                    If they're wrong, say "Misconception: [brief description]"
                    """
                    
                    ai_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Using 3.5 for speed/cost
                        messages=[
                            {"role": "system", "content": "You are an expert educator analyzing student responses."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=30,
                        temperature=0.2
                    )
                    cluster_name = ai_response.choices[0].message.content.strip()
                    # Clean up the response
                    cluster_name = cluster_name.replace('"', '').replace("'", "")
                    
                except Exception as e:
                    st.warning(f"Couldn't generate name for cluster {cluster_id}: {e}")
                    cluster_name = f"Pattern {cluster_id + 1} ({len(cluster_responses)} students)"
            
            cluster_summaries[cluster_id] = cluster_name
            cluster_examples[cluster_id] = examples
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return {
            'dataframe': df,
            'cluster_summaries': cluster_summaries,
            'cluster_examples': cluster_examples,
            'n_clusters': n_clusters,
            'n_responses': len(responses),
            'sensitivity': sensitivity
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# -------------------------------
# 6. ANALYSIS TRIGGER
# -------------------------------
if student_text and api_key:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_clicked = st.button("🔍 Analyze Misunderstandings", 
                                   type="primary", 
                                   use_container_width=True)
    
    with col2:
        st.caption(f"Responses: {len([r for r in student_text.split('\\n') if r.strip()])}")
    
    if analyze_clicked:
        with st.spinner("Starting analysis..."):
            results = analyze_responses(student_text, api_key, sensitivity)
            
        if results:
            df = results['dataframe']
            cluster_summaries = results['cluster_summaries']
            cluster_examples = results['cluster_examples']
            
            # -------------------------------
            # 7. RESULTS DISPLAY
            # -------------------------------
            st.success(f"✅ Found **{results['n_clusters']} conceptual patterns** in {results['n_responses']} responses")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Visualization", "🧩 Detailed Patterns", "📋 Summary"])
            
            with tab1:
                # -------------------------------
                # VISUALIZATION: Cluster Plot
                # -------------------------------
                st.subheader("Response Clusters")
                
                # Prepare hover text
                df['hover_text'] = df['response'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
                
                # Create visualization
                fig = px.scatter(
                    df,
                    x='embedding_x',
                    y='embedding_y',
                    color=df['cluster'].apply(lambda x: cluster_summaries.get(x, "Unknown")),
                    hover_data=['response'],
                    title="Student Responses Grouped by Conceptual Similarity",
                    labels={'color': 'Pattern'},
                    width=800,
                    height=500
                )
                fig.update_traces(
                    marker=dict(size=10, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                    selector=dict(mode='markers')
                )
                fig.update_layout(
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("💡 *Points closer together represent similar understandings/misunderstandings*")
            
            with tab2:
                # -------------------------------
                # CLUSTER DETAILS
                # -------------------------------
                st.subheader("Detected Patterns & Examples")
                
                # Sort clusters by size (largest first)
                cluster_sizes = {cid: len(df[df['cluster'] == cid]) for cid in cluster_summaries.keys()}
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                
                for cluster_id, size in sorted_clusters:
                    summary = cluster_summaries[cluster_id]
                    examples = cluster_examples.get(cluster_id, [])
                    
                    with st.expander(f"**{summary}** — {size} student{'s' if size != 1 else ''}", expanded=size > 5):
                        # Color code based on content
                        if "misconception" in summary.lower() or "incorrect" in summary.lower():
                            st.markdown("**Status:** ❌ Common misconception")
                        elif "partial" in summary.lower():
                            st.markdown("**Status:** ⚠️ Partial understanding")
                        elif "correct" in summary.lower():
                            st.markdown("**Status:** ✅ Correct understanding")
                        
                        st.markdown("**Example responses:**")
                        for i, example in enumerate(examples[:4], 1):
                            st.markdown(f"{i}. *\"{example}\"*")
                        
                        if len(examples) > 4:
                            st.caption(f"... and {len(examples) - 4} more similar responses")
                        
                        # Teacher insight suggestions
                        st.divider()
                        st.markdown("**💡 Teacher Insight:**")
                        
                        # Generate targeted insight
                        insight_prompt = f"""
                        As an expert teacher, seeing that {size} students have this pattern:
                        Pattern: {summary}
                        Examples: {examples[:2]}
                        
                        Suggest ONE specific, actionable teaching strategy to address this.
                        Be brief (1-2 sentences).
                        """
                        
                        try:
                            client = OpenAI(api_key=api_key)
                            insight_response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an experienced teacher providing concise, actionable advice."},
                                    {"role": "user", "content": insight_prompt}
                                ],
                                max_tokens=80,
                                temperature=0.3
                            )
                            insight = insight_response.choices[0].message.content.strip()
                            st.info(insight)
                        except:
                            # Fallback generic insights
                            if size > results['n_responses'] * 0.3:  # >30% of class
                                st.info("This is a major misconception affecting many students. Consider a whole-class reteach with a hands-on demonstration.")
                            elif size > 1:
                                st.info("Small group intervention recommended. Pair these students with peers who have correct understanding.")
                            else:
                                st.info("Individual check-in needed. This student may have a unique perspective or confusion.")
            
            with tab3:
                # -------------------------------
                # SUMMARY REPORT
                # -------------------------------
                st.subheader("Analysis Summary")
                
                # Create summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Responses", results['n_responses'])
                
                with col2:
                    st.metric("Patterns Found", results['n_clusters'])
                
                with col3:
                    largest_cluster = max([len(df[df['cluster'] == cid]) for cid in cluster_summaries.keys() if cid != -1], default=0)
                    percentage = (largest_cluster / results['n_responses']) * 100
                    st.metric("Largest Pattern", f"{largest_cluster} ({percentage:.1f}%)")
                
                # Summary table
                summary_data = []
                for cluster_id, summary in cluster_summaries.items():
                    size = len(df[df['cluster'] == cluster_id])
                    percentage = (size / results['n_responses']) * 100
                    summary_data.append({
                        "Pattern": summary,
                        "Students": size,
                        "Percentage": f"{percentage:.1f}%",
                        "Priority": "⚠️ Address" if "misconception" in summary.lower() and percentage > 10 else "📝 Note"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df.sort_values("Students", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download options
                st.divider()
                st.subheader("Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download
                    csv = df[['response', 'cluster']].copy()
                    csv['pattern'] = csv['cluster'].map(cluster_summaries)
                    csv = csv[['response', 'pattern', 'cluster']]
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv.to_csv(index=False),
                        file_name=f"insight_lens_analysis_{time.strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Summary report
                    report_text = f"""Insight Lens Analysis Report
Generated: {time.strftime('%Y-%m-%d %H:%M')}
Total Responses: {results['n_responses']}
Patterns Identified: {results['n_clusters']}

PATTERNS FOUND:
"""
                    for cluster_id, summary in cluster_summaries.items():
                        size = len(df[df['cluster'] == cluster_id])
                        percentage = (size / results['n_responses']) * 100
                        report_text += f"\n{summary}: {size} students ({percentage:.1f}%)"
                    
                    st.download_button(
                        label="📄 Download Summary",
                        data=report_text,
                        file_name=f"insight_summary_{time.strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                # Next steps suggestion
                st.divider()
                st.markdown("**🎯 Recommended Next Steps:**")
                
                # Find the largest misconception
                misconceptions = [(cid, s) for cid, s in cluster_summaries.items() 
                                if "misconception" in s.lower() and cid != -1]
                if misconceptions:
                    largest_misconception = max(misconceptions, key=lambda x: len(df[df['cluster'] == x[0]]))
                    st.info(f"**Focus area:** {largest_misconception[1]} is the most common misconception.")
                
                st.markdown("""
                1. **Plan a 5-10 minute reteach** for the largest misconception
                2. **Create heterogeneous groups** mixing different understanding levels
                3. **Follow up individually** with students in the "Uncategorized" group
                4. **Share findings** with your teaching team or specialist
                """)

# -------------------------------
# 9. FOOTER & HELP
# -------------------------------
st.divider()
with st.expander("ℹ️ About & Help"):
    st.markdown("""
    **How It Works:**
    1. **Embeddings:** Converts text to numerical vectors capturing meaning
    2. **Clustering:** Groups similar vectors together (similar understandings)
    3. **Analysis:** Identifies patterns and generates insights
    
    **Best Practices:**
    - Use open-ended questions (not yes/no)
    - Aim for 10-50 responses for best results
    - Responses should be in students' own words
    - Works best with 3rd grade through high school
    
    **Privacy & Security:**
    - No data is stored or saved
    - All processing happens in your browser/our server
    - API calls go directly to OpenAI
    - You can run this locally for complete privacy
    """)

st.caption("""
**Insight Lens v0.2** | Built for teachers | Python 3.12+ Compatible | [Report Issues]
""")
