import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
from matching_system import (
    parse_uploaded_file, df_jobs, run_comprehensive_matching, 
    SKILL_GROUPS, precision_at_k, build_synthetic_ground_truth,
    CEFR_REVERSE
)

st.set_page_config(page_title='DE Matcher Pro', layout='wide', initial_sidebar_state='expanded')

# --- Header ---
st.title('🚀 Data Engineering Matcher Pro')
st.markdown("""
This advanced matcher extracts **structured signals** (Experience, Cloud Stack, Tools) 
to provide a precise ranking based on Data Engineering best practices.
""")

# --- Caching ---
@st.cache_data
def cached_matching(cv_dict, jobs_df, weights, allow_comp):
    # Use a hash of the CV text to avoid re-running for the same CV
    return run_comprehensive_matching(cv_dict, jobs_df, weights, allow_project_compensation=allow_comp)


def _make_run_key(cv_dict, weights, allow_comp):
    cv_text = (cv_dict.get('text_clean') or cv_dict.get('text') or '').encode('utf-8', errors='ignore')
    cv_hash = hashlib.sha256(cv_text).hexdigest()
    weights_json = json.dumps(weights, sort_keys=True)
    return hashlib.sha256(f"{cv_hash}|{weights_json}|{bool(allow_comp)}".encode('utf-8')).hexdigest()

# --- Sidebar: CV Upload & Summary ---
st.sidebar.header('📄 CV Analysis')
uploaded_file = st.file_uploader('Upload CV (PDF/DOCX/TXT)', type=['pdf', 'docx', 'txt'])

if uploaded_file:
    cv = parse_uploaded_file(uploaded_file)
    
    with st.sidebar:
        st.success(f"Parsed: {cv['name']}")
        st.metric("Experience", cv.get('total_experience_str', f"{cv['total_years']}y"), cv['seniority'])
        
        with st.expander("Extracted Skills (Work vs Projects)"):
            for skill, details in cv['skills_modeled'].items():
                source_icons = "".join(["💼" if s == 'work' else "🛠️" if s == 'projects' else "❔" for s in details['sources']])
                st.write(f"{skill} {source_icons}")
        
        if cv.get('certifications'):
            with st.expander("🏅 Certifications"):
                for cert in cv['certifications']:
                    st.caption(f"- {cert}")
                    
        if cv['expected_salary']:
            st.info(f"Expected Salary: {cv['expected_salary']}")
        
        if cv['languages']:
            st.write("**Languages**")
            for l in cv['languages']:
                level_str = CEFR_REVERSE.get(l['level'], 'A1').upper()
                st.caption(f"{l['language']}: {level_str}  (Badge: `{level_str}`)")

        if cv.get('soft_skills'):
            with st.expander("🤝 Soft Skills"):
                st.write(", ".join(cv['soft_skills']))

    # --- Main UI Tabs ---
    tab_res, tab_analysis, tab_settings = st.tabs(["🎯 Top Matches", "📊 Analysis", "⚙️ Weight Settings"])

    # --- Settings Tab (Define Weights) ---
    with tab_settings:
        st.subheader("Fine-tune Matching Factor Weights")
        col1, col2, col3 = st.columns(3)
        with col1:
            w_skills = st.slider("Skills Proficiency", 0.0, 1.0, 0.30)
            w_exp = st.slider("Experience Duration/Seniority", 0.0, 1.0, 0.15)
            w_qual = st.slider("Education & Certs", 0.0, 1.0, 0.10)
        with col2:
            w_sem = st.slider("Semantics (Deep Learning)", 0.0, 1.0, 0.15)
            w_tfidf = st.slider("Keywords (TF-IDF)", 0.0, 1.0, 0.10)
            w_lda = st.slider("Topics (LDA)", 0.0, 1.0, 0.10)
        with col3:
            w_lang = st.slider("Languages", 0.0, 1.0, 0.05)
            w_const = st.slider("Salary/Constraints", 0.0, 1.0, 0.05)
            w_soft = st.slider("Soft skills", 0.0, 1.0, 0.08)
        
        st.divider()
        allow_comp = st.toggle("Allow Projects to compensate for missing Work Experience", value=True)
        
        weights = {
            'skills': w_skills,
            'experience': w_exp,
            'qualifications': w_qual,
            'responsibilities': w_sem,
            'tfidf': w_tfidf,
            'lda': w_lda,
            'languages': w_lang,
            'constraints': w_const,
            'soft_skills': w_soft
        }

    # --- Results Tab ---
    with tab_res:
        top_k = st.slider('Number of results', 1, 20, 5)

        # Run the matching only when CV/weights/toggle changes (k-only changes reuse stored results)
        run_key = _make_run_key(cv, weights, allow_comp)
        if st.session_state.get('last_run_key') != run_key or 'last_results_df' not in st.session_state:
            st.session_state['last_run_key'] = run_key
            st.session_state['last_results_df'] = cached_matching(cv, df_jobs, weights, allow_comp)

        results_df = st.session_state['last_results_df']
        
        st.subheader(f"Top {top_k} Matches for {cv['name']}")

        view_df = results_df.head(int(top_k))
        for i in range(len(view_df)):
            row = view_df.iloc[i]
            exp = row['explanation']
            
            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"### {i+1}. {row['job_title']}")
                    st.write(f"**Seniority**: {row['seniority']} | **Required**: {row['required_years']}y")
                with c2:
                    st.metric("Match Score", f"{row['score']*100:.1f}%")

                with st.expander("Job details"):
                    meta_cols = st.columns(2)
                    with meta_cols[0]:
                        if row.get('company'):
                            st.write(f"**Company**: {row.get('company')}")
                        if row.get('job_location'):
                            st.write(f"**Location**: {row.get('job_location')}")
                        if row.get('job_type'):
                            st.write(f"**Type**: {row.get('job_type')}")
                        if row.get('job_level'):
                            st.write(f"**Level**: {row.get('job_level')}")
                        if row.get('first_seen'):
                            st.caption(f"First seen: {row.get('first_seen')}")
                        if row.get('job_link'):
                            st.markdown(f"[Open job link]({row.get('job_link')})")
                    with meta_cols[1]:
                        if row.get('job_skills'):
                            st.write("**Skills**")
                            st.write(row.get('job_skills'))

                        must_skills = row.get('must_have_skills')
                        nice_skills = row.get('nice_to_have_skills')
                        if isinstance(must_skills, (list, tuple)) and must_skills:
                            st.write("**Parsed Must-have Skills**")
                            st.write(", ".join(must_skills))
                        if isinstance(nice_skills, (list, tuple)) and nice_skills:
                            st.write("**Parsed Nice-to-have Skills**")
                            st.write(", ".join(nice_skills))

                        if row.get('salary_range') and row.get('salary_range') != 'N/A':
                            st.write(f"**Salary range**: {row.get('salary_range')}")

                    if row.get('job_summary'):
                        st.write("**Summary / Description**")
                        st.write(row.get('job_summary'))

                    if row.get('job_text'):
                        st.write("**Canonical job_text (used by matchers)**")
                        st.write(row.get('job_text'))
                
                # New Structured Explanation
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Why Matched** ✅")
                    for reason in exp['why_matched']:
                        st.write(f"- {reason}")
                    if exp['project_highlights']:
                        for highlight in exp['project_highlights']:
                            st.info(f"💡 {highlight}")
                
                with cols[1]:
                    st.write("**Gaps / Risks** ⚠️")
                    if exp['gaps']:
                        for gap in exp['gaps']:
                            st.write(f"- {gap}")
                    else:
                        st.write("No major gaps detected.")
                    st.caption(f"Language: {exp['language_status']}")
                    if 'title_relevance' in exp:
                        st.caption(f"Role Match: {exp['title_relevance']}")
                
                st.divider()

    # --- Analysis Tab ---
    with tab_analysis:
        st.subheader("Evaluation & Model Insights")
        
        # precision@k logic
        gt = build_synthetic_ground_truth([cv], df_jobs)
        if gt.get(cv['name']):
            st.write("#### Synthetic Precision@k")
            metrics = []
            # We only have ensemble_results in a different format now, 
            # let's adapt or use a simplified version for this CV
            scores_array = np.zeros(len(df_jobs))
            for _, r in results_df.iterrows():
                scores_array[r['job_index']] = r['score']
            
            for k in [1, 3, 5]:
                # Mock results dict for precision_at_k function
                mock_res = {cv['name']: scores_array}
                mean_p, _ = precision_at_k(mock_res, gt, k=k)
                metrics.append({'k': k, 'precision': mean_p})
            
            m_cols = st.columns(3)
            for j, m in enumerate(metrics):
                m_cols[j].metric(f"Precision@{m['k']}", f"{m['precision']*100:.1f}%")
        else:
            st.warning("Could not build a synthetic ground truth for this CV (no skill overlap).")

        st.subheader("Score Contribution (Example for Top Match)")
        if not results_df.empty:
            top_exp = results_df.iloc[0]['explanation']
            if 'breakdown' in top_exp:
                # Filter out zero contributions for cleaner chart
                data = {k: v for k, v in top_exp['breakdown'].items() if v != 0}
                st.bar_chart(pd.DataFrame({'Contribution': data.values()}, index=data.keys()))
            else:
                st.write("Detailed breakdown not available.")

else:
    st.info("Please upload a CV to start the matching process.")
    
# Footer
st.sidebar.divider()
st.sidebar.caption("v2.0 - Attribute-Aware Ranking Engine")
