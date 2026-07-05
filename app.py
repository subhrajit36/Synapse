import sys
import pickle
sys.path.insert(0, "src")

import streamlit as st
from synapse.matching.matcher import Matcher
from synapse.ingest.resume import read_resume
from synapse.ingest.skill_extractor import SkillExtractor

st.set_page_config(page_title="Synapse", page_icon="🧠", layout="wide")

GRAPH_PATH = "data/skill_graph.pkl"


@st.cache_resource
def load_engine():
    """Load the prebuilt graph; set up matcher + extractor. No PyTorch at runtime."""
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    return Matcher(G), SkillExtractor(skills), len(skills)


matcher, extractor, n_skills = load_engine()

st.title("🧠 Synapse — Relationship-Aware Talent Matching")
st.caption(f"Knowledge graph of {n_skills} software skills · ranks candidates by "
           "bridgeable-gap reasoning, not keyword overlap.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 1. Required skills for the role")
    jd_text = st.text_area(
        "Enter the skills this job requires — one per line:",
        "Kubernetes\nDocker\nAWS\nTerraform\nJenkins\nPython\nLinux\nPrometheus",
        height=240)
with col2:
    st.markdown("#### 2. Upload candidate resumes")
    files = st.file_uploader(
        "Upload one or more resumes (PDF, DOCX, or TXT):",
        type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Rank candidates", type="primary"):
    jd = extractor.extract(jd_text)          # gazetteer over the JD text

    if not files:
        st.warning("Upload at least one resume to rank.")
        st.stop()

    candidates, extracted = {}, {}
    for f in files:
        text = read_resume(f.getvalue(), f.name)
        skills = extractor.extract(text)
        candidates[f.name] = skills
        extracted[f.name] = skills

    ranked = matcher.rank(jd, candidates)

    st.caption("Job skills recognized: " + ", ".join(jd))
    st.subheader("Ranking")
    for i, r in enumerate(ranked, 1):
        with st.container(border=True):
            head = st.columns([3, 1])
            head[0].markdown(f"### {i}. {r['name']}")
            head[1].metric("Fit", f"{r['score']:.2f}")
            st.progress(min(r["score"], 1.0))
            st.caption(f"Extracted {len(extracted[r['name']])} skills from this resume")
            st.markdown("✅ **Matches JD:** " + (", ".join(r["matched"]) or "—"))
            bridge = [g for g in r["gaps"] if g["bridgeable"]]
            if bridge:
                st.markdown("🟡 **Bridgeable:** " + ", ".join(
                    f"{g['skill']} (← {g['via']}, {g['distance']:.2f})" for g in bridge))
            real = [g["skill"] for g in r["gaps"] if not g["bridgeable"]]
            if real:
                st.markdown("❌ **Real gaps:** " + ", ".join(real))
