import sys
sys.path.insert(0, "src")

import streamlit as st
from synapse.graph.build_graph import build_skill_graph, add_semantic_edges
from synapse.matching.entity_linker import EntityLinker
from synapse.matching.matcher import Matcher
from synapse.ingest.resume import read_resume
from synapse.ingest.skill_extractor import SkillExtractor

st.set_page_config(page_title="Synapse", page_icon="🧠", layout="wide")


@st.cache_resource
def load_engine():
    """Build graph + engine ONCE and reuse across reruns."""
    G = build_skill_graph()
    G = add_semantic_edges(G)
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    return EntityLinker(skills), Matcher(G), SkillExtractor(skills), len(skills)


linker, matcher, extractor, n_skills = load_engine()

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
    jd = linker.extract([s.strip() for s in jd_text.splitlines() if s.strip()])

    if not files:
        st.warning("Upload at least one resume to rank.")
        st.stop()

    # Read each uploaded resume and extract its skills.
    candidates, extracted = {}, {}
    for f in files:
        text = read_resume(f.getvalue(), f.name)
        skills = extractor.extract(text)
        candidates[f.name] = skills
        extracted[f.name] = skills

    ranked = matcher.rank(jd, candidates)

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
