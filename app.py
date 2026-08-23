import sys
sys.path.insert(0, "src")

import streamlit as st
from synapse.matching.matcher import Matcher
from synapse.matching.entity_linker import EntityLinker
from synapse.ingest.resume import read_resume
from synapse.ingest.extractor import SkillExtractor as LLMExtractor
from synapse.ingest.skill_extractor import SkillExtractor as GazetteerExtractor
from synapse.graph.build_graph import add_semantic_edges, build_skill_graph

st.set_page_config(page_title="Synapse", page_icon="🧠", layout="wide")

GRAPH_PATH = "data/skill_graph.pkl"


@st.cache_resource
def load_engine():
    """Load the enriched graph; set up matcher + extractors + entity linker."""
    # Load base graph and add semantic edges (skill-similarity links)
    G = add_semantic_edges(build_skill_graph())
    skills = [n for n, d in G.nodes(data=True) if d["node_type"] == "skill"]
    node_texts = {n: f"{n} ({G.nodes[n].get('category', '')})" for n in skills}

    # Entity linker: canonicalizes extracted skills to graph nodes
    linker = EntityLinker(
        skills,
        node_texts=node_texts,
        min_score=0.60,  # threshold for embedding fallback
        use_embeddings=True,
    )

    # Gazetteer extractor: fast string-match fallback (no API key needed)
    gazetteer = GazetteerExtractor(skills)

    # LLM extractor: Gemini Flash with structured output (requires GEMINI_API_KEY)
    llm_extractor = None
    try:
        llm_extractor = LLMExtractor()
    except ValueError:
        pass  # No API key; will use gazetteer

    return Matcher(G), linker, llm_extractor, gazetteer, len(skills)


matcher, linker, llm_extractor, gazetteer, n_skills = load_engine()

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
    # Determine which extractor to use (with runtime fallback)
    use_llm = llm_extractor is not None
    extractor_name = "Gemini Flash (LLM)" if use_llm else "Gazetteer (string match)"
    st.info(f"Using {extractor_name} for skill extraction")

    # Use a mutable container for use_llm so nested function can modify it
    use_llm_flag = [use_llm]

    def extract_and_link(text: str, source_id: str):
        """Extract skills from text and canonicalize via entity linker."""
        extraction = None
        if use_llm_flag[0] and llm_extractor is not None:
            try:
                extraction = llm_extractor.extract_from_text(text)
            except Exception as e:
                st.warning(f"LLM extraction failed ({type(e).__name__}), falling back to gazetteer...")
                use_llm_flag[0] = False  # fallback for subsequent calls

        if not use_llm_flag[0] or extraction is None:
            # Gazetteer returns list of node names directly
            skills = gazetteer.extract(text)
            # Wrap in ExtractionResult-like object
            from synapse.ingest.schemas import ExtractedSkill
            extraction = [ExtractedSkill(skill=s, weight=1.0, context="gazetteer") for s in skills]

        profile = linker.link_extraction(
            type("Extraction", (), {"skills": extraction, "source_id": source_id})()
        )
        return profile.skills, extraction

    # Step 1: Extract skills from JD, then canonicalize via entity linker
    with st.spinner("Extracting skills from job description..."):
        jd_skills, jd_raw = extract_and_link(jd_text, "jd")

    if not files:
        st.warning("Upload at least one resume to rank.")
        st.stop()

    candidates, extracted = {}, {}
    for f in files:
        text = read_resume(f.getvalue(), f.name)
        # Step 2: Extract skills from resume
        with st.spinner(f"Extracting skills from {f.name}..."):
            cand_skills, cand_raw = extract_and_link(text, f.name)

        candidates[f.name] = cand_skills
        extracted[f.name] = cand_skills

    ranked = matcher.rank(jd_skills, candidates)

    st.caption("Job skills recognized: " + ", ".join(sorted(jd_skills.keys())))
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
