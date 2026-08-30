"""Phase C4: the FastMCP server. FR6 - ranking and gap analysis as MCP tools.

Every tool here is a one-line delegation to `engine.py`. That is the C4.4
contract: the transport layer must reuse Phase A's tested scoring verbatim, so
there is nothing in this file to unit-test beyond "the tools are registered with
the right schemas" - which is exactly how it should be.

Run it:

    python -m synapse.mcp.server                    # HTTP on :8000/mcp
    python -m synapse.mcp.server --transport sse    # SSE, same tools
    python -m synapse.mcp.server --transport stdio  # local MCP clients
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from .engine import (
    CandidateInput,
    ExplainResponse,
    GapResponse,
    GraphStats,
    RankingResponse,
    SkillWeight,
    get_engine,
)

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Synapse ranks candidates against a job description using a skill knowledge
graph rather than keyword or embedding overlap alone.

The distinction it adds is the *bridgeable gap*: a skill the candidate lacks but
which sits within a short weighted path of one they hold, and which they can
therefore be expected to pick up quickly. A gap with no such path is a real gap.
Every score is returned decomposed into direct match, bridge credit and gap
penalty, with the graph path that justified each bridge.

Skill names are canonicalized onto the graph before scoring, so 'K8s', 'k8s'
and 'Kubernetes' are the same node. Surfaces that reach no node are reported in
`*_unresolved` rather than silently dropped - check that field before trusting
a low score.
"""

mcp = FastMCP(name="synapse", instructions=INSTRUCTIONS, version="0.2.0")


# ----------------------------------------------------------------- FR5 / FR4


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"ranking"},
)
def rank_candidates(
    jd_skills: Annotated[
        list[SkillWeight],
        Field(description="Skills the role requires; weight = how much it demands each."),
    ],
    candidates: Annotated[
        list[CandidateInput],
        Field(description="The candidate pool; weight = that candidate's proficiency."),
    ],
    top_k: Annotated[
        int | None, Field(None, ge=1, description="Return only the best K. Null = all.")
    ] = None,
    max_hops: Annotated[
        int | None,
        Field(None, ge=1, le=4, description="Bridging radius in hops. Null = tuned default (2)."),
    ] = None,
    use_weights: Annotated[
        bool | None,
        Field(None, description="False scores every skill at 1.0 (the B4.1 uniform arm)."),
    ] = None,
    enable_bridging: Annotated[
        bool | None,
        Field(None, description="False falls back to direct-match-only (the B4.2 arm)."),
    ] = None,
    link: Annotated[
        bool,
        Field(True, description="Canonicalize input names first. False = names are already graph nodes."),
    ] = True,
) -> RankingResponse:
    """Rank candidates against a job description, best fit first.

    Each result carries its score components (direct match, bridge credit, gap
    penalty), the skills that matched, the gaps that were bridgeable and via
    which held skill, and the gaps that were not.
    """
    return get_engine().rank_candidates(
        jd_skills=jd_skills, candidates=candidates, top_k=top_k, max_hops=max_hops,
        use_weights=use_weights, enable_bridging=enable_bridging, link=link,
    )


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"gaps"},
)
def get_bridgeable_gaps(
    candidate_skills: Annotated[list[SkillWeight], Field(description="Skills the candidate holds.")],
    jd_skills: Annotated[list[SkillWeight], Field(description="Skills the role requires.")],
    max_hops: Annotated[
        int | None,
        Field(None, ge=1, le=4, description="Bridging radius in hops. Null = tuned default (2)."),
    ] = None,
    link: Annotated[bool, Field(True, description="Canonicalize input names first.")] = True,
) -> GapResponse:
    """Split a candidate's missing skills into bridgeable ones and real gaps.

    A skill is bridgeable when a short weighted path connects it to something the
    candidate already holds - that path (`via`, `distance`, `hops`) is returned
    so the classification can be checked rather than trusted. Anything not
    bridgeable carries a `reason` naming the condition that blocked it.
    """
    return get_engine().get_bridgeable_gaps(
        candidate_skills=candidate_skills, jd_skills=jd_skills,
        max_hops=max_hops, link=link,
    )


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    tags={"ranking", "explainability"},
)
def explain_score(
    jd_skills: Annotated[list[SkillWeight], Field(description="Skills the role requires.")],
    candidate_skills: Annotated[list[SkillWeight], Field(description="Skills the candidate holds.")],
    candidate_name: Annotated[str, Field("candidate", description="Label for the output.")] = "candidate",
    max_hops: Annotated[int | None, Field(None, ge=1, le=4)] = None,
    link: Annotated[bool, Field(True, description="Canonicalize input names first.")] = True,
) -> ExplainResponse:
    """Derive one candidate's score end to end.

    Returns the same components as `rank_candidates` plus a readable breakdown
    and the full linking trace - which surface became which node, by which
    method (alias, surface, embedding) and at what similarity. Use this when a
    score looks wrong: most surprises are linking, not scoring.
    """
    return get_engine().explain_score(
        jd_skills=jd_skills, candidate_skills=candidate_skills,
        candidate_name=candidate_name, max_hops=max_hops, link=link,
    )


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True}, tags={"diagnostics"})
def graph_stats() -> GraphStats:
    """Report what graph is loaded and how it is configured.

    Loads the graph if it is not loaded yet, so it doubles as a warm-up call
    against a cold Render instance.
    """
    return get_engine().stats()


# ------------------------------------------------------------- health (C6.3)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe that does NOT touch the graph.

    Render's free tier sleeps and cold-starts; a probe that unpickled the graph
    would report "unhealthy" for the several seconds a healthy instance spends
    waking up. Graph readiness is a separate question - ask `graph_stats`.
    """
    return JSONResponse({"status": "ok", "service": "synapse", "version": "0.2.0"})


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synapse FastMCP server (Phase C4).")
    parser.add_argument(
        "--transport", default=os.getenv("SYNAPSE_MCP_TRANSPORT", "http"),
        choices=["http", "sse", "stdio"],
        help="http (streamable HTTP, default) | sse | stdio for local clients.",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    # Render injects PORT; binding anything else makes the service unreachable.
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--path", default="/mcp", help="Mount path for http/sse.")
    parser.add_argument(
        "--warm", action="store_true",
        help="Load the graph at startup instead of on the first request.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.warm:
        stats = get_engine().stats()
        logger.info(
            "Graph ready: %d skills, %d similar edges", stats.skill_nodes, stats.similar_edges
        )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport=args.transport, host=args.host, port=args.port, path=args.path
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
