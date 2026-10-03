"""Calibrate the entity linker's embedding threshold (`DEFAULT_MIN_SCORE`).

The embedding fallback only sees surfaces that miss the alias table and the
surface index, so the labelled set is written in exactly those terms:

  POSITIVES  - realistic résumé/JD phrasings of skills that ARE graph nodes
               (typos, versions, suffixes, spacing), each with its correct node.
  NEGATIVES  - real skills and concepts that are NOT graph nodes. Any link for
               one of these is wrong: it hands a candidate a skill they never
               claimed ("Feature Engineering" -> C++).

For each threshold it reports, over the surfaces that reach the embedding layer:

  correct  - positives linked to their own node
  wrong    - positives linked to a different node, plus negatives linked at all
  missed   - positives left unresolved (visible, recoverable - see NFR6)

A wrong link corrupts every downstream score invisibly; a missed one shows up in
`unresolved`. So the selection rule is: the lowest threshold with zero wrong
links on this set, and it is reported alongside how many positives it costs.

    python scripts/calibrate_link_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synapse.matching.entity_linker import METHOD_EMBEDDING  # noqa: E402
from synapse.mcp.engine import MatchEngine  # noqa: E402

# surface -> correct node
POSITIVES: dict[str, str] = {
    "kubernities": "Kubernetes",
    "kubernetes cluster": "Kubernetes",
    "Postgres SQL": "PostgreSQL",
    "PostgreSQL 15": "PostgreSQL",
    "ReactJS": "React",
    "React.js": "React",
    "VueJS": "Vue.js",
    "Node JS": "Node.js",
    "Scikit learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "Tensor Flow": "TensorFlow",
    "TensorFlow 2": "TensorFlow",
    "Py Torch": "PyTorch",
    "Pandas library": "pandas",
    "numpy arrays": "NumPy",
    "Spring boot microservices": "Spring Boot",
    "MS Excel": "Microsoft Excel",
    "Excel spreadsheets": "Microsoft Excel",
    "Power BI dashboards": "Microsoft Power BI",
    "Azure": "Microsoft Azure software",
    "AWS EC2": "Amazon Elastic Compute Cloud EC2",
    "Amazon SageMaker": "Amazon Web Services AWS SageMaker",
    "Golang": "Go",
    "Rust": "Rust programming language",
    "Java 8": "Oracle Java",
    "Jenkins pipelines": "Jenkins CI",
    "Hugging Face Transformers": "Hugging Face",
    "Kafka Streams": "Apache Kafka",
    "Airflow DAGs": "Apache Airflow",
    "Shell scripting": "Shell script",
    "bash scripting": "Bash",
    "Linux administration": "Linux",
    "MongoDB Atlas": "MongoDB",
    "Redis cache": "Redis",
    "REST APIs": "RESTful API",
    "Selenium WebDriver": "Selenium",
    "Tableau dashboards": "Tableau",
    "XGBoost classifier": "XGBoost",
    "LightGBM model": "LightGBM",
    "Llama 2": "Llama",
    "LangChain agents": "LangChain",
    "MS SQL Server": "Microsoft SQL Server",
    "T-SQL": "Transact-SQL",
    "PL SQL": "Oracle PL/SQL",
    "TypeScript 5": "TypeScript",
    "AngularJS": "Google Angular",
    "Docker Compose": "Docker",
    "dockerized services": "Docker",
    "Prometheus monitoring": "Prometheus",
    "Git version control": "Git",
    "GitLab CI": "GitLab",
    "Unity 3D": "Unity Technologies Unity",
    "Unreal Engine 5": "Unreal Technology Unreal Engine",
    "Elastic search": "Elasticsearch",
    "Snowflake data warehouse": "Snowflake",
    "C sharp": "C#",
    "CPP": "C++",
    "FastAPI backend": "FastAPI",
    "Django REST framework": "Django",
    "GraphQL APIs": "GraphQL",
    "PySpark jobs": "PySpark",
    "Apache Spark Streaming": "Apache Spark",
}

# Not graph nodes. Any link is wrong.
NEGATIVES: list[str] = [
    "Object Oriented Programming", "Feature Engineering", "CNNs", "RNNs", "LSTMs",
    "neo4j", "graph database", "RAG", "SHAP", "Machine Learning", "Deep Learning",
    "Data Structures", "Algorithms", "Computer Vision", "Natural Language Processing",
    "Statistics", "Linear Regression", "Model Evaluation", "Microservices",
    "System Design", "Agile", "Scrum", "Communication", "Leadership",
    "Problem Solving", "WebSockets", "Pydantic", "Flask", "Next.js", "OpenCV",
    "Streamlit", "SpeechBrain", "PCGrad", "GANs", "Reinforcement Learning",
    "Firebase", "Supabase", "Heroku", "Nginx", "RabbitMQ", "Celery", "gRPC",
    "OAuth", "JWT", "CI/CD", "Unit testing", "Data Visualization", "Matplotlib",
    "Seaborn", "Plotly", "SciPy", "NLTK", "spaCy", "OpenAI API",
    "Prompt Engineering", "Vector databases", "Pinecone", "FAISS", "ChromaDB",
    "Cypher", "Redux", "Express.js", "Jest", "Cypress", "Operating Systems",
    "Computer Networks", "DBMS", "Cloud Computing", "Kotlin", "Dart", "Flutter",
    "Haskell", "Julia", "Verilog", "Arduino", "Blockchain", "Web3", "LLMs",
    "Transformer models", "Fine-tuning", "MLOps", "ETL pipelines", "A/B testing",
]

THRESHOLDS = [0.60, 0.65, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90]


def main() -> int:
    linker = MatchEngine().linker
    nodes = set(linker.skills)
    bad = sorted(v for v in POSITIVES.values() if v not in nodes)
    if bad:
        print(f"Labelled targets missing from the graph: {bad}")
        return 1

    # Score every surface once at threshold 0, so each one's best node and
    # cosine score is known; a threshold then only decides accept vs. withhold.
    linker.min_score = 0.0
    pos, neg, skipped = [], [], []
    for surface, gold in POSITIVES.items():
        r = linker.link(surface)
        if r.method != METHOD_EMBEDDING:
            skipped.append((surface, r.method))
            continue
        pos.append((surface, gold, r.node, r.score))
    for surface in NEGATIVES:
        r = linker.link(surface)
        if r.method != METHOD_EMBEDDING:
            skipped.append((surface, r.method))
            continue
        neg.append((surface, r.node, r.score))

    print(f"{len(pos)} positives and {len(neg)} negatives reach the embedding layer "
          f"({len(skipped)} resolved earlier by alias/surface and excluded)\n")
    print(f"{'threshold':>9} {'correct':>8} {'missed':>7} {'wrong(pos)':>11} {'wrong(neg)':>11}")
    for t in THRESHOLDS:
        correct = sum(1 for _, g, n, s in pos if s >= t and n == g)
        wrong_pos = sum(1 for _, g, n, s in pos if s >= t and n != g)
        missed = sum(1 for _, _, _, s in pos if s < t)
        wrong_neg = sum(1 for _, _, s in neg if s >= t)
        print(f"{t:>9.2f} {correct:>8} {missed:>7} {wrong_pos:>11} {wrong_neg:>11}")

    print("\nHighest-scoring negatives (would be wrong links):")
    for surface, node, score in sorted(neg, key=lambda x: -x[2])[:12]:
        print(f"  {score:.3f}  {surface!r} -> {node}")
    print("\nPositives, lowest score first:")
    for surface, gold, node, score in sorted(pos, key=lambda x: x[3])[:15]:
        flag = "" if node == gold else f"   WRONG (got {node})"
        print(f"  {score:.3f}  {surface!r} -> {gold}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
