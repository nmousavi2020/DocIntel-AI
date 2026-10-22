# ============================================================
# DocIntel AI
# Enterprise Document Intelligence Engine
# ============================================================

import json
import re
from io import BytesIO

import fitz
import pandas as pd
import requests
import streamlit as st

from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocIntel AI",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📄 DocIntel AI")
st.caption(
    "Enterprise Document Intelligence Engine"
)


# ============================================================
# CONSTANTS
# ============================================================

COLLECTION_NAME = "docintel_documents"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

VECTOR_SIZE = 1024

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

RRF_K = 60


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL_NAME
    )


@st.cache_resource
def get_qdrant():

    return QdrantClient(":memory:")


embedding_model = load_embedding_model()
reranker = load_reranker()
qdrant = get_qdrant()


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class ExperienceItem(BaseModel):

    company: str = ""

    location: str = ""

    role: str = ""

    period: str = ""

    description: str = ""


class EducationItem(BaseModel):

    institution: str = ""

    degree: str = ""

    field: str = ""

    period: str = ""

    location: str = ""

    description: str = ""


class PublicationItem(BaseModel):

    title: str = ""

    authors: list[str] = Field(
        default_factory=list
    )

    journal: str = ""

    year: str = ""

    doi: str = ""

    url: str = ""

    description: str = ""


class DocumentInformation(BaseModel):

    name: str = ""

    email: str = ""

    phone: str = ""

    location: str = ""

    github: str = ""

    linkedin: str = ""

    profile: str = ""

    programming_languages: list[str] = Field(
        default_factory=list
    )

    ml_frameworks: list[str] = Field(
        default_factory=list
    )

    ml_tools: list[str] = Field(
        default_factory=list
    )

    ml_methods: list[str] = Field(
        default_factory=list
    )

    data_science_tools: list[str] = Field(
        default_factory=list
    )

    experience: list[ExperienceItem] = Field(
        default_factory=list
    )

    education: list[EducationItem] = Field(
        default_factory=list
    )

    publications: list[PublicationItem] = Field(
        default_factory=list
    )

    languages: list[str] = Field(
        default_factory=list
    )


# ============================================================
# SECTION DEFINITIONS
# ============================================================

SECTION_PATTERNS = {

    "profile": [
        r"\bprofile\b",
    ],

    "machine_learning": [
        r"\bmachine learning\b",
    ],

    "data_science": [
        r"\bdata\s*&\s*scientific computing\b",
        r"\bdata science\b",
    ],

    "programming": [
        r"\bprogramming\b",
    ],

    "ml_tools": [
        r"\bml engineering\s*&\s*mlops\b",
        r"\bmlops\b",
    ],

    "languages": [
        r"\blanguages\b",
    ],

    "projects": [
        r"\bselected project\b",
        r"\bselected projects\b",
    ],

    "experience": [
        r"\bprofessional experience\b",
    ],

    "education": [
        r"\beducation\b",
    ],

    "publications": [
        r"\bselected publication\b",
        r"\bselected publications\b",
        r"\bpublications\b",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text(
            "text",
            sort=True,
        )

        text = normalize_text(text)

        pages.append(
            {
                "page": page_number + 1,
                "text": text,
            }
        )

    document.close()

    return pages


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_sections(text):

    text = normalize_text(text)

    if not text:
        return []


    # --------------------------------------------------------
    # Find all known section headings
    # --------------------------------------------------------

    matches = []

    for section_name, patterns in SECTION_PATTERNS.items():

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                matches.append(
                    {
                        "section": section_name,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )


    # --------------------------------------------------------
    # Remove overlapping / duplicate matches
    # --------------------------------------------------------

    matches.sort(
        key=lambda x: x["start"]
    )

    filtered = []

    occupied = []

    for match in matches:

        overlap = False

        for start, end in occupied:

            if (
                match["start"] < end
                and match["end"] > start
            ):

                overlap = True
                break

        if not overlap:

            filtered.append(match)

            occupied.append(
                (
                    match["start"],
                    match["end"],
                )
            )


    # --------------------------------------------------------
    # Build sections
    # --------------------------------------------------------

    sections = []

    for index, match in enumerate(filtered):

        start = match["end"]

        if index + 1 < len(filtered):

            end = filtered[index + 1]["start"]

        else:

            end = len(text)

        section_text = text[
            start:end
        ].strip()

        if section_text:

            sections.append(
                {
                    "section": match["section"],
                    "text": section_text,
                }
            )


    # --------------------------------------------------------
    # Header before first detected section
    # --------------------------------------------------------

    if filtered:

        header_text = text[
            :filtered[0]["start"]
        ].strip()

        if header_text:

            sections.insert(
                0,
                {
                    "section": "header",
                    "text": header_text,
                },
            )

    else:

        sections.append(
            {
                "section": "header",
                "text": text,
            }
        )


    return sections


# ============================================================
# SPECIAL CV SECTION PARSER
# ============================================================

def split_cv_sections(text):

    text = normalize_text(text)

    sections = []


    # --------------------------------------------------------
    # Main CV headings
    # --------------------------------------------------------

    main_heading_patterns = [

        (
            "skills",
            r"\bSkills\b",
        ),

        (
            "profile",
            r"\bProfile\b",
        ),

        (
            "projects",
            r"\bSelected Project\b",
        ),

        (
            "experience",
            r"\bProfessional Experience\b",
        ),

        (
            "education",
            r"\bEducation\b",
        ),

        (
            "publications",
            r"\bSelected Publication\b",
        ),
    ]


    matches = []

    for name, pattern in main_heading_patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            matches.append(
                {
                    "name": name,
                    "start": match.start(),
                    "end": match.end(),
                }
            )


    matches.sort(
        key=lambda x: x["start"]
    )


    # Remove duplicate / nested matches
    unique = []

    seen_ranges = set()

    for match in matches:

        key = (
            match["start"],
            match["end"],
        )

        if key not in seen_ranges:

            unique.append(match)
            seen_ranges.add(key)


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    if unique:

        header = text[
            :unique[0]["start"]
        ].strip()

        if header:

            sections.append(
                {
                    "section": "header",
                    "text": header,
                }
            )


    # --------------------------------------------------------
    # Main sections
    # --------------------------------------------------------

    for index, match in enumerate(unique):

        start = match["end"]

        if index + 1 < len(unique):

            end = unique[index + 1]["start"]

        else:

            end = len(text)

        content = text[
            start:end
        ].strip()

        if not content:
            continue


        if match["name"] == "skills":

            # ----------------------------------------------
            # Split Skills into sub-sections
            # ----------------------------------------------

            skill_patterns = [

                (
                    "machine_learning",
                    r"\bMachine Learning\b",
                ),

                (
                    "data_science",
                    r"\bData\s*&\s*Scientific Computing\b",
                ),

                (
                    "programming",
                    r"\bProgramming\b",
                ),

                (
                    "ml_tools",
                    r"\bML Engineering\s*&\s*MLOps\b",
                ),

                (
                    "languages",
                    r"\bLanguages\b",
                ),
            ]

            skill_matches = []

            for name, pattern in skill_patterns:

                for sm in re.finditer(
                    pattern,
                    content,
                    flags=re.IGNORECASE,
                ):

                    skill_matches.append(
                        {
                            "name": name,
                            "start": sm.start(),
                            "end": sm.end(),
                        }
                    )


            skill_matches.sort(
                key=lambda x: x["start"]
            )


            if skill_matches:

                # text before first skill subsection
                prefix = content[
                    :skill_matches[0]["start"]
                ].strip()

                if prefix:

                    sections.append(
                        {
                            "section": "skills",
                            "text": prefix,
                        }
                    )


                for sindex, sm in enumerate(
                    skill_matches
                ):

                    start2 = sm["end"]

                    if (
                        sindex + 1
                        < len(skill_matches)
                    ):

                        end2 = skill_matches[
                            sindex + 1
                        ]["start"]

                    else:

                        end2 = len(content)

                    subsection_text = content[
                        start2:end2
                    ].strip()

                    if subsection_text:

                        sections.append(
                            {
                                "section": sm["name"],
                                "text": subsection_text,
                            }
                        )

            else:

                sections.append(
                    {
                        "section": "skills",
                        "text": content,
                    }
                )

        else:

            sections.append(
                {
                    "section": match["name"],
                    "text": content,
                }
            )


    return sections


# ============================================================
# SECTION-AWARE CHUNKING
# ============================================================

def create_chunks(
    pages,
    document_name="",
    chunk_size=1200,
    overlap=150,
):

    chunks = []

    global_chunk_id = 0


    for page in pages:

        page_text = normalize_text(
            page["text"]
        )

        if not page_text:
            continue


        sections = split_cv_sections(
            page_text
        )


        # fallback
        if not sections:

            sections = [
                {
                    "section": "general",
                    "text": page_text,
                }
            ]


        for section_data in sections:

            section_name = section_data[
                "section"
            ]

            text = section_data[
                "text"
            ].strip()

            if not text:
                continue


            # ------------------------------------------------
            # Chunk section
            # ------------------------------------------------

            start = 0

            local_chunk = 0

            while start < len(text):

                end = min(
                    start + chunk_size,
                    len(text),
                )

                chunk_text = text[
                    start:end
                ].strip()


                if chunk_text:

                    chunks.append(
                        {
                            "id": global_chunk_id,
                            "document": document_name,
                            "page": page["page"],
                            "chunk": local_chunk,
                            "section": section_name,
                            "text": chunk_text,
                        }
                    )

                    global_chunk_id += 1
                    local_chunk += 1


                if end >= len(text):
                    break


                start = max(
                    0,
                    end - overlap,
                )


    return chunks


# ============================================================
# QDRANT COLLECTION
# ============================================================

def create_collection():

    existing = [
        collection.name
        for collection in
        qdrant.get_collections().collections
    ]

    if COLLECTION_NAME in existing:

        qdrant.delete_collection(
            collection_name=COLLECTION_NAME
        )


    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


# ============================================================
# VECTOR INDEXING
# ============================================================

def index_chunks(
    chunks,
    document_name,
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


    points = []


    for index, embedding in enumerate(
        embeddings
    ):

        chunk = chunks[index]


        points.append(
            PointStruct(
                id=int(chunk["id"]),
                vector=embedding.tolist(),
                payload={
                    "id": chunk["id"],
                    "document": document_name,
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                },
            )
        )


    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


    return len(points)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question,
    top_k=10,
):

    query_embedding = embedding_model.encode(
        question,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )


    return result.points


# ============================================================
# BM25
# ============================================================

def build_bm25(chunks):

    tokenized_chunks = [
        re.findall(
            r"\b[\w+#.-]+\b",
            chunk["text"].lower(),
        )
        for chunk in chunks
    ]


    return BM25Okapi(
        tokenized_chunks
    )


def bm25_search(
    question,
    top_k=10,
):

    bm25 = st.session_state[
        "bm25"
    ]

    chunks = st.session_state[
        "bm25_chunks"
    ]


    query_tokens = re.findall(
        r"\b[\w+#.-]+\b",
        question.lower(),
    )


    scores = bm25.get_scores(
        query_tokens
    )


    ranked_indices = (
        scores.argsort()[::-1][:top_k]
    )


    results = []


    for index in ranked_indices:

        results.append(
            {
                "index": int(index),
                "score": float(
                    scores[index]
                ),
                "chunk": chunks[index],
            }
        )


    return results


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_sections(question):

    q = question.lower()

    sections = set()


    if any(
        term in q
        for term in [
            "programming language",
            "programming languages",
            "coding language",
            "coding languages",
            "what languages does",
            "what programming",
        ]
    ):

        sections.add(
            "programming"
        )


    if any(
        term in q
        for term in [
            "framework",
            "frameworks",
            "pytorch",
            "tensorflow",
        ]
    ):

        sections.add(
            "programming"
        )


    if any(
        term in q
        for term in [
            "mlops",
            "machine learning tools",
            "ml tools",
            "deployment tools",
            "engineering tools",
        ]
    ):

        sections.add(
            "ml_tools"
        )


    if any(
        term in q
        for term in [
            "machine learning method",
            "methods",
            "techniques",
            "algorithm",
            "algorithms",
            "models",
        ]
    ):

        sections.add(
            "machine_learning"
        )


    if any(
        term in q
        for term in [
            "education",
            "degree",
            "phd",
            "university",
            "study",
            "studied",
        ]
    ):

        sections.add(
            "education"
        )


    if any(
        term in q
        for term in [
            "publication",
            "publications",
            "paper",
            "papers",
            "doi",
            "journal",
        ]
    ):

        sections.add(
            "publications"
        )


    if any(
        term in q
        for term in [
            "experience",
            "worked",
            "job",
            "company",
            "position",
            "role",
        ]
    ):

        sections.add(
            "experience"
        )


    if any(
        term in q
        for term in [
            "project",
            "projects",
        ]
    ):

        sections.add(
            "projects"
        )


    if any(
        term in q
        for term in [
            "language",
            "english",
            "german",
        ]
    ):

        if not sections:

            sections.add(
                "languages"
            )


    return sections


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_overlap(
    question,
    text,
):

    q_tokens = set(
        re.findall(
            r"\b[\w+#.-]+\b",
            question.lower(),
        )
    )

    t_tokens = set(
        re.findall(
            r"\b[\w+#.-]+\b",
            text.lower(),
        )
    )


    if not q_tokens:

        return 0.0


    return len(
        q_tokens.intersection(
            t_tokens
        )
    ) / len(q_tokens)


# ============================================================
# HYBRID SEARCH - RRF + SECTION BOOST
# ============================================================

def hybrid_search(
    question,
    top_k=8,
):

    retrieval_k = max(
        10,
        top_k * 3,
    )


    semantic_results = semantic_search(
        question,
        top_k=retrieval_k,
    )


    bm25_results = bm25_search(
        question,
        top_k=retrieval_k,
    )


    target_sections = detect_query_sections(
        question
    )


    scores = {}

    chunks_by_id = {}


    # --------------------------------------------------------
    # Semantic ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        payload = result.payload

        key = payload.get(
            "id"
        )


        chunks_by_id[key] = payload

        scores.setdefault(
            key,
            0.0,
        )


        scores[key] += (
            1.0 /
            (
                RRF_K + rank
            )
        )


    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        chunk = result[
            "chunk"
        ]

        key = chunk.get(
            "id"
        )


        chunks_by_id[key] = chunk

        scores.setdefault(
            key,
            0.0,
        )


        scores[key] += (
            1.0 /
            (
                RRF_K + rank
            )
        )


    # --------------------------------------------------------
    # Additional lexical + section boost
    # --------------------------------------------------------

    for key, payload in chunks_by_id.items():

        text = payload.get(
            "text",
            "",
        )

        section = payload.get(
            "section",
            "",
        )


        lexical_score = lexical_overlap(
            question,
            text,
        )


        scores[key] += (
            lexical_score * 0.02
        )


        if (
            target_sections
            and section in target_sections
        ):

            scores[key] += 0.05


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


    results = []


    for key, score in ranked[:top_k]:

        results.append(
            {
                "rrf_score": float(
                    score
                ),
                "payload": chunks_by_id[
                    key
                ],
            }
        )


    return results


# ============================================================
# RERANKER
# ============================================================

def rerank_results(
    question,
    hybrid_results,
    top_k=5,
):

    if not hybrid_results:

        return []


    pairs = []


    for result in hybrid_results:

        text = result[
            "payload"
        ][
            "text"
        ]


        pairs.append(
            [
                question,
                text,
            ]
        )


    scores = reranker.predict(
        pairs
    )


    reranked = []


    for index, score in enumerate(
        scores
    ):

        reranked.append(
            {
                "reranker_score": float(
                    score
                ),
                "rrf_score": hybrid_results[
                    index
                ][
                    "rrf_score"
                ],
                "payload": hybrid_results[
                    index
                ][
                    "payload"
                ],
            }
        )


    reranked.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True,
    )


    return reranked[:top_k]


# ============================================================
# FORCE RELEVANT SECTION
# ============================================================

def ensure_query_section(
    question,
    results,
):

    target_sections = detect_query_sections(
        question
    )


    if not target_sections:

        return results


    available_sections = {
        r["payload"].get(
            "section"
        )
        for r in results
    }


    missing = (
        target_sections
        - available_sections
    )


    if not missing:

        return results


    all_chunks = st.session_state.get(
        "chunks",
        [],
    )


    forced = []


    for chunk in all_chunks:

        if chunk.get(
            "section"
        ) in missing:

            forced.append(
                {
                    "reranker_score": 999.0,
                    "rrf_score": 999.0,
                    "payload": chunk,
                }
            )


    # Put forced relevant sections first
    return (
        forced
        + results
    )


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(
    question,
    reranked_results,
):

    context_parts = []


    for index, result in enumerate(
        reranked_results,
        start=1,
    ):

        payload = result[
            "payload"
        ]


        context_parts.append(
            f"""
SOURCE [{index}]
Document: {payload.get("document", "")}
Page: {payload.get("page", "")}
Chunk: {payload.get("chunk", "")}
Section: {payload.get("section", "")}

Text:
{payload.get("text", "")}
"""
        )


    context = "\n".join(
        context_parts
    )


    programming_question = (
        "programming"
        in detect_query_sections(
            question
        )
    )


    if programming_question:

        special_rules = """

SPECIAL RULE FOR PROGRAMMING-LANGUAGE QUESTIONS:

The CV may contain a section called "Programming".

Return ONLY actual programming, scripting,
or query languages.

Examples of programming/query languages:
Python, SQL, MATLAB, Fortran, Bash.

Do NOT classify these as programming languages:
NumPy, pandas, scikit-learn, PyTorch,
Git, Linux/HPC, MLflow, FastAPI, Docker.

If the document explicitly lists an item under
Programming, preserve actual languages from that list,
but do not promote libraries, frameworks, operating systems,
or engineering tools into programming languages.
"""

    else:

        special_rules = ""


    prompt = f"""
You are an enterprise document intelligence assistant.

Answer the user's question using ONLY the supplied CV/document context.

Rules:

1. Never invent information.
2. Do not use outside knowledge.
3. If the information is not present, say:
   "The information is not available in the provided document."
4. Be concise.
5. Preserve exact names, dates, titles and technologies.
6. For list questions, return a clean bullet list.
7. Do not mix categories.
8. Cite relevant source numbers when useful.

{special_rules}

USER QUESTION:

{question}

DOCUMENT CONTEXT:

{context}

Answer:
"""


    return prompt


# ============================================================
# QWEN GENERATION
# ============================================================

def ask_qwen(
    question,
    reranked_results,
):

    prompt = build_rag_prompt(
        question,
        reranked_results,
    )


    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        },
        timeout=300,
    )


    response.raise_for_status()


    data = response.json()


    return data.get(
        "response",
        "",
    ).strip()


# ============================================================
# CITATIONS
# ============================================================

def generate_citations(
    reranked_results,
):

    citations = []


    for index, result in enumerate(
        reranked_results,
        start=1,
    ):

        payload = result[
            "payload"
        ]


        citation = (
            f"[{index}] "
            f"{payload.get('document', 'Unknown')} | "
            f"Page: {payload.get('page', 'Unknown')} | "
            f"Chunk: {payload.get('chunk', 'Unknown')} | "
            f"Section: {payload.get('section', 'Unknown')}"
        )


        citations.append(
            citation
        )


    return citations


# ============================================================
# IDP PROMPT
# ============================================================

def build_idp_prompt(
    document_text,
):

    return f"""
You are a professional CV information extraction system.

Extract structured information from the CV below.

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.
Do not add explanations.
Do not add fields outside the schema.

SCHEMA:

{{
  "name": "",
  "email": "",
  "phone": "",
  "location": "",
  "github": "",
  "linkedin": "",
  "profile": "",

  "programming_languages": [],

  "ml_frameworks": [],

  "ml_tools": [],

  "ml_methods": [],

  "data_science_tools": [],

  "experience": [],

  "education": [],

  "publications": [],

  "languages": []
}}

EXPERIENCE ITEM:

{{
  "company": "",
  "location": "",
  "role": "",
  "period": "",
  "description": ""
}}

EDUCATION ITEM:

{{
  "institution": "",
  "degree": "",
  "field": "",
  "period": "",
  "location": "",
  "description": ""
}}

PUBLICATION ITEM:

{{
  "title": "",
  "authors": [],
  "journal": "",
  "year": "",
  "doi": "",
  "url": "",
  "description": ""
}}

CLASSIFICATION RULES:

1. programming_languages:
   Only actual programming, scripting, or query languages.

   Examples:
   Python
   SQL
   Bash
   MATLAB
   Fortran

2. ml_frameworks:
   Frameworks or major ML libraries.

   Examples:
   PyTorch
   TensorFlow
   scikit-learn
   XGBoost

3. ml_tools:
   Engineering, MLOps, deployment, infrastructure,
   or development tools.

   Examples:
   MLflow
   FastAPI
   Docker
   Git
   Linux/HPC

4. ml_methods:
   ML methods and techniques.

   Examples:
   Gradient Boosting
   Random Forest
   Deep Learning
   CNN
   Neural Networks
   Feature Engineering
   Hyperparameter Optimization
   Ensemble Learning
   Uncertainty Quantification
   Explainable AI

5. data_science_tools:
   Data processing, scientific computing,
   visualization, geospatial, or data-analysis tools/concepts.

6. Do not place the same item into multiple categories
   unless the document clearly uses it in multiple roles.

7. Do not invent missing information.

8. Preserve exact names and dates.

9. Education must be a list of objects.

10. Publications must be a list of objects.

11. Authors must be a list of strings.

12. Extract the actual publication journal names.

13. Extract DOI URLs into the "doi" field when a DOI is available.

14. If a URL is explicitly provided separately,
    use the "url" field.

15. Languages means human languages such as:
    English (C2), German (A2).

16. For name/email/phone/location/social profiles,
    extract directly from the document.

17. For experience, preserve the actual company,
    role, location and period.

18. Empty information must use:
    ""
    or
    []

19. Return valid JSON only.

DOCUMENT:

{document_text}
"""


# ============================================================
# JSON CLEANING
# ============================================================

def extract_json_object(text):

    if not text:

        raise ValueError(
            "Empty response from Qwen."
        )


    text = text.strip()


    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    ).strip()


    # Find first JSON object
    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )


    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found in Qwen response."
        )


    return text[
        start:end + 1
    ]


# ============================================================
# IDP EXTRACTION
# ============================================================

def extract_structured_data(
    document_text,
):

    prompt = build_idp_prompt(
        document_text
    )


    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        },
        timeout=300,
    )


    response.raise_for_status()


    raw = response.json().get(
        "response",
        "",
    )


    cleaned = extract_json_object(
        raw
    )


    data = json.loads(
        cleaned
    )


    validated = DocumentInformation.model_validate(
        data
    )


    return validated


# ============================================================
# POST-PROCESS IDP
# ============================================================

def postprocess_structured_data(
    data,
    document_text,
):

    text = document_text


    # --------------------------------------------------------
    # Extract email if Qwen missed it
    # --------------------------------------------------------

    if not data.email:

        match = re.search(
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
            text,
        )

        if match:

            data.email = match.group(
                0
            )


    # --------------------------------------------------------
    # GitHub
    # --------------------------------------------------------

    if not data.github:

        match = re.search(
            r"https?://(?:www\.)?github\.com/[^\s)\]]+",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            data.github = match.group(
                0
            )


    # --------------------------------------------------------
    # LinkedIn
    # --------------------------------------------------------

    if not data.linkedin:

        match = re.search(
            r"https?://(?:www\.)?linkedin\.com/in/[^\s)\]]+",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            data.linkedin = match.group(
                0
            )


    # --------------------------------------------------------
    # DOI normalization
    # --------------------------------------------------------

    for publication in data.publications:

        if publication.doi:

            doi_match = re.search(
                r"https?://doi\.org/[^\s)\]]+",
                publication.doi,
                flags=re.IGNORECASE,
            )

            if doi_match:

                publication.doi = doi_match.group(
                    0
                )


    # --------------------------------------------------------
    # Remove obvious false programming languages
    # --------------------------------------------------------

    forbidden_programming = {
        "numpy",
        "pandas",
        "scikit-learn",
        "sklearn",
        "pytorch",
        "tensorflow",
        "git",
        "linux",
        "linux/hpc",
        "mlflow",
        "fastapi",
        "docker",
    }


    data.programming_languages = [
        item
        for item in data.programming_languages
        if item.strip().lower()
        not in forbidden_programming
    ]


    # --------------------------------------------------------
    # Normalize common Bash formatting
    # --------------------------------------------------------

    normalized = []


    for item in data.programming_languages:

        if item.lower().startswith(
            "bash"
        ):

            normalized.append(
                "Bash"
            )

        else:

            normalized.append(
                item
            )


    data.programming_languages = list(
        dict.fromkeys(
            normalized
        )
    )


    return data


# ============================================================
# JSON DOWNLOAD
# ============================================================

def create_json_download(
    data,
):

    json_text = json.dumps(
        data.model_dump(),
        indent=2,
        ensure_ascii=False,
    )


    return json_text.encode(
        "utf-8"
    )


# ============================================================
# EXCEL EXPORT
# ============================================================

def create_excel_download(
    data,
):

    model_data = data.model_dump()


    output = BytesIO()


    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        # -----------------------------------------------
        # Simple fields
        # -----------------------------------------------

        simple_rows = []

        for key, value in model_data.items():

            if not isinstance(
                value,
                list,
            ):

                simple_rows.append(
                    {
                        "field": key,
                        "value": value,
                    }
                )


        pd.DataFrame(
            simple_rows
        ).to_excel(
            writer,
            index=False,
            sheet_name="Basic",
        )


        # -----------------------------------------------
        # List fields
        # -----------------------------------------------

        for key in [
            "programming_languages",
            "ml_frameworks",
            "ml_tools",
            "ml_methods",
            "data_science_tools",
            "languages",
        ]:

            values = model_data.get(
                key,
                [],
            )


            rows = [
                {
                    "value": value
                }
                for value in values
            ]


            pd.DataFrame(
                rows
            ).to_excel(
                writer,
                index=False,
                sheet_name=key[
                    :31
                ],
            )


        # -----------------------------------------------
        # Experience
        # -----------------------------------------------

        pd.DataFrame(
            model_data[
                "experience"
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Experience",
        )


        # -----------------------------------------------
        # Education
        # -----------------------------------------------

        pd.DataFrame(
            model_data[
                "education"
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Education",
        )


        # -----------------------------------------------
        # Publications
        # -----------------------------------------------

        pd.DataFrame(
            model_data[
                "publications"
            ]
        ).to_excel(
            writer,
            index=False,
            sheet_name="Publications",
        )


    output.seek(0)


    return output.getvalue()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval():

    if "chunks" not in st.session_state:

        return None


    chunks = st.session_state[
        "chunks"
    ]


    if not chunks:

        return None


    test_question = (
        "What programming languages "
        "does the candidate know?"
    )


    expected_keywords = [
        "Python",
        "SQL",
        "Bash",
        "MATLAB",
        "Fortran",
    ]


    hybrid = hybrid_search(
        test_question,
        top_k=min(
            10,
            len(chunks),
        ),
    )


    reranked = rerank_results(
        test_question,
        hybrid,
        top_k=min(
            10,
            len(chunks),
        ),
    )


    reranked = ensure_query_section(
        test_question,
        reranked,
    )


    retrieved_text = " ".join(
        [
            r["payload"]["text"]
            for r in reranked
        ]
    ).lower()


    hits = sum(
        keyword.lower()
        in retrieved_text
        for keyword in expected_keywords
    )


    recall = (
        hits /
        len(expected_keywords)
        if expected_keywords
        else 0.0
    )


    return {
        "test_question": test_question,
        "keyword_recall": recall,
        "keywords_found": hits,
        "keywords_expected": len(
            expected_keywords
        ),
    }


# ============================================================
# UPLOAD
# ============================================================

st.header("📂 Upload PDF")


uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    if st.button(
        "🔍 Extract & Index Document",
        type="primary",
    ):

        with st.spinner(
            "Extracting PDF..."
        ):

            pages = extract_pdf(
                uploaded_file
            )


        non_empty_pages = [
            page
            for page in pages
            if page["text"].strip()
        ]


        if not non_empty_pages:

            st.error(
                "No text could be extracted from the PDF."
            )

            st.stop()


        chunks = create_chunks(
            pages,
            document_name=uploaded_file.name,
        )


        if not chunks:

            st.error(
                "No chunks could be created."
            )

            st.stop()


        with st.spinner(
            "Creating BGE-M3 embeddings and indexing Qdrant..."
        ):

            create_collection()


            vector_count = index_chunks(
                chunks,
                uploaded_file.name,
            )


        # BM25
        bm25 = build_bm25(
            chunks
        )


        # Session state
        st.session_state[
            "pages"
        ] = pages


        st.session_state[
            "non_empty_pages"
        ] = non_empty_pages


        st.session_state[
            "chunks"
        ] = chunks


        st.session_state[
            "vector_count"
        ] = vector_count


        st.session_state[
            "bm25"
        ] = bm25


        st.session_state[
            "bm25_chunks"
        ] = chunks


        st.session_state[
            "document_name"
        ] = uploaded_file.name


        st.session_state[
            "document_indexed"
        ] = True


        # Clear previous outputs

        for key in [
            "semantic_results",
            "bm25_results",
            "hybrid_results",
            "reranked_results",
            "qwen_answer",
            "citations",
            "structured_data",
        ]:

            st.session_state.pop(
                key,
                None,
            )


        st.success(
            f"Successfully extracted "
            f"{len(pages)} pages, "
            f"{len(non_empty_pages)} non-empty pages, "
            f"{len(chunks)} section-aware chunks, and "
            f"indexed {vector_count} vectors."
        )


# ============================================================
# STATISTICS
# ============================================================

if (
    "document_indexed"
    in st.session_state
):

    st.header(
        "📊 Document Statistics"
    )


    col1, col2, col3, col4 = st.columns(
        4
    )


    col1.metric(
        "Pages",
        len(
            st.session_state[
                "pages"
            ]
        ),
    )


    col2.metric(
        "Non-empty Pages",
        len(
            st.session_state[
                "non_empty_pages"
            ]
        ),
    )


    col3.metric(
        "Chunks",
        len(
            st.session_state[
                "chunks"
            ]
        ),
    )


    col4.metric(
        "Qdrant Vectors",
        st.session_state[
            "vector_count"
        ],
    )


# ============================================================
# SEARCH / RAG
# ============================================================

if (
    "document_indexed"
    in st.session_state
):

    st.header(
        "🧠 Document Intelligence"
    )


    question = st.text_input(
        "Enter your question",
        placeholder=(
            "e.g. What programming languages "
            "does the candidate know?"
        ),
    )


    max_k = min(
        10,
        len(
            st.session_state[
                "chunks"
            ]
        ),
    )


    default_k = min(
        4,
        max_k,
    )


    top_k = st.slider(
        "Top-K retrieval results",
        min_value=1,
        max_value=max_k,
        value=default_k,
    )


    if st.button(
        "🚀 Run Full RAG Pipeline",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # Semantic
            # ------------------------------------------------

            with st.spinner(
                "1/5 Semantic search..."
            ):

                semantic_results = semantic_search(
                    question,
                    top_k=max(
                        10,
                        top_k * 3,
                    ),
                )


                st.session_state[
                    "semantic_results"
                ] = semantic_results


            # ------------------------------------------------
            # BM25
            # ------------------------------------------------

            with st.spinner(
                "2/5 BM25 search..."
            ):

                bm25_results = bm25_search(
                    question,
                    top_k=max(
                        10,
                        top_k * 3,
                    ),
                )


                st.session_state[
                    "bm25_results"
                ] = bm25_results


            # ------------------------------------------------
            # Hybrid
            # ------------------------------------------------

            with st.spinner(
                "3/5 Hybrid RRF..."
            ):

                hybrid_results = hybrid_search(
                    question,
                    top_k=max(
                        10,
                        top_k * 3,
                    ),
                )


                st.session_state[
                    "hybrid_results"
                ] = hybrid_results


            # ------------------------------------------------
            # Reranker
            # ------------------------------------------------

            with st.spinner(
                "4/5 BGE Reranker..."
            ):

                reranked_results = rerank_results(
                    question,
                    hybrid_results,
                    top_k=max(
                        5,
                        top_k,
                    ),
                )


                reranked_results = ensure_query_section(
                    question,
                    reranked_results,
                )


                # Remove duplicate IDs
                unique = {}

                for result in reranked_results:

                    chunk_id = result[
                        "payload"
                    ].get(
                        "id"
                    )

                    if chunk_id not in unique:

                        unique[
                            chunk_id
                        ] = result


                reranked_results = list(
                    unique.values()
                )[:top_k]


                st.session_state[
                    "reranked_results"
                ] = reranked_results


            # ------------------------------------------------
            # Qwen
            # ------------------------------------------------

            with st.spinner(
                "5/5 Qwen RAG generation..."
            ):

                try:

                    answer = ask_qwen(
                        question,
                        reranked_results,
                    )


                    st.session_state[
                        "qwen_answer"
                    ] = answer


                except Exception as e:

                    st.session_state[
                        "qwen_answer"
                    ] = (
                        "Qwen could not be reached.\n\n"
                        f"Error: {e}"
                    )


            # ------------------------------------------------
            # Citations
            # ------------------------------------------------

            st.session_state[
                "citations"
            ] = generate_citations(
                reranked_results
            )


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

if (
    "reranked_results"
    in st.session_state
):

    st.header(
        "🎯 Retrieved Context"
    )


    for rank, result in enumerate(
        st.session_state[
            "reranked_results"
        ],
        start=1,
    ):

        payload = result[
            "payload"
        ]


        st.markdown(
            f"### #{rank} — "
            f"Reranker Score: "
            f"`{result['reranker_score']:.4f}`"
        )


        st.caption(
            f"Original RRF Score: "
            f"{result['rrf_score']:.6f}"
        )


        col1, col2, col3, col4 = st.columns(
            4
        )


        col1.write(
            f"**Document:** "
            f"{payload.get('document', 'Unknown')}"
        )


        col2.write(
            f"**Page:** "
            f"{payload.get('page', 'Unknown')}"
        )


        col3.write(
            f"**Chunk:** "
            f"{payload.get('chunk', 'Unknown')}"
        )


        col4.write(
            f"**Section:** "
            f"{payload.get('section', 'Unknown')}"
        )


        with st.expander(
            "View chunk"
        ):

            st.write(
                payload.get(
                    "text",
                    "",
                )
            )


        st.divider()


# ============================================================
# QWEN ANSWER
# ============================================================

if (
    "qwen_answer"
    in st.session_state
):

    st.header(
        "🤖 Qwen Answer"
    )


    st.write(
        st.session_state[
            "qwen_answer"
        ]
    )


# ============================================================
# CITATIONS
# ============================================================

if (
    "citations"
    in st.session_state
):

    st.header(
        "📚 Sources"
    )


    for citation in st.session_state[
        "citations"
    ]:

        st.markdown(
            f"**{citation}**"
        )


# ============================================================
# IDP
# ============================================================

if (
    "document_indexed"
    in st.session_state
):

    st.header(
        "🏢 Structured Document Extraction"
    )


    st.caption(
        "Section-aware extraction using "
        "Qwen + Pydantic validation."
    )


    if st.button(
        "🏢 Extract Structured Data",
        type="secondary",
    ):

        all_text = "\n\n".join(
            [
                page["text"]
                for page in st.session_state[
                    "pages"
                ]
                if page["text"].strip()
            ]
        )


        with st.spinner(
            "Extracting structured information with Qwen..."
        ):

            try:

                structured_data = (
                    extract_structured_data(
                        all_text
                    )
                )


                structured_data = (
                    postprocess_structured_data(
                        structured_data,
                        all_text,
                    )
                )


                st.session_state[
                    "structured_data"
                ] = structured_data


                st.success(
                    "Structured extraction and "
                    "Pydantic validation successful."
                )


            except json.JSONDecodeError as e:

                st.error(
                    "Qwen returned invalid JSON."
                )


                st.code(
                    str(e)
                )


            except ValidationError as e:

                st.error(
                    "Pydantic validation failed."
                )


                st.code(
                    str(e)
                )


            except Exception as e:

                st.error(
                    "Structured extraction failed."
                )


                st.code(
                    str(e)
                )


# ============================================================
# DISPLAY STRUCTURED DATA
# ============================================================

if (
    "structured_data"
    in st.session_state
):

    data = st.session_state[
        "structured_data"
    ]


    st.subheader(
        "📋 Validated Structured Data"
    )


    st.json(
        data.model_dump()
    )


    # --------------------------------------------------------
    # Programming
    # --------------------------------------------------------

    st.subheader(
        "💻 Skills Classification"
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        st.markdown(
            "**Programming Languages**"
        )

        st.write(
            data.programming_languages
        )


        st.markdown(
            "**ML Frameworks**"
        )

        st.write(
            data.ml_frameworks
        )


        st.markdown(
            "**ML / AI Methods**"
        )

        st.write(
            data.ml_methods
        )


    with col2:

        st.markdown(
            "**ML / Engineering Tools**"
        )

        st.write(
            data.ml_tools
        )


        st.markdown(
            "**Data Science Tools**"
        )

        st.write(
            data.data_science_tools
        )


        st.markdown(
            "**Human Languages**"
        )

        st.write(
            data.languages
        )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_bytes = create_json_download(
        data
    )


    st.download_button(
        label="⬇️ Download JSON",
        data=json_bytes,
        file_name="docintel_output.json",
        mime="application/json",
    )


    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    excel_bytes = create_excel_download(
        data
    )


    st.download_button(
        label="⬇️ Download Excel",
        data=excel_bytes,
        file_name="docintel_output.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

if (
    "document_indexed"
    in st.session_state
):

    st.header(
        "📈 Evaluation"
    )


    st.caption(
        "Small smoke benchmark for pipeline validation. "
        "This is not a production accuracy benchmark."
    )


    if st.button(
        "📊 Run Evaluation",
    ):

        with st.spinner(
            "Running evaluation..."
        ):

            evaluation = evaluate_retrieval()


        if evaluation:

            col1, col2, col3 = st.columns(
                3
            )


            col1.metric(
                "Keyword Recall",
                f"{evaluation['keyword_recall']:.2f}",
            )


            col2.metric(
                "Keywords Found",
                evaluation[
                    "keywords_found"
                ],
            )


            col3.metric(
                "Keywords Expected",
                evaluation[
                    "keywords_expected"
                ],
            )


            st.write(
                f"Test question: "
                f"**{evaluation['test_question']}**"
            )


        else:

            st.info(
                "No document available for evaluation."
            )


# ============================================================
# DEBUG - DETECTED SECTIONS
# ============================================================

if (
    "chunks"
    in st.session_state
):

    with st.expander(
        "🔬 Debug: Detected Sections"
    ):

        displayed_sections = set()


        for chunk in st.session_state[
            "chunks"
        ]:

            section = chunk.get(
                "section",
                "unknown",
            )


            if section in displayed_sections:
                continue


            displayed_sections.add(
                section
            )


            st.markdown(
                f"### `{section}`"
            )


            st.write(
                chunk["text"]
            )


# ============================================================
# DEBUG - SEMANTIC
# ============================================================

if (
    "semantic_results"
    in st.session_state
):

    with st.expander(
        "🔬 Debug: Semantic Search"
    ):

        for result in st.session_state[
            "semantic_results"
        ]:

            payload = result.payload


            st.write(
                {
                    "score": result.score,
                    "id": payload.get(
                        "id"
                    ),
                    "document": payload.get(
                        "document"
                    ),
                    "page": payload.get(
                        "page"
                    ),
                    "chunk": payload.get(
                        "chunk"
                    ),
                    "section": payload.get(
                        "section"
                    ),
                }
            )


# ============================================================
# DEBUG - BM25
# ============================================================

if (
    "bm25_results"
    in st.session_state
):

    with st.expander(
        "🔬 Debug: BM25 Search"
    ):

        for result in st.session_state[
            "bm25_results"
        ]:

            chunk = result[
                "chunk"
            ]


            st.write(
                {
                    "score": result[
                        "score"
                    ],
                    "id": chunk.get(
                        "id"
                    ),
                    "document": chunk.get(
                        "document"
                    ),
                    "page": chunk.get(
                        "page"
                    ),
                    "chunk": chunk.get(
                        "chunk"
                    ),
                    "section": chunk.get(
                        "section"
                    ),
                }
            )


# ============================================================
# DEBUG - HYBRID
# ============================================================

if (
    "hybrid_results"
    in st.session_state
):

    with st.expander(
        "🔬 Debug: Hybrid RRF"
    ):

        for result in st.session_state[
            "hybrid_results"
        ]:

            payload = result[
                "payload"
            ]


            st.write(
                {
                    "rrf_score": result[
                        "rrf_score"
                    ],
                    "id": payload.get(
                        "id"
                    ),
                    "document": payload.get(
                        "document"
                    ),
                    "page": payload.get(
                        "page"
                    ),
                    "chunk": payload.get(
                        "chunk"
                    ),
                    "section": payload.get(
                        "section"
                    ),
                }
            )


# ============================================================
# EXTRACTED CHUNKS
# ============================================================

if (
    "chunks"
    in st.session_state
):

    st.header(
        "🔎 Extracted Chunks"
    )


    for chunk in st.session_state[
        "chunks"
    ]:

        with st.expander(
            f"Page {chunk['page']} — "
            f"Chunk ID {chunk['id']} — "
            f"Local Chunk {chunk['chunk']} — "
            f"{chunk['section']}"
        ):

            st.write(
                chunk["text"]
            )


# ============================================================
# RAW PDF TEXT
# ============================================================

if (
    "pages"
    in st.session_state
):

    with st.expander(
        "📄 Extracted PDF Text"
    ):

        for page in st.session_state[
            "pages"
        ]:

            st.markdown(
                f"### Page {page['page']}"
            )

            st.text(
                page["text"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "DocIntel AI — PDF → Section-aware Extraction → "
    "BGE-M3 → Qdrant → BM25 → Hybrid RRF → "
    "BGE Reranker → Qwen RAG → Citations → "
    "Section-based Qwen IDP → Pydantic → JSON/Excel → Evaluation"
)
st.markdown(
    """
    <div style="
        text-align: center;
        padding: 30px 0 15px 0;
        margin-top: 50px;
        border-top: 1px solid rgba(128,128,128,0.25);
        color: #888;
        font-size: 13px;
    ">
        © 2026 <strong>Naeim Mousavi</strong> · DocIntel AI<br>
        Enterprise Document Intelligence Engine
    </div>
    """,
    unsafe_allow_html=True
)