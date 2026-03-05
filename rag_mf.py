# ==========================================
# OPTIMIZED MUTUAL FUND RAG BACKEND
# ==========================================

import os
import re
import json
from dotenv import load_dotenv
import chromadb
from google import genai

# ==========================================
# CONFIG
# ==========================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Dataset + scheme mapping
SCHEME_FILES = {
    "PPLF_dataset.txt": (
        "PPLF",
        "https://amc.ppfas.com/downloads/factsheet/2026/ppfas-mf-factsheet-for-January-2026.pdf"
    ),

    "PPFC_dataset.txt": (
        "PPFCF",
        "https://amc.ppfas.com/downloads/factsheet/2026/flexi-cap-fund.pdf"
    ),

    "taxsaver_data.txt": (
        "PPTSF",
        "https://amc.ppfas.com/downloads/factsheet/2026/tax-saver-fund.pdf"
    )
}

# ==========================================
# PERMANENT CACHE
# ==========================================

CACHE_FILE = "bot_memory.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        response_cache = json.load(f)
else:
    response_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(response_cache, f)

# ==========================================
# VECTOR STORE
# ==========================================

def build_vector_store():

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("mf_data")

    if collection.count() > 0:
        print(f"Loaded existing DB with {collection.count()} chunks")
        return collection

    print("Building new vector DB...")

    chunk_id = 0

    for filename, (scheme, source) in SCHEME_FILES.items():

        if not os.path.exists(filename):
            print("Missing dataset:", filename)
            continue

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = [c.strip() for c in text.split("[END]") if c.strip()]

        print(f"{filename}: {len(chunks)} chunks")

        for chunk in chunks:

            embedding = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            ).embeddings[0].values

            collection.add(
                ids=[str(chunk_id)],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "scheme": scheme,
                    "source": source
                }]
            )

            chunk_id += 1

    print("Vector DB ready")
    return collection

# ==========================================
# ROUTER
# ==========================================

def route_query(query):

    q = query.lower()

    if "quantity" in q or "units" in q:
        return "quantity"

    if "expense ratio" in q:
        return "expense_ratio"

    if "exit load" in q:
        return "exit_load"

    if "aum" in q:
        return "aum"

    if "benchmark" in q:
        return "benchmark"

    if "riskometer" in q:
        return "riskometer"

    if any(x in q for x in ["should i", "best fund", "better fund"]):
        return "refuse"

    return "llm"

# ==========================================
# SIMPLE FACT SEARCH
# ==========================================

def simple_fact_search(collection, query, scheme):

    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        where={"scheme": scheme}
    )

    if not results["documents"][0]:
        return "Information not found."

    doc = results["documents"][0][0]
    source = results["metadatas"][0][0]["source"]

    return f"{doc}\n\nSource: {source}\nLast updated: Feb 2026"

# ==========================================
# QUANTITY HANDLER
# ==========================================

def handle_quantity(collection, query, scheme):

    results = collection.get(
        include=["documents", "metadatas"]
    )

    docs = results["documents"]
    metas = results["metadatas"]

    total = 0
    entries = []
    name = None
    source = None

    clean_query = re.sub(r"[^\w\s]", "", query.lower())

    for i, text in enumerate(docs):

        if metas[i]["scheme"] != scheme:
            continue

        name_match = re.search(r"INSTRUMENT:\s*(.*?)\s*\(", text)
        qty_match = re.search(r"QUANTITY:\s*([\d,\.]+)", text)

        if name_match and qty_match:

            instrument = name_match.group(1).lower()

            terms = [
                w for w in clean_query.split()
                if len(w) > 3
            ]

            if terms and all(t in instrument for t in terms):

                qty = float(qty_match.group(1).replace(",", ""))

                total += qty
                entries.append(str(int(qty)))

                name = instrument
                source = metas[i]["source"]

    if total == 0:
        return "Holding not found."

    return (
        f"Total quantity of {name}: {int(total)} units\n\n"
        f"Entries: {', '.join(entries)}\n\n"
        f"Source: {source}"
    )

# ==========================================
# LLM HANDLER
# ==========================================

def handle_llm(collection, query, scheme):

    if query in response_cache:
        return response_cache[query] + "\n\n⚡ Loaded from cache"

    embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        where={"scheme": scheme}
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    sources = list(set([m["source"] for m in metas]))

    context = "\n\n".join(docs)

    prompt = f"""
You are a strictly factual mutual fund assistant.

Rules:
Use ONLY the provided context.
Do not guess.
Maximum 3 sentences.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    answer = response.text.strip()

    final = (
        f"{answer}\n\nSources:\n"
        + "\n".join(sources)
        + "\nLast updated: Feb 2026"
    )

    response_cache[query] = final
    save_cache()

    return final

# ==========================================
# MAIN FUNCTION
# ==========================================

def ask_question(collection, query, scheme):

    route = route_query(query)

    if route == "refuse":
        return (
            "I provide factual scheme information only. "
            "Consult a financial advisor for investment decisions."
        )

    if route == "quantity":
        return handle_quantity(collection, query, scheme)

    if route in [
        "expense_ratio",
        "exit_load",
        "aum",
        "benchmark",
        "riskometer"
    ]:
        return simple_fact_search(collection, query, scheme)

    return handle_llm(collection, query, scheme)