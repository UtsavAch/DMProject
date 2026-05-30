#!/usr/bin/env python3
"""
rdf_loader_fuseki.py
Reads MIND news.tsv and behaviors.tsv and emits Turtle triples for Fuseki.
Uses rdflib and SentenceTransformer for embeddings (embedding files saved externally).
"""

import os
import json
import csv
import random
import pathlib
from urllib.parse import quote, urlparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD
from sentence_transformers import SentenceTransformer
import requests

# -------------------------
# Configuration (edit as needed)
# -------------------------
DATA_DIR = "./MINDsmall_train"
NEWS_TSV = os.path.join(DATA_DIR, "news.tsv")
BEHAVIORS_TSV = os.path.join(DATA_DIR, "behaviors.tsv")
OUTPUT_TTL = "data_combined.ttl"
EMBEDDING_DIR = "./embeddings"          # where .npy vectors will be stored
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # model used in your PG loader
FUSEKI_ENDPOINT = "http://localhost:3030/dataset"  # change if needed
USE_FUSEKI_UPLOAD = False               # set True to POST to Fuseki automatically
GRAPH_URI = "http://example.org/graph/news"  # named graph to upload to Fuseki

# Create directories
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Initialize embedding model (same as your PG loader)
random.seed(42)
model = SentenceTransformer(EMBEDDING_MODEL)

# Namespaces
EX = Namespace("http://example.org/news#")
SCHEMA = Namespace("http://schema.org/")
WIKIDATA = "http://www.wikidata.org/entity/"

# Helper: safe IRI creation
def make_iri(prefix: str, raw_id: str) -> URIRef:
    safe = quote(str(raw_id).strip().replace(" ", "_"), safe="-_.~")
    return URIRef(f"{prefix}{safe}")

# Helper: create Wikidata IRI if looks like Q12345
def wikidata_iri_if_present(wikidata_id: str):
    if not wikidata_id:
        return None
    wikidata_id = str(wikidata_id).strip()
    if wikidata_id.upper().startswith("Q") and wikidata_id[1:].isdigit():
        return URIRef(WIKIDATA + wikidata_id)
    return None

# Ground truth synthetic pool fallback list for sources
GLOBAL_SOURCES = [
    {"name": "BBC News", "url": "https://bbc.com"},
    {"name": "CNN", "url": "https://cnn.com"},
    {"name": "Reuters", "url": "https://reuters.com"},
    {"name": "The New York Times", "url": "https://nytimes.com"},
    {"name": "The Guardian", "url": "https://theguardian.com"},
    {"name": "Bloomberg", "url": "https://bloomberg.com"},
    {"name": "ESPN", "url": "https://espn.com"},
    {"name": "TechCrunch", "url": "https://techcrunch.com"}
]


def clean_source_mapping(url_string, article_id):
    """
    Extracts actual domain names from real data URLs when possible,
    otherwise matches down to a rich synthetic source group pool.
    """
    if url_string and "msn.com" not in url_string:
        try:
            parsed = urlparse(url_string)
            domain = parsed.netloc.replace("www.", "")
            if domain:
                return domain.split(".")[0].upper(), f"https://{domain}"
        except Exception:
            pass


    

    try:
        num_id = int(''.join(filter(str.isdigit, article_id)))
    except ValueError:
        num_id = len(article_id)

    selected = GLOBAL_SOURCES[num_id % len(GLOBAL_SOURCES)]
    return selected["name"], selected["url"]

# Initialize RDF graph
g = Graph()
g.bind("ex", EX)
g.bind("schema", SCHEMA)
g.bind("rdfs", RDFS)
g.bind("xsd", XSD)

# -------------------------
# Phase 1: Process news.tsv -> Articles, Topics, Sources, Entities, Embeddings
# -------------------------
print("Loading news.tsv and creating Article triples...")
news_cols = ["nid", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
df_news = pd.read_csv(NEWS_TSV, sep="\t", names=news_cols, keep_default_na=False, quoting=csv.QUOTE_NONE)

for _, row in tqdm(df_news.iterrows(), total=len(df_news)):
    nid = row["nid"]
    if not nid:
        continue
    article_iri = make_iri(EX, f"article_{nid}")
    # Types
    g.add((article_iri, RDF.type, EX.Article))
    g.add((article_iri, RDF.type, SCHEMA.Article))
    # Basic properties
    if row.get("title"):
        g.add((article_iri, SCHEMA.headline, Literal(row["title"])))
    if row.get("abstract"):
        g.add((article_iri, SCHEMA.articleBody, Literal(row["abstract"])))
    if row.get("url"):
        g.add((article_iri, SCHEMA.url, Literal(row["url"])))
    # published_at: use ingestion time as fallback
    g.add((article_iri, SCHEMA.datePublished, Literal(datetime.utcnow().isoformat(), datatype=XSD.dateTime)))
    # identifier
    g.add((article_iri, SCHEMA.identifier, Literal(str(nid))))

    # Topic / Category
    cat = row.get("category") or row.get("subcategory")
    if cat:
        cat_id = cat.strip().lower().replace(" ", "_")
        cat_iri = make_iri(EX, f"topic_{cat_id}")
        g.add((cat_iri, RDF.type, EX.Topic))
        g.add((cat_iri, SCHEMA.name, Literal(cat)))
        g.add((article_iri, EX.hasTopic, cat_iri))

    # Source: derive domain or fallback synthetic mapping with notebook-aligned source names and URLs
    src_name, src_url = clean_source_mapping(row.get("url", ""), row["nid"])
    src_id = src_name.lower().replace(" ", "_").replace(".", "_")
    src_iri = make_iri(EX, f"source_{src_id}")
    g.add((src_iri, RDF.type, EX.Source))
    g.add((src_iri, SCHEMA.name, Literal(src_name)))
    g.add((src_iri, SCHEMA.url, Literal(src_url)))
    # link article -> source
    g.add((article_iri, EX.publishedBy, src_iri))
    g.add((src_iri, EX.published, article_iri))
    g.add((article_iri, EX.fromSource, src_iri))

    # Embedding: compute and save externally, create Embedding node with ref
    text_to_embed = f"{row.get('title','')}. {row.get('abstract','')}"
    try:
        vec = model.encode(text_to_embed)
        emb_filename = f"emb_{nid}.npy"
        emb_path = os.path.join(EMBEDDING_DIR, emb_filename)
        np.save(emb_path, vec)
        emb_iri = make_iri(EX, f"emb_{nid}")
        g.add((emb_iri, RDF.type, EX.Embedding))
        g.add((emb_iri, EX.embeddingVectorRef, Literal(str(pathlib.Path(emb_path).absolute()))))
        g.add((emb_iri, SCHEMA.identifier, Literal(f"emb_{nid}")))
        g.add((article_iri, EX.hasEmbedding, emb_iri))
    except Exception as e:
        # skip embedding if model fails
        pass

    # Entities: title_entities and abstract_entities are JSON arrays in MIND
    for col in ["title_entities", "abstract_entities"]:
        raw = row.get(col)
        if raw:
            try:
                ents = json.loads(raw)
                for ent in ents:
                    # ent expected to have keys: 'WikidataId', 'Label', 'Type'
                    wikidata_id = ent.get("WikidataId") or ent.get("WikidataID") or ent.get("wikidataId")
                    ent_label = ent.get("Label") or ent.get("label") or ent.get("Name") or ""
                    ent_type = ent.get("Type") or ent.get("type") or "Other"
                    ent_iri = None
                    if wikidata_id:
                        ent_iri = wikidata_iri_if_present(wikidata_id)
                    if not ent_iri:
                        ent_id = ent_label or (wikidata_id or "unknown")
                        ent_iri = make_iri(EX, f"entity_{ent_id}")
                    g.add((ent_iri, RDF.type, EX.Entity))
                    if ent_label:
                        g.add((ent_iri, SCHEMA.name, Literal(ent_label)))
                    g.add((ent_iri, EX.entityType, Literal(ent_type)))
                    # link article -> entity
                    g.add((article_iri, EX.mentions, ent_iri))
            except Exception:
                continue

# -------------------------
# Phase 2: Process behaviors.tsv -> Users and ReadEvents, Follows, InterestedIn
# -------------------------
print("Loading behaviors.tsv and creating Subscriber triples...")
beh_cols = ["impression_id", "uid", "time", "history", "impressions"]
df_beh = pd.read_csv(BEHAVIORS_TSV, sep="\t", names=beh_cols, keep_default_na=False, quoting=csv.QUOTE_NONE)

# Create Subscriber nodes (unique uids)
unique_users = df_beh[["uid", "history"]].drop_duplicates(subset=["uid"])
for _, row in tqdm(unique_users.iterrows(), total=len(unique_users)):
    uid = row["uid"]
    if not uid:
        continue
    user_iri = make_iri(EX, f"user_{uid}")
    g.add((user_iri, RDF.type, EX.Subscriber))
    g.add((user_iri, SCHEMA.identifier, Literal(str(uid))))
    g.add((user_iri, SCHEMA.name, Literal(f"User_{uid}")))
    g.add((user_iri, SCHEMA.email, Literal(f"{uid}@example.com")))
    freq = random.choice(["daily", "weekly", "monthly"])
    g.add((user_iri, EX.frequency, Literal(freq)))

    # history -> READ edges (we model as ReadEvent nodes to carry ts/clicked if needed)
    history = row.get("history")
    if history:
        history_list = [h for h in history.split(" ") if h]
        for artid in history_list:
            art_iri = make_iri(EX, f"article_{artid}")
            # create ReadEvent reified node
            re = BNode()
            g.add((re, RDF.type, EX.ReadEvent))
            g.add((re, EX.sourceUser, user_iri))
            g.add((re, EX.targetArticle, art_iri))
            g.add((re, EX.ts, Literal(datetime.utcnow().isoformat(), datatype=XSD.dateTime)))
            g.add((re, EX.clicked, Literal(True, datatype=XSD.boolean)))
            # optional direct triple for convenience
            g.add((user_iri, EX.read, art_iri))

# Synthetic FOLLOWS: sample a few sources per user (reproducible)
all_sources = set(g.subjects(RDF.type, EX.Source))
all_sources = list(all_sources)
for _, row in tqdm(unique_users.iterrows(), total=len(unique_users)):
    uid = row["uid"]
    if not uid:
        continue
    user_iri = make_iri(EX, f"user_{uid}")
    num_to_follow = random.randint(0, 3)
    if num_to_follow > 0 and all_sources:
        followed = random.sample(all_sources, min(num_to_follow, len(all_sources)))
        for src in followed:
            g.add((user_iri, EX.follows, src))

# Infer INTERESTED_IN: if user read >=2 articles in same topic, create ex:interestedIn
print("Inferring interested topics for users...")
# naive approach: count per user-topic by scanning ReadEvent triples
user_topic_counts = {}
for re in g.subjects(RDF.type, EX.ReadEvent):
    user = next(g.objects(re, EX.sourceUser), None)
    art = next(g.objects(re, EX.targetArticle), None)
    if user and art:
        for t in g.objects(art, EX.hasTopic):
            key = (str(user), str(t))
            user_topic_counts[key] = user_topic_counts.get(key, 0) + 1

for (user_uri, topic_uri), count in user_topic_counts.items():
    if count >= 2:
        g.add((URIRef(user_uri), EX.interestedIn, URIRef(topic_uri)))

# -------------------------
# Phase 3: Optional: precompute top-k embedding similarity and create Similarity nodes
# (This is optional and can be expensive; here we do a tiny sample for demonstration)
# -------------------------
print("Optionally creating a small set of similarity triples (sample)...")
# Collect a small sample of embeddings (first N) to compute pairwise cosine
sample_articles = []
for s in g.subjects(RDF.type, EX.Embedding):
    # embedding id like ex:emb_N
    emb_ref = next(g.objects(s, EX.embeddingVectorRef), None)
    if emb_ref:
        sample_articles.append((s, str(emb_ref)))
    if len(sample_articles) >= 200:
        break

# compute pairwise for sample (if you want full dataset, use a vector DB)
def load_vec(path):
    try:
        return np.load(path)
    except Exception:
        return None

sample_vectors = []
sample_article_iris = []
for emb_iri, path in sample_articles:
    vec = load_vec(path)
    if vec is not None:
        # find article that links to this embedding
        for art in g.subjects(EX.hasEmbedding, emb_iri):
            sample_article_iris.append(art)
            sample_vectors.append(vec)
            break

if len(sample_vectors) >= 2:
    import sklearn.metrics.pairwise as pw
    sims = pw.cosine_similarity(np.vstack(sample_vectors))
    top_k = 5
    for i, art_i in enumerate(sample_article_iris):
        # find top_k neighbors
        idxs = sims[i].argsort()[::-1][1:top_k+1]
        for j in idxs:
            art_j = sample_article_iris[j]
            score = float(sims[i][j])
            sim_node = BNode()
            g.add((sim_node, RDF.type, EX.Similarity))
            g.add((sim_node, EX.sourceArticle, art_i))
            g.add((sim_node, EX.targetArticle, art_j))
            g.add((sim_node, EX.score, Literal(score, datatype=XSD.float)))
            g.add((sim_node, EX.method, Literal("embedding")))

# -------------------------
# -------------------------
# SPARQL query helpers and convenience functions (top-level)
# -------------------------
PREFIXES = """
PREFIX ex: <http://example.org/news#>
PREFIX schema: <http://schema.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def _sparql_result_to_df(result):
    vars = list(result.vars)
    rows = []
    for r in result:
        row = {}
        for v in vars:
            try:
                val = r.get(v)
            except Exception:
                val = r[v] if v in r else None
            row[str(v)] = str(val) if val is not None else None
        rows.append(row)
    if rows:
        return pd.DataFrame(rows, columns=[str(v) for v in vars])
    else:
        return pd.DataFrame(columns=[str(v) for v in vars])


def graph_overlap_recommendations(nid, limit=5):
    q = PREFIXES + """
SELECT ?candidateId ?Title (COUNT(DISTINCT ?sharedNode) AS ?SharedConnections)
WHERE {{
  BIND(ex:article_{nid} AS ?target)
  ?target (ex:hasTopic|ex:mentions) ?sharedNode .
  ?candidate (ex:hasTopic|ex:mentions) ?sharedNode .
  FILTER(?candidate != ?target)
  ?candidate schema:identifier ?candidateId .
  OPTIONAL {{ ?candidate schema:headline ?Title }}
}}
GROUP BY ?candidate ?candidateId ?Title
ORDER BY DESC(?SharedConnections)
LIMIT {limit}
""".format(nid=nid, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def similarity_recommendations(nid, limit=5):
    q = PREFIXES + """
SELECT ?candidateId ?Title ?Content ?SimilarityScore
WHERE {{
  ?sim rdf:type ex:Similarity ;
       ex:sourceArticle ex:article_{nid} ;
       ex:targetArticle ?candidate ;
       ex:score ?SimilarityScore ;
       ex:method "embedding" .
  ?candidate schema:identifier ?candidateId .
  OPTIONAL {{ ?candidate schema:headline ?Title }}
  OPTIONAL {{ ?candidate schema:articleBody ?Content }}
  FILTER(?candidate != ex:article_{nid})
}}
ORDER BY DESC(?SimilarityScore)
LIMIT {limit}
""".format(nid=nid, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def hybrid_recommendations_for_user(uid, limit=5):
    q = PREFIXES + """
SELECT ?candidateId ?Title ?URL ?FinalScore
WHERE {{
  {{ SELECT ?u ?last WHERE {{
      ?u a ex:Subscriber ; schema:identifier "{uid}" .
      ?re a ex:ReadEvent ; ex:sourceUser ?u ; ex:targetArticle ?last ; ex:ts ?ts .
    }} ORDER BY DESC(?ts) LIMIT 1 }}
  ?sim a ex:Similarity ; ex:sourceArticle ?last ; ex:targetArticle ?candidate ; ex:score ?score ; ex:method "embedding" .
  FILTER NOT EXISTS {{ ?re2 a ex:ReadEvent ; ex:sourceUser ?u ; ex:targetArticle ?candidate . }}
  BIND( IF(EXISTS {{ ?u ex:follows ?src . ?candidate ex:fromSource ?src }} , 0.3, 0.0) AS ?sourceBoost )
  BIND( IF(EXISTS {{ ?u ex:interestedIn ?t . ?candidate ex:hasTopic ?t }} , 0.2, 0.0) AS ?topicBoost )
  BIND( (xsd:float(?score) + xsd:float(?sourceBoost) + xsd:float(?topicBoost)) AS ?FinalScore )
  ?candidate schema:identifier ?candidateId .
  OPTIONAL {{ ?candidate schema:headline ?Title }}
  OPTIONAL {{ ?candidate schema:url ?URL }}
}}
ORDER BY DESC(?FinalScore)
LIMIT {limit}
""".format(uid=uid, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def detect_trending_topics_sparql(min_article_count=3, min_velocity=5.0, limit=5):
    q = PREFIXES + """
SELECT ?topicName ?ArticleCount ?TotalReads (ROUND((?TotalReads / ?ArticleCount), 2) AS ?MomentumScore)
WHERE {{
  {{ SELECT ?t (COUNT(DISTINCT ?a) AS ?ArticleCount) (COUNT(?s) AS ?TotalReads) WHERE {{
      ?s a ex:Subscriber .
      ?re a ex:ReadEvent ; ex:sourceUser ?s ; ex:targetArticle ?a .
      ?a ex:hasTopic ?t .
    }} GROUP BY ?t }}
  ?t schema:name ?topicName .
  FILTER(?ArticleCount > {min_article_count})
  BIND( (xsd:float(?TotalReads) / xsd:float(?ArticleCount)) AS ?TrendVelocity )
  FILTER(?TrendVelocity > {min_velocity})
}}
ORDER BY DESC(?TrendVelocity)
LIMIT {limit}
""".format(min_article_count=min_article_count, min_velocity=min_velocity, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def serendipity_recommendations_sparql(uid, limit=5):
    q = PREFIXES + """
SELECT DISTINCT ?SurpriseArticleId ?Title ?UnfamiliarPublisher (GROUP_CONCAT(DISTINCT ?ename; separator=", ") AS ?OverlappingEntities)
WHERE {{
  ?u a ex:Subscriber ; schema:identifier "{uid}" .
  ?u ex:interestedIn ?t .
  ?u ex:read ?pastArticle .
  ?pastArticle ex:hasTopic ?t .
  ?pastArticle ex:mentions ?e .
  ?e schema:name ?ename .
  ?candidate ex:mentions ?e .
  ?candidate ex:hasTopic ?t .
  FILTER NOT EXISTS {{ ?u ex:read ?candidate }}
  ?candidate ex:fromSource ?altSource .
  FILTER NOT EXISTS {{ ?u ex:follows ?altSource }}
  ?candidate schema:identifier ?SurpriseArticleId .
  OPTIONAL {{ ?candidate schema:headline ?Title }}
  OPTIONAL {{ ?altSource schema:name ?UnfamiliarPublisher }}
}}
GROUP BY ?SurpriseArticleId ?Title ?UnfamiliarPublisher
ORDER BY RAND()
LIMIT {limit}
""".format(uid=uid, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def analyze_entity_co_occurrence_sparql(entity_qid, limit=10):
    # entity_qid expected like 'Q43274' or full URI
    if entity_qid.startswith('http'):
        target = f"<{entity_qid}>"
    elif entity_qid.upper().startswith('Q'):
        target = f"<http://www.wikidata.org/entity/{entity_qid}>"
    else:
        target = f"<http://example.org/news#entity_{entity_qid}>"
    q = PREFIXES + """
SELECT ?RelatedEntity ?EntityType (COUNT(DISTINCT ?a) AS ?NumberOfCoMentions)
WHERE {{
  BIND({target} AS ?target)
  ?a ex:mentions ?target .
  ?a ex:mentions ?coOccurring .
  FILTER(?coOccurring != ?target)
  ?coOccurring schema:name ?RelatedEntity .
  OPTIONAL {{ ?coOccurring ex:entityType ?EntityType }}
}}
GROUP BY ?coOccurring ?RelatedEntity ?EntityType
ORDER BY DESC(?NumberOfCoMentions)
LIMIT {limit}
""".format(target=target, limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


def publisher_hub_analysis_sparql(limit=50):
    q = PREFIXES + """
SELECT ?FromPublisherName ?ToPublisherName (COUNT(DISTINCT ?s) AS ?MigratedUsers)
WHERE {{
  ?s a ex:Subscriber .
  ?re1 a ex:ReadEvent ; ex:sourceUser ?s ; ex:targetArticle ?a1 .
  ?re2 a ex:ReadEvent ; ex:sourceUser ?s ; ex:targetArticle ?a2 .
  ?a1 ex:fromSource ?src1 .
  ?a2 ex:fromSource ?src2 .
  FILTER(?src1 != ?src2)
  ?a1 schema:datePublished ?d1 .
  ?a2 schema:datePublished ?d2 .
  FILTER(xsd:dateTime(?d1) < xsd:dateTime(?d2))
  ?src1 schema:name ?FromPublisherName .
  ?src2 schema:name ?ToPublisherName .
}}
GROUP BY ?src1 ?src2 ?FromPublisherName ?ToPublisherName
HAVING (COUNT(DISTINCT ?s) > 1)
ORDER BY DESC(?MigratedUsers)
LIMIT {limit}
""".format(limit=limit)
    res = g.query(q)
    return _sparql_result_to_df(res)


# -------------------------
# Write Turtle file
# -------------------------
print(f"Serializing RDF to {OUTPUT_TTL} ...")
g.serialize(destination=OUTPUT_TTL, format="turtle")
print("Serialization complete.")

# -------------------------
# Optional: upload to Fuseki via Graph Store Protocol (HTTP POST)
# -------------------------
if USE_FUSEKI_UPLOAD:
    print("Uploading TTL to Fuseki...")
    url = f"{FUSEKI_ENDPOINT}/data?graph={quote(GRAPH_URI, safe='')}"
    headers = {"Content-Type": "text/turtle"}
    with open(OUTPUT_TTL, "rb") as fh:
        r = requests.post(url, data=fh, headers=headers)
    if r.status_code in (200, 201, 204):
        print("Upload successful.")
    else:
        print("Upload failed:", r.status_code, r.text)

print("Done.")
