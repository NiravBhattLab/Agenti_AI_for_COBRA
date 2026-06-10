"""
retrieve.py
===========
Retrieval module for the GSM workflow planning system.

Two-phase retrieval strategy
-----------------------------
Phase 1 — Coarse (Tier 0 + Tier 1)
  Embeds the user query and searches tier 0 (paper level) and tier 1
  (category level) separately. Identifies which papers and pipeline
  stages are relevant to the query. Produces a shortlist of paper_ids.

Phase 2 — Fine (Tier 2 + Tier 3)
  Searches tier 2 (sub-workflow level) and tier 3 (individual step level).
  Tier 2 hits are the primary plan seeds — each carries a complete step_ids
  list representing a validated sub-workflow. Tier 3 hits provide
  supplementary individual steps not captured by any tier 2 group.

  Phase 2 can optionally be filtered to only search within the paper_ids
  shortlisted in phase 1. This improves precision but risks missing
  relevant workflows from papers that didn't surface at tier 0/1.
  Default is unfiltered — recommended unless recall is too low.

Output
------
retrieve() returns a RetrievalResult dict consumed by the planning module:

  {
    "query": str,
    "phase1": {
      "tier0_hits": [Hit, ...],      # paper-level matches
      "tier1_hits": [Hit, ...],      # category-level matches
      "shortlisted_paper_ids": [...] # union of paper_ids from tier 0+1
    },
    "phase2": {
      "tier2_hits": [Hit, ...],      # sub-workflow matches (primary plan seeds)
      "tier3_hits": [Hit, ...]       # individual step matches (supplementary)
    }
  }

  Hit = {
    "paper_id":      str,
    "entry_id":      str,
    "score":         float,   # similarity 0-1 (higher = more relevant)
    "document":      str,     # the text that was embedded, for inspection
    "step_ids":      [str],   # decoded from JSON string; the steps in this entry
    "step_category": str,     # pipeline category, empty for tier 0
    "technique":     str,     # technique field from raw step, tier 3 only
    "tier":          int      # 0 / 1 / 2 / 3
  }

BGE asymmetric prompting
------------------------
Documents were indexed with BGE_DOC_PREFIX. Queries MUST use BGE_QUERY_PREFIX.
Using the wrong prefix degrades retrieval quality. These constants are defined
here and should not be changed without re-indexing.

Usage
-----
  from planner.retrieve import GSMRetriever

  retriever = GSMRetriever()  # uses defaults from planner/config.py

  result = retriever.retrieve("how do I simulate SDH inhibition and run FVA?")

  # Access phase 2 sub-workflow hits (primary plan seeds)
  for hit in result["phase2"]["tier2_hits"]:
      print(hit["score"], hit["step_ids"], hit["document"][:80])

  # Access individual step hits (supplementary)
  for hit in result["phase2"]["tier3_hits"]:
      print(hit["score"], hit["technique"], hit["document"][:80])

Dependencies
------------
  pip install chromadb sentence-transformers
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from planner.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# ── BGE asymmetric prefix constants ──────────────────────────────────────────
# Documents in the index were embedded with BGE_DOC_PREFIX.
# User queries MUST be embedded with BGE_QUERY_PREFIX.
# Do not change these without rebuilding the index.

BGE_DOC_PREFIX   = "Represent this document for retrieval: "
BGE_QUERY_PREFIX = "Represent this query for retrieval: "


class GSMRetriever:
    """
    Retrieves relevant GSM workflow steps from a ChromaDB collection
    using a two-phase search strategy.

    The model is loaded once at init to avoid reload overhead on every query.
    Instantiate once and call retrieve() repeatedly.
    """

    def __init__(
        self,
        chroma_dir:      str  = CHROMA_DB_PATH,
        collection_name: str  = COLLECTION_NAME,
        model_name:      str  = EMBEDDING_MODEL,
    ):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  Embedding dimension : {self.model.get_sentence_embedding_dimension()}")

        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(name=collection_name)
        print(f"  Collection '{collection_name}': {self.collection.count():,} entries\n")

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,

        # Number of results to return per tier
        n_tier0: int = 5,
        n_tier1: int = 5,
        n_tier2: int = 5,
        n_tier3: int = 15,

        # If True, phase 2 is filtered to only the papers shortlisted in phase 1.
        # Increases precision, reduces recall. Default False is recommended.
        filter_by_phase1: bool = False,

        # Optional hard filter on technique field (tier 3 only).
        # e.g. technique_filter="fluxVariability" restricts tier 3 search
        # to steps that use FVA before semantic ranking runs.
        # Pass None to disable (default).
        technique_filter: Optional[str] = None,

    ) -> dict:
        """
        Run the full two-phase retrieval for a user query.
        Returns a RetrievalResult dict (see module docstring for schema).
        """
        query_embedding = self._embed_query(query)

        # ── Phase 1: Coarse ───────────────────────────────────────────────────
        tier0_hits = self._search(query_embedding, tiers=[0], n_results=n_tier0)
        tier1_hits = self._search(query_embedding, tiers=[1], n_results=n_tier1)

        # Collect all paper_ids surfaced in phase 1 for optional phase 2 filtering
        shortlisted = list({
            hit["paper_id"]
            for hit in tier0_hits + tier1_hits
        })

        # ── Phase 2: Fine ─────────────────────────────────────────────────────
        # Tier 2: sub-workflow hits — primary plan seeds
        tier2_hits = self._search(
            query_embedding,
            tiers=[2],
            n_results=n_tier2,
            paper_ids=shortlisted if filter_by_phase1 else None,
        )

        # Tier 3: individual step hits — supplementary
        tier3_hits = self._search(
            query_embedding,
            tiers=[3],
            n_results=n_tier3,
            paper_ids=shortlisted if filter_by_phase1 else None,
            technique=technique_filter,
        )

        return {
            "query": query,
            "phase1": {
                "tier0_hits":             tier0_hits,
                "tier1_hits":             tier1_hits,
                "shortlisted_paper_ids":  shortlisted,
            },
            "phase2": {
                "tier2_hits": tier2_hits,
                "tier3_hits": tier3_hits,
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """Embed a user query with the BGE query prefix."""
        vec = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
        )
        return vec.tolist()

    def _search(
        self,
        query_embedding: list[float],
        tiers:           list[int],
        n_results:       int,
        paper_ids:       Optional[list[str]] = None,
        technique:       Optional[str]       = None,
    ) -> list[dict]:
        """
        Run a single ChromaDB query against the specified tiers with
        optional metadata filters.

        paper_ids and technique are combined as AND conditions when both
        are provided. If paper_ids is an empty list, returns [] immediately
        to avoid a ChromaDB error on empty $in filters.
        """
        # Guard: empty paper_ids list would cause a ChromaDB filter error
        if paper_ids is not None and len(paper_ids) == 0:
            return []

        # Guard: n_results must not exceed collection size
        n_results = min(n_results, self.collection.count())
        if n_results == 0:
            return []

        where = self._build_where(tiers, paper_ids, technique)

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(raw)

    def _build_where(
        self,
        tiers:     list[int],
        paper_ids: Optional[list[str]],
        technique: Optional[str],
    ) -> dict:
        """
        Construct a ChromaDB where-filter dict.

        ChromaDB requires $and to have at least 2 conditions.
        Single-condition filters are passed directly without $and.
        """
        conditions = []

        # Tier filter
        if len(tiers) == 1:
            conditions.append({"tier": tiers[0]})
        else:
            conditions.append({"tier": {"$in": tiers}})

        # Paper_id filter (phase 1 shortlist)
        if paper_ids is not None:
            if len(paper_ids) == 1:
                conditions.append({"paper_id": paper_ids[0]})
            else:
                conditions.append({"paper_id": {"$in": paper_ids}})

        # Technique hard filter (tier 3 only)
        if technique is not None:
            conditions.append({"technique": technique})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_results(self, raw: dict) -> list[dict]:
        """
        Convert raw ChromaDB query output into a list of clean Hit dicts.
        Decodes step_ids from JSON string back to Python list.
        Converts cosine distance to similarity score (1 - distance).
        """
        hits = []

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
            hits.append({
                "paper_id":      meta.get("paper_id", ""),
                "entry_id":      meta.get("entry_id", ""),
                "score":         round(1.0 - dist, 4),
                "document":      doc,
                "step_ids":      json.loads(meta.get("step_ids", "[]")),
                "step_category": meta.get("step_category", ""),
                "technique":     meta.get("technique", ""),
                "tier":          meta.get("tier", -1),
            })

        # Sort by score descending (ChromaDB returns by distance ascending,
        # which is equivalent, but explicit sort makes the output predictable)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits


# ── Convenience print helper ──────────────────────────────────────────────────

def print_result(result: dict, truncate: int = 120) -> None:
    """Pretty-print a RetrievalResult for debugging and inspection."""
    q = result["query"]
    print(f"\n{'═' * 70}")
    print(f"  Query: {q}")
    print(f"{'═' * 70}\n")

    p1 = result["phase1"]
    print(f"── Phase 1 — Coarse {'─' * 48}")
    print(f"  Shortlisted papers : {p1['shortlisted_paper_ids']}\n")

    print(f"  Tier 0 hits ({len(p1['tier0_hits'])}):")
    for h in p1["tier0_hits"]:
        print(f"    [{h['score']:.3f}] {h['paper_id']}")
        print(f"           {h['document'][:truncate]}{'...' if len(h['document']) > truncate else ''}")

    print(f"\n  Tier 1 hits ({len(p1['tier1_hits'])}):")
    for h in p1["tier1_hits"]:
        print(f"    [{h['score']:.3f}] {h['paper_id']} / {h['step_category']}")
        print(f"           {h['document'][:truncate]}{'...' if len(h['document']) > truncate else ''}")

    p2 = result["phase2"]
    print(f"\n── Phase 2 — Fine {'─' * 50}")

    print(f"\n  Tier 2 hits — sub-workflows ({len(p2['tier2_hits'])}):")
    for h in p2["tier2_hits"]:
        print(f"    [{h['score']:.3f}] {h['paper_id']}  steps: {h['step_ids']}")
        print(f"           {h['document'][:truncate]}{'...' if len(h['document']) > truncate else ''}")

    print(f"\n  Tier 3 hits — individual steps ({len(p2['tier3_hits'])}):")
    for h in p2["tier3_hits"]:
        technique = f"  technique: {h['technique']}" if h['technique'] else ""
        print(f"    [{h['score']:.3f}] {h['paper_id']} / {h['step_ids']}{technique}")
        print(f"           {h['document'][:truncate]}{'...' if len(h['document']) > truncate else ''}")

    print()


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the retrieval module.")
    parser.add_argument("--query",       required=True, help="User query string")
    parser.add_argument("--chroma-dir",  default=CHROMA_DB_PATH)
    parser.add_argument("--collection",  default=COLLECTION_NAME)
    parser.add_argument("--n-tier0",     type=int, default=5)
    parser.add_argument("--n-tier1",     type=int, default=5)
    parser.add_argument("--n-tier2",     type=int, default=5)
    parser.add_argument("--n-tier3",     type=int, default=15)
    parser.add_argument("--filter-phase1",  action="store_true",
                        help="Restrict phase 2 to papers shortlisted in phase 1")
    parser.add_argument("--technique",   default=None,
                        help="Hard-filter tier 3 by technique (e.g. fluxVariability)")
    args = parser.parse_args()

    retriever = GSMRetriever(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
    )

    result = retriever.retrieve(
        query=args.query,
        n_tier0=args.n_tier0,
        n_tier1=args.n_tier1,
        n_tier2=args.n_tier2,
        n_tier3=args.n_tier3,
        filter_by_phase1=args.filter_phase1,
        technique_filter=args.technique,
    )

    print_result(result)
