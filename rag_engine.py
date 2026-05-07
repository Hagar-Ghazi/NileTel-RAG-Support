"""
NileTel RAG Engine
==================
Hybrid retrieval (FAISS semantic + BM25 keyword) fused with Reciprocal Rank Fusion (RRF)
All state lives inside the NileTelRAG class
Embeddings, BM25 index and chunks are persisted to disk and reloaded on startup
"""

from __future__ import annotations
import hashlib
import json
import os
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Optional
import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



# DEFAULT CONFIG  
_DEFAULTS = {
    "embedding_model": "intfloat/multilingual-e5-large",
    "groq_model":      "llama-3.1-8b-instant",
    "top_k":           6,
    "score_threshold": 0.35,   # minimum cosine similarity kept after FAISS search
    "rrf_k":           60,     # RRF constant  (higher = less penalty for low-rank docs)
    "chunk_size":      600,    # target characters per chunk
    "chunk_overlap":   80,     # overlap between consecutive chunks (context continuity)
}





# CHUNKING  (recursive character splitter)
class RecursiveChunker:
    """
    Recursive character-level splitter

    Why recursive instead of paragraph-only?
    ─────────────────────────────────────────
    Our docs mix Arabic prose, English bullet lists and headers
    A pure paragraph split produces some very large chunks (long sections)
    and some very small ones (single bullet)
    Recursive splitting tries a priority list of separators from coarsest
    to finest:
        1. Double newline  (paragraph / section break)
        2. Single newline  (bullet list items)
        3. Arabic sentence end (. ، ؟ !)
        4. Space           (word boundary — last resort)
    It splits on the first separator that keeps chunks ≤ chunk_size
    then recurses on any piece still too large
    Overlap adds a trailing context window so retrieval doesn't miss
    answers that span chunk boundaries
    """

    SEPARATORS = ["\n\n", "\n", ".", "،", "؟", "!", " "]

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    

    def split(self, text: str) -> list[str]:
        """Return a list of non-empty chunks with overlap applied"""
        raw = self._split_recursive(text.strip(), self.SEPARATORS)
        return self._apply_overlap(raw)

    

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            cleaned = re.sub(r"[ \t]+", " ", text).strip()
            return [cleaned] if cleaned else []

        sep = separators[0]
        fallback = separators[1:] if len(separators) > 1 else [" "]
        parts = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current = ""


        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
        
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_recursive(part, fallback))
                    current = ""
                else:
                    current = part.strip()

        if current:
            chunks.append(current)

        return [re.sub(r"[ \t]+", " ", c).strip() for c in chunks if c.strip()]




    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of the previous chunk to each chunk (except the first)."""
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        merged: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.chunk_overlap:]
            merged.append((tail + " " + chunks[i]).strip())

        return merged


# PERSISTENCE MANAGER
class PersistenceManager:
    """
    Saves and loads the artefacts produced by indexing so we never
    recompute embeddings on every run

    Files written to  <cache_dir>/:
        chunks.json      — list of chunk strings + metadata dicts
        embeddings.npy   — float32 matrix  (N × D)
        bm25.pkl         — serialised BM25Okapi object
        manifest.json    — hash of the source documents → invalidate cache if changed
    """

    CHUNKS_FILE    = "chunks.json"
    EMBEDDINGS_FILE = "embeddings.npy"
    BM25_FILE      = "bm25.pkl"
    MANIFEST_FILE  = "manifest.json"

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)



    def compute_hash(self, data_dir: str) -> str:
        """MD5 of all .md file contents in sorted order."""
        h = hashlib.md5()
        for p in sorted(Path(data_dir).glob("*.md")):
            h.update(p.read_bytes())
        return h.hexdigest()



    def is_cache_valid(self, current_hash: str) -> bool:
        manifest_path = self.cache_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return False
        saved = json.loads(manifest_path.read_text())
        return saved.get("hash") == current_hash



    def save_manifest(self, current_hash: str) -> None:
        (self.cache_dir / self.MANIFEST_FILE).write_text(
            json.dumps({"hash": current_hash})
        )

    

    def save(
        self,
        chunks: list[str],
        metadata: list[dict],
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        source_hash: str,
    ) -> None:
        payload = [{"text": c, **m} for c, m in zip(chunks, metadata)]
        (self.cache_dir / self.CHUNKS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        np.save(str(self.cache_dir / self.EMBEDDINGS_FILE), embeddings)
        with open(self.cache_dir / self.BM25_FILE, "wb") as f:
            pickle.dump(bm25, f)
        self.save_manifest(source_hash)
        print(f"[Cache] Saved {len(chunks)} chunks to {self.cache_dir}/")

   

    def load(self) -> tuple[list[str], list[dict], np.ndarray, BM25Okapi]:
        payload = json.loads((self.cache_dir / self.CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks   = [p["text"] for p in payload]
        metadata = [{k: v for k, v in p.items() if k != "text"} for p in payload]
        embeddings = np.load(str(self.cache_dir / self.EMBEDDINGS_FILE))
        with open(self.cache_dir / self.BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        print(f"[Cache] Loaded {len(chunks)} chunks from {self.cache_dir}/")
        return chunks, metadata, embeddings, bm25



# ROUTER
class QueryRouter:
    """
    Multi-intent two-stage router

    Instead of returning a single label it returns a routing dict so the
    pipeline can handle combined queries like:
        "مرحبا، عندي مشكلة في النت"         →  greeting + ticket
        "ايه عروض العيد وكمان ايه الـ SLA؟" →  two chat sub-questions

    Stage 1 — fast keyword lookup per sub-part  (no LLM cost, < 1 ms)
    Stage 2 — LLM fallback for genuinely ambiguous sub-parts

    route() returns:
    {
        "primary":      str,        # dominant intent (ticket > out_of_scope > chat)
        "intents":      list[str],  # all detected intents in order
        "has_greeting": bool,
        "sub_queries":  list[str],  # non-greeting parts for retrieval
    }
    """

    # Arabic normalisation helpers 
    _ALEF_MAP = str.maketrans({
        "\u0623": "\u0627", "\u0625": "\u0627",
        "\u0622": "\u0627", "\u0671": "\u0627",
    })

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower().strip()
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Mn" and ch != "\u0640"
        )
        text = text.translate(QueryRouter._ALEF_MAP)
        return re.sub(r"\s+", " ", text)



    # Keyword lists 
    _GREETINGS: list[str] = [
        "ازيك", "ازيكم", "مرحبا", "اهلا", "اهلا وسهلا",
        "صباح الخير", "مساء الخير", "السلام عليكم", "عليكم السلام",
        "هاي", "هلو", "ايه الاخبار", "عامل ايه", "كيف الحال", "كيف حالك",
        "hi", "hello", "hey", "good morning", "good evening",
        "how are you", "what's up", "sup",
    ]

    _TICKET_TRIGGERS: list[str] = [
        # Arabic explicit requests
        "افتح تذكرة", "اعمل تذكرة", "ارفع تذكرة", "فتح شكوى",
        "ابعت مهندس", "ابعتلي مهندس", "محتاج مهندس", "عايز مهندس",
        "escalate", "اعمل escalation",
        # Arabic problem indicators 
        "مقطوع تماما", "مش شغال خالص", "انقطع النت", "انقطع الخط",
        "ضوء احمر", "ضوء أحمر", "ont ضوء", "روتر مش شغال",
        "فاتورة غلط", "رسوم غلط", "بيشحن غلط",
        # English
        "open ticket", "create ticket", "raise ticket", "dispatch engineer",
        "send technician", "total outage", "complete outage",
        "no internet", "no connection", "disconnected",
    ]

    _OUT_OF_SCOPE: list[str] = [
        # Entertainment
        "فيلم", "مسلسل", "انمي", "كرتون", "يوتيوب", "نتفليكس",
        "game", "جيم", "لعبة",
        # Food
        "اكل", "مطعم", "طبخ", "وجبة", "كافيه", "قهوة",
        # Sports
        "رياضة", "كورة", "كرة القدم", "مباراة", "دوري",
        "football", "sport", "match",
        # Politics / news
        "سياسة", "انتخابات", "برلمان", "اخبار", "جريدة",
        "politics", "election",
        # Weather / travel
        "طقس", "درجة الحرارة", "سفر", "فندق", "طيارة",
        "weather", "hotel", "flight",
        # Finance (non-telecom)
        "بورصة", "ذهب", "دولار", "بنك",
        "stock", "gold", "currency",
        # Medical
        "دكتور", "مستشفى", "علاج", "دواء",
        "doctor", "hospital", "medicine",
    ]



    # Sentence splitters (split on these to detect multi-intent)
    # ، kept  → splits "مرحبا، انقطع النت" into greeting + ticket correctly
    # .  removed → avoid breaking mid-sentence dots
    # و  removed → was incorrectly splitting tokens like "الـ SLA"
    _SPLIT_PATTERN = re.compile(
        r"[،؟!\n]"
        r"|كمان"
        r"|وكمان"
        r"|\band\b|\balso\b",
        re.IGNORECASE,
    )

    def __init__(self, groq_client: Groq, groq_model: str):
        self._groq  = groq_client
        self._model = groq_model

    

    def route(self, query: str) -> dict:
        """
        Returns a routing decision dict:
        {
            "primary":      str,        # dominant intent
            "intents":      list[str],  # all detected intents in order
            "has_greeting": bool,
            "sub_queries":  list[str],  # split parts if multi-intent
        }
        """
        norm = self._normalise(query)

        # Step 1: Check if there's a greeting anywhere in the query
        has_greeting = self._matches_any(norm, self._GREETINGS)

        # Step 2: Split query into sub-parts and classify each
        parts = self._split_into_parts(query)
        intents: list[str] = []

        for part in parts:
            part_norm = self._normalise(part)
            if not part_norm.strip():
                continue
            intent = self._classify_part(part_norm, part)
            intents.append(intent)

        # Remove duplicate consecutive intents
        intents = self._deduplicate(intents)

        # If only greeting found and nothing else it's pure greeting
        if not intents or all(i == "greeting" for i in intents):
            return {
                "primary":      "greeting",
                "intents":      ["greeting"],
                "has_greeting": True,
                "sub_queries":  [],
            }

        # Filter out greetings from the substantive intents list
        substantive = [i for i in intents if i != "greeting"]
        primary = self._pick_primary(substantive)

        return {
            "primary":      primary,
            "intents":      intents,
            "has_greeting": has_greeting,
            "sub_queries":  [
                p for p in parts
                if len(self._normalise(p)) > 5
                and not self._matches_any(self._normalise(p), self._GREETINGS)
            ],
        }

 


    def _split_into_parts(self, query: str) -> list[str]:
        """Split a query into sub-parts at sentence boundaries."""
        parts = self._SPLIT_PATTERN.split(query)
        return [p.strip() for p in parts if p and p.strip()]




    def _classify_part(self, norm_part: str, raw_part: str) -> str:
        """Classify a single sub-part using keyword matching then LLM fallback."""
        # Check greetings FIRST (before length guard) so short words like
        # "ازيك" (4 chars) and "هاي" are still caught correctly
        if self._matches_any(norm_part, self._GREETINGS):
            return "greeting"
        if self._matches_any(norm_part, self._TICKET_TRIGGERS):
            return "ticket"
        if self._matches_any(norm_part, self._OUT_OF_SCOPE):
            return "out_of_scope"
        # Skip short fragments before LLM to avoid misclassification of
        # leftover tokens like "ايه ال" produced by splitting
        if len(norm_part.strip()) < 8:
            return "chat"
        return self._llm_classify(raw_part)




    def _llm_classify(self, text: str) -> str:
        _SYSTEM = (
            "أنت نظام تصنيف لشركة اتصالات NileTel.\n"
            "صنّف النص التالي إلى واحدة فقط من هذه الفئات:\n"
            "  greeting    = تحية أو كلام اجتماعي\n"
            "  ticket      = مشكلة تقنية تحتاج مهندس أو تذكرة دعم (انقطاع، عطل، شكوى)\n"
            "  out_of_scope = سؤال خارج نطاق الاتصالات (طعام، رياضة، سياسة، طقس)\n"
            "  chat        = سؤال مشروع عن خدمات أو باقات أو معلومات الشركة\n"
            "أجب بكلمة واحدة فقط بدون أي شرح."
        )
        try:
            resp = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": text},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            label = resp.choices[0].message.content.strip().lower()
            if label in {"greeting", "ticket", "out_of_scope", "chat"}:
                return label
        except Exception as e:
            print(f"[Router] LLM error: {e}")
        return "chat"   # safe default



    @staticmethod
    def _matches_any(norm_text: str, keywords: list[str]) -> bool:
        return any(kw in norm_text for kw in keywords)



    @staticmethod
    def _deduplicate(lst: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result



    @staticmethod
    def _pick_primary(intents: list[str]) -> str:
        """Priority order: ticket > out_of_scope > chat."""
        priority = {"ticket": 0, "out_of_scope": 1, "chat": 2}
        return min(intents, key=lambda i: priority.get(i, 99)) if intents else "chat"




# HYBRID RETRIEVER  (FAISS + BM25 + RRF)
class HybridRetriever:
    """
    Two independent retrieval signals fused with Reciprocal Rank Fusion

    FAISS (semantic):
        – Encodes query + chunks with multilingual-e5-large
        – IndexFlatIP on L2-normalised vectors → cosine similarity
        – Catches paraphrase matches and cross-language questions

    BM25 (keyword / exact):
        – Classic TF-IDF variant, language-agnostic tokenisation
        – Catches exact product names, error codes, account IDs
        – Operates on raw token overlap no embeddings needed

    RRF fusion:
        – Gives each doc a fused score = Σ 1/(k + rank_in_list)
        – k=60 standard larger k softens the advantage of top positions
        – Combines lists without needing score normalisation
        – Consistently outperforms either signal alone on mixed queries
    """

    def __init__(
        self,
        model: SentenceTransformer,
        index: faiss.IndexFlatIP,
        bm25: BM25Okapi,
        chunks: list[str],
        metadata: list[dict],
        top_k: int = 6,
        score_threshold: float = 0.35,
        rrf_k: int = 60,
    ):
        self.model           = model
        self.index           = index
        self.bm25            = bm25
        self.chunks          = chunks
        self.metadata        = metadata
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self.rrf_k           = rrf_k

    

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-k results as dicts:
            { "text": str, "source": str, "score": float, "method": str }
        """
        semantic_ranking = self._semantic_search(query)
        keyword_ranking  = self._bm25_search(query)
        fused            = self._rrf(semantic_ranking, keyword_ranking)
        print(f"[Retriever] Fused results: {len(fused)} docs returned.")
        return fused

   

    def _semantic_search(self, query: str) -> list[int]:
        """Return chunk indices ranked by FAISS cosine similarity."""
        q_emb = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_emb, self.top_k * 2)
        ranked: list[int] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) >= self.score_threshold:
                ranked.append(int(idx))
        return ranked




    def _bm25_search(self, query: str) -> list[int]:
        """Return chunk indices ranked by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Rank all docs; take top_k * 2 as candidates
        ranked_indices = np.argsort(scores)[::-1][: self.top_k * 2]
        # Filter out zero-score docs
        return [int(i) for i in ranked_indices if scores[i] > 0]



    def _rrf(
        self,
        list_a: list[int],
        list_b: list[int],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        score(doc) = 1/(k + rank_in_A)  +  1/(k + rank_in_B)
        """
        rrf_scores: dict[int, float] = {}

        for rank, idx in enumerate(list_a, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, idx in enumerate(list_b, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        results: list[dict] = []
        for idx, score in top:
            results.append({
                "text":   self.chunks[idx],
                "source": self.metadata[idx].get("source", "unknown"),
                "score":  round(score, 4),
                "method": "hybrid-rrf",
            })
        return results



# GENERATOR

_SYSTEM_PROMPT = """
أنت مساعد دعم عملاء احترافي في شركة NileTel للاتصالات.

قواعد صارمة:
1. أجب فقط من المعلومات الموجودة في السياق المقدم — لا تخترع إجابات.
2. إذا لم تجد الإجابة في السياق، قل "يا فندم، مش عندي معلومات كافية عن ده حالياً."
3. استخدم اللغة العربية الواضحة مع مصطلحات تقنية إنجليزية عند الضرورة.
4. كن مهذباً ومحترفاً في كل ردودك.
5. في نهاية إجابتك، ضع سطراً واحداً يبدأ بـ "NEEDS_ACTION:" يكون قيمته "YES" أو "NO".
   YES  = الحالة تحتاج تذكرة دعم فني أو إجراء خارجي
   NO   = الإجابة اكتملت بالكلام فقط

مثال على صياغة الرد:
---
يا فندم، بخصوص سؤالك عن...
[الإجابة هنا]

NEEDS_ACTION: NO
---
""".strip()


class Generator:
    """Builds the prompt, calls Groq, parses the structured output."""

    def __init__(self, groq_client: Groq, groq_model: str):
        self._groq  = groq_client
        self._model = groq_model

    

    def generate(self, query: str, retrieved: list[dict]) -> dict:
        """
        Returns:
            { "answer": str, "needs_action": "YES"|"NO", "sources": list[str] }
        """
        if not retrieved:
            return {
                "answer":       "يا فندم، مش لاقي معلومات كافية عشان أرد على سؤالك. ممكن تعيد الصياغة؟",
                "needs_action": "NO",
                "sources":      [],
            }

        context_block = self._build_context(retrieved)
        user_msg = (
            f"السياق المتاح (استخدمه فقط):\n{context_block}\n\n"
            f"سؤال العميل: {query}"
        )

        try:
            resp = self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Generator] LLM error: {e}")
            raw = "يا فندم، حصلت مشكلة تقنية. حاول تاني.\nNEEDS_ACTION: NO"

        answer, needs_action = self._parse(raw)
        return {
            "answer":       answer,
            "needs_action": needs_action,
            "sources":      list({r["source"] for r in retrieved}),
        }

 

    @staticmethod
    def _build_context(retrieved: list[dict]) -> str:
        parts = [
            f"[{i}] ({r['source']} | rrf={r['score']:.4f})\n{r['text']}"
            for i, r in enumerate(retrieved, 1)
        ]
        return "\n\n---\n\n".join(parts)




    @staticmethod
    def _parse(raw: str) -> tuple[str, str]:
        """Extract answer text and NEEDS_ACTION flag from LLM output."""
        needs_action = "NO"
        match = re.search(r"NEEDS_ACTION\s*:\s*(YES|NO)", raw, re.IGNORECASE)
        if match:
            needs_action = match.group(1).upper()

        # Remove the NEEDS_ACTION line from the displayed answer
        answer = re.sub(
            r"\n?NEEDS_ACTION\s*:\s*(YES|NO)\s*", "", raw, flags=re.IGNORECASE
        ).strip()

        return answer, needs_action



# NileTelRAG  (orchestrator class)
class NileTelRAG:
    """
    Main entry point.  Call  rag = NileTelRAG(data_dir="data")  to build
    or reload the index, then  result = rag.query("...")  to run a query

    Internal pipeline:
        1. Router   → decide how to handle the query
        2. Retriever → hybrid FAISS + BM25 + RRF  (only for 'chat' route)
        3. Generator → Groq LLM with structured prompt  (only for 'chat' route)
    """


    def __init__(
        self,
        data_dir:        str  = "data",
        cache_dir:       str  = "cache",
        embedding_model: str  = _DEFAULTS["embedding_model"],
        groq_model:      str  = _DEFAULTS["groq_model"],
        top_k:           int  = _DEFAULTS["top_k"],
        score_threshold: float = _DEFAULTS["score_threshold"],
        rrf_k:           int  = _DEFAULTS["rrf_k"],
        chunk_size:      int  = _DEFAULTS["chunk_size"],
        chunk_overlap:   int  = _DEFAULTS["chunk_overlap"],
    ):
        load_dotenv()
        self.data_dir  = data_dir
        self.groq_model = groq_model

        print("\n" + "=" * 60)
        print("NileTel RAG — initialising")
        print("=" * 60)


        # 1. Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY missing from .env file")
        self._groq = Groq(api_key=api_key)



        # 2. Embedding model
        print(f"[Init] Loading embedding model: {embedding_model}")
        self._embed_model = SentenceTransformer(embedding_model)



        # 3. Build or load index
        persistence = PersistenceManager(cache_dir)
        source_hash = persistence.compute_hash(data_dir)

        if persistence.is_cache_valid(source_hash):
            chunks, metadata, embeddings, bm25 = persistence.load()
        else:
            print("[Init] Cache miss so rebuilding index from documents")
            chunks, metadata = self._load_and_chunk(data_dir, chunk_size, chunk_overlap)
            embeddings       = self._encode(chunks)
            bm25             = self._build_bm25(chunks)
            persistence.save(chunks, metadata, embeddings, bm25, source_hash)

        faiss_index = self._build_faiss(embeddings)



        # 4. Compose sub-systems
        self._router = QueryRouter(self._groq, groq_model)

        self._retriever = HybridRetriever(
            model           = self._embed_model,
            index           = faiss_index,
            bm25            = bm25,
            chunks          = chunks,
            metadata        = metadata,
            top_k           = top_k,
            score_threshold = score_threshold,
            rrf_k           = rrf_k,
        )

        self._generator = Generator(self._groq, groq_model)
        print("[Init] System ready\n")



    # main query method 

    def query(self, user_query: str) -> dict:
        """
        Run the full pipeline for a user query

        Returns:
        {
            "route":        "greeting" | "ticket" | "out_of_scope" | "chat",
            "answer":       str,
            "needs_action": "YES" | "NO",
            "sources":      list[str],
        }
        """
        print(f"\n[Pipeline] Query: {user_query}")



        # Step 1 — Route (now returns a dict with multi-intent support)
        routing = self._router.route(user_query)
        primary      = routing["primary"]
        has_greeting = routing["has_greeting"]
        sub_queries  = routing["sub_queries"]
        print(f"[Pipeline] Routing: primary={primary}, intents={routing['intents']}, "
              f"has_greeting={has_greeting}, sub_queries={sub_queries}")

        # Build greeting prefix to prepend when greeting + another intent
        greeting_prefix = (
            "أهلاً بيك يا فندم! 😊\n\n"
            if has_greeting and primary != "greeting"
            else ""
        )



        # Step 2 — Short-circuit responses (no retrieval needed)
        # Pure greeting — no actionable question detected
        if primary == "greeting":
            return self._respond(
                "greeting",
                "أهلاً بيك يا فندم! 😊 أنا مساعد NileTel. إزاي أقدر أساعدك النهارده؟",
                "NO", [],
            )

        if primary == "out_of_scope":
            return self._respond(
                "out_of_scope",
                greeting_prefix + "آسف يا فندم، مش هقدر أساعدك في الموضوع ده. أنا متخصص في خدمات NileTel فقط.",
                "NO", [],
            )

        if primary == "ticket":
            return self._respond(
                "ticket",
                greeting_prefix + "تمام يا فندم، هبدأ في رفع التذكرة فوراً. فريق الدعم الفني هيتواصل معاك في أقرب وقت.",
                "YES", [],
            )




        # Step 3 — chat: handle one or multiple sub-questions
        if len(sub_queries) > 1:
            # Multi-question: retrieve and answer each separately, then combine
            combined_answer = greeting_prefix
            all_sources: list[str] = []

            for i, sub_q in enumerate(sub_queries, 1):
                retrieved = self._retriever.retrieve(sub_q)
                result    = self._generator.generate(sub_q, retrieved)
                short_q   = sub_q[:40] + ("..." if len(sub_q) > 40 else "")
                combined_answer += f"**{i}. بخصوص سؤالك عن '{short_q}':**\n{result['answer']}\n\n"
                all_sources.extend(result["sources"])

            return self._respond("chat", combined_answer.strip(), "NO", list(set(all_sources)))

        else:
            # Single question (possibly with greeting prefix)
            q = sub_queries[0] if sub_queries else user_query
            retrieved = self._retriever.retrieve(q)
            result    = self._generator.generate(q, retrieved)
            return self._respond(
                "chat",
                greeting_prefix + result["answer"],
                result["needs_action"],
                result["sources"],
            )

   

    @staticmethod
    def _respond(route: str, answer: str, needs_action: str, sources: list[str]) -> dict:
        return {
            "route":        route,
            "answer":       answer,
            "needs_action": needs_action,
            "sources":      sources,
        }

    @staticmethod
    def _load_and_chunk(
        data_dir: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> tuple[list[str], list[dict]]:
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks: list[str]  = []
        all_meta:   list[dict] = []

        for fpath in sorted(Path(data_dir).glob("*.md")):
            text = fpath.read_text(encoding="utf-8")
            doc_chunks = chunker.split(text)
            print(f"  {fpath.name}: {len(doc_chunks)} chunks")
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                all_meta.append({"source": fpath.name})

        print(f"[Chunking] Total: {len(all_chunks)} chunks")
        return all_chunks, all_meta

    def _encode(self, chunks: list[str]) -> np.ndarray:
        """Encode all chunks with e5 passage prefix."""
        print(f"[Embed] Encoding {len(chunks)} chunks...")
        prefixed = [f"passage: {c}" for c in chunks]
        embeddings = self._embed_model.encode(
            prefixed,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        print(f"[Embed] Shape: {embeddings.shape}")
        return embeddings

    @staticmethod
    def _build_bm25(chunks: list[str]) -> BM25Okapi:
        tokenised = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenised)
        print(f"[BM25] Index built on {len(chunks)} docs.")
        return bm25

    @staticmethod
    def _build_faiss(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"[FAISS] Index ready — {index.ntotal} vectors, dim={dim}.")
        return index
