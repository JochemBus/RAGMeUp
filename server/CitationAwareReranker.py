from _future_ import annotations

import os
import re
from typing import Optional, Sequence

import operator

from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker

class CitationAwareReranker(ScoredCrossEncoderReranker):
    """Document reranker that considers both semantic similarity and citation relevance."""
    
    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        self.citation_weight = float(os.getenv('citation_weight', '0.3'))
        self.context_window_size = int(os.getenv('context_window_size', '50'))

        # Enhanced citation patterns
        self.citation_patterns = _build_citation_patterns()

    def _extract_citations(self, text: str) -> list[str]:
        """Extract citations from text."""
        return _extract_citations(text, self.citation_patterns)

    def _get_citation_context(self, text: str, citation: str) -> str:
        """Extract the context around a citation."""
        citation_start = text.index(citation)
        start = max(0, citation_start - self.context_window_size)
        end = min(len(text), citation_start + len(citation) + self.context_window_size)
        return text[start:end]

    def _calculate_citation_relevance(self, query: str, citation_contexts: list[str]) -> list[float]:
        """Assess the relevance of citation contexts to the query."""
        relevance_scores = []
        for context in citation_contexts:
            # Implement a simple heuristic-based approach to assess citation relevance
            # e.g., check for keyword matches, content similarity, etc.
            relevance = self._simple_citation_relevance_check(query, context)
            relevance_scores.append(relevance)
        return relevance_scores

    def _simple_citation_relevance_check(self, query: str, context: str) -> float:
        """
        A simple heuristic-based approach to assess citation relevance.
        Can be replaced with a more advanced approach if needed.
        """
        # Example: Check for keyword matches between query and context
        query_keywords = set(query.lower().split())
        context_keywords = set(context.lower().split())
        overlap = query_keywords.intersection(context_keywords)
        return len(overlap) / max(len(query_keywords), len(context_keywords))

    def _calculate_citation_score(self, query_citations: list[str], doc_citations: list[str], citation_relevance_scores: list[float]) -> float:
        """Calculate citation relevance score."""
        if not query_citations:
            return 1.0  # No citations in query, neutral score

        # Count matching citations and weight by relevance
        weighted_matches = sum(
            relevance
            for qc in query_citations
            for dc, relevance in zip(doc_citations, citation_relevance_scores)
            if qc.lower() == dc.lower()
        )

        # Normalize score based on query citations
        return weighted_matches / len(query_citations) if query_citations else 0.0

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Rerank documents using both semantic similarity and citation relevance."""
        # Get semantic scores using parent class's model
        semantic_scores = self.model.score([(query, doc.page_content) for doc in documents])

        # Extract citations from query and documents
        query_citations = self._extract_citations(query)
        doc_citations_list = [self._extract_citations(doc.page_content) for doc in documents]

        # Get citation contexts and relevance scores
        citation_contexts_list = [
            [self._get_citation_context(doc.page_content, citation) for citation in doc_citations]
            for doc_citations in doc_citations_list
        ]
        citation_relevance_scores_list = [
            self._calculate_citation_relevance(query, citation_contexts)
            for citation_contexts in citation_contexts_list
        ]

        # Calculate citation scores
        citation_scores = [
            self._calculate_citation_score(query_citations, doc_citations, citation_relevance_scores)
            for doc_citations, citation_relevance_scores in zip(doc_citations_list, citation_relevance_scores_list)
        ]

        # Combine scores with weighting
        combined_scores = [
            (semantic_score * (1 - self.citation_weight) +
             citation_score * self.citation_weight)
            for semantic_score, citation_score in zip(semantic_scores, citation_scores)
        ]

        # Combine documents with their scores
        docs_with_scores = list(zip(documents, combined_scores, semantic_scores, citation_scores, citation_relevance_scores_list))

        # Sort by combined score
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        # Return top_n documents with metadata
        return [
            doc.copy(update={"metadata": {
                **doc.metadata,
                "relevance_score": combined_score,
                "semantic_score": semantic_score,
                "citation_score": citation_score,
                "citation_relevance_scores": citation_relevance_scores,
                "citations_found": doc_citations_list[i]
            }})
            for i, (doc, combined_score, semantic_score, citation_score, citation_relevance_scores) in enumerate(result[:self.top_n])
        ]

def _build_citation_patterns() -> list[re.Pattern]:
    """Construct the list of enhanced citation patterns."""
    return [
        # Standard formats
        r'Article\s*(\d+)(?:\s*\((\d+)\))?',    # Article 6(1)
        r'Art\.\s*(\d+)(?:\s*\((\d+)\))?',      # Art. 6(1)
        r'art\.\s*(\d+)(?:\s*\((\d+)\))?',      # art. 6(1)
        r'§\s*(\d+)(?:\s*\((\d+)\))?',          # § 6(1)
        
        # Asterisk/special character formats
        r'\\*Art\.\s(\d+)(?:\s*\((\d+)\))?[\]',   # *Art. 6(1)*
        r'\Art\.\s(\d+)(?:\s*\((\d+)\))?[\]',     # Art. 6(1)
        r'\\*Article\s(\d+)(?:\s*\((\d+)\))?[\]', # *Article 6(1)*
        
        # Different bracket styles
        r'Art(?:icle)?\s*(\d+)(?:\s*\[(\d+)\])',      # Article 6[1]
        r'Art(?:icle)?\s*(\d+)(?:\s*\{(\d+)\})',      # Article 6{1}
        
        # No space variations
        r'Art(?:icle)?(\d+)\((\d+)\)',                # Article6(1)
        r'Art(?:icle)?(\d+)\.(\d+)',                  # Article6.1
        
        # Period notation
        r'Art(?:icle)?\s*(\d+)\.(\d+)',              # Article 6.1
        
        # Different letter cases
        r'ARTICLE\s*(\d+)(?:\s*\((\d+)\))?',         # ARTICLE 6(1)
        r'ART\.\s*(\d+)(?:\s*\((\d+)\))?',           # ART. 6(1)
        
        # Slash notation
        r'Art(?:icle)?\s*(\d+)/(\d+)',               # Article 6/1
        
        # With sub-letters
        r'Art(?:icle)?\s*(\d+)(?:\s*\((\d+[a-z])\))?',  # Article 6(1a)
        
        # With roman numerals in subsection
        r'Art(?:icle)?\s*(\d+)(?:\s*\(([ivxIVX]+)\))?', # Article 6(iv)
    ]

def _extract_citations(text: str, citation_patterns: list[re.Pattern]) -> list[str]:
    """Extract citations from text using the given patterns."""
    citations = []
    for pattern in citation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            article_num = match.group(1)
            subsection = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
            citation = f"article_{article_num}"
            if subsection:
                citation += f"_{subsection}"
            citations.append(citation.lower())
    return list(set(citations))