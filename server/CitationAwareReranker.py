import re
import os
from typing import Optional, Sequence
import operator

from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker


class CitationAwareReranker(ScoredCrossEncoderReranker):
    """Document reranker that considers both semantic similarity and citation relevance."""
    citation_weight: float = 0.3
    citation_patterns: list[str] = [
        # Standard formats
        r'Article\s*(\d+)(?:\s*\((\d+)\))?',       # Article 6(1)
        r'Art\.\s*(\d+)(?:\s*\((\d+)\))?',         # Art. 6(1)
        r'art\.\s*(\d+)(?:\s*\((\d+)\))?',         # art. 6(1)
        r'§\s*(\d+)(?:\s*\((\d+)\))?',             # § 6(1)
        
        # Asterisk/special character formats
        r'\*Art\.\s*(\d+)(?:\s*\((\d+)\))?\*',     # *Art. 6(1)*
        r'Art\.\s*(\d+)(?:\s*\((\d+)\))?',         # Art. 6(1)
        r'\*Article\s*(\d+)(?:\s*\((\d+)\))?\*',   # *Article 6(1)*
        
        # Different bracket styles
        r'Art(?:icle)?\s*(\d+)(?:\s*\[(\d+)\])',   # Article 6[1]
        r'Art(?:icle)?\s*(\d+)(?:\s*\{(\d+)\})',   # Article 6{1}
        
        # No space variations
        r'Art(?:icle)?(\d+)\((\d+)\)',             # Article6(1)
        r'Art(?:icle)?(\d+)\.(\d+)',               # Article6.1
        
        # Period notation
        r'Art(?:icle)?\s*(\d+)\.(\d+)',            # Article 6.1
        
        # Different letter cases
        r'ARTICLE\s*(\d+)(?:\s*\((\d+)\))?',       # ARTICLE 6(1)
        r'ART\.\s*(\d+)(?:\s*\((\d+)\))?',         # ART. 6(1)
        
        # Slash notation
        r'Art(?:icle)?\s*(\d+)/(\d+)',             # Article 6/1
        
        # With sub-letters
        r'Art(?:icle)?\s*(\d+)(?:\s*\((\d+[a-z])\))?',  # Article 6(1a)
        
        # With roman numerals in subsection
        r'Art(?:icle)?\s*(\d+)(?:\s*\(([ivxIVX]+)\))?'  # Article 6(iv)
    ]

    
    class Config:
        arbitrary_types_allowed = True
        
    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        self.citation_weight = float(os.getenv('citation_weight', '0.3'))
        
        # Enhanced citation patterns
        

    def _extract_citations(self, text: str) -> list[str]:
        """
        Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of normalized citation strings
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the main article number and subsection
                article_num = match.group(1)
                subsection = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                # Create normalized citation format
                citation = f"article_{article_num}"
                if subsection:
                    citation += f"_{subsection}"
                    
                citations.append(citation.lower())
        
        return list(set(citations))  # Remove duplicates

    def _calculate_citation_score(self, query_citations: list[str], doc_citations: list[str]) -> float:
        """
        Calculate citation relevance score.
        
        Args:
            query_citations: List of citations from query
            doc_citations: List of citations from document
            
        Returns:
            Float score between 0 and 1
        """
        if not query_citations:
            return 1.0  # No citations in query, neutral score
            
        # Count matching citations
        matches = sum(1 for qc in query_citations for dc in doc_citations 
                     if qc.lower() == dc.lower())
                     
        # Normalize score based on query citations
        return matches / len(query_citations) if query_citations else 0.0

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using both semantic similarity and citation awareness.

        Args:
            documents: A sequence of documents to compress
            query: The query to use for compression
            callbacks: Callbacks for the compression process

        Returns:
            Reranked sequence of documents
        """
        # Get semantic scores using parent class's model
        semantic_scores = self.model.score([(query, doc.page_content) for doc in documents])
        
        # Extract citations from query and documents
        query_citations = self._extract_citations(query)
        doc_citations_list = [self._extract_citations(doc.page_content) for doc in documents]
        
        # Calculate citation scores
        citation_scores = [
            self._calculate_citation_score(query_citations, doc_citations)
            for doc_citations in doc_citations_list
        ]
        
        # Combine scores with weighting
        combined_scores = [
            (semantic_score * (1 - self.citation_weight) + 
             citation_score * self.citation_weight)
            for semantic_score, citation_score in zip(semantic_scores, citation_scores)
        ]
        
        # Combine documents with their scores
        docs_with_scores = list(zip(documents, combined_scores, semantic_scores, citation_scores))
        
        # Sort by combined score
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        
        # Return top_n documents with metadata
        return [
            doc.copy(update={"metadata": {
                **doc.metadata,
                "relevance_score": combined_score,
                "semantic_score": semantic_score,
                "citation_score": citation_score,
                "citations_found": doc_citations_list[i]  # Add found citations to metadata
            }}) 
            for i, (doc, combined_score, semantic_score, citation_score) in enumerate(result[:self.top_n])
        ]