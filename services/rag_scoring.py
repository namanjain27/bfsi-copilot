import os
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .config_loader import get_config
from .agent_schemas import KBDocument
from .logger_setup import setup_logger
logger = setup_logger()

class RAGScoringService:
    """
    Enhanced document scoring service for RAG retrieval combining multiple scoring algorithms:
    - Semantic similarity (vector search scores)
    - Keyword matching (BM25/TF-IDF)
    - Document quality (metadata-based)
    - Recency scoring
    """

    def __init__(self,
                 semantic_weight: Optional[float] = None,
                 keyword_weight: Optional[float] = None,
                 quality_weight: Optional[float] = None,
                 recency_weight: Optional[float] = None):
        """
        Initialize RAG scoring service with configurable weights

        Args:
            semantic_weight: Weight for semantic similarity scores (None to use config)
            keyword_weight: Weight for keyword matching scores (None to use config)
            quality_weight: Weight for document quality scores (None to use config)
            recency_weight: Weight for recency scores (None to use config)
        """
        # Load config and use provided weights or defaults from config
        config = get_config()
        scoring_config = config.get_section('rag_scoring')
        weights = scoring_config.get('weights', {})

        self.semantic_weight = semantic_weight if semantic_weight is not None else weights.get('semantic', 0.4)
        self.keyword_weight = keyword_weight if keyword_weight is not None else weights.get('keyword', 0.3)
        self.quality_weight = quality_weight if quality_weight is not None else weights.get('quality', 0.2)
        self.recency_weight = recency_weight if recency_weight is not None else weights.get('recency', 0.1)

        # Validate weights sum to 1.0
        total_weight = sum([self.semantic_weight, self.keyword_weight, self.quality_weight, self.recency_weight])
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total_weight}, not 1.0. Normalizing weights.")
            self.semantic_weight /= total_weight
            self.keyword_weight /= total_weight
            self.quality_weight /= total_weight
            self.recency_weight /= total_weight

        # Initialize TF-IDF vectorizer for keyword scoring using config values
        tfidf_config = scoring_config.get('tfidf', {})
        ngram_range = tfidf_config.get('ngram_range', [1, 2])
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=tfidf_config.get('stop_words', 'english'),
            max_features=tfidf_config.get('max_features', 1000),
            ngram_range=tuple(ngram_range),  # Convert list to tuple for TfidfVectorizer
            lowercase=True
        )
        self.tfidf_fitted = False

    def compute_semantic_scores(self, documents: List[Document], similarity_scores: List[float]) -> List[float]:
        """
        Normalize and return semantic similarity scores from vector search

        Args:
            documents: List of retrieved documents
            similarity_scores: Raw similarity scores from vector search

        Returns:
            List of normalized semantic scores (0-1 range)
        """
        if not similarity_scores:
            return [0.0] * len(documents)

        # Normalize scores to 0-1 range
        min_score = min(similarity_scores)
        max_score = max(similarity_scores)

        if max_score == min_score:
            return [1.0] * len(documents)

        normalized_scores = [(score - min_score) / (max_score - min_score) for score in similarity_scores]
        logger.debug(f"Semantic scores normalized: min={min_score:.3f}, max={max_score:.3f}")

        return normalized_scores

    def compute_keyword_scores(self, query: str, documents: List[Document]) -> List[float]:
        """
        Compute keyword matching scores using TF-IDF cosine similarity

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            List of keyword matching scores (0-1 range)
        """
        if not documents:
            return []

        # Extract document texts
        doc_texts = [doc.page_content for doc in documents]

        # Fit TF-IDF if not already fitted or refit for new corpus
        try:
            # Create corpus with query + documents
            corpus = [query] + doc_texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)

            # Query vector is first row, document vectors are remaining rows
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]

            # Compute cosine similarity between query and each document
            similarities = cosine_similarity(query_vector, doc_vectors)[0]

            # Normalize to 0-1 range
            if len(similarities) > 0:
                max_sim = max(similarities) if max(similarities) > 0 else 1.0
                normalized_scores = [sim / max_sim for sim in similarities]
            else:
                normalized_scores = [0.0] * len(documents)

            logger.debug(f"Keyword scores computed for {len(documents)} documents")
            return normalized_scores

        except Exception as e:
            logger.error(f"Error computing keyword scores: {e}")
            return [0.0] * len(documents)

    def compute_quality_scores(self, documents: List[Document]) -> List[float]:
        """
        Compute document quality scores based on metadata

        Args:
            documents: List of retrieved documents

        Returns:
            List of quality scores (0-1 range)
        """
        quality_scores = []

        for doc in documents:
            metadata = doc.metadata
            score = 0.0

            # Document type quality (formatted documents score higher)
            doc_type = metadata.get('document_type', 'unknown')
            if doc_type == 'formatted_document':
                score += 0.3
            elif doc_type == 'structured_text':
                score += 0.2
            elif doc_type == 'plain_text':
                score += 0.1

            # Content density (higher word density = better quality)
            content_density = metadata.get('content_density', 0.0)
            if content_density > 0:
                # Normalize density score (typical range 0.1-0.2 for good content)
                density_score = min(content_density * 5, 0.3)
                score += density_score

            # Position-based scoring (first chunks often contain important info)
            chunk_position_ratio = metadata.get('chunk_position_ratio', 0.5)
            if chunk_position_ratio <= 0.2:  # First 20% of document
                score += 0.2
            elif chunk_position_ratio <= 0.5:  # First 50% of document
                score += 0.1

            # Document size quality (medium-sized documents often better)
            word_count = metadata.get('word_count', 0)
            if 100 <= word_count <= 1000:  # Optimal chunk size range
                score += 0.1
            elif 50 <= word_count < 100 or 1000 < word_count <= 2000:
                score += 0.05

            # First page/chunk bonus (often contains summaries/introductions)
            if metadata.get('is_first_chunk', False):
                score += 0.1
            if metadata.get('is_first_page', False):
                score += 0.1

            # Ensure score is in 0-1 range
            score = min(score, 1.0)
            quality_scores.append(score)

        logger.debug(f"Quality scores computed: avg={np.mean(quality_scores):.3f}")
        return quality_scores

    def compute_recency_scores(self, documents: List[Document]) -> List[float]:
        """
        Compute recency scores based on file modification and ingestion timestamps

        Args:
            documents: List of retrieved documents

        Returns:
            List of recency scores (0-1 range)
        """
        recency_scores = []
        current_time = datetime.now()

        for doc in documents:
            metadata = doc.metadata
            score = 0.0

            # File modification recency (primary factor)
            file_modified_str = metadata.get('file_modified_timestamp')
            if file_modified_str:
                try:
                    file_modified = datetime.fromisoformat(file_modified_str)
                    days_old = (current_time - file_modified).days

                    # Scoring based on age (exponential decay)
                    if days_old <= 7:  # Within a week
                        score += 0.5
                    elif days_old <= 30:  # Within a month
                        score += 0.3
                    elif days_old <= 90:  # Within 3 months
                        score += 0.2
                    elif days_old <= 365:  # Within a year
                        score += 0.1
                    # Older than a year gets no recency bonus

                except ValueError:
                    logger.warning(f"Invalid file_modified_timestamp: {file_modified_str}")

            # Ingestion recency (secondary factor)
            ingestion_str = metadata.get('ingestion_timestamp')
            if ingestion_str:
                try:
                    ingestion_time = datetime.fromisoformat(ingestion_str)
                    hours_since_ingestion = (current_time - ingestion_time).total_seconds() / 3600

                    # Recent ingestion bonus (content freshness)
                    if hours_since_ingestion <= 24:  # Within 24 hours
                        score += 0.2
                    elif hours_since_ingestion <= 168:  # Within a week
                        score += 0.1

                except ValueError:
                    logger.warning(f"Invalid ingestion_timestamp: {ingestion_str}")

            # Default score for documents without timestamp info
            if score == 0.0:
                score = 0.3  # Neutral score for unknown recency

            recency_scores.append(score)

        logger.debug(f"Recency scores computed: avg={np.mean(recency_scores):.3f}")
        return recency_scores

    def compute_combined_scores(self,
                              query: str,
                              documents: List[Document],
                              similarity_scores: List[float]) -> List[Tuple[Document, float]]:
        """
        Compute combined weighted scores for all documents

        Args:
            query: User query
            documents: List of retrieved documents
            similarity_scores: Raw similarity scores from vector search

        Returns:
            List of (document, combined_score) tuples sorted by score (highest first)
        """
        if not documents:
            return []

        logger.info(f"Computing combined scores for {len(documents)} documents")

        # Compute individual scores
        semantic_scores = self.compute_semantic_scores(documents, similarity_scores)
        keyword_scores = self.compute_keyword_scores(query, documents)
        quality_scores = self.compute_quality_scores(documents)
        recency_scores = self.compute_recency_scores(documents)

        # Combine scores with weights
        combined_scores = []
        for i in range(len(documents)):
            combined_score = (
                self.semantic_weight * semantic_scores[i] +
                self.keyword_weight * keyword_scores[i] +
                self.quality_weight * quality_scores[i] +
                self.recency_weight * recency_scores[i]
            )
            combined_scores.append((documents[i], combined_score))

            logger.debug(f"Doc {i}: semantic={semantic_scores[i]:.3f}, "
                        f"keyword={keyword_scores[i]:.3f}, "
                        f"quality={quality_scores[i]:.3f}, "
                        f"recency={recency_scores[i]:.3f}, "
                        f"combined={combined_score:.3f}")

        # Sort by combined score (highest first)
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Combined scoring complete. Top score: {combined_scores[0][1]:.3f}")
        return combined_scores

    def filter_by_threshold(self,
                          scored_documents: List[Tuple[Document, float]],
                          threshold: float = 0.3) -> List[Tuple[Document, float]]:
        """
        Filter documents by minimum score threshold

        Args:
            scored_documents: List of (document, score) tuples
            threshold: Minimum score threshold (0-1)

        Returns:
            Filtered list of documents above threshold
        """
        filtered = [(doc, score) for doc, score in scored_documents if score >= threshold]

        logger.info(f"Filtered {len(scored_documents)} documents to {len(filtered)} "
                   f"above threshold {threshold}")

        return filtered

    def update_weights(self,
                      semantic_weight: float,
                      keyword_weight: float,
                      quality_weight: float,
                      recency_weight: float):
        """
        Update scoring weights (useful for tuning)

        Args:
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            quality_weight: Weight for document quality
            recency_weight: Weight for recency
        """
        total = semantic_weight + keyword_weight + quality_weight + recency_weight

        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            semantic_weight /= total
            keyword_weight /= total
            quality_weight /= total
            recency_weight /= total

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.quality_weight = quality_weight
        self.recency_weight = recency_weight

        logger.info(f"Updated scoring weights: semantic={self.semantic_weight:.2f}, "
                   f"keyword={self.keyword_weight:.2f}, "
                   f"quality={self.quality_weight:.2f}, "
                   f"recency={self.recency_weight:.2f}")

# Default scoring service instance (will use config values)
default_scoring_service = RAGScoringService()

def score_documents(query: str,
                   documents: List[Document],
                   similarity_scores: List[float],
                   threshold: Optional[float] = None) -> List[Tuple[Document, float]]:
    """
    Convenience function to score documents using default service

    Args:
        query: User query
        documents: Retrieved documents
        similarity_scores: Vector search similarity scores
        threshold: Minimum score threshold (None to use config default)

    Returns:
        List of (document, score) tuples above threshold, sorted by score
    """
    if threshold is None:
        config = get_config()
        threshold = config.get('rag_scoring.default_threshold', 0.3)

    scored_docs = default_scoring_service.compute_combined_scores(query, documents, similarity_scores)
    return default_scoring_service.filter_by_threshold(scored_docs, threshold)


def dedupe_documents(documents: List[Document]) -> List[Document]:
    """
    Deduplicate documents by (source, chunk_index) with stable ordering
    
    Args:
        documents: List of documents potentially containing duplicates
        
    Returns:
        List of unique documents maintaining order of first occurrence
    """
    seen = set()
    unique_docs = []
    
    for doc in documents:
        # Create unique key from source and chunk index
        source = doc.metadata.get('source', '')
        chunk_index = doc.metadata.get('chunk_index', 0)
        key = (source, chunk_index)
        
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    
    logger.debug(f"Deduplication: {len(documents)} -> {len(unique_docs)} documents")
    return unique_docs


def score_and_pack(query: str,
                  documents: List[Document],
                  similarity_scores: List[float],
                  threshold: float,
                  max_results: int) -> List[KBDocument]:
    """
    Score documents, apply threshold, dedupe, and pack into KBDocument format
    
    This is the main helper for multi-agent workflow retrieval strategy.
    
    Args:
        query: User query for scoring
        documents: Retrieved documents
        similarity_scores: Vector search similarity scores (can be empty/None)
        threshold: Minimum score threshold for filtering
        max_results: Maximum number of results to return
        
    Returns:
        List of KBDocument objects, scored, filtered, deduped, and limited to max_results
    """
    if not documents:
        logger.debug("No documents to score and pack")
        return []
    
    # Handle missing similarity scores with neutral fallback
    if not similarity_scores or len(similarity_scores) != len(documents):
        logger.warning("Missing or mismatched similarity scores, using neutral fallback")
        similarity_scores = [0.7] * len(documents)
    
    # Score documents using existing service
    try:
        scored_docs = default_scoring_service.compute_combined_scores(
            query, documents, similarity_scores
        )
        
        # Apply threshold
        filtered_docs = default_scoring_service.filter_by_threshold(
            scored_docs, threshold
        )
        
        if not filtered_docs:
            logger.info(f"No documents above threshold {threshold}")
            return []
        
        # Deduplicate before packing
        unique_docs_with_scores = []
        seen_keys = set()
        
        for doc, score in filtered_docs:
            source = doc.metadata.get('source', '')
            chunk_index = doc.metadata.get('chunk_index', 0)
            key = (source, chunk_index)
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_docs_with_scores.append((doc, score))
        
        logger.debug(f"After deduplication: {len(unique_docs_with_scores)} unique documents")
        
        # Limit to max_results
        limited_docs = unique_docs_with_scores[:max_results]
        
        # Pack into KBDocument format
        kb_documents = []
        for doc, score in limited_docs:
            kb_doc = KBDocument(
                content=doc.page_content,
                score=score,
                source=doc.metadata.get('source', 'unknown'),
                metadata=doc.metadata
            )
            kb_documents.append(kb_doc)
        
        logger.info(f"Scored and packed {len(kb_documents)} documents (threshold={threshold}, max={max_results})")
        return kb_documents
        
    except Exception as e:
        logger.error(f"Error in score_and_pack: {e}")
        return []