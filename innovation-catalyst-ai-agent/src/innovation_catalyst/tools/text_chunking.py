# src/innovation_catalyst/tools/text_chunking.py
"""
Intelligent text chunking with semantic boundary preservation.
Implements advanced text segmentation with overlap and metadata generation.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
from collections import Counter
import json 

# NLP libraries
import spacy
from spacy.lang.en import English
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from ..models.document import TextChunk, ProcessedDocument
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

from smolagents import tool

logger = get_logger(__name__)
config = get_config()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 500  # Target words per chunk
    overlap_size: int = 50  # Overlapping words
    min_chunk_size: int = 50  # Minimum words per chunk
    max_chunk_size: int = 1000  # Maximum words per chunk
    preserve_sentences: bool = True  # Respect sentence boundaries
    preserve_paragraphs: bool = True  # Respect paragraph boundaries

class SentenceTokenizer:
    """
    Advanced sentence tokenization with multiple fallback strategies.
    
    Features:
        - spaCy-based tokenization (primary)
        - NLTK fallback
        - Custom rule-based tokenization
        - Abbreviation handling
    """
    
    def __init__(self):
        self.spacy_nlp = self._load_spacy_model()
        self.nltk_available = True
        
    def _load_spacy_model(self) -> Optional[spacy.Language]:
        """Load spaCy model with fallback."""
        try:
            # Try to load the configured model
            nlp = spacy.load(config.model.spacy_model)
            # Disable unnecessary components for performance
            nlp.disable_pipes(['tagger', 'parser', 'ner', 'lemmatizer'])
            return nlp
        except OSError:
            logger.warning(f"spaCy model {config.model.spacy_model} not found, using fallback")
            try:
                # Try English model
                nlp = English()
                nlp.add_pipe('sentencizer')
                return nlp
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {str(e)}")
                return None
    
    @log_function_performance("sentence_tokenization")
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using best available method.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of sentences
        """
        if not text or not text.strip():
            return []
        
        # Try spaCy first
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                logger.warning(f"spaCy sentence tokenization failed: {str(e)}")
        
        # Fallback to NLTK
        if self.nltk_available:
            try:
                sentences = sent_tokenize(text)
                return [sent.strip() for sent in sentences if sent.strip()]
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {str(e)}")
                self.nltk_available = False
        
        # Final fallback: simple rule-based
        return self._simple_sentence_split(text)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple rule-based sentence splitting."""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Minimum sentence length
                cleaned_sentences.append(sent)
        
        return cleaned_sentences

class TextChunker:
    """
    Advanced text chunking with semantic boundary preservation.
    
    Features:
        - Sentence boundary preservation
        - Paragraph structure maintenance
        - Configurable overlap
        - Metadata generation
        - Performance optimization
    """
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.config = chunking_config or ChunkingConfig(
            chunk_size=config.processing.chunk_size,
            overlap_size=config.processing.chunk_overlap,
            min_chunk_size=config.processing.min_chunk_words
        )
        self.sentence_tokenizer = SentenceTokenizer()
        
        # Load stopwords for keyword extraction
        try:
            self.stopwords = set(stopwords.words('english'))
        except Exception:
            self.stopwords = set()
    
    @log_function_performance("text_chunking")
    def chunk_text_intelligently(
        self, 
        text: str, 
        document_id: str,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Chunk text intelligently with semantic boundary preservation.
        
        Args:
            text (str): Input text to chunk
            document_id (str): Source document identifier
            chunk_size (Optional[int]): Override default chunk size
            overlap_size (Optional[int]): Override default overlap size
            
        Returns:
            List[TextChunk]: List of text chunks with metadata
        """
        if not text or not text.strip():
            return []
        
        # Use provided parameters or defaults
        target_chunk_size = chunk_size or self.config.chunk_size
        target_overlap_size = overlap_size or self.config.overlap_size
        
        logger.info(f"Chunking text: {len(text)} characters, target size: {target_chunk_size} words")
        
        # Split into sentences
        sentences = self.sentence_tokenizer.tokenize_sentences(text)
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        # Create chunks from sentences
        chunks = self._create_chunks_from_sentences(
            sentences, 
            target_chunk_size, 
            target_overlap_size
        )
        
        # Generate TextChunk objects with metadata
        text_chunks = []
        for i, chunk_data in enumerate(chunks):
            chunk = self._create_text_chunk(
                chunk_data, 
                i, 
                document_id, 
                text
            )
            text_chunks.append(chunk)
        
        logger.info(f"Created {len(text_chunks)} chunks from {len(sentences)} sentences")
        return text_chunks
    
    def _create_chunks_from_sentences(
        self, 
        sentences: List[str], 
        target_chunk_size: int,
        target_overlap_size: int
    ) -> List[Dict[str, Any]]:
        """Create chunks from sentences with overlap."""
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed target size
            if (current_word_count + sentence_words > target_chunk_size and 
                current_chunk_sentences and 
                current_word_count >= self.config.min_chunk_size):
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunk_data = {
                    'text': chunk_text,
                    'sentences': current_chunk_sentences.copy(),
                    'word_count': current_word_count,
                    'sentence_count': len(current_chunk_sentences)
                }
                chunks.append(chunk_data)
                
                # Calculate overlap
                overlap_sentences = self._calculate_overlap_sentences(
                    current_chunk_sentences, 
                    target_overlap_size
                )
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences
                current_word_count = sum(len(s.split()) for s in overlap_sentences)
            
            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
            i += 1
        
        # Add final chunk if it has content
        if current_chunk_sentences and current_word_count >= self.config.min_chunk_size:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_data = {
                'text': chunk_text,
                'sentences': current_chunk_sentences,
                'word_count': current_word_count,
                'sentence_count': len(current_chunk_sentences)
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _calculate_overlap_sentences(
        self, 
        sentences: List[str], 
        target_overlap_words: int
    ) -> List[str]:
        """Calculate overlap sentences for next chunk."""
        if target_overlap_words <= 0 or not sentences:
            return []
        
        overlap_sentences = []
        overlap_word_count = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            
            if overlap_word_count + sentence_words <= target_overlap_words:
                overlap_sentences.insert(0, sentence)
                overlap_word_count += sentence_words
            else:
                break
        
        return overlap_sentences
    
    def _create_text_chunk(
        self, 
        chunk_data: Dict[str, Any], 
        chunk_index: int,
        document_id: str,
        full_text: str
    ) -> TextChunk:
        """Create TextChunk object with metadata."""
        chunk_text = chunk_data['text']
        
        # Calculate positions in full text
        start_pos = full_text.find(chunk_text)
        end_pos = start_pos + len(chunk_text) if start_pos != -1 else len(chunk_text)
        
        # Extract basic metadata
        keywords = self._extract_keywords(chunk_text)
        
        return TextChunk(
            content=chunk_text,
            chunk_index=chunk_index,
            word_count=chunk_data['word_count'],
            sentence_count=chunk_data['sentence_count'],
            start_position=max(0, start_pos),
            end_position=end_pos,
            source_document_id=document_id,
            keywords=keywords
        )
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from chunk text."""
        if not text:
            return []
        
        try:
            # Simple keyword extraction using word frequency
            words = word_tokenize(text.lower())
            
            # Filter out stopwords and short words
            filtered_words = [
                word for word in words 
                if (word.isalpha() and 
                    len(word) > 2 and 
                    word not in self.stopwords)
            ]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Return top keywords
            return [word for word, _ in word_freq.most_common(max_keywords)]
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {str(e)}")
            return []
    
    def validate_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Validate chunk quality and return statistics."""
        if not chunks:
            return {
                'valid': False,
                'error': 'No chunks provided'
            }
        
        stats = {
            'total_chunks': len(chunks),
            'total_words': sum(chunk.word_count for chunk in chunks),
            'total_sentences': sum(chunk.sentence_count for chunk in chunks),
            'avg_words_per_chunk': 0,
            'min_words': 0,
            'max_words': 0,
            'chunks_below_min': 0,
            'chunks_above_max': 0,
            'valid': True
        }
        
        if chunks:
            word_counts = [chunk.word_count for chunk in chunks]
            stats['avg_words_per_chunk'] = sum(word_counts) / len(word_counts)
            stats['min_words'] = min(word_counts)
            stats['max_words'] = max(word_counts)
            stats['chunks_below_min'] = sum(1 for wc in word_counts if wc < self.config.min_chunk_size)
            stats['chunks_above_max'] = sum(1 for wc in word_counts if wc > self.config.max_chunk_size)
        
        return stats

# Global chunker instance
text_chunker = TextChunker()

@tool
def chunk_text_intelligently(
    text: str, 
    document_id: str,
    chunk_size: Optional[int] = None,
    overlap_size: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Chunk text intelligently - main entry point for SmolAgent tool.
    
    Args:
        text (str): Input text to chunk
        document_id (str): Source document identifier
        chunk_size (Optional[int]): Target words per chunk
        overlap_size (Optional[int]): Overlapping words between chunks
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries
    """
    chunks = text_chunker.chunk_text_intelligently(
        text, document_id, chunk_size, overlap_size
    )
    
    # Convert TextChunk objects to dictionaries for SmolAgent compatibility
    return [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "word_count": chunk.word_count,
            "sentence_count": chunk.sentence_count,
            "start_position": chunk.start_position,
            "end_position": chunk.end_position,
            "source_document_id": chunk.source_document_id,
            "keywords": chunk.keywords,
            "created_at": chunk.created_at.isoformat()
        }
        for chunk in chunks
    ]
