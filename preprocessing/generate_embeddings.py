#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized version of generate_page_embeddings.py based on API limit testing.
Processes documents with maximum efficiency while maintaining correct data structure.
"""

import os
import json
import numpy as np
import openai
import tiktoken
import time
import random
from pathlib import Path
from collections import defaultdict
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import gc
import pickle
from datetime import datetime

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Azure OpenAI credentials
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "text-embedding-3-large")

# Embedding parameters
EMBEDDING_MODEL = AZURE_DEPLOYMENT
EMBEDDING_ENCODING = "cl100k_base"
MAX_TOKENS = 8191

# Optimized API limits based on testing
API_CALLS_PER_MINUTE = 280  # Conservative limit based on test results
MAX_BATCH_SIZE = 200  # Maximum items per API call
MAX_RETRIES = 6
BACKOFF_BASE = 2
BACKOFF_MAX = 60

# Processing parameters
CHECKPOINT_INTERVAL = 20  # Save progress every N documents
RATE_LIMIT_BUFFER = 0.8  # Use 80% of rate limit for safety


class OptimizedEmbeddingGenerator:
    """Generate embeddings with optimized batch processing"""
    
    def __init__(self, 
                 azure_endpoint: str = AZURE_ENDPOINT,
                 azure_key: str = AZURE_KEY,
                 azure_deployment: str = AZURE_DEPLOYMENT,
                 max_batch_size: int = MAX_BATCH_SIZE,
                 max_tokens: int = MAX_TOKENS,
                 api_calls_per_minute: int = API_CALLS_PER_MINUTE):
        """Initialize the embedding generator."""
        # Initialize Azure OpenAI client
        self.client = openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version="2023-05-15"
        )
        
        self.deployment = azure_deployment
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
        
        # Rate limiting
        self.api_calls_per_minute = api_calls_per_minute * RATE_LIMIT_BUFFER
        self.min_time_between_calls = 60.0 / self.api_calls_per_minute
        self.last_api_call_time = 0
        self.api_call_count = 0
        
        logger.info(f"=== Optimized Embedding Generator Initialized ===")
        logger.info(f"Deployment: {azure_deployment}")
        logger.info(f"Max batch size: {max_batch_size}")
        logger.info(f"API calls per minute: {self.api_calls_per_minute:.0f}")
        logger.info(f"Min time between calls: {self.min_time_between_calls:.3f}s")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limit"""
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.max_tokens:
            return [text]
        
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append(self.encoding.decode(chunk_tokens))
            i += self.max_tokens
        
        return chunks
    
    def wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.min_time_between_calls:
            sleep_time = self.min_time_between_calls - time_since_last_call
            time.sleep(sleep_time)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Request embeddings for a batch of texts with rate limiting and retries."""
        # Wait for rate limit
        self.wait_for_rate_limit()
        
        # Clean texts
        cleaned_texts = []
        for t in texts:
            cleaned = (t or "").replace("\n", " ").strip()
            # If text is empty, use a placeholder
            cleaned_texts.append(cleaned if cleaned else " ")
        
        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                # Make API call
                result = self.client.embeddings.create(
                    input=cleaned_texts,
                    model=self.deployment,
                )
                
                # Update rate limiting
                self.last_api_call_time = time.time()
                self.api_call_count += 1
                
                # Extract embeddings
                embeddings = [d.embedding for d in result.data]
                return embeddings
                
            except openai.RateLimitError as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    logger.error(f"Rate limit error after {MAX_RETRIES} retries: {e}")
                    return [None] * len(cleaned_texts)
                
                # Exponential backoff for rate limit errors
                backoff_seconds = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                logger.warning(f"Rate limit hit. Backing off {backoff_seconds:.1f}s...")
                time.sleep(backoff_seconds)
                
            except Exception as e:
                attempt += 1
                if attempt > MAX_RETRIES:
                    logger.error(f"Batch failed after {MAX_RETRIES} retries: {e}")
                    return [None] * len(cleaned_texts)
                
                backoff_seconds = min(BACKOFF_BASE ** attempt, BACKOFF_MAX) + random.random()
                logger.warning(f"API error (attempt {attempt}): {e}. Backing off {backoff_seconds:.1f}s...")
                time.sleep(backoff_seconds)
        
        return [None] * len(cleaned_texts)
    
    def read_page_content(self, file_path: str) -> Optional[str]:
        """Read and return the content of a page file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content if content.strip() else None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def process_documents_optimized(self, 
                                   documents: Dict[str, List[Tuple[int, str]]], 
                                   output_dir: str,
                                   resume: bool = False):
        """Process all documents with optimized batching"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load checkpoint if resuming
        processed_docs = set()
        checkpoint_file = os.path.join(output_dir, '.checkpoint.pkl')
        
        if resume and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    processed_docs = checkpoint_data.get('processed_docs', set())
                    self.api_call_count = checkpoint_data.get('api_call_count', 0)
                logger.info(f"Resumed from checkpoint: {len(processed_docs)} documents already processed")
                logger.info(f"Previous API calls made: {self.api_call_count}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                processed_docs = set()
        
        # Check existing files
        existing_files = set()
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_embeddings.json'):
                doc_id = file_name.replace('_embeddings.json', '')
                existing_files.add(doc_id)
        
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing embedding files")
            processed_docs.update(existing_files)
        
        # Prepare all texts to embed
        all_texts = []  # List of (doc_id, page_num, chunk_idx, text)
        doc_page_structure = defaultdict(lambda: defaultdict(list))
        
        logger.info("=== Preparing documents for batch processing ===")
        
        skipped_count = 0
        total_pages = 0
        total_chunks = 0
        
        for doc_id in sorted(documents.keys()):
            if doc_id in processed_docs:
                skipped_count += 1
                continue
            
            pages = documents[doc_id]
            pages.sort(key=lambda x: x[0])  # Sort by page number
            
            doc_has_content = False
            for page_num, file_path in pages:
                content = self.read_page_content(file_path)
                if content:
                    chunks = self.split_text_into_chunks(content)
                    total_pages += 1
                    total_chunks += len(chunks)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        text_idx = len(all_texts)
                        all_texts.append((doc_id, page_num, chunk_idx, chunk))
                        doc_page_structure[doc_id][page_num].append(text_idx)
                    doc_has_content = True
            
            if not doc_has_content:
                logger.warning(f"Document {doc_id} has no valid content")
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed documents")
        
        if not all_texts:
            logger.info("No new texts to process")
            return
        
        logger.info(f"Documents to process: {len(doc_page_structure)}")
        logger.info(f"Total pages: {total_pages}")
        logger.info(f"Total text chunks: {len(all_texts)}")
        logger.info(f"Average chunks per page: {len(all_texts)/total_pages:.1f}")
        
        # Calculate batches
        total_batches = (len(all_texts) + self.max_batch_size - 1) // self.max_batch_size
        estimated_time = (total_batches * self.min_time_between_calls) / 60
        
        logger.info(f"Total batches to process: {total_batches}")
        logger.info(f"Estimated minimum time: {estimated_time:.1f} minutes")
        
        # Process in batches
        all_embeddings = [None] * len(all_texts)
        start_time = time.time()
        batch_times = []
        
        logger.info("=== Starting batch processing ===")
        
        for batch_idx in range(0, len(all_texts), self.max_batch_size):
            batch_start = time.time()
            batch_end = min(batch_idx + self.max_batch_size, len(all_texts))
            batch_texts = [all_texts[i][3] for i in range(batch_idx, batch_end)]
            
            # Get embeddings for this batch
            batch_embeddings = self.get_embeddings_batch(batch_texts)
            
            # Store embeddings
            successful_embeddings = 0
            for i, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    all_embeddings[batch_idx + i] = embedding
                    successful_embeddings += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Progress update
            current_batch = (batch_idx // self.max_batch_size) + 1
            if current_batch % 5 == 0 or current_batch == total_batches:
                elapsed = time.time() - start_time
                processed_texts = batch_idx + len(batch_texts)
                rate = processed_texts / elapsed if elapsed > 0 else 0
                remaining_texts = len(all_texts) - processed_texts
                eta = remaining_texts / rate if rate > 0 else 0
                
                avg_batch_time = np.mean(batch_times[-10:])  # Average of last 10 batches
                
                logger.info(f"Batch {current_batch}/{total_batches} | "
                          f"Texts: {processed_texts}/{len(all_texts)} | "
                          f"Success: {successful_embeddings}/{len(batch_texts)} | "
                          f"Rate: {rate:.0f} texts/sec | "
                          f"Batch time: {batch_time:.2f}s | "
                          f"ETA: {eta/60:.1f} min")
        
        logger.info("=== Organizing embeddings by document ===")
        
        # Organize embeddings by document and save
        docs_saved = 0
        docs_failed = 0
        
        for doc_id, pages_data in sorted(doc_page_structure.items()):
            try:
                page_numbers = []
                page_embeddings = []
                
                for page_num in sorted(pages_data.keys()):
                    chunk_indices = pages_data[page_num]
                    chunk_embeddings = []
                    
                    for idx in chunk_indices:
                        if all_embeddings[idx] is not None:
                            chunk_embeddings.append(all_embeddings[idx])
                    
                    if chunk_embeddings:
                        # Average embeddings for this page
                        avg_embedding = np.mean(chunk_embeddings, axis=0)
                        page_embeddings.append(avg_embedding)
                        page_numbers.append(page_num)
                    else:
                        logger.warning(f"No valid embeddings for {doc_id} page {page_num}")
                
                if page_embeddings:
                    embeddings_array = np.array(page_embeddings)
                    
                    # Create result with correct structure
                    result = {
                        'doc_id': doc_id,
                        'page_numbers': page_numbers,
                        'embeddings': embeddings_array.tolist(),
                        'embedding_dim': embeddings_array.shape[1],
                        'num_pages': len(page_numbers),
                        'correct_order': sorted(page_numbers)  # Ground truth ordering
                    }
                    
                    # Save document embeddings
                    output_file = os.path.join(output_dir, f"{doc_id}_embeddings.json")
                    with open(output_file, 'w') as f:
                        json.dump(result, f)
                    
                    docs_saved += 1
                    processed_docs.add(doc_id)
                    
                    # Save checkpoint periodically
                    if docs_saved % CHECKPOINT_INTERVAL == 0:
                        checkpoint_data = {
                            'processed_docs': processed_docs,
                            'api_call_count': self.api_call_count,
                            'timestamp': datetime.now().isoformat()
                        }
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(checkpoint_data, f)
                    
                    if docs_saved % 10 == 0:
                        logger.info(f"Saved {docs_saved} documents...")
                    
                else:
                    logger.error(f"No valid embeddings for document {doc_id}")
                    docs_failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to save {doc_id}: {e}")
                docs_failed += 1
        
        # Final statistics
        elapsed_total = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("=== Processing Complete ===")
        logger.info(f"Total time: {elapsed_total/60:.2f} minutes")
        logger.info(f"Documents processed: {docs_saved}")
        logger.info(f"Documents failed: {docs_failed}")
        logger.info(f"Total texts embedded: {len(all_texts)}")
        logger.info(f"API calls made: {self.api_call_count}")
        logger.info(f"Average rate: {len(all_texts)/elapsed_total:.1f} texts/sec")
        logger.info(f"Average API call rate: {self.api_call_count/(elapsed_total/60):.1f} calls/min")
        
        # Save metadata
        metadata = {
            'total_documents': docs_saved,
            'failed_documents': docs_failed,
            'total_texts_embedded': len(all_texts),
            'total_pages': total_pages,
            'api_calls_made': self.api_call_count,
            'processing_time_minutes': elapsed_total / 60,
            'max_batch_size': self.max_batch_size,
            'api_calls_per_minute': self.api_calls_per_minute,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed")
        
        logger.info(f"Metadata saved to {metadata_file}")
        logger.info("=" * 60)


def load_documents(data_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """Load and organize document files from directory"""
    documents = defaultdict(list)
    
    logger.info(f"Loading documents from {data_dir}...")
    
    file_count = 0
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith('.txt'):
            continue
        
        try:
            # Parse document ID and page number
            # Expected format: {document_id}_page_{page_number}.txt
            parts = file_name.rsplit('_page_', 1)
            if len(parts) != 2:
                logger.warning(f"Skipping file with invalid format: {file_name}")
                continue
            
            doc_id = parts[0]
            page_num = int(parts[1].replace('.txt', ''))
            file_path = os.path.join(data_dir, file_name)
            
            documents[doc_id].append((page_num, file_path))
            file_count += 1
            
        except ValueError as e:
            logger.warning(f"Error parsing page number from {file_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error parsing filename {file_name}: {e}")
            continue
    
    logger.info(f"Loaded {file_count} files from {len(documents)} documents")
    
    # Show statistics
    page_counts = [len(pages) for pages in documents.values()]
    if page_counts:
        logger.info(f"Pages per document: min={min(page_counts)}, "
                   f"max={max(page_counts)}, avg={np.mean(page_counts):.1f}")
    
    return documents


def main():
    parser = argparse.ArgumentParser(description="Generate page embeddings with optimized processing")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default="./text_pages",
                       help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
                       help="Directory to save embeddings")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=200,
                       help="Number of texts per API call (default: 200)")
    parser.add_argument("--api_calls_per_minute", type=int, default=280,
                       help="API calls per minute limit (default: 280)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Page Embedding Generation - Optimized Version")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"API calls/minute: {args.api_calls_per_minute}")
    logger.info(f"Resume mode: {args.resume}")
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return
    
    # Load documents
    documents = load_documents(args.data_dir)
    
    if not documents:
        logger.error("No valid documents found!")
        return
    
    # Initialize generator and process
    generator = OptimizedEmbeddingGenerator(
        azure_endpoint=AZURE_ENDPOINT,
        azure_key=AZURE_KEY,
        azure_deployment=AZURE_DEPLOYMENT,
        max_batch_size=args.batch_size,
        max_tokens=MAX_TOKENS,
        api_calls_per_minute=args.api_calls_per_minute
    )
    
    generator.process_documents_optimized(
        documents=documents,
        output_dir=args.output_dir,
        resume=args.resume
    )


if __name__ == "__main__":
    main()