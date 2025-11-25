import os
import json
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), 'output.json')
GENERATION_MODEL = 'gemini-2.0-flash-001'
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

app = Flask(__name__)
CORS(app)

docs: List[Dict[str, Any]] = []
chunks: List[Dict[str, Any]] = []


def _load_documents(path: str) -> List[Dict[str, str]]:
    """Load URL->content mapping from output.json."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'output.json not found at {path}')
    with open(path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for url, content in data.items():
        if not content or not isinstance(content, str):
            continue
        documents.append({
            'url': url,
            'content': content
        })
    
    print(f"Loaded {len(documents)} documents from {path}")
    return documents


def _chunk_document(doc: Dict[str, str]) -> List[Dict[str, Any]]:
    """Split a document into overlapping chunks."""
    content = doc['content']
    url = doc['url']
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(content):
        end = start + CHUNK_SIZE
        chunk_text = content[start:end]
        
        # Try to break at sentence boundary
        if end < len(content):
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > CHUNK_SIZE * 0.7:  # Don't break too early
                chunk_text = chunk_text[:break_point + 1]
                end = start + break_point + 1
        
        if chunk_text.strip():
            chunks.append({
                'url': url,
                'content': chunk_text.strip(),
                'chunk_id': chunk_id,
                'start_pos': start
            })
            chunk_id += 1
        
        start = end - CHUNK_OVERLAP
    
    return chunks


def _build_search_index(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build searchable chunks from documents."""
    all_chunks = []
    for doc in documents:
        doc_chunks = _chunk_document(doc)
        all_chunks.extend(doc_chunks)
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def _tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    # Convert to lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def _bm25_search(query: str, chunks: List[Dict[str, Any]], k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    """BM25 ranking algorithm for better relevance than simple keyword matching."""
    query_tokens = set(_tokenize(query))
    
    if not query_tokens:
        return []
    
    # Calculate document frequency for each term
    N = len(chunks)
    df = defaultdict(int)  # document frequency
    
    for chunk in chunks:
        terms_in_doc = set(_tokenize(chunk['content']))
        for term in query_tokens:
            if term in terms_in_doc:
                df[term] += 1
    
    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avgdl = sum(len(_tokenize(chunk['content'])) for chunk in chunks) / N
    
    # Score each chunk
    scores = []
    for chunk in chunks:
        doc_tokens = _tokenize(chunk['content'])
        doc_len = len(doc_tokens)
        term_freq = defaultdict(int)
        
        for token in doc_tokens:
            term_freq[token] += 1
        
        score = 0.0
        for term in query_tokens:
            if term in term_freq:
                freq = term_freq[term]
                idf = np.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (doc_len / avgdl))
                score += idf * (numerator / denominator)
        
        if score > 0:
            scores.append((chunk, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:k]


def _simple_keyword_search(query: str, chunks: List[Dict[str, Any]], k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    """Simple keyword search with scoring based on match count and position."""
    query_lower = query.lower()
    query_tokens = _tokenize(query)
    
    matches = []
    for chunk in chunks:
        content_lower = chunk['content'].lower()
        
        # Exact phrase match gets highest score
        exact_match_count = content_lower.count(query_lower)
        
        # Token matches
        chunk_tokens = _tokenize(chunk['content'])
        token_matches = sum(1 for token in query_tokens if token in chunk_tokens)
        
        # Calculate score
        if exact_match_count > 0:
            score = 10.0 + exact_match_count * 5.0  # Heavily weight exact matches
        elif token_matches > 0:
            score = float(token_matches) / len(query_tokens)
        else:
            continue
        
        # Boost if query appears early in chunk
        if query_lower in content_lower[:100]:
            score *= 1.5
        
        matches.append((chunk, score))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches[:k]


def _exact_term_search(query: str, chunks: List[Dict[str, Any]], k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    """High-precision exact term search for short queries (e.g. names).

    If the raw query string appears in a chunk's content or URL (case-insensitive),
    that chunk is returned with a high fixed score so it ranks above fuzzy matches.
    """
    term = (query or '').strip().lower()
    if not term:
        return []

    results: List[Tuple[Dict[str, Any], float]] = []
    for chunk in chunks:
        text = (chunk.get('content') or '').lower()
        url = (chunk.get('url') or '').lower()
        if term in text or term in url:
            results.append((chunk, 100.0))  # fixed high score
            if len(results) >= k:
                break

    return results


def _search(query: str, k: int = TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    """Hybrid search combining exact term, BM25, and keyword matching."""
    if not chunks:
        return []

    # For short queries like names ("Keeling"), try an exact term search first.
    exact_results = _exact_term_search(query, chunks, k=k)
    if exact_results:
        print(f"\n=== Exact term hits for: '{query}' ===")
        for i, (chunk, score) in enumerate(exact_results, 1):
            print(f"{i}. Score: {score:.2f} | {chunk['url']}")
            print(f"   Content: {chunk['content'][:100]}...")
        return exact_results

    # Try BM25 first
    bm25_results = _bm25_search(query, chunks, k=k*2)
    
    # Also try simple keyword search
    keyword_results = _simple_keyword_search(query, chunks, k=k*2)
    
    # Merge results, preferring keyword matches for short queries
    if len(query.split()) <= 2:
        # For short queries, keyword search is more reliable
        primary = keyword_results
        secondary = bm25_results
    else:
        # For longer queries, BM25 handles it better
        primary = bm25_results
        secondary = keyword_results
    
    # Combine and deduplicate
    seen_chunks = {}
    combined = []
    
    for chunk, score in primary:
        key = (chunk['url'], chunk['chunk_id'])
        seen_chunks[key] = (chunk, score)
    
    for chunk, score in secondary:
        key = (chunk['url'], chunk['chunk_id'])
        if key not in seen_chunks:
            seen_chunks[key] = (chunk, score * 0.5)  # Lower weight for secondary
    
    combined = list(seen_chunks.values())
    combined.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n=== Search for: '{query}' ===")
    for i, (chunk, score) in enumerate(combined[:5], 1):
        print(f"{i}. Score: {score:.2f} | {chunk['url']}")
        print(f"   Content: {chunk['content'][:100]}...")
    
    return combined[:k]


def _ensure_genai_configured():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY environment variable is not set')
    genai.configure(api_key=api_key)


def _build_prompt(question: str, contexts: List[Dict[str, str]]) -> str:
    """Build prompt with retrieved contexts."""
    lines = [
        'You are BisonChat, Howard University\'s helpful AI assistant.',
        'Answer questions about Howard University directly and conversationally using the provided context.',
        'Be concise and natural - give direct answers without unnecessary citations unless asked for sources.',
        '',
        'Context from Howard University websites:',
    ]
    
    for i, c in enumerate(contexts, start=1):
        ctx = (c['content'] or '').strip()
        lines.append(f'[{i}] {ctx}')
        lines.append('')
    
    lines.append(f'Question: {question}')
    lines.append('')
    lines.append('Answer concisely and helpfully:')
    return '\n'.join(lines)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'documents': len(docs),
        'chunks': len(chunks)
    })


@app.route('/search_content', methods=['POST'])
def search_content():
    """Debug endpoint to search for specific terms in the content."""
    try:
        payload = request.get_json(force=True)
        search_term = (payload.get('term') or '').strip().lower()
        if not search_term:
            return jsonify({'error': 'term is required'}), 400
        
        matches = []
        for doc in docs:
            content_lower = doc['content'].lower()
            count = content_lower.count(search_term)
            if count > 0:
                # Find all occurrences
                contexts = []
                start_idx = 0
                for _ in range(min(count, 3)):  # Max 3 examples per doc
                    idx = content_lower.find(search_term, start_idx)
                    if idx == -1:
                        break
                    start = max(0, idx - 100)
                    end = min(len(doc['content']), idx + 100)
                    contexts.append(f"...{doc['content'][start:end]}...")
                    start_idx = idx + len(search_term)
                
                matches.append({
                    'url': doc['url'],
                    'occurrences': count,
                    'contexts': contexts
                })
        
        return jsonify({
            'search_term': search_term,
            'matches_found': len(matches),
            'total_occurrences': sum(m['occurrences'] for m in matches),
            'matches': matches
        }), 200
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/ask', methods=['POST'])
def ask():
    try:
        payload = request.get_json(force=True)
        question = (payload.get('query') or '').strip()
        if not question:
            return jsonify({'error': 'query is required'}), 400

        results = _search(question, TOP_K)
        if not results:
            return jsonify({
                'answer': 'I don\'t have information about that in my knowledge base.', 
                'sources': [], 
                'note': 'No relevant documents found'
            }), 200

        contexts = [r[0] for r in results]
        
        prompt = _build_prompt(question, contexts)

        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        answer_text = response.text if hasattr(response, 'text') else str(response)

        sources = [
            {
                'rank': i + 1,
                'url': chunk['url'],
                'score': score,
                'chunk_id': chunk.get('chunk_id', 0),
                'preview': chunk['content'][:150]
            }
            for i, (chunk, score) in enumerate(results)
        ]
        
        return jsonify({
            'answer': answer_text, 
            'sources': sources
        }), 200

    except Exception as e:
        print(f"Error in /ask: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def initialize_app():
    global docs, chunks
    load_dotenv()
    
    _ensure_genai_configured()
    
    docs = _load_documents(OUTPUT_JSON_PATH)
    if not docs:
        raise RuntimeError('No documents found in output.json')
    
    # Build search index
    chunks = _build_search_index(docs)
    print(f'RAG system ready with {len(docs)} documents and {len(chunks)} searchable chunks')


if __name__ == '__main__':
    initialize_app()
    port = int(os.getenv('PORT', '8000'))
    app.run(host='0.0.0.0', port=port)