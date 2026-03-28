"""
Document Preprocessing Module

Handles:
- Markdown parsing
- Document chunking by sections
- Thai text normalization
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
import markdown
from bs4 import BeautifulSoup


def normalize_thai_text(text: str) -> str:
    """
    Normalize Thai text by handling character variations.
    
    - Normalize tone marks and vowels
    - Remove extra whitespace
    - Standardize special characters
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize Thai digit characters to Arabic digits (optional)
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    for i, char in enumerate(thai_digits):
        text = text.replace(char, arabic_digits[i])
    
    return text


def parse_markdown_file(file_path: str) -> Dict[str, Any]:
    """
    Parse a markdown file and extract structured content.
    
    Returns:
        dict with keys:
        - filepath: original file path
        - filename: file name without extension
        - content: raw content
        - sections: list of sections with headers
        - metadata: extracted metadata (SKU, brand, etc.)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract filename
    filename = Path(file_path).stem
    
    # Parse markdown to HTML for structure extraction
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract sections by headers
    sections = []
    current_section = {'title': 'Introduction', 'content': []}
    
    for element in soup.children:
        if hasattr(element, 'name') and element.name in ['h1', 'h2', 'h3', 'h4']:
            # Save previous section
            if current_section['content']:
                sections.append(current_section)
            # Start new section
            current_section = {
                'title': element.get_text().strip(),
                'content': []
            }
        elif hasattr(element, 'get_text'):
            text = element.get_text().strip()
            if text:
                current_section['content'].append(text)
    
    # Add last section
    if current_section['content']:
        sections.append(current_section)
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(filename)
    
    # Also try to extract metadata from content
    metadata.update(extract_metadata_from_content(content))
    
    return {
        'filepath': file_path,
        'filename': filename,
        'content': content,
        'sections': sections,
        'metadata': metadata
    }


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract metadata from filename.
    
    Example: SF-SP-002_saifah_phone_x9_pro
    - SKU: SF-SP-002
    - Brand: saifah
    - Category: phone
    - Product: x9_pro
    """
    metadata = {
        'sku': '',
        'brand': '',
        'category': '',
        'product': ''
    }
    
    parts = filename.split('_')
    if parts:
        # First part is usually SKU
        metadata['sku'] = parts[0].upper()
        
        # Remaining parts describe the product
        if len(parts) > 1:
            metadata['brand'] = parts[1] if len(parts) > 1 else ''
            metadata['category'] = parts[2] if len(parts) > 2 else ''
            metadata['product'] = '_'.join(parts[3:]) if len(parts) > 3 else ''
    
    return metadata


def extract_metadata_from_content(content: str) -> Dict[str, str]:
    """
    Extract metadata from content (SKU, brand, price, etc.).
    """
    metadata = {}
    
    # Extract SKU (pattern: XXX-XX-XXX)
    sku_match = re.search(r'รหัสสินค้า:\s*([A-Z]{2,4}-[A-Z]{2,3}-\d{3})', content)
    if sku_match:
        metadata['sku'] = sku_match.group(1)
    
    # Extract brand
    brand_match = re.search(r'แบรนด์:\s*(.+?)(?:\n|$)', content)
    if brand_match:
        metadata['brand'] = brand_match.group(1).strip()
    
    # Extract price
    price_match = re.search(r'ราคา:\s*฿?([\d,]+)', content)
    if price_match:
        metadata['price'] = price_match.group(1)
    
    # Extract category
    category_match = re.search(r'หมวดหมู่:\s*(.+?)(?:\n|$)', content)
    if category_match:
        metadata['category'] = category_match.group(1).strip()
    
    return metadata


def chunk_document(doc: Dict[str, Any], 
                   chunk_size: int = 500,
                   chunk_overlap: int = 50,
                   chunk_by_section: bool = True) -> List[Dict[str, Any]]:
    """
    Chunk a document into smaller pieces for embedding.
    
    Args:
        doc: Parsed document from parse_markdown_file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        chunk_by_section: If True, respect section boundaries
    
    Returns:
        List of chunks with metadata
    """
    chunks = []
    
    if chunk_by_section:
        # Chunk by sections first
        for section in doc['sections']:
            section_text = '\n'.join(section['content'])
            section_chunks = chunk_text(
                section_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            for i, chunk_content in enumerate(section_chunks):
                chunks.append({
                    'content': normalize_thai_text(chunk_content),
                    'metadata': {
                        **doc['metadata'],
                        'filename': doc['filename'],
                        'filepath': doc['filepath'],
                        'section': section['title'],
                        'chunk_id': i
                    }
                })
    else:
        # Chunk entire document
        full_text = doc['content']
        text_chunks = chunk_text(
            full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        for i, chunk_content in enumerate(text_chunks):
            chunks.append({
                'content': normalize_thai_text(chunk_content),
                'metadata': {
                    **doc['metadata'],
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': i
                }
            })
    
    return chunks


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Respects sentence boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings (Thai: . ! ? | English: . ! ?)
            sentence_endings = ['.', '!', '?', '।', '๏']
            best_end = end
            
            # Search backwards for sentence ending
            for i in range(end, max(start, end - 200), -1):
                if text[i] in sentence_endings:
                    best_end = i + 1
                    break
            
            end = best_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - chunk_overlap
        
        # Prevent infinite loop
        if start <= 0 or start >= len(text):
            start = end
    
    return chunks


def load_knowledge_base(kb_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse all markdown files in the knowledge base.
    
    Args:
        kb_path: Path to knowledge_base directory
    
    Returns:
        List of parsed documents
    """
    documents = []
    kb_path = Path(kb_path)
    
    # Walk through all subdirectories
    for subdir in ['products', 'policies', 'store_info']:
        subdir_path = kb_path / subdir
        if not subdir_path.exists():
            continue
        
        for file_path in subdir_path.glob('*.md'):
            try:
                doc = parse_markdown_file(str(file_path))
                doc['metadata']['category_folder'] = subdir
                documents.append(doc)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
    
    return documents


def prepare_chunks(kb_path: str, 
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Load knowledge base and prepare all chunks.
    
    Args:
        kb_path: Path to knowledge_base directory
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of all chunks ready for embedding
    """
    documents = load_knowledge_base(kb_path)
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_by_section=True
        )
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == '__main__':
    # Test preprocessing
    import json
    
    kb_path = 'data/knowledge_base'
    chunks = prepare_chunks(kb_path)
    
    print(f"Total chunks: {len(chunks)}")
    print(f"\nSample chunk:")
    print(json.dumps(chunks[0], indent=2, ensure_ascii=False))
