import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import docx2txt
from pathlib import Path
from langchain.schema import Document
from .services import vector_store
from datetime import datetime
import hashlib
from .config_loader import get_config
from .logger_setup import setup_logger
## want this to be a separate layer for data ingestion into the vector db - chromaDB
## a function that takes multi-file input and stores them in the vector db
logger = setup_logger()

# Load configuration
config = get_config()
doc_processing_config = config.get_section('document_processing')

def create_enhanced_metadata(file_path: Path, chunk_index: int, total_chunks: int, word_count: int, char_count: int, page_number: int = None, tenant_id: str = "default", access_roles: list = None, document_visibility: str = "Public") -> dict:
    """
    Create comprehensive metadata for document chunks to enable effective retrieval scoring

    Args:
        file_path: Path to the source file
        chunk_index: Index of this chunk within the document (0-based)
        total_chunks: Total number of chunks for this document
        word_count: Number of words in the chunk
        char_count: Number of characters in the chunk
        page_number: Page number if applicable (for PDFs)
        tenant_id: Unique identifier for tenant (default: "default")
        access_roles: List of roles that can access this document (default: ["customer"])
        document_visibility: Document visibility level (default: "Public")

    Returns:
        dict: Enhanced metadata for scoring during retrieval and tenant filtering
    """
    file_stats = file_path.stat()

    # Set default access roles if not provided
    if access_roles is None:
        access_roles = ["customer"]

    # Basic file information
    metadata = {
        # File identification
        "source": str(file_path),
        "filename": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "file_size_bytes": file_stats.st_size,

        # Multi-tenant and RBAC metadata
        "tenant_id": tenant_id,
        "document_visibility": document_visibility,

        # Temporal information for recency scoring
        "ingestion_timestamp": datetime.now().isoformat(),
        "file_modified_timestamp": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "file_created_timestamp": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),

        # Document structure for position-based scoring
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_position_ratio": chunk_index / max(total_chunks - 1, 1),  # 0.0 to 1.0

        # Content quality metrics
        "word_count": word_count,
        "char_count": char_count,
        "content_density": word_count / max(char_count, 1),  # words per character

        # Document type for format-based scoring
        "document_type": get_document_type(file_path.suffix.lower()),

        # Quality indicators
        "is_first_chunk": chunk_index == 0,
        "is_last_chunk": chunk_index == total_chunks - 1,
        "relative_chunk_size": char_count,  # Will be used for size-based scoring
    }

    # Add boolean fields for each access role (denormalized approach for ChromaDB compatibility)
    for role in access_roles:
        metadata[f"access_role_{role}"] = True

    # Add page number for PDFs
    if page_number is not None:
        metadata["page_number"] = page_number
        metadata["is_first_page"] = page_number == 1

    return metadata

def get_document_type(file_extension: str) -> str:
    """
    Determine document type for quality scoring
    """
    type_mapping = {
        '.pdf': 'formatted_document',
        '.docx': 'formatted_document',
        '.txt': 'plain_text',
        '.md': 'structured_text'
    }
    return type_mapping.get(file_extension, 'unknown')

def get_supported_extensions():
    """
    Get supported file extensions from config
    """
    return doc_processing_config.get('supported_extensions', ['.pdf', '.docx', '.txt', '.md'])

## ------Extraction processors--------
def extract_docx(file_path) -> list:
    """Extract text from DOCX files and return as Document list with enhanced metadata"""
    text = docx2txt.process(file_path)
    file_path_obj = Path(file_path)

    # Create enhanced metadata for single document extraction
    metadata = create_enhanced_metadata(
        file_path=file_path_obj,
        chunk_index=0,
        total_chunks=1,
        word_count=len(text.split()),
        char_count=len(text)
    )

    document = Document(page_content=text, metadata=metadata)
    return [document]

def extract_pdf(file_path) -> list:
    """Extract text from PDF files and return as Document list with enhanced metadata"""
    pdf_loader = PyPDFLoader(file_path)
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")

        # Enhance metadata for each page with position and page information
        file_path_obj = Path(file_path)
        enhanced_pages = []

        for idx, page in enumerate(pages):
            # Get original page number from metadata if available
            original_page_num = page.metadata.get('page', idx + 1)

            # Create enhanced metadata for this page
            enhanced_metadata = create_enhanced_metadata(
                file_path=file_path_obj,
                chunk_index=idx,
                total_chunks=len(pages),
                word_count=len(page.page_content.split()),
                char_count=len(page.page_content),
                page_number=original_page_num
            )

            # Preserve any existing metadata and merge with enhanced metadata
            enhanced_metadata.update(page.metadata)

            enhanced_page = Document(
                page_content=page.page_content,
                metadata=enhanced_metadata
            )
            enhanced_pages.append(enhanced_page)

        return enhanced_pages
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def extract_txt(file_path) -> list:
    """Extract text from TXT and MD files and return as Document list with enhanced metadata"""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Enhance metadata for text documents
    file_path_obj = Path(file_path)
    enhanced_documents = []

    for idx, doc in enumerate(documents):
        enhanced_metadata = create_enhanced_metadata(
            file_path=file_path_obj,
            chunk_index=idx,
            total_chunks=len(documents),
            word_count=len(doc.page_content.split()),
            char_count=len(doc.page_content)
        )

        # Preserve any existing metadata and merge with enhanced metadata
        enhanced_metadata.update(doc.metadata)

        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=enhanced_metadata
        )
        enhanced_documents.append(enhanced_doc)

    return enhanced_documents

def ingest_file_with_feedback(file_path: str, original_file_name: str = None, tenant_id: str = "rentomojo", access_roles: list = None, document_visibility: str = "Public") -> dict:
    """Modified version of file ingestion that returns detailed status for UI with tenant support"""
    try:
        file_path = Path(file_path)
        file_name = original_file_name if original_file_name else file_path.name
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"success": False, "message": f"File not found: {file_name}", "file_name": file_name}
        
        # Get supported file processors mapping from config
        supported_extensions = get_supported_extensions()
        supported_types = {}
        for ext in supported_extensions:
            if ext == '.pdf':
                supported_types[ext] = extract_pdf
            elif ext == '.docx':
                supported_types[ext] = extract_docx
            elif ext in ['.txt', '.md']:
                supported_types[ext] = extract_txt
        
        file_extension = file_path.suffix.lower()
        
        # Check if file extension is supported
        if file_extension not in supported_types:
            return {"success": False, "message": f"Unsupported file type: {file_extension}", "file_name": file_name}
        
        # Process file based on type
        processor = supported_types[file_extension]
        file_content = processor(str(file_path))
        
        if not file_content:
            return {"success": False, "message": f"No content extracted from file", "file_name": file_name}
        
        # Chunking Process initiate using config values
        chunking_config = doc_processing_config.get('chunking', {})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get('chunk_size', 1000),
            chunk_overlap=chunking_config.get('chunk_overlap', 200)
        )

        pages_split = text_splitter.split_documents(file_content)

        # Update metadata for chunked documents with proper chunk indexing and tenant information
        enhanced_chunks = []
        for chunk_idx, chunk in enumerate(pages_split):
            # Create enhanced metadata for this chunk with tenant information
            enhanced_metadata = create_enhanced_metadata(
                file_path=file_path,
                chunk_index=chunk_idx,
                total_chunks=len(pages_split),
                word_count=len(chunk.page_content.split()),
                char_count=len(chunk.page_content),
                page_number=chunk.metadata.get('page_number'),  # Preserve page number if exists
                tenant_id=tenant_id,
                access_roles=access_roles,
                document_visibility=document_visibility
            )

            # Preserve any existing metadata and merge with enhanced metadata
            original_metadata = chunk.metadata.copy()
            original_metadata.update(enhanced_metadata)

            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=original_metadata
            )
            enhanced_chunks.append(enhanced_chunk)

            # Log ingestion metadata before storing
            logger.info(f"Ingesting file into vector DB - File: {file_name}, "
                       f"Access Roles: {access_roles or ['customer']}, "
                       f"File Type: {file_extension}, "
                       f"Tenant ID: {tenant_id}, "
                       f"Document Visibility: {document_visibility}, "
                       f"Chunks: {len(pages_split)}")
            
            # Store in vector DB with enhanced metadata
            vector_store.add_documents(documents=enhanced_chunks)
            
            return {"success": True, "message": f"Successfully processed {len(pages_split)} chunks", "file_name": file_name}
        
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}", "file_name": file_path.name if file_path else "unknown"}

## ----------main ingestion---------
def ingest_file_to_vectordb(file_paths, tenant_id: str = "rentomojo", access_roles: list = None, document_visibility: str = "Public") -> None:
    """
    Main function to ingest one or multiple files into ChromaDB vector store
    Supports: PDF, DOCX, TXT, MD file extensions with multi-tenant support

    Args:
        file_paths (str or list): Path(s) to the file(s) to ingest
        tenant_id (str): Unique identifier for tenant (default: "rentomojo")
        access_roles (list): List of roles that can access documents (default: ["customer"])
        document_visibility (str): Document visibility level (default: "Public")

    Note:
        Skips unsupported or missing files and continues processing others
    """
    # Get supported file processors mapping from config
    supported_extensions = get_supported_extensions()
    supported_types = {}
    for ext in supported_extensions:
        if ext == '.pdf':
            supported_types[ext] = extract_pdf
        elif ext == '.docx':
            supported_types[ext] = extract_docx
        elif ext in ['.txt', '.md']:
            supported_types[ext] = extract_txt
    
    # Convert single file path to list for uniform processing
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    successful_files = []
    
    for file_path in file_paths:
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Skipping: File not found - {file_path}")
                continue
            
            file_extension = file_path.suffix.lower()
            
            # Check if file extension is supported
            if file_extension not in supported_types:
                print(f"Skipping: Unsupported file type - {file_path}")
                continue
            
            # Process file based on type
            processor = supported_types[file_extension]
            file_content = processor(str(file_path))
            
            if not file_content:
                logger.warning(f"No content extracted from: {file_path}")
                continue
            
            # Chunking Process initiate using config values
            chunking_config = doc_processing_config.get('chunking', {})
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get('chunk_size', 1000),
                chunk_overlap=chunking_config.get('chunk_overlap', 200)
            )

            pages_split = text_splitter.split_documents(file_content)

            # Update metadata for chunked documents with proper chunk indexing and tenant information
            enhanced_chunks = []
            for chunk_idx, chunk in enumerate(pages_split):
                # Create enhanced metadata for this chunk with tenant information
                enhanced_metadata = create_enhanced_metadata(
                    file_path=file_path,
                    chunk_index=chunk_idx,
                    total_chunks=len(pages_split),
                    word_count=len(chunk.page_content.split()),
                    char_count=len(chunk.page_content),
                    page_number=chunk.metadata.get('page_number'),  # Preserve page number if exists
                    tenant_id=tenant_id,
                    access_roles=access_roles,
                    document_visibility=document_visibility
                )

                # Preserve any existing metadata and merge with enhanced metadata
                original_metadata = chunk.metadata.copy()
                original_metadata.update(enhanced_metadata)

                enhanced_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=original_metadata
                )
                enhanced_chunks.append(enhanced_chunk)

            # Log ingestion metadata before storing
            logger.info(f"Ingesting file into vector DB - File: {file_path.name}, "
                       f"Access Roles: {access_roles or ['customer']}, "
                       f"File Type: {file_extension}, "
                       f"Tenant ID: {tenant_id}, "
                       f"Document Visibility: {document_visibility}, "
                       f"Chunks: {len(pages_split)}")
            
            # Store in vector DB with enhanced metadata
            vector_store.add_documents(documents=enhanced_chunks)
            print(f"Successfully ingested {file_path.name}")
            successful_files.append(file_path.name)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if successful_files:
        print(f"Total files processed: {len(successful_files)}")
    else:
        print("No files were successfully processed")
        
if __name__ == "__main__":
    file_paths_input = input("Give the file path(s) to ingest (comma-separated for multiple): ")
    file_paths = [path.strip() for path in file_paths_input.split(',')]
    ingest_file_to_vectordb(file_paths)
