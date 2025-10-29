from .data_ingestion import extract_pdf, extract_txt, extract_docx 
import re
import base64
import os
import tempfile
from pathlib import Path
# Image processing helper function
def process_image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string for Gemini multi-modal input
    Supports: PNG, JPG, JPEG, GIF, WEBP formats
    """
    try:
        image_path = Path(image_path)
        
        # Validate file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check supported formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        if image_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        print(f"Successfully processed image: {image_path.name}")
        return encoded_string
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def process_document_to_text(file_path: str) -> str:
    """
    Extract text content from document files using existing processors
    Supports: PDF, TXT, MD, DOCX formats
    """
    try:
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        # Map file extensions to processors
        file_extension = file_path.suffix.lower()
        processor_map = {
            '.pdf': extract_pdf,
            '.txt': extract_txt,
            '.md': extract_txt,
            '.docx': extract_docx
        }
        
        if file_extension not in processor_map:
            raise ValueError(f"Unsupported document format: {file_extension}")
        
        # Process file and extract text
        processor = processor_map[file_extension]
        documents = processor(str(file_path))
        
        if not documents:
            raise ValueError(f"No content extracted from: {file_path}")
        
        # Combine all document pages/content into single text
        combined_text = ""
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
        
        print(f"Successfully processed document: {file_path.name}")
        return combined_text.strip()
        
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
        return None

def parse_multimodal_input(user_input: str) -> tuple:
    """
    Parse user input to extract text and file paths (images and documents)
    Supports syntax: 
    - Images: 'image:/path/to/file.png' or 'img:/path/to/file.png'
    - Documents: 'pdf:/path/to/file.pdf', 'txt:/path/to/file.txt', 'md:/path/to/file.md', 'doc:/path/to/file.docx'
    Returns: (clean_text, list_of_image_paths, list_of_document_paths)
    """
    # Pattern to match image references
    image_pattern = r'(?:image?:|img:)([^\s]+)'
    
    # Pattern to match document references
    doc_pattern = r'(?:pdf:|txt:|md:|doc:)([^\s]+)'
    
    # Find all file references
    image_matches = re.findall(image_pattern, user_input, re.IGNORECASE)
    doc_matches = re.findall(doc_pattern, user_input, re.IGNORECASE)
    
    # Remove all file references from text
    clean_text = re.sub(image_pattern, '', user_input, flags=re.IGNORECASE)
    clean_text = re.sub(doc_pattern, '', clean_text, flags=re.IGNORECASE).strip()
    
    return clean_text, image_matches, doc_matches

def process_uploaded_files(uploaded_files):
    """
    Process uploaded files from Streamlit and categorize them into images and documents
    Returns dictionary with temporary file paths and attachment info
    """
    image_files = []
    doc_files = []
    attachments = []
    
    # Define supported file types
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    doc_extensions = {'.pdf', '.txt', '.md', '.docx'}
    
    for uploaded_file in uploaded_files:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        # TODO: what does getvalue() do here? in image_file isn't there data duplication?
        # TODO: are we storing these uploaded temp files? if not, do it.

        if file_extension in image_extensions:
            # Image file
            image_files.append(tmp_file_path)
            attachments.append({
                "type": "image",
                "name": uploaded_file.name,
                "data": uploaded_file.getvalue()
            })
            print(f"Processed image file: {uploaded_file.name}")
            
        elif file_extension in doc_extensions:
            # Document file
            doc_files.append(tmp_file_path)
            attachments.append({
                "type": "document", 
                "name": uploaded_file.name,
                "data": None
            })
            print(f"Processed document file: {uploaded_file.name}")
            
        else:
            # Unsupported file type - clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            print(f"Skipping unsupported file type: {uploaded_file.name}")
    
    return {
        "image_files": image_files,
        "doc_files": doc_files,
        "attachments": attachments
    }