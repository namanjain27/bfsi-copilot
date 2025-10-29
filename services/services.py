
# Import SQLite3 fix BEFORE any ChromaDB imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utility run files'))

from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple
import os
from .logger_setup import setup_logger
logger = setup_logger()


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding_model = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

persist_directory = 'knowledgeBase'
db_collection_name = "general_rentomojo"

# Create directory if it doesn't exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vector_store = Chroma(
    collection_name=db_collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

def create_tenant_aware_retriever(tenant_id: str, 
                                 user_role: str, 
                                 search_kwargs: Dict[str, Any] = None,
                                 search_type: Optional[str] = None) -> VectorStoreRetriever:
    """
    Create a tenant-aware retriever with metadata filtering for multi-tenant RBAC

    Args:
        tenant_id (str): Unique identifier for the tenant
        user_role (str): User role for RBAC filtering (customer, vendor, associate, leadership, hr)
        search_kwargs (Dict[str, Any], optional): Additional search parameters
        search_type (str, optional): Search type ('similarity' or 'mmr'). If None, uses config default.

    Returns:
        VectorStoreRetriever: Configured retriever with tenant and role-based filtering
    """
    # Import config loader here to avoid circular imports
    from .config_loader import get_config
    config = get_config()
    
    # Get search type from parameter or config
    if search_type is None:
        search_type = config.get('retrieval.search_type', 'similarity')
    
    # Default search parameters
    default_search_kwargs = {
        "k": 4,  # Number of documents to retrieve
        "filter": {}
    }
    
    # Add MMR-specific parameters if using MMR search
    if search_type == 'mmr':
        diversity_lambda = config.get('retrieval.diversity_lambda', 0.5)
        default_search_kwargs['lambda_mult'] = diversity_lambda
        logger.debug(f"Using MMR search with lambda_mult={diversity_lambda}")

    # Merge with provided search_kwargs
    if search_kwargs:
        default_search_kwargs.update(search_kwargs)

    # Build metadata filter for tenant and role-based access
    metadata_filter = build_metadata_filter(tenant_id, user_role)

    # Merge tenant filter with any existing filters
    if "filter" in default_search_kwargs:
        default_search_kwargs["filter"].update(metadata_filter)
    else:
        default_search_kwargs["filter"] = metadata_filter

    logger.info(f"Created tenant-aware retriever for tenant_id: {tenant_id}, user_role: {user_role}, search_type: {search_type}")
    logger.debug(f"Retriever filter: {default_search_kwargs['filter']}")

    # Create and return the retriever with tenant-aware filtering
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=default_search_kwargs
    )

def build_metadata_filter(tenant_id: str, user_role: str) -> Dict[str, Any]:
    """
    Build metadata filter for tenant isolation and role-based access control

    Args:
        tenant_id (str): Unique identifier for the tenant
        user_role (str): User role for RBAC filtering

    Returns:
        Dict[str, Any]: Metadata filter for ChromaDB
    """
    # Tenant isolation filter - documents must belong to the tenant
    metadata_filter = {
        "tenant_id": tenant_id
    }

    # Role-based access control filter
    # Documents are accessible if:
    # 1. User role has corresponding boolean field set to True, OR
    # 2. Document visibility is "Public"

    # ChromaDB filtering using denormalized boolean fields for access roles
    # Documents accessible if access_role_{user_role} is True OR document_visibility is "Public"
    role_filter = {
        "$or": [
            {f"access_role_{user_role}": True},
            {"document_visibility": "Public"}
        ]
    }

    # Combine tenant and role filters using $and operator
    combined_filter = {
        "$and": [
            metadata_filter,
            role_filter
        ]
    }

    logger.debug(f"Built metadata filter for tenant_id: {tenant_id}, user_role: {user_role}")
    return combined_filter

def retrieve_with_scores(query: str, 
                        tenant_id: str, 
                        user_role: str,
                        search_type: Optional[str] = None,
                        k: Optional[int] = None) -> Tuple[List[Document], List[float]]:
    """
    Retrieve documents with relevance scores using tenant-aware filtering
    
    This function uses ChromaDB's similarity_search_with_relevance_scores when available,
    falling back to regular search with neutral scores if needed.
    
    Args:
        query: Search query text
        tenant_id: Unique identifier for the tenant
        user_role: User role for RBAC filtering
        search_type: Search type ('similarity' or 'mmr'). If None, uses config default.
        k: Number of documents to retrieve. If None, uses config default.
        
    Returns:
        Tuple of (documents, scores) where scores are relevance scores (0-1 range)
    """
    from .config_loader import get_config
    config = get_config()
    
    # Get parameters from config if not provided
    if search_type is None:
        search_type = config.get('retrieval.search_type', 'similarity')
    if k is None:
        k = config.get('retrieval.k', 5)
    
    # Build metadata filter
    metadata_filter = build_metadata_filter(tenant_id, user_role)
    
    try:
        # Try to use similarity search with scores
        if search_type == 'mmr':
            # MMR doesn't support scores directly in Chroma, use regular MMR
            diversity_lambda = config.get('retrieval.diversity_lambda', 0.5)
            documents = vector_store.max_marginal_relevance_search(
                query,
                k=k,
                filter=metadata_filter,
                lambda_mult=diversity_lambda
            )
            # Return neutral scores for MMR (scores not available)
            scores = [0.7] * len(documents)
            logger.debug(f"MMR search returned {len(documents)} documents (neutral scores)")
            
        else:
            # Use similarity search with scores
            docs_and_scores = vector_store.similarity_search_with_relevance_scores(
                query,
                k=k,
                filter=metadata_filter
            )
            
            if docs_and_scores:
                documents, scores = zip(*docs_and_scores)
                documents = list(documents)
                scores = list(scores)
            else:
                documents = []
                scores = []
            
            logger.debug(f"Similarity search with scores returned {len(documents)} documents")
        
        return documents, scores
        
    except Exception as e:
        logger.warning(f"Error in retrieve_with_scores, falling back to regular retrieval: {e}")
        
        # Fallback to regular retrieval without scores
        retriever = create_tenant_aware_retriever(
            tenant_id=tenant_id,
            user_role=user_role,
            search_kwargs={"k": k},
            search_type=search_type
        )
        documents = retriever.invoke(query)
        scores = [0.7] * len(documents)  # Neutral fallback scores
        
        logger.debug(f"Fallback retrieval returned {len(documents)} documents with neutral scores")
        return documents, scores


def get_vector_store_status(tenant_id: str = None) -> Dict[str, Any]:
    """
    Get vector store status with optional tenant filtering

    Args:
        tenant_id (str, optional): Filter status by tenant

    Returns:
        Dict[str, Any]: Vector store status information
    """
    try:
        # Get collection info
        collection = vector_store._collection

        if tenant_id:
            # Get tenant-specific document count
            # Note: ChromaDB doesn't have direct count with filter, so we get all tenant documents
            tenant_filter = {"tenant_id": tenant_id}
            try:
                results = collection.get(
                    where=tenant_filter
                )
                # Get actual count of tenant documents
                tenant_doc_count = len(results['ids']) if results['ids'] else 0
                has_tenant_docs = tenant_doc_count > 0

                # For tenant-specific requests, return tenant document count as main count
                status = {
                    "status": "ready" if tenant_doc_count > 0 else "not_found",
                    "document_count": tenant_doc_count,
                    "collection_name": db_collection_name,
                    "tenant_document_count": tenant_doc_count,
                    "has_tenant_documents": has_tenant_docs
                }
            except Exception as e:
                logger.warning(f"Error getting tenant-specific count: {e}")
                # Return error status for tenant-specific requests
                status = {
                    "status": "error",
                    "document_count": 0,
                    "collection_name": db_collection_name,
                    "tenant_document_count": 0,
                    "has_tenant_documents": False,
                    "error_message": f"Error accessing tenant data: {str(e)}"
                }
        else:
            # Get total collection count for general requests
            total_count = collection.count()
            status = {
                "status": "ready" if total_count > 0 else "empty",
                "document_count": total_count,
                "collection_name": db_collection_name,
                "tenant_document_count": None,
                "has_tenant_documents": None
            }

        logger.info(f"Vector store status retrieved for tenant: {tenant_id}")
        return status

    except Exception as e:
        logger.error(f"Error getting vector store status: {e}")
        return {
            "status": "error",
            "document_count": 0,
            "collection_name": db_collection_name,
            "error_message": str(e),
            "tenant_document_count": 0,
            "has_tenant_documents": False
        }
