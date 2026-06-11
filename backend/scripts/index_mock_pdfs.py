import sys
from pathlib import Path

# Add the backend directory to python path so we can import 'app'
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BACKEND_DIR))

from app.config import get_settings
from app.indexing.store import KnowledgeBase

def index_pdfs():
    settings = get_settings()
    print(f"Loading KnowledgeBase from storage: {settings.storage_dir}")
    kb = KnowledgeBase(settings)
    
    # Path to the generated mock PDFs
    mock_pdf_dir = BACKEND_DIR.parent / "data" / "mock_conflict_pdfs" / "files"
    
    if not mock_pdf_dir.exists():
        print(f"Error: Mock PDF directory {mock_pdf_dir} does not exist. Please run generate_mock_pdfs.py first.")
        return
    
    pdf_files = list(mock_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found to ingest.")
        return

    # Delete existing documents with the same name if they exist to avoid duplicate chunks
    existing_docs = kb.repo.list_documents()
    for doc in existing_docs:
        if doc["source"] in [f.name for f in pdf_files]:
            print(f"Deleting existing document: {doc['source']} (ID: {doc['id']})")
            kb.delete_document(doc["id"])

    print("\nStarting ingestion...")
    for pdf_path in pdf_files:
        print(f"Ingesting {pdf_path.name}...")
        try:
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            result = kb.ingest_pdf(pdf_data, pdf_path.name)
            print(f"Successfully ingested {pdf_path.name}: {result}")
        except Exception as e:
            print(f"Failed to ingest {pdf_path.name}: {e}")

    print("\nRebuilding vector & BM25 indexes...")
    kb.rebuild_indexes()
    print("All mock files ingested and indexed successfully!")

if __name__ == "__main__":
    index_pdfs()
