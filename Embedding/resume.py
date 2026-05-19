import os
import json
import uuid
import shutil
import re
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

 
chroma_base_dir = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\chroma_db_resume"   
resume_json_folder = r"C:\Users\SibaPrasadPatra\Desktop\new_hugging_face\resume_json"    

def load_json_from_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load JSON from file {json_path}: {e}")
        return {}

def init_chromadb(persist_dir):
    try:
        return PersistentClient(path=persist_dir)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Chroma DB: {e}")
        return None

def sanitize_collection_name(name):
    """Sanitize file name to valid ChromaDB collection name."""
    name = os.path.splitext(name)[0]
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    if len(name) < 3:
        name = (name + "___")[:3]
    return name

def delete_collection_folder(collection_name):
    """Delete Chroma persisted collection data from disk."""
    collection_path = os.path.join(chroma_base_dir, "collections", collection_name)
    if os.path.exists(collection_path):
        try:
            shutil.rmtree(collection_path)
            print(f"[INFO] Deleted collection folder: {collection_path}")
        except Exception as e:
            print(f"[ERROR] Could not delete collection folder '{collection_path}': {e}")

def delete_chromadb_collection(client, collection_name):
    """Delete ChromaDB collection if it exists."""
    try:
        client.delete_collection(name=collection_name)
        print(f"[INFO] Deleted existing ChromaDB collection: {collection_name}")
    except Exception as e:
        print(f"[DEBUG] Collection '{collection_name}' did not exist or could not be deleted: {e}")

def embed_and_store_fields(data, collection_name, persist_dir):
    # Always delete the collection folder and ChromaDB collection before embedding
    delete_collection_folder(collection_name)
    client = init_chromadb(persist_dir)
    if not client:
        print("[ERROR] ChromaDB client initialization failed.")
        return False

    delete_chromadb_collection(client, collection_name)

    collection = client.get_or_create_collection(name=collection_name)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if isinstance(data, dict):
        data = [data]

    total_chunks = 0

    for idx, resume in enumerate(data):
        texts = []
        metadatas = []
        ids = []

        print(f"\n[INFO] Embedding fields for resume #{idx+1} -> Collection: {collection_name}")

        # Embed all fields present in the JSON
        for field in resume:
            content = resume.get(field)

            if content is None or (isinstance(content, str) and content.strip() == ""):
                content_str = "null"
            elif isinstance(content, dict):
                content_str = "; ".join([f"{k}: {v}" for k, v in content.items()])
            elif isinstance(content, list):
                content_str = "; ".join(map(str, content))
            else:
                content_str = str(content).strip()

            labeled_text = f"{field}: {content_str}"

            print(f" Field: {field}")
            print(f" Content: {content_str[:150]}...\n")

            texts.append(labeled_text)
            metadatas.append({"field": field})
            ids.append(str(uuid.uuid4()))

        if not texts:
            print(f"[WARNING] No valid fields to embed for resume #{idx+1}")
            continue

        embeddings = embedder.encode(texts)
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        total_chunks += len(texts)

    print(f"[INFO] Stored {total_chunks} total embedded field(s) in collection '{collection_name}'")

     
    try:
        count = collection.count()
        print(f"[SUCCESS] Collection '{collection_name}' contains {count} documents.\n")
        return count > 0
    except Exception as e:
        print(f"[ERROR] Verification failed for collection '{collection_name}': {e}")
        return False

def remove_orphan_collections(folder_path, persist_dir):
    """Remove any ChromaDB collections that do not have a corresponding JSON file."""
    client = init_chromadb(persist_dir)
    if not client:
        return
    existing_collections = set(c.name for c in client.list_collections())
    json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    valid_collections = set(sanitize_collection_name(f) for f in json_files)
    orphan_collections = existing_collections - valid_collections
    for orphan in orphan_collections:
        try:
            client.delete_collection(name=orphan)
            delete_collection_folder(orphan)
            print(f"[INFO] Removed orphan collection: {orphan}")
        except Exception as e:
            print(f"[ERROR] Could not remove orphan collection '{orphan}': {e}")

def embed_all_jsons_from_folder(folder_path, persist_dir):
    os.makedirs(persist_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    print(f"\n[INFO] Found {len(files)} JSON files in folder '{folder_path}'.")

    # Remove collections that do not have a corresponding JSON file
    remove_orphan_collections(folder_path, persist_dir)

    if not files:
        print("[WARNING] No JSON files found to embed.")
        return

    for file in files:
        json_path = os.path.join(folder_path, file)
        collection_name = sanitize_collection_name(file)

        print(f"\n[INFO] Processing file: {file} -> Collection: '{collection_name}'")

        try:
            json_data = load_json_from_file(json_path)
            success = embed_and_store_fields(json_data, collection_name=collection_name, persist_dir=persist_dir)
            if success:
                print(f"[SUCCESS] Embedding completed and verified for: {file}")
            else:
                print(f"[WARNING] Embedding failed or collection is empty for: {file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")

# if __name__ == "__main__":
#     print("\n=== Starting Resume Embedding Process ===")
#     embed_all_jsons_from_folder(resume_json_folder, persist_dir=chroma_base_dir)
#     print("=== Embedding Process Completed ===\n")
