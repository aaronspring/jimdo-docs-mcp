#!/usr/bin/env python3
"""
Generate embeddings for Jimdo docs and upload them to Qdrant.

Usage (with uv):
  uv run python generate_and_upload_embeddings.py --recreate
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from tqdm.auto import tqdm
from fastembed import SparseTextEmbedding


# Constants
DATA_FILE = "jimdo_docs.csv"
RETRIEVAL_COLUMN = "page_content"
DOCUMENT_METADATA_COLUMNS = ["url", "breadcrumbs_path", "lastmod"]

# Load environment variables early so Qdrant config is available
load_dotenv(dotenv_path=".envrc")

# Model configurations for FastEmbed - dense vectors
DENSE_MODEL_CONFIGS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "dimensions": 384,
        "max_length": 512,
        "batch_size": 32,
    },
    "jinaai/jina-embeddings-v2-base-de": {
        "dimensions": 768,
        "max_length": 8192,
        "batch_size": 2,
    },
}

# Sparse embedding models
SPARSE_MODEL_CONFIGS = {
    "Qdrant/bm25": {
        "requires_idf": True,
    },
    "Qdrant/bm42-all-minilm-l6-v2-attentions": {
        "requires_idf": True,
    },
}


def get_vector_name(embedding_model_name: str) -> str:
    return f"fast-{embedding_model_name.split('/')[-1].lower()}"


def load_data(data_file: str, n_documents: Optional[int] = None) -> pd.DataFrame:
    print(f"Loading data from {data_file} ...")
    df = pd.read_csv(data_file)
    if n_documents:
        df = df.head(n_documents)
        print(f"Using first {n_documents} documents")
    # Basic cleanup
    df = df.drop_duplicates(subset=["url"], keep="first")
    df[RETRIEVAL_COLUMN] = df[RETRIEVAL_COLUMN].fillna("")
    print(f"Loaded {len(df)} rows after dropping duplicates")
    return df


def parse_metadata_cell(cell: str) -> dict:
    """Parse the `metadata` column (JSON string) into a dict."""
    if not isinstance(cell, str):
        return {}
    try:
        parsed = json.loads(cell)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Expand metadata JSON into useful columns."""
    df = df.copy()
    expanded_rows = []
    for _, row in df.iterrows():
        meta = parse_metadata_cell(row.get("metadata", ""))
        breadcrumbs_path = meta.get("breadcrumbs_path")
        lastmod = meta.get("lastmod")
        expanded_rows.append(
            {
                "url": row.get("url"),
                "page_content": row.get(RETRIEVAL_COLUMN, ""),
                "breadcrumbs_path": breadcrumbs_path,
                "lastmod": lastmod,
                "raw_metadata": meta,
            }
        )
    return pd.DataFrame(expanded_rows)


def expand_content_to_chunks(
    df: pd.DataFrame, chunk_size: int = 750, chunk_overlap: int = 100, mode: str = "recursive"
) -> pd.DataFrame:
    """Split content into chunks."""
    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        if mode == "recursive"
        else TokenTextSplitter(
            encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    )

    rows = []
    for idx, row in df.iterrows():
        chunks = splitter.split_text(str(row[RETRIEVAL_COLUMN]))
        for chunk_idx, chunk in enumerate(chunks):
            rows.append(
                {
                    **row,
                    RETRIEVAL_COLUMN: chunk,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "original_row_index": idx,
                }
            )
    print(f"Expanded {len(df)} rows into {len(rows)} chunk rows")
    return pd.DataFrame(rows)


def create_documents(df: pd.DataFrame, metadata_columns: List[str]) -> List[Document]:
    documents: List[Document] = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating documents"):
        metadata = {}
        for column in metadata_columns:
            value = row.get(column)
            if pd.isna(value):
                metadata[column] = None
            else:
                metadata[column] = value
        raw_metadata = row.get("raw_metadata") or {}
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        documents.append(Document(page_content=str(row[RETRIEVAL_COLUMN]), metadata=metadata))
    print(f"{len(documents)} documents loaded")
    return documents


def create_qdrant_collection(collection_name: str, client: QdrantClient, recreate: bool = True) -> None:
    existing = [col.name for col in client.get_collections().collections]
    if collection_name in existing:
        if recreate:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"Collection {collection_name} already exists (use --recreate to overwrite)")
            return

    vectors_config = {}
    for model_name, config in DENSE_MODEL_CONFIGS.items():
        vector_name = get_vector_name(model_name)
        vectors_config[vector_name] = VectorParams(size=config["dimensions"], distance=Distance.COSINE)
        print(f"  Dense: {vector_name} ({config['dimensions']}d)")

    sparse_vectors_config = {}
    for model_name, config in SPARSE_MODEL_CONFIGS.items():
        vector_name = get_vector_name(model_name)
        params = SparseVectorParams(index=SparseIndexParams())
        if config.get("requires_idf"):
            params.modifier = Modifier.IDF
        sparse_vectors_config[vector_name] = params
        print(f"  Sparse: {vector_name} (IDF={config.get('requires_idf', False)})")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def upload_to_qdrant(
    df: pd.DataFrame, collection_name: str, recreate: bool = True, batch_size: int = 50
) -> None:
    print("Connecting to Qdrant ...")
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    client.get_collections()
    print(f"Connected to {os.getenv('QDRANT_URL')}")

    create_qdrant_collection(collection_name, client, recreate)

    metadata_columns = DOCUMENT_METADATA_COLUMNS.copy()
    if "chunk_index" in df.columns:
        metadata_columns.append("chunk_index")

    total_texts = len(df)

    # Dense models
    print("\nLoading dense embedding models ...")
    dense_models = {}
    for model_name, config in DENSE_MODEL_CONFIGS.items():
        print(f"  Loading {model_name}")
        dense_models[model_name] = FastEmbedEmbeddings(
            model_name=model_name,
            max_length=config["max_length"],
            batch_size=config.get("batch_size", 32),
        )

    # Sparse models
    print("\nLoading sparse embedding models ...")
    sparse_models = {}
    from qdrant_client.http.models import SparseVector as SparseVectorModel
    for model_name in SPARSE_MODEL_CONFIGS.keys():
        print(f"  Loading {model_name}")
        if model_name == "Qdrant/bm25":
            avg_len = df[RETRIEVAL_COLUMN].str.split().str.len().mean()
            sparse_models[model_name] = SparseTextEmbedding(
                model_name=model_name, language="german", avg_len=avg_len
            )
        else:
            sparse_models[model_name] = SparseTextEmbedding(model_name=model_name)

    print(f"\nUploading {total_texts} documents to '{collection_name}' in batches of {batch_size} ...")
    for start in tqdm(range(0, total_texts, batch_size), desc="Uploading"):
        end = min(start + batch_size, total_texts)
        batch_df = df.iloc[start:end]
        batch_texts = batch_df[RETRIEVAL_COLUMN].tolist()

        dense_vectors = {}
        for model_name, model in dense_models.items():
            vector_name = get_vector_name(model_name)
            dense_vectors[vector_name] = model.embed_documents(batch_texts)

        sparse_vectors = {}
        for model_name, model in sparse_models.items():
            vector_name = get_vector_name(model_name)
            vectors = []
            for embedding in model.embed(batch_texts):
                vectors.append(
                    SparseVectorModel(
                        indices=[int(idx) for idx in embedding.indices],
                        values=[float(val) for val in embedding.values],
                    )
                )
            sparse_vectors[vector_name] = vectors

        batch_ids = [str(uuid4()) for _ in range(len(batch_texts))]
        points = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            metadata = {}
            for column in metadata_columns:
                value = row.get(column)
                if pd.isna(value):
                    metadata[column] = None
                else:
                    if column == "lastmod" and isinstance(value, str):
                        try:
                            metadata[column] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        except Exception:
                            metadata[column] = value
                    else:
                        metadata[column] = value

            raw_metadata = row.get("raw_metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)

            vector_payloads = {}
            for vector_name, embeddings in dense_vectors.items():
                vector_payloads[vector_name] = embeddings[idx]
            for vector_name, embeddings in sparse_vectors.items():
                vector_payloads[vector_name] = embeddings[idx]

            points.append(
                PointStruct(
                    id=batch_ids[idx],
                    vector=vector_payloads,
                    payload={"document": batch_texts[idx], "metadata": metadata},
                )
            )

        client.upsert(collection_name=collection_name, points=points)

    print("Upload complete.")
    info = client.get_collection(collection_name)
    print(f"Collection now has {info.points_count} points.")


def process_and_upload_embeddings(
    document_splitting: Optional[str],
    n_documents: Optional[int],
    data_file: str,
    recreate: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    df = load_data(data_file, n_documents)
    df = preprocess_metadata(df)

    if document_splitting == "recursive":
        df = expand_content_to_chunks(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode="recursive")
    elif document_splitting == "tokens":
        df = expand_content_to_chunks(df, chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode="tokens")

    collection_name = "jimdo_docs"
    if document_splitting in ("recursive", "tokens"):
        collection_name += f"_{document_splitting}-{chunk_size}-{chunk_overlap}"
    if n_documents is not None:
        collection_name += "_test"

    upload_to_qdrant(df, collection_name, recreate=recreate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and upload embeddings for Jimdo docs to Qdrant")
    parser.add_argument("--document-splitting", choices=["none", "recursive", "tokens"], default="recursive")
    parser.add_argument("--n-documents", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--data-file", type=str, default=DATA_FILE, help="Path to CSV")
    parser.add_argument("--chunk-size", type=int, default=750)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--list-collections", action="store_true", help="List existing collections and exit")
    args = parser.parse_args()

    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    if args.list_collections:
        collections = client.get_collections().collections
        print("Existing collections:")
        for col in collections:
            info = client.get_collection(col.name)
            print(f"  - {col.name}: {info.points_count} points")
        return

    document_splitting = None if args.document_splitting == "none" else args.document_splitting
    print(
        f"Config: splitting={document_splitting or 'none'}, chunk_size={args.chunk_size}, "
        f"chunk_overlap={args.chunk_overlap}, recreate={args.recreate}"
    )

    process_and_upload_embeddings(
        document_splitting=document_splitting,
        n_documents=args.n_documents,
        data_file=args.data_file,
        recreate=args.recreate,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
