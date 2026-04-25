"""Build the FAISS vector store for the CRS project.

Run from the src/ directory:
    python -m crs.scripts.build_index
"""
import sys
from pathlib import Path

# Ensure the src dir is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crs.config import get_settings
from crs.data.loaders import DatasetLoader
from crs.retrieval.vector_store import VectorStore
from crs.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    logger.info("Building vector store...")

    loader = DatasetLoader(settings)
    logger.info("Loaded %d movies from item_map", len(loader.item_map))

    vs = VectorStore(settings=settings)
    vs.build(loader)
    index_path, meta_path = vs.save()

    logger.info("Done! Saved index to %s and metadata to %s", index_path, meta_path)


if __name__ == "__main__":
    main()
