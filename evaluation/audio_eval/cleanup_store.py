"""
Clean up the Qdrant vector store between evaluation runs.

Two modes:

  --from-files  (default) — parse result JSON files, collect all user_ids that
                were created during those runs, and delete them one by one via
                the mem0 Memory API.  Safe: only removes what those runs wrote.

  --nuke        — drop the entire Qdrant collection and recreate it empty.
                Use before a fresh start so there is zero risk of stale data
                polluting the next run.

Usage:
    # Delete memories created by specific result files
    python -m audio_eval.cleanup_store \\
        results/audio_eval/run_0_100_final.json \\
        results/audio_eval/run_100_500_final.json

    # Wipe the whole collection (nuclear reset)
    python -m audio_eval.cleanup_store --nuke

    # Dry run — show what would be deleted without actually deleting
    python -m audio_eval.cleanup_store --dry-run results/audio_eval/run_0_100_final.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_user_ids(result_files: List[str]) -> List[str]:
    """Extract every unique user_id stored in the given result JSON files."""
    user_ids = []
    seen = set()
    for path in result_files:
        p = Path(path)
        if not p.exists():
            logger.warning(f"File not found, skipping: {path}")
            continue
        with open(p) as f:
            data = json.load(f)
        for r in data.get("results", []):
            uid = r.get("user_id")
            if uid and uid not in seen:
                user_ids.append(uid)
                seen.add(uid)
        logger.info(f"  {path}: found {len(data.get('results', []))} results")
    return user_ids


# ---------------------------------------------------------------------------
# Delete by user_id (safe mode)
# ---------------------------------------------------------------------------

def delete_by_user_ids(result_files: List[str], dry_run: bool = False):
    """Delete only the memories that belong to the given result files."""
    user_ids = _collect_user_ids(result_files)
    if not user_ids:
        logger.warning("No user_ids found in the supplied result files. Nothing to delete.")
        return

    logger.info(f"Found {len(user_ids)} unique user_ids to clean up")

    if dry_run:
        logger.info("[DRY RUN] Would delete memories for the following user_ids:")
        for uid in user_ids:
            print(f"  {uid}")
        return

    from mem0 import Memory
    from . import config

    mem0_config = config.get_mem0_config()
    memory = Memory.from_config(mem0_config)

    deleted = 0
    failed = 0
    for uid in user_ids:
        try:
            memory.delete_all(user_id=uid)
            deleted += 1
            logger.debug(f"  Deleted memories for user_id: {uid}")
        except Exception as e:
            logger.warning(f"  Failed to delete {uid}: {e}")
            failed += 1

    logger.info(f"Done. Deleted: {deleted}, Failed: {failed}")


# ---------------------------------------------------------------------------
# Nuclear reset (whole collection)
# ---------------------------------------------------------------------------

def nuke_collection(dry_run: bool = False, qdrant_path: Optional[str] = None):
    """Drop and recreate the entire Qdrant collection."""
    from . import config

    collection = config.COLLECTION_NAME
    qdrant_path = qdrant_path or config.QDRANT_PATH

    logger.info(f"Target collection : {collection}")
    logger.info(f"Qdrant local path : {qdrant_path}")

    if dry_run:
        logger.info("[DRY RUN] Would drop and recreate collection — no changes made.")
        return

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=qdrant_path)
        existing = [c.name for c in client.get_collections().collections]

        if collection not in existing:
            logger.info(f"Collection '{collection}' does not exist — nothing to delete.")
            return

        client.delete_collection(collection_name=collection)
        logger.info(f"Collection '{collection}' dropped successfully.")
        logger.info(
            "The collection will be recreated automatically on the next evaluation run."
        )

    except ImportError:
        logger.error("qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to drop collection: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean up the Qdrant vector store between evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        metavar="FILE",
        help="Result JSON files whose user_ids should be deleted (used without --nuke)",
    )
    parser.add_argument(
        "--nuke",
        action="store_true",
        default=False,
        help="Drop the entire Qdrant collection (full reset, ignores result files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting anything",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help="Qdrant storage path to clean (default: from config.py). Must match the path used during the run.",
    )

    args = parser.parse_args()

    if args.nuke:
        nuke_collection(dry_run=args.dry_run, qdrant_path=args.qdrant_path)
    elif args.result_files:
        delete_by_user_ids(args.result_files, dry_run=args.dry_run)
    else:
        parser.print_help()
        print(
            "\nError: provide result JSON files or use --nuke for a full reset.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
