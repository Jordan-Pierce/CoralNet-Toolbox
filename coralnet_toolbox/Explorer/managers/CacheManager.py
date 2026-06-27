import os
import glob
import sqlite3
import threading
import warnings

import faiss

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class CacheManager:
    """
    Manages storing and retrieving annotation features for MULTIPLE models
    using a single SQLite database and multiple, model-specific FAISS indexes.
    
    Cache files stored in .cache/embedding/ directory.
    """
    CACHE_SUBDIR = '.cache/embedding'
    
    def __init__(self, db_path='manager.db', index_path_base='features'):
        # Ensure cache directory exists
        os.makedirs(self.CACHE_SUBDIR, exist_ok=True)
        
        # Prepend cache directory to paths
        self.db_path = os.path.join(self.CACHE_SUBDIR, db_path)
        self.index_path_base = os.path.join(self.CACHE_SUBDIR, index_path_base)
        
        # Note: do NOT keep a long-lived sqlite3 connection here — sqlite3
        # connections are thread-affine. Open connections per-call instead.
        self._create_table()

        # FAISS index files are read-modify-written on disk and FAISS objects
        # are not thread-safe. Multiple pipeline workers (e.g. a superseded
        # worker still draining plus a freshly started one) can call into the
        # cache concurrently, which corrupts the read/add/write cycle and
        # surfaces as opaque (empty-message) FAISS exceptions. Serialize all
        # index access through a single lock.
        self._index_lock = threading.Lock()

        # We do not keep in-memory FAISS indexes to avoid sharing
        # FAISS objects across threads. Index files are read from and
        # written to disk as needed.

    def _create_table(self):
        """Create the metadata table if it doesn't exist."""
        # Use a short-lived connection to avoid cross-thread usage
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    annotation_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    faiss_index INTEGER NOT NULL,
                    PRIMARY KEY (annotation_id, model_key)
                )
            ''')
            conn.commit()

    def _get_or_load_index(self, model_key):
        """
        Retrieves an index from memory or loads it from disk if it exists.
        Returns the index object or None if not found in memory or on disk.
        """
        # Check for a corresponding file on disk and load it afresh.
        index_path = f"{self.index_path_base}_{model_key}.faiss"
        if os.path.exists(index_path):
            print(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            return index

        # If not on disk, return None
        return None

    def _drop_model_locked(self, model_key):
        """Delete all cached data for a model_key (SQLite rows + index file).

        Caller must hold self._index_lock. Used to discard a stale index whose
        dimension no longer matches the current feature definition so it can be
        rebuilt cleanly.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM features WHERE model_key = ?", (model_key,))
            conn.commit()

        index_path = f"{self.index_path_base}_{model_key}.faiss"
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
            except OSError as e:
                print(f"Error removing stale index file {index_path}: {e}")

    def add_features(self, data_items, features, model_key):
        """
        Adds new features to the store for a specific model.
        """
        if not len(features):
            return

        feature_dim = features.shape[1]

        # Serialize the whole read-modify-write so concurrent workers cannot
        # interleave on the same on-disk index (see _index_lock in __init__).
        with self._index_lock:
            # Get the specific index for this model, loading it if necessary
            # Load index from disk if it exists, otherwise create a new index
            index = self._get_or_load_index(model_key)

            # A stale on-disk index can have a different dimension than the
            # features we're adding now (e.g. the feature definition for a
            # fixed model_key like 'Color_Features' changed between versions).
            # FAISS would assert forever in that case, so rebuild from scratch.
            if index is not None and index.d != feature_dim:
                print(
                    f"FAISS index for '{model_key}' has dimension {index.d} but "
                    f"features are {feature_dim}-d; rebuilding stale index."
                )
                self._drop_model_locked(model_key)
                index = None

            if index is None:
                print(f"Creating new FAISS index for model '{model_key}' with dimension {feature_dim}.")
                index = faiss.IndexFlatL2(feature_dim)

            # Add vectors to the FAISS index
            start_index = index.ntotal
            index.add(features.astype('float32'))

            # Add metadata to SQLite using a short-lived connection
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                for i, item in enumerate(data_items):
                    faiss_row_index = start_index + i
                    cur.execute(
                        "INSERT OR REPLACE INTO features (annotation_id, model_key, faiss_index) VALUES (?, ?, ?)",
                        (item.annotation.id, model_key, faiss_row_index)
                    )
                conn.commit()

            # Persist FAISS index to disk immediately
            index_path = f"{self.index_path_base}_{model_key}.faiss"
            faiss.write_index(index, index_path)

    def get_features(self, data_items, model_key):
        """
        Retrieves features for given data items and a specific model.
        Chunks the query to avoid SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (~999).
        """
        ids_to_query = [item.annotation.id for item in data_items]
        if not ids_to_query:
            return {}, []

        # Query in chunks to avoid SQLite bind parameter limits
        max_bind_params = 900
        rows = []

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for start in range(0, len(ids_to_query), max_bind_params):
                chunk = ids_to_query[start:start + max_bind_params]
                placeholders = ','.join('?' for _ in chunk)
                query = (f"SELECT annotation_id, faiss_index FROM features "
                         f"WHERE model_key=? AND annotation_id IN ({placeholders})")
                params = [model_key] + chunk
                cur.execute(query, params)
                rows.extend(cur.fetchall())

        if not rows:
            return {}, data_items

        faiss_map = {ann_id: faiss_idx for ann_id, faiss_idx in rows}

        # Load the FAISS index fresh from disk and reconstruct vectors under the
        # lock so we never read an index file mid-write from a concurrent flush.
        with self._index_lock:
            index = self._get_or_load_index(model_key)
            if index is None:
                # Index file should exist for these rows; if not, treat as not found
                return {}, data_items

            faiss_indices = list(faiss_map.values())
            retrieved_vectors = index.reconstruct_batch(faiss_indices)

        found_features = {}
        not_found_items = []

        id_to_vector = {ann_id: retrieved_vectors[i] for i, ann_id in enumerate(faiss_map.keys())}

        for item in data_items:
            ann_id = item.annotation.id
            if ann_id in id_to_vector:
                found_features[ann_id] = id_to_vector[ann_id]
            else:
                not_found_items.append(item)

        return found_features, not_found_items

    def get_faiss_index_to_annotation_id_map(self, model_key):
        """
        Retrieves a mapping from FAISS row index to annotation_id for a given model.
        """
        query = "SELECT faiss_index, annotation_id FROM features WHERE model_key = ?"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (model_key,))
            rows = cur.fetchall()
        return {faiss_idx: ann_id for faiss_idx, ann_id in rows}

    def save_faiss_index(self, model_key):
        """Saves a specific FAISS index to disk."""
        # Indexes are written to disk immediately after modification in
        # add_features(), so this method is a no-op but kept for API
        # compatibility.
        return
            
    def remove_features_for_annotation(self, annotation_id):
        """
        Removes an annotation's feature metadata from the SQLite database.
        This effectively orphans the vector in the FAISS index, invalidating it.
        """
        self.remove_features_for_annotations([annotation_id])

    def remove_features_for_annotations(self, annotation_ids):
        """
        Removes multiple annotations' feature metadata in a single SQLite transaction.

        This is the fast path for bulk deletes. The FAISS vectors themselves are
        still left in place and become unreachable once their metadata rows are removed.
        """
        unique_ids = list(dict.fromkeys(annotation_ids))
        if not unique_ids:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()

                # SQLite limits the number of bind parameters per statement, so chunk
                # larger bulk deletes into safe batches.
                max_bind_params = 900
                for start in range(0, len(unique_ids), max_bind_params):
                    chunk = unique_ids[start:start + max_bind_params]
                    placeholders = ','.join('?' for _ in chunk)
                    cur.execute(
                        f"DELETE FROM features WHERE annotation_id IN ({placeholders})",
                        chunk,
                    )

                conn.commit()

            if len(unique_ids) == 1:
                print(f"Invalidated features for annotation_id: {unique_ids[0]}")
            else:
                print(f"Invalidated features for {len(unique_ids)} annotations")
        except sqlite3.Error as e:
            if len(unique_ids) == 1:
                print(f"Error removing feature for annotation {unique_ids[0]}: {e}")
            else:
                print(f"Error removing features for annotations {unique_ids}: {e}")

    def close(self):
        """Closes the database connection."""
        # Nothing to close for sqlite (we use short-lived connections)
        return

    def delete_storage(self):
        """
        Closes connection and deletes the DB and ALL FAISS index files.
        """
        self.close()

        with self._index_lock:
            if os.path.exists(self.db_path):
                try:
                    os.remove(self.db_path)
                    print(f"Deleted feature database: {self.db_path}")
                except OSError as e:
                    print(f"Error removing database file {self.db_path}: {e}")

            # Use glob to find and delete all matching index files
            for index_file in glob.glob(f"{self.index_path_base}_*.faiss"):
                try:
                    os.remove(index_file)
                    print(f"Deleted FAISS index: {index_file}")
                except OSError as e:
                    print(f"Error removing index file {index_file}: {e}")
