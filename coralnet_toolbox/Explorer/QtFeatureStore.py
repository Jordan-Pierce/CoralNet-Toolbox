import os
import glob
import sqlite3
import warnings

import faiss

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class FeatureStore:
    """
    Manages storing and retrieving annotation features for MULTIPLE models
    using a single SQLite database and multiple, model-specific FAISS indexes.
    """
    def __init__(self, db_path='feature_store.db', index_path_base='features'):
        self.db_path = db_path
        self.index_path_base = index_path_base  # Base name for index files, e.g., 'features'
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

        # A dictionary to hold multiple FAISS indexes, keyed by model_key
        self.faiss_indexes = {}

    def _create_table(self):
        """Create the metadata table if it doesn't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                annotation_id TEXT NOT NULL,
                model_key TEXT NOT NULL,
                faiss_index INTEGER NOT NULL,
                PRIMARY KEY (annotation_id, model_key)
            )
        ''')
        self.conn.commit()

    def _get_or_load_index(self, model_key):
        """
        Retrieves an index from memory or loads it from disk if it exists.
        Returns the index object or None if not found in memory or on disk.
        """
        # 1. Check if the index is already loaded in memory
        if model_key in self.faiss_indexes:
            return self.faiss_indexes[model_key]

        # 2. If not in memory, check for a corresponding file on disk
        index_path = f"{self.index_path_base}_{model_key}.faiss"
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            self.faiss_indexes[model_key] = index  # Cache it in memory
            return index

        # 3. If not in memory or on disk, return None
        return None

    def add_features(self, data_items, features, model_key):
        """
        Adds new features to the store for a specific model.
        """
        if not len(features):
            return

        # Get the specific index for this model, loading it if necessary
        index = self._get_or_load_index(model_key)

        # If no index exists yet, create one
        if index is None:
            feature_dim = features.shape[1]
            print(f"Creating new FAISS index for model '{model_key}' with dimension {feature_dim}.")
            index = faiss.IndexFlatL2(feature_dim)
            self.faiss_indexes[model_key] = index

        # Add vectors to the specific FAISS index
        start_index = index.ntotal
        index.add(features.astype('float32'))

        # Add metadata to SQLite. The table already supports multiple models.
        for i, item in enumerate(data_items):
            faiss_row_index = start_index + i
            self.cursor.execute(
                "INSERT OR REPLACE INTO features (annotation_id, model_key, faiss_index) VALUES (?, ?, ?)",
                (item.annotation.id, model_key, faiss_row_index)
            )
        self.conn.commit()
        self.save_faiss_index(model_key)  # Save the specific index that was modified

    def get_features(self, data_items, model_key):
        """
        Retrieves features for given data items and a specific model.
        """
        # Get the specific index for this model
        index = self._get_or_load_index(model_key)

        if index is None:
            # No features have ever been stored for this model
            return {}, data_items

        found_features = {}
        not_found_items = []

        ids_to_query = [item.annotation.id for item in data_items]

        # Query SQLite for the given model_key
        placeholders = ','.join('?' for _ in ids_to_query)
        query = (f"SELECT annotation_id, faiss_index FROM features "
                 f"WHERE model_key=? AND annotation_id IN ({placeholders})")
        params = [model_key] + ids_to_query
        self.cursor.execute(query, params)

        faiss_map = {ann_id: faiss_idx for ann_id, faiss_idx in self.cursor.fetchall()}

        if not faiss_map:
            return {}, data_items

        # Reconstruct vectors from the correct FAISS index
        faiss_indices = list(faiss_map.values())
        retrieved_vectors = index.reconstruct_batch(faiss_indices)

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
        self.cursor.execute(query, (model_key,))
        return {faiss_idx: ann_id for faiss_idx, ann_id in self.cursor.fetchall()}

    def save_faiss_index(self, model_key):
        """Saves a specific FAISS index to disk."""
        if model_key in self.faiss_indexes:
            index_to_save = self.faiss_indexes[model_key]
            index_path = f"{self.index_path_base}_{model_key}.faiss"
            print(f"Saving FAISS index for '{model_key}' to {index_path}")
            faiss.write_index(index_to_save, index_path)

    def close(self):
        """Closes the database connection."""
        self.conn.close()

    def delete_storage(self):
        """
        Closes connection and deletes the DB and ALL FAISS index files.
        """
        self.close()

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