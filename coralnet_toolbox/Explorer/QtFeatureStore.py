import warnings

import os

import faiss
import sqlite3

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------------------------------------------------
# Viewers
# ----------------------------------------------------------------------------------------------------------------------


class FeatureStore:
    """
    Manages storing and retrieving annotation features using SQLite and FAISS.
    """
    def __init__(self, db_path='feature_store.db', index_path='features.faiss'):
        self.db_path = db_path
        self.index_path = index_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

        self.faiss_index = None
        self.feature_dim = -1
        self._load_faiss_index()

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

    def _load_faiss_index(self):
        """Load the FAISS index from disk if it exists."""
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
            self.faiss_index = faiss.read_index(self.index_path)
            self.feature_dim = self.faiss_index.d
        else:
            print("No FAISS index found. A new one will be created upon adding features.")

    def add_features(self, data_items, features, model_key):
        """
        Adds new features to the store.

        Args:
            data_items (list[AnnotationDataItem]): The data items whose features were computed.
            features (np.ndarray): The computed feature vectors.
            model_key (str): A unique identifier for the model used (e.g., 'yolo_v8_embed').
        """
        if not len(features):
            return

        # Initialize FAISS index if it's the first time adding data
        if self.faiss_index is None:
            self.feature_dim = features.shape[1]
            # Using IndexFlatL2, a simple baseline. It stores the full vectors.
            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)

        # Add vectors to FAISS
        start_index = self.faiss_index.ntotal
        self.faiss_index.add(features.astype('float32'))

        # Add metadata to SQLite
        for i, item in enumerate(data_items):
            faiss_row_index = start_index + i
            self.cursor.execute(
                "INSERT OR REPLACE INTO features (annotation_id, model_key, faiss_index) VALUES (?, ?, ?)",
                (item.annotation.id, model_key, faiss_row_index)
            )
        self.conn.commit()
        self.save_faiss_index()  # Save after every addition for robustness

    def get_features(self, data_items, model_key):
        """
        Retrieves features for given data items and a model.

        Returns:
            A tuple: (found_features, not_found_items)
            - found_features (dict): {annotation_id: feature_vector}
            - not_found_items (list): List of AnnotationDataItems for which features were not found.
        """
        if self.faiss_index is None:
            return {}, data_items  # Nothing is cached yet

        found_features = {}
        not_found_items = []
        
        ids_to_query = [item.annotation.id for item in data_items]
        
        # Query SQLite in a single batch
        placeholders = ','.join('?' for _ in ids_to_query)
        query = (
            "SELECT annotation_id, faiss_index "
            "FROM features "
            "WHERE model_key=? AND annotation_id IN ("
            f"{placeholders}"
            ")"
        )
        params = [model_key] + ids_to_query
        self.cursor.execute(query, params)
        
        # Map faiss_index to annotation_id
        faiss_map = {ann_id: faiss_idx for ann_id, faiss_idx in self.cursor.fetchall()}
        
        if not faiss_map:
            return {}, data_items
        
        # Reconstruct vectors from FAISS
        faiss_indices = list(faiss_map.values())
        retrieved_vectors = self.faiss_index.reconstruct_batch(faiss_indices)

        # Create the final dictionary of found features
        id_to_vector = {ann_id: retrieved_vectors[i] for i, ann_id in enumerate(faiss_map.keys())}
        
        # Separate found from not found
        for item in data_items:
            ann_id = item.annotation.id
            if ann_id in id_to_vector:
                found_features[ann_id] = id_to_vector[ann_id]
            else:
                not_found_items.append(item)

        return found_features, not_found_items

    def save_faiss_index(self):
        """Saves the current FAISS index to disk."""
        if self.faiss_index:
            print(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.faiss_index, self.index_path)

    def close(self):
        """Closes the database connection."""
        self.conn.close()

    def delete_storage(self):
        """
        Closes the connection and deletes the database and FAISS index files.
        """
        # First, ensure the connection is closed to release any file locks
        self.close()

        # Delete the SQLite database file
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                print(f"Deleted feature database: {self.db_path}")
            except OSError as e:
                print(f"Error removing database file {self.db_path}: {e}")

        # Delete the FAISS index file
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                print(f"Deleted FAISS index: {self.index_path}")
            except OSError as e:
                print(f"Error removing index file {self.index_path}: {e}")