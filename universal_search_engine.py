#!/usr/bin/env python3
"""
Universal Movie Subtitle Semantic Search Engine
Works with any subtitle database (original sample, Kaggle datasets, etc.)
"""

import os
import warnings
import sqlite3
import zipfile
import io
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

class UniversalMovieSearchEngine:
    def __init__(self, database_path=None, use_semantic=True):
        """Initialize the search engine with flexible database support"""
        self.available_databases = self.find_available_databases()
        self.db_path = database_path or self.select_database()
        self.data = []
        self.use_semantic = use_semantic
        self.vectorizer = None
        self.tfidf_matrix = None
        self.sentence_model = None
        self.semantic_embeddings = None

    def find_available_databases(self):
        """Find all available subtitle databases"""
        databases = {}

        # Check for common database files
        db_files = [
            ("eng_subtitles_database.db", "Original Sample Database (8 movies)"),
            ("kaggle_sample_database.db", "Kaggle-style Sample Database"),
            ("kaggle_subtitles_database.db", "Kaggle Subtitles Database"),
            ("real_subtitles_database.db", "Real Subtitles Database"),
            ("free_subtitles_database.db", "Free Subtitles Database")
        ]

        for db_file, description in db_files:
            if os.path.exists(db_file):
                # Check how many records
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM zipfiles")
                    count = cursor.fetchone()[0]
                    conn.close()
                    databases[db_file] = f"{description} ({count} entries)"
                except:
                    databases[db_file] = f"{description} (unknown size)"

        return databases

    def select_database(self):
        """Let user select which database to use"""
        if not self.available_databases:
            print("No subtitle databases found!")
            print("Run: python3 kaggle_data_processor.py (option 4) to create sample data")
            return None

        if len(self.available_databases) == 1:
            db_path = list(self.available_databases.keys())[0]
            print(f"📁 Using database: {self.available_databases[db_path]}")
            return db_path

        print("📁 Available Databases:")
        db_list = list(self.available_databases.items())
        for i, (db_path, description) in enumerate(db_list, 1):
            print(f"   {i}. {description}")

        while True:
            try:
                choice = input(f"\nSelect database (1-{len(db_list)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(db_list):
                    selected_db = db_list[idx][0]
                    print(f"Selected: {self.available_databases[selected_db]}")
                    return selected_db
                else:
                    print("Invalid choice, try again.")
            except ValueError:
                print("Please enter a number.")

    def load_and_process_data(self):
        """Load subtitle data from database and process it"""
        if not self.db_path:
            return []

        print(f"Loading subtitles from {self.db_path}...")

        # Connect to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num, name, content FROM zipfiles")
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
            return []

        processed_data = []

        for subtitle_id, name, content in rows:
            # Extract zip content
            try:
                with io.BytesIO(content) as bio:
                    with zipfile.ZipFile(bio, "r") as zipf:
                        for file_name in zipf.namelist():
                            with zipf.open(file_name) as file:
                                text = file.read().decode("latin-1")

                                # Clean text thoroughly
                                text = re.sub(r'\d{1,2}:\d{2}:\d{2},\d{3} --> \d{1,2}:\d{2}:\d{2},\d{3}\r?\n', '', text)
                                text = re.sub(r'\r?\n', ' ', text)
                                text = re.sub(r'<[^>]+>', '', text)
                                text = re.sub(r'[^a-zA-Z\s]', '', text)
                                text = re.sub(r'\s+', ' ', text)

                                # Remove processing artifacts
                                text = re.sub(r'end of subtitle chunk.*?$', '', text, flags=re.IGNORECASE)
                                text = re.sub(r'subtitle chunk.*?$', '', text, flags=re.IGNORECASE)
                                text = text.lower().strip()

                                # Extract meaningful movie name
                                movie_name = name.replace('.zip', '').replace('.', ' ')
                                movie_name = re.sub(r'\beng\b.*?\d+cd\b', '', movie_name, flags=re.IGNORECASE)
                                movie_name = re.sub(r'\bkaggle\b.*?\bsubtitle\b', '', movie_name, flags=re.IGNORECASE)
                                movie_name = re.sub(r'\b\d+\b', '', movie_name)  # Remove standalone numbers
                                movie_name = re.sub(r'\s+', ' ', movie_name).strip().title()

                                if not movie_name or len(movie_name) < 3:
                                    movie_name = f"Movie {subtitle_id}"

                                # Only keep substantial content
                                if len(text) > 50 and len(text.split()) > 8:  # Meaningful content
                                    processed_data.append({
                                        'id': subtitle_id,
                                        'movie': movie_name,
                                        'content': text
                                    })
                                break
            except:
                continue

        self.data = processed_data
        print(f"Processed {len(processed_data)} movie subtitles")
        return processed_data

    def create_search_index(self):
        """Create search index using semantic embeddings or TF-IDF"""
        if not self.data:
            print("No data to index")
            return False

        documents = [item['content'] for item in self.data]

        if self.use_semantic:
            print("Creating semantic search index with sentence embeddings...")
            try:
                # Load pre-trained sentence transformer model
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

                # Create embeddings for all documents
                print("Generating semantic embeddings (this may take a moment)...")
                self.semantic_embeddings = self.sentence_model.encode(documents, show_progress_bar=True)

                print(f"Semantic index created with {len(documents)} documents")
                return True
            except Exception as e:
                print(f"Semantic model failed ({e}), falling back to TF-IDF")
                self.use_semantic = False

        if not self.use_semantic:
            print("Creating TF-IDF search index...")
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # Fit and transform documents
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)

            print(f"TF-IDF index created with {len(documents)} documents")
            return True

    def search(self, query, num_results=5):
        """Search for movies using semantic similarity or TF-IDF"""
        if self.use_semantic and self.sentence_model is not None and self.semantic_embeddings is not None:
            return self._semantic_search(query, num_results)
        elif self.vectorizer and self.tfidf_matrix is not None:
            return self._tfidf_search(query, num_results)
        else:
            return []

    def _semantic_search(self, query, num_results=5):
        """Perform semantic search using sentence embeddings"""
        # Encode the query
        query_embedding = self.sentence_model.encode([query])

        # Calculate semantic similarities
        similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()

        # Get top results
        top_indices = similarities.argsort()[-num_results:][::-1]

        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0.1:  # Lower threshold for semantic search
                results.append({
                    'rank': i + 1,
                    'movie': self.data[idx]['movie'].title(),
                    'similarity': round(similarities[idx], 3),
                    'preview': self.data[idx]['content'][:200] + "..." if len(self.data[idx]['content']) > 200 else self.data[idx]['content'],
                    'type': 'semantic'
                })

        return results

    def _tfidf_search(self, query, num_results=5):
        """Perform TF-IDF search"""
        # Transform query
        query_vector = self.vectorizer.transform([query.lower()])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top results
        top_indices = similarities.argsort()[-num_results:][::-1]

        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only return relevant results
                results.append({
                    'rank': i + 1,
                    'movie': self.data[idx]['movie'].title(),
                    'similarity': round(similarities[idx], 3),
                    'preview': self.data[idx]['content'][:200] + "..." if len(self.data[idx]['content']) > 200 else self.data[idx]['content'],
                    'type': 'tfidf'
                })

        return results

    def setup(self):
        """Setup the entire search engine"""
        if not self.db_path:
            return False

        self.load_and_process_data()
        return self.create_search_index()

    def show_database_info(self):
        """Show information about the current database"""
        print(f"\n📊 Database Information")
        print("=" * 25)
        print(f"Database: {self.db_path}")
        print(f"Description: {self.available_databases.get(self.db_path, 'Unknown')}")
        print(f"Processed entries: {len(self.data)}")

        if self.data:
            print(f"\nSample entries:")
            for i in range(min(3, len(self.data))):
                movie = self.data[i]['movie']
                preview = self.data[i]['content'][:100]
                print(f"  {i+1}. {movie}: \"{preview}...\"")

def main():
    """Main function"""
    print("🎬 Universal Movie Subtitle Search Engine (Semantic)")
    print("=" * 55)
    print("Supports: True semantic search with sentence embeddings")
    print("Features: Original sample, Kaggle datasets, and custom databases")

    # Initialize and setup
    engine = UniversalMovieSearchEngine()

    if not engine.setup():
        print("Setup failed!")
        print("\n💡 To create sample data:")
        print("   python3 kaggle_data_processor.py")
        print("   Choose option 4 for Kaggle-style sample")
        return

    # Show database info
    engine.show_database_info()

    print(f"\n Ready! Search engine loaded with {len(engine.data)} movies")
    print("Examples: 'betrayal friend', 'life wisdom', 'love story', 'action hero'")

    # Interactive search
    while True:
        try:
            query = input(f"\n Search: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            results = engine.search(query, num_results=3)

            if results:
                search_type = results[0].get('type', 'unknown')
                print(f"\n Found {len(results)} results ({search_type} search):")
                for result in results:
                    print(f"\n{result['rank']}. {result['movie']} (similarity: {result['similarity']})")
                    print(f"   \"{result['preview']}\"")
            else:
                print("No results found")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
