#!/usr/bin/env python3
"""
Kaggle Dataset Processor for Movie Subtitles
Downloads and processes real subtitle datasets from Kaggle
"""

import os
import pandas as pd
import sqlite3
import zipfile
import io
import json
import requests
from pathlib import Path

class KaggleSubtitleProcessor:
    def __init__(self):
        self.datasets = {
            "movie_subtitle_dataset": {
                "name": "Movie Subtitle Dataset",
                "kaggle_url": "https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset",
                "description": "5k timestamped subtitles with IMDB metadata",
                "files": ["subtitles.csv", "movies.csv"]
            },
            "english_subtitles": {
                "name": "English Subtitles (opensubtitles.org)",
                "kaggle_url": "https://www.kaggle.com/datasets/kaushikrahul/english-subtitles-opensubtitles-org",
                "description": "Large collection of English subtitles",
                "files": ["subtitles.csv"]
            },
            "multilingual_subtitles": {
                "name": "Open Subtitles Multilingual Translation",
                "kaggle_url": "https://www.kaggle.com/datasets/thedevastator/open-subtitles-multilingual-translation",
                "description": "Multilingual subtitle corpus",
                "files": ["data.csv", "translations.csv"]
            }
        }

    def show_available_datasets(self):
        """Display available Kaggle datasets"""
        print("📊 Available Kaggle Subtitle Datasets")
        print("=" * 45)

        for key, dataset in self.datasets.items():
            print(f"\n{len([d for d in self.datasets.keys() if d <= key])}. {dataset['name']}")
            print(f"   URL: {dataset['kaggle_url']}")
            print(f"   Description: {dataset['description']}")
            print(f"   Files: {', '.join(dataset['files'])}")

        print(f"\n💡 To download:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Setup API key: https://www.kaggle.com/docs/api")
        print("3. Run this script to process downloaded data")

    def setup_kaggle_instructions(self):
        """Show detailed Kaggle setup instructions"""
        print("🔧 Kaggle Setup Instructions")
        print("=" * 30)

        print("Step 1: Install Kaggle CLI")
        print("   pip install kaggle")

        print("\nStep 2: Get API Credentials")
        print("   1. Go to: https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Download kaggle.json")
        print("   4. Place it in: ~/.kaggle/kaggle.json")
        print("   5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")

        print("\nStep 3: Download Dataset")
        print("   kaggle datasets download -d adiamaan/movie-subtitle-dataset")
        print("   unzip movie-subtitle-dataset.zip")

        print("\nStep 4: Process Data")
        print("   python3 kaggle_data_processor.py")

    def process_csv_to_database(self, csv_file_path, dataset_type="subtitles"):
        """Process CSV file and create database"""
        print(f"📁 Processing {csv_file_path}...")

        if not os.path.exists(csv_file_path):
            print(f"❌ File not found: {csv_file_path}")
            return False

        try:
            # Read CSV
            df = pd.read_csv(csv_file_path)
            print(f"   Loaded {len(df)} rows")

            # Display columns to understand structure
            print(f"   Columns: {list(df.columns)}")

            # Process based on dataset type
            processed_data = self.process_dataframe(df, dataset_type)

            if processed_data:
                # Create database
                db_name = f"kaggle_{dataset_type}_database.db"
                self.create_database(processed_data, db_name)
                return True
            else:
                print("   ❌ No data to process")
                return False

        except Exception as e:
            print(f"   ❌ Error processing CSV: {e}")
            return False

    def process_dataframe(self, df, dataset_type):
        """Process DataFrame based on dataset type"""
        processed_data = []

        # Common column names to look for
        text_columns = ['text', 'subtitle', 'content', 'dialogue', 'line', 'subtitle_text']
        movie_columns = ['movie', 'title', 'film', 'movie_name', 'imdb_title']

        # Find the text column
        text_col = None
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break

        # Find movie column
        movie_col = None
        for col in movie_columns:
            if col in df.columns:
                movie_col = col
                break

        if not text_col:
            print(f"   ❌ No text column found. Available: {list(df.columns)}")
            return []

        print(f"   Using text column: '{text_col}'")
        if movie_col:
            print(f"   Using movie column: '{movie_col}'")

        # Process rows
        for idx, row in df.iterrows():
            try:
                text = str(row[text_col])
                if len(text) > 20:  # Only meaningful content

                    movie_name = "Unknown Movie"
                    if movie_col and pd.notna(row[movie_col]):
                        movie_name = str(row[movie_col])

                    processed_data.append({
                        'id': idx + 1,
                        'name': f"{movie_name.lower().replace(' ', '.')}.kaggle.subtitle",
                        'content': text.lower().strip()
                    })

                # Limit to first 1000 for testing
                if len(processed_data) >= 1000:
                    break

            except Exception as e:
                continue

        print(f"   Processed {len(processed_data)} subtitle entries")
        return processed_data

    def create_database(self, movie_data, db_name):
        """Create SQLite database from processed data"""
        print(f"💾 Creating database: {db_name}")

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create table (same format as existing system)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zipfiles (
                num INTEGER PRIMARY KEY,
                name TEXT,
                content BLOB
            )
        ''')

        for movie in movie_data:
            # Create subtitle format content
            subtitle_content = f"""1
00:00:10,000 --> 00:00:15,000
{movie['content']}

2
00:00:20,000 --> 00:00:25,000
End of subtitle chunk"""

            # Create zip content (to match existing format)
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(f"{movie['name']}.srt", subtitle_content)

            # Insert into database
            cursor.execute(
                "INSERT INTO zipfiles (num, name, content) VALUES (?, ?, ?)",
                (movie['id'], movie['name'], zip_buffer.getvalue())
            )

        conn.commit()
        conn.close()

        print(f"✅ Database created: {db_name} with {len(movie_data)} entries")

    def download_sample_kaggle_data(self):
        """Download a sample from Kaggle-style data (for demo)"""
        print("📥 Creating Kaggle-style Sample Data")
        print("=" * 35)

        # Simulate real Kaggle subtitle data
        sample_data = [
            {"movie": "The Shawshank Redemption", "subtitle": "Hope is a good thing, maybe the best of things, and no good thing ever dies"},
            {"movie": "The Godfather", "subtitle": "I'm gonna make him an offer he can't refuse"},
            {"movie": "The Dark Knight", "subtitle": "Why so serious? Let's put a smile on that face"},
            {"movie": "Pulp Fiction", "subtitle": "The path of the righteous man is beset on all sides"},
            {"movie": "Forrest Gump", "subtitle": "Life was like a box of chocolates, you never know what you're gonna get"},
            {"movie": "Fight Club", "subtitle": "The first rule of Fight Club is you do not talk about Fight Club"},
            {"movie": "Inception", "subtitle": "We need to go deeper into the dream within the dream"},
            {"movie": "The Matrix", "subtitle": "There is no spoon, Neo, the spoon does not exist"},
            {"movie": "Goodfellas", "subtitle": "As far back as I can remember, I always wanted to be a gangster"},
            {"movie": "Casablanca", "subtitle": "Here's looking at you, kid, in this moment forever"},
            {"movie": "Star Wars", "subtitle": "May the Force be with you, always and forever"},
            {"movie": "Titanic", "subtitle": "I'll never let go Jack, I promise you that"},
            {"movie": "The Departed", "subtitle": "I'm the guy doing his job, you must be the other guy"},
            {"movie": "Scarface", "subtitle": "Say hello to my little friend, this is the end"},
            {"movie": "Casino", "subtitle": "When you love someone, you've gotta trust them completely"}
        ]

        # Create DataFrame (simulate Kaggle CSV)
        df = pd.DataFrame(sample_data)
        csv_path = "kaggle_sample.csv"
        df.to_csv(csv_path, index=False)

        print(f"✅ Created sample CSV: {csv_path}")

        # Process the sample
        success = self.process_csv_to_database(csv_path, "sample")

        # Clean up
        os.remove(csv_path)

        return success

def main():
    """Main function"""
    print("🎬 Kaggle Subtitle Dataset Processor")
    print("=" * 40)

    processor = KaggleSubtitleProcessor()

    print("Choose an option:")
    print("1. Show available Kaggle datasets")
    print("2. Show Kaggle setup instructions")
    print("3. Process local CSV file")
    print("4. Create Kaggle-style sample database")
    print("5. Process specific Kaggle dataset")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        processor.show_available_datasets()

    elif choice == "2":
        processor.setup_kaggle_instructions()

    elif choice == "3":
        csv_path = input("Enter CSV file path: ").strip()
        dataset_type = input("Dataset type (subtitles/movies/general): ").strip() or "subtitles"
        processor.process_csv_to_database(csv_path, dataset_type)

    elif choice == "4":
        if processor.download_sample_kaggle_data():
            print("\n🎉 Sample database created!")
            print("Run: python3 search_engine.py")
            print("Change database path to: kaggle_sample_database.db")

    elif choice == "5":
        print("\nTo download specific Kaggle datasets:")
        print("1. Setup Kaggle CLI (option 2)")
        print("2. Download dataset:")
        print("   kaggle datasets download -d adiamaan/movie-subtitle-dataset")
        print("3. Extract and run option 3 with the CSV file")

    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()