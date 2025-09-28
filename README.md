# 🎬 Movie Subtitle Semantic Search Engine

This project implements a **true semantic search engine** for movie subtitles using **Sentence-BERT embeddings**, allowing users to find movies and dialogues based on meaning rather than exact keyword matches.

## ✨ Features

- **🧠 True Semantic Search:** Uses Sentence-BERT neural embeddings to understand meaning
- **🎯 Concept Understanding:** Finds "betrayal" when you search "backstab" and vice versa
- **📊 High-Quality Data:** Clean, processed subtitle database with meaningful content
- **⚡ Fast Performance:** Optimized for quick semantic similarity search
- **📚 Multiple Interfaces:** Command-line tool and Jupyter notebook analysis

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Semantic Search
```bash
python3 universal_search_engine.py
```

### 3. Try These Semantic Queries
- `"betrayal by friend"` → Finds backstabbing scenes without exact words
- `"space adventure"` → Finds sci-fi exploration themes
- `"fear and terror"` → Finds scary/frightening situations
- `"friendship bond"` → Finds loyalty and companionship

### 4. Jupyter Analysis (Optional)
```bash
jupyter notebook Movie_Subtitle_Search_Engine.ipynb
```

## 📁 Project Structure

- `universal_search_engine.py` - **Main semantic search engine**
- `Movie_Subtitle_Search_Engine.ipynb` - **Jupyter notebook analysis**
- `kaggle_data_processor.py` - **Dataset processor for adding more data**
- `eng_subtitles_database.db` - Sample database (8 movies)
- `kaggle_subtitles_database.db` - High-quality database (160 entries)
- `requirements.txt` - Python dependencies

## 🎯 Semantic Search Examples

**Input → Semantic Understanding:**
- `"betrayal"` → Finds "backstabbing murderer", "sworn enemy"
- `"space adventure"` → Finds "secret mission in uncharted space"
- `"friendship"` → Finds "best friends handshake", "friend in me"
- `"fear"` → Finds "scared", "panic", "atrocities"
- `"evil villain"` → Finds "evil emperor", "backstabbing murderer"

## 🏆 Perfect For

✅ **Internship Applications** - Demonstrates semantic search mastery
✅ **Portfolio Projects** - Professional implementation with real data
✅ **Technical Interviews** - Shows ML, NLP, and data processing skills
✅ **Academic Research** - Jupyter notebook with detailed analysis

## 🛠 Technical Stack

- **🐍 Python** - Core language
- **🤗 Sentence-Transformers** - Neural sentence embeddings (all-MiniLM-L6-v2)
- **📊 scikit-learn** - Cosine similarity and fallback TF-IDF
- **🐼 pandas** - Data processing and manipulation
- **🗄️ SQLite** - Database storage and retrieval
- **📓 Jupyter** - Interactive analysis and visualization

## 📊 Technical Approach

- **Neural Embeddings:** Uses pre-trained Sentence-BERT model for semantic understanding
- **High-Quality Data:** Clean processing removes artifacts and ensures meaningful content
- **Semantic Similarity:** Cosine similarity on 384-dimensional sentence embeddings
- **Fallback System:** TF-IDF backup if neural model fails
- **Optimized Performance:** Efficient batch processing and caching

## 🎖️ Perfect For Portfolios

✅ **Internship Applications** - Shows advanced NLP and semantic search skills
✅ **Technical Interviews** - Demonstrates ML, embeddings, and system design
✅ **Academic Projects** - Includes detailed Jupyter analysis and methodology
✅ **Production Systems** - Professional code structure and error handling