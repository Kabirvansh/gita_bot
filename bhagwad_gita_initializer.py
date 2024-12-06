import json
import sqlite3
from sentence_transformers import SentenceTransformer

def initialize_gita_database(json_file_path: str, db_path: str = 'gita_verses.db'):
    """
    Initialize SQLite database with Bhagavad Gita verses and pre-compute embeddings.
    
    Args:
        json_file_path (str): Path to JSON file with Gita verses
        db_path (str): Path to SQLite database file
    """
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create verses table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verses (
            id TEXT PRIMARY KEY,
            chapter TEXT,
            verse_number TEXT,
            original_verse TEXT,
            commentary TEXT,
            embedding BLOB
        )
    ''')
    
    # Load data from JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        gita_data = json.load(file)
    
    # Insert verses and compute embeddings
    for chapter_key, chapter in gita_data.get('chapters', {}).items():
        for verse_num, verse in chapter.get('verses', {}).items():
            # Compute embedding
            original_verse = verse.get('original_verse', '')
            embedding = model.encode(original_verse).tobytes()
            
            # Insert verse with embedding
            cursor.execute('''
                INSERT OR REPLACE INTO verses 
                (id, chapter, verse_number, original_verse, commentary, embedding) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"{chapter_key}_{verse_num}",
                chapter.get('chapter_number', ''),
                verse_num,
                original_verse,
                verse.get('commentary', {}).get('shankaracharya', ''),
                embedding
            ))
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized successfully at {db_path}")

# Usage
if __name__ == "__main__":
    initialize_gita_database('chapter2.json')