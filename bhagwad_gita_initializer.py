
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
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            gita_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verses (
                id TEXT PRIMARY KEY,
                chapter TEXT,
                verse_number TEXT,
                original_verse TEXT,
                speaker TEXT,
                commentary TEXT,
                tags TEXT,
                embedding BLOB
            )
        ''')

        # Process and insert verses
        total_verses = sum(len(chapter.get('verses', {})) for chapter in gita_data.get('chapters', {}).values())
        processed = 0

        for chapter_key, chapter in gita_data.get('chapters', {}).items():
            for verse_num, verse in chapter.get('verses', {}).items():
                processed += 1

                original_verse = verse.get('original_verse', '')
                speaker = verse.get('speaker', '')
                commentary = verse.get('commentary', {}).get('shankaracharya', '')
                tags = ", ".join(verse.get('tags', []))  # Convert list of tags to comma-separated string
                embedding = model.encode(original_verse).tobytes()

                cursor.execute('''
                    INSERT OR REPLACE INTO verses 
                    (id, chapter, verse_number, original_verse, speaker, commentary, tags, embedding) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{chapter_key}_{verse_num}",
                    chapter_key,
                    verse_num,
                    original_verse,
                    speaker,
                    commentary,
                    tags,
                    embedding
                ))
                
                print(f"Processed {processed}/{total_verses} verses...", end="\r")
        
        print(f"\nDatabase initialized successfully at {db_path}")

if __name__ == "__main__":
    initialize_gita_database('chapter2.json')