import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import sqlite3
import re
import streamlit as st

class OptimizedBhagavadGitaChatbot:
    def __init__(self, json_file_path: str, db_path: str = 'gita_verses.db'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.conn = sqlite3.connect(db_path)
        self.create_database(json_file_path)

        self.pre_compute_embeddings()

    def create_database(self, json_file_path: str):
        """Create SQLite database from JSON data."""
        cursor = self.conn.cursor()
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

        with open(json_file_path, 'r', encoding='utf-8') as file:
            gita_data = json.load(file)

        for chapter_key, chapter in gita_data.get('chapters', {}).items():
            for verse_num, verse in chapter.get('verses', {}).items():
                cursor.execute('''
                    INSERT OR REPLACE INTO verses 
                    (id, chapter, verse_number, original_verse, commentary) 
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    f"{chapter_key}_{verse_num}",
                    chapter.get('chapter_number', ''),
                    verse_num,
                    verse.get('original_verse', ''),
                    verse.get('commentary', {}).get('shankaracharya', '')
                ))

        self.conn.commit()

    def pre_compute_embeddings(self):
        """Pre-compute and store embeddings for all verses."""
        cursor = self.conn.cursor()
        verses = cursor.execute('SELECT id, original_verse FROM verses').fetchall()

        for verse_id, original_verse in verses:
            embedding = self.model.encode(original_verse).tobytes()
            cursor.execute(
                'UPDATE verses SET embedding = ? WHERE id = ?', 
                (embedding, verse_id)
            )

        self.conn.commit()

    def normalize_text(self, text: str) -> str:
        """Normalize input text for better matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def find_most_similar_verse(self, question: str) -> Dict[str, str]:
        """Find most semantically similar verse using embedding similarity."""
        normalized_question = self.normalize_text(question)
        question_embedding = self.model.encode(normalized_question)

        cursor = self.conn.cursor()
        verses = cursor.execute('SELECT id, chapter, verse_number, original_verse, commentary, embedding FROM verses').fetchall()

        similarities = []
        for verse_id, chapter, verse_num, original_verse, commentary, db_embedding in verses:
            embedding = np.frombuffer(db_embedding, dtype=np.float32)

            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((similarity, verse_id, chapter, verse_num, original_verse, commentary))

        best_match = max(similarities, key=lambda x: x[0])
        
        return {
            "similarity_score": best_match[0],
            "chapter": best_match[2],
            "verse_number": best_match[3],
            "original_verse": best_match[4],
            "commentary": best_match[5]
        }

    def chat(self, question: str) -> str:
        """Generate a response for the user question."""
        relevant_verse = self.find_most_similar_verse(question)

        response = (f"Bhagavad Gita\n\n"
                    f"Question: {question}\n"
                    f"Similarity Score: {relevant_verse['similarity_score']:.2f}\n\n"
                    f"Verse: {relevant_verse['original_verse']}\n\n"
                    f"Commentary (Shankaracharya): {relevant_verse['commentary']}\n"
                    f"Chapter: {relevant_verse['chapter']}, Verse: {relevant_verse['verse_number']}")
        return response

    def __del__(self):
        """Close database connection."""
        self.conn.close()

def main():
    st.title("🕉️ Bhagavad Gita Wisdom Chatbot")
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = OptimizedBhagavadGitaChatbot('chapter2.json')
    
    user_question = st.text_input("Ask a question about life, philosophy, or spirituality:")
    
    if user_question:
        response = st.session_state.chatbot.chat(user_question)
        
        st.markdown("### Response")
        st.markdown(response)

if __name__ == "__main__":
    main()