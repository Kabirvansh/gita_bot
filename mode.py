import streamlit as st
import sqlite3
import numpy as np
import re
from sentence_transformers import SentenceTransformer

class GitaChatbot:
    def __init__(self, db_path: str = 'gita_verses.db'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def find_most_similar_verse(self, question: str):
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
        relevant_verse = self.find_most_similar_verse(question)

        response = (f"Bhagavad Gita\n\n"
                    f"Question: {question}\n"
                    f"Similarity Score: {relevant_verse['similarity_score']:.2f}\n\n"
                    f"Verse: {relevant_verse['original_verse']}\n\n"
                    f"Commentary (Shankaracharya): {relevant_verse['commentary']}\n"
                    f"Chapter: {relevant_verse['chapter']}, Verse: {relevant_verse['verse_number']}")
        return response

def main():
    st.title("🕉️ Bhagavad Gita Chatbot")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GitaChatbot()
    
    user_question = st.text_input("Ask a question about life, philosophy, or spirituality:")
    
    if user_question:
        response = st.session_state.chatbot.chat(user_question)
        
        st.markdown("### Response")
        st.markdown(response)

if __name__ == "__main__":
    main()