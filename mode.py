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
        cursor.execute('SELECT id, chapter, verse_number, original_verse, commentary, embedding FROM verses')

        best_match = None
        highest_similarity = -1  

        for verse_id, chapter, verse_num, original_verse, commentary, db_embedding in cursor:
            if db_embedding is None:
                continue  

            embedding = np.frombuffer(db_embedding, dtype=np.float32)

            similarity = np.dot(question_embedding, embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
            )

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {
                    "similarity_score": similarity,
                    "chapter": chapter,
                    "verse_number": verse_num,
                    "original_verse": original_verse,
                    "commentary": commentary
                }

        return best_match


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