import streamlit as st
import sqlite3
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import anthropic
import os

class GitaChatbot:
    def __init__(self, db_path: str = 'gita_verses.db'):
       
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def normalize_text(self, text: str) -> str:
        """Normalize input text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get_verse_by_chapter_and_number(self, chapter: int, verse_number: int):
        """Retrieve a specific verse from the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT original_verse, commentary 
            FROM verses 
            WHERE chapter = ? AND verse_number = ?
        ''', (chapter, verse_number))
        
        result = cursor.fetchone()
        if result:
            return {
                "original_verse": result[0],
                "commentary": result[1]
            }
        return None

    def generate_philosophical_response(self, question: str):
        """Generate a philosophical response."""
        
        condition_verse_guide = {
            "ANGER": {
                "chapters": [2, 16],
                "key_verses": [
                    {"chapter": 2, "verses": [56, 62, 63]},
                    {"chapter": 16, "verses": [2, 3, 21]}
                ],
                "guidance": "Focus on verses that discuss controlling anger, emotional regulation, and the destructive nature of uncontrolled rage."
            },
            "FEELING SINFUL": {
                "chapters": [4, 5, 10, 14, 18],
                "key_verses": [
                    {"chapter": 4, "verses": [36, 33]},
                    {"chapter": 5, "verses": [30]},
                    {"chapter": 10, "verses": [37]},
                    {"chapter": 14, "verses": [6]},
                    {"chapter": 18, "verses": [66]}
                ],
                "guidance": "Select verses that emphasize divine forgiveness, spiritual purification, and transcending past mistakes."
            },
            "PRACTISING FORGIVENESS": {
                "chapters": [11, 12, 13, 16],
                "key_verses": [
                    {"chapter": 11, "verses": [44]},
                    {"chapter": 12, "verses": [14]},
                    {"chapter": 13, "verses": [16]},
                    {"chapter": 16, "verses": [2, 3]}
                ],
                "guidance": "Choose verses that highlight compassion, humility, and the spiritual strength of forgiveness."
            },
            
            "DEPRESSION": {
                "chapters": [2, 5],
                "key_verses": [
                    {"chapter": 2, "verses": [3, 14]},
                    {"chapter": 5, "verses": [21]}
                ],
                "guidance": "Select verses that offer hope, resilience, and spiritual perspective during emotional low points."
            },
            "FEAR": {
                "chapters": [2, 4, 18],
                "key_verses": [
                    {"chapter": 2, "verses": [50]},
                    {"chapter": 4, "verses": [10]},
                    {"chapter": 18, "verses": [30]}
                ],
                "guidance": "Focus on verses that discuss overcoming fear through spiritual wisdom and inner strength."
            },
            "DEMOTIVATED": {
                "chapters": [11, 18],
                "key_verses": [
                    {"chapter": 11, "verses": [33]},
                    {"chapter": 18, "verses": [66, 78]}
                ],
                "guidance": "Choose verses that inspire action, divine support, and finding purpose beyond temporary setbacks."
            }
        }

        try:
            condition = None
            for key in condition_verse_guide.keys():
                if key.lower() in question.lower():
                    condition = key
                    break
   
            guidance = "Provide a philosophical response with personal touch based on the Bhagavad Gita"
            if condition and condition in condition_verse_guide:
                condition_data = condition_verse_guide[condition]
                guidance += f"\n\nSpecific Guidance for {condition}:\n"
                guidance += condition_data.get('guidance', '')
                guidance += "\n\nREQUIREMENTS:"
            else:
                guidance += "\n\nREQUIREMENTS:"

            guidance += """
            - DO NOT GIVE ANSWERS TO FACTUAL QUESTIONS, JUST SAY DONT KNOW
            - 75-100 words long
            - Ensure the ENTIRE response is generated completely
            - Do NOT truncate or leave the response incomplete
            - Inspired by Krishna's teachings
            - Add a PERSONAL TOUCH to the answer
            - MUST include chapter and verse number in format: (Chapter X, Verse Y)
            - Ensure ONLY ONE response is generated
            - Ensure ONLY ONE verse is generated 
            - First try to find verses from other chapters(3-18) if couldnt find then search chapter 2 
            """

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[
                    {
                        "role": "user", 
                        "content": f"{guidance}\n\nQuestion: {question}"
                    }
                ]
            )

            raw_response = response.content[0].text.strip()
            print(f"Raw API Response: {raw_response}")


            verse_match = re.search(r'\(Chapter (\d+), Verse (\d+)\)', raw_response)
            
            if not verse_match:
                print("No verse reference found in the response")
                return None

            full_verse_ref = verse_match.group(0)
            chapter = int(verse_match.group(1))
            verse_number = int(verse_match.group(2))

            response_text = raw_response.replace(full_verse_ref, '').strip()
            
            return {
                "response": f"{response_text}",
                "chapter": chapter,
                "verse_number": verse_number
            }
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def chat(self, question: str):
        """Main chat method to process question and retrieve verse."""

        try:
            api_response = self.generate_philosophical_response(question)
            

            if not api_response or isinstance(api_response, str):
                return "Unable to generate a response."
            
            verse_details = self.get_verse_by_chapter_and_number(
                api_response['chapter'], 
                api_response['verse_number']
            )
            
            if not verse_details:
                return "Verse not found in the database."

            return {
                "question": question,
                "philosophical_response": api_response['response'],
                "chapter": api_response['chapter'],
                "verse_number": api_response['verse_number'],
                "original_verse": verse_details['original_verse'],
                "commentary": verse_details['commentary']
            }
        
        except Exception as e:
            print(f"Error in chat method: {e}")
            return "An error occurred while processing your question."

def main():

    st.set_page_config(page_title="🕉️ Bhagavad Gita Chatbot)", page_icon="🕉️")
    
    st.title("🕉️ Bhagavad Gita Chatbot")

    st.sidebar.markdown("""
    ### About this App
    Project for RELIG 397\n
    Team :\n  
            Kabirvansh Chadha\n
        Aaron Lad\n 
        Hriday Nijhawan\n

    """)

    st.sidebar.markdown("""
    #### Note:
    - This chatbot is in an **early development stage** and might make mistakes or provide incomplete responses.
    - If the bot is unable to generate a response, please try rephrasing your question or ask again.
    """)
    
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = GitaChatbot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            return

    user_question = st.text_input("Ask a question:", 
                                  placeholder="What is the meaning of life?")
    
    if st.button("Seek Wisdom") or user_question:
        if user_question:
            try:
                response = st.session_state.chatbot.chat(user_question)
                
                if isinstance(response, str):
                    st.warning(response)
                    return
                
                st.markdown("### AI Response:")
                st.markdown(response['philosophical_response'])

                st.markdown("### Relevant Verse(Patton)")
                st.markdown(f"**Chapter {response['chapter']}, Verse {response['verse_number']}**")
                st.markdown(response['original_verse'])

                st.markdown("### Shankracharya's Commentary")
                st.markdown(response['commentary'])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
