import os
from typing import List, Dict, Generator
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from openai import OpenAI

# Load environment variables first
load_dotenv()

@st.cache_resource
def load_vector_store():
    """Charge les PDFs depuis le dossier docs/ et crée le vector store"""
    if not os.path.exists("docs"):
        os.makedirs("docs")
        return None
    
    pdf_files = [os.path.join("docs", f) for f in os.listdir("docs") if f.endswith(".pdf")]
    
    if not pdf_files:
        return None

    sections = []
    
    for pdf_path in pdf_files:
        school_name = os.path.splitext(os.path.basename(pdf_path))[0].upper()
        
        pdf_text = []
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            if text := page.extract_text():
                pdf_text.append(text)
        
        full_text = '\n'.join(pdf_text)
        current_section = []
        
        for line in full_text.split('\n'):
            if line.strip().startswith('#'):
                if current_section:
                    section = f"[FORMATION: {school_name}]\n" + '\n'.join(current_section)
                    sections.append(section)
                    current_section = []
                current_section.append(line.strip())
            else:
                current_section.append(line)
        
        if current_section:
            section = f"[Formation (Ecole): {school_name}]\n" + '\n'.join(current_section)
            sections.append(section)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return FAISS.from_texts(
        texts=sections,
        embedding=embeddings
    )

class PDFChatbot:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.vector_store = load_vector_store()
        self.chat_history = []

    def generate_response(self, user_query: str) -> Generator[str, None, None]:
        """Generate response with streaming support"""
        if not self.vector_store:
            yield "⚠️ Aucun document détecté. Veuillez :\n1. Créer un dossier 'docs/'\n2. Y placer les fichiers PDF\n3. Recharger l'application"
            return

        relevant_docs = self.vector_store.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {
                "role": "system",
                "content": f"""
                **Rôle** : Assistant spécialisé dans les écoles et programmes de l'Université Mohammed VI Polytechnique (UM6P). Réponds dans la même langue que l'utilisateur.

                **Contexte UM6P** :
                - Fondée en 2013, l'UM6P se concentre sur la recherche, l'innovation et l'enseignement en sciences, ingénierie, technologies, sciences sociales et énergies renouvelables.
                - Mission : Promouvoir le développement durable et l'innovation au Maroc et en Afrique.

                **Établissements et Contacts** :
                [...]  # (Keep your original system prompt here)
                
                **Contexte actuel** :
                {context}"""
            }
        ]

        for msg in self.chat_history:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        messages.append({"role": "user", "content": user_query})

        try:
            stream = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )

            full_response = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text_chunk = chunk.choices[0].delta.content
                    full_response.append(text_chunk)
                    yield text_chunk

            self.chat_history.append({
                "user": user_query,
                "assistant": "".join(full_response)
            })

        except Exception as e:
            yield f"Erreur de génération de réponse: {str(e)}"

def main():
    st.set_page_config(page_title="UM6P Chatbot", page_icon="🎓")
    
    st.markdown("""
    <style>
        .streaming {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stMarkdown p {
            font-size: 16px !important;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
        .blink-cursor::after {
            content: "▌";
            animation: blink 1s step-end infinite;
            color: #2d3436;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Assistant UM6P")
    st.caption("Posez vos questions sur les programmes et écoles de l'UM6P")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()

    if not st.session_state.chatbot.vector_store:
        st.warning("Configuration requise :")
        st.markdown("""
        1. Créer un dossier `docs/`
        2. Placer les fichiers PDF des formations dedans
        3. Redémarrer l'application
        """)

    user_input = st.chat_input("Écrivez votre question ici...")
    
    for msg in st.session_state.chatbot.chat_history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["assistant"])

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            for chunk in st.session_state.chatbot.generate_response(user_input):
                full_response += chunk
                response_container.markdown(
                    f'<div class="streaming blink-cursor">{full_response}</div>', 
                    unsafe_allow_html=True
                )
            
            response_container.markdown(f'<div class="streaming">{full_response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()