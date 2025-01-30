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
    def __init__(self, model_choice: str):
        self.model_choice = model_choice
        # Configurer le client en fonction du modèle choisi
        if model_choice == "DeepSeek":
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        else:  # GPT-4o
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
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
            1. **EMINES (School of Industrial Management)** :
            - Programme : Cycle Ingénieur en Management Industriel
            - Contact : 
                📧 contact@emines-ingenieur.org | 🌐 emines-ingenieur.org

            2. **CC (College of Computing)** :
            - Programme : Cycle Ingénieur en Computer Sciences
            - Contact : 
                📧 cc@um6p.ma | 📞 06 69 93 51 50 | 🌐 cc.um6p.ma/engineering-degree

            3. **GTI (Green Tech Institute)** :
            - Programme : 
                - Master Ingénierie Electrique pour les Energies Renouvelables et les Réseaux Intelligents (📧 Master.RESMA@um6p.ma) 
                - Master Technologies Industrielles pour l’Usine du Futur (📧 master.TIUF@um6p.ma)
            - Contact :
                📧  admission@um6p.ma | 📞 +212 525 073 308 | 🌐 um6p.ma/index.php/fr/green-tech-institute

            4. **SoCheMiB-IST&I** :
            - Programme : Cycle Ingénieur en Génie Chimique, Minéralogique et Biotechnologique
            - Contact : 
                📧 admission@um6p.ma | 📞 +212 525 073 308 | 🌐 um6p.ma/fr/institute-science-technology-innovation

            5. **SAP+D (School of Architecture)** :
            - Programme : 
                - Cycle Architecte (Bac+6)
                - Master Ingénierie des Bâtiments Verts et Efficacité Energétique 
            - Contact : 
                📧 contactsapd@um6p.ma | 📞 06 69 93 51 50 | 🌐 um6p.ma/fr/sapd-school-architecture-planning-design

            6. **ABS (Africa Business School)** :
            - Programmes : 
                - Master AgriBusiness Innovation
                - Master Financial Engineering
                - Master International Management
            - Contact : 
                📧 ali.assmirasse@um6p.ma | 📞 +212 659 46 59 79 | 🌐 abs.um6p.ma

            7. **SHBM (School of Hospitality)** :
            - Programme : Bachelor in Hospitality Business & Management
            - Contact : 
                📧 admissions.shbm@um6p.mah | 📞 +212 6 62 10 47 63 | 🌐 www.shbm-um6p.ma

            8. **FMS (Faculty of Medical Sciences)** :
            - Programmes : Doctorat en Médecine, Doctorat en Pharmacie
            - Contact : 
                📧 admission-fms@um6p.ma | 📞 +212 525-073051 / +212 665-693326 | 🌐 um6p.ma/fr/faculty-medical-sciences-0

            9. **ISSB-P (Institut Supérieur des Sciences Biologiques et Paramédicales)** :
            - Programmes : Licence Soins infirmiers, option infirmier polyvalent 
            - Contact : 
                📧 admissionISSBP@um6p.ma | 📞 +212 669 936 049 | 🌐 um6p.ma/en/institute-biological-sciences

            10. 10. *FGSES (Faculty of Governance, Economics and Social Sciences)* :
            - Programmes : 
                Programmes Post-Bac :
                * Licence en économie Appliquée 
                * Licence en Science Politique
                * Licence en Sciences Comportementales pour
                les Politiques Publiques
                * Licence en Relations Internationales
                Programmes Master :
                * Master Behavioral and Social Sciences for
                Public Policy 
                * Master Global Affairs
                * Master Political Science
                * Master Analyse Economique et Politiques
                Publiques 
                * Master Economie Quantitative
            - Contact : 
                📧 Info.fgses@um6p.ma | 📞 +212 (0) 530 431 217 | 🌐 www.fgses-um6p.ma    

            **Directives Strictes** :
            1. **Identification de l'École** :
            - Vérifie TOUJOURS le nom exact de l'école dans la question (ex: "EMINES", "CC", "SAP+D").
            - Si la question mentionne un programme (ex: "Ingénieur en Computer Sciences"), associe-le à l'école correspondante (ex: CC).

            2. **Règles de Réponse** :
            - Répondre uniquement sur la base du contexte:
            - Utilise UNIQUEMENT les informations fournies dans le contexte.
            - Si l'information n'est pas dans le contexte dirigé l'utilisatuer vers le contact de l'école concernée, réponds : "Je ne trouve pas cette information dans ma base de connaissances. Veuillez consulter le site : 🌐 "
            - Si l'école n'est pas claire → Demande : "Veuillez préciser l'école (ex: EMINES, CC, SAP+D)".
            - Toujours donner les contacts officiels de l'école concernée en se basant sur la liste fournie des etablissemtn et contacts.

            3. **Interdictions** :
            - Aucun mélange d'informations entre écoles (ex: ne pas utiliser les contacts de l'ABS pour une question sur le CC).
            - Ne pas inventer de contacts ou de liens. Utilise UNIQUEMENT ceux fournis.

            4. **En cas d'erreur** :
            - Si l'information est manquante → Réponds : 
                "Pour plus de détails, consultez le site officiel de l'UM6P : 🌐 https://um6p.ma/fr".

                                    **Contexte actuel** :
                                    {context}"""
                        }
        ]

        for msg in self.chat_history:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        messages.append({"role": "user", "content": user_query})

        try:
            # Choisir le modèle approprié
            model_name = "deepseek-chat" if self.model_choice == "DeepSeek" else "gpt-4o"
            
            stream = self.client.chat.completions.create(
                model=model_name,
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
    
    # Ajouter la sélection de modèle dans la sidebar
    with st.sidebar:
        st.header("Paramètres")
        model_choice = st.selectbox(
            "Choix du modèle",
            ("DeepSeek", "GPT-4o"),
            index=0
        )

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

    # Réinitialiser le chatbot si le modèle change
    if 'chatbot' not in st.session_state or st.session_state.current_model != model_choice:
        st.session_state.chatbot = PDFChatbot(model_choice=model_choice)
        st.session_state.current_model = model_choice
        st.session_state.chatbot.chat_history = []

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
