from dotenv import load_dotenv
import os
from openai import OpenAI
from pypdf import PdfReader
from typing import Generator
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chargement des variables d'environnement
load_dotenv()

class InteractiveClarifier:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="ttps://api.fireworks.ai/inference/v1/chat/completions"
        )
        self.context = []

    def clarify_question(self, user_query: str) -> str:
        """Clarifie la question utilisateur"""
        clarification_prompt = [
            {
                "role": "system",
                "content": f"""
                **Répondre toujours dans la même langue que l'utilisateur**
                Corrige les noms d'écoles UM6P et ajoute le nom correct de l'école et la formation si la question est imcomplète.
                

                **Établissements et Contacts** :
            1. **EMINES (School of Industrial Management)** :
            - Programme : 
                - Cycle Préparatoire Intégré en Management Industriel 
                - Cycle Ingénieur en Management Industriel 
           
            2. **CC (College of Computing)** :
            - Programme : 
                    - Cycle Préparatoire Intégré en Computer Science 
                    - Cycle Ingénieur en Computer Science 
                    - Cycle Ingénieur en Cyber Security 

            3. **GTI (Green Tech Institute)** :
            - Programme : 
                - Master Ingénierie Electrique pour les Energies Renouvelables et les Réseaux Intelligents(📧 Master.RESMA@um6p.ma) 
                - Master Technologies Industrielles pour l'Usine du Futur (📧 master.TIUF@um6p.ma) 
          
            4. **SoCheMiB-IST&I** :
            - Programme : Cycle Ingénieur en Génie Chimique, Minéralogique et Biotechnologique 
        
            5. **SAP+D (School of Architecture)** :
            - Programme : 
                - Cycle Architecte a Ben Guerir (Bac+6) 
                - Master Ingénierie des Bâtiments Verts et Efficacité Energétique 

   

            6. **ABS (Africa Business School)** :
            - Programmes : 
                - Master AgriBusiness Innovation 
                - Master Financial Engineering 
                - Master International Management 

            7. **SHBM (School of Hospitality)** :
            - Programme : Bachelor in Hospitality Business & Management 
            
            8. **FMS (Faculty of Medical Sciences)** :
            - Programmes : Doctorat en Médecine, Doctorat en Pharmacie 
            
            9. **ISSB-P (Institut Supérieur des Sciences Biologiques et Paramédicales)** :
            - Programmes : Licence Soins infirmiers, option infirmier polyvalent 
 
            10. *FGSES (Faculty of Governance, Economics and Social Sciences)* :
            - Programmes : 
                Programmes Post-Bac :
                    * Licence en économie Appliquée 
                    * Licence en Science Politique 
                    * Licence en Sciences Comportementales pour les Politiques Publiques 
                    * Licence en Relations Internationales 
                    * Licence Droit public
                Programmes Master :
                    * Master Behavioral and Social Sciences for Public Policy 
                    * Master Global Affairs 
                    * Master Political Science  
                    * Master Analyse Economique et Politiques Publiques 
                    * Master Economie Quantitative 
   
            11. **SASE (School of Applied Sciences and Engineering)** :
            - Programme : 
                - Bachelor of Science (B.Sc.) en Applied Sciences and Engineering (4 ans) avec spécialisations :
                    * Mechanical Engineering
                    * Electrical Engineering
                    * Chemical & Material Engineering
                    * Environmental Engineering
                    * Applied Mathematics & Business
            
            12. **CASE (Cycle Ingénieur en Systèmes Agri-Environnementaux)** :
            - Programmes :
                - Cycle Préparatoire Intégré + 
                - Cycle Ingénieur en Systèmes Agro-Environnementaux (5 ans, Bac+5)

            13. **School of collective intelligence
            - Programmes : 
                - Master of collective intelligence

                Contexte précédent: {self.context[-1] if self.context else "Aucun"}
                
                        **Règles Strictes** :
                            1. Reformuler la question de l'uilisateur en :
                                - Corrigeant les noms d'écoles
                                - Ajoutant le nom de l'école correct si la question est incomplète en utilisant les noms d'écoles ci-dessus et le contexte precedent
                                - Ajoutant le nom du programme si necessaire
                                - Gardant la structure originale de la question de l'utilisateur
                                - Ne pas inventer d'informations
                                - Ne reformuler pas si la question est déjà claire
                                - Traiter CHAQUE question de manière indépendante
                                - Ne jamais combiner avec des questions précédentes
                            2. Ne jamais ajouter de réponse ou d'explication
                            3. Ne pas ajouter de commentaires
                            4. Tu DOIS traiter la question dans la même langue que celle utilisée par l'utilisateur.
                            5. Ne traduis pas la question, clarifie-la uniquement dans sa langue d'origine.
                    

                """
            },
            {"role": "user", "content": user_query}
        ]

        try:
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v3p1",
                messages=clarification_prompt,
                temperature=0.1,
                max_tokens=100
            )
            clarified = response.choices[0].message.content
            self.context.append(clarified)
            return clarified
        except Exception as e:
            return user_query

@st.cache_resource
def load_vector_store():
    """Charge les PDFs et crée le vector store"""
    if not os.path.exists("docs"):
        os.makedirs("docs")
        return None
    
    pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
    if not pdf_files:
        return None

    sections = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join("docs", pdf_file)
        school_name = os.path.splitext(pdf_file)[0].upper()
        
        pdf_reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        sections.append(f"[FORMATION: {school_name}]\n{text}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return FAISS.from_texts(
        texts=sections,
        embedding=embeddings
    )

class PDFChatbot:
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature
        self.clarifier = InteractiveClarifier()
        self.client = OpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="ttps://api.fireworks.ai/inference/v1/chat/completions"
        )
        self.vector_store = load_vector_store()
        self.chat_history = []
        self.last_limitations = None


    def generate_response(self, user_query: str) -> Generator[str, None, None]:
        """Génère une réponse avec streaming"""
        if self.last_limitations != st.session_state.limitations:
            self.chat_history.clear()
            self.last_limitations = st.session_state.limitations

        clarified_query = self.clarifier.clarify_question(user_query)

        
        if not self.vector_store:
            yield "⚠️ Créez un dossier 'docs/' avec des PDFs des formations"
            return

        relevant_docs = self.vector_store.similarity_search(clarified_query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        messages = [
            {
            "role": "system",
            "content": f"""
            **Répondre toujours dans la même langue que l'utilisateur**
            **Rôle** : Assistant spécialisé dans les écoles et programmes de l'Université Mohammed VI Polytechnique (UM6P). Réponds dans la même langue que l'utilisateur.

            **Contexte UM6P** :
            - Fondée en 2013, l'UM6P se concentre sur la recherche, l'innovation et l'enseignement en sciences, ingénierie, technologies, sciences sociales et énergies renouvelables.
            - Mission : Promouvoir le développement durable et l'innovation au Maroc et en Afrique.

            **Établissements et Contacts** :
            1. **EMINES (School of Industrial Management)** :
            - Date de création : 2013
            - Programme : 
                - Cycle Préparatoire Intégré en Management Industriel a Ben Guerir (Date limite de candidature : 1 juin 2025)
                - Cycle Ingénieur en Management Industriel a Ben Guerir (Date limite de candidature : 15 mai 2025)
            - Contact : 
                📧 contact@emines-ingenieur.org | 🌐 emines-ingenieur.org
                

            2. **CC (College of Computing)** :
            - Date de création : 2020
            - Programme : 
                    - Cycle Préparatoire Intégré en Computer Science a Ben Guerir (Date limite de candidature : 20 juin 2025)
                    - Cycle Ingénieur en Computer Science a Ben Guerir (Date limite de candidature : 20 avril 2025)
                    - Cycle Ingénieur en Cyber Security a Rabat (Date limite de candidature : 20 avril 2025)
            - Contact : 
                📧 cc@um6p.ma | 📞 06 69 93 51 50 | 🌐 cc.um6p.ma/engineering-degree

            3. **GTI (Green Tech Institute)** :
            - Date de création : 2020
            - Programme : 
                - Master Ingénierie Electrique pour les Energies Renouvelables et les Réseaux Intelligents a Ben Guerir(📧 Master.RESMA@um6p.ma) (Date limite de candidature : 15 avril 2025)
                - Master Technologies Industrielles pour l'Usine du Futur a Ben Guerir(📧 master.TIUF@um6p.ma) (Date limite de candidature : 15 avril 2025)
            - Contact :
                📧  admission@um6p.ma | 📞 +212 525 073 308 | 🌐 um6p.ma/index.php/fr/green-tech-institute

            4. **SoCheMiB-IST&I** :
            - Programme : Cycle Ingénieur en Génie Chimique, Minéralogique et Biotechnologique a Ben Guerir(Date limite de candidature : 20 juin 2025)
            - Contact : 
                📧 admission@um6p.ma | 📞 +212 525 073 308 | 🌐 um6p.ma/fr/institute-science-technology-innovation

            5. **SAP+D (School of Architecture)** :
            - Date de création : 2019
            - Programme : 
                - Cycle Architecte a Ben Guerir (Bac+6) (Date limite de candidature : 20 juin 2025)
                - Master Ingénierie des Bâtiments Verts et Efficacité Energétique a Ben Guerir (Date limite de candidature : 15 avril 2025)
            - Contact : 
                📧 contactsapd@um6p.ma | 📞 06 69 93 51 50 | 🌐 um6p.ma/fr/sapd-school-architecture-planning-design

            6. **ABS (Africa Business School)** :
            - Date de création : 2016
            - Programmes : 
                - Master AgriBusiness Innovation a Rabat(Date limite de candidature : 15 avril 2025)
                - Master Financial Engineering a Rabat(Date limite de candidature : 15 avril 2025)
                - Master International Management a Rabat(Date limite de candidature : 15 avril 2025)
            - Contact : 
                📧 ali.assmirasse@um6p.ma | 📞 +212 659 46 59 79 | 🌐 abs.um6p.ma

            7. **SHBM (School of Hospitality)** :
            - Date de création : 2020
            - Programme : Bachelor in Hospitality Business & Management a Ben Guerir(Date limite de candidature : 21 avril 2025)
            - Contact : 
                📧 admissions.shbm@um6p.mah | 📞 +212 6 62 10 47 63 | 🌐 www.shbm-um6p.ma

            8. **FMS (Faculty of Medical Sciences)** :
            - Date de création : 2022
            - Programmes : Doctorat en Médecine, Doctorat en Pharmacie a Ben Guerir(Date limite de candidature : 20 juin 2025)
            - Contact : 
                📧 admission-fms@um6p.ma | 📞 +212 525-073051 / +212 665-693326 | 🌐 um6p.ma/fr/faculty-medical-sciences-0

            9. **ISSB-P (Institut Supérieur des Sciences Biologiques et Paramédicales)** :
            - Date de création : 2021
            - Programmes : Licence Soins infirmiers, option infirmier polyvalent a Ben Guerir(Date limite de candidature : 20 juin 2025)
            - Contact : 
                📧 admissionISSBP@um6p.ma | 📞 +212 669 936 049 | 🌐 um6p.ma/en/institute-biological-sciences

            10. **FGSES (Faculty of Governance, Economics and Social Sciences)** :
            - Date de création : 2014
            - Programmes : 
                Programmes Post-Bac :
                * Licence en économie Appliquée a Rabat(Date limite de candidature : 15 mars 2025)
                * Licence en Science Politique a Rabat(Date limite de candidature : 15 mars 2025)
                * Licence en Sciences Comportementales pour
                les Politiques Publiques a Rabat(Date limite de candidature : 15 mars 2025)
                * Licence en Relations Internationales a Rabat(Date limite de candidature : 15 mars 2025)
                * Licence Droit public (Date limite de candidature a Rabat: 15 mars 2025)
                Programmes Master :
                * Master Behavioral and Social Sciences for
                Public Policy (Date limite de candidature a Rabat: 31 mars 2025)
                * Master Global Affairs (Date limite de candidature a Rabat: 31 mars 2025)
                * Master Political Science (Date limite de candidature a Rabat: 31 mars 2025)
                * Master Analyse Economique et Politiques
                Publiques a Rabat (Date limite de candidature : 31 mars 2025)
                * Master Economie Quantitative a Rabat(Date limite de candidature : 31 mars 2025)
            - Contact : 
                📧 Info.fgses@um6p.ma | 📞 +212 (0) 530 431 217 | 🌐 www.fgses-um6p.ma    

            11. **SASE (School of Applied Sciences and Engineering)** :
            - Programme : 
                - Bachelor of Science (B.Sc.) en Applied Sciences and Engineering (4 ans) avec spécialisations a Rabat & Benguerir :(Date limite de candidature : 20 juin 2025)
                    * Mechanical Engineering
                    * Electrical Engineering
                    * Chemical & Material Engineering
                    * Environmental Engineering
                    * Applied Mathematics & Business`
            - Contact : 📧 sase@um6p.ma | 📞 +212 662 32 29 53 | 🌐 sase.um6p.ma
            
            12. **ESAFE (Cycle Ingénieur en Systèmes Agri-Environnementaux)** :
            - Programmes a Benguerir :
                - Cycle Préparatoire Intégré (Date limite de candidature : 20 juin 2025)
                - Cycle Ingénieur en Systèmes Agro-Environnementaux
            - Contact : 
                📧 esafe@um6p.ma | 📞 +212 662 324983 | 🌐 www.um6p.ma/en/ecole-des-sciences-de-lagriculture-de-la-fertilisation-et-de-lenvironnement-esafe        

            13. **School of collective intelligence
            - Programmes : 
                - Master of collective intelligence (Date limite de candidature : 15 april 2025)
            - Contact :
                📧 sci@um6p.ma | https://sci.um6p.ma/
                
                
            
            **Directives Strictes** :
            0. **LIMITATION STRICTE**: 
            {st.session_state.limitations}

            1. **Identification de l'École** :
            - Vérifie TOUJOURS le nom exact de l'école dans la question (ex: "EMINES", "CC", "SAP+D").
            - Si la question mentionne un programme (ex: "Ingénieur en Computer Sciences"), associe-le à l'école correspondante (ex: CC).

            2. **Règles de Réponse** :
            - Répondre uniquement sur la base du contexte:
            - Utilise UNIQUEMENT les informations fournies dans le contexte.
            - Si l'information n'est pas dans le contexte dirigé l'utilisatuer vers le contact de l'école concernée, réponds : "Je ne trouve pas cette information dans ma base de connaissances. Veuillez consulter le site : 🌐 "
            - Si l'école n'est pas claire → Demande : "Veuillez préciser l'école (ex: EMINES, CC, SAP+D)".
            - Toujours donner les contacts officiels de l'école concernée en se basant sur la liste fournie des etablissemtn et contacts.
            - Tu DOIS répondre dans la même langue que celle utilisée par l'utilisateur
            - Si l'utilisateur écrit en français, réponds en français
            - Si l'utilisateur écrit en anglais, réponds en anglais
            - Si l'utilisateur écrit en darija, réponds en darija
            - Si l'utilisateur écrit dans une autre langue, réponds dans cette même langue
            

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

        messages.append({"role": "user", "content": clarified_query})

        try:
            model_name = "accounts/fireworks/models/deepseek-v3p1"
            
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=float(self.temperature),  # Use the temperature parameter
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
            yield f"Erreur : {str(e)}"

    def update_temperature(self, new_temperature: float):
        """Met à jour la température pour les futures réponses"""
        self.temperature = new_temperature

def main():
    st.set_page_config(page_title="UM6P Chatbot", page_icon="🎓")
    
    # Initialisation des limitations
    if 'limitations' not in st.session_state:
        st.session_state.limitations = """     - Tu ne peux répondre qu'aux questions concernant l'UM6P.
            - Pour TOUTE question non liée à l'UM6P, réponds UNIQUEMENT: "Je suis un assistant spécialisé uniquement pour les informations concernant l'UM6P. Je ne peux pas répondre à cette question car elle ne concerne pas l'Université Mohammed VI Polytechnique."
            - Ne jamais répondre à des questions générales, culturelles ou personnelles (par exemple : musique, célébrités, actualités, politique ...)"""

    with st.sidebar:
        st.header("Configuration")
        temperature = st.slider(
            "Créativité des réponses",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="0 = Précis, 1 = Créatif"
        )
        
        limitations = st.text_area(
            "Règles de restriction",
            value=st.session_state.limitations,
            height=200,
            help="Une règle par ligne, utiliser des tirets (-)"
        )
        
        if limitations != st.session_state.limitations:
            st.session_state.limitations = limitations
            st.rerun()

    st.markdown("""
    <style>
    .streaming {background: #f1f3f4; padding: 15px; border-radius: 8px; margin: 10px 0;}
    @keyframes blink {50% {opacity: 0;}}
    .blink-cursor::after {content: "▌"; animation: blink 1s step-end infinite;}
    </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Assistant UM6P")
    st.caption("Posez vos questions sur les programmes et formations de l'UM6P")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFChatbot()
        st.session_state.current_temperature = 0.2

    if st.session_state.current_temperature != temperature:
        st.session_state.chatbot.update_temperature(temperature)
        st.session_state.current_temperature = temperature

    if not st.session_state.chatbot.vector_store:
        st.warning("1. Créez un dossier 'docs/' 2. Ajoutez des PDFs 3. Actualisez la page")

    user_input = st.chat_input("Votre question...")
    
    for msg in st.session_state.chatbot.chat_history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["assistant"])

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response = st.empty()
            full_response = ""
            
            for chunk in st.session_state.chatbot.generate_response(user_input):
                full_response += chunk
                response.markdown(f'<div class="streaming blink-cursor">{full_response}</div>', unsafe_allow_html=True)
            
            response.markdown(f'<div class="streaming">{full_response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
