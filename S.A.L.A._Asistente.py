import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import logging
from datetime import datetime
import tenacity
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import io
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
from gtts import gTTS  # Para generar audio en español

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Asistente S.A.L.A. - Summum Projects",
    page_icon="https://summumcorp.com/wp-content/uploads/2022/03/cropped-logo_summit.png",
    layout="wide"
)

# --- Funciones Utilitarias: Estilo y Fondo ---
def render_suggested_questions(questions, callback):
    st.markdown('<div class="suggestions-container">', unsafe_allow_html=True)
    st.write("### Preguntas Sugeridas")
    for question in questions:
        if st.button(question, key=f"suggestion_{question}"):
            callback(question)
    st.markdown('</div>', unsafe_allow_html=True)
def set_background_image(image_url):
    """Establece una imagen de fondo con un overlay y estilos personalizados."""
    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url('{image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        line-height: 1.6;
    }}

    h1, h2, h3, label {{
        color: white !important;
        text-shadow: 1.5px 1.5px 4px rgba(0,0,0,0.8);
        font-weight: 700;
    }}

    label {{
        font-size: 1.1em !important;
        text-shadow: 1px 1px 2px black;
    }}

    .stTextInput > div > div > input {{
        background-color: #f0f2f6;
        color: #000000;
        border: 2px solid #003366;
        border-radius: 10px;
        padding: 10px 15px;
        font-size: 1.1em;
        font-weight: 500;
    }}

    .stSelectbox > div > div > div {{
        background-color: #f0f2f6;
        color: #000000;
    }}

    section[data-testid="stFileUploader"] label {{
        color: white !important;
        font-size: 1.1em;
        text-shadow: 1px 1px 2px black;
    }}

    .stButton > button, .stFormSubmitButton > button {{
        background-color: #FFFFFF;
        color: #003366 !important;
        border: 2px solid #003366;
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: 700;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin: 5px 0;
    }}

    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background-color: #003366;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,51,102,0.4);
    }}

    .chat-container .chat-message .assistant-response {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Establecer la imagen de fondo
set_background_image("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80")

# --- Inicialización de Cliente Groq y Modelos ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Clave API de Groq no encontrada. Revisa tu archivo '.env'.")
    st.stop()

try:
    client = Groq(api_key=api_key)
    LLM_MODEL_NAME = "llama3-8b-8192"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    STT_MODEL_NAME = "whisper-large-v3"
except Exception as e:
    st.error(f"Error al inicializar el cliente Groq: {e}")
    st.stop()

@st.cache_resource
def load_embedding_model():
    """Carga el modelo de embeddings para RAG."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Error al cargar el modelo de embedding '{EMBEDDING_MODEL_NAME}': {e}")
        st.warning("Asegúrate de tener conexión a internet la primera vez que se descarga el modelo.")
        return None

embedding_model = load_embedding_model()
if embedding_model is None:
    st.stop()

# --- Funciones para Extracción de Texto ---
def extract_text_from_pdf(file):
    """Extrae texto de un archivo PDF."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo PDF: {e}")
        return None

def extract_text_from_docx(file):
    """Extrae texto de un archivo DOCX."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return text
    except Exception as e:
        st.error(f"Error al leer el archivo DOCX: {e}")
        return None

def extract_text_from_txt(file):
    """Extrae texto de un archivo TXT."""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error al leer el archivo TXT: {e}")
        return None

def extract_text_from_xlsx(file):
    """Extrae texto de un archivo XLSX."""
    try:
        df = pd.read_excel(file, engine="openpyxl")
        return df.to_string()
    except Exception as e:
        st.error(f"Error al leer el archivo XLSX: {e}")
        return None

def extract_text_from_image(file):
    """Extrae texto de una imagen usando OCR."""
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang="spa")
        return text
    except Exception as e:
        st.error(f"Error al extraer texto de la imagen: {e}")
        return None

from moviepy.editor import VideoFileClip
import tempfile

def extract_text_from_audio(file):
    """Extrae texto de un archivo de audio usando Whisper vía API Groq."""
    try:
        audio_bytes = file.read()
    # Determinar tipo MIME según extensión del archivo
        if file.name.endswith(".wav"):
            mime_type = "audio/wav"
        elif file.name.endswith(".m4a"):
            mime_type = "audio/mp4"
        elif file.name.endswith(".mp3"):
            mime_type = "audio/mpeg"
        else:
            mime_type = "application/octet-stream"
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL_NAME,
            file=(file.name, audio_bytes, mime_type),
            language="es"
        )
        return transcription.text
    except Exception as e:
        st.error(f"Error al transcribir el audio: {e}")
        return None

def extract_text_from_video(file):
    """Extrae texto desde el audio de un video utilizando Whisper vía API Groq."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        video_clip = VideoFileClip(tmp_path)
        audio_clip = video_clip.audio

        if not audio_clip:
            st.error("El video no contiene pista de audio.")
            return None

        audio_path = tmp_path.replace(".mp4", ".wav")
        audio_clip.write_audiofile(audio_path, codec='pcm_s16le')

        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        transcription = client.audio.transcriptions.create(
            model=STT_MODEL_NAME,
            file=("audio.wav", audio_bytes, "audio/wav"),
            language="es"
        )

        return transcription.text

    except Exception as e:
        st.error(f"Error al procesar el video: {e}")
        return None

def extract_text_from_file(file):
    """Extrae texto de un archivo según su tipo."""
    if file is None or file.size == 0:
        st.error("Archivo no válido o vacío.")
        return None
    file_extension = file.name.split('.')[-1].lower()
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "txt": extract_text_from_txt,
        "xlsx": extract_text_from_xlsx,
        "png": extract_text_from_image,
        "jpg": extract_text_from_image,
        "jpeg": extract_text_from_image,
        "mp3": extract_text_from_audio,
        "wav": extract_text_from_audio,
        "mp4": extract_text_from_video,
        "m4a": extract_text_from_audio,
        "avi": extract_text_from_video
    }
    extractor = extractors.get(file_extension)
    if extractor:
        return extractor(file)
    else:
        st.error(f"Formato de archivo no soportado: {file_extension}")
        return None

# --- Funciones para Extracción de Métricas ---
def extract_metrics_from_text(text):
    """Extrae métricas como Plan, Real, Desviaciones y fechas (meses) del texto."""
    metrics = {
        "Plan": [],
        "Real": [],
        "Desviaciones": [],
        "Meses": []
    }
    
    # Patrones para identificar métricas
    plan_pattern = re.compile(r"Plan(?:ificado)?:\s*(\d+\.?\d*)", re.IGNORECASE)
    real_pattern = re.compile(r"Real(?:izado)?:\s*(\d+\.?\d*)", re.IGNORECASE)
    desv_pattern = re.compile(r"Desviaci(?:ón|ones):\s*([-]?\d+\.?\d*)", re.IGNORECASE)
    month_pattern = re.compile(r"(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)\s+\d{4}", re.IGNORECASE)

    # Extraer datos
    for line in text.split("\n"):
        # Plan
        plan_matches = plan_pattern.findall(line)
        if plan_matches:
            metrics["Plan"].extend([float(val) for val in plan_matches])
            month_match = month_pattern.search(line)
            if month_match:
                metrics["Meses"].append(month_match.group(0))
        
        # Real
        real_matches = real_pattern.findall(line)
        if real_matches:
            metrics["Real"].extend([float(val) for val in real_matches])
            month_match = month_pattern.search(line)
            if month_match and month_match.group(0) not in metrics["Meses"]:
                metrics["Meses"].append(month_match.group(0))
        
        # Desviaciones
        desv_matches = desv_pattern.findall(line)
        if desv_matches:
            metrics["Desviaciones"].extend([float(val) for val in desv_matches])
            month_match = month_pattern.search(line)
            if month_match and month_match.group(0) not in metrics["Meses"]:
                metrics["Meses"].append(month_match.group(0))

    # Asegurar que las listas tengan la misma longitud
    min_len = min(len(metrics["Plan"]), len(metrics["Real"]), len(metrics["Desviaciones"]), len(metrics["Meses"]))
    metrics["Plan"] = metrics["Plan"][:min_len]
    metrics["Real"] = metrics["Real"][:min_len]
    metrics["Desviaciones"] = metrics["Desviaciones"][:min_len]
    metrics["Meses"] = metrics["Meses"][:min_len]

    return metrics

# --- Funciones para RAG ---
def get_text_chunks(text):
    """Divide el texto en fragmentos para RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    """Crea un índice vectorial para los fragmentos de texto."""
    if not text_chunks:
        return None, None
    try:
        with st.spinner("Generando embeddings y creando índice vectorial..."):
            embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
        return index, text_chunks
    except Exception as e:
        st.error(f"Error al crear el vector store: {e}")
        return None, None

def get_relevant_chunks(query, vector_store, original_chunks, top_k=3):
    """Recupera los fragmentos más relevantes para una consulta."""
    if vector_store is None or not original_chunks:
        return []
    try:
        query_embedding = embedding_model.encode([query])[0].astype('float32').reshape(1, -1)
        distances, indices = vector_store.search(query_embedding, top_k)
        return [original_chunks[i] for i in indices[0]]
    except Exception as e:
        st.error(f"Error al recuperar chunks relevantes: {e}")
        return []

# --- Funciones de Visualización ---
def generate_summary_chart(metrics, selected_months):
    """Genera gráficos comparativos de Plan vs Real y Desviaciones."""
    if not metrics["Plan"] or not metrics["Real"] or not metrics["Desviaciones"]:
        return None

    # Filtrar datos según los meses seleccionados
    filtered_plan = []
    filtered_real = []
    filtered_desv = []
    filtered_labels = []

    for i, month in enumerate(metrics["Meses"]):
        if month in selected_months:
            filtered_plan.append(metrics["Plan"][i])
            filtered_real.append(metrics["Real"][i])
            filtered_desv.append(metrics["Desviaciones"][i])
            filtered_labels.append(month)

    if not filtered_plan:
        return None

    # Configuración del gráfico
    chart_config = {
        "type": "line",
        "data": {
            "labels": filtered_labels,
            "datasets": [
                {
                    "label": "Plan",
                    "data": filtered_plan,
                    "borderColor": "#FF4500",
                    "backgroundColor": "rgba(255, 69, 0, 0.2)",
                    "fill": True,
                    "tension": 0.4
                },
                {
                    "label": "Real",
                    "data": filtered_real,
                    "borderColor": "#003366",
                    "backgroundColor": "rgba(0, 51, 102, 0.2)",
                    "fill": True,
                    "tension": 0.4
                }
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"labels": {"color": "#FFFFFF"}},
                "title": {"display": True, "text": "Comparativa Plan vs Real", "color": "#FFFFFF"}
            },
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Valores", "color": "#FFFFFF"}, "ticks": {"color": "#FFFFFF"}},
                "x": {"ticks": {"color": "#FFFFFF"}}
            }
        }
    }

    # Gráfico de Desviaciones
    desv_chart_config = {
        "type": "bar",
        "data": {
            "labels": filtered_labels,
            "datasets": [{
                "label": "Desviaciones",
                "data": filtered_desv,
                "backgroundColor": "#f0f2f6",
                "borderColor": "#f0f2f6",
                "borderWidth": 1
            }]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"labels": {"color": "#FFFFFF"}},
                "title": {"display": True, "text": "Desviaciones por Mes", "color": "#FFFFFF"}
            },
            "scales": {
                "y": {"title": {"display": True, "text": "Desviación", "color": "#FFFFFF"}, "ticks": {"color": "#FFFFFF"}},
                "x": {"ticks": {"color": "#FFFFFF"}}
            }
        }
    }

    return chart_config, desv_chart_config

# --- Funciones para Sugerencias de Preguntas ---
def generate_suggested_questions(text):
    """Genera preguntas sugeridas basadas en el contenido del texto."""
    suggestions = []
    
    # Identificar temas clave
    if "riesgo" in text.lower():
        suggestions.append("¿Qué riesgos destacados hay?")
    if "avance" in text.lower() or "real" in text.lower():
        suggestions.append("¿Cuál es el avance acumulado?")
    if "desviaci" in text.lower():
        suggestions.append("¿Qué desviaciones se reportan?")
    if "plan" in text.lower():
        suggestions.append("¿Cómo se compara el Plan con el Real?")

    return suggestions if suggestions else ["¿Qué información clave contiene este documento?"]

# --- Funciones de Procesamiento ---
@st.cache_data(show_spinner=False)
def generate_audio_bytes(text_to_speak: str) -> io.BytesIO | None:
    """Genera un archivo de audio en memoria a partir de texto."""
    if not text_to_speak:
        st.warning("[Depuración] Texto para audio vacío.")
        return None
    cleaned_text = re.sub(r'```.*?```', '', text_to_speak, flags=re.DOTALL).strip()
    if not cleaned_text:
        st.warning("[Depuración] Texto limpio vacío.")
        return None
    if len(cleaned_text) > 4500:
        cleaned_text = cleaned_text[:4500] + "..."
    
    try:
        tts = gTTS(text=cleaned_text, lang='es', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.write("[Depuración] Audio generado con gTTS exitosamente. Tamaño:", audio_fp.getbuffer().nbytes, "bytes")
        return audio_fp
    except Exception as e:
        logger.error(f"Error al generar audio con gTTS: {e}")
        st.error(f"Error al generar audio: {e}")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Reintentando API de Groq... Intento {retry_state.attempt_number}")
)
def call_groq_api(messages, model_name):
    """Llama a la API de Groq para procesar una consulta."""
    if not messages or not isinstance(messages, list):
        raise ValueError("Los mensajes para la API de Groq deben ser una lista no vacía.")
    return client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0.2,
        max_tokens=1000
    )

# --- Funciones de Datos Externos ---
def fetch_summum_data():
    """Obtiene datos de la página de Summum Projects o una página de prueba."""
    url = "https://summumcorp.com/"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = [element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if element.get_text(strip=True) and len(element.get_text(strip=True)) > 20]
        return {"general": " ".join(texts) if texts else "No se pudo obtener información de la página de Summum Projects."}
    except Exception as e:
        logger.error(f"Error al extraer datos de Summum Projects: {e}")
        st.warning(f"No se pudo conectar con la página de Summum Projects. Error: {e}. Usando página de prueba.")
        url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            texts = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20]
            return {"general": " ".join(texts) if texts else "No se pudo obtener información de la página de prueba."}
        except Exception as e:
            logger.error(f"Error al extraer datos de la página de prueba: {e}")
            return {"general": "No se pudo obtener información de ninguna página."}

def get_system_prompt(user_name, preferences=None):
    """Genera el prompt del sistema para el asistente, adaptado a las preferencias del usuario."""
    preferences_str = f"Prioriza información sobre: {', '.join(preferences)}." if preferences else ""
    return f"""
**ROL:**
Eres "S.A.L.A.", un asistente inteligente diseñado por Summum Projects para proporcionar respuestas prácticas y claras basadas en archivos cargados o lecciones aprendidas de los proyectos de Summum Projects. Personaliza tus respuestas dirigiéndote al usuario por su nombre, que es {user_name}, y responde exclusivamente en español.

**CONOCIMIENTO BASE:**
Tu conocimiento se basa en archivos cargados (PDF, Word, Texto, XLSX, Imágenes) o en el contenido extraído de la página principal de Summum Projects (https://summumcorp.com/) o una página de prueba si no se puede acceder a Summum.

**TAREA PRINCIPAL:**
Cuando un usuario te haga una consulta:
1. Busca palabras clave en el contenido del archivo cargado (si hay uno) o en las lecciones aprendidas.
2. Proporciona una respuesta clara y concisa basada en la información encontrada, usando el nombre del usuario ({user_name}) de manera natural.
3. Si no hay archivo cargado y no encuentras información relevante en las lecciones aprendidas, indica que no hay datos disponibles y sugiere cargar un archivo.
4. Adapta el lenguaje para que sea claro y profesional, adecuado para ingenieros y profesionales de proyectos, siempre en español.
5. Cita la fuente si aplica.
6. {preferences_str}

**LIMITACIONES:**
- No inventes datos ni compartas información confidencial.
- No emitas opiniones personales.

**FORMATO DE RESPUESTA:**
- Claridad y concisión.
- Estructura lógica.
- Usa el nombre del usuario ({user_name}) de manera natural.
- Responde exclusivamente en español.
"""

# --- Gestión del Estado ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "original_chunks" not in st.session_state:
    st.session_state.original_chunks = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "welcome_audio_played" not in st.session_state:
    st.session_state.welcome_audio_played = False
if "processing_input" not in st.session_state:
    st.session_state.processing_input = False
if "audio_info_from_mic_recorder" not in st.session_state:
    st.session_state.audio_info_from_mic_recorder = None
if "user_gender" not in st.session_state:
    st.session_state.user_gender = None
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "audio_played_manually" not in st.session_state:
    st.session_state.audio_played_manually = False
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = []
if "welcome_message_shown" not in st.session_state:
    st.session_state.welcome_message_shown = False

# --- Interfaz de Usuario: Barra Lateral ---
with st.sidebar:
    st.image("https://summumcorp.com/wp-content/uploads/2022/03/cropped-logo_summit.png", width=200, use_container_width=True)
    st.header("Opciones de S.A.L.A.")
    st.write("- Responder tus consultas por texto o voz")
    st.write("- Analizar archivos (PDF, Word, Texto, XLSX, Imágenes)")
    if st.session_state.user_name and st.session_state.user_gender:
        st.write("Idioma: Español")

# Menú de navegación principal
    st.header("Sección de Trabajo")
    st.session_state.seccion_seleccionada = st.radio(
        "Selecciona una sección:",
        ["Chat Asistente", "Conversión Multimedia"],
    index=0,
    key="menu_seccion"
)
   
    st.header("Cargar Documento")
    uploaded_file = st.file_uploader(
        "Cargar archivos en formatos PDF, Word, Texto, XLSX, Imagen, Audio o Video (máx. 500MB)", 
        type=["pdf", "docx", "txt", "xlsx", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "avi", "m4a"],
        label_visibility="visible"
    )
    
    if uploaded_file is not None:
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.original_chunks = None
            st.session_state.file_processed = False
            st.session_state.file_name = uploaded_file.name
            st.session_state.raw_text = None
            st.session_state.metrics = None
            st.info(f"Nuevo archivo '{uploaded_file.name}' cargado. Procesando...")
        
        if not st.session_state.file_processed:
            with st.spinner(f"Procesando '{uploaded_file.name}'..."):
                raw_text = extract_text_from_file(uploaded_file)
                if raw_text:
                    st.session_state.raw_text = raw_text
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        st.session_state.vector_store, st.session_state.original_chunks = create_vector_store(text_chunks)
                        st.session_state.metrics = extract_metrics_from_text(raw_text)
                        if st.session_state.vector_store is not None:
                            st.session_state.file_processed = True
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 08:05 PM -05, 01/07/2025
                            st.success(f"Archivo '{uploaded_file.name}' procesado y listo para consultas!")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"¡Hola, {st.session_state.user_name}! He procesado el archivo {uploaded_file.name}. ¿Qué te gustaría saber sobre su contenido?", "timestamp": timestamp}
                            )
                        else:
                            st.error("No se pudo crear el índice vectorial.")
                    else:
                        st.warning("No se pudo extraer texto útil o dividirlo en chunks.")
                else:
                    st.error("No se pudo extraer texto del archivo.")
    elif st.session_state.file_name:  # Si se quita el archivo
        st.session_state.file_processed = False
        st.session_state.file_name = None
        st.session_state.vector_store = None
        st.session_state.original_chunks = None
        st.session_state.raw_text = None
        st.session_state.metrics = None
        st.session_state.messages = []
        st.info("Se ha quitado el archivo. Por favor, carga uno nuevo para continuar.")

    # Preferencias del usuario
    st.header("Tus Preferencias")
    preferences_options = ["Impactos", "Acciones Correctivas", "Riesgos", "Buenas Prácticas","Recomendaciones"]
    selected_preferences = st.multiselect(
        "Selecciona tus temas prioritarios:",
        preferences_options,
        default=st.session_state.user_preferences,
        key="user_preferences_select"
    )
    if selected_preferences != st.session_state.user_preferences:
        st.session_state.user_preferences = selected_preferences
        st.success("Preferencias actualizadas.")

    # Botón de limpieza siempre visible
    if st.button("Limpiar Conversación y Archivo"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.original_chunks = None
        st.session_state.file_processed = False
        st.session_state.file_name = None
        st.session_state.raw_text = None
        st.session_state.metrics = None
        st.session_state.welcome_audio_played = False
        st.session_state.audio_played_manually = False
        st.session_state.user_preferences = []
        st.session_state.welcome_message_shown = False
        st.rerun()

# --- Interfaz de Usuario: Configuración Inicial ---
if st.session_state.user_name is None or st.session_state.user_gender is None:
    st.markdown('<div class="title-container">', unsafe_allow_html=True)
    st.title("Bienvenido(a) a S.A.L.A.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("user_info_form"):
        st.subheader("🧾 Datos del Usuario")
        user_name_input = st.text_input("¿Cuál es tu nombre?", placeholder="Ej: Quiño", key="user_name_input", label_visibility="visible")
        gender_selection = st.selectbox("¿Con qué género te identificas?", ["Seleccionar", "Hombre", "Mujer", "Otro"], key="gender_select")
        submit_user_info = st.form_submit_button("Continuar")
        
        if submit_user_info:
            if user_name_input.strip() and gender_selection != "Seleccionar":
                st.session_state.user_name = user_name_input.strip().capitalize()
                st.session_state.user_gender = gender_selection
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 08:05 PM -05, 01/07/2025
                welcome_message = f"¡Hola, {st.session_state.user_name}! Soy S.A.L.A., tu asistente de Summum Projects. Estoy diseñada principalmente cómo Asistente para responder tus consultas por texto o voz analizando archivos según los criterios que definas y realizar procesos de Conversión de Archivos Multimedia. ¿En qué puedo ayudarte hoy?"
                
                if not st.session_state.welcome_message_shown:
                    st.session_state.messages.append({"role": "assistant", "content": welcome_message, "timestamp": timestamp})
                    st.session_state.welcome_message_shown = True
                
                welcome_audio = generate_audio_bytes(welcome_message)
                if welcome_audio:
                    st.session_state.last_audio_bytes = welcome_audio
                    st.audio(welcome_audio, format="audio/mp3", start_time=0)
                    st.session_state.welcome_audio_played = True
                    st.success("¡Bienvenido(a)! Puedes comenzar tu consulta en texto o voz.")
                else:
                    st.warning("No se pudo generar el audio de bienvenida. Verifica tu conexión a internet.")
                st.rerun()
            else:
                st.warning("Por favor, ingresa tu nombre y selecciona tu género para continuar.")

elif st.session_state.seccion_seleccionada == "Chat Asistente":
    # --- Interfaz de Usuario: Área Principal ---
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image("https://summumcorp.com/wp-content/uploads/2022/03/cropped-logo_summit.png", width=200, use_container_width=True)
    with col2:
        st.markdown('<div class="title-container">', unsafe_allow_html=True)
        st.title("S.A.L.A. (Summum Assistant de Lecciones Aprendidas)")
        st.caption("Respuestas prácticas y claras de archivos o lecciones aprendidas de Summum Projects. Puedes hablar o escribir tu consulta.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.write("")  # Espacio reservado

    # --- Área de Chat ---
    def process_user_query(user_prompt: str):
        """Procesa la consulta del usuario y genera una respuesta."""
        if not user_prompt:
            return
        st.session_state.processing_input = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 08:05 PM -05, 01/07/2025
        st.session_state.messages.append({"role": "user", "content": user_prompt, "timestamp": timestamp})

        with st.chat_message("assistant", avatar="https://summumcorp.com/wp-content/uploads/2022/03/cropped-logo_summit.png"):
            st.markdown("**S.A.L.A. está procesando tu consulta...**")

        if st.session_state.file_processed:
            relevant_chunks = get_relevant_chunks(user_prompt, st.session_state.vector_store, st.session_state.original_chunks)
            context_str = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else "No se encontró información relevante en el archivo."
            system_prompt_rag = f"""
Eres un asistente de IA especializado en responder preguntas basadas EXCLUSIVAMENTE en el siguiente contexto extraído de un documento.
Si la respuesta no se encuentra en el contexto, indica claramente que no puedes responder con la información proporcionada.
No inventes información. Sé conciso y directo.

Contexto del documento:
{context_str}
"""
            messages_for_api = [
                {"role": "system", "content": system_prompt_rag},
                {"role": "user", "content": user_prompt}
            ]
        else:
            context = ""
            for key, value in fetch_summum_data().items():
                if value and isinstance(value, str):
                    for sentence in value.split('.'):
                        if sentence.strip() and any(kw in sentence.lower() for kw in user_prompt.lower().split()):
                            context += sentence.strip() + ". "
            if not context:
                context = "No se encontró información relevante en las lecciones aprendidas o en la página de prueba."
            system_prompt = get_system_prompt(st.session_state.user_name, st.session_state.user_preferences) + f"\n**Contenido Extraído:**\n{context}"
            messages_for_api = [{"role": "system", "content": system_prompt}]
            messages_for_api.extend([{"role": "user", "content": user_prompt}])

        assistant_response = "Lo siento, no pude generar una respuesta en este momento."
        try:
            with st.spinner("Buscando información... ⚡"):
                chat_completion = call_groq_api(messages_for_api, LLM_MODEL_NAME)
                assistant_response = chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al contactar la API LLM de Groq: {e}")
            st.error(f"Error al procesar la consulta: {e}")
        finally:
            st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": timestamp
        })
        st.session_state.last_audio_bytes = generate_audio_bytes(assistant_response)
        st.session_state.processing_input = False
        st.session_state.audio_info_from_mic_recorder = None
        st.rerun()

        # Generar gráficos y sugerencias si hay archivo procesado
        if st.session_state.file_processed:
            # Filtros para gráficos
            st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
            st.write("### Visualización de Datos")
            if st.session_state.metrics and st.session_state.metrics["Meses"]:
                available_months = st.session_state.metrics["Meses"]
                selected_months = st.multiselect(
                    "Filtrar por Meses:",
                    available_months,
                    default=available_months,
                    key="month_filter"
                )
                chart_config, desv_chart_config = generate_summary_chart(st.session_state.metrics, selected_months)
                if chart_config:
                    st.write("#### Comparativa Plan vs Real")
                    st.code(chart_config, "chartjs")
                if desv_chart_config:
                    st.write("#### Desviaciones")
                    st.code(desv_chart_config, "chartjs")
            else:
                st.write("No se encontraron métricas suficientes para generar gráficos.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Sugerencias de preguntas

    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["content"] == "**S.A.L.A. está procesando tu consulta...**" and not st.session_state.processing_input:
                continue
            if message["role"] == "user":
                avatar = "user_man.png" if st.session_state.user_gender == "Hombre" else "user_woman.png" if st.session_state.user_gender == "Mujer" else None
            else:
                avatar = "https://summumcorp.com/wp-content/uploads/2022/03/cropped-logo_summit.png"
            with st.chat_message(message["role"], avatar=avatar if avatar and os.path.exists(avatar) else None):
                if message["role"] == "assistant":
                    st.markdown(f"<div style='color:white; font-size: 17px'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Entrada de Consultas ---
    col_mic, col_input = st.columns([1, 5])
    with col_mic:
        audio_info_mic_recorder_result = mic_recorder(
            start_prompt="▶️ Grabar",
            stop_prompt="⏹️ Detener",
            key='global_recorder_widget',
            use_container_width=True,
            just_once=True
        )
        if audio_info_mic_recorder_result and audio_info_mic_recorder_result.get('bytes') and not st.session_state.processing_input:
            st.session_state.audio_info_from_mic_recorder = audio_info_mic_recorder_result
            st.rerun()
        if st.session_state.audio_info_from_mic_recorder and st.session_state.audio_info_from_mic_recorder.get('duration'):
            st.progress(st.session_state.audio_info_from_mic_recorder['duration'] / 10.0, text="Grabando audio...")
        elif st.session_state.audio_info_from_mic_recorder and st.session_state.audio_info_from_mic_recorder.get('bytes'):
            st.success("¡Audio grabado! Procesando transcripción...")

    with col_input:
        with st.form(key="chat_form", clear_on_submit=True):
            text_prompt_input = st.text_input(
                "Escribe tu consulta aquí...",
                key="text_input_widget",
                placeholder="Ej: ¿Qué dice el archivo sobre X? o ¿Qué lecciones aprendidas hay en proyectos?",
                label_visibility="collapsed",
                disabled=st.session_state.processing_input
            )
            submit_button = st.form_submit_button("Enviar", disabled=st.session_state.processing_input)
            if submit_button and text_prompt_input and not st.session_state.processing_input:
                process_user_query(text_prompt_input)

                if st.session_state.raw_text:
                 suggested_questions = generate_suggested_questions(st.session_state.raw_text)
                render_suggested_questions(suggested_questions, process_user_query)
                
    # --- Manejo de Audio desde Micrófono ---
    if st.session_state.audio_info_from_mic_recorder and st.session_state.audio_info_from_mic_recorder.get('bytes') and not st.session_state.processing_input:
        try:
            with st.spinner(f"Transcribiendo con {STT_MODEL_NAME}... 🎤"):
                audio_bytes = st.session_state.audio_info_from_mic_recorder['bytes']
                transcription = client.audio.transcriptions.create(
                    model=STT_MODEL_NAME,
                    file=("audio.wav", audio_bytes, "audio/wav"),
                    language="es"
                )
            user_prompt_from_audio = transcription.text
            st.info(f"Texto transcrito: \"{user_prompt_from_audio}\"")
            process_user_query(user_prompt_from_audio)
        except Exception as e:
            logger.error(f"Error durante la transcripción: {e}")
            st.error(f"Error durante la transcripción de audio: {e}")
            st.session_state.processing_input = False
            st.session_state.audio_info_from_mic_recorder = None

    if st.session_state.last_audio_bytes:
        # Intentar reproducción automática
        st.audio(st.session_state.last_audio_bytes, format="audio/mp3", autoplay=True)
        st.write(f"[Depuración] Intentando reproducir audio automáticamente para: {st.session_state.user_name}")
        # Botón manual como respaldo
        if st.button("Reproducir Respuesta en Voz Manualmente", key="play_response_audio"):
            st.audio(st.session_state.last_audio_bytes, format="audio/mp3", autoplay=True)
            st.session_state.audio_played_manually = True
            st.write("[Depuración] Audio de respuesta reproducido manualmente.")
        if st.session_state.welcome_audio_played:
            st.session_state.welcome_audio_played = False

elif st.session_state.seccion_seleccionada == "Conversión Multimedia":
    st.title("🔄 Conversor Multimedia S.A.L.A.")
    st.markdown("Convierte entre texto, audio, video, PDF y Word.")

    opcion = st.selectbox("Selecciona la conversión deseada:", [
        "Texto → Audio",
        "Texto → Video",
        "Audio → Video",
        "Video → Audio",
        "PDF/Word → Texto"
    ])

    if opcion == "Audio → Video":
        audio_file = st.file_uploader("Sube un archivo de audio (.mp3)", type=["mp3"])
        texto_opcional = st.text_input("Texto opcional que quieres mostrar en el video:")

    # --- Saludo hablado de S.A.L.A. ---
        saludo = "¡Hola! Soy S.A.L.A., y estoy lista para ayudarte a convertir tu archivo de audio en un video personalizado. Sube tu archivo MP3 para comenzar."
        st.markdown(f"<p style='color:white; font-size:20px; font-weight:bold'>{saludo}</p>", unsafe_allow_html=True)
        st.write("Puedes subir un archivo de audio y opcionalmente un texto para mostrar en el video.")
     
    try:
        tts = gTTS(text=saludo, lang='es')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_saludo:
            tts.save(tmp_saludo.name)

            # Reproducir el saludo (funciona localmente)
            if os.name == "nt":  # Windows
                os.system(f'start {tmp_saludo.name}')
            elif os.name == "posix":  # macOS o Linux
                os.system(f'afplay {tmp_saludo.name}')
    except Exception as e:
        st.warning(f"No se pudo reproducir el saludo: {e}")

    # --- Procesamiento del audio del usuario ---
    if audio_file and st.button("Convertir a Video", key="btn_convertir_audio_video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(audio_file.read())
            ruta_video = audio_a_video(tmp_audio.name)
            st.video(ruta_video)
            with open(ruta_video, "rb") as f:
                st.download_button("Descargar video", f, file_name="audio_en_video.mp4")


        # --- Procesamiento del audio del usuario ---
        if audio_file and st.button("Convertir a Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tmp_audio.write(audio_file.read())
                ruta_video = audio_a_video(tmp_audio.name)
                st.video(ruta_video)
                with open(ruta_video, "rb") as f:
                    st.download_button("Descargar video", f, file_name="audio_en_video.mp4")

        if audio_file and st.button("Convertir a Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tmp_audio.write(audio_file.read())
                ruta_video = audio_a_video(tmp_audio.name)
                st.video(ruta_video)
                with open(ruta_video, "rb") as f:
                    st.download_button("Descargar video", f, file_name="audio_en_video.mp4")

    elif opcion == "Video → Audio":
        video_file = st.file_uploader("Sube un archivo de video (.mp4)", type=["mp4"])
        if video_file and st.button("Extraer Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(video_file.read())
                ruta_audio = video_a_audio(tmp_video.name)
                st.audio(ruta_audio)
                with open(ruta_audio, "rb") as f:
                    st.download_button("Descargar audio extraído", f, file_name="extraido.mp3")

    elif opcion == "PDF/Word → Texto":
        archivo = st.file_uploader("Sube archivo PDF o Word", type=["pdf", "docx"])
        if archivo and st.button("Extraer texto"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(archivo.name)[1]) as tmp_file:
                tmp_file.write(archivo.read())
                texto = texto_desde_pdf_word(tmp_file.name)
                st.text_area("Texto extraído:", texto, height=300)
                st.download_button("Descargar como TXT", texto, file_name="texto_extraido.txt")

    from moviepy.editor import ImageClip, AudioFileClip

from moviepy.editor import ImageClip, AudioFileClip
import os

def audio_a_video(audio_path, video_output="audio_a_video.mp4", imagen_path="Imagen_SALA.png"):
    # Cargar el audio
    audio = AudioFileClip(audio_path)
    duracion = audio.duration

    # Validar si la imagen existe en la ruta esperada
    if not os.path.exists(imagen_path):
        raise FileNotFoundError(f"No se encontró la imagen: {os.path.abspath(imagen_path)}")

    # Crear clip de imagen con duración y tamaño
    imagen = ImageClip(imagen_path).set_duration(duracion).resize(height=720)

    # Asociar el audio al clip de imagen
    imagen = imagen.set_audio(audio)

    # Escribir el archivo de salida en video
    imagen.write_videofile(video_output, fps=24)

    return video_output


# --- Pie de Página con Contacto ---
st.markdown("""<hr style="margin-top:50px; margin-bottom:10px; border: 1px solid #FF4500;">""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px; color: white;'>
    📞 Para mayor información, por favor contacta a <strong>Summum Projects</strong> al teléfono 
    <strong>+57 321 456 7890</strong> o al correo <strong>contacto@summumcorp.com</strong>. 
    <br>Estaremos encantados de ayudarte.
</div>
""", unsafe_allow_html=True)
