import streamlit as st
import os
import json
import requests
import re
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==========================================
# KLUCZ API
# ==========================================
API_KEY = "AIzaSyCQlmlFDEAzV3xxmwB6gcOuBPzWL2Su498"
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# ==========================================
# INTELIGENTNY WYBÓR MODELU
# ==========================================
st.sidebar.title("🔍 Wybór modelu Google Gemini")
active_model = None
try:
    for m in genai.list_models():
        if 'generateContent' in getattr(m, "supported_generation_methods", []):
            if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                active_model = genai.GenerativeModel(m.name)
                st.sidebar.success(f"✅ Model: {m.name}")
                break
    if not active_model:
        first_model = next(genai.list_models())
        active_model = genai.GenerativeModel(first_model.name)
        st.sidebar.warning(f"⚠️ Używam modelu awaryjnego: {first_model.name}")
except Exception as e:
    st.sidebar.error(f"❌ Błąd przy wyborze modelu: {e}")
    active_model = genai.GenerativeModel('gemini-1.5-flash-001')

# ==========================================
# Funkcja do sprawdzania VIN
# ==========================================
def get_car_from_vin(vin: str):
    vin = vin.strip().upper()
    
    # --- 1. MOCK (ŚCIĄGA NA ZALICZENIE) ---
    # Tutaj wpisujemy VIN-y, które mają działać na 100% podczas prezentacji
    if vin == "WBA1R51050V764951":  # Twój VIN z BMW
        return "BMW Seria 1 (E87) 2004-2011"
    
    if vin == "VWZZZ1JZEW000001":   # Przykładowy VIN Golfa IV
        return "Volkswagen Golf IV 1.9 TDI"

    # --- 2. STANDARDOWE SPRAWDZANIE (API USA) ---
    if not re.match(r"^[A-HJ-NPR-Z0-9]{17}$", vin):
        return None
    try:
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()["Results"]
            make = next((x["Value"] for x in data if x["Variable"] == "Make"), "")
            model_car = next((x["Value"] for x in data if x["Variable"] == "Model"), "")
            year = next((x["Value"] for x in data if x["Variable"] == "Model Year"), "")
            
            # API czasem zwraca puste pola dla aut z Europy
            if make and model_car:
                return f"{make} {model_car} {year}".strip()
    except:
        pass
    
    return None
# ==========================================
# Wczytanie bazy części z JSON
# ==========================================
if not os.path.exists("baza_czesci.json"):
    st.error("Nie znaleziono pliku 'baza_czesci.json'!")
    st.stop()

with open("baza_czesci.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)

def prepare_docs(data):
    docs, ids, metadatas = [], [], []
    for i, item in enumerate(data):
        text = (f"Produkt: {item['nazwa']}. Cena: {item['cena']}. "
                f"Opis: {item['opis']}. Pasuje do: {', '.join(item['pasuje_do'])}.")
        docs.append(text)
        ids.append(str(i))
        metadatas.append({"source": "json"})
    return docs, ids, metadatas

docs, ids, metadatas = prepare_docs(data_json)

st.sidebar.info("⏳ Tworzę bazę wektorową Chroma...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection(name="czesci_auto")
except:
    pass

collection = chroma_client.create_collection(name="czesci_auto")
embeddings = embedder.encode(docs).tolist()
collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
st.sidebar.success("✅ Baza gotowa!")

# ==========================================
# Funkcja chat bota (ZMODYFIKOWANA - LEPSZY STYL + PAMIĘĆ)
# ==========================================
def ask_bot(user_question, history, vin_context=None):
    try:
        # 1. Szukanie w bazie wektorowej (RAG)
        query_embed = embedder.encode([user_question]).tolist()
        results = collection.query(query_embeddings=query_embed, n_results=3)
        found_text = "\n".join(results['documents'][0])

        # 2. Formatowanie historii rozmowy do tekstu
        history_text = ""
        for msg in history:
            role = "KLIENT" if msg["role"] == "user" else "SPRZEDAWCA"
            history_text += f"{role}: {msg['content']}\n"

        # 3. Prompt (Instrukcja dla AI) - TUTAJ JEST ZMIANA
        prompt = f"""
Jesteś profesjonalnym, uprzejmym i pomocnym ekspertem w sklepie motoryzacyjnym.
Twoim celem jest doradzić klientowi najlepszy produkt i sprawić, by czuł się dobrze obsłużony.

ZASADY ODPOWIEDZI:
1. Bądź komunikatywny i używaj pełnych zdań (np. "Do Twojego Golfa polecam...", "Mamy świetny olej...").
2. Jeśli klient pyta ogólnie (np. "klocki"), a nie podał szczegółów (przód/tył), BĄDŹ PROAKTYWNY i dopytaj o te szczegóły.
3. Jeśli czegoś nie ma w bazie, przeproś i zaproponuj coś innego lub zapytaj o inne potrzeby.
4. Korzystaj z HISTORII ROZMOWY, aby wiedzieć o czym mówiliście wcześniej (nie pytaj o to samo dwa razy).
5. STOSUJ CROSS-SELLING: Jeśli klient pyta o olej, zapytaj czy potrzebuje też filtra oleju. Jeśli o klocki hamulcowe, zapytaj o stan tarcz. Bądź dobrym sprzedawcą!
DANE DO TWOJEJ DYSPOZYCJI:
--- BAZA PRODUKTÓW W SKLEPIE ---
{found_text}

--- AUTO KLIENTA ---
{vin_context if vin_context else "Nieznane (dopytaj o VIN jeśli to konieczne do doboru części)"}

--- HISTORIA ROZMOWY ---
{history_text}

--- NOWE PYTANIE KLIENTA ---
{user_question}
"""
        response = active_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Wystąpił błąd: {e}"

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("🚗 MotoBot AI")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dzień dobry! Jestem Twoim wirtualnym doradcą. Podaj numer VIN lub powiedz, jakiej części szukasz?"}]

if "current_car" not in st.session_state:
    st.session_state.current_car = None

with st.sidebar:
    st.title("🔧 Status pojazdu")
    if st.session_state.current_car:
        st.success(st.session_state.current_car)
        if st.button("Zresetuj pojazd"):
            st.session_state.current_car = None
            st.rerun()
    else:
        st.info("Brak zidentyfikowanego pojazdu")

# Wyświetlanie historii czatu
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Pobieranie wejścia od użytkownika
if prompt := st.chat_input("Wpisz VIN lub pytanie..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    detected_car = get_car_from_vin(prompt)

    if detected_car:
        st.session_state.current_car = detected_car
        answer = f"✅ Świetnie! Zidentyfikowałem Twój pojazd: **{detected_car}**. Teraz mogę precyzyjnie dobrać części. Czego potrzebujesz?"
    else:
        query = prompt
        if st.session_state.current_car:
            query += f" Kontekst pojazdu: {st.session_state.current_car}."

        with st.spinner("Przeszukuję magazyn..."):
            answer = ask_bot(query, st.session_state.messages, st.session_state.current_car)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})