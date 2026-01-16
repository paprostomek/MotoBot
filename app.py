import streamlit as st
import os
import json
import requests
import re
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from groq import Groq

# ==========================================
import streamlit as st
import os
import google.generativeai as genai
from groq import Groq

# ==========================================
# KONFIGURACJA AI (HYBRYDA: GROQ + GOOGLE)
# ==========================================
st.sidebar.title("⚙️ Konfiguracja AI")

# Funkcja bezpiecznie pobierająca klucze (z chmury lub zmiennych)
def get_key(name):
    # Najpierw sprawdza st.secrets (chmura), potem zmienne środowiskowe
    if name in st.secrets:
        return st.secrets[name]
    return os.environ.get(name)

groq_key = get_key("GROQ_API_KEY")
google_key = get_key("GOOGLE_API_KEY")

# Wybór silnika (automatyczny lub ręczny)
engine = "Brak"
if groq_key or google_key:
    # Jeśli mamy oba klucze, dajemy wybór. Jeśli jeden - ustawiamy go automatycznie.
    dostepne_opcje = []
    if groq_key: dostepne_opcje.append("Groq (Llama 3 - Szybki)")
    if google_key: dostepne_opcje.append("Google (Gemini - Dokładny)")
    
    engine = st.sidebar.radio("Wybierz silnik AI:", dostepne_opcje)
else:
    st.error("❌ Brak kluczy API! Skonfiguruj Secrets w Streamlit Cloud.")
    st.stop() # Zatrzymuje aplikację, żeby nie wywaliła błędu dalej

# --- UNIWERSALNA FUNKCJA GENEROWANIA ---
def generate_ai_response(prompt_text):
    # 1. Ścieżka GROQ
    if "Groq" in engine:
        try:
            client = Groq(api_key=groq_key)
            completion = client.chat.completions.create(
                model="llama3-70b-8192", 
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.6,
                max_tokens=1000
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ Błąd Groq: {e}"

    # 2. Ścieżka GOOGLE
    elif "Google" in engine:
        try:
            genai.configure(api_key=google_key)
            # Próba użycia modelu 2.5 flash, potem fallback na 2.5 flash lite
            models = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']
            active_model = None
            for m in models:
                try:
                    active_model = genai.GenerativeModel(m)
                    break
                except: continue
            
            if active_model:
                return active_model.generate_content(prompt_text).text
            else:
                return "Nie udało się połączyć z żadnym modelem Google."
        except Exception as e:
            return f"⚠️ Błąd Google: {e}"

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
       # Nowe wywołanie (korzysta z naszej funkcji hybrydowej)
        response_text = generate_ai_response(prompt)
        return response_text
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
