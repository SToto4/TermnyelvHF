import os
import json
import re
import time

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import SKOS
from rank_bm25 import BM25Okapi
from groq import Groq, RateLimitError
from graph_visualize import visualize_kg

client = Groq(api_key="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")

DOCS_FOLDER = "./documents"
os.makedirs(DOCS_FOLDER, exist_ok=True)

# RDF Tudásgráf előkészítése
kg = Graph()
EX = Namespace("http://example.org/ontology/")
kg.bind("skos", SKOS)
kg.bind("ex", EX)


def get_active_groq_model():
    available = [m.id for m in client.models.list().data]

    # Konkrét, biztosan működő modellek prioritási sorrendje
    preferred_models = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b"
    ]

    for model in preferred_models:
        if model in available:
            return model

    for m in available:
        if any(prefix in m.lower() for prefix in ("llama", "openai/gpt", "gemma", "mixtral")):
            if not any(bad in m.lower() for bad in ("guard", "vision", "whisper", "tts")):
                return m

    raise RuntimeError(f"Nem található megfelelő szöveges modell. Elérhetőek: {available}")

ACTIVE_MODEL = get_active_groq_model()
print(f"[Info] Kiválasztott aktív Groq modell: {ACTIVE_MODEL}")

def call_llm_safe(prompt, json_mode=False):
    kwargs = {
        "model": ACTIVE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except RateLimitError:
            print("    [!] Token limit elérve, 20 mp várakozás...")
            time.sleep(20)
    raise RuntimeError("Nem sikerült az API hívás 3 próbálkozás után sem.")

def make_safe_uri(text):
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', text.lower())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean if clean else "entity"

# ==========================================
# 1. KATEGORIZÁLÁS ÉS ENTITÁSKINYERÉS (LLM)
# ==========================================
def analyze_and_extract_knowledge(text, filename):
    prompt = f"""
        Elemezd az alábbi magyar szöveget!
        1. Határozd meg a szöveg fő kategóriáját 1-2 szóban (pl. "Egészségügy", "Oktatás/TVSZ", "Informatika", "Sport").
        2. Gyűjtsd ki a kulcsfogalmakat, szakszavakat, és mindegyikhez adj meg 2-3 hétköznapi szinonimát vagy rokon értelmű szót (pl. mérkőzés -> meccs, találkozó, játék).

        KIZÁRÓLAG érvényes JSON formátumban válaszolj:
        {{
          "category": "fő kategória neve",
          "concepts": [
            {{"pref": "szakszó vagy fő fogalom", "alt": "szinonima1, szinonima2, szinonima3"}}
          ]
        }}

        Szöveg:
        \"\"\"{text}\"\"\"
        """
    try:
        raw = call_llm_safe(prompt, json_mode=True)
        return json.loads(raw)
    except Exception as e:
        print(f"Hiba a feldolgozáskor ({filename}): {e}")
        return {"category": "Általános", "concepts": []}


def tokenize_hu(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    tokens = []
    for w in words:
        tokens.append(w)
        if len(w) >= 6:
            tokens.append(w[:6])
    return tokens

def save_to_kg(analysis, filename):
    category = analysis.get("category", "Általános")
    cat_uri = URIRef(EX[make_safe_uri(category)])
    doc_uri = URIRef(EX[make_safe_uri(filename)])

    kg.add((doc_uri, RDF.type, EX.Document))
    kg.add((doc_uri, EX.hasCategory, cat_uri))
    kg.add((cat_uri, SKOS.prefLabel, Literal(category, lang="hu")))

    for item in analysis.get("concepts", []):
        pref = item.get("pref")
        alt = item.get("alt")
        if pref:
            c_uri = URIRef(EX[make_safe_uri(pref)])
            kg.add((c_uri, RDF.type, SKOS.Concept))
            kg.add((c_uri, SKOS.prefLabel, Literal(pref, lang="hu")))
            if alt:
                kg.add((c_uri, SKOS.altLabel, Literal(alt, lang="hu")))
            kg.add((c_uri, SKOS.broader, cat_uri))
            kg.add((c_uri, EX.sourceDoc, doc_uri))

# ==========================================
# 2. SZEMANTIKUS KATEGÓRIA- ÉS SZÓKERESÉS (SPARQL)
# ==========================================
STOPWORDS = {
    "egy", "ezt", "azt", "ami", "aki", "hol", "mikor", "kire", "mire",
    "hogy", "van", "lesz", "volt", "kell", "lehet", "akarok", "szeretnék",
    "milyen", "mennyi", "miért", "és", "vagy", "meg", "nem", "igen"
}

def semantic_routing_and_expansion(query):
    sparql = """
    SELECT ?pref ?alt ?catName WHERE {
        ?c a skos:Concept ;
           skos:prefLabel ?pref ;
           skos:broader ?cat .
        ?cat skos:prefLabel ?catName .
        OPTIONAL { ?c skos:altLabel ?alt }
    }
    """
    results = kg.query(sparql, initNs={"skos": SKOS, "ex": EX})

    clean_query = re.sub(r'[^\w\s]', '', query.lower())
    # Legalább 3 karakteres, nem töltelékszavak megtartása
    q_words = [w for w in clean_query.split() if len(w) >= 3 and w not in STOPWORDS]

    matched_categories = set()
    expanded_words = set(q_words)
    synonyms_found = []

    for r in results:
        pref = str(r.pref).lower()
        alt = str(r.alt).lower() if r.alt else ""
        cat = str(r.catName)
        cat_lower = cat.lower()

        # 1. Kategória nevének közvetlen ellenőrzése (pl. "tvsz" -> "Oktatás/TVSZ")
        for qw in q_words:
            if qw in re.split(r'[\s/]+', cat_lower):
                matched_categories.add(cat)

        # 2. Fogalmak és szinonimák ellenőrzése SZÓELEJI egyezéssel
        target_words = re.sub(r'[^\w\s]', '', f"{pref} {alt}").split()
        is_match = False

        for qw in q_words:
            # 5 karakteres szótő (rövidebb szavaknál a teljes szó)
            stem = qw[:5] if len(qw) >= 5 else qw
            for tw in target_words:
                # Csak akkor illeszkedik, ha a cél-szó EZZEL KEZDŐDIK (nem részszó!)
                if tw.startswith(stem) or (len(tw) >= 4 and qw.startswith(tw[:4])):
                    is_match = True
                    break
            if is_match:
                break

        if is_match:
            matched_categories.add(cat)
            expanded_words.update(re.sub(r'[^\w\s]', '', pref).split())
            if alt:
                expanded_words.update(re.sub(r'[^\w\s]', '', alt).split())
                pair = f"{pref} <-> {alt}"
                if pair not in synonyms_found:
                    synonyms_found.append(pair)

    return list(matched_categories), " ".join(expanded_words), synonyms_found
# ==========================================
# 3. FŐPROGRAM ÉS INTERAKTÍV MŰKÖDÉS
# ==========================================
def main():
    files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")]
    if not files:
        print(f"Tegyél legalább 2-3 .txt fájlt a '{DOCS_FOLDER}' mappába!")
        return

    documents_db = {}
    print("--- DOKUMENTUMOK FELDOLGOZÁSA ÉS BESOROLÁSA ---")
    for f in files:
        with open(os.path.join(DOCS_FOLDER, f), "r", encoding="utf-8") as file:
            txt = file.read().strip()

        print(f"[+] Elemzés: {f}...")
        analysis = analyze_and_extract_knowledge(txt, f)
        category = analysis.get("category", "Általános")
        save_to_kg(analysis, f)

        documents_db[f] = {"text": txt, "category": category}
        print(f"    -> Besorolt kategória: {category}")
        print(f"    -> Kinyert fogalmak száma: {len(analysis.get('concepts', []))}")

    # Gráf lementése fájlba a dokumentációhoz
    kg.serialize("knowledge_graph.ttl", format="turtle")
    kg.serialize("knowledge_graph.ttl", format="turtle")
    visualize_kg("knowledge_graph.ttl", "kg_abra.png")
    print(f"\n[OK] Tudásgráf lementve ('knowledge_graph.ttl', {len(kg)} RDF triplett).")
    print("Rendszer készen áll! Írj be egy kérdést (kilépéshez: exit).")

    while True:
        q = input("\nKérdés: ").strip()
        if q.lower() in ["exit", "q"]:
            break
        if not q:
            continue

        # 1. Kategória felismerése és szinonima-keresés a tudásgráfból
        categories, exp_query, synonyms = semantic_routing_and_expansion(q)

        print("\n--- Szemantikai döntés (Tudásgráf alapján) ---")
        if categories:
            print(f"  Felismerve mint kategória: {categories}")
        else:
            print("  Nem sikerült pontos kategóriát kötni a kérdéshez, minden dokumentumban keresünk.")

        if synonyms:
            print(f"  Aktivált szinonimák: {', '.join(synonyms)}")
        print(f"  Keresőkifejezés: '{exp_query}'")

        # 2. Szűrés: csak az adott kategóriájú dokumentumok kiválasztása
        if categories:
            filtered_docs = {k: v for k, v in documents_db.items() if v["category"] in categories}
            if not filtered_docs:
                filtered_docs = documents_db
        else:
            filtered_docs = documents_db

        # 3. Keresés a szűrt dokumentumok között (BM25)
        doc_names = list(filtered_docs.keys())
        tokenized_corpus = [tokenize_hu(filtered_docs[k]["text"]) for k in doc_names]
        bm25 = BM25Okapi(tokenized_corpus)

        scores = bm25.get_scores(tokenize_hu(exp_query))

        # A legjobb 2 találat kiválasztása a szűrt kategóriából
        top_indices = scores.argsort()[::-1][:2]
        selected_docs = [doc_names[i] for i in top_indices]

        print(
            f"\nKiválasztott forrásdokumentumok (Top {len(selected_docs)}): {selected_docs} (Kategória: {filtered_docs[selected_docs[0]]['category']})")

        combined_context = "\n\n---\n\n".join(
            [f"DOKUMENTUM ({name}):\n{filtered_docs[name]['text']}" for name in selected_docs]
        )

        # 4. Természetes nyelvű, igényes LLM válasz megfogalmazása
        rag_prompt = f"""
                A felhasználó kérdést tett fel. Az alábbi hivatalos források alapján válaszolj rá barátságos, 
                de szakszerű és tömör magyar mondatokkal. Csak olyan állítást tegyél, ami a forrásokból következik, 
                és jelöld meg, melyik dokumentumból származik az információ!

                FORRÁSDOKUMENTUMOK:
                {combined_context}

                FELHASZNÁLÓ KÉRDÉSE:
                {q}
                """
        answer_text = call_llm_safe(rag_prompt)

        print("\n=== Válasz a felhasználónak ===")
        print(answer_text)


if __name__ == "__main__":
    main()