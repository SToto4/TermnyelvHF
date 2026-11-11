import os
import json
import re
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import SKOS
from rank_bm25 import BM25Okapi
from google import genai

# Gemini Kliens inicializálása
client = genai.Client(api_key="AQ.Ab8RN6L6mupNIKzf-ElyaW301Cmm5PgTONMOGF7Foh9tmXGenQ")

DOCS_FOLDER = "./documents"
os.makedirs(DOCS_FOLDER, exist_ok=True)

# RDF Tudásgráf előkészítése
kg = Graph()
EX = Namespace("http://example.org/ontology/")
kg.bind("skos", SKOS)
kg.bind("ex", EX)


# ==========================================
# 1. KATEGORIZÁLÁS ÉS ENTITÁSKINYERÉS (LLM)
# ==========================================
def analyze_and_extract_knowledge(text, filename):
    """
    Az LLM kinyeri a dokumentum fő témakörét (kategóriáját),
    valamint a kulcsfogalmakat és szinonimákat.
    """
    prompt = f"""
    Elemezd az alábbi magyar szöveget!
    1. Határozd meg a szöveg fő kategóriáját 1-2 szóban (pl. "Egészségügy", "Oktatás/TVSZ", "Informatika", "Jog").
    2. Gyűjtsd ki a szakszavakat és azok hétköznapi szinonimáit.

    KIZÁRÓLAG érvényes JSON formátumban válaszolj:
    {{
      "category": "fő kategória neve",
      "concepts": [
        {{"pref": "szakszó vagy fő fogalom", "alt": "hétköznapi szinonima vagy rokon fogalom"}}
      ]
    }}

    Szöveg:
    \"\"\"{text}\"\"\"
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\n|\n```$", "", raw, flags=re.MULTILINE)
        return json.loads(raw)
    except Exception as e:
        print(f"Hiba a feldolgozáskor ({filename}): {e}")
        return {"category": "Általános", "concepts": []}


def save_to_kg(analysis, filename):
    """Eltárolja a kategóriát, a dokumentumot és a fogalmakat RDF-ben."""
    category = analysis.get("category", "Általános")
    cat_uri = URIRef(EX[category.replace(" ", "_").lower()])
    doc_uri = URIRef(EX[filename.replace(" ", "_")])

    # Doksi és kategória kapcsolata
    kg.add((doc_uri, RDF.type, EX.Document))
    kg.add((doc_uri, EX.hasCategory, cat_uri))
    kg.add((cat_uri, SKOS.prefLabel, Literal(category, lang="hu")))

    # Fogalmak felvétele a kategória alá
    for item in analysis.get("concepts", []):
        pref = item.get("pref")
        alt = item.get("alt")
        if pref:
            c_uri = URIRef(EX[pref.replace(" ", "_").lower()])
            kg.add((c_uri, RDF.type, SKOS.Concept))
            kg.add((c_uri, SKOS.prefLabel, Literal(pref, lang="hu")))
            if alt:
                kg.add((c_uri, SKOS.altLabel, Literal(alt, lang="hu")))
            kg.add((c_uri, SKOS.broader, cat_uri))
            kg.add((c_uri, EX.sourceDoc, doc_uri))


# ==========================================
# 2. SZEMANTIKUS KATEGÓRIA- ÉS SZÓKERESÉS (SPARQL)
# ==========================================
def semantic_routing_and_expansion(query):
    """
    Megkeresi a kérdés szavait a gráfban:
    - Kideríti, melyik kategóriába tartozik a kérdés
    - Kibővíti a kérdést szinonimákkal
    """
    sparql = """
    PREFIX skos: [http://www.w3.org/2004/02/skos/core#](http://www.w3.org/2004/02/skos/core#)
    PREFIX ex: [http://example.org/ontology/](http://example.org/ontology/)
    SELECT ?pref ?alt ?catName WHERE {
        ?c a skos:Concept ;
           skos:prefLabel ?pref ;
           skos:broader ?cat .
        ?cat skos:prefLabel ?catName .
        OPTIONAL { ?c skos:altLabel ?alt }
    }
    """
    results = kg.query(sparql)
    q_lower = query.lower()

    matched_categories = set()
    expanded_words = set(q_lower.split())
    synonyms_found = []

    for r in results:
        pref = str(r.pref).lower()
        alt = str(r.alt).lower() if r.alt else ""
        cat = str(r.catName)

        if pref in q_lower or (alt and alt in q_lower):
            matched_categories.add(cat)
            expanded_words.add(pref)
            if alt:
                expanded_words.add(alt)
                synonyms_found.append(f"{pref} <-> {alt}")

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

    print("\n[OK] Rendszer készen áll! Írj be egy kérdést (kilépéshez: exit).")

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
            if not filtered_docs:  # Ha valamiért üres lenne, visszaugrunk az összesre
                filtered_docs = documents_db
        else:
            filtered_docs = documents_db

        # 3. Keresés a szűrt dokumentumok között (BM25)
        doc_names = list(filtered_docs.keys())
        tokenized_corpus = [filtered_docs[k]["text"].lower().split() for k in doc_names]
        bm25 = BM25Okapi(tokenized_corpus)

        scores = bm25.get_scores(exp_query.lower().split())
        best_idx = scores.argmax()
        best_doc_name = doc_names[best_idx]
        best_context = filtered_docs[best_doc_name]["text"]

        print(f"\nLegrelevánsabb dokumentum: {best_doc_name} (Kategória: {filtered_docs[best_doc_name]['category']})")

        # 4. Természetes nyelvű, igényes LLM válasz megfogalmazása
        rag_prompt = f"""
        A felhasználó kérdést tett fel. Az alábbi hivatalos forrás alapján válaszolj rá barátságos, 
        de szakszerű és tömör magyar mondatokkal. Csak olyan állítást tegyél, ami a forrásból következik!

        FORRÁSDOKUMENTUM ({best_doc_name}):
        {best_context}

        FELHASZNÁLÓ KÉRDÉSE:
        {q}
        """
        answer = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rag_prompt,
        )

        print("\n=== Válasz a felhasználónak ===")
        print(answer.text)


if __name__ == "__main__":
    main()