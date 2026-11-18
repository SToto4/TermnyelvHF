import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph
from rdflib.namespace import SKOS

def visualize_kg(ttl_path="knowledge_graph.ttl", output_image="kg_abra.png"):
    g = Graph()
    g.parse(ttl_path, format="turtle")

    G = nx.DiGraph()
    for s, p, o in g:
        if p == SKOS.altLabel:
            pref = g.value(s, SKOS.prefLabel)
            if pref:
                G.add_edge(str(pref), str(o), label="szinonima")
        elif p == SKOS.broader:
            pref = g.value(s, SKOS.prefLabel)
            cat = g.value(o, SKOS.prefLabel)
            if pref and cat:
                G.add_edge(str(pref), str(cat), label="kategória")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="#A7C7E7", node_size=2200, font_size=8, font_weight="bold", arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Generált RDF/SKOS Tudásgráf részlete")
    plt.savefig(output_image, dpi=200, bbox_inches="tight")
    plt.close() # Bezárja a memóriából, ne akassza meg a konzolos ciklust!
    print(f"[OK] Tudásgráf ábra elmentve: '{output_image}'")

# Ha külön futtatod a fájlt, akkor is működjön:
if __name__ == "__main__":
    visualize_kg()