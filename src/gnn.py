import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np

def json_to_graph(obj, model, parent=None, graph=None, path=""):
    """
    Recursively convert JSON to a graph with path tracking.
    """
    if graph is None:
        graph = nx.DiGraph()
    
    node_id = len(graph.nodes)
    
    # Node label and type
    if isinstance(obj, dict):
        label = "object"
        node_type = "dict"
    elif isinstance(obj, list):
        label = "array"
        node_type = "list"
    else:
        label = str(obj)
        node_type = type(obj).__name__
    
    # Compute embedding with path context
    context_label = f"{path}:{label}" if path else label
    embedding = model.encode([context_label])[0]
    
    graph.add_node(node_id, label=label, feat=torch.tensor(embedding, dtype=torch.float), 
                   type=node_type, path=path)
    
    if parent is not None:
        graph.add_edge(parent, node_id)
    
    # Recurse for dicts and lists
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_id = len(graph.nodes)
            key_emb = model.encode([k])[0]
            new_path = f"{path}.{k}" if path else k
            graph.add_node(key_id, label=k, feat=torch.tensor(key_emb, dtype=torch.float),
                          type="key", path=new_path)
            graph.add_edge(node_id, key_id)
            json_to_graph(v, model, key_id, graph, new_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            json_to_graph(item, model, node_id, graph, new_path)
    
    return graph

def extract_graph_features(graph):
    """
    Extract structural features from the graph.
    """
    if len(graph.nodes) == 0:
        return torch.zeros(10)
    
    features = []
    
    # Basic statistics
    features.append(len(graph.nodes))  # number of nodes
    features.append(len(graph.edges))  # number of edges
    
    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    features.append(np.mean(degrees) if degrees else 0)
    features.append(np.std(degrees) if len(degrees) > 1 else 0)
    features.append(max(degrees) if degrees else 0)
    
    # Type distribution
    type_counts = {}
    for node in graph.nodes:
        node_type = graph.nodes[node].get("type", "unknown")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    features.append(type_counts.get("dict", 0))
    features.append(type_counts.get("list", 0))
    features.append(type_counts.get("key", 0))
    
    # Depth
    if len(graph.nodes) > 0:
        try:
            paths = nx.single_source_shortest_path_length(graph, 0)
            features.append(max(paths.values()) if paths else 0)
        except:
            features.append(0)
    else:
        features.append(0)
    
    # Clustering coefficient (for undirected version)
    try:
        features.append(nx.average_clustering(graph.to_undirected()))
    except:
        features.append(0)
    
    return torch.tensor(features, dtype=torch.float)

def compute_combined_embedding(graph):
    """
    Combine semantic embeddings with structural features.
    """
    if len(graph.nodes) == 0:
        return torch.zeros(384 + 10)
    
    # Semantic embedding (weighted average)
    embeddings = []
    for node in graph.nodes:
        embeddings.append(graph.nodes[node]["feat"])
    
    embeddings = torch.stack(embeddings)
    semantic_embedding = embeddings.mean(dim=0)
    
    # Structural features
    structural_features = extract_graph_features(graph)
    
    # Normalize structural features
    structural_features = F.normalize(structural_features.unsqueeze(0), dim=1).squeeze(0)
    
    # Combine (you can adjust the weight of structural features)
    # Scale structural features to have similar magnitude
    structural_features = structural_features * 10
    
    combined = torch.cat([semantic_embedding, structural_features])
    return combined

def compare_json(json1, json2, model):
    g1 = json_to_graph(json1, model)
    g2 = json_to_graph(json2, model)
    
    z1 = compute_combined_embedding(g1)
    z2 = compute_combined_embedding(g2)
    
    sim = F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0))
    return sim.item()

# Example Usage
if __name__ == "__main__":
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    json_a = {
        "user": {"name": "Alice", "age": 30},
        "active": True
    }
    json_b = {
        "user": {"name": "Alicia", "age": 31},
        "active": "yes"
    }
    
    similarity = compare_json(json_a, json_b, text_model)
    print(f"Similarity score: {similarity:.4f}")