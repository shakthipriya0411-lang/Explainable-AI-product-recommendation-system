import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np

torch.manual_seed(42)

# -----------------------------
# BUILD GRAPH
# -----------------------------

def build_graph(df):

    G = nx.Graph()

    for _, row in df.iterrows():

        user = f"user_{row['user_id']}"
        product = f"product_{row['product_id']}"

        weight = float(row["sentiment_score"])

        # convert to positive safe range
        weight = abs(weight)

        if weight == 0:
            weight = 0.01

        G.add_edge(
            user,
            product,
            weight=weight
        )

    return G


# -----------------------------
# CREATE NODE FEATURES
# -----------------------------

def create_node_features(G, df):

    feature_dict = {}

    product_stats = df.groupby("product_id").agg({
        "sentiment_score": "mean",
        "rating": "mean",
        "review_text": "count",
        "membership": "mean",
        "non_membership": "mean",
        "hesitation": "mean"
    }).reset_index()

    max_reviews = product_stats["review_text"].max()

    if max_reviews == 0:
        max_reviews = 1

    for _, row in product_stats.iterrows():

        node = f"product_{row['product_id']}"

        feature_dict[node] = [

            float(row["sentiment_score"]),

            float(row["rating"]) / 5.0,

            float(row["review_text"]) / max_reviews,

            float(row["membership"]),

            float(row["non_membership"]),

            float(row["hesitation"])

        ]

    for node in G.nodes():

        if node.startswith("user_"):

            feature_dict[node] = [0, 0, 0, 0, 0, 0]

    return feature_dict


# -----------------------------
# GRAPH TO DATA
# -----------------------------

def graph_to_data(G, df):

    nodes = list(G.nodes())

    node_index = {
        node: i
        for i, node in enumerate(nodes)
    }

    edges = []
    weights = []

    for u, v, data in G.edges(data=True):

        weight = float(data.get("weight", 1.0))

        edges.append([
            node_index[u],
            node_index[v]
        ])

        edges.append([
            node_index[v],
            node_index[u]
        ])

        weights.append(weight)
        weights.append(weight)

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    edge_weight = torch.tensor(
        weights,
        dtype=torch.float
    )

    feature_dict = create_node_features(G, df)

    x = torch.tensor(
        [feature_dict[node] for node in nodes],
        dtype=torch.float
    )

    # normalize features
    x = F.normalize(x, p=2, dim=1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight
    )

    return data, node_index


# -----------------------------
# GCN MODEL
# -----------------------------

class GCNModel(torch.nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(
        self,
        x,
        edge_index,
        edge_weight
    ):

        x = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight
        )

        return x


# -----------------------------
# TRAIN GCN
# -----------------------------

def train_gcn(df):

    print("Building graph...")

    G = build_graph(df)

    data, node_index = graph_to_data(G, df)

    model = GCNModel(
        input_dim=data.num_features
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("Training GCN...")

    for epoch in range(50):

        model.train()

        optimizer.zero_grad()

        out = model(
            data.x,
            data.edge_index,
            data.edge_weight
        )

        loss = torch.mean(out.pow(2))

        if torch.isnan(loss) or torch.isinf(loss):

            print("Invalid loss — skipping epoch")

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch} | Loss: {loss.item():.4f}"
            )

    print("GCN training finished.")

    return model, node_index, data


# -----------------------------
# RECOMMEND PRODUCTS
# -----------------------------

def recommend_products_gcn(
    model,
    node_index,
    data,
    user_id,
    top_n=5
):

    model.eval()

    with torch.no_grad():

        embeddings = model(
            data.x,
            data.edge_index,
            data.edge_weight
        )

    user_node = f"user_{user_id}"

    if user_node not in node_index:

        return []

    user_vector = embeddings[
        node_index[user_node]
    ]

    scores = {}

    for node, idx in node_index.items():

        if node.startswith("product_"):

            product_vector = embeddings[idx]

            score = torch.dot(
                user_vector,
                product_vector
            ).item()

            if np.isnan(score):

                score = 0

            product_id = node.replace(
                "product_",
                ""
            )

            scores[product_id] = score

    sorted_products = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_products[:top_n]