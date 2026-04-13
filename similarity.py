import torch


def get_similar_users(
    model,
    node_index,
    data,
    user_id,
    top_k=3
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

        if node.startswith("user_"):

            if node == user_node:
                continue

            other_vector = embeddings[idx]

            similarity = torch.dot(
                user_vector,
                other_vector
            )

            scores[node] = similarity.item()

    sorted_users = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        u[0]
        for u in sorted_users[:top_k]
    ]