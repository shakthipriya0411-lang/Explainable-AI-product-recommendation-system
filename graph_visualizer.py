from pyvis.network import Network
import os

def create_graph_visualization(
    G,
    recommended_product=None
):

    if not os.path.exists("static"):
        os.makedirs("static")

    net = Network(
        height="520px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black"
    )

    # LIMIT NODES (IMPORTANT)
    nodes = list(G.nodes())[:60]

    for node in nodes:

        node_str = str(node)

        # USER NODES

        if node_str.startswith("user_"):

            net.add_node(
                node,
                label=node_str,
                color="blue",
                size=10
            )

        # PRODUCT NODES

        elif node_str.startswith("product_"):

            product_id = node_str.replace(
                "product_",
                ""
            )

            # Highlight recommended product

            if (
                recommended_product
                and product_id == str(
                    recommended_product
                )
            ):

                net.add_node(
                    node,
                    label=node_str,
                    color="red",
                    size=25
                )

            else:

                net.add_node(
                    node,
                    label=node_str,
                    color="green",
                    size=12
                )

        else:

            net.add_node(
                node,
                label=node_str,
                color="gray",
                size=8
            )

    # EDGES

    for u, v in G.edges():

        if u in nodes and v in nodes:

            net.add_edge(u, v)

    net.save_graph(
        "static/graph.html"
    )