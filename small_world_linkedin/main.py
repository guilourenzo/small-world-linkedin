import functools

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network
from scipy.stats import kstest, powerlaw

FILE_PATH = "https://raw.githubusercontent.com/guilourenzo/small-world-linkedin/refs/heads/main/small_world_linkedin/data/linkedin_graph.csv"


# Funções do script original
def safe_read_csv(csv_file, separator="|", **kwargs):
    """
    Safely read CSV with error handling and logging.

    Args:
        csv_file (str): Path to CSV file
        separator (str, optional): CSV separator. Defaults to '|'

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    try:
        df = pd.read_csv(csv_file, sep=separator, **kwargs)
        df["Connected On"] = pd.to_datetime(df["Connected On"], errors="coerce")

        # Basic data validation
        if df.empty:
            st.warning("The dataset is empty. Please check the data source.")

        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()


@functools.lru_cache(maxsize=128)
def process_data(csv_file=FILE_PATH):
    """
    Cached data processing with memoization.

    Args:
        csv_file (str, optional): CSV file path

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    return safe_read_csv(csv_file)


def add_proximity_weights(graph, attribute="company"):
    """
    Add edge weights to the graph based on proximity/similarity of a given attribute.

    Parameters:
        graph (networkx.Graph): The undirected graph.
        attribute (str): Node attribute to calculate similarity (e.g., 'Company').

    Returns:
        graph (networkx.Graph): Updated graph with weighted edges.
    """
    for u, v in graph.edges():
        attr_u = graph.nodes[u].get(attribute, None)
        attr_v = graph.nodes[v].get(attribute, None)
        weight = (
            1 if attr_u and attr_v and attr_u == attr_v else 0.5
        )  # Example weight logic
        graph[u][v]["weight"] = weight

    return graph


def create_graph(dataframe):
    graph = nx.Graph()

    for _, row in dataframe.iterrows():
        main_node = f'{row["First Name"]} {row["Last Name"]}'
        graph.add_node(
            main_node,
            title=f'Company: {row["Company"]}, Position: {row["Position"]}, Days Connected: {row["Days Connected"]}',
            group="Main",
            company=row["Company"],
            position=row["Position"],
            days_connected=row["Days Connected"],
        )

        connection_node = row["Connection Name"]
        graph.add_node(connection_node, group="Connection")
        graph.add_edge(main_node, connection_node)

    graph = add_proximity_weights(graph)

    return graph


def calculate_betweenness(graph):
    """
    Calculate betweenness centrality for nodes in the graph.

    Parameters:
        graph (networkx.Graph): The undirected graph.

    Returns:
        betweenness_dict (dict): Betweenness centrality for each node.
    """
    betweenness_dict = nx.betweenness_centrality(graph, weight="weight")
    return (
        pd.DataFrame.from_dict(betweenness_dict, "index", columns=["BTW"])
        .reset_index()
        .rename(columns={"index": "Nome"})
        .sort_values(by="BTW", ascending=False)
    )


def calculate_average_geodesic_length(graph):
    """
    Calculate the average geodesic length for each node in the graph.

    Parameters:
        graph (networkx.Graph): The undirected graph.

    Returns:
        geodesic_length_dict (dict): Average geodesic length for each node.
    """
    geodesic_length_dict = {}
    for node in graph.nodes():
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="weight")
        geodesic_length_dict[node] = sum(lengths.values()) / (
            len(lengths) - 1
        )  # Exclude self

    return (
        pd.DataFrame.from_dict(geodesic_length_dict, "index", columns=["AGL"])
        .reset_index()
        .rename(columns={"index": "Nome"})
        .sort_values(by="AGL", ascending=False)
    )


def interest_clustering_coefficient_undirected(graph, attribute="Company"):
    """
    Calculate the Interest Clustering Coefficient (ICC) for each node in an undirected graph.

    Parameters:
        graph (networkx.Graph): The undirected graph.
        attribute (str): Node attribute to measure similarity (e.g., 'Company').

    Returns:
        icc_dict (dict): Interest clustering coefficient for each node.
    """
    icc_dict = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 2:
            icc_dict[node] = 0.0
            continue

        same_interest_count = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                attr_i = graph.nodes[neighbors[i]].get(attribute, None)
                attr_j = graph.nodes[neighbors[j]].get(attribute, None)
                if attr_i and attr_j and attr_i == attr_j:
                    same_interest_count += 1

        possible_pairs = len(neighbors) * (len(neighbors) - 1) / 2
        icc_dict[node] = (
            same_interest_count / possible_pairs if possible_pairs > 0 else 0.0
        )

    return (
        pd.DataFrame.from_dict(icc_dict, "index", columns=["ICC"])
        .reset_index()
        .rename(columns={"index": "Nome"})
        .sort_values(by="ICC", ascending=False)
    )


def plot_degree_distribution(graph, min_samples=30):
    try:
        degrees = [degree for _, degree in graph.degree()]
        if len(degrees) < min_samples:
            st.warning(
                f"Dataset too small for comprehensive degree distribution analysis (needs at least {min_samples} samples)."
            )
            return

        # Frequência dos graus
        degree_counts = pd.Series(degrees).value_counts().sort_index()

        # Gráfico de Distribuição
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Bar(
                x=degree_counts.index,
                y=degree_counts.values,
                name="Degree Distribution",
                marker=dict(color="blue"),
            )
        )
        fig_hist.update_layout(
            title="Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Frequency",
            template="plotly_dark",
        )

        # Log-Log Plot
        fig_log = go.Figure()
        fig_log.add_trace(
            go.Scatter(
                x=degree_counts.index,
                y=degree_counts.values,
                mode="markers+lines",
                name="Log-Log Distribution",
                line=dict(dash="dash"),
                marker=dict(color="red", size=8),
            )
        )
        fig_log.update_layout(
            title="Log-Log Degree Distribution",
            xaxis_title="Degree",
            yaxis_title="Frequency",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_dark",
        )

        # Power-Law Fit e Teste KS
        degrees_array = np.array(degrees)
        alpha, loc, scale = powerlaw.fit(degrees_array)
        D, p_value = kstest(degrees_array, "powerlaw", args=(alpha, loc, scale))

        # Gráfico Power-Law
        fig_powerlaw = go.Figure()
        x = np.linspace(min(degrees), max(degrees), 100)
        fig_powerlaw.add_trace(
            go.Histogram(
                x=degrees,
                histnorm="probability density",
                name="Empirical",
                marker=dict(color="green"),
            )
        )
        fig_powerlaw.add_trace(
            go.Scatter(
                x=x,
                y=powerlaw.pdf(x, alpha, loc, scale),
                mode="lines",
                name=f"Power Law Fit (α={alpha:.2f})",
                line=dict(color="red"),
            )
        )
        fig_powerlaw.update_layout(
            title="Power Law Distribution",
            xaxis_title="Degree",
            yaxis_title="Density",
            template="plotly_dark",
        )

        # Exibir gráficos
        st.plotly_chart(fig_hist)
        st.plotly_chart(fig_log)
        st.plotly_chart(fig_powerlaw)

        st.write(f"Power Law Fit: α = {alpha:.2f}")
        st.write(f"Kolmogorov-Smirnov Test: D = {D:.3f}, p-value = {p_value:.3f}")

    except Exception as e:
        st.error(f"Error in degree distribution analysis: {e}")


def compute_centrality_measures(G):
    """
    Compute centrality measures with optimized computation.

    Args:
        G (nx.Graph): Input graph

    Returns:
        tuple: Centrality DataFrame and degree centrality
    """
    # Parallel computation of centrality measures
    centrality_funcs = {
        "Degree Centrality": nx.degree_centrality,
        "Betweenness Centrality": lambda g: nx.betweenness_centrality(
            g, weight="weight"
        ),
        "Eigenvector Centrality": nx.eigenvector_centrality,
    }

    centrality_data = {name: func(G) for name, func in centrality_funcs.items()}

    centrality_df = pd.DataFrame(
        {
            "Node": list(G.nodes()),
            **{name: list(values.values()) for name, values in centrality_data.items()},
            "Days Connected": [
                G.nodes[node].get("days_connected", None) for node in G.nodes()
            ],
        }
    )

    centrality_df["Score"] = centrality_df[list(centrality_funcs.keys())].mean(axis=1)
    return centrality_df.sort_values(
        "Degree Centrality", ascending=False
    ), centrality_data["Degree Centrality"]


# Step 4: Analyze the graph
def analyze_graph(G):
    clustering_coefficient = nx.average_clustering(G)
    average_path_length = (
        nx.average_shortest_path_length(G) if nx.is_connected(G) else None
    )
    return clustering_coefficient, average_path_length


@st.cache_data
def display_pyvis_graph(_graph):
    net = Network(
        notebook=False,
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
    )
    net.from_nx(_graph)
    net_file = "scx_graph.html"
    net.write_html(net_file)
    return net_file


def main():
    st.set_page_config(layout="wide")
    # Centralizar o conteúdo
    st.markdown(
        "<style>.css-1kyxreq {justify-content: center;}</style>", unsafe_allow_html=True
    )
    # Configuração do Streamlit
    st.title("Small-World Analysis of LinkedIn Connections")

    sample_df = process_data()
    if sample_df.empty:
        st.error("Unable to load network data.")
        return

    st.write("Sample Dataset created for analysis")
    st.markdown("""
            This dataset were created extracting a sample from an actual LinkedIn account.  
            To filter the connections, we have used connections related to companies with a minimun of 5 employees.  
  
            To create the connections, the mutual connections were scrapped using Python.
    """)
    st.dataframe(sample_df.head(100))

    st.markdown("---")
    st.subheader("Visual Graph of LinkedIn connection")
    # Criar o grafo
    graph = create_graph(sample_df)
    graph_file = display_pyvis_graph(graph)
    st.components.v1.html(
        open(graph_file, "r", encoding="utf-8").read(), height=800, scrolling=True
    )
    # st.html(graph_file)

    st.markdown("---")
    st.subheader("Graph Evaluation Metrics")

    # Calcular Interest Clustering Coefficient
    icc = interest_clustering_coefficient_undirected(graph, attribute="company")

    # Calcular Betweenness Centrality
    betweenness = calculate_betweenness(graph)

    # Calcular Geodesic Length
    geodesic_lengths = calculate_average_geodesic_length(graph)

    clustering_coefficient, average_path_length = analyze_graph(graph)

    top_influencers, _ = compute_centrality_measures(graph)

    clustering, avg_path = st.columns(2)

    with clustering:
        st.metric(label="Clustering Coefficient", value=f"{clustering_coefficient:.3f}")

    with avg_path:
        st.metric(label="Average Path Length", value=f"{average_path_length:.3f}")

    icc_avg, btw_avg, geo_avg = st.columns(3)

    with icc_avg:
        st.metric(
            label="Average Interest clustering coefficient",
            value=f"{icc.ICC.mean():.3f}",
        )
        st.write("ICC calculated dataset")
        st.dataframe(icc.head(10))

    with btw_avg:
        st.metric(
            label="Average Betweenness Centrality",
            value=f"{betweenness.BTW.mean():.3f}",
        )
        st.write("Betweenness Centrality calculated dataset")
        st.dataframe(betweenness.head(10))

    with geo_avg:
        st.metric(
            label="Average Geodesic Length", value=f"{geodesic_lengths.AGL.mean():.3f}"
        )
        st.write("Geodesic Length calculated dataset")
        st.dataframe(geodesic_lengths.head(10))

    top_k, top_score = st.columns(2)

    with top_k:
        st.write("Top 10 Influencers and Centrality Measures")
        st.dataframe(top_influencers.head(10))

    with top_score:
        st.write("Top 10 Influencer Scores")
        st.dataframe(top_influencers[["Node", "Score"]].head(10))

    st.markdown("---")
    st.subheader("Degree Distribution of LinkedIn Connections")
    st.markdown("""
        The following visuals were created to help evaluate if the current graph has the 'Scale-free' property of a 'Small-world'.  
        
        To evaluate that, we have 3 main Distributions:  
                1. Degree: shows the Degree histogram  
                2. Log-log: evaluates if the Degrees Distribuition follows a 'Scale-free' scale  
                3. Power-law: fits a Power-law and evaluates to evaluate if it follows a 'Scale-free' scale  
          
        Beyond the distributions, we have evaluated the 'Scale-free' using Kolmogorov-Smirnov Test.
        """)
    plot_degree_distribution(graph)

    st.markdown("---")
    st.caption(
        "Projeto desenvolvido durante as aulas do curso SCX5002 - Sistemas Complexos I - EACH USP"
    )
    st.caption("Autor: Guilherme Lourenço | Prof.: Camilo Rodrigues")


if __name__ == "__main__":
    main()
