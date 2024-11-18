import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np


# Streamlit UI Elements to interact with the user
st.sidebar.title("Network Configuration")
network_size = st.sidebar.slider(
    "Select number of nodes", min_value=10, max_value=1000, value=550, step=10
)


top_k = st.sidebar.slider(
    "Select top k most influential nodes", min_value=3, max_value=20, value=5, step=1
)

# Button to reset to default values
default_button = st.sidebar.button("Reset to Default")
if default_button:
    network_size = 550
    top_k = 5


# Carregando o dataset
file_path = 'https://raw.githubusercontent.com/guilourenzo/small-world-linkedin/refs/heads/main/small_world_linkedin/data/simulated_linkedin_connections.csv'
data = pd.read_csv(file_path)

# Criando um grafo com NetworkX baseado nos dados fornecidos
graph = nx.from_pandas_edgelist(
    data.loc[:network_size],
    source='Origem',
    target='Destino',
    edge_attr=['Conexões em Comum (Peso)', 'Tipo de Conexão', 'Empresa (Grupo)'],
    create_using=nx.Graph()
)

# Adicionando atributos de peso às arestas
for u, v, data in graph.edges(data=True):
    data['weight'] = data['Conexões em Comum (Peso)']


# Creating a graph with NetworkX based on user input
# graph = nx.erdos_renyi_graph(n=network_size, p=connection_prob)

# Calculating Small-World Metrics
avg_path_length = (
    nx.average_shortest_path_length(graph, weight='weight')
    if nx.is_connected(graph)
    else "Graph is not connected"
)
clustering_coeff = nx.average_clustering(graph)

# Displaying metrics
st.title("Small-World Analysis of LinkedIn Connections")
st.write(
    f"Average Path Length: {avg_path_length if isinstance(avg_path_length, str) else round(avg_path_length, 3)}"
)
st.write(f"Clustering Coefficient: {round(clustering_coeff, 3)}")

# Calculating Degree Distribution to check for Scale-Free properties
degrees = [degree for node, degree in graph.degree()]
degree_count = np.bincount(degrees)
degree_range = range(len(degree_count))

# Visualizing the Graph with Plotly
pos = nx.spring_layout(graph)
xn, yn = zip(*[pos[k] for k in graph.nodes()])
xe, ye = [], []
for e in graph.edges():
    xe += [pos[e[0]][0], pos[e[1]][0], None]
    ye += [pos[e[0]][1], pos[e[1]][1], None]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=xe, y=ye, mode="lines", line=dict(color="lightblue"), hoverinfo="none")
)
fig.add_trace(
    go.Scatter(
        x=xn,
        y=yn,
        mode="markers",
        marker=dict(size=10, color="red"),
        text=list(graph.nodes()),
        hoverinfo="text",
    )
)
fig.update_layout(
    title="Graph Representation of LinkedIn Connections", showlegend=False
)

st.plotly_chart(fig)

# Creating Degree Distribution Graph with both bar and line chart
fig_deg = go.Figure()
fig_deg.add_trace(
    go.Bar(x=list(degree_range), y=list(degree_count), name="Degree histogram")
)
fig_deg.add_trace(
    go.Scatter(
        x=list(degree_range),
        y=list(degree_count),
        mode="lines+markers",
        name="Degree distribution",
    )
)
fig_deg.update_layout(
    title="Degree Distribution of LinkedIn Connections",
    xaxis_title="Degree",
    yaxis_title="Node count",
)

st.plotly_chart(fig_deg)


centrality_col, top_k_col = st.columns(2)
# Calculating Centrality Measures
degree_centrality = nx.degree_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
eigenvector_centrality = nx.eigenvector_centrality(graph)

with centrality_col:
    # Displaying Centrality Measures
    st.subheader("Centrality Measures")
    centrality_df = pd.DataFrame(
        {
            "Node": list(graph.nodes()),
            "Degree Centrality": [degree_centrality[node] for node in graph.nodes()],
            "Betweenness Centrality": [
                betweenness_centrality[node] for node in graph.nodes()
            ],
            "Eigenvector Centrality": [
                eigenvector_centrality[node] for node in graph.nodes()
            ],
        }
    )

    st.write(centrality_df)

with top_k_col:
    # Displaying Top k Most Influential Nodes
    st.subheader(f"Top {top_k} Most Influential Nodes")
    centrality_df["Score"] = centrality_df[
        ["Degree Centrality", "Betweenness Centrality", "Eigenvector Centrality"]
    ].mean(axis=1)
    top_k_nodes = centrality_df.nlargest(top_k, "Score")

    st.write(top_k_nodes[["Node", "Score"]])

st.markdown("---")
st.caption("Projeto desenvolvido durante as aulas do curso SCX5002 - Sistemas Complexos I - EACH USP")
st.caption("Autor: Guilherme Lourenço | Prof.: Camilo Rodrigues")