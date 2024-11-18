import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit UI Elements to interact with the user
st.sidebar.title("Network Configuration")
network_size = st.sidebar.slider(
    "Select number of nodes", min_value=10, max_value=500, value=10, step=10
)
connection_prob = st.sidebar.slider(
    "Select connection probability",
    min_value=0.01,
    max_value=1.0,
    value=0.01,
    step=0.01,
)

# Button to reset to default values
default_button = st.sidebar.button("Reset to Default")
if default_button:
    network_size = 10
    connection_prob = 0.01

# Creating a graph with NetworkX based on user input
graph = nx.erdos_renyi_graph(n=network_size, p=connection_prob)

# Calculating Small-World Metrics
avg_path_length = nx.average_shortest_path_length(graph) if nx.is_connected(graph) else "Graph is not connected"
clustering_coeff = nx.average_clustering(graph)

# Displaying metrics
st.title("Small-World Analysis of LinkedIn Connections")
st.write(f"Average Path Length: {avg_path_length}")
st.write(f"Clustering Coefficient: {clustering_coeff}")

# Calculating Degree Distribution to check for Scale-Free properties
degrees = [degree for node, degree in graph.degree()]
degree_count = np.bincount(degrees)
degree_range = range(len(degree_count))

fig_deg = go.Figure()
fig_deg.add_trace(go.Scatter(x=list(degree_range), y=list(degree_count), mode='lines+markers'))
fig_deg.update_layout(title="Degree Distribution of LinkedIn Connections", xaxis_title="Degree", yaxis_title="Count")

st.plotly_chart(fig_deg)

# Visualizing the Graph with Plotly
pos = nx.spring_layout(graph)
xn, yn = zip(*[pos[k] for k in graph.nodes()])
xe, ye = [], []
for e in graph.edges():
    xe += [pos[e[0]][0], pos[e[1]][0], None]
    ye += [pos[e[0]][1], pos[e[1]][1], None]

fig = go.Figure()
fig.add_trace(go.Scatter(x=xe, y=ye, mode='lines', line=dict(color='lightblue'), hoverinfo='none'))
fig.add_trace(go.Scatter(x=xn, y=yn, mode='markers', marker=dict(size=10, color='red'), text=list(graph.nodes()), hoverinfo='text'))
fig.update_layout(title="Graph Representation of LinkedIn Connections", showlegend=False)

st.plotly_chart(fig)

# Calculating Centrality Measures
degree_centrality = nx.degree_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
eigenvector_centrality = nx.eigenvector_centrality(graph)

# Displaying Centrality Measures
st.subheader("Centrality Measures")
centrality_df = pd.DataFrame({
    'Node': list(graph.nodes()),
    'Degree Centrality': [degree_centrality[node] for node in graph.nodes()],
    'Betweenness Centrality': [betweenness_centrality[node] for node in graph.nodes()],
    'Eigenvector Centrality': [eigenvector_centrality[node] for node in graph.nodes()]
})

st.write(centrality_df)