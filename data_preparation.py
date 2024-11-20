import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data

def prepare_data():
    # Step 1: Create a synthetic dataset
    nodes_data = {
        "node_id": range(8),
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah"],
        "hobby": ["Reading", "Gaming", "Cooking", "Traveling", "Music", "Gaming", "Cooking", "Traveling"],
        "favorite_color": ["Red", "Blue", "Green", "Yellow", "Purple", "Blue", "Green", "Yellow"],
        "occupation": ["Engineer", "Artist", "Doctor", "Teacher", "Student", "Artist", "Doctor", "Teacher"],
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Los Angeles", "Chicago", "Houston"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame({
        "source": [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6],
        "target": [1, 2, 3, 4, 5, 6, 7, 0, 7, 5, 3, 1],
    })

    # Step 2: Encode features
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(nodes_df[["hobby", "favorite_color", "occupation", "city"]])
    node_features = torch.tensor(encoded_features, dtype=torch.float)
    node_labels = torch.tensor(nodes_df["label"].values, dtype=torch.long)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    # Step 3: Create PyTorch Geometric data object
    graph_data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    return graph_data

if __name__ == "__main__":
    prepare_data()