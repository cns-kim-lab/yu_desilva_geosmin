from color_utils import * 
from make_network import *
import numpy as np 
import pandas as pd 
import os 
import pickle
from pathlib import Path


GRAY_RGBA = {
    "a": 1.0,
    "r": 125,
    "g": 125,
    "b": 125
}


def load_gexf_graphs(graph_paths):
    """
    여러 GEXF 파일을 dictionary 형태로 불러온다.

    Parameters
    ----------
    graph_paths : dict[str, str | Path]
        예:
        {
            "tarsal_sweet": "...gexf",
            "labial_sweet": "...gexf",
        }

    Returns
    -------
    dict[str, nx.Graph]
    """
    return {
        name: nx.read_gexf(Path(path))
        for name, path in graph_paths.items()
    }


def get_rank_counts_and_selected_nodes(
    rank_df,
    top_n=20,
    min_count=3,
):
    """
    각 node가 여러 조건에서 top_n 안에 포함된 횟수를 계산하고,
    min_count 이상 포함된 node를 선택한다.
    """
    counts = (rank_df <= top_n).sum(axis=1).astype(int)

    counts.index = counts.index

    selected_nodes = (
        counts[counts >= min_count]
        .index
        .to_numpy(dtype=int)
    )

    return counts.to_dict(), selected_nodes


def group_ids_to_cell_ids(
    group_ids,
    cell_ids,
    group_info,
):
    """
    group/type ID에 해당하는 FlyWire cell ID를 반환한다.
    """
    group_ids = np.asarray(group_ids, dtype=int)
    group_info = np.asarray(group_info)

    return np.asarray(cell_ids)[np.isin(group_info, group_ids)]


def rgba_to_gephi_dict(rgba):
    """
    RGBA array를 Gephi/GEXF node color dictionary로 변환한다.
    """
    rgba = np.asarray(rgba, dtype=float)

    return {
        "a": float(rgba[3]),
        "r": int(rgba[0]),
        "g": int(rgba[1]),
        "b": int(rgba[2]),
    }


def make_rank_color_info(
    rank_counts,
    gid2canonical_name,
    base_color,
    max_count=4,
    lightening_factor=0.5,
):
    """
    rank 포함 횟수에 따라 node 색상을 생성한다.

    현재 기준:
    - 모든 조건에서 선택된 node: base_color
    - 그 외: 밝게 처리된 base_color
    """
    base_rgba = np.asarray(hex_to_rgba(base_color), dtype=float)

    light_rgb = lighten_color(
        base_rgba[:3],
        factor=lightening_factor,
    )
    light_rgba = np.array([*light_rgb, base_rgba[3]])

    color_info = {}

    for group_id, count in rank_counts.items():
        node_name = str(gid2canonical_name[group_id])

        rgba = base_rgba if count == max_count else light_rgba
        color_info[node_name] = rgba_to_gephi_dict(rgba)

    return color_info


def set_gray_node_colors(
    color_info,
    nodes,
    gray_color=None,
):
    """
    지정한 node들의 색상을 회색으로 설정한다.
    """
    if gray_color is None:
        gray_color = GRAY_RGBA

    for node in nodes:
        color_info[str(node)] = gray_color.copy()

    return color_info


def combine_unique_cells(*cell_groups):
    """
    여러 cell ID array를 합치고 중복을 제거한다.
    """
    arrays = [
        np.asarray(group)
        for group in cell_groups
        if group is not None and len(group) > 0
    ]

    if not arrays:
        return np.array([], dtype=int)

    return np.unique(np.concatenate(arrays))


def add_dummy_nodes_from_pickle(
    graph,
    pickle_path,
    color_info,
    gray_color=None,
):
    """
    pickle에 저장된 node 중 graph에 없는 node를 dummy node로 추가한다.
    """
    if gray_color is None:
        gray_color = GRAY_RGBA

    with open(pickle_path, "rb") as file:
        dummy_nodes = pickle.load(file)

    existing_nodes = set(map(str, graph.nodes))
    dummy_nodes = [str(node) for node in dummy_nodes]

    nodes_to_add = [
        node for node in dummy_nodes
        if node not in existing_nodes
    ]

    graph.add_nodes_from(nodes_to_add)

    for node in nodes_to_add:
        color_info[node] = gray_color.copy()

    return graph, nodes_to_add


def assign_node_modalities(
    graph,
    tarsal_nodes,
    labial_nodes,
    dummy_nodes=None,
):
    """
    graph node에 tarsal, labial, geosmin, dummy modality를 지정한다.
    """
    tarsal_nodes = set(map(str, tarsal_nodes))
    labial_nodes = set(map(str, labial_nodes))
    dummy_nodes = set(map(str, dummy_nodes or []))

    for node in graph.nodes:
        node_str = str(node)

        if node_str in dummy_nodes:
            modality = "dummy"
        elif node_str in tarsal_nodes:
            modality = "tarsal"
        elif node_str in labial_nodes:
            modality = "labial"
        else:
            modality = "geosmin"

        graph.nodes[node]["modality"] = modality

    return graph


def load_node_locations(csv_path):
    """
    Gephi에서 export한 CSV의 Id, X, Y column을 node 위치 dictionary로 변환한다.
    """
    info = pd.read_csv(csv_path)

    return {
        str(row.Id): {
            "x": float(row.X),
            "y": float(row.Y),
            "z": 0.0,
        }
        for row in info.itertuples(index=False)
    }


def add_missing_node_locations(
    graph,
    node_locations,
    reference_node="horntail",
    x_spacing=100,
):
    """
    기존 위치 정보가 없는 node를 reference_node 오른쪽에 순차 배치한다.
    """
    reference_node = str(reference_node)

    if reference_node not in node_locations:
        raise KeyError(
            f"Reference node {reference_node!r} is not present "
            "in node_locations."
        )

    missing_nodes = sorted(
        set(map(str, graph.nodes)) - set(node_locations)
    )

    for index, node in enumerate(missing_nodes, start=1):
        location = node_locations[reference_node].copy()
        location["x"] += x_spacing * index
        node_locations[node] = location

    return node_locations


def threshold_and_transform_edges(
    graph,
    min_weight=7,
    weight_transform=np.log2,
):
    """
    weight가 min_weight보다 작은 edge를 제거하고,
    남은 edge의 weight를 변환한다.
    """
    graph = graph.copy()

    edges_to_remove = []

    for source, target, edge_data in graph.edges(data=True):
        weight = float(edge_data["weight"])

        if weight < min_weight:
            edges_to_remove.append((source, target))
        else:
            edge_data["weight"] = float(weight_transform(weight))

    graph.remove_edges_from(edges_to_remove)

    return graph


def remove_cross_modality_edges(
    graph,
    source_modality,
    target_modality,
):
    """
    특정 modality 조합의 directed edge를 제거한다.
    """
    graph = graph.copy()

    edges_to_remove = [
        (source, target)
        for source, target in graph.edges
        if (
            graph.nodes[source].get("modality"),
            graph.nodes[target].get("modality"),
        ) == (source_modality, target_modality)
    ]

    graph.remove_edges_from(edges_to_remove)

    return graph