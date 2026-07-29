


def hex_to_rgb(hexcode):
    value = hexcode.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    rgb = tuple(int(x * 255) for x in rgb)
    return '#%02x%02x%02x' % rgb

    
def make_color_map(cluster_df):
    """
    Create a mapping from neuron type to display color.

    Parameters
    ----------
    cluster_df : pandas.DataFrame
        DataFrame containing 'type' and 'color' columns.

    Returns
    -------
    dict
        Dictionary mapping neuron type -> color.
    """
    return (
        cluster_df[["type", "color"]]
        .drop_duplicates(subset="type")
        .set_index("type")["color"]
        .to_dict()
    )