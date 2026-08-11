import matplotlib.colors as mcolors
import numpy as np 

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

def get_graded_color(base_hex, count):
    base_rgb = mcolors.to_rgb(base_hex)
    factor = 1 - ((count-1) / 4)

    new_rgb = [((val + (1.0 - val) * factor)) for val in base_rgb]
    
    return rgb_to_hex(new_rgb) #{'a': 1, 'r': new_rgb[0], 'g': new_rgb[1], 'b': new_rgb[2]}

def lighten_color(rgb, factor=0.5):
    white = np.array([255, 255, 255])
    return (rgb[:3] + (white - rgb[:3]) * factor).astype(int)


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([r, g, b, int(alpha)])