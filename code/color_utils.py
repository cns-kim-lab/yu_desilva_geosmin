


def hex_to_rgb(hexcode):
    value = hexcode.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    rgb = tuple(int(x * 255) for x in rgb)
    return '#%02x%02x%02x' % rgb
