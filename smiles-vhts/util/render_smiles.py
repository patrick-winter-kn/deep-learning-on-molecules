def render(smiles, path, size_factor=1, heatmap=None):
    # We use a size of 6 per character and 2 for padding before and after the text
    width = (len(smiles) * 6 + 2) * size_factor
    # Text hight of 10 plus 2 padding
    height = 12 * size_factor
    x = 1 * size_factor
    y = 10 * size_factor
    font_size = 10 * size_factor
    if heatmap is not None:
        text = ''
        # heatmap contains RGB values for every letter
        for i in range(len(smiles)):
            text += '<tspan fill="rgb(' + str(heatmap[i][0]) + ',' + str(heatmap[i][1]) + ',' + str(heatmap[i][2])\
                    + ')">' + smiles[i] + '</tspan>\n'
    else:
        text = smiles
    with open(path, 'w') as file:
        file.write('<svg viewBox="0 0 ' + str(width) + ' ' + str(height) + '">\n<text x="' + str(x) + '" y="' + str(y)
                   + '" fill="black" font-family="monospace" font-size="' + str(font_size) + '">\n' + text
                   + '</text>\n</svg>')
