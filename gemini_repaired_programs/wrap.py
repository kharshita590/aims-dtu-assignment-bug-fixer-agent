def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(' ', 0, cols)
        if end == -1:
            end = cols
        line, text = text[:end], text[end:].lstrip()
        lines.append(line)

    lines.append(text)
    return lines