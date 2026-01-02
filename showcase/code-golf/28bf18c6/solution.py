solve=lambda g:[[*r]*2for r in zip(*filter(any,zip(*g)))if any(r)]
