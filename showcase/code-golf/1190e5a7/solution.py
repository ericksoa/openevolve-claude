solve=lambda g:[[sum({*g[0]})-min(map(max,g))]*-~sum(len({*c})<2for c in zip(*g))for r in g[:-~sum(len({*r})<2for r in g)]]
