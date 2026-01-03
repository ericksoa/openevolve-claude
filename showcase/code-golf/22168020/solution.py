def solve(g):
 for r in g:
  for c in{*r}-{0}:a=r.index(c);b=len(r)-r[::-1].index(c);r[a:b]=[c]*(b-a)
 return g
