def solve(g):
 q=lambda l:l[0]*l[-1]and 1>max(l[1:-1])
 return[[x or 3*(q(r)or q([*zip(*g)][c]))for c,x in enumerate(r)]for r in g]
