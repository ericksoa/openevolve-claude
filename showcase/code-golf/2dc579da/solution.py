def solve(g):
 v,u=len(g)//2,len(g[0])//2;S={g[v][u],g[0][0]}
 return next(q for R in(g[:v],g[v+1:])for q in[[r[:u]for r in R],[r[u+1:]for r in R]]if any(c not in S for r in q for c in r))
