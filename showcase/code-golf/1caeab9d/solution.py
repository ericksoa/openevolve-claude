def solve(g):
 S={};o=[[0]*len(r)for r in g];[S.setdefault(v,[]).append((i,j))for i,r in enumerate(g)for j,v in enumerate(r)if v]
 for v,P in S.items():
  for i,j in P:o[i-P[0][0]+S[1][0][0]][j]=v
 return o
