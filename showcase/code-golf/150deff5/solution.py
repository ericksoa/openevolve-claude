def solve(g):
 R,C=len(g),len(g[0]);V=[(i,j)for i in range(R-1)for j in range(C-1)if 5==g[i][j]==g[i+1][j]==g[i][j+1]==g[i+1][j+1]];F={(i,j)for i in range(R)for j in range(C)if g[i][j]>4}
 from itertools import combinations as X
 B=[];Z=999
 for s in range(len(V),0,-1):
  for c in X(V,s):
   if all(abs(a[0]-b[0])>1or abs(a[1]-b[1])>1for a,b in X(c,2)):
    T=F-{p for i,j in c for p in[(i,j),(i+1,j),(i,j+1),(i+1,j+1)]};iso=sum(not T&{(i-1,j),(i+1,j),(i,j-1),(i,j+1)}for i,j in T)
    if iso<Z or iso==Z and len(c)>len(B):Z=iso;B=c
  if B and Z<1:break
 o=[[0]*C for _ in range(R)]
 for i,j in B:o[i][j]=o[i+1][j]=o[i][j+1]=o[i+1][j+1]=8
 for i,j in F:o[i][j]=o[i][j]or 2
 return o
