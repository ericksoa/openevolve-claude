def solve(g):
 R=[[0]*len(r)for r in g]
 for k in{c for r in g for c in r if c}:
  M=max(j for r in g for j,c in enumerate(r)if c==k)
  for i,r in enumerate(g):
   b=[]
   for j in[j for j,c in enumerate(r)if c==k][::-1]:R[i][n:=j+(j<M)-(j+1in b)]=k;b+=n,
 return R
