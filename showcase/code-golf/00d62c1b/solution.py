def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o];s=[(0,0)]
 while s:
  a,b=s.pop()
  if len(g)>a>=0<=b<w and g[a][b]<1:g[a][b]=1;s+=(a+1,b),(a-1,b),(a,b+1),(a,b-1),
 return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
