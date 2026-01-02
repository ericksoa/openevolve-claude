def solve(G):
 w=len(G[0])+2;g=[o:=[0]*w,*[[0,*r,0]for r in G],o]
 def f(a,b):
  if len(g)>a>~0<b<w>1>g[a][b]:g[a][b]=1;f(a+1,b);f(a-1,b);f(a,b+1);f(a,b-1)
 f(0,0);return[[[4,0,0,3][c]for c in r[1:-1]]for r in g[1:-1]]
