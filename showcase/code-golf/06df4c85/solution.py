def solve(g):
 s=g[2][0];R=[*map(list,g)];H=-(-len(g)//3);W=-(-len(g[0])//3)
 for k in 0,1:
  for a in range([W,H][k]):
   d={}
   for b in range([H,W][k]):v=g[(a,b)[k]*3][(b,a)[k]*3];d[v]=d.get(v,[])+[b]*(v!=s)
   for c,p in d.items():
    for b in p[1:]and range(p[0],p[-1]+1):
     for i in 0,1:
      for j in 0,1:x,y=(a,b)[k]*3+i,(b,a)[k]*3+j;R[x][y]=R[x][y]or c
 return R
