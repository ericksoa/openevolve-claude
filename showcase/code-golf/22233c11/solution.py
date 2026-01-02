def solve(g):
 R,C=len(g),len(g[0]);o=eval(str(g));S={(r,c)for r in range(R)for c in range(C)if g[r][c]==3};V=set()
 for s in S-V:
  G={s};V|=G;Q=[s]
  while Q:r,c=Q.pop();N={(r+i,c+j)for i in(-1,0,1)for j in(-1,0,1)}&S-V;V|=N;Q+=[*N];G|=N
  rs,cs=zip(*G);a,b,e,f=min(rs),max(rs),min(cs),max(cs);h,w=b-a+1>>1,f-e+1>>1;u,v=(f+1,e-w)if g[a][e]else(e-w,f+1)
  [o[t].__setitem__(k,8)for i in range(h)for j in range(w)for t,k in((a-h+i,u+j),(b+1+i,v+j))if R>t>=0<=k<C]
 return o
