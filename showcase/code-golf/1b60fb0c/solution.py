def solve(g):
 R,C=len(g),len(g[0]);o=eval(str(g));O=lambda r:sorted(c for c in range(C)if g[r][c])
 rm=max(c for r in range(R)for c in range(C)if g[r][c])
 cp=lambda r:(X:=O(r))and len(X)==X[-1]-X[0]+1and len(X)>2and X[-1]<rm
 cs=[r for r in range(R)if cp(r)]
 if cs<[1]:return o
 tc,bc=min(cs),max(cs);cw=len(O(tc));wr=-1;wo=[]
 for r in range(R):
  X=O(r)
  if X and X[-1]==rm and len(X)==X[-1]-X[0]+1>len(wo):wo=X;wr=r
 if wr<0:return o
 wl=wo[0];tp=set(O(tc));sy=tp==set(O(bc));ci=tp&set(O(bc));th=len(tp)-1;ax=wl-1+sy
 gp=lambda r:(X:=O(r))and len(X)>1and X[-1]-X[0]+1-len(X)or 0
 hc=lambda r:any(O(r)[i+1]==O(r)[i]+1for i in range(len(O(r))-1))
 bd={wr}
 for r in range(wr-1,tc,-1):
  if not set(O(r))&set(O(r+1))or sy*(gp(r)>th or set(O(r))<=ci)or(1-sy)*(hc(r)or gp(r)>th):break
  bd.add(r)
 for r in range(wr+1,bc):
  if not set(O(r))&set(O(r-1))or sy*(set(O(r))<=ci)or(1-sy)*(set(O(r))==ci):break
  bd.add(r)
 for r in bd:
  X=O(r);ic=len(X)==X[-1]-X[0]+1;rg=gp(r);sg=[];s=[]
  for c in X:
   if s and c-s[-1]>1:sg+=[s];s=[]
   s+=[c]
  if s:sg+=[s]
  lc={c for s in sg if s[0]<=ax for c in s}
  M=[2*ax-c for c in X if c>ax and 0<2*ax-c<ax]if r==wr else[2*ax-max([c for c in X if c>ax]+[ax+1])]if ic else list(range(max(0,ax-cw+1),ax+1))if rg<2else[2*ax-c for c in X if c>ax and c not in lc and 0<2*ax-c<ax]if sy else[2*wl+1-c for c in X if 0<2*wl+1-c<wl]
  M=[m for m in M if 0<m<ax or rg<2and m<=ax]
  for m in M:o[r][m]=o[r][m]or 2
 return o
