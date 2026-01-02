def solve(g):
 h,w=len(g),len(g[0]);p={}
 for r in range(h):
  for c in range(w):
   v=g[r][c]
   if v:p[v]=p.get(v,[])+[(r,c)]
 f=max(p,key=lambda k:len(p[k]));V=set();C=[]
 def B(s):
  q=[s];c=set()
  while q:
   r,z=q.pop(0)
   if(r,z)in V:continue
   V.add((r,z))
   if g[r][z]:
    c.add((r,z))
    for d in(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1):
     n=r+d[0],z+d[1]
     if h>n[0]>=0<=n[1]<w and n not in V and g[n[0]][n[1]]:q+=n,
  return c
 for x in p[f]:
  if x not in V:c=B(x);C+=[c]*bool(c)
 A=set().union(*C);I={k:[x for x in p[k]if x not in A]for k in p if k!=f}
 o=[[0]*w for _ in range(h)];u=set()
 for c in C:
  e=[(r,z,g[r][z])for r,z in c];K={j:(r,z)for r,z,j in e if j!=f}
  if K:
   R,*_=K;Rr,Rc=K[R]
   L=[(r-Rr,z-Rc,j)for r,z,j in e];X={j:(r-Rr,z-Rc)for j,(r,z)in K.items()}
   for a,b,d,e in(1,0,0,1),(-1,0,0,1),(1,0,0,-1),(-1,0,0,-1),(0,1,1,0),(0,-1,1,0),(0,1,-1,0),(0,-1,-1,0):
    Y=[(r*a+z*b,r*d+z*e,j)for r,z,j in L];Z={j:(r*a+z*b,r*d+z*e)for j,(r,z)in X.items()}
    if R in I:
     for ir,ic in I[R]:
      if(ir,ic)not in u:
       m=1;M=[(ir,ic)]
       for cv,(dr,dc)in Z.items():
        if cv!=R:
         er,ec=ir+dr,ic+dc
         if cv not in I or(er,ec)not in I[cv]:m=0;break
         M+=(er,ec),
       if m:
        for dr,dc,j in Y:
         nr,nc=ir+dr,ic+dc
         if h>nr>=0<=nc<w:o[nr][nc]=j
        u|=set(M)
 return o
