def solve(g):
 h,w=len(g),len(g[0]);o=[r[:]for r in g];C={}
 for r,R in enumerate(g):
  for c,v in enumerate(R):
   if v:C[v]=C.get(v,[])+[(r,c)]
 M=C[m:=max(C,key=lambda x:len(C[x]))];R,K=zip(*M);a,b,e,f=min(R),max(R),min(K),max(K)
 cr,ce,ph,pw=(a+b)/2,(e+f)/2,b-a+1,f-e+1;P={(i-a,j-e)for i,j in M}
 for v,L in C.items():
  if v-m:
   for u,k in L:
    dr=(u-cr>1.4)-(cr-u>1.4);dc=(k-ce>1.4)-(ce-k>1.4)
    if dr|dc:
     for s in range(1,50):
      nr,nc=a+s*-~ph*dr,e+s*-~pw*dc
      if-ph<nr<h>-pw<nc<w:
       for p,q in P:r,c=nr+p,nc+q;h>r>=0<=c<w and(o[r].__setitem__(c,v))
 return o
