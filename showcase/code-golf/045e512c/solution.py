def solve(g):
 h,w=len(g),len(g[0]);o=eval(str(g));C={}
 for i in range(h*w):
  if v:=g[i//w][i%w]:C[v]=C.get(v,[])+[(i//w,i%w)]
 M=C[m:=max(C,key=lambda x:len(C[x]))];R,K=zip(*M);a,b,e,f=min(R),max(R),min(K),max(K)
 P={(i-a,j-e)for i,j in M}
 for v,L in C.items():
  if v-m:
   for u,k in L:
    dr=(2*u-a-b>2)-(a+b-2*u>2);dc=(2*k-e-f>2)-(e+f-2*k>2)
    if dr|dc:[h>(r:=a+s*(b-a+2)*dr+p)>=0<=(c:=e+s*(f-e+2)*dc+q)<w and(o[r].__setitem__(c,v))for s in range(1,h)for p,q in P]
 return o
