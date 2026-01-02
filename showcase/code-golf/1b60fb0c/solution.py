def solve(g):
 R,C=len(g),len(g[0]);o=eval(str(g));O=lambda r:[c for c in range(C)if g[r][c]];S=lambda r:set(O(r))
 rm=max(c for r in range(R)for c in range(C)if g[r][c]);Q=lambda X:len(X)==X[-1]-X[0]+1
 cs=[];wr=-1;wo=[]
 for r in range(R):
  if(X:=O(r))and Q(X):
   if 2<len(X)and X[-1]<rm:cs+=[r]
   if X[-1]==rm>len(wo)-len(X):wo=X;wr=r
 if wr<1or len(cs)<2:return o
 tc,bc=cs[0],cs[-1];wl=wo[0];tp=S(tc);ci=tp&S(bc);sy=tp==S(bc);ax=wl-1+sy;th=len(tp)-1
 G=lambda X:(len(X)>1)*(X[-1]-X[0]+1-len(X));H=lambda X:any(X[i+1]==X[i]+1for i in range(len(X)-1))
 bd={wr}
 for d in-1,1:
  r=wr+d
  while tc<r<bc and S(r)&S(r-d)and not(sy*(G(O(r))>th or S(r)<=ci)or(1-sy)*(H(O(r))or G(O(r))>th if d<0else S(r)==ci)):bd.add(r);r+=d
 A=2*ax;W=2*wl+1
 for r in bd:
  X=O(r);rg=G(X)
  M=[A-c for c in X if c>ax and 0<A-c<ax]if r==wr or sy*rg>sy else[A-max(c for c in X if c>ax)]if Q(X)else[*range(max(0,ax-th),ax+1)]if rg<2else[W-c for c in X if 0<W-c<wl]
  for m in M:
   if(0<m<ax or rg<2and m<=ax)and not g[r][m]:o[r][m]=2
 return o
