def solve(g):
 C={}
 for r,R in enumerate(g):
  for c,v in enumerate(R):
   if v:C[v]=C.get(v,[])+[(r,c)]
 a,b=zip(*C[m:=min(C,key=lambda _:len(C[_]))])
 return[[m]*-~(max(b)-min(b))]*-~(max(a)-min(a))