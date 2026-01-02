def solve(g):
 S=sum;n=len(g);m=len(g[0])
 def T(q):
  while S(q[0])<1:q=q[1:]
  return q if S(q[-1])else T(q[:-1])
 r=[i for i in range(n)if S(g[i])<1][0]
 c=[j for j in range(m)if S(g[i][j]for i in range(n))<1][0]
 Q=[[x[:c]for x in g[:r]],[x[c+1:]for x in g[:r]],[x[:c]for x in g[r+1:]],[x[c+1:]for x in g[r+1:]]]
 for q in Q:
  if{x for y in q for x in y}-{0,*[x for p in Q if p!=q for y in p for x in y]}:return[[*x]for x in zip(*T([*zip(*T(q))]))]
