def solve(g):
 I=range;R,C=len(g),len(g[0]);o=[[0]*C for _ in I(R)];V={g[0][c]:c for c in I(C)if g[0][c]and len({g[r][c]for r in I(R)})<2};H={g[r][0]:r for r in I(R)if g[r][0]and len({g[r][c]for c in I(C)})<2}
 for i in I(R):
  for j in I(C):
   v=g[i][j]
   if v in V:c=V[v];o[i][c]=v;j!=c and-1<(n:=c+(j>c or-1))<C and exec('o[i][n]=v')
   elif v in H:r=H[v];o[r][j]=v;i!=r and-1<(n:=r+(i>r or-1))<R and exec('o[n][j]=v')
 return o
