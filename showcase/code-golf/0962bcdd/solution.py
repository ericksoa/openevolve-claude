def solve(g):
 o=[*map(list,g)]
 for k in range(64):
  i,j=k//8+2,k%8+2;c=g[i][j];u=g[i-1][j]
  if c*u and u==g[i+1][j]==g[i][j-1]==g[i][j+1]!=c:
   for d in-2,2:o[i+d][j]=o[i][j+d]=u
   for x in-2,-1,1,2:o[i+x][j+x]=o[i+x][j-x]=c
 return o
