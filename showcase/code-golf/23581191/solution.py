def solve(g):
 I=enumerate;o=eval(str(g))
 for i,r in I(g):
  for j,v in I(r):
   if v:
    for k in range(len(g)):o[i][k]=o[k][j]=v
    if v>7:a,b=i,j
    else:c,d=i,j
 o[a][d]=o[c][b]=2
 return o
