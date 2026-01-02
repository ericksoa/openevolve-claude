def solve(g):
 o=eval(str(g));i=0
 for r in g:
  for j,v in enumerate(r):
   if v==2:
    for k in range(len(g)):o[k][j]=o[k][j]or 2
   if v&1:
    for k in range(len(r)):
     if o[i][k]<3:o[i][k]=v
  i+=1
 return o
