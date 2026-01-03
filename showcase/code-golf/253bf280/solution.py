def solve(g):
 E=enumerate;o=eval(str(g));p=[(r,c)for r,R in E(g)for c,v in E(R)if v>7]
 for a,b in p:
  for c,d in p:
   if a==c:o[a][b+1:d]=[3]*(d+~b)
   for j in[*range(a+1,c)]*(b==d):o[j][b]=3
 return o
