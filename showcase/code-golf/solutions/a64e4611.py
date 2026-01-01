def solve(G):
 I=range;R,C=len(G),len(G[0]);O=[*map(list,G)]
 if(b:=max([-~(c-a)*-~(k-d),a,d,c,k]for a in I(R)for d in I(C)for c in I(a,R)for k in I(d,C)if all(O[r][j]<1for r in I(a,c+1)for j in I(d,k+1)))or[0,])[0]<1:return G
 _,e,f,g,j=b;H=j-f>g-e;e+=e>0;g-=H*(g<R-1);f+=1-H;j-=1-H
 for r in I(e,g+1):G[r][f:j+1]=[3]*(j-f+1)
 E=lambda i,L,z,d=0:d or all(O[(v,i)[z]][(i,v)[z]]<1for v in L)
 for i,A,B,P,z in[(r,e,g,[I(f),I(j+1,C)],1)for r in I(e,g+1)]+[(c,f,j,[I(e),I(g+1,R)],0)for c in I(f,j+1)]:
  for L in P:
   if L and E(i,L,z)*E(i-1,L,z,i==A)*E(i+1,L,z,i==B):
    for v in L:G[(v,i)[z]][(i,v)[z]]=3
 return G
