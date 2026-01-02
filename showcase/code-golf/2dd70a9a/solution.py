def solve(g):
 R,C=len(g),len(g[0]);o=[r[:]for r in g];F=lambda v:[(i,j)for i in range(R)for j in range(C)if g[i][j]==v]
 if[]in[p:=F(2),q:=F(3)]:return o
 a,b,c,d=[sorted({z[k]for z in t})for t in[p,q]for k in(0,1)];G=b[0]-d[-1];D=lambda i,j:o[i][j]or o[i].__setitem__(j,3);I=range;m=min;x=max
 if G<3:V=m(x(*b,*d)+(G*(G>0)or-~abs(a[0]-c[0])),~-C);[D(a[0],j)for j in I(-~b[-1],-~V)];[D(c[0],j)for j in I(-~d[-1],-~V)];[D(i,V)for i in I(-~m(a[0],c[0]),x(a[-1],c[-1]))]
 else:u=c[0]>a[-1];X=(c[-1],a[-1])[u];Z=(a[0],c[0])[u];K=X+(Z-X)//3;e,f=(d[0],b[-1])[u],(b[-1],d[0])[u];[D(i,e)for i in I(-~X,-~K)];[D(K,j)for j in I(m(e,f),-~x(e,f))];[D(i,f)for i in I(-~K,Z)]
 return o
