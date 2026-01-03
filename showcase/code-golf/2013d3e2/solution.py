def solve(g):e=enumerate;R,C=zip(*[(i,j)for i,r in e(g)for j,x in e(r)if x]);r=min(R);c=min(C);h=-~max(R)-r>>1;return[g[i][c:c+h]for i in range(r,r+h)]
