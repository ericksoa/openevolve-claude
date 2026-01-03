def solve(g):E=enumerate;c=min(f:=[*filter(bool,sum(g,[]))],key=f.count);R,C=zip(*[(i,j)for i,r in E(g)for j,v in E(r)if v==c]);return[r[min(C):max(C)+1]for r in g[min(R):max(R)+1]]
