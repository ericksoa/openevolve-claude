def solve(g):o=eval(str(g));[exec("for k in 0,1,2:o[i-i%3+k][j-1:j+2]=g[k][:3]")for i,r in enumerate(g)for j,v in enumerate(r)if v==1];return o
