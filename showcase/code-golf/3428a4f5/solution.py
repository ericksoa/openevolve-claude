def solve(g):s=len(g)//2;return[[(a!=b)*3for a,b in zip(*r)]for r in zip(g[:s],g[-s:])]
