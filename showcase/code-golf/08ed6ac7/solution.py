def solve(g):
 c=[*map(sum,zip(*g))];s=sorted(range(len(c)),key=lambda i:-c[i])
 return[[v and-~s.index(x)for x,v in enumerate(r)]for r in g]
