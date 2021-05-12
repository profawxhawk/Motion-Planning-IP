import networkx as nx
from scipy.spatial import distance
import random
from copy import deepcopy
import numpy as np
from time import sleep
from numpy.linalg import lstsq
from numpy import ones,vstack
import math
import time
import sys
import warnings
warnings.filterwarnings("ignore")
class Selector:
    def weightsamp(self, G, p_candidate):
        weight_sum = 0
        for i in range(len(p_candidate) - 1):
            node = G.edges[p_candidate[i], p_candidate[i + 1]]
            if node['eval'] is False:
                weight_sum += node['est']
        x = np.random.uniform(low=0, high=weight_sum)
        weight_sum = 0
        for i in range(len(p_candidate) - 1):
            node = G.edges[p_candidate[i], p_candidate[i + 1]]
            if x >= weight_sum and x < weight_sum + node['est'] and node['eval'] is False:
                return [(p_candidate[i], p_candidate[i + 1])]
            if node['eval'] is False:
                weight_sum += node['est']
    def expand(self, G, p_candidate):
        E_selected = []
        for i in range(len(p_candidate) - 1):
            if G.edges[p_candidate[i], p_candidate[i + 1]]['eval'] is False:
                v_frontier = p_candidate[i]
                for e in G.edges(v_frontier):
                    if G.edges[e]['eval'] is False:
                        E_selected.append(e)
                break
        return E_selected
    def forward(self, G, p_candidate):
        for i in range(len(p_candidate) - 1):
            if G.edges[p_candidate[i], p_candidate[i + 1]]['eval'] is False:
                return [(p_candidate[i], p_candidate[i + 1])]
    
    def reverse(self, G, p_candidate):
        for i in range(len(p_candidate) - 2, -1, -1):
            if G.edges[p_candidate[i], p_candidate[i + 1]]['eval'] is False:
                return [(p_candidate[i], p_candidate[i + 1])]

    def alternate(self, G, p_candidate, alternate_n):
        if alternate_n:
            return self.forward(G, p_candidate), 1 - alternate_n
        else:
            return self.reverse(G, p_candidate), 1 - alternate_n
    
    def bisection(self, G, p_candidate):
        dists = [len(p_candidate)] * (len(p_candidate) - 1)
        for i in range(len(p_candidate) - 1): # forward and backward, check eval
            if G.edges[p_candidate[i], p_candidate[i + 1]]['eval'] is True:
                dist = 0
            elif i == 0:
                dist = 1
            else:
                dist = dists[i - 1] + 1
            if dists[i] > dist:
                dists[i] = dist

            j = len(p_candidate) - 1 - i
            if G.edges[p_candidate[j - 1], p_candidate[j]]['eval'] is True:
                dist = 0
            elif i == 0:
                dist = 1
            else:
                dist = dists[j] + 1
            if dists[j - 1] > dist:
                dists[j - 1] = dist
        
        i_max = dists.index(max(dists))
        return [(p_candidate[i_max], p_candidate[i_max + 1])]
def checkCollision(a, b, c, x, y, radius):
    dist = ((abs(a * x + b * y + c)) /
            math.sqrt(a * a + b * b))
    if (radius == dist):
        return -1
    elif (radius > dist):
        return -1
    else:
        return 1

class LazySP:
    def __init__(self, G, u, v):
        self.G = G
        self.u = u
        self.v = v
    
    def estimate_distance(self, u, v):
        if 'pos' in self.G.nodes[u].keys():
            return distance.euclidean(self.G.nodes[u]['pos'], self.G.nodes[v]['pos'])
        return 0 # only for randomly connected graphs, with no actual positions
    
    def get_weight(self,e):
        start=e[0]
        end=e[1]
        points = [start,end]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        for i in obstacles:
            if (checkCollision(m,-1,c,i[0],i[1],i[2]))==-1:
                return 1000000
        return self.G.edges[e]['actual']

    def solve(self, sp_solver, selector, beta):
        eval_n = 0
        E_eval = set([])
        alternate_n = 0
        while True:
            try:
                p_candidate = nx.shortest_path(self.G, source=self.u, target=self.v, weight='est', method=sp_solver) # use 'est' as current knowledge of map edge weights
            except nx.exception.NetworkXNoPath:
                print('no path')
                return [], 0
            p_evaled = True
            for i in range(len(p_candidate) - 1):
                if tuple(p_candidate[i:i+2]) not in E_eval:
                    p_evaled = False
                    break
            if p_evaled:
                print("path found successfully")
                return p_candidate, eval_n
            if selector.__name__ == 'alternate':
                E_selected, alternate_n = selector(G=self.G, p_candidate=p_candidate, alternate_n=alternate_n)
            else:
                E_selected = selector(G=self.G, p_candidate=p_candidate)
            for e in set(E_selected).difference(E_eval):
                self.G.edges[e]['est'] = self.get_weight(e) 
                self.G.edges[e]['eval'] = True
                eval_n += 1
                E_eval.add(e)
                u, v = e
                E_eval.add((v, u))

grid_size=10                
G1 = nx.grid_2d_graph(grid_size,grid_size,0.2)
S = Selector()
for (u, v) in G1.edges():
    G1.edges[u, v]['est'] = 1.0
    G1.edges[u, v]['actual'] = 1.0
    G1.edges[u, v]['eval'] = False
obstacles=[]
for i in range(6):
    obstacles.append((random.randint(0,10),random.randint(0,10),random.randint(1,2)))

selector='weightsamp'
st = random.sample(range(grid_size),1)[0],random.sample(range(grid_size),1)[0]
goal = random.sample(range(grid_size),1)[0],random.sample(range(grid_size),1)[0]

print("Start "+str(st),"Goal "+str(goal))

sp = LazySP(deepcopy(G1), st, goal)
start = time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (weightsamp) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

selector='forward'
sp = LazySP(deepcopy(G1), st, goal)
start= time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (forward) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

selector='bisection'
sp = LazySP(deepcopy(G1), st, goal)
start = time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (bisection) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

selector='alternate'
sp = LazySP(deepcopy(G1), st, goal)
start = time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (alternate) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

selector='expand'
sp = LazySP(deepcopy(G1), st, goal)
start = time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (expand) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

selector='reverse'
sp = LazySP(deepcopy(G1), st, goal)
start = time.time()
path, eval_n = sp.solve('dijkstra', eval('S.' + selector),2)
end = time.time()
print("Runtime of the program (reverse) is "+str(end - start)+" seconds")
print(path,eval_n,len(path))

Counter = 0
lim = 100000
G=[]
G.append(st)
par={st:None}
start = time.time()
def Nearest(point):
    mindis=sys.maxsize
    minpoint=point
    count=0
    for i in G:
        start=point
        end=i
        points = [start,end]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        fl=1
        for j in obstacles:
            count+=1
            if (checkCollision(m,-1,c,j[0],j[1],j[2]))==-1:
                fl=-1
                break
        if fl==-1:
            continue
        if math.sqrt((point[0]-i[0])**2 + (point[1]-i[1])**2)<mindis:
            mindis=math.sqrt((point[0]-i[0])**2 + (point[1]-i[1])**2)
            minpoint=i
    return minpoint,mindis,count

def getPath(point):
    ans=[point]
    dis=0
    while(par[point]!=None):
        ans.insert(0,par[point][0])
        point=par[point][0]
        if par[point]==None:
            break
        dis+=par[point][1]
    return ans,dis
comp=0
allowed=list(G1.nodes())
allowed.remove(st)
while Counter < lim:
    Counter+=1
    Xnew  = random.choice(allowed)
    Xnearest,dis,count = Nearest(Xnew)
    comp+=count
    if Xnearest==None or Xnearest==Xnew:
        continue
    G.append(Xnew)
    allowed.remove(Xnew)
    par[Xnew]=[Xnearest,dis]
    if Xnew == goal:
        print("success")
        print(getPath(Xnew),comp)
        break
    Counter+=1
end = time.time()
print("Runtime of the program (rrt) is "+str(end - start)+" seconds")