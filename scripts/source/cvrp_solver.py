import cvrplib
import numpy as np
from functools import reduce

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
font = {'family' : 'arial', 'weight' : 'bold', 'size'   : 22}
plt.rc('font', **font)
plt.rcParams["figure.figsize"] = (12,8)


class AntColonySolver:
    
    def __init__(self,    
                 iterations=100, 
                 ants = 5, 
                 alpha=1, 
                 beta=1, 
                 sigma=1, 
                 q=0.8, 
                 theta=80):
        
        self.iterations = iterations
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.q = q
        self.theta = theta
        
    def fit(self, folder, dataset, verbose=False, early_stop=None):
        
        instance, solution = cvrplib.read(instance_path=f"{folder}{dataset}.vrp",
                                          solution_path=f"{folder}{dataset}.sol")
        
        self.bench_routes = solution.routes
        self.bench_cost = solution.cost
        
        self.graph, self.demand, self.edges, self.feromones = dict(), dict(), dict(), dict()

        self.capacity = instance.capacity

        for i in range(instance.n_customers+1):
            self.graph[i+1] = instance.coordinates[i]
            self.demand[i+1] = instance.demands[i]

        distances = instance.distances
        customers = list(self.graph.keys())
        self.vertices = customers[1:]

        for a in customers:
            for b in customers:
                self.edges[(min(a,b),max(a,b))] = distances[a-1][b-1]
                if a!=b:
                    self.feromones[(min(a,b),max(a,b))] = 1 
                    
        self.solve(verbose, early_stop)
    
    def one_ant_solution(self) -> list:
        
        solution = list()
        vertices = self.vertices.copy()
        capacity_limit = self.capacity
        
        while(len(vertices)!=0):
            path = list()
            city = np.random.choice(vertices)
            capacity = capacity_limit - self.demand[city]
            path.append(city)
            vertices.remove(city)
            while(len(vertices)!=0):
                probabilities = list(map(lambda x: ((self.feromones[(min(x,city), max(x,city))])**self.alpha)
                                         *((1/self.get_edges(x, city))**self.beta), vertices))
                probabilities /= np.sum(probabilities)

                city = np.random.choice(vertices, p=probabilities)
                capacity -= self.demand[city]

                if(capacity>0):
                    path.append(city)
                    vertices.remove(city)
                else:
                    break
            solution.append(path)
        return solution
    
    def get_edges(self, a , b):
        dist = self.edges[(min(a,b), max(a,b))]
        if dist!=0:
            return dist
        return 1
    
    def solution_cost(self, solution) -> int:
        s = 0
        for i in solution:
            a = 1
            for j in i:
                b = j
                s += self.get_edges(a , b)
                a = b
            b = 1
            s += self.get_edges(a , b)
        return s
    
    def get_feromones(self, path, i) -> tuple:
        return (min(path[i],path[i+1]), max(path[i],path[i+1]))
        
    def update_feromone(self, solutions, best_solution) -> int:
        
        L_mean = reduce(lambda x,y: x+y, (i[1] for i in solutions))/len(solutions)
        self.feromones = {k : (self.q + self.theta/L_mean)*v for (k,v) in self.feromones.items()}
        solutions.sort(key = lambda x: x[1])
        
        if(best_solution!=None):
            if(solutions[0][1] < best_solution[1]):
                best_solution = solutions[0]
            for path in best_solution[0]:
                for i in range(len(path)-1):
                    self.feromones[self.get_feromones(path, i)]\
                    = self.sigma/best_solution[1] + self.feromones[self.get_feromones(path, i)]
        else:
            best_solution = solutions[0]
        for l in range(self.sigma):
            paths = solutions[l][0]
            L = solutions[l][1]
            for path in paths:
                for i in range(len(path)-1):
                    self.feromones[self.get_feromones(path, i)]\
                    = (self.sigma-(l+1)/L**(l+1)) + self.feromones[self.get_feromones(path, i)]
        return best_solution
    
    def solve(self, verbose, early_stop):
        
        self.best_solution = None
        prev_solution = None
        stop = 0
        
        for i in range(self.iterations):
            solutions = list()
            for _ in range(self.ants):
                solution = self.one_ant_solution()
                solutions.append((solution, self.solution_cost(solution)))
            self.best_solution = self.update_feromone(solutions, self.best_solution)
            if verbose and i%verbose==0:
                print(str(i)+":\t"+str(int(self.best_solution[1])))
            if prev_solution == self.best_solution:
                stop+=1
                if stop == early_stop:
                    break
            else:
                stop = 0
                prev_solution = self.best_solution
    
    def get_best_solution(self) -> tuple:
        solution = list(self.best_solution)
        solution[0] = [list((map(lambda x: x-1, e))) for e in solution[0]]
        return tuple(solution)
    
    def evaluate_solution(self) -> float:
        return round((self.best_solution[1]-self.bench_cost)/self.bench_cost, 3)
    
    def plot_solution_routes(self):
        
        self.plot_routes([list((map(lambda x: x+1, e))) for e in self.bench_routes], title="Bench plot")
        self.plot_routes(self.best_solution[0], title="Solution plot")
    
    def plot_routes(self, routes, title="Routes"):
        
        for i, sol in enumerate(routes):

            x, y = [], []

            x.append(self.graph[1][0])
            y.append(self.graph[1][1])

            for n in sol:
                x.append(self.graph[n][0])
                y.append(self.graph[n][1])

            x.append(self.graph[1][0])
            y.append(self.graph[1][1])

            plt.plot(x, y, label=f"Route {i+1}")
        
        plt.title(title)
        plt.legend()
        plt.show()