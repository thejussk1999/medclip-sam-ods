class OptimizedUnionFind:
    def __init__(self, num_node):
        self.parent = [i for i in range(num_node)] # Assigns the parent number for each node
        self.rank = [0 for i in range(num_node)] # Assigns rank 0 for all the nodes
        self.size = [1 for i in range(num_node)] # Every node is independent
        self.num_set = num_node

    def size_of(self, u):
        return self.size[u] # Size of u

    def find(self, u):
        # Conducts a recursive search until the major parent(root) is found
        if self.parent[u] == u:
            return u

        self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def merge(self, u, v):
        u = self.find(u) # Finds the root of u
        v = self.find(v) # Finds the root of v

        if u != v: # If the roots are not same

            # Ensures that u has a lower rank than v
            if self.rank[u] > self.rank[v]: 
                u, v = v, u

            # Higher rank will be the parent => v has higher rank
            self.parent[u] = v

            # Size of the parent is increased 
            self.size[v] += self.size[u]

            # If the ranks are equal, rank of v is added by 1
            if self.rank[u] == self.rank[v]:
                self.rank[v] += 1
            
            # Number of nodes are reduced by 1. 
            self.num_set -= 1