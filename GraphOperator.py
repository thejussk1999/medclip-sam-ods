import math

import OptimizedUnionFind as uf

def get_diff(img, x1, y1, x2, y2):
    # Calculate the distance between 2 points
    r = (img[0][y1, x1] - img[0][y2, x2]) ** 2
    g = (img[1][y1, x1] - img[1][y2, x2]) ** 2
    b = (img[2][y1, x1] - img[2][y2, x2]) ** 2
    return math.sqrt(r + g + b)

def get_threshold(k, size):
    # Calculate the threshold
    return (k / size) # Controls the segment size

def create_edge(img, width, x1, y1, x2, y2):
    # Vetrex id = x+y*width
    vertex_id = lambda x, y: y * width + x

    # Get the Eucledian distance between 2 points and image
    w = get_diff(img, x1, y1, x2, y2)

    # Calculate vertex id of two points and the distance Edge=(Vert_ID(p1),Vert_ID(p2), distance)
    return (vertex_id(x1, y1), vertex_id(x2, y2), w)

def build_graph(img, width, height):
    graph = []

    for y in range(height):
        for x in range(width):
            if x < width - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y)) # Width level edge building
            if y < height - 1:
                graph.append(create_edge(img, width, x, y, x, y + 1)) # Height level edge 
            if x < width - 1 and y < height - 1:
                graph.append(create_edge(img, width, x, y, x + 1, y + 1)) # Diagonal level edge
            if x < width - 1 and y > 0:
                graph.append(create_edge(img, width, x, y, x + 1, y - 1)) # Off diagonal level edge

    return graph

def remove_small_component(ufset, sorted_graph, min_size):
    for edge in sorted_graph:
        u = ufset.find(edge[0]) # Find the parent of p1
        v = ufset.find(edge[1]) # Find the parent of p2

        # If parents are not same
        if u != v: 
            if ufset.size_of(u) < min_size or ufset.size_of(v) < min_size:
                ufset.merge(u, v) # Merge u and v if either of their size is less than min_size

    return ufset

def segment_graph(sorted_graph, num_node, k):
    ufset = uf.OptimizedUnionFind(num_node)
    threshold = [get_threshold(k, 1)] * num_node  # Initializes the size of segment for each node

    for edge in sorted_graph:
        u = ufset.find(edge[0]) # Find the parent of p1
        v = ufset.find(edge[1]) # Find the parent of p2
        w = edge[2] # Find the distance between p1 and p2

        if u != v:
            # If they are not same parents
            if w <= threshold[u] and w <= threshold[v]:
                # If the distance between them is less than the threshold of each of their parents
                ufset.merge(u, v) # Merge the points
                parent = ufset.find(u) # Find the parent of u
                threshold[parent] = w + get_threshold(k, ufset.size_of(parent)) # Add the distance to the threshold

    return ufset
