# define the structure of graph
# define a graph which is a length * width graticule


class Graph(object):
    def __init__(self, length, width):
        '''parameters:
           length: the column number of graph
           width: the row number of graph
           the index of node is from 1'''
        self.length = length
        self.width = width
        self.node = length * width
        self.adj = [[] for _ in range(1, self.node + 1)]
        count = 1
        #connect the node(can modify)
        for i in range(self.node):
            if i + length < self.node:
                self.adj[i].append(i + length + 1)
                self.adj[i + length].append(i + 1)
            if count != length:
                self.adj[i].append(i + 2)
                self.adj[i + 1].append(i + 1)
                count += 1
            else:
                count = 1

    #print every node's neighbor index
    def print_graph(self):
        for i in range(self.node):
            print('node:', i, self.adj[i])

    #return the neighbor index of specific node
    def adj_of_vertex(self, node_number):
        neighbor_node = self.adj[node_number]
        return neighbor_node


if __name__ == '__main__':
    g = Graph(3, 4)
    g.print_graph()
    action = g.adj_of_vertex(2)
    print(action)