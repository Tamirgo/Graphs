import numpy as np
import random
import math

class Graphs:
    def __init__(self,vertecies,clique=False,directed = False):

        #THE USER MIGHT WANT A KLIQUE GRAPH, Kn SO I GAVE THEM THE CHOICE WITH DEFAULT SET TO NO.
        self.num_of_vert = len(vertecies)#number of vertecites.
        if clique == True:
            self.Graph =  np.ones((self.num_of_vert,self.num_of_vert),dtype='uint8')
        else:
            self.Graph = np.zeros((self.num_of_vert,self.num_of_vert),dtype='uint8')

        #THE USER IS GIVEN THE OPTION TO CREATE A DIRECTED AND UNDIRECTED GRAPH.
        self.directed = directed
        #with dictionary it will be faster to get to each vert.
        self.graph_dict = {} # these are the vertecies
        #filling up the graph dictionary.
        self.edges = {} # these are the edges.
        for i in range(self.num_of_vert):
            self.graph_dict[vertecies[i]] = i

        #time is for the DFS algorithm which will be written later on.
        dfs_time = 0

        #if the user sets up weights to the edges of the graph, if there are negative weights, Dijkstra Algorithm
        #will not work correctly. hence, it is needed to know whether there are negative weights or not.
        self.negative = 0


    def set_Neighbours(self,vert1,vert2):
        '''
        this method will get tuples of two vertecies, which will be considered neighbors.
        will act differently if the graph is directed or undirected.
        :return: it just affects the self.Graph.
        '''
        self.Graph[self.graph_dict[vert1]][self.graph_dict[vert2]] = 1
        self.edges[(vert1,vert2)] = 'No' # giving it no weight until the user will want to set weights for it.

        #this statement is for the case the graph is undirected, then if a,b are neighbouts, b,a are also neighbours.
        if self.directed is False:
            self.Graph[self.graph_dict[vert2]][self.graph_dict[vert1]] = 1

    def set_Whole_Neighbours(self,tuple_list):
        '''
        this will use the self.set_Neighbours method to set up all the neighbours in the graph
        :param tuple_list: list of pairs of vertecies.
        :return: nothing.
        '''
        for i in tuple_list:
            self.set_Neighbours(i[0],i[1])


    def unset_Neighbours(self,vert1,vert2):
        '''
        this method deletes an edge from two vertecies.
        :param vert1:
        :param vert2:
        :return:
        '''
        self.Graph[self.graph_dict[vert1]][self.graph_dict[vert2]] = 0
        self.edges.pop(vert1,vert2)#deleting the edge.

        #this statement is for the case the graph is undirected, then if a,b are neighbours, b,a are also neighbours.
        if self.directed is False:
            self.Graph[self.graph_dict[vert2]][self.graph_dict[vert1]] = 0




    def areNeighbours(self,node1,node2):
        '''

        :param node1: self explanitory.
        :param node2:
        :return: returning the value of the matrice in the position node1,node2. if its 1 then it will return 1
        else it will be zero...
        '''

        return self.Graph.item(self.graph_dict[node1],self.graph_dict[node2])
    def getVertecies(self):
        '''

        :return: the Vertecies of the graph.
        '''
        return self.graph_dict
    def getEdges(self):
        '''

        :return: the edges of the graph.
        '''
        return self.edges
    def getDirecition(self):
        '''
        in case the user wants to know if the Graph is Directed or not.
        :return:
        '''
        return self.directed

    def setWeights(self,randomWeights = False):

        #using this flag on both cases (randomized weights of manual).
        flag = 1
        # IN CASE THE USER WANTS RANDOM WEIGHTS ON THEIR EDGES.
        if randomWeights is True:


            # making a loop which makes sure the user inputs a right input in case of random weights chosen.
            while(flag):
                print('Please enter a range of weights:')

                # the try statement is in case the user tried to input a non numeric value.
                try:
                    begining = np.float(input('Begining:'))
                    end = np.float(input('End:'))

                    #also, the beggining should be lesser then the end.
                    if end <=begining:
                        print("Invalid input! The ending Cannot be lesser than the Beginning.")
                        flag = 1
                    else:
                        flag = 0
                except:
                    print("Invalid inputs,Please try again")
                    flag = 1


             # end while

            #checking if there is an option for negative weight in the graph.
            if begining < 0:
                self.negative = 1
            #giving the edges the random weights.
            for keys in self.edges.keys():
                self.edges[keys] = random.uniform(begining,end)
            return self.edges
            #end for.


        #END OF RANDOM CASE.

        # IN CASE THE USER WANTS TO INPUT THEIR OWN WEIGHTS.
        else:#the else is not a nessacity but its more readable.
            #an outter for loop for all the edges.
            for keys in self.edges.keys():
                flag = 1 # will be used for the while loop just like before.
                #while the input is not acceptable this while will run.
                while(flag):
                    # trying to convert it into float
                    try:
                        weight = np.float(input('Weight of the edge :' + str(keys)))
                        self.edges[keys] = weight

                        #again, recording the fact that there are negative weights.
                        if weight < 0 and self.negative == 0:
                            self.negative = 1
                        flag = 0
                    #if the weights are non numeric
                    except:
                        print("Invalid Weights' Please try again")
                        flag = 1
            # end for.
            # END OF NON RANDOM CHOICE.
            return self.edges

    def BFS(self,startingNode):
        '''
        input = self, Graph.
        :return: a Dictionary  of Dictionaries since its very easy to understand the result that way.
        '''


        # Dictionary which will store all the results of each vertecie. the key will be the ver and the value will
        #be a list with the parent node,distance,color.
        resultDict = {}
        # creating an empty queue for the algorithm.
        algoQueue = []


        '''
            The Algorithm first passes on ALL nodes and reset their attributes. the color is white, the parent is NIL
            and the distance is infinity.
            
            then, it uses a Queue to traverse on the nodes and give them the right data.
            BFS is moving on each layer of nodes and according to the layer's index, all the nodes in that layer
            get the distance which is the layer's index. the parent is the node the algoritm reached them from and 
            the color determines if they are inside the queue or not. white means they have yet to be visited,gray
            means the node is inside the queue and black means they are done.
            
            in the end the algorithm returns the result dictionary.
        '''

        #RESETTING ALL THE NODES TO HAVE THE DEFAULT VALUES.
        for key in self.graph_dict.keys():
            resultDict[key] = {'Parent':'Nil','Distance':float('Inf'),'Color':'White'}

        # GETTING THE STARTING NODE ITS RIGHT VALUES. COLOR WILL BE BLACK AFTER WE FINISH WITH IT.
        algoQueue.append(startingNode)
        resultDict[startingNode]['Distance'] = 0
        resultDict[startingNode]['Color'] = 'Gray'

        #RUNNING AS LONG AS THE QUEUE IS NOT EMPTY
        while(len(algoQueue)):

            # always getting the first node in the queue.
            node1 = algoQueue[0]
            for node2 in self.graph_dict.keys():
                # only white neighbours.
                if self.areNeighbours(node1,node2) and resultDict[node2]['Color'] == 'White':

                    # if it is a white neighbour - inserting it into the queue
                    algoQueue.append(node2)

                    # the parent is node1 - the node that reached the current node.
                    resultDict[node2]['Parent'] = node1

                    # parent distance which is node1 distance+1.
                    resultDict[node2]['Distance'] = resultDict[node1]['Distance']+1

                    # color will be gray since all the nodes in the queue are gray.
                    resultDict[node2]['Color'] = 'Gray'

            #end of for loop.

            # when we finish with that node, we color it with black and pop it out of the Queue
            resultDict[node1]['Color'] = 'Black'
            algoQueue.pop(0)

        #end of while loop
        return resultDict#in the end, the result dict which stores all the information needed is returned.

    def DFS_VISIT(self, currentNode,resultDict):
        '''

        :param currentNode: the node which we will go from, searching for a path.
        :param resultDict: the graph itself, in order to find the next white neighbour, the next node in that path.
        :return: not returning anything, just changing the dictionary values if needed.

        DFS_VISIT is the main algorithm in DFS, since DFS is like a maze, we are going from one node to another
        in a path, finding all the nodes in that path and marking them with time,color and parent.
        Beggining is the time we reached that node, Finish is the time we ended that node, which means there are
        no more paths from that node that our DFS has not discovered through that node or through some other node.

        '''
        # turning the color of the current node into gray since the node is now spotted by DFS.
        resultDict[currentNode]['Color'] = 'Gray'

        # getting the beggining time to be the last recorded time.
        resultDict[currentNode]['Beggining'] = self.dfs_time

        # incremeting the recorded time.
        self.dfs_time+=1

        # now running on all the neighbours in the search for a white neighbour:
        for neighbour in resultDict.keys():
            #if we find a node which is a neighbour and also white we will summon the dfs visit method on it.
            if self.areNeighbours(currentNode,neighbour) and resultDict[neighbour]['Color'] == 'White':

                # since we got to that node from currentNode, currentNode is the parent.
                resultDict[neighbour]['Parent'] = currentNode
                self.DFS_VISIT(neighbour,resultDict)#using recursion on the white neighbour.

        # The node has finished and it has no white neighbours, hence we will paint it in black and give it a finish time.
        resultDict[currentNode]['Color'] = 'Black'
        resultDict[currentNode]['Finish'] = self.dfs_time
        #every time a node finish,we give it the finish time and then increment the time as DFS goes.
        self.dfs_time+=1



    def DFS(self,startingList = None):
        '''
        input: a Graph - self.
        :return: a dictionary, the keys will be the nodes, the values will be (parent,Start Time,Finish time,color)
        '''
        # Dictionary which will store all the results of each vertecie. the key will be the ver and the value will
        #be a list with the parent node,distance,color.
        resultDict = {}
        #resetting the time whenever DFS has been called.
        self.dfs_time = 0
        ########################################################
        # IN CASE NO SPECIFIC NODE OR NODE LIST WERE GIVEN:
        # resetting all the nodes. i am shuffeling the keys so the DFS will always run differently in case no node was given.
        if startingList is None:
            nodeList = list(self.graph_dict.keys())
            random.shuffle(nodeList)
        # IN CASE A NODE OR A LIST OF NODES WERE GIVEN:
        #######################################################
        else:
            # making sure the nodes are actually in the graph.
            for i in startingList:
                if i not in self.graph_dict:
                    print('The list consists of a node which is not in the Graph. make sure you have entered a valid list')
            # so the nodes are all valid, now we need to make a nodelist which starts with the nodes the user gave us.

            # in this case, the user gave me the whole graph list which means they know the order of the nodes need to be
            #checked. (will be used in SCC for example.
            if len(startingList) == len(self.graph_dict.keys()):
                nodeList = startingList

             # so in the case the starting list is not equal to the whole V of the Graph, we will want a nodeList
            # in the order of the starting list, and the rest will be shuffled.
            #so what i did was creating a temp variable called nodesNotGiven which will be a list of all the nodes that
            #were not given. then i use python operator overload where '-' is difference between two lists.
            #so i get nodeList to be the startinglist + the nodes that were not given. the starting list maintain the order
            #given, and the nodesNotGiven gets to be shuffeled. and then i concat with '+' the two lists.
            else:
                nodesNotGiven = list(set(self.graph_dict.keys()) - set(startingList))#the - operator works on sets only.
                random.shuffle(nodesNotGiven)
                nodeList = startingList + nodesNotGiven
        ########################################################
        for node in nodeList:
            resultDict[node] = {'Parent':'Nil','Color':'White','Beggining':'0','Finish':'0'}

        # THE DFS ALGO:

        for node in nodeList:
            if resultDict[node]['Color'] == 'White':#if the node is white, we need to summon the recursive call.
                self.DFS_VISIT(node,resultDict)

        return resultDict


    def relax(self,targetNode,sourceNode,resultdict):
        '''

        :param targetNode: the node that we try to relax.
        :param sourceNode: the node which we see if can relax its neighbours.
        :param resultdict: the current weights of all the nodes.

        so we check if the graph is directed or not. since our edge dict includes only ordered pairs
        we have to make sure the graph's directness. if it is undirected we need to take into account both
        (u,v) and (v,u) - in the Dijkstra algo we checked if (u,v) is an edge. it is an edge but in the edge
        dict we might see only (v,u) so if we try to relax (u,v) we won't have it in the dict and it will cause
        an exception. hence if it did, it means the edge dict have  (v,u) and that is why we will relax (v,u)
        which is the same edge.
        and if the graph is directed we don't have to worry about the directions since we have only one.
        '''

        try:
            current_Weight = resultdict[targetNode]['WeightedDistance']
            relaxation_Weight = resultdict[sourceNode]['WeightedDistance'] + self.edges[(sourceNode,targetNode)]
            if current_Weight > relaxation_Weight:
                resultdict[targetNode]['WeightedDistance'] = relaxation_Weight
                resultdict[targetNode]['Parent'] = sourceNode
        except:
            current_Weight = resultdict[targetNode]['WeightedDistance']
            relaxation_Weight = resultdict[sourceNode]['WeightedDistance'] + self.edges[(targetNode,sourceNode)]
            if current_Weight > relaxation_Weight:
                resultdict[targetNode]['WeightedDistance'] = relaxation_Weight
                resultdict[targetNode]['Parent'] = sourceNode

    def Dijkstra(self,startingNode):
        '''
        input: a weighted Graph.
        :return: a Dictionary - keys = nodes, values = (Parent,Distance,Color)

        ALGORITHM:
        the algorithm creates a queue with nodes and weights. it will run untill the queue will be empty and will
        always pick the node with a minimal weight.
        from that node, just like BFS and will check all its neighbours but whenever it will find a non-black neighbour
        it will try to "relax" it. relaxation is the operation where we check if the current path is lighter than
        the path which we got to that node from the other path.

        in the end we return the result dict which is a dict that holds the information about each node in the graph.
        '''


        #WARNING TO THE USER IN CASE THE ALGO SPOTTED THE OPTION FOR NEGATIVE WEIGHTS:

        if self.negative:
            print ("We have spotted that there might be negative weights in the Graph, Dijkstra Does not always give" \
                  " correct results when there are negative weights.")

        # Dictionary which will store all the results of each vertecie. the key will be the ver and the value will
        #be a dictionary with the parent node,distance,color.
        resultDict = {}
        #making the starting node the first position in the dictionary.
        resultDict[startingNode] = {'Parent':'Nil','WeightedDistance':0,'Color':'White'}
        for keys in self.graph_dict.keys():
            if keys != startingNode:
                resultDict[keys] = {'Parent':'Nil','WeightedDistance':float('inf'),'Color':'White'}
        #end for

        # THE ALGORITHM:

        #getting an empty queue, and inserting it with the starting node which the user gave the method.
        myQueue = []
        myQueue.append(startingNode)

        #the main loop.
        while(len(myQueue)):#runnig as long as the queue isn't empty.

            # in case its the first itteration its obvious, in other cases we need a min to compare it to.
            minNode = myQueue[0]


            #finding the minimum in the queue.
            #even though its not very pythonic, it is confortable, in case we have multiple minimums it will find the first one.
            for i in myQueue:
                if resultDict[i]['WeightedDistance'] < resultDict[minNode]['WeightedDistance']:
                    minNode = i


        # now finding that node's neighbours, relaxing them if needed and inserting them into the queue.
            for node in resultDict.keys():
                #making sure i need to insert the node into the Queue.
                if self.areNeighbours(minNode,node) and resultDict[node]['Color'] == 'White':

            #end for

                    # making sure im not reappending the node.
                    if node not in myQueue:
                        myQueue.append(node)

                    #relaxing the edge. the relax method will change the node's weight if needed.
                    self.relax(node,minNode,resultDict)

            # removing the node and coloring it in black so we won't consider it again.
            myQueue.remove(minNode)
            resultDict[minNode]['Color'] = 'Black'

        #end while loop

        return resultDict

    def compute_TransposeGraph(self, G):
        '''

        :param G: a Directed Graph which we will reverse.
        :return: a new graph with the opposite edges.
        '''

        if G.getDirecition() is False:
            print('This graph is undirected, Transpose has no meaning in an undirected Graph.')
            return None
        else:

            new_Graph = Graphs(list(G.getVertecies().keys()), directed=True)  # creating a new graph with no edges yet.
            new_tuple_list = []  # empty list.

            # now setting the new edges to be in the opposite direction from the original
            for keys in G.getEdges().keys():
                new_tuple_list.append((keys[-1], keys[-2]))

            # now setting the graph edges with the method
            new_Graph.set_Whole_Neighbours(new_tuple_list)

            # returning the Graph to the user.
            return new_Graph

    def check_Connectivity(self,G):
        '''

        :param G: a (non) Directed Graph.
        :return: boolean if the graph is connected or not.
        acts differently if the graph is directed or not.
        '''
        # IN CASE OF AN UNDIRECTED GRAPH:
        if G.getDirecition() is False:
            test_res = G.BFS(random.choice(list(G.getVertecies().keys())))
            counter = 0
            for keys in test_res.keys():
                if test_res[keys]['Parent'] == 'Nil':
                    counter+=1
                if counter > 1:
                    return False
            print(counter)
            return True
        # IN CASE THE GRAPH IS DIRECTED - A NODE HAS TO BE A ROOT IN BOTH G AND G TRANSPOSE.
        else:
            #picking a random node.
            selected_vert = random.choice(G.getVertecies().keys())

            #calling BFS on it.
            temp_dict = G.BFS(selected_vert)

            #a counter to count the num_of_nils.
            counter = 0
            #now traverse on the result dict couting nils.
            for keys in temp_dict.keys():
                if temp_dict[keys]['Parent'] == 'Nil':
                    counter+=1
                #if more than one found we can stop right here.
                if counter > 1:
                    return False


            #computing the Transpose graph:
            transpose = G.compute_TransposeGraph(G)
            #now we will traverse on the tranpose graph with BFS with the same node as before:
            temp_dict = transpose.BFS(selected_vert)
            counter = 0

            #making sure its a root in the transpose as well.
            for keys in temp_dict.keys():
                if temp_dict[keys]['Parent'] == 'Nil':
                    counter+=1

                #if not then the graph isn't strongly connected.
                if counter > 1:
                    return False

            #else it it.
            return True

    def isAcycle(self):
        '''
        input - an (un)directed graph.
        :return: boolean if the  DIRECTED graph has any cycles in it.

         if the graph is undirected, a simple BFS will do the trick , after we run BFS on a random node we need to
         traverse the edges and find out if the next condition is True:
         for each (x,y) that belongs to the set E do:
            if the parent of x is not y and the parent of y is not x then we have a cycle.
        explaination:
            when we got to an edge which is built from two nodes, if bfs found x through one path and y through a different
            path, then we have two different paths which were closed by the edge (x,y). hence we found a cycle.

         in case its a directed graph
        we need to look for a back edge. if we find one - the Graph has cycles.
        else return false.

        a backedge is an edge where the DFS found an edge (u,v) such that :
        v is an ancesstor of u. this means there is a cycle since u->v but DFS found a path that v was found before
        u.
        '''

        if self.directed is False:
            #calling the bfs algorithm.
            bfs_result = self.BFS(random.choice(self.graph_dict.keys()))

            # traversing through the edges and looking for the condition which was explained earlier.
            for edges in self.edges.keys():
                #using x,y to make it easier to write and more readable.
                x = edges[0]
                y = edges[1]
                #basically asking for each(x,y) which belong to the set E do:
                if bfs_result[x]['Parent'] != y and bfs_result[y]['Parent'] != x:#we have a cycle!
                    return False
            #end for
            return True


        #END OF UNDIRECTED ALGORITHM.



        else:
            # calling DFS, doesnt matter which node is the start node, hence - im not inserting a list of nodes for the order
            dfs_result = self.DFS()

            #a loop that traverse through the graph edges:
            for edge in self.edges: #(u,v)=> u = edge[0],v = edge[1]. if edge[1] is an anccestron then return True
                u = dfs_result[edge[0]]
                v = dfs_result[edge[1]]
                #so if we got a back edge, we have a cycle - its not Acycle => false.
                if v['Beggining'] < u['Beggining']< u['Finish']< v['Finish']:
                    return False
            #end for
            return True


    def isBiPartite(self):
        '''

        input  - Un Directed Graph
        :return: boolean - is the graph bipartite or not


        The algorithm: the known claim is :
        an undirected graph is bipartite if and only if it does not contain a cycle with odd number of vertecies.
        a bipartite graph has a chromatic number of at most 2, a cycle with odd number of vertecies has a chromatic
        number of 3, hence it cannot contain it.

        so we will use BFS and we will see if it is connected (for the Kn,m part) and also traverse the edges in the
        lookout for an odd cycle. the way we do that is to check if for each (x,y) that belongs to the set E
        x[distance]  =  y[distance]. if the source node found two differenct paths => one to x and one to y and they are
        the same lengh. it means the two paths of lengh k + the lengh of an edge (1) is 2k+1 = odd number.
        '''

        #the algorithm:

        #running BFS with a random node since it doesn't matter.
        bfs_result = self.BFS(random.choice(self.graph_dict.keys()))
        for edges in self.edges.keys():#for each (x,y) that belongs to E do:
            x = edges[0]
            y = edges[1]
            if bfs_result[x]['Distance'] == bfs_result[y]['Distance']:
                return False
        #end for

        #so if we got here it means we did not find any odd circle, hence we know its bipartite.
        return True








