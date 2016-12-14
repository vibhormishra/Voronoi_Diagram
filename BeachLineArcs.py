#Global variable which stores final output of edges.
output = [];

# This class stores points's x nd y coordinates point
class Point(object):
    
    x = 0.0;
    y = 0.0;

    def __init__(self, nx, ny):
        self.x = nx; 
        self.y = ny;
    
    
class Event(object):    

    def __init__(self, x, pt, arc, cEvent=True):
        self.x = x;
        self.point = pt;
        self.arc = arc;
        self.valid = True;          
        self.isCircleEvent = cEvent;
            
class Edge(object):

    def __init__(self, site):
        self.start = site;
        self.end = Point(0.0,0.0);
        self.done = False;      
        output.append(self);

    # Set the end point and mark as "done."
    def finish(self, site):
        if (self.done):
            return; 
        self.end = site; 
        self.done = True; 

#    Beachline consists of arcs from various sites. This class is used to store all such arcs information
#    It can represent an arch of parabola or an intersection between two archs (which defines an edge).
class Arc(object):
    
    def __init__(self, pt, leftChild=None, rightChild=None):
        self.point = pt;                #point corresponding to the this arc
        if pt is None:                  
            self.isLeaf    = False;     #boolean value to check if this node is leaf or not.
        else:
            self.isLeaf    = True;
        self._left = leftChild;         #stores left child of this current node(site)
        self._right = rightChild;       #stores right child of this current node
        self.s0 = None;                 #stores first edge from this site
        self.s1 = None;                 #stores second edge from this site
        self.event = None;                  #stores the event information related to this
        self.parent = None;             #Stores the parent of current node to allow backtracking.
        self.leftNeighbor = None;
        self.rightNeighbor = None;
    
    
    def SetLeft (self, point):
        self._left  = point; 
        point.parent = self;
        
    def SetRight(self, point):
        self._right = point; 
        point.parent = self;
        

    def Left (self):
        return self._left;
    
    def Right(self):
        return self._right; 
    
      
    @staticmethod
    def GetLeftParent(p):
    #returns the closest parent which is on the left of current node.
        par        = p.parent;
        pLast    = p;
        while(par.Left() == pLast):
        
            if(par.parent is None or par == par.parent):
                return None;
            pLast = par; 
            par = par.parent; 
                    
        return par;
    
        
    @staticmethod
    def GetRightParent(p):
    #returns the closest parent which is on the right of current node.    
        par        = p.parent;
        pLast    = p;
        while(par.Right() == pLast):
            
            if(par.parent is None or par == par.parent): 
                return None;
            
            pLast = par; 
            par = par.parent; 

        return par;
        
    @staticmethod
    def GetLeftChild(p):
    #returns the closest leaf node which is on the left of current node
        if(p is None):
            return None;
        
        par = p.Left();
        while(par is not None and not par.isLeaf):
            par = par.Right();
            
        return par;
        
    @staticmethod
    def GetRightChild(p):
    #returns the closest leaf which is on the right of current node    
        if(p is None):
            return None;
        par = p.Right();
        while(par is not None and not par.isLeaf):
            par = par.Left();
        return par;    
    
    @staticmethod
    def GetLeft(p):
    #returns the closest left leaf of current node    
        #return Arc.GetLeftChild(Arc.GetLeftParent(p));
        return p.leftNeighbor;
        
    @staticmethod
    def GetRight(p):
    #returns the closest right leaf of current node        
        #return Arc.GetRightChild(Arc.GetRightParent(p));
        return p.rightNeighbor;
    

