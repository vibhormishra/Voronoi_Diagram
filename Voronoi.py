from Queue import PriorityQueue
from BeachLineArcs import Point, Arc, Event, Edge, output;
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import Tkinter as tk
import time
from math import log

##########################Global Variables#####################################

events = PriorityQueue(); # to store  site events and circle events
root = None; # root node of the tree whose leaf nodes represent parabolic arc.

#Initialize the grid dimensions
x0 = 0.0;
grid_x = 1000;
y0 = 0.0; 
grid_y = 1000;

#===============================================================================#

def main():
    
    global x0, y0, grid_x,grid_y, events;    
    
    #Initialize the number of sites(points) in the grid
    no_of_ver = 20;
    print "Running for [%d] random points/sites" % (no_of_ver);
    
    
    #Generate points randomly
    x = np.ndarray.tolist(np.random.random_integers(10,grid_y, no_of_ver));
    y = np.ndarray.tolist(np.random.random_integers(10,grid_y, no_of_ver));
                          
    for i in range(0, no_of_ver):       
        events.put(((x[i],y[i]), Event(x[i], Point(x[i],y[i]), None, False)));
    
    print 'X: ',x, '\nY: ', y;    
    
    
    #Add margins to the bounding box.
    dx = (grid_x-x0+1.0)/5.0;
    dy = (grid_y-y0+1.0)/5.0;
    x0 -= dx;  
    grid_x += dx;  
    y0 -= dy;  
    grid_y += dy;
    
    lines = [];
    x_ = [];
    y_ = [];
    
    #Make the plot interactive which can be updated at each step
    plt.ion()
    
    ptsLeftOfBeachLine_x = [];
    ptsLeftOfBeachLine_y = [];    
    
    ptsRightOfBeachLine_x = x[:];
    ptsRightOfBeachLine_y = y[:];
    
    # Process the queues; select the top element with smaller x coordinate.
    while (not events.empty()):    
        
        del lines[:];
        del x_[:];
        del y_[:];
        event = events.get()[1];
        
        try:
            index_x = ptsRightOfBeachLine_x.index(event.point.x);
            index_y = ptsRightOfBeachLine_y.index(event.point.y);
            if index_x == index_y:
                ptsLeftOfBeachLine_x.append(event.point.x);
                ptsLeftOfBeachLine_y.append(event.point.y);
                ptsRightOfBeachLine_x.pop(index_x);
                ptsRightOfBeachLine_y.pop(index_y);
        except ValueError:
            "Do nothing";
        
        if(event.isCircleEvent):        
            lines.append([event.x,event.x]);
            lines.append([0,grid_y]);    
            circleEvent(event);
        else:
            lines.append([event.point.x,event.point.x]);
            lines.append([0,grid_y]);
            siteEvent(event.point);
                
        for i in range(0,len(output)):
            if output[i].start.x == 0 and output[i].start.y == 0:
                x_.append(output[i].end.x);
                y_.append(output[i].end.y);
            elif output[i].end.x == 0 and output[i].end.y == 0:           
                x_.append(output[i].start.x);
                y_.append(output[i].start.y); 
            else:
                lines.append([output[i].start.x, output[i].end.x]);
                lines.append([output[i].start.y, output[i].end.y]);   
            
        args = tuple(lines);    
        
        plt.axis([0, grid_x, 0, grid_x])
        plt.plot(ptsLeftOfBeachLine_x, ptsLeftOfBeachLine_y, 'go', ptsRightOfBeachLine_x, ptsRightOfBeachLine_y, 'ro', x_, y_, 'k+',*args);       
        plt.pause(0.1); 
        plt.clf()          
    
    
    # Compute the start or end points of dangling edges.
    completeEdges(); 
    
    
    lines = [];
    for i in range(0,len(output)):
        lines.append([output[i].start.x, output[i].end.x]);
        lines.append([output[i].start.y, output[i].end.y]);
        print i, ': Start =>', output[i].start.x,', ', output[i].start.y, '] End=>[', output[i].end.x,', ', output[i].end.y,']'
     
    args = tuple(lines);
    print lines
    print '\n',args;
       
    plt.plot(x,y,'go', *args);
    plt.axis([0, grid_x, 0, grid_x])
    plt.pause(0.2);
    
    while plt.get_fignums():
        try:
            plt.pause(0.2);
        except tk.TclError:
            break;
        
#===============================================================================#
        
def PerformanceAnalyzer():
        
    global x0, y0, grid_x,grid_y, events, root;    
    
    itr = 10;    
    timeTaken = [];
    numOfPoints = [];
    while itr <= 10000:
        
        grid_x = itr*10;        
        grid_y = itr*10;
        
        #Initialize the number of sites(points) in the grid
        no_of_ver = itr;                    
        
        events = PriorityQueue(); 
        x = [];
        y = [];
        try: 
            
            
            #Generate points randomly
            x = np.ndarray.tolist(np.random.random_integers(10,grid_y, no_of_ver));
            y = np.ndarray.tolist(np.random.random_integers(10,grid_y, no_of_ver));
                        
            #Start timer
            tic = time.time();
                               
            for i in range(0, no_of_ver):       
                events.put(((x[i],y[i]), Event(x[i], Point(x[i],y[i]), None, False)));
            
            interval = time.time();
            print x, '\n', y;    
            
            
            #Add margins to the bounding box.
            dx = (grid_x-x0+1.0)/10.0;
            dy = (grid_y-y0+1.0)/10.0;        
            x0 -= dx;  
            grid_x += dx;  
            y0 -= dy;  
            grid_y += dy;                    
            
            # Process the queues; select the top element with smaller x coordinate.
            while (not events.empty()):    
                #Pop element from priority queue    
                event = events.get()[1];
                
                if(event.isCircleEvent):        
                    circleEvent(event);
                else:
                    siteEvent(event.point);
            
            
            # Compute the start or end points of dangling edges.
            completeEdges(); 
            
            toc = time.time();
            print "Time interval: (%0.3f) for #points (%d): %0.3f" % (interval-tic, no_of_ver, toc-tic);
            
            timeTaken.append(toc-tic);
            numOfPoints.append(itr);
        
        except (AttributeError, ZeroDivisionError, ValueError) as err:
            #print "Got error from itr = ",itr
            #print err            
            toc = time.time();
            print "Time interval for #points (%d): %0.3f" % (no_of_ver, toc-tic);                    
        
        itr = int(round(itr * 1.2));        
                
        DeleteTree(root);
        del x[:], y[:]; 
        root = None;
        events = None;
 
    
    fig, ax1 = plt.subplots();

    fig.canvas.set_window_title('Voronoi Runtime Plot');

    plt.plot(numOfPoints, timeTaken, 'k+', [0]+numOfPoints, [0]+timeTaken)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Fortune''s Sweep Line Algorithm');
    ax1.set_xlabel('N (# Points in Voronoi Diagram)')
    ax1.set_ylabel('timetaken (in secs)')
    
    top = timeTaken[len(timeTaken)-1]*2;
    bottom = 0.0
    ax1.set_ylim(bottom, top)
    ax1.set_xlim(0, itr/2 +100);
    plt.show();

#===============================================================================#

def circleEvent(event):

    #print 'circleEvent:: Starts for Event: Point[',event.point.x,', ',event.point.y,']'
    if (event.valid):
        # Start a new edge.
        edge = Edge(event.point);            
        
        # Remove the associated arc from the beach line.
        arc = event.arc;
        #print 'circleEvent:: Removing Arc: Point[',arc.point.x,', ',arc.point.y,']'
        l = Arc.GetLeft(arc);
        r = Arc.GetRight(arc);
        
        gparent = arc.parent.parent;    #Fetching grand parent of the current node
        arcP = arc.parent;              #Fetching parent of the current node
        if(arc == arcP.Left()):
            if(gparent.Left() == arcP):
                gparent.SetLeft(arcP.Right());                
            else:
                gparent.SetRight(arcP.Right());
            r.leftNeighbor = arc.leftNeighbor;
            if arc.leftNeighbor is not None:
                arc.leftNeighbor.rightNeighbor = r;
            #print ("circleEvent:: Set left node: Left: [%d,%d] Arc: [%d, %d] Right: [%d, %d]" % (arcP.Right().leftNeighbor.point.x if arcP.Right().leftNeighbor and arcP.Right().leftNeighbor.point else 0,  arcP.Right().leftNeighbor.point.y if arcP.Right().leftNeighbor and arcP.Right().leftNeighbor.point else 0,  arcP.Right().point.x, arcP.Right().point.y,  arcP.Right().rightNeighbor.point.x if arcP.Right().rightNeighbor and arcP.Right().rightNeighbor.point else 0,  arcP.Right().rightNeighbor.point.y if arcP.Right().rightNeighbor and arcP.Right().rightNeighbor.point else 0)); 
        else:
            if(gparent.Left() == arcP):
                gparent.SetLeft(arcP.Left());
            else:
                gparent.SetRight(arcP.Left());
            l.rightNeighbor = arc.rightNeighbor;
            if arc.rightNeighbor is not None:
                arc.rightNeighbor.leftNeighbor = l;
            #print ("circleEvent:: Set right node: Left: [%d,%d] Arc: [%d, %d] Right: [%d, %d]" % (arcP.Left().leftNeighbor.point.x if arcP.Left().leftNeighbor and arcP.Left().leftNeighbor.point else 0,  arcP.Left().leftNeighbor.point.y if arcP.Left().leftNeighbor and arcP.Left().leftNeighbor.point else 0,  arcP.Left().point.x, arcP.Left().point.y,  arcP.Left().rightNeighbor.point.x if arcP.Left().rightNeighbor and arcP.Left().rightNeighbor.point else 0,  arcP.Left().rightNeighbor.point.y if arcP.Left().rightNeighbor and arcP.Left().rightNeighbor.point else 0)); 
        
        
        #print ("circleEvent:: right's stats: Left: [%d,%d] Arc: [%d, %d] Right: [%d, %d]" % (r.leftNeighbor.point.x if r.leftNeighbor and r.leftNeighbor.point else 0, r.leftNeighbor.point.y if r.leftNeighbor and r.leftNeighbor.point else 0, r.point.x, r.point.y, r.rightNeighbor.point.x if r.rightNeighbor and r.rightNeighbor.point else 0, r.rightNeighbor.point.y if r.rightNeighbor and r.rightNeighbor.point else 0));
        
        if (l):            
            l.s1 = edge;            
        
        if (r):            
            r.s0 = edge;        
    
        # Finish the edges before and after a.
        if (arc.s0 is not None):
            arc.s0.finish(event.point);
        if (arc.s1 is not None):
            arc.s1.finish(event.point);
        
        # Check for the circle events on either side of the newly added
        if (l is not None):
            checkForCircleEvent(l, event.x);
        if (r is not None):
            checkForCircleEvent(r, event.x);
            
        del event, arc, arcP;

#===============================================================================#

def siteEvent(pt):
    
    #print "siteEvent:: Starts for Points [",pt.x,',',pt.y,']' 
    global root;
    if (root is None):
        root = Arc(pt);
        root.parent = root;
        return;    

    # Find the current arc(s) at height p.y (if there are any).    
    pointItr = root;
    while (pointItr is not None):                        
        #Find the two points on beach line between which this new point will lie.
        left = Arc.GetLeftChild(pointItr);
        right = Arc.GetRightChild(pointItr);
        
        if (not left):  #if this is the extreme left node for which there would exist no node on the left
            left = pointItr;
        if (not right): #if this is the extreme right node for which there would exist no node on the right
            right = pointItr;
        
        z = Point(0.0,0.0);         
        
        (ret, direc) = doPtIntersectParabola(pt,left, z);
        
        if(ret):
            pointItr = left;
        
        if(not ret and right != left):
            (ret, direc) = doPtIntersectParabola(pt,right, z);
            if(ret):
                pointItr = right;
        
        if (ret):
            # Add the arc from this new point to the beachline, this dividing it into two                                   
            arcP = pointItr.parent;
            
            newArc1 = Arc(None);            
            newArc1.SetRight(Arc(pt));
            newArc1.SetLeft(pointItr); 
            #newArc1.point = Point(z);           
            
            newArc0 = Arc(None);
            newArc0.SetLeft(newArc1);
            newArc0.SetRight(Arc(pointItr.point));
            newArc0.Right().s1 = pointItr.s1;
            
            if (root.isLeaf):
                root = newArc0;
                root.parent = root;
            elif (arcP.Left() == pointItr):
                arcP.SetLeft(newArc0);
            else:
                arcP.SetRight(newArc0);
            
            newArc1.Left().s1 = newArc1.Right().s0 = Edge(z);
            newArc0.Right().s0 = newArc1.Right().s1 = Edge(z);                                                 
            
            newArc0.Right().rightNeighbor = newArc1.Left().rightNeighbor;
            if newArc0.Right().rightNeighbor is not None:
                newArc0.Right().rightNeighbor.leftNeighbor = newArc0.Right();
            newArc1.Left().rightNeighbor = newArc1.Right();
            newArc1.Right().leftNeighbor = newArc1.Left();
            newArc1.Right().rightNeighbor = newArc0.Right();
            newArc0.Right().leftNeighbor = newArc1.Right();
            
            # Check for new circle events around the new arc:
            #checkForCircleEvent(pointItr, pt.x);
            checkForCircleEvent(Arc.GetLeft(newArc1.Right()), pt.x);
            checkForCircleEvent(Arc.GetRight(newArc1.Right()), pt.x);                        
            
            #print "siteEvent:: Iterator"
            #DisplayTreeTraversal(root);
            #print "#####################Display Ends.###############################3"
            return;
    
        if(direc == 0):
            pointItr = pointItr.Left();
            #print "Going left"
        else:
            pointItr = pointItr.Right(); 
            #print "Going Right"

    #If straight horizontal line from p never intersects a arc between its neighbors on beach line, 
    #then just add it to the rightmost or leftmost node of the tree.    
    pointItr = root;
    while (pointItr.Right() is not None ): # Find the last rightmost node.                
        pointItr=pointItr.Right()
        
    arcP = pointItr.parent;
    newArc = Arc(None);
    newArc.SetRight(Arc(pt));
    newArc.SetLeft(pointItr);    
    
    arcP.SetRight(newArc);
        
    # Insert edge between p and node   
    start = Point(0.0,0.0); 
    start.x = x0;
    start.y = (newArc.Right().point.y + pointItr.point.y) / 2.0;
    pointItr.s1 = newArc.Right().s0 = Edge(start);
    
#===============================================================================#

def checkForCircleEvent(arc, x0):
    # Look for a new circle event for arc node.
    #print "checkForCircleEvent:: Arc: Points: [",arc.point.x,",",arc.point.y,"] "
    global events;
        
    l = Arc.GetLeft(arc);
    r = Arc.GetRight(arc);
    
    #if(l == arc or r == arc):
    #    return;
    
    # Invalidate any old event.
    if (arc.event is not None and arc.event.x != x0): 
        arc.event.valid = False;
        
    arc.event = None;

    if (not l or not r):
        return;
    
    o = Point(0.0,0.0);
        
    [x, ret] = getCircle(l.point, arc.point, r.point, o);
    
    #print "checkForCircleEvent:: Got values: x:[",x,"] ret:[",ret,"] o:[",o.x,',',o.y,']';
    
    if ( ret and x > x0):      
        #print "checkForCircleEvent:: Inserting Circle Event";  
        # Create new event.
        arc.event = Event(x, o, arc);
        events.put(((x,0), arc.event));        
   
#===============================================================================#

def getCircle(a, b, c, o):
    # Find the rightmost point on the circle through a,b,c.
        
    #print "getCircle::For points a[%lf, %lf], b[%lf, %lf], c[%lf, %lf]" % (a.x,a.y,b.x,b.y,c.x,c.y);
    # Check that bc is a "right turn" from ab.
    if ((b.x-a.x)*(c.y-a.y) - (c.x-a.x)*(b.y-a.y) > 0):        
        return 0.0, False;

    # Compute the center and rightmost point of the circle.    
    c0 = (b.x - a.x)*(a.x+b.x) + (b.y - a.y)*(a.y+b.y);
    c1 = (c.x - a.x)*(a.x+c.x) + (c.y - a.y)*(a.y+c.y);
    c2 = 2.0*((b.x - a.x)*(c.y-b.y) - (b.y - a.y)*(c.x-b.x));
    
    # Return in case all three points are co-linear.
    if (c2 == 0):
        return 0.0, False;  

    #Point o is the center of the circle.
    o.x = ((c.y - a.y)*c0-(b.y - a.y)*c1)/(c2*1.0);
    o.y = ((b.x - a.x)*c1-(c.x - a.x)*c0)/(c2*1.0);

    # o.x plus radius equals max x coordinate.
    x = o.x + math.sqrt( pow(a.x - o.x, 2.0) + pow(a.y - o.y, 2.0) );
    #print "circle::the value of x: [%lf] and o[%lf, %lf]\n" % (x,o.x,o.y);
    return x, True;

#===============================================================================#

def doPtIntersectParabola(p, arc, res):
    # Check whether the point intersect the arc of beachline and if it does, then compute the point of intersection
    #print "doPtIntersectParabola:: Checking intersect for points:p: [",p.x,p.y,"], i:[",arc.point.x,arc.point.y,"]";
    direc = -1;    # 0:Left , 1:Right => Tells direction to which we should proceed next in case False is returned.
    if (arc.point.x == p.x):
        if(arc.point.y > p.y):
            direc = 0;
        else:
            direc = 1;
        return False, direc;

    ly = 0.0;
    ry = 0.0;
    
    leftArc = Arc.GetLeft(arc);
    rightArc = Arc.GetRight(arc);
    
    if (leftArc): # Get the intersection of left arc with this arc.
        ly = getParaIntersectionPoint(leftArc.point, arc.point, p.x).y;
    if (rightArc): # Get the intersection of right arc with this arc.
        ry = getParaIntersectionPoint(arc.point, rightArc.point, p.x).y;

    if ((not leftArc or ly <= p.y) and (not rightArc or p.y <= ry)):        
        res.y = p.y;

        #Plug it back into the parabola equation.
        res.x = (arc.point.x*arc.point.x + (arc.point.y-res.y)*(arc.point.y-res.y) - p.x*p.x)/ (2.0*arc.point.x - 2.0*p.x);
        
        #print "doPtIntersectParabola::For point p: [",p.x,", ",p.y,"]";
        #print "doPtIntersectParabola::res [: ",res.x,", ",res.y,"]\n";
      
        return True, direc;
    
    if (not leftArc or ly > p.y):
        direc = 0;
    else:
        direc =1;
    return False, direc;

#===============================================================================#

def getParaIntersectionPoint(p0, p1, px):
    # Calculate intersection of parabolas
    
    res = Point(0.0,0.0);
    p = p0;

    if (p0.x == p1.x):
        res.y = (p0.y + p1.y) / 2.0;
    elif (p1.x == px):
        res.y = p1.y;
    elif (p0.x == px):
        res.y = p0.y;
        p = p1;

    else:
        # solving quadratic equations of parabolas using standard formula: -b +- sqrt(b*b - 4*a*c)/(2*a)
        z0 = 2.0*(p0.x - px);
        z1 = 2.0*(p1.x - px);

        a = 1.0/z0 - 1.0/z1;
        b = -2.0*(p0.y/z0 - p1.y/z1);
        c = (p0.y*p0.y + p0.x*p0.x - px*px)/z0 - (p1.y*p1.y + p1.x*p1.x - px*px)/z1;

        res.y = ( -b - math.sqrt(b*b - 4.0*a*c) ) / (2.0*a);
   
    # Substitute the above computed y in parabola equations.
    res.x = (p.x*p.x + (p.y-res.y)*(p.y-res.y) - px*px)/(2.0*p.x-2.0*px);
    #print "getParaIntersectionPoint::For point p0: [" ,p0.x,p0.y,"]","p1: [" ,p1.x,p1.y,"]";
    #print "getParaIntersectionPoint::res.x: ",res.x,"\tres.y: ",res.y;
    return res;

#===============================================================================#

def completeEdges():
    # Advance the sweep line so no parabolas can cross the bounding box.
    l = grid_x + (grid_x-x0) + (grid_y-y0);
    # Extend each remaining edge to the new parabola intersections.
    TreeTraversal(root, l);

#===============================================================================#

def TreeTraversal(node, l):
    
    if (node.isLeaf):
        if (node.s1):
            node.s1.finish(getParaIntersectionPoint(node.point, Arc.GetRight(node).point, l*2));
    else:
        TreeTraversal(node.Left(), l);
        TreeTraversal(node.Right(), l);

#===============================================================================#
  
def DisplayTreeTraversal(node):
    if (node.isLeaf):
        print ("DisplayTreeTraversal:: Left: [%d,%d] Arc: [%d, %d] Right: [%d, %d]"
            % (node.leftNeighbor.point.x if node.leftNeighbor and node.leftNeighbor.point else 0, 
               node.leftNeighbor.point.y if node.leftNeighbor and node.leftNeighbor.point else 0, 
               node.point.x, node.point.y, 
               node.rightNeighbor.point.x if node.rightNeighbor and node.rightNeighbor.point else 0, 
               node.rightNeighbor.point.y if node.rightNeighbor and node.rightNeighbor.point else 0)); 
        if node.event is not None:
            print " Event(Points): [", node.event.point.x,",",node.event.point.y,"]";
        print ("DisplayTreeTraversal:: Segments: S0: start[%lf, %lf] end[%lf, %lf]   S1: start[%lf, %lf] end[%lf, %lf]\n"
           %    (node.s0.start.x if node.s0 else 0, node.s0.start.y if node.s0 else 0, node.s0.end.x if node.s0 else 0, node.s0.end.y if node.s0 else 0
                 ,node.s1.start.x if node.s1 else 0, node.s1.start.y if node.s1 else 0, node.s1.end.x if node.s1 else 0, node.s1.end.y if node.s1 else 0));
    else:
        DisplayTreeTraversal(node.Left());
        DisplayTreeTraversal(node.Right());        

#===============================================================================#
  
def DeleteTree(node):
    
    if node is None:
        return;
    while(node is not None):
        
        if (node.isLeaf):            
            temp = node;
            node = Arc.GetRight(node);            
            del temp;
        else:
            node = Arc.GetLeft(node)
#===============================================================================#

###################################Execution starts here############################################
#Call main function
main();
#PerformanceAnalyzer();
