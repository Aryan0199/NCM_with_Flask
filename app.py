# Importing necesssary libraries
import numpy as np

# To handle indeterminate variable value in matrix
import sympy as sym
from sympy import *
J=symbols('J')  # We are using 'J' to indicate that the value is indeterminate
import random
import networkx as nx
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
import json
import os
import colorsys
from PIL import Image
PEOPLE_FOLDER = os.path.join('static', 'images')
E=sym.Matrix()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
numOfNodes=0
def file_exists(file_path):
    return os.path.exists(file_path)
@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/input')
def input():
  return render_template('input.html')

@app.route('/inputNeu')
def inputNeu():
  return render_template('inputE.html')

@app.route('/inputLt')
def inputLt():
  return render_template('inputL.html')


@app.route('/nodes', methods=['POST'])
def my_form_post():
  file_path="static/images/graph.png"
  if file_exists(file_path):
      os.remove(file_path)

  global E
  matrix = sym.Matrix()
  row=[]
  numOfNodes=request.form['nodes']
  print("Number of nodes+1 are: ")
  numOfNodes=int(numOfNodes)
  # numOfNodes=numOfNodes
  print(numOfNodes)
  

  for i in range(numOfNodes):
    string = request.form['field_{}'.format(i+1)]
    temp = string.split() 
    temp=[eval(i) for i in temp] #to int
    v1=np.array(temp)
    print(v1)
  
    # v2.append
    row.append(temp)
    print(row)

  string2 = request.form['cncpt']
  cncpt = string2.split()
  print(cncpt)
  # Inserting each row into matrix.
  for i in range(0,numOfNodes):
    matrix = matrix.row_insert(i,Matrix([row[i]]))
  print("The matrix is: ")
  print(matrix)
  G = nx.DiGraph()
  i=0
  j=0
  E=matrix
  while(i<shape(E)[0]):
    j=1
    while(j<shape(E)[0]):
    #  print(int(M.row(i)[j]))
      if(E.row(i)[j]!=0):
        if(E.row(i)[j]!=J):
          G.add_edge(i+1,j+1,weight=int(E.row(i)[j]))
        else:
          G.add_edge(i+1,j+1,weight=-100)
        

      j+=1
    i+=1
  elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] !=-100]
  esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == -100]
  # Create a dictionary that maps node names to random colors
  node_colors = {}
  for node in G.nodes():
    # Generate a random color with light shade
    hue = random.uniform(0,1)
    # saturation = random.uniform(0.5, 1.0)
    saturation = 0.8

    lightness = random.uniform(0.7, 1.0)
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    node_colors[node] = hex_color
  pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility

  # nodes
  nx.draw_networkx_nodes(G, pos, node_size=1000,node_color=list(node_colors.values()))

  # edges
  nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
  nx.draw_networkx_edges(
      G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="r", style="dashed"
  )


  # node labels
  mapping = dict(zip(G.nodes(), cncpt))
  nx.draw_networkx_labels(G, pos,labels=None, font_size=20, font_family="sans-serif")
  
  # edge weight labels
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels)


  # Create the legend
  legend_handles = []
  i=0
  for node, color in node_colors.items():
    legend_handles.append(plt.Line2D([], [], linewidth=0, marker='o', markersize=10, markerfacecolor=color, label=cncpt[i]))
    i=i+1
  # plt.legend(handles=legend_handles, loc='lower center', ncol=len(node_colors))
  plt.legend(handles=legend_handles, bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure)

  ax = plt.gca()
  ax.margins(0.08)
  plt.axis("off")
  # plt.tight_layout()
  img = Image.new("RGB", (100, 100), color=(255, 255, 255))
  img.save("static/images/graph.png")
  plt.savefig('static/images/graph.png')
  plt.show()
  # m_list = matrix.tolist()
  print("E matrix is: ")
  print(E)
  return render_template('form2.html')

@app.route('/nodesL',methods=['POST'])
def my_form_post2():
  file_path="static/images/graph.png"
  if file_exists(file_path):
      os.remove(file_path)
  print("my_form_post2 started")
  global E
  # matrix = sym.Matrix()
  t=[]
  numOfNodes=request.form['nodes']
  print("Number of nodes+1 are: ")
  numOfNodes=int(numOfNodes)
  string2 = request.form['cncpt']
  cncpt = string2.split()
  for i in range(numOfNodes):
    string = request.form['field_{}'.format(i+1)]
    temp = string.split() 
    # temp=[eval(i) for i in temp] #to int
    # v1=np.array(temp)
    # print("Numpy array is: ")
    # print(v1)
    t.append(temp)
    # print(t)
    matrix = sym.Matrix()

  row=[]
  for i in range(numOfNodes) : 
    string = ""
    for j in range(numOfNodes) : 
      if i == j : 
        element = 0 

      else : 
        term = t[i][j]
        if term == '-VH' : 
          element = round(random.uniform(-5,-2.75), 2)
        elif term == '-H' : 
          element = round(random.uniform(-5,-1.5), 2)
        elif term == '-M' : 
          element = round(random.uniform(-2,-1.25), 2)
        elif term == '-L' : 
          element = round(random.uniform(-2.5,-1), 2) 
        elif term == '-VL' : 
          element = round(random.uniform(-1.5,0), 2) 
        elif term == 'NC' : 
          #element = round(random.uniform(-0.001 , 0.001) , 2)
          #if element == -0.0 : 
          element = 0 
        
        elif term == '+VH' : 
          element = round(random.uniform(2.75, 5), 2)
        elif term == '+H' : 
          element = round(random.uniform(1.5, 5), 2)
        elif term == '+M' : 
          element = round(random.uniform(1.25, 2), 2)
        elif term == '+L' : 
          element = round(random.uniform(1, 2.5), 2) 
        elif term == '+VL' : 
          element = round(random.uniform(0, 1.5), 2) 
        
        elif term == 'NaN' : 
          element = 'J'

      string += str(element)
      string += " "
      #print(element)

    temp = string.split(" ")
    temp.pop()
    for i in temp:
      if i=='J':
        continue
      i=float(i)
    row.append(temp)
    # print(row)
  
  for i in range(0,numOfNodes):
    matrix = matrix.row_insert(i,Matrix([row[i]]))
  matrix=matrix.applyfunc(lambda x: round(x, 2) if x.is_number else x)
  E=matrix
  print("Final: ")
  print(matrix) 
  G = nx.DiGraph()
  i=0
  j=0
  while(i<shape(E)[0]):
    j=1
    while(j<shape(E)[0]):
    #  print(int(M.row(i)[j]))
      if(E.row(i)[j]!=0):
        if(E.row(i)[j]!=J):
          G.add_edge(i+1,j+1,weight=int(E.row(i)[j]))
        else:
          G.add_edge(i+1,j+1,weight=-100)
        

      j+=1
    i+=1
  elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] !=-100]
  esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == -100]
    # Create a dictionary that maps node names to random colors
  node_colors = {}
  for node in G.nodes():
    # Generate a random color with light shade
    # Generate a random color with light shade
    hue = random.uniform(0,1)
    # saturation = random.uniform(0.5, 1.0)
    saturation = 0.8

    lightness = random.uniform(0.7, 1.0)
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    node_colors[node] = hex_color
  pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility

  # nodes
  nx.draw_networkx_nodes(G, pos, node_size=1000,node_color=list(node_colors.values()))

  # edges
  nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
  nx.draw_networkx_edges(
      G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="r", style="dashed"
  )


  # node labels
  mapping = dict(zip(G.nodes(), cncpt))
  nx.draw_networkx_labels(G, pos,labels=None, font_size=20, font_family="sans-serif")
  
  # edge weight labels
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels)


  # Create the legend
  legend_handles = []
  i=0
  for node, color in node_colors.items():
    legend_handles.append(plt.Line2D([], [], linewidth=0, marker='o', markersize=10, markerfacecolor=color, label=cncpt[i]))
    i=i+1
  # plt.legend(handles=legend_handles, loc='lower center', ncol=len(node_colors))
  plt.legend(handles=legend_handles, bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure)

  ax = plt.gca()
  ax.margins(0.08)
  plt.axis("off")
  # plt.tight_layout()
  img = Image.new("RGB", (100, 100), color=(255, 255, 255))
  img.save("static/images/graph.png")
  plt.savefig('static/images/graph.png')
  plt.show()
  print("E matrix is: ")
  print(E)



  # Inserting each row into matrix.
  # for i in range(0,numOfNodes):
  #   matrix = matrix.row_insert(i,Matrix([row[i]]))
  # print("The matrix is: ")
  # print(matrix)
  # matrix = sym.Matrix()
  
 

  # n = int(input("Enter number of nodes : "))
  # matrix=sym.Matrix()
  # print(matrix)
  return render_template('/form2.html')

@app.route('/tval', methods=['POST'])
def  generate_result():
  global E
  print("gen result function is running")
  print(E)
  threshold_value=int(request.form['tval'])
  state=int(request.form['state'])
  # print(threshold_value)
  if state > (np.shape(E)[1]) or state < 0: 
    # print("INVALID STATE ENTERED")
    return render_template('errorpage.html')

  elif state != 0:
    table2={}
    res = iteration(E , state, threshold_value )
    if(res[1]==0):
        table2["Fix-point: "]=res[0][-1]
        print("Fix-point:")
        print(res[0][-1])
    else:
        table2["Limit Cycle: "]=res[0]
        print("Limit Cycle")
        print(res[0])
    # table.append(res[0])
     # create an empty list to store the rows of the matrix
    rows2 = []
    
    # iterate over each row of the matrix
    for i in range(E.shape[0]):
        # create an empty list to store the elements of the row
        row2 = []
        
        # iterate over each element in the row
        for j in range(E.shape[1]):
            # add the element to the row list
            row2.append(str(E[i, j]))
        
        # add the row list to the rows list
        rows2.append(row2)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph.png')
    
    # table2.append(res[0])
    print("table2")
    print(table2)
    data=[rows2,full_filename,table2]
    return render_template('single_state_result.html',result=data)

  else :
    table = {}

    for x in range(1 , (np.shape(E)[1]) +1 ) : 
      cnt=0
      print("FOR ACTIVE STATE " , x)
      res = iteration(E , x, threshold_value )
      # cnt=cnt+1
      if(res[1]==0):
        table["Fix-point{}: ".format(x)]=res[0][-1]
        print("Fix-point:")
        print(res[0][-1])
      else:
        table["Limit Cycle{}: ".format(x)]=res[0]
        print("Limit Cycle")
        print(res[0])

    print("FULL TABLE")
    print(table)
    
    print("E= ")
    print(E)
        # create an empty list to store the rows of the matrix
    rows = []
    
    # iterate over each row of the matrix
    for i in range(E.shape[0]):
        # create an empty list to store the elements of the row
        row = []
        
        # iterate over each element in the row
        for j in range(E.shape[1]):
            # add the element to the row list
            row.append(str(E[i, j]))
        
        # add the row list to the rows list
        rows.append(row)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'graph.png')
    result=[rows,table,full_filename]

    return render_template('multi_state_result.html',result=result)




"""
  Check whether a cycle of state vector exists or not.
 i.e To check whether we are getting a previously calculated 
 state vector or not after applying threholding operation. 

  Function Parameters:
    b : a 1D vector array
"""

def check_cycle(b,E,tval) : 
  res=[]
  for i in range(len(b) - 1) : 
    for j in range(i + 1 , len(b)) : 
      
      v1 = b[i]
      v2 = b[j]
  
      if(compare(v1 , v2) == True) :
        tv1=multiply(v1,E)
        tv1=thresholdAndUpdate(tv1,tval)
      
        if(tv1==v1):
          res=[True,0]
          return res # 0 means there is a fix point 
        else:
          res=[True,1]
          return res
        
  res=[False,0]     
  return res

"""
  Comparing whether two state vectors are equal or not.
  
  Function Parameters:
    x, y : a 1D vector array
"""
def compare(x , y) :

  res = True 
  
  for k in range(len(x)) : 
  
    if x[k] != y[k] : 
      return False

  return True

"""
  Converts the quadratic polynomial of indeterminate as linear polynomial
  
  Function Parameters:
  x: 1D vector array

"""
def updateIPower(x):
  for i in range((np.shape(x))[1]):

    if x[i].find(J**2):
      x[i]=x[i].xreplace({J**2:J})

  return x


"""
A function to accept user input for neutorsophic matrix.
This function accepts the matrix input in row-wise manner.

"""

def inputE():
#   global v23

  # Taking the number of nodes.
  n=int(input("Enter the number of nodes required:"))

  matrix = sym.Matrix()

  row=[]
  
  # Taking row-wise input for the matrix.
  for i in range (0,n):
    
    string = input("Enter elements (Space-Separated): ")
    temp = string.split() 
    print(type(temp))
    temp=[eval(i) for i in temp] #to int
    # v1=np.array(temp)
    # print(v1)
  
    # v2.append
    row.append(temp)
  
# Inserting each row into matrix.
  for i in range(0,n):
    matrix = matrix.row_insert(i,Matrix([row[i]]))

  return matrix

"""
  A function to take input from user in terms of linguistic terms
"""
def inputL() : 
  n = int(input("Enter number of nodes : "))
  
  matrix = sym.Matrix()

  row=[]
  for i in range(n) : 
    string = ""
    for j in range(n) : 
      if i == j : 
        element = 0 

      else : 
        term = input("Enter the linguistic term for the relation between C{} and C{} : ".format(i + 1 , j + 1))
        if term == '-VH' : 
          element = round(random.uniform(-5,-2.75), 2)
        elif term == '-H' : 
          element = round(random.uniform(-5,-1.5), 2)
        elif term == '-M' : 
          element = round(random.uniform(-2,-1.25), 2)
        elif term == '-L' : 
          element = round(random.uniform(-2.5,-1), 2) 
        elif term == '-VL' : 
          element = round(random.uniform(-1.5,0), 2) 
        elif term == 'NC' : 
          #element = round(random.uniform(-0.001 , 0.001) , 2)
          #if element == -0.0 : 
          element = 0 
        
        if term == '+VH' : 
          element = round(random.uniform(2.75, 5), 2)
        elif term == '+H' : 
          element = round(random.uniform(1.5, 5), 2)
        elif term == '+M' : 
          element = round(random.uniform(1.25, 2), 2)
        elif term == '+L' : 
          element = round(random.uniform(1, 2.5), 2) 
        elif term == '+VL' : 
          element = round(random.uniform(0, 1.5), 2) 
        
        elif term == 'NaN' : 
          element = 'J'

      string += str(element)
      string += ","
      #print(element)

    temp = string.split(",")
    temp.pop()
    for i in temp:
      if i=='J':
        continue
      i=float(i)
    row.append(temp)
    print(row)
  
  for i in range(0,n):
    matrix = matrix.row_insert(i,Matrix([row[i]]))

  return matrix
"""
 Multipying a vector with matrix

 Function Paramerts:
   a : a 1D vector 
   B : a 2D matrix 
"""
def multiply(a, B) : 
  result = a * B
  return result

"""
Performing the thresholding operation.

Function Parameters:
  X : a 1D vector
  threshold_value : an integer (threshold value to make necessary updation )
"""
def thresholdAndUpdate(X , threshold_value) :

  X=updateIPower(X)

  for i in range((np.shape(X)[1])):
     temp_expr=X[i].subs(J,0)

     if(temp_expr>=threshold_value):
      X[i]=X[i].subs(X[i],threshold_value)

     elif(temp_expr==0):

       if(X[i]!=0):
        X[i]=J

     else:
      X[i]=X[i].subs(X[i],0)

  X[0]=1
 
  return X

"""
Performing the iterative operations of multiplication & thresholding.

Function Parameters:
  E : Adjaency/Connection matrix
  threshold_value : an integer
  state : an integer denoting which state to activate
"""
def iteration(E , state, threshold_value = 1) : 

  # Creating a vector of zeroes of length equal to coloumn length of E
  c1 = np.zeros((np.shape(E)[1]))
  c1[state - 1] = 1

  # Creating an matrix object with row c1
  start = sym.Matrix(c1)

  flag = False
  start = start.T
  vectors = []
  fix_point=0
  while flag == False :
    y = multiply(start , E)
    
    # Performing the thresholding operation on output vector y
    y = thresholdAndUpdate(y , threshold_value)

    vectors.append(y)

    # Updating start vector to start with new state vector
    start = y

    # Checking for cycle among state vectors
    res = check_cycle(vectors,E,threshold_value)
    flag=res[0]
    fix_point=res[1]


  return [vectors,fix_point]

"""
  A function
"""
def startE() : 
  E=inputE()
  print("The E matrix is : ")
  print(E)
  G = nx.DiGraph()
  i=0
  j=0
  while(i<shape(E)[0]):
    j=1
    while(j<shape(E)[0]):
    #  print(int(M.row(i)[j]))
      if(E.row(i)[j]!=0):
        if(E.row(i)[j]!=J):
          G.add_edge(i+1,j+1,weight=int(E.row(i)[j]))
        else:
          G.add_edge(i+1,j+1,weight=-100)
        

      j+=1
    i+=1
  elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] !=-100]
  esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == -100]
  pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility

  # nodes
  nx.draw_networkx_nodes(G, pos, node_size=700)

  # edges
  nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
  nx.draw_networkx_edges(
      G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="r", style="dashed"
  )

  # node labels
  nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
  # edge weight labels
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels)

  ax = plt.gca()
  ax.margins(0.08)
  plt.axis("off")
  plt.tight_layout()
  plt.show()

  print("Enter the threshold value : ")
  threshold_value = int(input())

  print("Enter the state which is to be active (If an iteration of all states active one by one enter 0): ")
  state = int(input())

  if state > (np.shape(E)[1]) or state < 0: 
    print("INVALID STATE ENTERED")

  elif state != 0:
    res = iteration(E , state, threshold_value )
    if(res[1]==0):
      print("Fix-point:")
      print(res[0][-1])
    else:
      print("Limit Cycle:")
      print(res[0])

  else :
    table = []

    for x in range(1 , (np.shape(E)[1]) +1 ) : 
      print("FOR ACTIVE STATE " , x)
      res = iteration(E , x, threshold_value )
      if(res[1]==0):
        print("Fix-point:")
        print(res[0][-1])
      else:
        print("Limit Cycle")
        print(res[0])
      table.append(res[0])

    print("FULL TABLE")
    print(table)

"""
 A function 
"""
def startL():
  E=inputL()
  print("The E matrix is : ")
  print(E)
  G = nx.DiGraph()
  i=0
  j=0
  while(i<shape(E)[0]):
    j=1
    while(j<shape(E)[0]):
    #  print(int(M.row(i)[j]))
      if(E.row(i)[j]!=0):
        if(E.row(i)[j]!=J):
          G.add_edge(i+1,j+1,weight=int(E.row(i)[j]))
        else:
          G.add_edge(i+1,j+1,weight=-100)
        

      j+=1
    i+=1
  elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] !=-100]
  esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == -100]
  pos = nx.circular_layout(G)  # positions for all nodes - seed for reproducibility

  # nodes
  nx.draw_networkx_nodes(G, pos, node_size=700)

  # edges
  nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
  nx.draw_networkx_edges(
      G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="r", style="dashed"
  )

  # node labels
  nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
  # edge weight labels
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G, pos, edge_labels)

  ax = plt.gca()
  ax.margins(0.08)
  plt.axis("off")
  plt.tight_layout()
  plt.show()

  print("Enter the threshold value : ")
  threshold_value = int(input())

  print("Enter the state which is to be active (If an iteration of all states active one by one enter 0): ")
  state = int(input())

  if state > (np.shape(E)[1]) or state < 0: 
    print("INVALID STATE ENTERED")

  elif state != 0:
    res = iteration(E , state, threshold_value )
    if(res[1]==0):
      print("Fix-point:")
      print(res[0][-1])
    else:
      print("Limit Cycle:")
      print(res[0][0])


  else :
    table = []

    for x in range(1 , (np.shape(E)[1]) +1 ) : 
      print("FOR ACTIVE STATE " , x)
      res = iteration(E , x, threshold_value )
      if(res[1]==0):
        print("Fix-point:")
        print(res[0][-1])
      else:
        print("Limit Cycle")
        print(res[0][0])
      table.append(res[0])

    print("FULL TABLE")
    print(table)
if __name__ == '__main__':
    app.run()