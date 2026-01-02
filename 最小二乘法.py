import numpy as np
import matplotlib.pyplot as plt
data = np.array([[32,31],[53,68],[61,62],[47,71],[59,87],[55,78],[52,79],[39,59],[48,75],[52,71],
          [45,55],[54,82],[44,62],[58,75],[56,81],[48,60],[44,82],[60,97],[45, 48],[38,56],
          [66,83],[65,118],[47,57],[41,51],[51,75],[59,74],[57,95],[63,95],[46,79],[50,83]])
x=data[:,0]
y=data[:,1]
#plt.scatter(x,y)
#plt.show()
def compute_cost(w,b,points):
    total_cost=0;
    l=len(points)
    for i in range(l):
        x=points[i,0]
        y=points[i,1]
        total_cost+=(x*w+b-y)**2
    return total_cost/l
def average(data):
    sum=0
    num=len(data)
    for i in range(num):
          sum+=data[i]
    return sum/num
def fit(points):
    M=len(points)
    x_bar=average(points[:,0])
    sum_yx=0
    sum_x2=0
    sum_delta=0
    sum_x=0
    sum_y=0

    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        sum_yx+=y*(x-x_bar)
        sum_x2+=x**2
    w=sum_yx/(sum_x2-M*(x_bar**2))
    for i in range(M):
        x=points[i,0]
        y=points[i,1]
        sum_delta+=y-w*x
    b=sum_delta/M
    return w,b
w,b=fit(data)
plt.scatter(x,y)
pred_y=w*x+b
plt.plot(x,pred_y,c="r")
plt.show()