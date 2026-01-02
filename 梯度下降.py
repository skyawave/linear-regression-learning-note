import numpy as np
import matplotlib.pyplot as plt
data = np.array([[34, 58], [42, 65], [55, 72], [49, 81], [61, 89], [38, 63], [52, 77], [46, 71], [57, 84], [43, 68],
[50, 73], [59, 92], [47, 69], [53, 79], [41, 62], [56, 87], [44, 67], [62, 95], [39, 61], [51, 76],
[48, 74], [54, 83], [58, 91], [45, 70], [63, 98], [40, 64], [49, 78], [55, 85], [42, 66], [60, 94]])     #数据集
x=data[:,0]
y=data[:,1]      #分别读取两列的数据
def comput_cost(w,b,data):
    cost=0
    l=len(data)
    for i in range(l):
        x=data[i,0]
        y=data[i,1]
        cost+=(w*x+b-y)**2
    average_cost=cost/l       #计算成本
    return average_cost
alpha=0.0001
initial_w=0
initial_b=0
num_iter=15             #初始化w,b,学习率，计算次数
def grad_desc(data,initial_w,initial_b,alpha,num_iter):
    w=initial_w
    b=initial_b
    cost_list=[]
    for i in range(num_iter):
        cost_list.append(comput_cost(w,b,data))
        w,b=step_grad_desc(w,b,alpha,data)
    return [w,b,cost_list]
def step_grad_desc(current_w,current_b,alpha,data):
    sum_grad_w=0
    sum_grad_b=0
    l=len(data)
    for i in range(l):
        x=data[i,0]
        y=data[i,1]
        sum_grad_w+=(current_w*x+current_b-y)*x
        sum_grad_b +=current_w * x + current_b - y       #偏导数
    grad_w=2/l*sum_grad_w
    grad_b=2/l*sum_grad_b
    update_w=current_w-alpha*grad_w
    update_b=current_b-alpha*grad_b              #分别计算新的w,b
    return update_w,update_b                       #返回w,b
w,b,cost_list=grad_desc(data,initial_w, initial_b, alpha, num_iter)
cost=comput_cost(w,b,data)
print(cost_list)
plt.plot(cost_list)
plt.show()
plt.scatter(x,y)              #绘制散点图
pred_y=w*x+b
plt.plot(x,pred_y,c='r')  #绘制图像
plt.show()

