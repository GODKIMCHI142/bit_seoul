# 이 소스를 분석하시오


# 넘파이를 선언하기 위해 임포트함
import numpy as np 


# 데이터 준비
dataset = np.array(range(1,11)) # 1~10 
size = 5                        # 원하는 열의 길이
print("dataset : >>> \n",dataset) # [ 1  2  3  4  5  6  7  8  9 10]
print("len(dataset) : >>> \n",len(dataset)) # 10

def split_x(seq, size):                        # 함수선언
    aaa = []                                   # 빈 리스트를 만듬
    for i in range(len(seq) - size +1):        # for문이 도는 횟수는 dataset의 길이 - 열의 길이 + 1
        subset = seq[i : (i+size)]             # 인덱스 i ~ i+5까지 리스트로 뽑고 1만큼 순차적으로 증가한다
        aaa.append([item for item in subset])  # subset안에서 가져온 데이터를 aaa에 넣는다
    print("type(aaa) : >>> \n",type(aaa))      # aaa의 타입을 출력해본다.                  
    return np.array(aaa)                       # aaa를 numpy.ndarray 타입으로 바꾼다.


datasets = split_x(dataset, size) # 함수를 호출하고 dataset과 size를 넘겨준뒤 새로운 리스트를 리턴받는다.
print("=================")
print("datasets : >>>> \n",datasets)
print("type(datasets) : >>>> \n",type(datasets))




# x_len = x.__len__()
# print("x.__len__() : ",x.__len__())
# new_x = []
# np.array(new_x)
# for i in range(x[1,].size):
#     # print("s",x[1,].size)
#     # new_x += [x[0,i],x[1,i],x[2,i]]
#     # on = np.array(x[0,i])
#     # tw = np.array(x[1,i])
#     # th = np.array(x[2,i])
#     # new_x = np.append(new_x,[on],axis=0)
#     # new_x = np.append(new_x,[tw],axis=0)
#     # new_x = np.append(new_x,[th],axis=0)
#     new_x.append([x[0,i],x[1,i],x[2,i]])
# new_x = np.array(new_x)
# print(new_x)
# print("new_x.shape >>> ",new_x.shape)

