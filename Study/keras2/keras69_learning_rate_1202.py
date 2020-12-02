weight = 0.5
input  = 0.5
goal_prediction = 0.8
lr = 0.1 # 0.0001 0.001 0.01 0.1 1
ee = 1
for iteration in range(1101):
    prediction = input * weight
    error      = (prediction - goal_prediction) ** 2
    if error < ee : 
        print(iteration+1,"\tError : ",str(error),"\tPrediction : ",str(prediction))
        ee = error
    print(iteration+1,"\tError : ",str(error),"\tPrediction : ",str(prediction))
    
    up_prediction   = input * (weight + lr)
    up_error        = (goal_prediction - up_prediction) ** 2

    down_prediction = input * (weight - lr)
    down_error      = (goal_prediction - down_prediction) ** 2

    if(down_error < up_error):
        weight = weight - lr
    if(down_error > up_error):
        weight = weight + lr
# 0.0001
# 1101    Error :  0.24502500000000604    Prediction :  0.30499999999999394

# 0.001
# 1101    Error :  1.0799505792475652e-27 Prediction :  0.7999999999999672

# 0.01
# 111     Error :  1.9721522630525295e-31 Prediction :  0.8000000000000005

# 0.1
# 12      Error :  1.232595164407831e-32  Prediction :  0.8000000000000002

# 1
# 2       Error :  0.0025000000000000044  Prediction :  0.75