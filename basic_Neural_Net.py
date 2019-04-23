# Basic Neural Network in (not so) pure Python


#Follow along tutorial by freecodecamp.org @ https://youtu.be/u7n9t1cBei8


# Problem definition:

# Combination of three lights --> ON(1)/OFF(0) states determines --> WALK!(1) /STOP!(0)


# Streetlights--------------------------------WALK/STOP
# 1    0    1                                   0 
# 0    1    1                                   1 
# 0    0    1                                   0 
# 1    1    1                                   1 
# 0    1    1                                   1 
# 1    0    1                                   0 


# middle light's ON state is correlated with WALK/STOP state
# expected O/P is that NN will detect this correlation
# Train the NN and test using a test input light combination



import numpy as np

np.random.seed(1)

weights = np.random.rand(3)  # random initial weights
alpha = 0.1                  # Gradient descent parameters prevents overshooting



streetlights = np.array ([[0,0,1],
                        [0,1,1],
                        [0,0,1],
                        [1,1,1],
                        [0,1,1],
                        [1,0,1]])

walks_vs_stop = np.array([0,1,0,1,1,0])

#walks_vs_stop = np.array([1,0,1,0,1,1])

#NN function
def neural_network(input,weight):
  prediction = np.dot(input,weight)
  return (prediction)

# I worked here directly with matrix algebra,
# in tutorial nested FOR loops have been used to work with individual elements 
# of vectors
  
for i in range(40):
  error_for_all_lights = 0
      
  prediction = neural_network(streetlights,weights)
  
  ms_error = (prediction - walks_vs_stop)**2 # mean squared error -to keep things simple
  
  error_for_all_lights = np.sum(ms_error)
  
  delta = prediction-walks_vs_stop
  
  streetlights_T = np.transpose(streetlights)
  
  weights = weights - (alpha*(np.dot(streetlights_T,delta))) # update weights GD

 # print data for detailed insights   
    
 # print (f"Prediction: {prediction}") 
print (f"Weights: {weights}")
 # print (f"Errors: {ms_error}")   


test = np.array([1,1,0]) # test input

pred_new = np.rint((neural_network(test,weights))) # predict on test data and round to int

print (f"Test Input: {test}")

if pred_new >= 1:
        
    print (f"Test Output: WALK!!")

else:
    print(f"Test Output: STOP!!")
