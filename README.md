# MLwithCIFAR-10
Basic neural network trained to recognize images from the CIFAR-10 database

Libraries and Resources Used

-Uses member functions and functionality from the python PyTorch machine learning library.
-Uses the CIFAR-10 dataset of 60,000 32x32 color images in 10 different classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks

File Structure

-trainnet.py : 
  The training file for the neural network. Uses a subset of the whole CIFAR-10 dataset and uses the Stochastic Gradient Descent loss function to calculate the cost and the gradient for each set of nodes. 
  
  Then, the net backpropagates using the pytorch backward() function and the gradients calculated from the SGD function to adjust the buffers. 
  
  The training file runs through the training subset overall three times and afterward saves the state of the net as a whole as a file which is accessed by the training function.

-testnnet.py:
  This file loads in the PATH file created by the training file and then tests the neural network on the entire dataset. It then prints out the overall accuracy of the net by comparing its guesses with the ground truth.
  Additional output breaks down the accuracy of the neural net in recognizing images from each of the individual classes.
  
