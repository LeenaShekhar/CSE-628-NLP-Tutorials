""" Variable examples
Created by Chip Huyen (huyenn@stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 02
modified by: Leena Shekhar
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


### Example 1: creating and initializing variables ###

s = tf.Variable(2, name='scalar_s') 
m = tf.Variable([[0, 1], [2, 3]], name='matrix') 
W = tf.Variable(tf.zeros([10, 1]), name='big_matrix')
V = tf.Variable(tf.truncated_normal([4, 4]), name='normal_matrix')

#gets an existing variable or create a new one. The initializer can be a Tensor, variable is initialized to this value and shape.
s = tf.get_variable('scalar_s', initializer=tf.constant(2)) 
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(10, 1), initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(4, 4), initializer=tf.truncated_normal_initializer())

#create a session
with tf.Session() as sess:
    #initialize the variables
    sess.run(tf.global_variables_initializer())
    #evaluate
    print("\nExample 1 output")
    print(V.eval())
sess.close()

### Example 2: assigning values to variables ###
               
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    #iitialize the variable
    sess.run(W.initializer)
    print("Initial value: %d"%sess.run(W))                  # >> 10
    sess.run(assign_op)
    print("Final value: %d"%W.eval())                     	# >> 100
sess.close()

# create a variable whose original value is 2
a = tf.get_variable('scalar_a', initializer=tf.constant(2)) 
a_times_two = a.assign(a * 2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    print("First multiplication: %d"%sess.run(a_times_two) )  # >> 4                 	
    print("Second multiplication: %d"%sess.run(a_times_two))  # >> 8                	
    print("Third multiplication: %d"%sess.run(a_times_two))   # >> 16
sess.close()

W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print("Add operation: %d"%sess.run(W.assign_add(10)))     	# >> 20
    print("Subtract operation: %d"%sess.run(W.assign_sub(2)))   # >> 18
sess.close()

### Example 3: Each session has its own copy of a variable ###

W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print("Session1 add operation: %d"%sess1.run(W.assign_add(10)))        	# >> 20
print("Session2 sub operation: %d"%sess2.run(W.assign_sub(2)))          # >> 8
print("Session1 add operation: %d"%sess1.run(W.assign_add(100)))        # >> 120
print("Session2 sub operation: %d"%sess2.run(W.assign_sub(50)))         # >> -42
sess1.close()
sess2.close()


### Example 4: create a variable with the initial value depending on another variable ###

W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(W * 2)


### Example 5: Simple ways to create log file writer ###

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graphs/simple', tf.get_default_graph()) 
with tf.Session() as sess:
    print("Add operation: %d"%sess.run(x))
writer.close() # close the writer 

            
### Example 6: The wonderful wizard of div ###

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
    print("First div")
    print(sess.run(tf.div(b, a)))
    print("Second div")
    print(sess.run(tf.divide(b, a)))

### Example 7: multiplying tensors ###

a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')

with tf.Session() as sess:
    print("Multiplication")
    print(sess.run(tf.multiply(a, b)))