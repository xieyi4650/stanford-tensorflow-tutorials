"""
Simple exercises to get used to TensorFlow API
You should thoroughly test your code.
TensorFlow's official documentation should be your best friend here
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

sess = tf.InteractiveSession()
###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: x + y, lambda: x - y)
print("outa:")
print(sess.run(out))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

# YOUR CODE
xb = tf.random_uniform([], minval=-1, maxval=1)  # Empty array as shape creates a scalar.
yb = tf.random_uniform([], minval=-1, maxval=1)
outb = tf.case([(tf.greater(xb, yb), lambda: xb - yb), (tf.less(xb, yb), lambda: xb + yb)], default=lambda:0.)
print("outb:")
print(sess.run(outb))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# YOUR CODE
xc = tf.constant([[0, -2, -1], [0, 1, 2]])
yc = tf.zeros_like(xc)
outc = tf.equal(xc, yc)
print("outc:")
print(sess.run(outb))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

# YOUR CODE
xd = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
                  30.97266006,  26.67541885,  38.08450317,  20.74983215,
                  34.94445419,  34.45999146,  29.06485367,  36.01657104,
                  27.88236427,  20.56035233,  30.20379066,  29.51215172,
                  33.71149445,  28.59134293,  36.05556488,  28.66994858])
outd1 = tf.where(tf.greater(xd, tf.constant(30.)))
print("outd1:")
print(sess.run(outd1))

outd2 = tf.gather(xd, tf.reshape(outd1, [-1]))
print("outd2:")
print(sess.run(outd2))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
xe = tf.range(1, 7)
oute = tf.diag(xe)
print("oute:")
print(sess.run(oute))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

# YOUR CODE
xf = tf.random_uniform([10, 10])
outf = tf.matrix_determinant(xf)
print("outf:")
print(sess.run(outf))

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

# YOUR CODE
xg = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
outg, idx_g = tf.unique(xg)
print("outg:")
print(sess.run(outg))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

# YOUR CODE
xh = tf.random_uniform([300])
yh = tf.random_uniform([300])
average = tf.reduce_mean(xh - yh)
def f1(): return tf.reduce_mean(tf.square(xh - yh))
def f2(): return tf.reduce_mean(tf.abs(xh - yh))
outh = tf.cond(average < 0, f1, f2)
print("outh:")
print(sess.run(outh))
