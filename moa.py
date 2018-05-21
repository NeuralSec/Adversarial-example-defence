import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess

class MOA:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = -0.5, boxmax = 0.5,
                 image_size, num_channels, num_labels):

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus
        
        # prediction BEFORE-SOFTMAX of the model
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = model.predict(self.newimg)
        self.output = mrcnn_class
        self.boxposition = mrcnn_bbox
        self.mask = mrcnn_mask
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-(tf.tanh(self.timg) * self.boxmul + self.boxplus)),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)