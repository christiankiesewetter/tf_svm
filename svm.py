import tensorflow as tf
from tensorflow.keras.models import Model

class SVMModel(Model):

    def rbf_kernel(self, X):
        y = (X - self.means) / (2 * self.sigma**2)
        return y


    def linear_kernel(self, X):
        y_hat = self.slope * X + self.intercept
        return y_hat


    def init_rbf_kernel(self):
        init_mu = tf.random.uniform(shape = (1, self.dimensions), minval = 0., maxval = 1, dtype=tf.float32)
        self.means = tf.Variable(init_mu, trainable = True, name = 'rbf_mu')

        init_sigma = tf.random.uniform(shape = (1, self.dimensions), minval = 0., maxval = 0.5, dtype=tf.float32)
        self.sigma = tf.Variable(init_sigma, trainable = True, name = 'rbf_sigma')


    def init_linear_kernel(self):
        init_intercept = tf.random.normal(shape = (1, self.dimensions), mean=0.1, stddev=0.5, dtype=tf.float32)
        self.intercept = tf.Variable(init_intercept, trainable=True)

        init_slope = tf.random.normal(shape = (1, self.dimensions), mean=-0.1, stddev=0.5, dtype=tf.float32)
        self.slope = tf.Variable(init_slope, trainable=True)


    def hinge(self, y_hat):
        return tf.maximum(y_hat + 1, 0)


    def h(self, X):
        if self.use_rbf:
            X = self.rbf_kernel(X)
        y_hat = self.linear_kernel(X)
        y_hat = tf.expand_dims(tf.reduce_sum(y_hat, axis = 1), axis=-1)
        #y_hat = tf.maximum(y_hat, 0)
        return y_hat


    def svm_loss(self, y, y_hat):
        loss = self.C * (
                (0.5 * (1 + y) * tf.reduce_sum(self.hinge(-y_hat)))
                + (0.5 * (1. - y) * tf.reduce_sum(self.hinge(y_hat)))) \
                + 0.5 * (tf.reduce_sum(self.slope**2) + tf.reduce_sum(self.intercept**2))
        return loss


    def __init__(self, dimensions, C=1, rbf = False, **kwargs):
        super(SVMModel,self).__init__()
        self.use_rbf = rbf
        self.dimensions = dimensions
        if self.use_rbf:
            self.init_rbf_kernel()
        self.init_linear_kernel()
        self.C = C


    def call(self, X):
        y_hat = self.h(X)
        return y_hat

    @tf.function
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_hat = self.call(X)
            loss = self.svm_loss(y, y_hat)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_hat)
        accuracy = 1. - tf.reduce_mean(tf.abs(y - tf.where(y_hat > 0, 1., -1.)))
        res = {
            'acuracy': accuracy,
            'loss':loss
        }
        return res
