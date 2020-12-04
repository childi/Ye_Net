import tensorflow as tf
w1 = tf.Variable(3.0)
w2 = tf.Variable(4.0)
y_hat = tf.Variable(1.0)

gen = tf.multiply(0.1, w1)
gen_stopped = tf.stop_gradient(gen)
alpha = tf.multiply(0.2, w2)
alpha = tf.add(alpha, tf.multiply(0.3, w1))
y = tf.add(gen_stopped, alpha)
loss = tf.multiply(tf.log(y), y_hat)

w1_gradients = tf.gradients(loss, w1)
w2_gradients = tf.gradients(loss, w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(w1_gradients))
    print(sess.run(w2_gradients))
