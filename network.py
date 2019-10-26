from keras.layers import GaussianNoise, Dense, Activation
from sklearn.cluster import KMeans
from loss import *
import os


class autoencoder(object):
    def __init__(self, dataname, distribution, self_training, dims, cluster_num, t_alpha, alpha, gamma, learning_rate, noise_sd=1.5, init='glorot_uniform', act='relu'):
        self.dataname = dataname
        self.distribution = distribution
        self.self_training = self_training
        self.dims = dims
        self.cluster_num = cluster_num
        self.t_alpha = t_alpha
        self.alpha = alpha
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act

        self.n_stacks = len(self.dims) - 1

        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))

        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        self.h = self.x
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)

        for i in range(self.n_stacks - 1):
            self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
            self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
            self.h = Activation(self.act)(self.h)

        self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)

        self.num, self.latent_p = cal_latent(self.latent, self.t_alpha)

        self.latent_q = target_dis(self.latent_p)
        self.latent_p = self.latent_p + tf.linalg.diag(tf.linalg.diag_part(self.num))
        self.latent_q = self.latent_q + tf.linalg.diag(tf.linalg.diag_part(self.num))

        self.latent_dist1, self.latent_dist2 = cal_dist(self.latent, self.clusters)

        self.h = self.latent
        for i in range(self.n_stacks - 1, 0, -1):
            self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(self.h)

        if self.distribution == "ZINB":
            self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(
                self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
        elif self.distribution == "NB":
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.likelihood_loss = NB(self.disp, self.x_count, self.output, mask=False, debug=False, mean=True)

        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))

        if self_training:
            self.cross_entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_p))
            self.entropy = -tf.reduce_sum(self.latent_q * tf.log(self.latent_q))
            self.kl_loss = self.cross_entropy - self.entropy
            self.total_loss = self.likelihood_loss + self.alpha * self.kmeans_loss + self.gamma * self.kl_loss
        else:
            self.total_loss = self.likelihood_loss + self.alpha * self.kmeans_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.likelihood_loss)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def pretrain(self, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("begin the pretraining")
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        self.sess = tf.Session(config=config_)
        self.sess.run(init)

        self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        pre_index = 0
        for ite in range(pretrain_epoch):
            while True:
                if (pre_index + 1) * batch_size > X.shape[0]:
                    last_index = np.array(list(range(pre_index * batch_size, X.shape[0])) + list(
                        range((pre_index + 1) * batch_size - X.shape[0])))
                    _, likelihood_loss, latent = self.sess.run([self.pretrain_op, self.likelihood_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[last_index],
                            self.x: X[last_index],
                            self.x_count: count_X[last_index]})
                    self.latent_repre[last_index] = latent
                    pre_index = 0
                    break
                else:
                    _, likelihood_loss, latent = self.sess.run(
                        [self.pretrain_op, self.likelihood_loss, self.latent],
                        feed_dict={
                            self.sf_layer: size_factor[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x: X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)],
                            self.x_count: count_X[(pre_index * batch_size):(
                                    (pre_index + 1) * batch_size)]})
                    self.latent_repre[(pre_index * batch_size):((pre_index + 1) * batch_size)] = latent
                    pre_index += 1


    def funetrain(self, X, count_X, size_factor, batch_size, funetrain_epoch, update_epoch, error):
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++")
        self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.latent_repre)
        self.last_pred = np.copy(self.kmeans_pred)
        self.sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))
        print("begin the funetraining")

        fune_index = 0
        for i in range(1, funetrain_epoch + 1):
            if i % update_epoch == 0:
                dist, likelihood_loss, kmeans_loss = self.sess.run(
                    [self.latent_dist1, self.likelihood_loss, self.kmeans_loss],
                    feed_dict={
                        self.sf_layer: size_factor,
                        self.x: X,
                        self.x_count: X})
                self.Y_pred = np.argmin(dist, axis=1)
                if np.sum(self.Y_pred != self.last_pred) / len(self.last_pred) < error:
                    break
                else:
                    self.last_pred = self.Y_pred
            else:
                while True:
                    if (fune_index + 1) * batch_size > X.shape[0]:
                        last_index = np.array(list(range(fune_index * batch_size, X.shape[0])) + list(
                            range((fune_index + 1) * batch_size - X.shape[0])))
                        _, likelihood_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.likelihood_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[last_index],
                                self.x: X[last_index],
                                self.x_count: count_X[last_index]})
                        fune_index = 0
                        break
                    else:
                        _, likelihood_loss, Kmeans_loss = self.sess.run(
                            [self.train_op, self.likelihood_loss, self.kmeans_loss],
                            feed_dict={
                                self.sf_layer: size_factor[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x: X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)],
                                self.x_count: count_X[(fune_index * batch_size):(
                                        (fune_index + 1) * batch_size)]})
                        fune_index += 1
        self.sess.close()
        return self.Y_pred
