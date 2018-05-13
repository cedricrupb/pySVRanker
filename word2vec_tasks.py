import numpy as np
import tensorflow as tf
from pyTasks.task import Task, Parameter
from pyTasks.task import Optional, containerHash
from pyTasks.target import CachedTarget, LocalTarget
from pyTasks.target import JsonService, FileTarget
from .gram_tasks import PrepareKernelTask
import logging
import math
from time import time
import os
from tensorflow.contrib.tensorboard.plugins import projector
from scipy.spatial.distance import cdist
from .graph_tasks import EdgeType


class WVSkipgram(object):

    def __init__(self, num_words, learning_rate, embedding_size,
                 num_steps, neg_sampling, unigrams, log="./log/"):
        self.num_words = num_words
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.num_steps = num_steps
        self.neg_sampling = neg_sampling
        self.unigrams = unigrams
        self.log_dir = log
        self.graph, self.batch_inputs, self.batch_labels,self.normalized_embeddings,\
        self.loss, self.optimizer = self.trainer_initial()

    def trainer_initial(self):
        graph = tf.Graph()
        with graph.as_default():

            # logging
            self.logger = tf.summary.FileWriter(self.log_dir)

            with tf.name_scope("embedding"):
                batch_inputs = tf.placeholder(tf.int64, shape=([None, ]))
                batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))

                graph_embeddings = tf.Variable(
                        tf.random_uniform([self.num_words, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size),
                        name='word_embedding')

                batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, batch_inputs) #hiddeb layer

                weights = tf.Variable(tf.truncated_normal([self.num_words, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size))) #output layer wt
                biases = tf.Variable(tf.zeros(self.num_words)) #output layer biases

                #negative sampling part
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=weights,
                                   biases=biases,
                                   labels=batch_labels,
                                   inputs=batch_graph_embeddings,
                                   num_sampled=self.neg_sampling,
                                   num_classes=self.num_words,
                                   sampled_values=tf.nn.fixed_unigram_candidate_sampler(
                                       true_classes=batch_labels,
                                       num_true=1,
                                       num_sampled=self.neg_sampling,
                                       unique=True,
                                       range_max=self.num_words,
                                       distortion=0.75,
                                       unigrams=self.unigrams)#word_id_freq_map_as_list is the
                                   # frequency of each word in vocabulary
                                   ))
                norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
                normalized_embeddings = graph_embeddings/norm

                # summary
                tf.summary.histogram("weights", weights)
                tf.summary.histogram("biases", biases)
                tf.summary.scalar("loss", loss)

                config = projector.ProjectorConfig()
                emb = config.embeddings.add()
                emb.tensor_name = normalized_embeddings.name
                emb.metadata_path = os.path.join(self.log_dir, 'vocab.tsv')
                projector.visualize_embeddings(self.logger, config)

            with tf.name_scope('descent'):
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                           global_step, 100000, 0.96, staircase=True) #linear decay over time

                learning_rate = tf.maximum(learning_rate,0.001) #cannot go below 0.001 to ensure at least a minimal learning

                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

                self.logger.add_graph(graph)
        return graph, batch_inputs, batch_labels, normalized_embeddings, loss, optimizer

    def train(self, dataset):
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)) as sess:

            merged_summary = tf.summary.merge_all()
            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess.run(init)

            sess.run(tf.tables_initializer())

            step = 0

            for i in range(self.num_steps):
                t0 = time()

                feed_it = dataset.make_initializable_iterator()
                next_element = feed_it.get_next()

                sess.run(feed_it.initializer)
                while True:
                    try:
                        feed_dict = sess.run([next_element])
                        feed_dict = {self.batch_inputs: feed_dict[0][0],
                                     self.batch_labels:
                                     sess.run(
                                        tf.reshape(feed_dict[0][1], [-1, 1])
                                        )
                                     }
                        loss_val = 0
                        _,  loss_val = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                        if step % 10 == 0:
                            s = sess.run(merged_summary, feed_dict=feed_dict)
                            self.logger.add_summary(s, step)

                        if step % 1000 == 0:
                            saver.save(sess, os.path.join(self.log_dir, "model.ckpt"), step)

                        step += 1
                    except tf.errors.OutOfRangeError:
                        break

                epoch_time = time() - t0
                loss = 0

            #done with training
            final_embeddings = self.normalized_embeddings.eval()
        return final_embeddings


def collect_ast(G, nodes):
    stack = []
    stack.extend(nodes)

    out = []

    while len(stack) > 0:
        act = stack.pop()
        out.append(act)

        for in_node, _, _, d in G.in_edges(act, keys=True, data='type'):
            if d is EdgeType.se:
                stack.append(in_node)

    return out


def is_ast_node(G, node):
    ast_node = True

    for out_node, _, _, d in G.out_edges(node, keys=True, data='type'):
        ast_node &= d is EdgeType.se

    for out_node, _, _, d in G.in_edges(node, keys=True, data='type'):
        ast_node &= d is EdgeType.se

    return ast_node


class WVGraphSentenceTask(Task):
    out_dir = Parameter('./w2v/sentences/')

    def __init__(self, name, h, D):
        self.name = name
        self.h = h
        self.D = D

    def require(self):
        return PrepareKernelTask(self.name, self.h, self.D)

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.txt'
        return FileTarget(path)

    def __taskid__(self):
        return 'W2VGraphSentence_%s_%d_%d' % (self.name, self.h, self.D)

    def run(self):
        with self.input()[0] as i:
            G = i.query()

        L = []

        with self.output() as output:
            for node in G:

                in_nodes = []
                ast_nodes = []

                for in_node, _, _, d in G.in_edges(node, keys=True, data='type'):
                    if d is EdgeType.se:
                        ast_nodes.append(in_node)
                    elif d is EdgeType.de:
                        in_nodes.append(in_node)

                in_nodes.extend(collect_ast(G, ast_nodes))

                if len(in_nodes) == 0:
                    continue

                in_nodes = [G.node[n]['label'] for n in in_nodes]

                output.write(
                    str(G.node[node]['label']) + ' ' + ' '.join(in_nodes)+'\n'
                )


class WVVocabulary(Task):
    out_dir = Parameter('./w2v/')

    def __init__(self, graph_list, length, h, D):
        self.graph_list = graph_list
        self.h = h
        self.D = D
        self.length = length

    def require(self):
        return [
            WVGraphSentenceTask(
                name,
                self.h,
                self.D
            )
            for name in self.graph_list
        ]

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'W2VVocabulary_%d_%d_%d' % (self.h, self.D,
                                           containerHash(self.graph_list))

    def run(self):
        vocab = {}
        overall = 0
        for inp in self.input():
            with inp as i:
                for line in i.readlines():
                    for w in line.split():
                        if w not in vocab:
                            vocab[w] = 0
                        vocab[w] += 1
                    overall += 1
        vocab = [x for x in sorted(
                    list(vocab.items()), key=lambda x: x[1], reverse=True
                 )][:self.length]
        vocab = {k[0]: (v, k[1]) for v, k in enumerate(vocab)}

        print('### Parsed %s samples ###' % overall)

        with self.output() as o:
            o.emit(vocab)


class WVEmbeddingTask(Task):
    out_dir = Parameter('./w2v/')
    embedding_size = Parameter(10)
    learning_rate = Parameter(0.001)
    num_steps = Parameter(3)
    neg_sampling = Parameter(15)
    batch_size = Parameter(100)
    log_dir = Parameter('./log/embedded/')

    def __init__(self, graph_list, length, h, D):
        self.graph_list = graph_list
        self.h = h
        self.D = D
        self.length = length

    def require(self):
        out = [WVVocabulary(self.graph_list, self.length, self.h, self.D)]
        out.extend([
            WVGraphSentenceTask(
                name,
                self.h,
                self.D
            )
            for name in self.graph_list
        ])
        return out

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'W2VEmbeddingTask_%d_%d_%d' % (self.h, self.D,
                                           containerHash(self.graph_list))

    def _get_vocab(self, vocab):
        vocab = [x[0] for x in
                 sorted(list(vocab.items()),
                        key=lambda v: v[1][0])]

        with open(os.path.join(self.log_dir.value, 'vocab.tsv'), 'w') as o:
            for v in vocab:
                o.write(v+'\n')

        return vocab

    def run(self):
        with self.input()[0] as i:
            vocab = i.query()
        inp = (self.input()[i] for i in range(1, len(self.input())))
        filenames = [f.sandBox + f.path for f in inp]

        unigrams = [x[1][1] for x in
                    sorted(list(vocab.items()),
                    key=lambda v: v[1][0])]

        model_skipgram = WVSkipgram(
            len(vocab),
            self.learning_rate.value,
            self.embedding_size.value,
            self.num_steps.value,
            self.neg_sampling.value,
            unigrams,
            self.log_dir.value
        )

        with tf.Session(graph=model_skipgram.graph,
                        config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)) as sess:

            vocab_mapping = tf.constant(self._get_vocab(vocab))
            table = tf.contrib.lookup.index_table_from_tensor(
                                        mapping=vocab_mapping, num_oov_buckets=1,
                                        default_value=-1)

            def parse_mapping(line):
                line = tf.string_split([line], ' ').values
                line = table.lookup(line)
                label = line[0:1]
                features = line[1:]
                return features, tf.tile(label, [tf.shape(features)[0]])

            dataset = tf.data.TextLineDataset(filenames)
            dataset = dataset.map(parse_mapping)
            dataset = dataset.flat_map(lambda features, labels:
                                        tf.data.Dataset().zip((
                                          tf.data.Dataset().from_tensor_slices(features),
                                          tf.data.Dataset().from_tensor_slices(labels))
                                        ))
            dataset = dataset.shuffle(1000).batch(self.batch_size.value)

        embedding = model_skipgram.train(dataset)

        with self.output() as o:
            o.emit(embedding.tolist())


class WVSimilarWords(Task):
    out_dir = Parameter('./w2v/')

    def __init__(self, graph_list, length, h, D):
        self.graph_list = graph_list
        self.h = h
        self.D = D
        self.length = length

    def require(self):
        out = [WVVocabulary(self.graph_list, self.length, self.h, self.D),
               WVEmbeddingTask(self.graph_list, self.length,
                               self.h, self.D)]

        return out

    def output(self):
        path = self.out_dir.value + self.__taskid__() + '.json'
        return CachedTarget(
            LocalTarget(path, service=JsonService)
        )

    def __taskid__(self):
        return 'W2VSimilarWords_%d_%d_%d' % (self.h, self.D,
                                           containerHash(self.graph_list))

    def run(self):
        with self.input()[0] as i:
            vocab = i.query()
        with self.input()[1] as i:
            embedding = np.array(i.query())

        inv_vocab = [None]*len(vocab)
        for k, v in vocab.items():
            inv_vocab[v[0]] = k
        inv_vocab = inv_vocab

        dis = cdist(embedding, embedding, 'cosine')
        arg_sort = np.argsort(dis, axis=1)[:, 1:6]

        near = {}

        for i, k in enumerate(inv_vocab):
            row = arg_sort[i]
            near[k] = []
            for j in range(row.shape[0]):
                near[k].append([inv_vocab[row[j]], 1-dis[i, j]])

        with self.output() as o:
            o.emit(near)
