from __future__ import absolute_import
from __future__ import division
import logging
import tensorflow as tf
from modules import BiRNNEncoder,SimpleSoftmaxLayer
from tensorflow.python.ops import variable_scope as vs
#from tensorflow.python.ops import embedding_ops
import os
import time
import sys
import numpy as np
import data_batcher

logging.basicConfig(level=logging.INFO)

class SentimentModel(object):
    def __init__(self,FLAGS,word2id,char2id,word_embed_matrix,char_embed_matrix,review_path,rating_path,batch_size,review_length,word_length,discard_long,test_size=0.1):
        self.FLAGS=FLAGS
        self.dataObject=data_batcher.YelpDataFromFile(word2id,char2id,word_embed_matrix,char_embed_matrix,review_path,rating_path,batch_size,review_length,word_length,discard_long,test_size=0.1)

        with tf.variable_scope("SentimentModel",initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,uniform=True)):
            self.add_placeholders()
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params=tf.trainable_variables()
        gradients=tf.gradients(self.loss,params)
        self.gradient_norm=tf.global_norm(gradients)
        clipped_gradients,_=tf.clip_by_global_norm(gradients,FLAGS.max_gradient_norm)
        self.param_norm=tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step=tf.Variable(0,name="global_step",trainable=False)
        opt=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # you can try other optimizers
        self.updates=opt.apply_gradients(zip(clipped_gradients,params),global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.keep)
        self.bestmodel_saver=tf.train.Saver(tf.global_variables(),max_to_keep=1)
        self.summaries=tf.summary.merge_all()

    def add_placeholders(self):
        self.review_words=tf.placeholder(tf.int32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_embedding_size])
        self.review_mask=tf.placeholder(tf.int32,shape=[None,self.FLAGS.review_length])
        self.review_char_ids=tf.placeholder(tf.int32,shape=[None,self.FLAGS.review_length,self.FLAGS.word_length,self.FLAGS.character_embedding_size])
        self.labels=tf.placeholder(tf.float32,shape=[None,5])
        self.keep_prob=tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        encoder1=BiRNNEncoder(self.FLAGS.hidden_Size,self.keep_prob)
        review_hidden=encoder1.build_graph(self.review_words,self.review_mask)

        with vs.variable_scope("Softmax"):
            softmax_layer=SimpleSoftmaxLayer()
            self.logits,self.probdist=softmax_layer.build_graph(review_hidden,self.review_mask)

    def add_loss(self):
        with vs.variable_scope("loss"):
            # Calculate loss for prediction of start position
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels)
            self.loss=tf.reduce_mean(loss)
            tf.summary.scalar('loss',self.loss)

    def run_train_iter(self,session, review_words,review_char_ids,review_mask,review_labels,summary_writer):
        """
                This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

                Inputs:
                  session: TensorFlow session
                  batch: a Batch object
                  summary_writer: for Tensorboard

                Returns:
                  loss: The loss (averaged across the batch) for this batch.
                  global_step: The current number of training iterations we've done
                  param_norm: Global norm of the parameters
                  gradient_norm: Global norm of the gradients
                """
        # Match up our input data with the placeholders
        input_feed={}
        input_feed[self.review_words]=review_words
        input_feed[self.review_char_ids]=review_char_ids
        input_feed[self.review_mask]=review_mask
        input_feed[self.labels]=review_labels
        input_feed[self.keep_prob]=1.0-self.FLAGS.dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def get_loss(self, session, review_words,review_char_ids,review_mask,review_labels):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.review_words] = review_words
        input_feed[self.review_char_ids] = review_char_ids
        input_feed[self.review_mask] = review_mask
        input_feed[self.labels] = review_labels
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_prob_dist(self,session,review_words,review_char_ids,review_mask,review_labels):
        input_feed = {}
        input_feed[self.review_words] = review_words
        input_feed[self.review_char_ids] = review_char_ids
        input_feed[self.review_mask] = review_mask
        input_feed[self.labels] = review_labels
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed=[self.probdist]
        [probdist]=session.run(output_feed,input_feed)
        return probdist

    def get_value(self, session, review_words,review_char_ids,review_mask,review_labels):
        """
        Run forward-pass only; get the most likely value.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        prob_dist= self.get_prob_dists(session, review_words,review_char_ids,review_mask,review_labels)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        value=np.argmax(prob_dist, axis=1)
        return value

    def get_test_loss(self, session):
        """
        Get loss for entire test set.

        Inputs:
          session: TensorFlow session

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        for review_words,review_chars,review_mask,ratings in self.dataObject.generate_test_data(batch_size=self.FLAGS.batch_size):
            # Get loss for this batch
            loss = self.get_loss(session,review_words,review_chars,review_mask,ratings)
            curr_batch_size = len(review_words)
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print("Computed test loss over %i examples in %.2f seconds" % (total_num_examples, toc - tic))
        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)
        return dev_loss

    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path,
              dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0
        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for review_words,review_chars,review_mask,ratings in self.dataObject.generate_one_epoch(batch_size=self.FLAGS.batch_size):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session,review_words,review_chars,review_mask,ratings , summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss:  # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info('epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %(epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    test_loss = self.get_test_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, test loss: %f" % (epoch, global_step, test_loss))
                    write_summary(test_loss, "test/loss", summary_writer, global_step)

                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if test_loss is None or test_loss < best_test_loss:
                        best_test_loss=test_loss
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)

            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc - epoch_tic))

        sys.stdout.flush()

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)