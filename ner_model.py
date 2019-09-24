from modeling import *
from preprocess import *
from random import shuffle
import time

flags = tf.flags

FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
        "data_dir", None,
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string("ds_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
        "output_dir", config["data_dir"],
        "The output directory where the model checkpoints will be written.")


## Other parameters
flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
        "do_lower_case", False,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_bool(
        "use_iobul", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer(
        "max_seq_length", 640,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
        "do_predict", False,
        "Whether to run the model in inference mode on the test set.")
flags.DEFINE_bool(
        "extract_feature", False,
        "Whether to extract features or not")

flags.DEFINE_bool(
        "crf", False,
        "Whether to extract features or not")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 54, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                                         "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                                         "How many steps to make in each estimator call.")

# old_model
flags.DEFINE_float("clip_grad_norm", 5.0, "")
flags.DEFINE_integer("report_frequency", 100, "")
tf.flags.DEFINE_string("optimizer", "adam", "")
flags.DEFINE_float("lr_decay", 0.8, "")
flags.DEFINE_float("log_interval", 500, "")



class Model:
    def __init__(self, config, emb_matrix, fix_emb_matrix):

        # Define input and target tensors
        self.hidden_dropout_prob = tf.placeholder(tf.float32)
        self.attention_probs_dropout_prob = tf.placeholder(tf.float32)

        self.input_ids = tf.placeholder(tf.int32, [None, None], name="input_ids")
        self.input_mask = tf.placeholder(tf.int32, [None, None], name="input_mask")
        self.feat_ids = tf.placeholder(tf.int32, [None, None, config.feat_num], name="feat_ids")

        s = [None, None, config.max_char_len]
        self.char_ids = tf.placeholder(tf.int32, s, name="char_ids") if config.use_char else None
        s = [None, None, config.elmo_size, config.elmo_lnum]
        self.elmo_emb = tf.placeholder(tf.float32, s, name="elmo_emb") if config.use_elmo else None
        s = [None, None, config.edoc_size, config.edoc_lnum]
        self.edoc_emb = tf.placeholder(tf.float32, s, name="edoc_emb") if config.use_edoc else None
        s = [None, None, config.flair_size, config.flair_lnum]
        self.flair_emb = tf.placeholder(tf.float32, s, name="flair_emb") if config.use_flair else None
        s = [None, None, config.bert_size, config.bert_lnum]
        self.bert_emb = tf.placeholder(tf.float32, s, name="bert_emb") if config.use_bert else None

        self.label_positions = tf.placeholder(tf.int32, [None, None], name="label_positions")
        self.label_ids = tf.placeholder(tf.int32, [None, None], name="label_ids")
        self.label_weights = tf.placeholder(tf.float32, [None, None], name="label_weights")

        seq_len = tf.to_int32(tf.reduce_sum(self.input_mask, -1)) -1 # remove [SEP]
        word_input = []

        # pre-trained embeddings
        if config.use_pretemb:
            pret_embedding = tf.get_variable(name="pretemb",
                                             shape=emb_matrix.shape,
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(emb_matrix),
                                             trainable= not config.freeze)

            emb_input = tf.nn.embedding_lookup(pret_embedding, self.input_ids)
            word_input.append(emb_input)

        if config.use_fixemb:
            fix_embedding = tf.get_variable(name="fixemb",
                                            shape=fix_emb_matrix.shape,
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(fix_emb_matrix),
                                            trainable= False)

            emb_input = tf.nn.embedding_lookup(fix_embedding, self.input_ids)
            word_input.append(emb_input)

        if config.use_feat:
            feat_embedding = tf.get_variable(name="cap_embedding",
                                             initializer=tf.random_uniform(
                                                [config.feat_vocab_size, config.feat_emb_size],
                                                minval=-(3. / config.feat_emb_size) ** .5,
                                                maxval=(3. / config.feat_emb_size) ** .5),
                                             trainable=True)

            feat_input = tf.nn.embedding_lookup(feat_embedding, self.feat_ids)
            input_shape = get_shape_list(feat_input)
            feat_input = tf.reshape(feat_input, input_shape[0:-2] + [input_shape[-2] * config.feat_emb_size])
            word_input.append(feat_input)

        if config.use_char:
            char_input = char_cnn_model(config, self.char_ids)
            word_input.append(char_input)

        # add elmo and edoc embeddings
        for enc in ["elmo", "edoc", "flair", "bert"]:
            if getattr(config, "use_%s" % enc):
                with tf.variable_scope(enc):
                    word_input.append(
                        embedding_elmo(getattr(self, "%s_emb" % enc), concat=getattr(config, "concat_%s" % enc)))

        word_input = tf.concat(word_input, 2)
        all_layer_outputs = lstm_encoder(word_input, seq_len, config.num_lstm_layers, config.hidden_size,
                                               self.hidden_dropout_prob, config.initializer_range)#lstm_contextualize lstm_encoder

        output = all_layer_outputs[-1]
        # output_shape = get_shape_list(output, expected_rank=3)
        # output = tf.reshape(output, [-1, output_shape[2]])
        self.logits = tf.layers.dense(output, config.tags_num, activation=None)

        if FLAGS.crf:
            # self.logits = tf.reshape(self.logits, [-1, output_shape[1], config.tags_num])
            log_likelihood, self.transition_params = \
                tf.contrib.crf.crf_log_likelihood(self.logits, self.label_ids, seq_len)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            self.loss, self.log_probs = get_softmax_output(config, self.logits, self.label_ids, self.label_weights)

        tvars = tf.trainable_variables()
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            tf.logging.info("    name = %s, shape = %s", var.name, var.shape)
        self._lr = tf.Variable(0.0, trainable=False)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), FLAGS.clip_grad_norm)

        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer,
            "mom": tf.train.MomentumOptimizer
        }

        if FLAGS.optimizer == "mom":
            optimizer = optimizers[FLAGS.optimizer](self._lr, 0.9)
        else:
            optimizer = optimizers[FLAGS.optimizer](self._lr)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def get_softmax_output(bert_config, logits, label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    logits_shape = get_shape_list(logits, expected_rank=3)
    logits = tf.reshape(logits, [-1, bert_config.tags_num])
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_ids, depth=bert_config.tags_num, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

    log_probs = tf.reshape(log_probs, [-1, logits_shape[1], bert_config.tags_num])
    return loss, log_probs


class Batcher:
    def __init__(self, data, portion="train"):
        self.portion = portion
        self.is_train = portion == "train"

        self.word_to_emb = data.word_to_emb
        self.word_to_char = data.word_to_char
        self.batch_size = FLAGS.train_batch_size if self.is_train else FLAGS.predict_batch_size
        self.instances = data.instances
        self.instance_idx = list(range(len(self.instances)))
        if self.is_train:
            shuffle(self.instance_idx)

        cont_emb_dir = os.path.join(config["data_dir"], FLAGS.ds_name)
        self.encoders = []
        for cont_emb in ["elmo", "edoc", "edoc", "flair", "bert"]:
            self.encoders.append((cont_emb, h5py.File(os.path.join(cont_emb_dir, "cache_%s.hdf5" % cont_emb), 'r')))

    def iterator(self, config):
        for j in range(0, len(self.instances), self.batch_size):
            idx_lst = []
            for k in range(j, j+self.batch_size):
                if k >= len(self.instances):
                    break
                idx_lst.append(self.instance_idx[k])
            max_seq_length = max([len(self.instances[i].orig_tokens) for i in idx_lst])

            data = []
            for i in idx_lst:
                data.append(self.instances[i].to_numpy(config, max_seq_length,
                                                       self.portion, encoders=self.encoders,
                                                       word_to_emb=self.word_to_emb,
                                                       word_to_char=self.word_to_char))

            yield dict([(k, np.stack([inst[k] for inst in data], 0)) for k in data[0]])

        if self.is_train:
            shuffle(self.instance_idx)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def get_dico(config, model, data, is_train=True):
    if is_train:
        dico = {model.hidden_dropout_prob: config.hidden_dropout_prob,
                model.attention_probs_dropout_prob: config.attention_probs_dropout_prob}
    else:
        dico = {model.hidden_dropout_prob: 0,
                model.attention_probs_dropout_prob: 0}

    for k, v in data.items():
        if hasattr(model, k) and getattr(model, k) is not None:
            dico[getattr(model, k)] = v

    return dico


def evaluate(bert_config, sess, model, writer, tf_global_step, portion, batcher, dataset):
    avg_cost = 0.
    gen = batcher.iterator(bert_config)
    predictions = []

    for data in gen:
        dico = get_dico(bert_config, model, data, False)
        if FLAGS.crf:
            tf_loss, logits, tf_transition_params = sess.run([model.loss, model.logits, model.transition_params], dico)
            prediction = viterbi_decoder(logits, np.sum(data["input_mask"], -1), tf_transition_params)
        else:
            tf_loss, log_probs = sess.run([model.loss, model.log_probs], dico)
            prediction = np.argmax(log_probs, axis=-1)
            prediction = np.reshape(prediction, [-1, data["input_mask"].shape[1]]).tolist()

        predictions += prediction
        avg_cost += tf_loss * data["input_mask"].shape[0]

    avg_cost /= len(dataset.instances)

    output = []
    for instance, prediction in zip(dataset.instances, predictions):
        output.append(instance.decode_pred(prediction, dataset.id_to_tag, seq_split="orig"))
    output = "\n\n".join(output)

    # Write predictions to disk and run CoNLL script externally
    output_pred_file = os.path.join(FLAGS.output_dir, "%s.predictions.txt" % portion)
    output_score_file = os.path.join(FLAGS.output_dir, "%s.scores.txt" % portion)
    f = open(output_pred_file, 'w')
    f.write(output)
    f.close()
    os.system("%s < %s > %s" % (config["eval_script"], output_pred_file, output_score_file))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in open(output_score_file, 'r')]
    val = float(eval_lines[1].strip().split()[-1])
    tf.logging.info("    F1 score on %s= %s", portion, val)
    writer.add_summary(make_summary({"%s_loss" % portion: avg_cost, "%s_score" % portion: val}), tf_global_step)

    return val


def viterbi_decoder(output, length, tf_transition_params):
    prediction = []

    for tf_unary_scores_, sequence_length_ in zip(output, length):
        # Remove padding from the scores and tag sequence.
        tf_unary_scores_ = tf_unary_scores_[:sequence_length_]

        # Compute the highest scoring sequence.
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, tf_transition_params)

        prediction.append(viterbi_sequence)

    return prediction


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS.output_dir = os.path.join(config["data_dir"], FLAGS.ds_name, "model")
    # tensorboard
    writer = tf.summary.FileWriter(FLAGS.output_dir, flush_secs=20)
    bert_config = SeqConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    ds_name = FLAGS.ds_name
    dataset, emb_matrix, file_dict, fix_emb_matrix = load_dataset(ds_name)
    bert_config.tags_num = dataset["train"].tags_num

    tf.logging.info("Creating batchers")
    batcher = {}
    for portion, data in dataset.items():
        batcher[portion] = Batcher(data, portion)

    g = tf.Graph()
    with g.as_default():
        with tf.variable_scope("seq_%s" % ds_name):
            model = Model(bert_config, emb_matrix, fix_emb_matrix)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    max_score = 0
    tf_global_step = 0

    with tf.Session(config=gpu_config, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if FLAGS.do_train:
            train_examples = dataset["train"].instances
            num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            tf.logging.info("***** Running training *****")
            tf.logging.info("    Num examples = %d", len(train_examples))
            tf.logging.info("    Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("    Num steps = %d", num_train_steps)
            cur_lr = FLAGS.learning_rate
            FLAGS.num_train_epochs = int(FLAGS.num_train_epochs)
            max_batch_num = len(batcher["train"].instances) // FLAGS.train_batch_size

            for e in range(FLAGS.num_train_epochs):
                cur_lr *= 1 / (1.02 + FLAGS.lr_decay * e)
                model.assign_lr(sess, cur_lr)
                gen = batcher["train"].iterator(bert_config)
                batch, avg_cost = 0, 0.
                start_time = time.time()
                for data in gen:
                    dico = get_dico(bert_config, model, data)
                    tf_loss, lr, _ = sess.run([model.loss, model.lr, model.train_op], dico)
                    avg_cost += tf_loss * data["input_ids"].shape[0]
                    batch += data["input_ids"].shape[0]

                    if batch and batch % FLAGS.log_interval == 0:
                        cur_loss = avg_cost / batch
                        writer.add_summary(make_summary({"loss": cur_loss, "lr": lr}), tf_global_step)
                        elapsed = time.time() - start_time
                        tf.logging.info(
                              f'| epoch {e:3d} | {batch// FLAGS.train_batch_size:5d}/{max_batch_num:0d} batches' +
                              f'| lr {lr:1.5f} | ms/batch {elapsed * 1000 / FLAGS.log_interval:5.2f} |' +
                              f'loss {cur_loss:5.6f}')
                        start_time = time.time()
                    tf_global_step += 1

                avg_cost /= batch
                print("Epoch:", '%04d' % (e + 1), "cost=", "{:.9f}".format(avg_cost))
                writer.add_summary(make_summary({"loss": avg_cost, "lr": lr}), tf_global_step)
                for portion, examples in batcher.items():
                    if portion != "train":
                        score = evaluate(bert_config, sess, model, writer, tf_global_step,
                                         portion, examples, dataset[portion])

                    if portion == "dev" and score > max_score:
                        saver.save(sess, FLAGS.output_dir)
                        max_score = score

        saver.restore(sess, FLAGS.output_dir)
        print("model restored")

        for portion, examples in batcher.items():
            if portion != "train":
                evaluate(bert_config, sess, model, writer, tf_global_step,portion, examples, dataset[portion])


    del g
    sess.close()


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    flags.mark_flag_as_required("ds_name")
    flags.mark_flag_as_required("bert_config_file")
    FLAGS.log_interval = FLAGS.log_interval * FLAGS.train_batch_size
    FLAGS.lr_decay = FLAGS.learning_rate / FLAGS.num_train_epochs

    tf.app.run()
