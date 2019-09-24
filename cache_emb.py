# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import itertools
import modeling
import optimization



from preprocess import *
flags = tf.flags

FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string("ds_name", None, "The name of the dataset")
flags.DEFINE_string("emb_type", None, "[edoc, bert]")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
        "data_dir", None,
        "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
        "output_dir", "/tmp/model/",
        "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
        "do_lower_case", False,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_bool("use_iobul", True, "")
flags.DEFINE_bool("mask", False, "")

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
        "doc_agr", False,
        "Whether to extract features or not")

flags.DEFINE_bool(
        "data_agr", False,
        "Whether to extract features or not")

flags.DEFINE_bool("add_mask", False,"")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

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

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

tf.flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def file_based_convert_examples_to_features(dataset, output_file, tokenizer, is_train=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for doc_id, instances in enumerate(dataset.docs):
        instances = doc_level_aggregation(instances, FLAGS.max_seq_length, doc_agr=FLAGS.doc_agr)
        dataset.docs[doc_id] = instances

    # combine 2 docs if thier len(tokens) less than max_seq_length
    #  and each contains one block
    dataset.instances = dataset_level_aggregation(dataset.docs, FLAGS.max_seq_length, FLAGS.data_agr)

    for instance in dataset.instances:
        tf_example = instance.to_tfrecord(tokenizer, FLAGS.max_seq_length, is_bert_emb= FLAGS.emb_type == "bert")
        writer.write(tf_example.SerializeToString())

    writer.close()


def file_based_input_fn_builder(input_file, max_seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {"input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
                        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
                        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
                        "masked_lm_positions": tf.FixedLenFeature([max_seq_length], tf.int64),
                        "masked_lm_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
                        "masked_lm_weights": tf.FixedLenFeature([max_seq_length], tf.float32)
                        }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(tf.contrib.data.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                  batch_size=batch_size,
                                                  drop_remainder=drop_remainder))

        return d

    return input_fn


def get_masked_lm_output(bert_config, input_tensor, positions, label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(input_tensor,
                                           units=bert_config.hidden_size,
                                           activation=modeling.get_activation(bert_config.hidden_act),
                                           kernel_initializer=modeling.create_initializer(
                                               bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_weights = tf.get_variable("output_weights_task",
                                         shape=[bert_config.tags_num, bert_config.hidden_size],
                                         initializer=modeling.create_initializer(bert_config.initializer_range))

        output_bias = tf.get_variable("output_bias_task",
                                      shape=[bert_config.tags_num],
                                      initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
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

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, layer_indexes):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # This will fall back to BertModel if num_cnn_layer=0
        model = modeling.DCNNModel(config=bert_config,
                                   is_training=is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=use_one_hot_embeddings)


        (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), masked_lm_positions, masked_lm_ids, masked_lm_weights)


        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(labels=masked_lm_ids,
                                                         predictions=masked_lm_predictions,
                                                         weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(values=masked_lm_example_loss, weights=masked_lm_weights)

                return {"masked_lm_accuracy": masked_lm_accuracy, "masked_lm_loss": masked_lm_mean_loss}

            eval_metrics = (metric_fn, [masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,masked_lm_weights])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          eval_metrics=eval_metrics,
                                                          scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            if FLAGS.extract_feature:
                all_layers = model.get_all_encoder_layers()
                layer_output = tf.stack([all_layers[layer_index] for layer_index in layer_indexes], -1)
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                              predictions=layer_output,
                                                              scaffold_fn=scaffold_fn)

        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    # FLAGS.max_seq_length = bert_config.max_position_embeddings

    tf.gfile.MakeDirs(FLAGS.output_dir)

    ds_name = FLAGS.ds_name
    tokenizer = FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=("uncased" in FLAGS.vocab_file))
    dataset, _, _, _ = load_dataset(ds_name)

    bert_config.tags_num = dataset["train"].tags_num
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(FLAGS.tpu_name,
                                                                              zone=FLAGS.tpu_zone,
                                                                              project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver,
                                          master=FLAGS.master,
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          tpu_config=tf.contrib.tpu.TPUConfig(
                                              iterations_per_loop=FLAGS.iterations_per_loop,
                                              num_shards=FLAGS.num_tpu_cores,
                                              per_host_input_for_training=is_per_host)
                                          )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    train_file = os.path.join(FLAGS.output_dir, "%s_train.tf_record" % ds_name)

    if FLAGS.do_train:
        file_based_convert_examples_to_features(dataset["train"], train_file, tokenizer, True)
        train_examples = dataset["train"].instances
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    ext_layers = [-1, -2, -3, -4] if FLAGS.emb_type == "bert" else [-1, -2]
    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_tpu=FLAGS.use_tpu,
                                use_one_hot_embeddings=FLAGS.use_tpu,
                                layer_indexes=ext_layers)#

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.train_batch_size,
                                            eval_batch_size=FLAGS.eval_batch_size,
                                            predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("    Num examples = %d", len(train_examples))
        tf.logging.info("    Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("    Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     max_seq_length=FLAGS.max_seq_length,
                                                     is_training=True,
                                                     drop_remainder=FLAGS.use_tpu)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        for portion, examples in dataset.items():
            if portion == "train":
                continue

            eval_file = os.path.join(FLAGS.output_dir, "%s.tf_record"%portion)
            file_based_convert_examples_to_features(examples, eval_file, tokenizer)
            tf.logging.info("***** Running evaluation on %s*****" % portion)
            tf.logging.info("    Num examples = %d", len(examples.instances))
            tf.logging.info("    Batch size = %d", FLAGS.eval_batch_size)

            # This tells the estimator to run through the entire set.
            eval_steps = None
            eval_input_fn = file_based_input_fn_builder(input_file=eval_file,
                                                        max_seq_length=FLAGS.max_seq_length,
                                                        is_training=False,
                                                        drop_remainder=FLAGS.use_tpu)

            # eval
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results on %s*****" % portion)
                for key in sorted(result.keys()):
                    tf.logging.info("    %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    # here use use predict to get representation
    if FLAGS.do_predict:
        features_dict = defaultdict(list)
        for portion, examples in dataset.items():
            if portion == "train" and not FLAGS.extract_feature :
                continue

            pred_file = "/tmp/%s_%s.tf_record"% (ds_name, portion)
            if not FLAGS.do_eval:
                file_based_convert_examples_to_features(examples, pred_file, tokenizer)
            tf.logging.info("***** Running prediction on %s*****" % portion)
            tf.logging.info("    Num examples = %d", len(examples.instances))
            tf.logging.info("    Batch size = %d", FLAGS.eval_batch_size)

            # This tells the estimator to run through the entire set.
            pred_input_fn = file_based_input_fn_builder(input_file=pred_file,
                                                        max_seq_length=FLAGS.max_seq_length,
                                                        is_training=False,
                                                        drop_remainder=FLAGS.use_tpu)

            # predict
            result = estimator.predict(input_fn=pred_input_fn)
            if FLAGS.extract_feature:
                for instance, prediction in zip(examples.instances, result):
                    features_dict[portion] += instance.decode_feature(prediction)
                features_dict[portion] = sorted(features_dict[portion], key=lambda x: (x[0], x[1]))

        if FLAGS.extract_feature:
            with h5py.File(os.path.join(FLAGS.data_dir, ds_name, "cache_%s.hdf5" % FLAGS.emb_type), "w") as out_file:
                for portion, examples in dataset.items():
                    for doc_id, features in itertools.groupby(features_dict[portion], lambda x: x[0]):
                        file_key = "%s_%s" % (portion, doc_id)
                        group = out_file.create_group(file_key)
                        for d, sent_num, feature in features:
                            group[str(sent_num)] = feature

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("ds_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("data_dir")

    tf.app.run()
