# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from . import tokenization

__all__ = ['convert_lst_to_features']


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        # self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_lst_to_features(lst_str, max_seq_length, max_position_embeddings,
                            tokenizer, logger, is_tokenized=False, mask_cls_sep=False):
    label_list=[]
    for i in lst_str:
        label_list.append('')
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    """Loads a data file into a list of `InputBatch`s."""
    def get_features(lst_str, max_seq_length, max_position_embeddings, tokenizer, logger, is_tokenized=False, mask_cls_sep=False):
        examples = read_tokenized_examples(lst_str) if is_tokenized else read_examples(lst_str)

        _tokenize = lambda x: tokenizer.mark_unk_tokens(x) if is_tokenized else tokenizer.tokenize(x)

        all_tokens = [(_tokenize(ex.text_a), _tokenize(ex.text_b) if ex.text_b else []) for ex in examples]

        # user did not specify a meaningful sequence length
        # override the sequence length by the maximum seq length of the current batch
        if max_seq_length is None:
            max_seq_length = max(len(ta) + len(tb) for ta, tb in all_tokens)
            # add special tokens into account
            # case 1: Account for [CLS], tokens_a [SEP], tokens_b [SEP] -> 3 additional tokens
            # case 2: Account for [CLS], tokens_a [SEP] -> 2 additional tokens
            max_seq_length += 3 if any(len(tb) for _, tb in all_tokens) else 2
            max_seq_length = min(max_seq_length, max_position_embeddings)
            logger.warning('"max_seq_length" is undefined, '
                           'and bert config json defines "max_position_embeddings"=%d. '
                           'hence set "max_seq_length"=%d according to the current batch.' % (
                               max_position_embeddings, max_seq_length))

        for (tokens_a, tokens_b) in all_tokens:
            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            input_type_ids = [0] * len(tokens)
            input_mask = [int(not mask_cls_sep)] + [1] * len(tokens_a) + [int(not mask_cls_sep)]

            if tokens_b:
                tokens += tokens_b + ['[SEP]']
                input_type_ids += [1] * (len(tokens_b) + 1)
                input_mask += [1] * len(tokens_b) + [int(not mask_cls_sep)]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length. more pythonic
            pad_len = max_seq_length - len(input_ids)
            input_ids += [0] * pad_len
            input_mask += [0] * pad_len
            input_type_ids += [0] * pad_len

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(input_type_ids) == max_seq_length

            logger.debug('tokens: %s' % ' '.join([tokenization.printable_text(x) for x in tokens]))
            logger.debug('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
            logger.debug('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
            logger.debug('input_type_ids: %s' % ' '.join([str(x) for x in input_type_ids]))
            
            label_id = label_map[example.label]
            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                label_id=label_id,
                is_real_example=True,
                segment_ids=input_type_ids)
            
     #get features
     #features=get_ans()
     #write to tfrecord
     #file_based_convert_examples_to_features(get_ans())
     #load classifier for result
     #init_ckpt="./qnli2/bert_model
     #result=get_ans(init_ckpt)   
     
            

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(lst_strs):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for ss in lst_strs:
        line = tokenization.convert_to_unicode(ss)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1

def file_based_convert_examples_to_features(features):
  """Convert a set of `InputExample`s to a TFRecord file."""
  #saves features to file
  output_file='./tmp/preds.tfrecord'  
  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    #if ex_index % 10000 == 0:
      #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    features = get_features(ex_index, example, label_list, max_seq_length, tokenizer)
    for feature in features:
        def create_int_feature(values):
          f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
          return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
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

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def read_tokenized_examples(lst_strs):
    unique_id = 0
    lst_strs = [[tokenization.convert_to_unicode(w) for w in s] for s in lst_strs]
    for ss in lst_strs:
        text_a = ss
        text_b = None
        try:
            j = ss.index('|||')
            text_a = ss[:j]
            text_b = ss[(j + 1):]
        except ValueError:
            pass
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn 

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def ans(init_checkpoint):
  model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels=len(label_list),
  init_checkpoint=init_checkpoint,
  learning_rate=5e-5,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=False,
  use_one_hot_embeddings=False)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=32,
      eval_batch_size=32,
      predict_batch_size=32)

  #no need for data_dir
  #if FLAGS.do_predict:
   #predict_examples = processor.get_test_examples(data_dir)
   #num_actual_eval_examples = len(eval_examples)
  predict_file = "./tmp/preds.tf_record"
   #eval_steps = None
  predict_drop_remainder = False

  max_seq_legnth=80  
  predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

  result = estimator.predict(input_fn=predict_input_fn)
  return result
