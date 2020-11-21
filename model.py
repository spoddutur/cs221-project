import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import backend as K

import os
os.chdir("/Users/sruthip/Downloads/bert-entity-tagging")
import bert
os.chdir("/Users/sruthip/Downloads/bert-entity-tagging/bert")
from bert import modeling
from bert import run_classifier
from bert import optimization
from bert import tokenization
os.chdir("/Users/sruthip/Downloads/bert-entity-tagging/")

BERT_VOCAB = '/Users/sruthip/Downloads/uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = '/Users/sruthip/Downloads/uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = '/Users/sruthip/Downloads/uncased_L-12_H-768_A-12/bert_config.json'
tag2idx = {'PAD': 0, 'I-STR':1, 'B-STR':2, 'O': 3}

tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

import pandas as pd
INPUT_FILE_PATH = "/Users/sruthip/Downloads/part0-parsed-combined-street1.csv"
INPUT_FILE_PATH = "/Users/sruthip/Downloads/part0-parsed-combined-street1lakhlabels.csv"
df = pd.read_csv(INPUT_FILE_PATH, header=0)[["id", "model_input", "model_label"]]

# CONVERT LABEL in df TO NER TAGS
def label_to_string_streets(model_input, model_label):
    model_input = model_input.split(" ")
    start = -1
    end = -1
    index = 0
    out = []
#     print(label)
    for l in model_label.split(","):
#         print(l)
        if l == "I-STR":
            if start == -1:
                start = index
            end = index
        elif start != -1 and end != -1:
            out.append(" ".join(model_input[start:end+1]))
            start = -1
            end = -1
        index = index + 1
    return out

label_to_string_streets_fn = np.vectorize(label_to_string_streets, otypes=[list])
df["final_prediction"] = label_to_string_streets_fn(df["model_input"], df["model_label"])

def listoflist_to_array(listoflist):
    arr = []
    for l in listoflist:
        arr.append(np.array(l))
    return np.array(arr)

addresses = listoflist_to_array(df.apply(lambda x: x['model_input'].split(" "), axis=1))
labels = listoflist_to_array(df.apply(lambda x: x['model_label'].split(","), axis=1))

# SHUFFLE DATA BEFORE TAKING 50K sample
np.random.shuffle(addresses)
left_train, right_train = addresses[:50000], labels[:50000]
left_test, right_test = addresses[50000:55000], labels[50000:55000]

# LOAD DATA
def XYandRaw(left_train, right_train):
    X, Y, RAW_TO_X = [], [], []
    for i in tqdm(range(len(left_train))):
        left = left_train[i]
        right = right_train[i]
        bert_tokens = ['[CLS]']
        y = ['PAD']
        r_to_x = [1] 
        for no, orig_token in enumerate(left):
            y.append(right[no])
            t = tokenizer.tokenize(orig_token)
            bert_tokens.extend(t)
            y.extend(['PAD'] * (len(t) - 1))
            r_to_x.append(len(t))
        bert_tokens.append("[SEP]")
        y.append('PAD')
        r_to_x.append(1)
        X.append(tokenizer.convert_tokens_to_ids(bert_tokens))
        Y.append([tag2idx[i] for i in y])
        RAW_TO_X.append(r_to_x)
        # print(len(tokenizer.convert_tokens_to_ids(bert_tokens)), len(y), len(r_to_x), r_to_x, left.shape)
    return X, Y, RAW_TO_X
train_X, train_Y, train_RAW_TO_X = XYandRaw(left_train, right_train)
test_X, test_Y, test_RAW_TO_X = XYandRaw(left_test, right_test)

# PADDING OF DATA
import keras
train_X = keras.preprocessing.sequence.pad_sequences(train_X, padding='post')
train_Y = keras.preprocessing.sequence.pad_sequences(train_Y, padding='post')
test_X = keras.preprocessing.sequence.pad_sequences(test_X, padding='post')
test_Y = keras.preprocessing.sequence.pad_sequences(test_Y, padding='post')

# DEFINING LAYERS
import tensorflow_hub as hub

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean", "sequence_output"]:
            raise NameError("0000000 Undefined pooling type (must be either first or mean or sequence_output, but is " + self.pooling)

        super(BertLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape + (768,)

    def build(self, input_shape):
#         self.bert = hub.Module(
#             self.bert_path, trainable=self.trainable, name=self.name+"_module"
#         )

        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name="bert"
        )
        
        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean" or self.pooling == "sequence_output":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                "111111 Undefined pooling type (must be either first or mean or sequence_output, but is" + self.pooling
            )

        # print([v.name for v in trainable_vars])
        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append("encoder/layer_"+ str(11 - i))

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids, sequence_lengths = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )

        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        elif self.pooling == "sequence_output":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]
        else:
            raise NameError("2222222 Undefined pooling type (must be either first or mean or sequence_output, but is "+ self.pooling)

        return pooled

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_size)

class CRFLayer(tf.keras.layers.Layer):

    def __init__(self, num_tags, max_seq_len, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_tags = num_tags
        self.max_seq_len = max_seq_len

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.num_tags,)

    def build(self, input_shape):
        """Creates the layer weights.
        Args:
            input_shape (list(tuple, tuple)): [(batch_size, n_steps, n_classes), (batch_size, 1)]
        """
        self.transitions = self.add_weight(shape=(self.num_tags, self.num_tags),
                                                 initializer='glorot_uniform',
                                                 name='transitions',
                                                trainable=True)
        self.built = True

    def call(self, inputs, mask=None, **kwargs):

        logits, sequence_lengths = inputs
        sequence_lengths = K.flatten(sequence_lengths)        
        tags_seq, tags_score = tf.contrib.crf.crf_decode(
            logits, self.transitions, sequence_lengths
        )
        tags_seq = tf.identity(tags_seq, name = 'logits')
        outputs = K.one_hot(tags_seq, self.num_tags)
        return K.in_train_phase(logits, outputs)

epoch = 3
batch_size = 16
warmup_proportion = 0.1
num_train_steps = int(len(train_X) / batch_size * epoch)
num_warmup_steps = int(num_train_steps * warmup_proportion)
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
MAX_SEQ_LENGTH = train_X.shape[1]
NUM_TAGS = len(tag2idx)

# init session
sess = tf.Session()
K.set_session(sess)

# DEFINING MODEL
in_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32',name="input_ids")
in_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32',name="input_masks")
in_segment = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32',name="segment_ids")
in_sequence_lengths = tf.keras.layers.Input(shape=(1,), dtype='int32',name="sequence_lengths")
model_inputs = np.asarray([in_id, in_mask, in_segment, in_sequence_lengths])
bert_inputs = [in_id, in_mask, in_segment, in_sequence_lengths]

# define layers
bert_layer = BertLayer(n_fine_tune_layers=3, bert_path="/Users/sruthip/Downloads/1", pooling="sequence_output")
dense_layer = tf.keras.layers.Dense(NUM_TAGS)
td_layer = tf.keras.layers.TimeDistributed(dense_layer)
crf_layer = CRFLayer(num_tags=NUM_TAGS, max_seq_len=MAX_SEQ_LENGTH)

# # invoke layers
bert_output = bert_layer(bert_inputs)
# dense_output = dense_layer(bert_output)
td_output = td_layer(bert_output)
crf_output = crf_layer((td_output, in_sequence_lengths))
mask = tf.sequence_mask(in_sequence_lengths, maxlen = MAX_SEQ_LENGTH)
model = tf.keras.models.Model(inputs=bert_inputs, outputs=crf_output)
model.summary()

optimizer = tf.train.AdamOptimizer(learning_rate = 2e-5)

# Define custom accuracy and loss functions for CRF
def accuracy(y_true, y_pred):
    print("In Accuracy")
    crf, idx = y_pred._keras_history[:2]

    # prepare inputs i.e., reshape and cast inputs
    inputs = crf.get_input_at(idx)
    logits, sequence_lengths = inputs
    sequence_lengths = K.flatten(sequence_lengths)
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    # y_true = tf.cast(y_true, tf.int32)
    
    # compute tags_seq
    tags_seq, tags_score = tf.contrib.crf.crf_decode(
            logits, crf.transitions, sequence_lengths)
    tags_seq = tf.identity(tags_seq, name = 'logits')

    mask = tf.sequence_mask(sequence_lengths, maxlen = crf.max_seq_len)
    
    print("SHAPES IN Accuracy:", y_true, tags_seq, mask)

    masked_prediction = tf.boolean_mask(tags_seq, mask)
    masked_label = tf.boolean_mask(y_true, mask)
    correct_pred = tf.equal(masked_prediction, masked_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def loss(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    
    # prepare inputs i.e., reshape and cast inputs
    inputs = crf.get_input_at(idx)
    logits, sequence_lengths = inputs
    sequence_lengths = K.flatten(sequence_lengths)
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    # y_true = tf.cast(y_true, tf.int32)
    
    print("SHAPES IN LOSS:", y_true, y_pred, sequence_lengths)
    log_likelihood, crf.transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, y_true, sequence_lengths, transition_params=crf.transitions
    )
    cost = tf.reduce_mean(-log_likelihood)
    return cost

# COMPILE MODEL
model.compile(loss=loss, metrics=[accuracy], optimizer=optimizer)

# INIT MODEL
def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

initialize_vars(sess)

# TRAIN MODEL
num_train_samples = 50000
model.fit(
    [train_input_ids[:num_train_samples], train_input_masks[:num_train_samples], train_segment_ids[:num_train_samples], train_sequence_lengths[:num_train_samples]],
    one_hot_train_Y[:num_train_samples],
    validation_data=(
        [test_input_ids, test_input_masks, test_segment_ids, test_sequence_lengths],
        one_hot_test_Y
    ),
    epochs=3,
    batch_size=512
)

# SAVEDMODEL
tf.saved_model.simple_save(
            sess,
            'street_ner_as_savedmodel_50K_samples',
            inputs={'input_ids': model.inputs[0], "input_masks":  model.inputs[1], "segment_ids": model.inputs[2], "sequence_lengths": model.inputs[3]},
            outputs={'output': model.outputs[0]},
            legacy_init_op = tf.tables_initializer()
        )


# PREDICTION FROM SAVED MODEL
latest = "street_ner_as_savedmodel_50K_samples"
from tensorflow.contrib import predictor
predict_fn = predictor.from_saved_model(latest)
# computing test accuracy
test_pred = predict_fn({'input_ids':test_input_ids,
    "input_masks": test_input_masks,
    "segment_ids":test_segment_ids,
    "sequence_lengths":test_sequence_lengths})["output"]

correct_pred = tf.equal(one_hot_test_Y[:10], test_pred[:10])
test_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
test_accuracy.eval(session=sess)  

