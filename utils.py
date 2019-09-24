import re, sys, string, unicodedata, os, itertools, pickle, copy, fasttext, joblib
from collections import Counter, OrderedDict, defaultdict
from glob import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py
import tqdm
import pyhocon

from tokenization import convert_to_unicode, FullTokenizer

#############
## Classes ##
#############
class Dataset():
    def __init__(self, docs, id_to_tag, feature_size, feature_num):
        """See base class.
        Args:
            file_dict: dict. k: portion name (e.g train, dev, test)or dev_brown for srl
                             v: path to that portion
            train_key: the key in file_dict that refer to train portion.
                       It is used to get tag mapping
            use_iobul: convert tags from iob to iobul
        """

        self.docs = docs
        self.instances = None
        self.id_to_tag = id_to_tag
        self.tags_num = len(id_to_tag)
        self.feature_size = feature_size
        self.feature_num = feature_num
        self.max_sent = -1
        self.word_to_emb = None
        self.word_to_char = None

    def get_word_list(self):
        vocab = []
        for doc in self.docs:
            for instance in doc:
                vocab += instance.orig_tokens

        return vocab


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, doc_id, sent_num, orig_tokens, orig_tags, orig_features,
                 bert_tokens, orig_to_tok_map, bert_tags, bert_features):
        """Constructs a InputExample.

        Args:
            tokens: list. A list of string where each item is a word of the original tokenization.
            sent_idx: list. The sent_num of each tokens in tokens.
            tags: list. Tags for each token in tokens.
            features: (Optional) list. A list of features associated with each token.

        """

        self.orig_tokens = orig_tokens
        self.orig_tags = orig_tags
        self.orig_features = orig_features
        self.orig_guids = ["%s_%s" % (doc_id, sent_num)] * len(self.orig_tokens)

        self.orig_to_tok_map = orig_to_tok_map

        # bert
        self.bert_tokens = bert_tokens
        self.bert_tags = bert_tags
        self.bert_features = bert_features
        self.bert_guids = ["%s_%s" % (doc_id, sent_num)] * len(self.bert_tokens)

    def merge(self, instance):
        self.orig_to_tok_map += [i+len(self.bert_tokens) for i in instance.orig_to_tok_map]
        for scheme in ["orig", "bert"]:
            for att in ["tokens", "tags", "features", "guids"]:
                nw = getattr(self, "%s_%s" %(scheme, att)) + getattr(instance, "%s_%s" %(scheme, att))
                setattr(self, "%s_%s" %(scheme, att), nw)

    def to_tfrecord(self, tokenizer, max_seq_length, seq_split="bert",
                    encoders=None, word_to_emb=None, is_bert_emb=False):
        """Converts a single `InputExample` into a single `TFRecord`."""
        tokens = getattr(self, "%s_tokens" % seq_split)
        tags = getattr(self, "%s_tags" % seq_split)
        feats = copy.deepcopy(getattr(self, "%s_features" % seq_split))


        # set word begin as index
        if seq_split == "bert":
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            if not is_bert_emb:
                for i in range(len(self.bert_tokens)):
                    if i in self.orig_to_tok_map:
                        segment_ids[i] = 1
        else:
            input_ids = [word_to_emb[tok] for tok in tokens] if word_to_emb else [0] * len(tokens)
            segment_ids = [1] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            feats.append([0] * len(feats[0]))
        if len(input_ids) > max_seq_length:
            print(len(self.orig_tokens), len(input_ids), self.orig_tokens)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(feats) == max_seq_length

        positions, ids, weights = list(range(len(tokens))), copy.deepcopy(tags), [1.0] * len(tokens)
        # if the scheme is bert than don't predict on intermidiare pieces or "[SEP]"
        for i in range(len(tokens)):
            if (seq_split == "bert" and i not in set(self.orig_to_tok_map)) or tokens[i] == "[SEP]":
                weights[i] = 0.0

        while len(positions) < max_seq_length:
            positions.append(0)
            ids.append(0)
            weights.append(0.0)

        assert len(positions) == max_seq_length
        assert len(ids) == max_seq_length
        assert len(weights) == max_seq_length

        features = OrderedDict()
        features["input_ids"] = _create_int_feature(input_ids)
        features["input_mask"] = _create_int_feature(input_mask)
        features["segment_ids"] = _create_int_feature(segment_ids)
        features["feat_ids"] = _create_int_feature(feats)
        if encoders:
            for enc_name, arr in encoders:
                features["%s_emb" % enc_name] = _create_float_feature(arr)

        features["masked_lm_positions"] = _create_int_feature(positions)
        features["masked_lm_ids"] = _create_int_feature(ids)
        features["masked_lm_weights"] = _create_float_feature(weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        return tf_example

    def to_numpy(self, config, max_seq_length, portion,
                 encoders=None, word_to_emb=None, word_to_char=None):
        features = OrderedDict()
        input_ids = [word_to_emb[tok] for tok in self.orig_tokens] if word_to_emb else [0] * len(self.orig_tokens)
        if word_to_char:
            char_ids = [word_to_char[tok] for tok in self.orig_tokens]
        else:
            char_ids = [[0] * config.max_char_len] * len(self.orig_tokens)

        segment_ids = [1] * len(input_ids)
        feat_ids = copy.deepcopy(self.orig_features)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            feat_ids.append([0] * len(feat_ids[0]))
            char_ids.append([0] * len(char_ids[0]))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(feat_ids) == max_seq_length
        assert len(char_ids) == max_seq_length

        positions, ids, weights = \
            list(range(len(self.orig_tokens))), copy.deepcopy(self.orig_tags), [1.0] * len(self.orig_tokens)
        while len(positions) < max_seq_length:
            positions.append(0)
            ids.append(0)
            weights.append(0.0)

        assert len(positions) == max_seq_length
        assert len(ids) == max_seq_length
        assert len(weights) == max_seq_length

        features["input_ids"] = np.array(input_ids, dtype=np.int32)
        features["input_mask"] = np.array(input_mask, dtype=np.int32)
        features["segment_ids"] = np.array(segment_ids, dtype=np.int32)
        features["feat_ids"] = np.array(feat_ids, dtype=np.int32)
        features["char_ids"] = np.array(char_ids, dtype=np.int32)

        for enc_name, file_input in encoders:
            if getattr(config, "use_%s" % enc_name):
                features["%s_emb" % enc_name] = _set_elmo(file_input, portion, self, max_seq_length, reshape=False)

        features["label_positions"] = np.array(positions, dtype=np.int32)
        features["label_ids"] = np.array(ids, dtype=np.int32)
        features["label_weights"] = np.array(weights, dtype=np.float32)

        return features

    def decode_pred(self, predictions, id_to_tag, seq_split="bert"):
        tokens = getattr(self, "%s_tokens" % seq_split)
        tags = getattr(self, "%s_tags" % seq_split)
        guids = getattr(self, "%s_guids" % seq_split)

        sentences, sentence = [], []
        prv_id = guids[0]
        for idx, (word, tag, pred, guid) in enumerate(zip(tokens, tags, predictions[:len(tags)], guids)):
            if seq_split == "bert":
                # if seq_split is "bert" drop intermidiare pieces
                if idx not in self.orig_to_tok_map:
                    continue
                # if it's first piece than get original token
                word = self.orig_tokens[self.orig_to_tok_map.index(idx)]

            tag, pred = id_to_tag[tag], id_to_tag[pred]
            line = " ".join([word, tag, pred])
            if guid == prv_id:
                sentence.append(line)

            elif sentence:
                # remove "[SEP]" tokens from the end of the sentence
                sentences.append("\n".join(sentence[:-1]))
                sentence = [line]

            prv_id = guid

        if sentence:
            sentences.append("\n".join(sentence[:-1]))

        return "\n\n".join(sentences)

    def decode_feature(self, features):
        guids = self.bert_guids
        tokens = defaultdict(list)

        for idx in self.orig_to_tok_map:
            guid = guids[idx]
            doc_id, sent_num = guid.split("_")
            feat = features[idx,...]
            tokens[(int(doc_id), int(sent_num))].append(feat)

        tokens = sorted(tokens.items(), key=lambda x: x[0])
        sentences = []

        for (doc_id, sent_num), tok_lst in tokens:
            sentences.append((doc_id, sent_num, np.stack(tok_lst, 0)))

        return sentences

    def len(self, seq_split="bert"):
        return len(getattr(self, "%s_tokens" % seq_split))



###################
##  Tags Methods ##
###################
def get_tag_map(files, use_iobul=False):
    tags = []
    for filename in files:
        tags += [line.strip().split()[-1] for line in open(filename, 'r').readlines() if line.strip() and " " in line]
    tags = [tag for tag, _ in Counter(tags).most_common()]
    tag_to_id = _convert_to_iobul(tags) if use_iobul else dict([(tag, idx) for idx, tag in enumerate(tags)])
    id_to_tag = _iobul_to_iob(tag_to_id)

    return tag_to_id, id_to_tag


def iob_to_iob2(tags):
    prev = "O"

    for i in range(len(tags)):
        tag = tags[i].replace("B-", "").replace("I-", "")
        if tags[i].startswith("I-") and not prev.endswith("-"+tag):
            tags[i] = "B-"+tag
        prev = tags[i]

    return tags


def _iobul_to_iob(tag_to_id):
    id_to_tag = {}

    for tag, id in tag_to_id.items():
        if tag.startswith('U-'):
            id_to_tag[id] = re.sub(r'^U-(.+)$', r'B-\1',tag)
        elif tag.startswith('L-'):
            id_to_tag[id] = re.sub(r'^L-(.+)$', r'I-\1',tag)
        else:
            id_to_tag[id] = tag

    return id_to_tag


def _convert_to_iobul(tags):
    tag_to_id, ul, count = {}, {"B-": "U-", "I-": "L-"}, 0

    for tag in tags:
        tag_to_id[tag] = count
        count += 1
        if tag[:2] in ul:
            tag = re.sub(r'^(%s)' % tag[:2], ul[tag[:2]], tag)
            tag_to_id[tag] = count
            count += 1

    return tag_to_id


def convert_tags(tags, tag_to_id, use_iobul=False):
    tags = iob_to_iob2(tags)
    if not use_iobul:
        return [tag_to_id[tag] for tag in tags]


    tags_lst = []

    for i in range(len(tags)):

        if tags[i] == "O":
            tags_lst.append(0)
            continue

        tag = re.sub(r'^B-|^I-', '', tags[i])
        if i != len(tags) - 1 and tags[i].startswith('B-') and not tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['U-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('B-') and tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['B-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('I-') and tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['I-' + tag])
        elif i != len(tags) - 1 and tags[i].startswith('I-') and not tags[i + 1].startswith('I-'):
            tags_lst.append(tag_to_id['L-' + tag])

        # last index
        elif i == len(tags) - 1 and tags[i].startswith('I-'):
            tags_lst.append(tag_to_id['L-' + tag])
        elif i == len(tags) - 1 and tags[i].startswith('B-'):
            tags_lst.append(tag_to_id['U-' + tag])

    return tags_lst


def replace_parantheses(word):
    word = word.replace('/.', '.')
    dico = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}'}
    return dico[word] if word in dico else word


######################
### Tokens Methods ###
######################
def wp_tokenizer(tokenizer, orig_tokens, orig_tags, orig_features=None):
    """
        orig_tokens = ["John", "Johanson", "'s",  "house"]
        bert_tokens == ["john", "johan", "##son", "'", "s", "house", "[SEP]"]
        orig_to_tok_map == [1, 2, 4, 6]
    """

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    # bert_tokens.append("[CLS]")
    for idx, orig_token in enumerate(orig_tokens):
        orig_to_tok_map.append(len(bert_tokens))
        if orig_token == "[MASK]":
            bert_tokens.extend(orig_token)
        else:
            bert_tokens.extend(tokenizer.tokenize(orig_token))

    # add "[SEP]" at the end of orig_token
    orig_to_tok_map.append(len(bert_tokens))
    orig_tokens.append("[SEP]")
    orig_tags.append(0)
    orig_features.append([0] * len(orig_features[0]))

    # add "[SEP]" at the end of bert_token
    bert_tokens.append("[SEP]")
    bert_tags = [0] * len(bert_tokens)
    bert_features = [[0] * len(orig_features[0])] * len(bert_tokens) if orig_features else None
    # print(orig_tokens)
    # print(orig_tags)

    # print(len(orig_tokens), len(orig_tags), len(bert_tags))
    for idx, val in enumerate(orig_to_tok_map):
        bert_tags[val] = orig_tags[idx]
        if orig_features:
            bert_features[val] = orig_features[idx]

    return bert_tokens, orig_to_tok_map, bert_tags, bert_features


def normalize_token(token):
    """Convert some tokens in ontonotes"""
    token = _strip_accents(token)
    dico = {'-LRB-': '(', '-RRB-': ')',
            '-LSB-': '[', '-RSB-': ']',
            '-LCB-': '{', '-RCB-': '}',
            '/.': '.', '/?': '?', '/-': '-'}

    if not token:
        print("Replacing empty token by -")
        token = "-"
    return dico[token] if token in dico else token


def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def shape_feature(s, b_idx=0):
    """
    Capitalization feature:
    0 = all upper
    1 = first letter upper
    2 = one capital (not first letter)
    3 = punctuation
    4 = numeric
    5 = contain no alphanum
    6 = all lower
    """

    if s in list(string.punctuation) or s in ["``", "''"]:return 3 + b_idx

    try:
        float(s)
        return 4 + b_idx
    except ValueError:
        pass

    if not s.isalnum():return 5 + b_idx
    if s.lower() == s:return 6 + b_idx
    if s.upper() == s:return 0 + b_idx
    if s[0].upper() == s[0]:return 1 + b_idx
    return 2 + b_idx


def get_char_maping(word_to_id, max_char_len=32):

    words = [*word_to_id]
    char_to_id, _ = zip(*Counter("".join(words)).most_common())
    char_to_id = dict([(v, k + 2) for k, v in enumerate(char_to_id[:86])])
    char_to_id = {**char_to_id, **{"<PAD>": 0, "<OOV>": 1}}

    # get word_to_char_features
    word_to_char = {}#np.zeros((len(word_to_id), max_char_len), np.int32)
    for word in words:
        lst = []

        for c in "<" + word + ">":
            c = char_to_id[c] if c in char_to_id else char_to_id["<OOV>"]
            lst.append(c)

        lst = lst[:max_char_len]

        if len(lst) < max_char_len:
            pad = [0] * (max_char_len - len(lst))
            pad_left, pad_right = pad[:len(pad)//2], pad[len(pad)//2:]
            lst = pad_left + lst + pad_right

        word_to_char[word] = lst + [0] * (max_char_len - len(lst))

    return word_to_char


def _create_int_feature(values):
    if isinstance(values[0], list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=
                                                              np.array(values, dtype=np.int64).reshape(-1).tolist()))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _create_float_feature(values):
    if not isinstance(values, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=values.reshape(-1).tolist()))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))


##################################
### Methods to merge instances ###
##################################
def split_long_sent(tokenizer, tokens, tags, shapes):
    """This method is never used"""
    max_len = 384
    lst = [tokenizer.tokenize(w) for w in tokens]
    lst = [item for sublist in lst for item in sublist]
    output = []

    if len(lst) >= max_len * 2:
        # print(len(lst))
        idx = len(tokens) // 3
        output.append({"orig_tokens": tokens[:idx], "orig_tags": tags[:idx], "orig_features": shapes[:idx]})
        output.append({"orig_tokens": tokens[idx:idx*2], "orig_tags": tags[idx:idx*2], "orig_features": shapes[idx:idx*2]})
        output.append({"orig_tokens": tokens[idx*2:], "orig_tags": tags[idx*2:], "orig_features": shapes[idx*2:]})

    elif len(lst) >= max_len:
        # print(len(lst))
        idx = len(tokens) // 2
        output.append({"orig_tokens": tokens[:idx], "orig_tags": tags[:idx], "orig_features": shapes[:idx]})
        output.append({"orig_tokens": tokens[idx:], "orig_tags": tags[idx:], "orig_features": shapes[idx:]})

    else:
        output.append({"orig_tokens": tokens, "orig_tags": tags, "orig_features": shapes})

    return output


def doc_level_aggregation(inst_lst, max_seq_length, seq_split="bert", doc_agr=True, consecutif=False):
    # if max_seq_length is set to None means that
    # we predict sentence by row so return instances as it
    if not max_seq_length or not doc_agr:
        return inst_lst

    rm = set()
    for i in range(len(inst_lst)):
        if i in rm:
            continue

        for j in range(i+1, len(inst_lst)):
            if j in rm:
                continue

            inst1, inst2 = inst_lst[i], inst_lst[j]
            if inst1.len(seq_split) + inst2.len(seq_split) < max_seq_length:
                inst1.merge(inst2)
                rm.add(j)
            elif consecutif:
                break

    return [inst for idx, inst in enumerate(inst_lst) if idx not in rm]


def dataset_level_aggregation(docs, max_seq_length, data_agr=True, seq_split="bert"):
    """ Merge 2 docs only if each contains one instance
        and thier sum is less than max_seq_length
    """

    docs_idx = [idx for idx, inst_lst in enumerate(docs) if len(inst_lst) == 1 and data_agr]

    rm = set()
    for i in range(len(docs_idx)):
        if docs_idx[i] in rm:
            continue
        for j in range(i + 1, len(docs_idx)):
            if docs_idx[j]in rm:
                continue

            inst1, inst2 = docs[docs_idx[i]][0], docs[docs_idx[j]][0]

            if inst1.len(seq_split) + inst2.len(seq_split) < max_seq_length:
                inst1.merge(inst2)
                rm.add(docs_idx[j])
                docs[docs_idx[i]] = [inst1]

    docs = [doc for idx, doc in enumerate(docs) if idx not in rm]

    # construct a list of input examples
    instances = [item for sublist in docs for item in sublist]
    return instances


#########################
### Word EMB Methods ####
#########################
def get_embeddings(config, dataset):
    vocab = []
    for name, data in dataset.items():
        vocab += data.get_word_list()

    vocab = set(vocab)
    word_to_emb, emb_matrix, word_to_char_mapping = get_emb_data(config, vocab)
    fix_emb_matrix = get_distance_embedings(config, word_to_emb)

    return emb_matrix, word_to_emb, word_to_char_mapping, fix_emb_matrix


def get_emb_data(config, data_vocab):
    # load pre-trained embeddings
    pretrained = []
    words = []
    emb_size = 100
    with open(config["wemb_file"], 'r') as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) > 2:
                words.append(vals[0])
                pretrained.append([float(x) for x in vals[1:]])
                emb_size = len(pretrained[-1])

    emb_vocab = {k: v for v, k in enumerate(words)}
    # '[SEP]' is the end sentence embeding in '</s>' SSKIP
    emb_vocab['[SEP]'] = emb_vocab['</s>']

    # get embedding vocab and matrix
    vocab_out = set()
    word_mapping = {}

    for word in data_vocab:
        if word in emb_vocab:
            word_mapping[word] = emb_vocab[word]
        elif word.lower() in emb_vocab:
            word_mapping[word] = emb_vocab[word.lower()]
        elif re.sub('\d', '0', word.lower()) in emb_vocab:
            word_mapping[word] = emb_vocab[re.sub('\d', '0', word.lower())]
        else:
            vocab_out.add(word)

    print("Embeddings coverage: %2.2f%%" % ((1 - (len(vocab_out) / len(data_vocab))) * 100))

    word_to_emb = {}
    # also use '[SEP]' as pad . It's ignored anyway ||| add embedding to unk
    vectors = [pretrained[emb_vocab['[SEP]']],
               np.random.uniform(-(3. / emb_size) ** .5, (3. / emb_size) ** .5, (emb_size)).tolist(),
               np.random.uniform(-(3. / emb_size) ** .5, (3. / emb_size) ** .5, (emb_size)).tolist()]
    word_to_emb["[MASK]"] = 2

    for w, idx in word_mapping.items():
        word_to_emb[w] = len(vectors)
        vectors.append(pretrained[idx])

    for w in vocab_out:
        word_to_emb[w] = 1

    vectors = np.asarray(vectors)

    word_to_char_mapping = get_char_maping(word_to_emb)

    return word_to_emb, vectors, word_to_char_mapping


def get_distance_embedings(config, word_to_id):
    embedding_matrix = fasttext.load_model(config["ls_model_file"])
    id_to_tag = {v: k for k, v in joblib.load(config["figer_tag_file"]).items()}
    tags_vectors = {key: np.asarray(embedding_matrix[value]) for key, value in id_to_tag.items()}
    tags_norm_vectors = {key: value / np.linalg.norm(value, ord=2) for key, value in tags_vectors.items()}

    cosine = np.zeros((len(word_to_id), len(id_to_tag) + 2), dtype=np.float16)

    for key, value in word_to_id.items():
        rank = np.asarray(embedding_matrix[normalize_token(key).lower()])
        rank = _most_similar_tag(rank, tags_norm_vectors, 'cosine')
        min_v, max_v = np.amin(rank), np.amax(rank)
        vec = -1 + 2. * (rank - min_v) / (max_v - min_v)
        cosine[value] = np.append(vec, np.append(max_v, min_v))

    return cosine


def _most_similar_tag(vec, vec_lst, metric):
    vec_norm = vec / np.linalg.norm(vec, ord=2)
    out = []

    for i in range(len(vec_lst)):
        if metric == 'cosine':
            out.append(np.dot(vec_lst[i], vec_norm))
        elif metric == 'euclidean':
            out.append(np.linalg.norm(vec_lst[i] - vec))
    return np.asarray(out, dtype=np.float16)


def _set_elmo(input_file, name, instance, max_seq_length, reshape=True):

    def get_key_idx(orig_guids, key):
        begin = -1
        for idx, guid in enumerate(orig_guids):
            if guid == key and begin == -1:
                begin = idx
            elif guid != key and begin != -1 or idx == len(orig_guids)-1:
                return begin, idx

        raise ValueError("Something Wrong!!!!!")

    doc_lst = [list(map(int, guid.split("_"))) for guid in instance.orig_guids]
    arr = None
    for doc_id, sent_lst in itertools.groupby(doc_lst, lambda x: x[0]):
        sent_lst = sorted(set([sent_num for _, sent_num in sent_lst]))
        file_key = "%s_%s" %(name, doc_id)
        group = input_file[file_key]

        _, lm_size, lm_layers = group[str(0)][...].shape
        arr = np.zeros([max_seq_length, lm_size, lm_layers], dtype=np.float32)
        for sent_num in sent_lst:
            begin, end = get_key_idx(instance.orig_guids, "%s_%s" % (doc_id, sent_num))
            if end - begin == group[str(sent_num)].shape[0]-1:
                arr[begin:end, ...] = group[str(sent_num)][:-1,...]
            else:
                arr[begin:end, ...] = group[str(sent_num)][...]

    # if arr.shape[-1] > arr.shape[-2]:
    #     arr = np.einsum('lij->lji', arr)
    if reshape:
        arr = arr.reshape([max_seq_length, -1])

    return arr

#########################
### Ontonotes Methods ###
#########################
def load_onto_file(filename):
    words = []
    tags = []

    sent_words = []
    tags_gold = []

    with open(filename) as data_file:
        for line in data_file:
            if line.strip():
                vals = line.strip().split()
                if vals[0] in ['#begin', '#end']:
                    continue

                words.append(replace_parantheses(vals[3]))
                tags.append(vals[10])
            elif len(words) > 0:
                tags = transform_onto_tags(tags)
                sent_words.append(copy.deepcopy(words))
                tags_gold.append(copy.deepcopy(tags))

                words = []
                tags = []

    return sent_words, tags_gold


def transform_onto_tags(lst):
    tags = ["O"] * len(lst)
    flag = False
    cur = "O"

    for i in range(len(lst)):
        if lst[i][0] == "(" and not flag:
            cur = lst[i].replace("(", "").replace(")", "").replace("*", "")
            tags[i] = "B-" + cur

            if lst[i][-1] != ")":
                flag = True

        elif flag and lst[i].startswith("*"):
            tags[i] = "I-" + cur
            if lst[i][-1] == ")":
                flag = False

    return tags