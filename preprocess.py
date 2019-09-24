from utils import *


config = pyhocon.ConfigFactory.parse_file("experiments.conf")


##########################
### Dataset to Tfrecords##
##########################
def load_dataset(ds_name):
    output_path = os.path.join(config["data_dir"], ds_name)

    with open(os.path.join(output_path, "dataset.pkl"), "rb") as data:
        dataset = pickle.load(data)

    emb_matrix = np.load(os.path.join(output_path, "emb_matrix.npy"))
    fix_emb_matrix = np.load(os.path.join(output_path, "fix_emb_matrix.npy"))
    file_dict = dict([(portion, os.path.join(output_path, "%s.tf_record" % portion)) for portion in config["portion"]])

    return dataset, emb_matrix, file_dict, fix_emb_matrix


def write_dataset(ds_name, use_iobul=True):

    output_path = os.path.join(config["data_dir"], ds_name)
    if not os.path.exists(output_path): os.mkdir(output_path)

    print("Loading dataset")
    do_lower_case = "uncased" in config["vocab_file"]
    tokenizer = FullTokenizer(vocab_file=config["vocab_file"], do_lower_case=do_lower_case)
    dataset = read_dataset(ds_name, tokenizer,  use_iobul=use_iobul)

    # set word embeddings index and get pretrained embedding matrix
    print("Loading embedding matrix")
    emb_matrix, word_to_emb, word_to_char, fix_emb_matrix = get_embeddings(config, dataset)

    for portion, data in dataset.items():
        print("Processing %s" % portion)
        for doc_id, instances in enumerate(data.docs):
            instances = doc_level_aggregation(instances, config["max_seq_length"],
                                              seq_split="orig", doc_agr=False, consecutif=False)
            dataset[portion].docs[doc_id] = instances
        dataset[portion].instances = dataset_level_aggregation(dataset[portion].docs, config["max_seq_length"],
                                                               seq_split="orig", data_agr=False)
        dataset[portion].word_to_emb = word_to_emb
        dataset[portion].word_to_char = word_to_char

    pickle.dump(dataset, open(os.path.join(output_path, "dataset.pkl"), "wb"))
    np.save(os.path.join(output_path, "emb_matrix"), emb_matrix)
    np.save(os.path.join(output_path, "fix_emb_matrix"), fix_emb_matrix)


def read_dataset(ds_name, tokenizer, use_iobul=True, use_shape=True):
    # set global attributes
    feature_size = 7 if use_shape else 0
    feature_num = 1 if use_shape else 0
    file_dict = {}
    for portion in config["portion"]:
        file_dict[portion] = os.path.join(config["data_dir"], "%s.%s.iob" % (ds_name, portion))

    tag_to_id, id_to_tag = get_tag_map(file_dict.values(), use_iobul)
    dataset = {}
    max_sent = -1
    for portion, filname in file_dict.items():
        docs = _read_ner(filname, tokenizer, tag_to_id, use_iobul)

        for doc_id, doc in enumerate(docs):
            instances = []
            for sent_num, sentence in enumerate(doc):
                bert_tokens, orig_to_tok_map, bert_tags, bert_features = wp_tokenizer(tokenizer,
                                                                                       sentence["orig_tokens"],
                                                                                       sentence["orig_tags"],
                                                                                       sentence["orig_features"])
                instance = InputExample(doc_id, sent_num,
                                        sentence["orig_tokens"], sentence["orig_tags"], sentence["orig_features"],
                                        bert_tokens, orig_to_tok_map, bert_tags, bert_features)

                if len(bert_tokens) > max_sent:
                    max_sent = len(bert_tokens)
                instances.append(instance)

            docs[doc_id] = instances

        dataset[portion] = Dataset(docs, id_to_tag, feature_size, feature_num)

    # set max_sent to the longest sequence in train/dev/test
    for portion in dataset:
        dataset[portion].max_sent = max_sent

    return dataset


def _read_ner(data_file, tokenizer, tag_to_id, use_iobul=False):
    docs, sentences, tokens, tags, shapes = [], [], [], [], []

    for line in open(data_file, 'r').readlines():
        if line.startswith("-DOCSTART-") and sentences:
            docs.append(sentences)
            sentences = []

        elif line.strip() and not line.startswith("-DOCSTART-"):
            vals = line.strip().split()
            tokens.append(convert_to_unicode(normalize_token(vals[0])))
            shapes.append([shape_feature(tokens[-1])])
            tag = convert_to_unicode(vals[-1])
            tag = tag.replace("\ufeff", "")
            if tag not in tag_to_id:
                raise ValueError("Tag %s not found in tag mapping!!!!!!" % tag)
            tags.append(tag)
        elif tokens:
            tags = convert_tags(tags, tag_to_id, use_iobul)
            assert len(tags) == len(tokens)
            sentences += split_long_sent(tokenizer, tokens, tags, shapes)
            tokens, tags, shapes = [], [], []

    if tokens:
        tags = convert_tags(tags, tag_to_id, use_iobul)
        sentences.append({"orig_tokens": tokens, "orig_tags": tags, "orig_features": shapes})
    if sentences:
        docs.append(sentences)

    return docs


###############################################
#     Read original CoNLL and OntoNotes       #
#     and convert them to tsv iob2 format     #
###############################################
def create_conll_raw(raw_path, portion):
    sent_words = []
    tags_gold = []

    words = []
    tags = []

    with open(os.path.join(raw_path, "conll.%s.txt"% portion)) as data_file:
        for line in data_file:
            if line.strip():
                vals = line.strip().split(" ")
                if vals[0] != "-DOCSTART-":
                    words.append(vals[0])
                    tags.append(vals[-1])

            elif len(words) > 0:
                tags = iob_to_iob2(tags)
                sent_words.append(copy.deepcopy(words))
                tags_gold.append(copy.deepcopy(tags))

                words = []
                tags = []

    output = [zip(x, y) for x, y in zip(sent_words, tags_gold)]
    st = '\n\n'.join(['\n'.join([' '.join(sub_lst) for sub_lst in lst]) for lst in output]) + "\n"

    with open("data/conll.%s.iob" % portion, 'w') as f:
        f.write(st + "\n")


def create_onto_raw(raw_path, portion):
    datafile = os.path.join(raw_path , portion) + "/data/english/annotations/"
    files = [y for x in os.walk(datafile) for y in glob(os.path.join(x[0], '*_gold_conll'))]

    words = []
    tags = []
    dico = defaultdict(int)

    for filename in files:
        if "/pt/nt" in filename:
            continue

        item = load_onto_file(filename)
        span = filename.replace(datafile, '').split('/')[0]
        dico[span] += len(item[0])

    for filename in files:
        if "/pt/nt" in filename:
            continue
        item = load_onto_file(filename)
        words += item[0]
        tags += item[1]

    output = [zip(x, y) for x, y in zip(words, tags)]
    st = '\n\n'.join(['\n'.join([' '.join(sub_lst) for sub_lst in lst]) for lst in output]) + "\n"

    with open("data/ontonotes.%s.iob" % portion, 'w') as f:
        f.write(st + "\n")


#####################
### ELMo Methods ####
#####################
def cache_cont_emb(ds_name):
    output_path = os.path.join(config["data_dir"], ds_name, "dataset.pkl")
    with open(output_path, "rb") as data:
        dataset = pickle.load(data)

    print("Catch elmo")
    cache_elmo(ds_name, dataset)

    print("Catch flair")
    cache_flair(ds_name, dataset)

def cache_elmo(ds_name, dataset):
    token_ph, len_ph, lm_emb = _build_elmo()
    out_path = os.path.join(config["data_dir"], ds_name, "cache_elmo.hdf5")

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        with h5py.File(out_path, "w") as out_file:
            for name, data in dataset.items():
                for doc_id, instances in tqdm.tqdm(enumerate(data.docs)):
                    max_sentence_length = max([len(instance.orig_tokens) for instance in instances])
                    tokens = [[""] * max_sentence_length for _ in range(len(instances))]

                    # -1 because the orig tokens ends with "[SEP]"
                    text_len = np.array([len(inst.orig_tokens)-1 for inst in instances])

                    for i, instance in enumerate(instances):
                        for j in range(len(instance.orig_tokens)):
                            tokens[i][j] = instance.orig_tokens[j]

                    tokens = np.array(tokens)
                    file_key = "%s_%s" % (name, doc_id)
                    group = out_file.create_group(file_key)

                    if len(data.docs) == 1:
                        bs = 32

                        for i in tqdm.tqdm(range(0, tokens.shape[0], bs)):
                            j = i+bs if i+bs < tokens.shape[0] else tokens.shape[0]
                            tf_lm_emb = session.run(lm_emb, feed_dict={token_ph: tokens[i:j], len_ph: text_len[i:j]})

                            for k, (e, l) in enumerate(zip(tf_lm_emb, text_len[i:j])):
                                # +1 to add dummy vec for ["SEP"]
                                e = e[:l + 1, :, :]
                                group[str(k+i)] = e
                    else:
                        tf_lm_emb = session.run(lm_emb, feed_dict={token_ph: tokens, len_ph: text_len})
                        for i, (e, l) in enumerate(zip(tf_lm_emb, text_len)):
                            # +1 to add dummy vec for ["SEP"]
                            e = e[:l+1, :, :]
                            group[str(i)] = e

    tf.Session().close()

def _build_elmo():
    token_ph = tf.placeholder(tf.string, [None, None])
    len_ph = tf.placeholder(tf.int32, [None])
    elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
    lm_embeddings = elmo_module(inputs={"tokens": token_ph, "sequence_len": len_ph},
                                signature="tokens",
                                as_dict=True)

    word_emb = lm_embeddings["word_emb"]# [num_sentences, max_sentence_length, 512]
    lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                       lm_embeddings["lstm_outputs1"],
                       lm_embeddings["lstm_outputs2"]], -1)# [num_sentences, max_sentence_length, 1024, 3]

    return token_ph, len_ph, lm_emb


def cache_flair(ds_name, dataset):
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings
    from flair.data import Sentence
    batch_size = 256

    lnum = 2
    out_path = os.path.join(config["data_dir"], ds_name, "cache_flair.hdf5")
    stacked_embeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')])

    with h5py.File(out_path, "w") as out_file:
        for name, data in dataset.items():
            for doc_id, instances in enumerate(tqdm.tqdm(data.docs)):
                file_key = "%s_%s" % (name, doc_id)
                group = out_file.create_group(file_key)
                sentences = []
                b_counter = 0

                for i, instance in enumerate(instances):
                    sentences.append(Sentence(' '.join([tok for tok in instance.orig_tokens[:-1]])))
                    if len(sentences) == batch_size or i == len(instances) -1:
                        stacked_embeddings.embed(sentences)
                        for k in range(len(sentences)):
                            arr = np.array([token.embedding.numpy().reshape(-1) for token in sentences[k]],
                                           dtype=np.float32)
                            arr = arr.reshape([arr.shape[0], -1, lnum])
                            group[str(batch_size*b_counter+k)] = arr

                        b_counter += 1
                        sentences = []






def main(argv):
    if not os.path.exists("models"):
        os.makedirs("models")

    ds_name = argv[0]
    if ds_name == "ontonotes":
        [create_onto_raw(config["raw_path"], p) for p in config["portion"]]

    elif ds_name == "conll":
        [create_conll_raw(config["raw_path"], p) for p in config["portion"]]
    else:
        print("Unknown dataset")
        sys.exit(1)

    write_dataset(ds_name)
    cache_cont_emb(ds_name)



if __name__ == "__main__":
    main(sys.argv[1:])



