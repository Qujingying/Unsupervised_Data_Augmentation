'''
Author : Bastien van Delft

'''


# Import
from argparse import ArgumentParser, FileType
import numpy as np
from tqdm import tqdm, trange
import pickle
from googletrans import Translator
import torch
import torch.nn.functional as F
from uda_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
from pathlib import Path
import spacy
from uda_bert.optimization import BertAdam
from uda_bert import BertForSequenceClassification,BertForPreTraining
from torch.nn import CrossEntropyLoss, NLLLoss, BCEWithLogitsLoss
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
writer = SummaryWriter('./log')


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randint(rand_start, rand_end-1) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = current_idx + randint(1, len(self.doc_lengths)-1)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
### Take as input a text



### Split it into sentences

#Use spacy and prepare all text for bert that produce document separated by \n\n where one line is one sentence

def split_into_sentences(input_folder: str, output_file: FileType, trim: bool):
    nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'tagger', 'ner', 'textcat'])
    nlp.max_length = 2000000
    text_to_write = []
    for file in tqdm(Path(input_folder).glob("*.txt")):
        # print(file)
        with open(file, 'r') as f:
            raw_text = f.read()
        text = raw_text.replace('\n\n', ' ')
        # print(text)
        doc = nlp(text)

        if trim:
            sentences = [sent.string.strip() for sent in doc.sents if len(sent.string.strip())>15]
        else:
            sentences = [sent.string.strip() for sent in doc.sents]
        text_to_write.append('\n'.join(sentences))
    output_file.write('\n\n'.join(text_to_write))
    output_file.close()






### Translate each sentence

def prepare_with_back_translate(text, translator, selected_lang, target_lang, epochs_to_generate,output_dir):

    with DocumentDatabase() as docs:

        with open(text.name, 'r') as f:
            liste = f.readlines()
            doc = []
            doc_translated = []
            for line in tqdm(liste, desc="Loading Dataset", unit=" lines"):

                line = line.strip()
                if line == "":
                    docs.add_document(np.array([doc, doc_translated]).T)
                    doc = []
                    doc_translated = []
                else:
                    # print('original: ', line)
                    translation = translator.translate(line, src=selected_lang, dest=target_lang)
                    back_translation = translator.translate(translation.text, src=target_lang, dest=selected_lang)
                    # tokens = tokenizer.tokenize(line)
                    doc.append(line)
                    # tokens_bis = tokenizer.tokenize(back_translation.text)
                    # print('new: ', back_translation.text)
                    doc_translated.append(back_translation.text)

    output_dir.mkdir(exist_ok=True)
    # for epoch in trange(epochs_to_generate, desc="Epoch"):
    #     epoch_filename = output_dir / f"epoch_{epoch}_unsup.json"
    #     num_instances = 0
    docs_instances_unsup = []
    # with epoch_filename.open('w') as epoch_file:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for doc_idx in trange(len(docs), desc="Document"):
        # print(docs[doc_idx])
        instance = convert_examplesUDA_to_features(docs[doc_idx],max_seq_length = 512, tokenizer = tokenizer,output_mode = "UDA")
        docs_instances_unsup.append(instance)
    docs_instances_unsup = np.array(docs_instances_unsup).reshape(-1)
    pickle.dump(docs_instances_unsup,open('data_unsup.p','wb'))
    return docs_instances_unsup
                # instance = {
                #     "tokens": tokens,
                #     "segment_ids": segment_ids,
                #     "is_random_next": is_random_next,
                #     "masked_lm_positions": masked_lm_positions,
                #     "masked_lm_labels": masked_lm_labels}



        #         doc_instances = create_instances_from_document(
        #             docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
        #             masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
        #             vocab_list=vocab_list)
        #         doc_instances = [json.dumps(instance) for instance in doc_instances]
        #         for instance in doc_instances:
        #             epoch_file.write(instance + '\n')
        #             num_instances += 1
        # metrics_file = args.output_dir / f"epoch_{epoch}_metrics_unsup.json"
        # with metrics_file.open('w') as metrics_file:
        #     metrics = {
        #         "num_training_examples": num_instances,
        #         "max_seq_len": args.max_seq_len
        #     }
        #     metrics_file.write(json.dumps(metrics))




def convert_examplesUDA_to_features(examples, max_seq_length,
                                 tokenizer, output_mode,label_list = None):
  """Loads a data file into a list of `InputBatch`s."""

#   label_map = {label : i for i, label in enumerate(label_list)}

  features = []
  for ex_index, example_ in enumerate(examples):
      # if ex_index % 10000 == 0:
      #     print('1000')
      example = example_[0]
      example_2 = example_[1]
      # print(example_, len(example_))
      tokens_a = tokenizer.tokenize(example)

      tokens_b = tokenizer.tokenize(example_2)
#       if example.text_b:
#           tokens_b = tokenizer.tokenize(example.text_b)
#           # Modifies `tokens_a` and `tokens_b` in place so that the total
#           # length is less than the specified length.
#           # Account for [CLS], [SEP], [SEP] with "- 3"
#           _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#       else:
          # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[:(max_seq_length - 2)]
      if len(tokens_b) > max_seq_length - 2:
          tokens_b = tokens_b[:(max_seq_length - 2)]

      # the entire model is fine-tuned.
      tokens1 = ["[CLS]"] + tokens_a + ["[SEP]"]
      segment_ids1 = [0] * len(tokens1)
      tokens2 = ["[CLS]"] + tokens_b + ["[SEP]"]
      segment_ids2 = [0] * len(tokens2)
      #


      input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
      input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask1 = [1] * len(input_ids1)
      input_mask2 = [1] * len(input_ids2)
      # Zero-pad up to the sequence length.
      padding1 = [0] * (max_seq_length - len(input_ids1))
      padding2 = [0] * (max_seq_length - len(input_ids2))
      input_ids1 += padding1
      input_mask1 += padding1
      segment_ids1 += padding1

      input_ids2 += padding2
      input_mask2 += padding2
      segment_ids2 += padding2

      assert len(input_ids1) == max_seq_length
      assert len(input_mask1) == max_seq_length
      assert len(segment_ids1) == max_seq_length
      assert len(input_ids2) == max_seq_length
      assert len(input_mask2) == max_seq_length
      assert len(segment_ids2) == max_seq_length

      if output_mode == "classification":
          label_id = label_list[ex_index]
      elif output_mode == "regression":
          label_id = float(label_list[ex_index])
      elif output_mode == "UDA":
          label_id = None
      else:
          raise KeyError(output_mode)


      features.append(InputFeatures(input_ids=[input_ids1,input_ids2],
                            input_mask=[input_mask1,input_mask2],
                            segment_ids=[segment_ids1,segment_ids2],
                            label_id=label_id))
  return features


def convertLABEL_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
  """Loads a data file into a list of `InputBatch`s."""

#   label_map = {label : i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):



      tokens_a = tokenizer.tokenize(example)

      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[:(max_seq_length - 2)]

      tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
      segment_ids = [0] * len(tokens)


      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      if output_mode == "classification":
          label_id = label_list[ex_index]
      elif output_mode == "regression":
          label_id = float(label_list[ex_index])
      else:
          raise KeyError(output_mode)


      features.append(InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id))
  return features
### Run Bert on that new text


def main():

    parser = ArgumentParser()
    parser.add_argument('--no_preprocessing', '-n', action='store_true')
    parser.add_argument('--input_folder',type=Path,default='processed_texts')
    parser.add_argument('--output_file_pretranslation',type=FileType(mode='w+', encoding='utf-8'),default='all_texts.txt')
    parser.add_argument('--trim',action='store_true')
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Size of batch")
    parser.add_argument("--selected_lang", "-s", type=str, default='en', help="Source language")
    parser.add_argument("--target_lang", "-t", type=str, default='fr', help="Target language")
    # parser.add_argument('--ouput_dir',type=FileType(mode='w', encoding='utf-8'),default='all_texts.txt')
    parser.add_argument("--output_dir", type=Path, default='training/')
    parser.add_argument('--translator', default = 'google', help = 'Choose which translator to use : bing, google or deepl')
    parser.add_argument('--num_labels', type = int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels).to(device)


    if args.translator == 'deepl':
        translator = DeeplTranslator()
    elif args.translator == 'bing':
        translator = BingTranslator()
    else:
        translator = Translator()

    if args.no_preprocessing != True:
        print('split into sentences')
        split_into_sentences(input_folder= args.input_folder, output_file = args.output_file_pretranslation, trim= args.trim)
        print('back_translation')
        train_features = prepare_with_back_translate(args.output_file_pretranslation, translator = translator, selected_lang = args.selected_lang, target_lang = args.target_lang, epochs_to_generate = args.epochs, output_dir = args.output_dir)
    else:
        train_features = pickle.load(open('data_unsup.p','rb'))
    original_input_ids = torch.tensor([f.input_ids[0] for f in train_features], dtype=torch.long)
    original_input_mask = torch.tensor([f.input_mask[0] for f in train_features], dtype=torch.long)
    original_segment_ids = torch.tensor([f.segment_ids[0] for f in train_features], dtype=torch.long)

    augmented_input_ids = torch.tensor([f.input_ids[1] for f in train_features], dtype=torch.long)
    augmented_input_mask = torch.tensor([f.input_mask[1] for f in train_features], dtype=torch.long)
    augmented_segment_ids = torch.tensor([f.segment_ids[1] for f in train_features], dtype=torch.long)

    # all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(original_input_ids, original_input_mask, original_segment_ids, augmented_input_ids,augmented_input_mask,augmented_segment_ids)

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [int(len(train_data) * 0.95),len(train_data) - int(len(train_data) * 0.95)])

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    param_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,lr=3 * 10e-5,warmup=-1,t_total=12000)

    loss_function = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')
    global_step = 0
    import os
    # print('cwd', os.getcwd())
    for epoch in tqdm(range(args.epochs)):
        writer.add_scalar('lol', epoch)
        model.train()
        print('train')
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            original_input_ids, original_input_mask, original_segment_ids, augmented_input_ids,augmented_input_mask,augmented_segment_ids = batch
            with torch.no_grad():
                logits_original = model(original_input_ids)  # , segment_ids, input_mask, labels=None)
            logits_augmented = model(augmented_input_ids)
            # print(logits_original)
            # print(logits_augmented)
            loss = loss_function(logits_augmented, logits_original)

            writer.add_scalar('KL_loss', loss.item(), global_step)
            loss.backward()
            # print('loss ', loss)
            optimizer.step()
            global_step += 1



if __name__ == '__main__':
    main()
    writer.close()

