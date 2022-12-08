import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class CongressionalHearingExplanationsProcessor(BertProcessor):
    NUM_CLASSES = 6
    IS_MULTILABEL = True

    def __init__(self, config=None, is_expert=False):
        super().__init__()
        self.config = config
        self.column_id = config.id_column
        self.column_label = config.label_column
        self.column_text_a = config.first_input_column
        self.use_text_b = config.use_second_input
        self.column_text_b = config.second_input_column
        self.use_text_c = config.use_third_input
        self.column_text_c = config.third_input_column
        self.use_text_d = config.use_fourth_input
        self.column_text_d = config.fourth_input_column
        if config and config.fold_num >= 0:
            self.NAME = os.path.join('CongressionalHearingFoldsExplanations', 'fold'+str(config.fold_num))
        else:
            self.NAME = 'CongressionalHearingExplanations'

    def get_train_examples(self, data_dir, is_expert=False):
        print('Getting train examples...')
        if is_expert:
            backup_column_text_a = self.column_text_a
            backup_use_text_b = self.use_text_b
            backup_use_text_c = self.use_text_c
            self.column_text_a = self.config.third_input_column
            self.use_text_b = False
            self.use_text_c = False
        else:
            backup_use_text_c = self.use_text_c
            self.use_text_c = False
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'train.tsv')))
        if is_expert:
            self.column_text_a = backup_column_text_a
            self.use_text_b = backup_use_text_b
            self.use_text_c = backup_use_text_c
        else:
            self.use_text_c = backup_use_text_c
        return examples

    def get_dev_examples(self, data_dir, is_expert=False):
        print('Getting dev examples...')
        if is_expert:
            backup_column_text_a = self.column_text_a
            backup_use_text_b = self.use_text_b
            backup_use_text_c = self.use_text_c
            self.column_text_a = self.config.third_input_column
            self.use_text_b = False
            self.use_text_c = False
        else:
            backup_use_text_c = self.use_text_c
            self.use_text_c = False
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'dev.tsv')))
        if is_expert:
            self.column_text_a = backup_column_text_a
            self.use_text_b = backup_use_text_b
            self.use_text_c = backup_use_text_c
        else:
            self.use_text_c = backup_use_text_c
        return examples


    def get_test_examples(self, data_dir, is_expert=False):
        print('Getting test examples...')
        if is_expert:
            backup_column_text_a = self.column_text_a
            backup_use_text_b = self.use_text_b
            backup_use_text_c = self.use_text_c
            self.column_text_a = self.config.third_input_column
            self.use_text_b = False
            self.use_text_c = False
        else:
            backup_use_text_c = self.use_text_c
            self.use_text_c = False
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'test.tsv')))
        if is_expert:
            self.column_text_a = backup_column_text_a
            self.use_text_b = backup_use_text_b
            self.use_text_c = backup_use_text_c
        else:
            self.use_text_c = backup_use_text_c
        return examples 

    def _create_examples(self, lines):
        examples = []
        text_b = None
        text_c = None
        text_d = None
        for (i, line) in enumerate(lines):
            if i == 0:
                print('Gold Label: ', line[self.column_label])
                print('Document Index: ', line[self.column_id])
                print('First Input: ', line[self.column_text_a])
                if self.use_text_b:
                    print('Second Input: ', line[self.column_text_b])
                    if self.use_text_c:
                        print('Third Input: ', line[self.column_text_c])
                        if self.use_text_d:
                            print('Fourth Input: ', line[self.column_text_d])
                continue
            guid = line[self.column_id]
            label = line[self.column_label]
            text_a = line[self.column_text_a]
            if self.use_text_b:
                text_b = line[self.column_text_b]
                if self.use_text_c:
                    text_c = line[self.column_text_c]
                    if self.use_text_d:
                        text_d = line[self.column_text_d]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, text_d=text_d, label=label))
        return examples