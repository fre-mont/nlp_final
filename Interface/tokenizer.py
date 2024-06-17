import sentencepiece as spm
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments


class CustomSPTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_file=tokenizer_path)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

    def _tokenize(self, text):
        return self._tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token):
        return self._tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self._tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep + [self.sep_token_id]

    @property
    def cls_token_id(self):
        return self._tokenizer.token_to_id("<cls>")

    @property
    def sep_token_id(self):
        return self._tokenizer.token_to_id("<sep>")

    @property
    def pad_token_id(self):
        return self._tokenizer.token_to_id("<pad>")

    @property
    def unk_token_id(self):
        return self._tokenizer.token_to_id("<unk>")

