"""A ChatML-style stub tokenizer (one id per character, special tokens as single ids)
so data tests need no model files."""

START, END = 1, 2     # <|im_start|>, <|im_end|>
EOS = {END}


class StubTokenizer:
    def _enc(self, s):
        return [ord(c) + 10 for c in s]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kw):
        out = []
        for m in messages:
            out += [START] + self._enc(m["role"] + "\n" + m["content"]) + [END] + self._enc("\n")
        if add_generation_prompt:
            out += [START] + self._enc("assistant\n")
        return out

    def decode(self, ids):
        return "".join(chr(i - 10) if i >= 10 else "" for i in ids)
