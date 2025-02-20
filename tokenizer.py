short_tokens = """
 0123456789./*-+=()><
"""

short_tokens += "\r"
short_tokens += "\n"

special_tokens = ["true", "false", ">=", "<=", "->", "<pad>", "<eof>", "<ans>"]

class Encoder:
    id2text = {}
    text2id = {}
    max_id = 0
    def size(self):
        return len(self.id2text)

    def init(self):
        for token in special_tokens:
            self.add_special_token(token)
        self.add_short_tokens(short_tokens)

    def add_special_token(self, token):
            token_id = self.max_id + 1
            self.max_id += 1
            self.id2text[token_id] = token
            self.text2id[token] = token_id

    def add_short_tokens(self, text : str):
        for c in text:
            if c in self.text2id:
                continue
            token_id = self.max_id + 1
            self.max_id = self.max_id + 1
            self.id2text[token_id] = c
            self.text2id[c] = token_id

    def encode(self, text : str):
        result = []
        i = 0
        while i < len(text):
            seq = text[i:]
            skip = False
            for t in special_tokens:
                if seq.startswith(t):
                    i += len(t)
                    result.append(self.text2id[t])
                    skip = True
                    break
            if skip:
                continue
            c = text[i]
            if c in self.text2id:
                result.append(self.text2id[c])
            else:
                raise Exception('unexpected ' + c + '(0x' + c.encode().hex() + ')')
            i += 1
        return result


if __name__ == "main":
    tok = Encoder()
    tok.init()
    tok.encode("9.11>9.9->0.11>0.9->11>90->0>79->false")