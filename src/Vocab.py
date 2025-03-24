from .trie import Trie
from .token import Token

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# 어휘 집합 클래스
class Vocabulary:
    '''
    어휘 집합을 관리하는 클래스
    '''
    def __init__(self, word_vocab: list, subword_vocab: list):
        '''
            word_vocab (list): 기본 어휘 집합
            subword_vocab (list): 기본 어휘 집합
        '''
        self.word_vocab = word_vocab
        self.subword_vocab = subword_vocab

        self.vocab = word_vocab + subword_vocab
        
        self.merge_rules = []

        self.word_dict = Trie()
        self.subword_dict = Trie()

        self.set_vocab(self.word_vocab, "word")
        self.set_vocab(self.subword_vocab, "subword")

    def __str__(self):
        return f"Word Vocabulary: {self.word_vocab}\nSubword Vocabulary: {self.subword_vocab}"

    def __len__(self):
        return len(self.word_vocab) + len(self.subword_vocab)

    def get_vocab(self):
        return self.word_vocab + self.subword_vocab
    
    def get_word_vocab(self):
        return self.word_vocab
    
    def get_subword_vocab(self):
        return self.subword_vocab
    
    def get_token(self, word: str, _from: str, method:str = "longest_matching"):
        '''
            word (str): 토큰을 찾을 단어
            _from (str): 토큰을 찾을 곳
    
            method (str): 토큰을 찾을 방법 ## 아니 첨에 병합규칙이 뭔질 몰라서 그냥 longest matching으로 했는데 병합규칙 으아아악 개 핵 느려서 Trie 구조 개열심히 공부했는데 쓸모 없어지고 그냥 죽고싶다
        '''
        ## 에러 처리 (한 번 호되게 당함..)
        if _from not in ["word", "subword"]:
            raise ValueError(f"word 또는 subword 중 하나를 입력해주세요.")

        # logger.debug(f"[{_from} 사전] {word} 토큰 찾기")
        if _from == "word":
        #     logger.debug(f"[{_from} 사전], {self.word_vocab}")
        #     logger.debug(f"[{_from} 사전] {word} 토큰 찾기 완료")
        #     logger.debug(f"[{_from} 사전] {word} 토큰: {self.word_dict.get_token(word)}")
            return Token(self.word_dict.get_token(word), False)
        elif _from == "subword":
            return Token(self.subword_dict.get_token(word), True)
    
    def set_vocab(self, vocab: list, to: str):
        '''
            vocab (list): 어휘 집합
            to (str): 어휘 집합을 추가할 곳
        '''
        ## 에러 처리 (한 번 호되게 당함..)
        if to not in ["word", "subword"]:
            raise ValueError(f"word 또는 subword 중 하나를 입력해주세요.")

        if to == "word":
            self.word_vocab.extend(vocab)
            for word in vocab:
                self.word_dict.insert(word)
        elif to == "subword":
            self.subword_vocab.extend(vocab)
            for subword in vocab:
                self.subword_dict.insert(subword)
    
    def _add(self, token_string: str, to: str):
        '''
            token_string (str): 추가할 토큰
            to (str): 토큰을 추가할 곳
        '''
        ## 에러 처리 (한 번 호되게 당함..)
        if to not in ["word", "subword"]:
            raise ValueError(f"word 또는 subword 중 하나를 입력해주세요.")

        if to == "word":
            self.word_vocab.append(token_string)
            self.word_dict.insert(token_string)
        elif to == "subword":
            self.subword_vocab.append(token_string)
            self.subword_dict.insert(token_string)

    def add(self, token: Token):
        '''
            token (str): 추가할 토큰
        '''
        self.vocab.append(token.string)
        
        if token.is_sub:
            self._add(token.string, "subword")
        else:
            self._add(token.string, "word")

## trie 자료구조 테스트용 코드
def main():
    word_vocab_input = ["hello", "world", "hello_world"]
    subword_vocab_input = ["he", "ll", "o", "wo", "rld", "hel", "lo_", "wor", "ld"]

    vocab = Vocabulary(word_vocab_input, subword_vocab_input)
    
    print(vocab)

    vocab.add(Token("hi", False))
    print(vocab)

    print("문자열 hello의 word 토큰: ", vocab.get_token("hello", "word"))
    print("문자열 hello의 subword 토큰: ", vocab.get_token("hello", "subword"))

if __name__ == "__main__":
    main()