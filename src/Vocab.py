from .trie import Trie

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# 어휘 집합 클래스
class Vobaulary:
    '''
    어휘 집합을 관리하는 클래스
    '''
    def __init__(self):
        '''
            vocab (list): 기본 어휘 집합
        '''
        self.word_vocab = []
        self.subword_vocab = []

        self.word_dict = Trie()
        self.subword_dict = Trie()

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
    
    def get_token(self, word: str, to: str):
        '''
            word (str): 토큰을 찾을 단어
            to (str): 토큰을 찾을 곳
        '''
        ## 에러 처리 (한 번 호되게 당함..)
        if to not in ["word", "subword"]:
            raise ValueError(f"word 또는 subword 중 하나를 입력해주세요.")

        if to == "word":
            return self.word_dict.get_token(word)
        elif to == "subword":
            return self.subword_dict.get_token(word)
    
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
    
    def add(self, token: str, to: str):
        '''
            token (str): 추가할 토큰
            to (str): 토큰을 추가할 곳
        '''
        ## 에러 처리 (한 번 호되게 당함..)
        if to not in ["word", "subword"]:
            raise ValueError(f"word 또는 subword 중 하나를 입력해주세요.")

        if to == "word":
            self.word_vocab.append(token)
            self.word_dict.insert(token)
        elif to == "subword":
            self.subword_vocab.append(token)
            self.subword_dict.insert(token)

## trie 자료구조 테스트용 코드
def main():
    word_vocab_input = ["hello", "world", "hello_world"]
    subword_vocab_input = ["he", "ll", "o", "wo", "rld", "hel", "lo_", "wor", "ld"]

    vocab = Vobaulary()
    vocab.set_vocab(word_vocab_input, "word")
    vocab.set_vocab(subword_vocab_input, "subword")
    
    print(vocab)

    vocab.add("hi", "word")
    print(vocab)

    print("문자열 hello의 word 토큰: ", vocab.get_token("hello", "word"))
    print("문자열 hello의 subword 토큰: ", vocab.get_token("hello", "subword"))

if __name__ == "__main__":
    main()