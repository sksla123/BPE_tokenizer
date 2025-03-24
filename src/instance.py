from collections import Counter
from .token import tokenize
from .vocab import Vocabulary
from .util import strip_token

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# Instace 클래스 정의
class Instance:
    word_to_instance = {}
    is_word_available_creating_bigram = {}
    bigram_counter = Counter()

    @classmethod
    def clear_static_variables(cls):
        cls.word_to_instance = {}
        cls.bigram_counter = Counter()
        cls.is_word_available_creating_bigram = {}

    def __init__(self, instance: str, instance_count: int):
        self.word = instance
        self.__class__.word_to_instance[instance] = self

        logger.debug(f"{self.word} 인스턴스 생성")

        # 최초 토큰화(알파벳 단위)
        self.tokens = ['[subword]' + token if i > 0 else '[word]' + token for i, token in enumerate(self.word)]
        logger.debug(f"{self.word}의 최초 토큰화 결과 -> {self.tokens}")

        self.token_count = len(self.tokens)

        ## 내부적으로 토큰 수가 1개라면 더 이상 바이그램 생성이 불가능
        if self.token_count == 1:
            self.__class__.is_word_available_creating_bigram[self.word] = True
        else:
            self.__class__.is_word_available_creating_bigram[self.word] = False

        self.instance_count = instance_count

        self.bigrams = self._create_bigrams()
        self.bigram_count = self._count_bigrams(self.bigrams)

        self.reverse_indexing()

    def tokenize(self, vocab: Vocabulary, mode: str):
        '''
        vocab (Vocabulary): 어휘 집합
        mode (str): "train" or "test"

        return (list): 토큰화된 인스턴스
        '''

        self.tokens = tokenize(self.word, vocab, mode)
        self.token_count = len(self.tokens)

        old_bigrams = self.bigrams
        old_bigram_count = self.bigram_count

        self.bigrams = self._create_bigrams()
        self.bigram_count = self._count_bigrams(self.bigrams)

        self.__class__.bigram_counter -= old_bigram_count
        self.__class__.bigram_counter += self.bigram_count

        return self.token_count

    def get_tokens(self):
        return self.tokens

    def _count_subword(self, subword: str):
        '''
        subword (str): 토큰 목록

        return (Counter): 토큰 목록과 각 토큰의 등장 횟수
        '''

        # 토큰 목록과 각 토큰의 등장 횟수 반환
        # self.word.count(subword) * self.instance_count 하는 이유: subword가 단어 안에서 여러 번 반복될 수 있기 때문

        # Counter 생성과 값 조정을 한 번에 수행
        counter = Counter({subword: self.word.count(subword) * self.instance_count})

        return counter

    def count_token(self):
        '''
        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''
        return self._count_subword(self.tokens)
    
    def _count_bigrams(self, bigrams: list):
        '''
        bigrams (list): [bigram1, bigram2, ...] 형식의 토큰 쌍 목록

        return (list): [(bigram1, count1), (bigram2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._get_token_count(bigrams)

    def _create_bigrams(self):
        '''
        현재 토큰을 가지고 인스턴스 내부에서 인접 토큰들과 연결하여, 임의의 토큰 쌍(bigram)을 생성하는 함수

        return (list): [pair1, pair2, ...] 형식의 토큰 쌍 목록
        '''
        tokens = [strip_token(token) for token in self.tokens]
        logger.debug(f"{self.word} 인스턴스의 토큰 목록: {tokens}")

        bigrams = []
        for i in range(len(tokens) - 1):
            if i == 0:
                bigrams.append('[word]' + tokens[i] + tokens[i + 1])
            else:
                bigrams.append('[subword]' + tokens[i] + tokens[i + 1])       
        # logger.debug(f"{self.word} 인스턴스의 인접 토큰 쌍 (bigrams): {bigrams}")

        return bigrams
    
    def create_bigrams(self):
        '''

        '''
