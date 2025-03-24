from collections import Counter
from .token import tokenize, Token
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

    updated_instances = [] ## 나중에 멀티프로세싱을 염두에 두고 만든 변수

    @classmethod
    def clear_static_variables(cls):
        cls.word_to_instance = {}
        cls.bigram_counter = Counter()
        cls.is_word_available_creating_bigram = {}
        cls.updated_instances = []
    
    def __init__(self, instance_string: str, instance_count: int):
        self.word = instance_string
        self.__class__.word_to_instance[self.word] = self

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

        self.bigrams = self.create_bigrams()
        self.bigram_count = self.count_bigrams()

    def tokenize(self, vocab: Vocabulary, mode: str):
        '''
        vocab (Vocabulary): 어휘 집합
        mode (str): "train" or "test"

        return (list): 토큰화된 인스턴스
        '''

        self.tokens = tokenize(self.word, vocab, mode)
        self.token_count = len(self.tokens)

        ## bigrams 업데이트
        old_bigram_count = self.bigram_count

        self.bigrams = self._create_bigrams()
        self.bigram_count = self._count_bigrams(self.bigrams)

        ## bigram_counter 변화량
        delta_bigram_count = self.bigram_count.subtract(old_bigram_count)

        ## 전체 bigram_counter 업데이트
        self.__class__.bigram_counter -= delta_bigram_count

        ## 업데이트된 인스턴스 목록에 추가
        self.__class__.updated_instances.append(self.word)

        return delta_bigram_count ## 추후 멀티프로세싱을 염두에 두고 반환

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

    def count_tokens(self):
        '''
        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''
        return self._count_subword(self.tokens)
    
    def count_bigrams(self):
        '''
        bigrams (list): [bigram1, bigram2, ...] 형식의 토큰 쌍 목록

        return (list): [(bigram1, count1), (bigram2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._count_subword(self.bigrams)

    def create_bigrams(self):
        '''
        현재 토큰을 가지고 인스턴스 내부에서 인접 토큰들과 연결하여, 임의의 토큰 쌍(bigram)을 생성하는 함수

        return (list): [pair1, pair2, ...] 형식의 토큰 쌍 목록
        '''
        logger.debug(f"{self.word} 인스턴스의 토큰 목록: {self.tokens}")

        bigrams = []
        for i in range(len(self.tokens) - 1):
            if i == 0:
                bigrams.append(Token(self.tokens[i].string + self.tokens[i + 1].string, False))
            else:
                bigrams.append(Token(self.tokens[i].string + self.tokens[i + 1].string, True))

        return bigrams

    def _count_bigrams(self):
        return self._count_subword(self.bigrams)
    
    def create_bigrams(self):
        '''

        '''
