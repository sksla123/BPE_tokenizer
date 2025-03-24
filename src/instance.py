from collections import Counter
from .tokenize import tokenize
from .token import Token
from .vocab import Vocabulary

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# Instace 클래스 정의
class Instance:
    word_to_instance = {}
    bigram_counter = Counter()
    bigram_to_instance = {}

    updated_instances = {} ## 나중에 멀티프로세싱을 염두에 두고 만든 변수

    @classmethod
    def init_static_variables(cls):
        cls.word_to_instance = {}
        cls.bigram_counter = Counter()
        cls.bigram_to_instance = {}

        cls.updated_instances = {} ## 나중에 멀티프로세싱을 염두에 두고 만든 변수

    @classmethod
    def update_instances(cls, bigram: Token, vocab: Vocabulary, mode: str):
        '''
        bigram (Token): 토큰 쌍
        vocab (Vocabulary): 어휘 집합
        mode (str): "train" or "test"
        '''
        logger.debug(f"vocab: {vocab.get_vocab()}")

        instances = cls.bigram_to_instance[bigram]
        total = len(instances)
        
        for i, instance in enumerate(instances):
            cls.word_to_instance[instance] = cls.word_to_instance[instance].tokenize(vocab, mode)
            print(f"\r{i+1} / {total}", end="")
        
        print(f"\n{total}개의 인스턴스 토큰화 완료")

    @classmethod
    def init_updated_instances(cls):
        cls.updated_instances = {}
    
    def __init__(self, instance_string: str, instance_count: int):
        self.word = instance_string
        self.__class__.word_to_instance[self.word] = self

        logger.debug(f"{self.word} 인스턴스 생성")

        # 최초 토큰화(알파벳 단위)
        self.tokens = [Token(token_string, True) if i > 0 else Token(token_string, False) for i, token_string in enumerate(self.word)]
        logger.debug(f"{self.word}의 최초 토큰화 결과 -> {self.tokens}")

        self.token_count = len(self.tokens)
        self.instance_count = instance_count

        self.bigrams = None
        self.create_bigrams()
        self.bigram_counter = self.count_bigrams()
        self.__class__.bigram_counter += self.bigram_counter
        
        self.bigram_count_delta = Counter() ## 나중에 멀티프로세싱을 염두에 두고 만든 변수

    def tokenize(self, vocab: Vocabulary, mode: str):
        '''
        토큰화하고 내부에 새로운 바이그램 쌍(후보군)을 생성
        전체 bigram_counter 업데이트

        vocab (Vocabulary): 어휘 집합
        mode (str): "train" or "test"

        return (self): 자기 반환
        '''

        self.tokens = tokenize(self.word, vocab, mode)
        self.token_count = len(self.tokens)

        ## bigrams 업데이트
        old_bigrams = self.bigrams
        # logger.debug(f"old_bigrams: {old_bigrams}")
        old_bigram_counter = self.bigram_counter
        # logger.debug(f"old_bigram_counter: {old_bigram_counter}")

        self.create_bigrams()
        # logger.debug(f"new_bigrams: {self.bigrams}")
        self.bigram_counter = self.count_bigrams()
        # logger.debug(f"new_bigram_counter: {self.bigram_counter}")

        # logger.debug(f"old_bigram_to_instance: {self.__class__.bigram_to_instance[old_bigrams[0]]}")
        ## 역인덱싱 업데이트
        for bigram in old_bigrams:
            self.__class__.bigram_to_instance[bigram].remove(self.word)
        # logger.debug(f"new_bigram_to_instance: {self.__class__.bigram_to_instance[old_bigrams[0]]}")

        ## bigram_counter 변화량
        bigram_counter = self.bigram_counter.copy()
        self.delta_bigram_count = bigram_counter.subtract(old_bigram_counter) ## 추후 멀티프로세싱을 염두에 두고 만든 변수

        ## 전체 bigram_counter 업데이트
        # logger.debug(f"정적 변수 bigram_counter old 값 업데이트 전:\n {self.__class__.bigram_counter}")
        self.__class__.bigram_counter -= old_bigram_counter
        # logger.debug(f"정적 변수 bigram_counter old 값 업데이트 후:\n {self.__class__.bigram_counter}")
        self.__class__.bigram_counter += self.bigram_counter
        # logger.debug(f"정적 변수 bigram_counter 업데이트 후:\n {self.__class__.bigram_counter}")

        ## 업데이트된 인스턴스 목록에 추가
        self.__class__.updated_instances[self.word] = self ## 추후 멀티프로세싱을 염두에 두고 만든 변수

        return self

    def get_tokens(self):
        return self.tokens

    def _count_subwords(self, subwords: list[Token]):
        '''
        subword (str): 토큰 목록

        return (Counter): 토큰 목록과 각 토큰의 등장 횟수
        '''

        # 토큰 목록과 각 토큰의 등장 횟수 반환
        # self.word.count(subword) * self.instance_count 하는 이유: subword가 단어 안에서 여러 번 반복될 수 있기 때문

        # Counter 생성과 값 조정을 한 번에 수행
        counter = Counter({subword: self.word.count(str(subword)) * self.instance_count for subword in subwords})

        return counter

    def count_tokens(self):
        '''
        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''
        return self._count_subwords(self.tokens)
    
    def count_bigrams(self):
        '''
        return (list): [(bigram1, count1), (bigram2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._count_subwords(self.bigrams)
    
    def create_bigrams(self):
        '''
        현재 토큰을 가지고 인스턴스 내부에서 인접 토큰들과 연결하여, 임의의 토큰 쌍(bigram)을 생성하는 함수
        '''

        self.bigrams = []
        for i in range(len(self.tokens) - 1):
            if i == 0:
                self.bigrams.append(Token(self.tokens[i].string + self.tokens[i + 1].string, False))
            else:
                self.bigrams.append(Token(self.tokens[i].string + self.tokens[i + 1].string, True))

        # 역인덱싱 업데이트
        for bigram in self.bigrams:
            if bigram not in self.__class__.bigram_to_instance:
                self.__class__.bigram_to_instance[bigram] = []
            self.__class__.bigram_to_instance[bigram].append(self.word)