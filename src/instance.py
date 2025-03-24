from collections import Counter
from .tokenize import tokenize
from .vocab import Vocabulary
from .util import strip_token

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# Instace 클래스 정의
class Instance:
    def __init__(self, instance: str, instance_count: int):
        self.word = instance
        logger.debug(f"{self.word} 인스턴스 생성")

        # 최초 토큰화(알파벳 단위)
        self.tokens = ['[subword]' + token if i > 0 else '[word]' + token for i, token in enumerate(self.word)]
        logger.debug(f"{self.word}의 최초 토큰화 결과 -> {self.tokens}")

        self.token_count = len(self.tokens)
        self.instance_count = instance_count

    def tokenize(self, vocab: Vocabulary):
        '''
        vocab (Vocabulary): 어휘 집합

        return (list): 토큰화된 인스턴스 ㅜ
        '''

        self.tokens = tokenize(self.word, vocab)
        self.token_count = len(self.tokens)

        return self.token_count

    def get_tokens(self):
        return self.tokens

    def _get_token_count(self, tokens: str):
        '''
        tokens (str): 토큰 목록

        return (Counter): 토큰 목록과 각 토큰의 등장 횟수
        '''

        # 토큰 목록과 각 토큰의 등장 횟수 반환
        # self.word.count(token) * self.instance_count 하는 이유: token이 단어 안에서 여러 번 반복될 수 있기 때문

        # Counter 생성과 값 조정을 한 번에 수행
        counter = Counter({token: tokens.count(token) * self.instance_count for token in set(tokens)})

        return counter

    
    def get_token_count(self):
        '''
        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''
        return self._get_token_count(self.tokens)
    
    def _get_bigram_count(self, bigrams: list):
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

        bigrams = []
        for i in range(len(tokens) - 1):
            if i == 0:
                bigrams.append('[word]' + tokens[i] + tokens[i + 1])
            else:
                bigrams.append('[subword]' + tokens[i] + tokens[i + 1])       
        # logger.debug(f"{self.word} 인스턴스의 인접 토큰 쌍 (bigrams): {bigrams}")

        return bigrams
    
    def get_bigram_count(self):
        '''
        임의로 생성한 인접 토큰 쌍과 임의로 생성된 인접 토큰 쌍의 개수를 카운트
        return (list): [(bigram1, count1), (bigram2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._get_bigram_count(self._create_bigrams())