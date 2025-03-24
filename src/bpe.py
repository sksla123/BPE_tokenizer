from collections import Counter

from .instance import Instance
from .vocab import Vobaulary

import re
import os
import itertools

import multiprocessing as mp ## 너무 느려서 추가한 모듈

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

# BPE 토크나이저 클래스 정의
class BPE():
    def __init__(self, config_data: dict):
        self.config_data = config_data
        self.train_corpus_path = config_data.get("train_corpus_path", None)
        self.train_corpus = ""
        self.pre_tokenized_corpus = None
        self.vocab_size = config_data.get("max_vocab", None)
        self.vocab_save_path = config_data.get("vocab_output_path", None)
        self.infer_vocab_path = config_data.get("infer_vocab_path", None)
        self.input_data_path = config_data.get("input_data_path", None)
        self.tokenized_result_path = config_data.get("tokenized_result_path", None)
        self.vocab = None

    def _load_corpus(self, corpus_path: str) -> str:
        '''
        코퍼스 파일을 로딩합니다.
        corpus_path (str): 코퍼스 파일 경로

        return (str): 코퍼스 파일 내용
        '''
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
            # BOM 제거
            corpus = corpus.lstrip("\ufeff")

        return corpus
    
    def _build_instances(self, tokenized_instances: list) -> list:
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (list): 인스턴스 목록
        '''

        precomputed_counts = Counter(tokenized_instances)
        instances = precomputed_counts.keys()

        result = [Instance(instance, precomputed_counts[instance]) for instance in instances]

        print("인스턴스 생성 완료")

        return result        
    
    def _update_instances(self, instances: list, vocab: Vobaulary):
        '''
        새로운 vocab으로 다시 토큰화

        instances (list): 인스턴스 목록
        vocab (Vobaulary): 어휘 집합

        return (list, int): 업데이트된 인스턴스 목록, 시그마 instance 내부의 토큰의 수(1 이라면 훈련 종료) 
        '''
        
        f = 1

        total = len(instances)
        i = 0
        print("인스턴스 업데이트 중 ...")
        for instance in instances:
            ## f에 곱하는 이유, 만약 instance 내부의 토큰의 수가 모두 1이라면 더 이상 토큰화가 불가능하기 때문
            f *= instance.tokenize(vocab)
            i += 1
            print(f"\r인스턴스 업데이트 중 {i} / {total}", end="")
        print(f"\n인스턴스 업데이트 완료")
        
        return instances, f

    def _build_base_vocab(self, corpus: str, instances: list) -> Vobaulary:
        '''
        instances (list): 인스턴스 목록

        return (Vobaulary): 어휘 집합
        '''
        
        whitespace_chars = {9, 10, 11, 12, 13, 32}
        base_vocab = [chr(i) for i in range(128)] 
        _char_in_corpus = list(set(corpus)) # 혹시 corpus에 ascii보다 큰 문자가 있을 수 있어서 처리하는 함수 추가
        
        base_vocab.extend(_char_in_corpus)
        base_vocab = set(base_vocab)

        # 화이트 스페이스 문자 제거
        base_vocab = [voc for voc in base_vocab if voc not in whitespace_chars]

        total = len(instances)
        i = 0
        instance_vocab = []
        for instance in instances:
            instance_vocab.append(instance.get_tokens())
            i += 1
            print(f"\r인스턴스 vocab 업데이트 중 {i} / {total}", end="")
        
        instance_vocab = list(set(itertools.chain.from_iterable(instance_vocab)))
        print(f"\n인스턴스 vocab 업데이트 완료")

        # instance_vocab = list(itertools.chain.from_iterable([instance.get_tokens() for instance in instances]))
        base_vocab = list(itertools.chain(base_vocab, instance_vocab))

        logger.debug(f"초기 어휘 집합 크기: {len(base_vocab)}")
        logger.debug(f"초기 어휘 집합: {base_vocab}")

        return Vobaulary(base_vocab)

    # pre-tokenize 함수 정의
    def pre_tokenize(self, corpus: str, method: str = "whitespace"):
        '''
        corpus (str): pre-tokenize 대상 코퍼스
        method (str): pre-tokenize 방법 [whitespace] # 현재는 whitespace만 지원(추후 개발)

        return (list): pre-tokenize 결과(tokenized-instances)
        ''' 

        if method == "whitespace":
            ret = re.split(r'\s+', corpus) ## whitespace 기준으로 분리
            
            # 이상한 "" 문자 제거
            if "" in ret:
                ret.remove("")
            
            return ret
        else:
            raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}\n 지원하는 메소드 목록: [whitespace]")

    # BPE 훈련 함수 정의
    def train_bpe(self):
        '''
        train_dataset (dict): 훈련 설정 데이터

        return (dict): 훈련 결과
        '''
        
        logger.info("훈련에 사용할 말 뭉치를 로딩합니다.")
        logger.debug(f"훈련 데이터 경로: {self.train_corpus_path}")
        self.train_corpus = self._load_corpus(self.train_corpus_path)
        
        logger.info("BPE 훈련을 진행합니다.")

        logger.info("Pre-Tokenization을 진행합니다.")
        tokenized_instances = self.pre_tokenize(self.train_corpus)
        logger.info(f"tokenized_instances 개수: {len(tokenized_instances)}")

        logger.info("BPE 훈련을 진행합니다.")
        self._train_bpe(tokenized_instances)
        
    def _train_bpe(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (dict): 훈련 결과
        '''

        vocab_size = self.vocab_size

        logger.info(f"초기 인스턴스 생성 중 ...")
        instances = self._build_instances(tokenized_instances)
        logger.info(f"초기 인스턴스 생성 완료")

        logger.info(f"초기 어휘 집합 생성 중 ...")
        vocab = self._build_base_vocab(self.train_corpus, instances)
        logger.info(f"초기 어휘 집합 생성 완료")

        train_loop_count = 0
        while len(vocab) < vocab_size:
            logger.info(f"현재 훈련 반복 횟수: {train_loop_count}")
            logger.debug(f"현재 어휘 집합 크기: {len(vocab)}")
            # logger.debug(f"현재 어휘 집합: {vocab}") ## 로그 파일 크기가 너무 커짐

            logger.info(f"현재 상황에서 가장 자주 등장하는 인접 토큰 쌍 검색 중 ...")
            # 현 상황에서 가장 자주 등장하는 인접 토큰 쌍 찾기
            pair_freq = {}
            for instance in instances:
                logger.debug(f"{instance.word} 인스턴스 내부의 인접 토큰 쌍 검색 중 ...")
                temp_pair_freq = instance.get_pair_and_count()
                for pair, count in temp_pair_freq:
                    if pair not in pair_freq:
                        pair_freq[pair] = count
                    else:
                        pair_freq[pair] += count
            
            # 가장 자주 등장하는 인접 토큰 쌍 찾기
            max_pair = max(pair_freq, key=pair_freq.get)
            logger.info(f"가장 자주 등장하는 토큰 쌍 검색 완료")
            logger.info(f"가장 자주 등장하는 토큰 쌍: {max_pair}, {pair_freq[max_pair]}")
            logger.info(f"어휘 집합에 새로운 단어를 추가합니다. {max_pair}")

            # 가장 자주 등장하는 인접 토큰 쌍을 vocab에 추가
            vocab.add(max_pair)
            logger.debug(f"변화된 어휘 집합 크기: {len(vocab)}")

            logger.info("인스턴스 업데이트 진행")
            # 인스턴스 업데이트
            instances, f = self._update_instances(instances, vocab)

            train_loop_count += 1

            if f == 1:
                logger.info("더 이상 instance를 토큰화하는 것이 불가능하여 훈련 종료")
                break

        logger.info(f"훈련 종료, 최종 훈련 반복 횟수: {train_loop_count}")
        logger.debug(f"훈련 종료 후 어휘 집합: {vocab}")

        self.vocab = vocab

    def save_vocab(self, vocab: Vobaulary):
        '''
        vocab (Vobaulary): 어휘 집합
        '''

        logger.info(f"어휘 집합을 저장합니다. {self.vocab_save_path}")
        logger.info(f"어휘 집합 크기: {len(vocab)}")
        os.makedirs(os.path.dirname(self.vocab_save_path), exist_ok=True)
        with open(self.vocab_save_path, 'w') as f:
            for voc in vocab.get_vocab():
                f.write(voc + '\n')
        logger.info(f"어휘 집합 저장 완료")

    def load_vocab(self, vocab_path: str) -> Vobaulary:
        '''
        vocab_path (str): 어휘 집합 파일 경로

        return (Vobaulary): 어휘 집합
        '''

        # 어휘 집합 존재 여부 검사
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"어휘 집합 파일을 찾을 수 없습니다. {vocab_path}")
        
        logger.info(f"어휘 집합을 로딩합니다. {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = f.read().splitlines()
        logger.info(f"어휘 집합 로딩 완료")
        logger.debug(f"어휘 집합 크기: {len(vocab)}")

        return Vobaulary(vocab)
    
    def infer(self, input_data: str):
        '''
        input_data (str): 추론 데이터
        '''

        logger.info(f"추론 데이터를 로딩합니다. {self.input_data_path}")
        with open(self.input_data_path, 'r') as f:
            input_data = f.read()
        logger.info(f"추론 데이터 로딩 완료")

        logger.info(f"추론에 사용할 어휘 집합을 로딩합니다. {self.infer_vocab_path}")
        self.vocab = self.load_vocab(self.infer_vocab_path)
        logger.info(f"추론에 사용할 어휘 집합 로딩 완료")