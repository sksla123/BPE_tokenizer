from collections import Counter

from .instance import Instance
from .vocab import Vocabulary
from .tokenize import pre_tokenize
from .util import strip_token

import os

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

        ## break 판단을 위한 카운터
        self.instances_token_counter = None

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
    
    def _build_instance_and_set_token_counter(self, instance: str, instance_count: int) -> Instance:
        '''
        이거 그냥 불러오면 오류일으킴

        instance (str): 인스턴스
        instance_count (int): 인스턴스 개수

        return (list): 인스턴스 목록
        '''

        _instance = Instance(instance, instance_count)
        self.instances_token_counter[instance] = _instance.get_token_count()
       
        return _instance

    def _build_instances(self, tokenized_instances: list) -> list:
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (list): 인스턴스 목록
        '''

        ## 인스턴스 토큰 카운터가 초기화되지 않았다면 오류 발생
        if self.instances_token_counter is None:
            raise ValueError("인스턴스 토큰 카운터가 초기화되지 않았습니다.")

        precomputed_counts = Counter(tokenized_instances)
        instances = precomputed_counts.keys()

        result = [self._build_instance_and_set_token_counter(instance, precomputed_counts[instance]) for instance in instances]

        print("인스턴스 생성 완료")

        return result        
    
    def _update_instances(self, instances: list, vocab: Vocabulary):
        '''
        새로운 vocab으로 다시 토큰화

        instances (list): 인스턴스 목록
        vocab (Vobaulary): 어휘 집합

        return (list, int): 업데이트된 인스턴스 목록, 시그마 instance 내부의 토큰의 수(1 이라면 훈련 종료) 
        '''
        for instance in instances:
            instance.tokenize(vocab)

            ## 토큰 카운터 업데이트
            self.instances_token_counter[instance] = instance.get_token_count()

        return instances

    def _build_base_vocab(self, corpus: str, instances: list) -> Vocabulary:
        '''
        instances (list): 인스턴스 목록

        return (Vobaulary): 어휘 집합
        '''
        
        whitespace_chars = {9, 10, 11, 12, 13, 32}
        word_vocab = [chr(i) for i in range(128)] 
        _char_in_corpus = list(set(corpus)) # 혹시 corpus에 ascii보다 큰 문자가 있을 수 있어서 처리하는 함수 추가(현 BPE에서는 ascii 문자만 사용함)
        word_vocab.extend(_char_in_corpus)

        # 화이트 스페이스 문자 제거
        word_vocab = [voc if voc not in whitespace_chars else voc for voc in set(word_vocab)]

        total = len(instances)
        i = 0
        subword_vocab = []
        for instance in instances:
            vocabs = instance.get_tokens()

            for vocab in vocabs:
                _vocab = strip_token(vocab)

                if vocab.startswith("[word]"):
                    word_vocab.append(_vocab)
                else:
                    subword_vocab.append(_vocab)
            
            i += 1
            print(f"\r각 instance로 부터 vocab 생성 중 {i} / {total}", end="")
        
        print(f"\n각 instance로 부터 vocab 생성 완료")

        word_vocab = list(set(word_vocab))
        subword_vocab = list(set(subword_vocab))
        base_vocab = Vocabulary(word_vocab, subword_vocab)        

        return base_vocab

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
        tokenized_instances = pre_tokenize(self.train_corpus, method="whitespace")

        logger.info(f"tokenized_instances 개수: {len(tokenized_instances)}")

        logger.info("BPE 훈련을 진행합니다.")
        self._train_bpe(tokenized_instances)
        
    def _train_bpe(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (dict): 훈련 결과
        '''

        vocab_size = self.vocab_size
        self.instances_token_counter = Counter()

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
            
            # 역인덱싱(instance 업데이트 과정에서 속도 증강을 위함)
            # bigram to instance 매핑
            bigram_to_instances = {}

            # 현 상황에서 가장 자주 등장하는 인접 토큰 쌍 찾기
            bigram_freq = Counter()
            for instance in instances:
                _bigram_freq = instance.get_bigram_count()
                # logger.debug(f"{instance.word}의 _bigram_freq: {_bigram_freq}")
                bigram_freq += _bigram_freq

                # 역인덱싱
                for bigram in _bigram_freq:
                    if bigram not in bigram_to_instances:
                        bigram_to_instances[bigram] = [instance]
                    else:
                        bigram_to_instances[bigram].append(instance)
            # logger.debug(f"bigram_freq: {bigram_freq}")

            # 가장 자주 등장하는 인접 토큰 쌍 찾기
            max_bigram = bigram_freq.most_common(1)[0][0]
            logger.info(f"가장 자주 등장하는 토큰 쌍 검색 완료")
            logger.info(f"가장 자주 등장하는 토큰 쌍: {max_bigram}, {bigram_freq[max_bigram]}")
            logger.info(f"어휘 집합에 새로운 단어를 추가합니다. {max_bigram}")

            # 가장 자주 등장하는 인접 토큰 쌍을 vocab에 추가
            vocab.add(max_bigram)
            logger.debug(f"변화된 어휘 집합 크기: {len(vocab)}")

            logger.info("인스턴스 업데이트 진행")
            # 인스턴스 업데이트
            instances = self._update_instances(bigram_to_instances[max_bigram], vocab)

            train_loop_count += 1

            if all(value == 1 for value in self.instances_token_counter.values()):
                logger.info("더 이상 instance를 토큰화하는 것이 불가능하여 훈련 종료")
                break

        logger.info(f"훈련 종료, 최종 훈련 반복 횟수: {train_loop_count}")
        logger.debug(f"훈련 종료 후 어휘 집합: {vocab}")

        self.vocab = vocab

    def save_vocab(self, vocab: Vocabulary):
        '''
        vocab (Vocabulary): 어휘 집합
        '''

        logger.info(f"어휘 집합을 저장합니다. {self.vocab_save_path}")
        logger.info(f"어휘 집합 크기: {len(vocab)}")
        os.makedirs(os.path.dirname(self.vocab_save_path), exist_ok=True)
        with open(self.vocab_save_path, 'w') as f:
            for voc in vocab.get_vocab():
                f.write(voc + '\n')
        logger.info(f"어휘 집합 저장 완료")

    def load_vocab(self, vocab_path: str) -> Vocabulary:
        '''
        vocab_path (str): 어휘 집합 파일 경로

        return (Vocabulary): 어휘 집합
        '''

        # 어휘 집합 존재 여부 검사
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"어휘 집합 파일을 찾을 수 없습니다. {vocab_path}")
        
        logger.info(f"어휘 집합을 로딩합니다. {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = f.read().splitlines()
        logger.info(f"어휘 집합 로딩 완료")
        logger.debug(f"어휘 집합 크기: {len(vocab)}")

        return Vocabulary(vocab)
    
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