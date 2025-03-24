from collections import Counter

from .instance import Instance
from .tokenize import pre_tokenize
from .token import Token
from .vocab import Vocabulary
from .util import token_to_string

import os
import json

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
    
    def _build_instances(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록
        '''

        Instance.init_static_variables()
        precomputed_counts = Counter(tokenized_instances)

        for instance in precomputed_counts.keys():
            Instance(instance, precomputed_counts[instance])

        print("인스턴스 생성 완료")
    
    def _update_instances(self, max_bigram: Token, vocab: Vocabulary):
        '''
        새로운 vocab으로 다시 토큰화

        target_instances (list): 업데이트할 인스턴스 목록
        vocab (Vobaulary): 어휘 집합

        return (list, int): 업데이트된 인스턴스 목록, 시그마 instance 내부의 토큰의 수(1 이라면 훈련 종료) 
        '''
        Instance.init_updated_instances()
        Instance.update_instances(max_bigram, vocab, mode="train")

    def _build_base_vocab(self, corpus: str) -> Vocabulary:
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

        total = len(Instance.word_to_instance.values())
        i = 0
        subword_vocab = word_vocab.copy()
        for instance in Instance.word_to_instance.values():
            base_tokens = instance.get_tokens()

            for base_token in base_tokens:
                if base_token.is_sub:
                    subword_vocab.append(base_token.string)
                else:
                    word_vocab.append(base_token.string)
            
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

        logger.info(f"초기 인스턴스 생성 중 ...")
        self._build_instances(tokenized_instances)
        logger.info(f"초기 인스턴스 생성 완료")

        logger.info(f"초기 어휘 집합 생성 중 ...")
        vocab = self._build_base_vocab(self.train_corpus)
        logger.info(f"초기 어휘 집합 생성 완료")

        train_loop_count = 0

        while len(vocab) < vocab_size:
            logger.info(f"현재 훈련 반복 횟수: {train_loop_count}")
            logger.debug(f"현재 어휘 집합 크기: {len(vocab)}")
            # logger.debug(f"현재 어휘 집합: {vocab}") ## 로그 파일 크기가 너무 커짐

            logger.info(f"현재 상황에서 가장 자주 등장하는 인접 토큰 쌍 검색 중 ...")

            # 가장 자주 등장하는 인접 토큰 쌍 찾기
            logger.debug(f"Instance.bigram_counter: {Instance.bigram_counter}")
            if len(Instance.bigram_counter.keys()) == 0:
                logger.info("더 이상 bigram을 생성할 수 없어 훈련 종료(전체 인스턴스에 대한 vocab 생성 완료)")
                break

            max_bigram = Instance.bigram_counter.most_common(1)[0][0]
            logger.info(f"가장 자주 등장하는 토큰 쌍 검색 완료")
            logger.info(f"가장 자주 등장하는 토큰 쌍: {max_bigram}, {Instance.bigram_counter[max_bigram]}")
            logger.info(f"어휘 집합에 새로운 단어를 추가합니다. {max_bigram}")

            # 가장 자주 등장하는 인접 토큰 쌍을 vocab에 추가
            vocab.add(max_bigram)
            logger.debug(f"변화된 어휘 집합 크기: {len(vocab)}")
            # logger.debug(f"변화된 어휘 집합: \n{vocab.get_vocab()}")

            logger.info("인스턴스 업데이트 진행")
            # 인스턴스 업데이트
            self._update_instances(max_bigram, vocab)

            train_loop_count += 1

        logger.info(f"훈련 종료, 최종 훈련 반복 횟수: {train_loop_count}")
        logger.debug(f"훈련 종료 후 어휘 집합: {vocab}")

        self.vocab = vocab

    def save_vocab(self, vocab: Vocabulary):
        '''
        vocab (Vocabulary): 어휘 집합
        '''

        logger.info(f"어휘 집합을 저장합니다. {self.vocab_save_path}")
        logger.info(f"어휘 집합 크기: {len(vocab)}")

        vocab_dict = {}
        vocab_dict["word_vocab"] = self.vocab.get_word_vocab()
        vocab_dict["subword_vocab"] = self.vocab.get_subword_vocab()
        
        os.makedirs(os.path.dirname(self.vocab_save_path), exist_ok=True)
        
        with open(self.vocab_save_path, 'w') as f:
            json.dump(vocab_dict, f)
        
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
            vocab_dict = json.load(f)
        
        vocab = Vocabulary(vocab_dict["word_vocab"], vocab_dict["subword_vocab"])
        
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

        logger.info("추론할 input_data 로딩")
        tokenized_instances = pre_tokenize(input_data, method="whitespace")
        logger.info(f"추론할 input_data 로딩 완료")

        logger.info("추론 진행")
        infer_output = self._infer(tokenized_instances)
        logger.info("추론 완료")

        self._save_infer_output(infer_output)

        return infer_output

    def _infer(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록
        '''

        Instance.init_static_variables()

        infer_tokens_list = []

        for instance in tokenized_instances:
            inst = Instance(instance, 1)
            inst.tokenize(self.vocab, mode="infer")
            infer_output.append(inst.get_tokens())
        
        infer_output = ""
        for infer_tokens in infer_output:
            infer_output += " ".join([token_to_string(token) for token in infer_tokens])
            infer_output += "\n"

        return infer_output

    def save_infer_output(self, infer_output: str):
        '''
        infer_output (str): 추론 결과
        '''

        logger.info(f"추론 결과를 저장합니다. {self.tokenized_result_path}")
        with open(self.tokenized_result_path, 'w') as f:
            f.write(infer_output)