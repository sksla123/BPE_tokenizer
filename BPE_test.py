import os
import argparse

import re

import logging

# 함수 정의 파트
# 타임스탬프 문자열 함수 정의
from datetime import datetime
from zoneinfo import ZoneInfo

def get_kst_timestamp() -> str:
    '''
    현재 시간을 기준으로 한국 타임스탬프 문자열 반환
    return (str): 한국 타임스탬프 문자열
    '''
    kst_time = datetime.now(ZoneInfo("Asia/Seoul"))
    return kst_time.strftime("%Y%m%d%H%M%S")

class Vobaulary:
    def __init__(self, vocab: list):
        self.vocab = []

    def __str__(self):
        return f"Vocabulary: {self.vocab}"

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab
    
    def set_vocab(self, vocab: list):
        

# Instace 클래스 정의
class Instance:
    def __init__(self, instance: str, instance_count: int):
        self.word = instance
        self.tokens = ['##' + token if i > 0 else token for i, token in enumerate(self.word)]
        self.token_count = len(self.tokens)
        self.instance_count = instance_count
    
    def tokenize(self, vocab: list):
        '''
        vocab (list): 어휘 집합

        return (list): 토큰화된 인스턴스
        '''

        self.tokens = 
        self.token_count = len(self.tokens)

    def get_tokens(self):
        return self.tokens

    def _get_token_count(self, tokens: str):
        '''
        tokens (str): 토큰 목록

        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''

        # 토큰 목록과 각 토큰의 등장 횟수 반환
        ## self.word.count(token) * self.instance_count하는 이유 token이 단어 안에서 여러번 반복 될 수 있기 때문
        return [(token, self.word.count(token.strip('##')) * self.instance_count) for token in tokens]

    def get_token_count(self):
        '''
        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''

        return self._get_token_count(self.tokens)
    
    def _get_pair_count(self, pairs: list):
        '''
        pairs (list): [pair1, pair2, ...] 형식의 토큰 쌍 목록

        return (list): [(pair1, count1), (pair2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._get_token_count(pairs)

    def _create_pair(self):
        '''
        현재 토큰을 가지고 인스턴스 내부에서 인접 토큰들과 연결하여, 임의의 토큰 쌍을 생성하는 함수

        return (list): [pair1, pair2, ...] 형식의 토큰 쌍 목록
        '''

        pairs = []
        for i in range(len(self.tokens) - 1):
            if i == 0:
                pairs.append(self.tokens[i] + self.tokens[i + 1])
            else:
                pairs.append('##' + self.tokens[i] + self.tokens[i + 1])
        
        return pairs
    
    def get_pair_and_count(self):
        '''
        return (list): [(pair1, count1), (pair2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._get_pair_count(self._create_pair())

# BPE 토크나이저 클래스 정의
class BPE():
    def __init__(self, config_data: dict):
        self.config_data = config_data
        self.train_corpus = config_data.get("train_corpus_path", None)
        self.pre_tokenized_corpus = None
        self.vocab_size = config_data.get("max_vocab", None)
        self.vocab_save_path = config_data.get("vocab_output_path", None)
        self.infer_vocab_path = config_data.get("infer_vocab_path", None)
        self.input_data_path = config_data.get("input_data_path", None)
        self.tokenized_result_path = config_data.get("tokenized_result_path", None)
        self.vocab_dict = {}

    def _load_corpus(self, corpus_path: str) -> str:
        '''
        코퍼스 파일을 로딩합니다.
        corpus_path (str): 코퍼스 파일 경로

        return (str): 코퍼스 파일 내용
        '''
        with open(corpus_path, 'r') as f:
            corpus = f.read()
        return corpus
    
    def _build_instances(self, tokenized_instances: list) -> list:
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (list): 인스턴스 목록
        '''

        instances = set(tokenized_instances)
        instance_count = {instance: tokenized_instances.count(instance) for instance in instances}
        
        return [Instance(instance, instance_count[instance]) for instance in instances]

    # pre-tokenize 함수 정의
    def pre_tokenize(self, corpus: str, method: str = "whitespace"):
        '''
        corpus (str): pre-tokenize 대상 코퍼스
        method (str): pre-tokenize 방법 [whitespace] # 현재는 whitespace만 지원(추후 개발)

        return (list): pre-tokenize 결과(tokenized-instances)
        ''' 

        if method == "whitespace":
            return re.split(r'\s+', corpus) ## whitespace 기준으로 분리
        else:
            raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}\n 지원하는 메소드 목록: [whitespace]")

    # BPE 훈련 함수 정의
    def train_bpe(self):
        '''
        train_dataset (dict): 훈련 설정 데이터

        return (dict): 훈련 결과
        '''
        
        logging.info("훈련에 사용할 말 뭉치를 로딩합니다.")
        logging.debug(f"훈련 데이터 경로: {self.train_corpus}")
        train_corpus = self._load_corpus(self.train_corpus)
        
        logging.info("BPE 훈련을 진행합니다.")

        logging.info("Pre-Tokenization을 진행합니다.")
        tokenized_instances = self.pre_tokenize(train_corpus)
        
    def _train_bpe(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (dict): 훈련 결과
        '''

        vocab_size = self.vocab_size
        

        train_loop_count = 0
        while len(vocab) < vocab_size:
            logging.info(f"현재 훈련 반복 횟수: {train_loop_count}")
            logging.debug(f"현재 어휘 집합: {vocab}")
            logging.debug(f"현재 어휘 집합 크기: {len(vocab)}")

            # 훈련 데이터에서 가장 자주 등장하는 토큰 쌍 찾기
            pair_freq = {}
            

## 디버그용 로그 설정
parser = argparse.ArgumentParser(prog="BPE Tokenizer")
## 디버그용 로그 설정
parser.add_argument('--log', action="store_false", help='학습에 사용할 코퍼스 파일 위치')

## 학습 모드와 추론 모드는 동시에 실행 불가능하게 막아놓음
group = parser.add_mutually_exclusive_group()
## 학습 모드에 사용될 관련 args
group.add_argument('--train', type=str, help='학습에 사용할 코퍼스 파일 위치') # infer와 베타적 그룹
parser.add_argument('--max_vocab', type=int, help='vocab 최대 크기')
parser.add_argument('--vocab', type=str, help='학습 결과를 저장할 vocab 파일 위치')

## 추론 모드에 사용될 관련 args
group.add_argument('--infer', type=str, help='추론에 사용할 저장된 vocab 파일 위치') # train 베타적 그룹
parser.add_argument('--input', type=str, help='추론할 입력 파일(텍스트)')
parser.add_argument('--output', type=str, help='추론된 결과를 저장할 파일 위치')

args = parser.parse_args()

logger = logging.getLogger(__name__)
log_file_path = f"./log/{get_kst_timestamp()}.log"

# 로그 폴더 생성
os.makedirs(log_file_path, exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 로그 비활성화
if not args.log:
    logging.disable(logging.CRITICAL)

logging.debug(f"디버그 로깅 활성화")
logging.info("로깅 활성화, BPE 토크나이저 프로그램 실행됨.")

if args.train:
    logging.debug("모드 설정: train")
    mode = "train"

    config_data = {
        "train_corpus_path": args.train,
        "max_vocab": args.max_vocab,
        "vocab_output_path": args.vocab
    }
    logging.debug(f"훈련 데이터 설정: {config_data}")
elif args.infer:
    logging.debug("모드 설정: infer")
    mode = "infer"

    config_data = {
        "infer_vocab_path": args.infer,
        "input_data_path": args.input,
        "tokenized_result_path": args.output
    }
    logging.debug(f"추론 데이터 설정: {config_data}")

if mode == "train":
    logging.info("훈련 모드로 프로그램이 동작합니다.")
    

