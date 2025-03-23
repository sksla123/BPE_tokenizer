import os
import argparse

import re

import logging

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

# 함수 정의 파트

# 어휘 집합 클래스
class Vobaulary:
    '''
    어휘 집합을 관리하는 클래스
    항상 정렬된 어휘 집합을 유지하기 위해 생성
    '''
    def __init__(self, vocab: list):
        self.vocab = []
        self.vocab = self._sort_vocab(vocab)
        self.hashed_vocab_start_idx = -1

        self.start_vocab = []
        self.hashed_vocab = []

    def __str__(self):
        return f"Vocabulary: {self.vocab}"

    def __len__(self):
        return len(self.vocab)

    def _sort_vocab(self, vocab: list):
        '''
        vocab (list): 어휘 집합

        return (list): 정렬된 어휘 집합
        '''
        def _sorting_rule(s):
            '''
            어휘 집합을 정렬하는 규칙
            1. ##으로 시작한다면 후순위로 배치
            2. 길이 기준 내림차순
            3. 사전순 오름차순
            '''

            is_hash = s.startwith('##')
            return (is_hash, -len(s), s)

        return sorted(vocab, key=self._sorting_rule)

    def get_vocab(self):
        return self.vocab

    def get_hashed_vocab_start_idx(self):
        return self.hashed_vocab_start_idx
    
    def get_start_vocab(self):
        return self.start_vocab
    
    def get_hashed_vocab(self):
        return self.hashed_vocab
    
    def set_vocab(self, vocab: list):
        '''
        어휘 집합을 설정하는 함수
        어휘 집합을 정렬하고, 해쉬로 시작하는 보캡의 위치를 찾아서 시작 보캡과 해쉬 보캡을 분리

        vocab (list): 어휘 집합
        '''
        self.vocab = self._sort_vocab(vocab)
        ## 해쉬로 시작하는 보캡의 위치를 찾기
        ## 그냥 for 문은 느릴 거 같아서 next 함수를 이용
        self.hashed_vocab_start_idx = next((idx for idx, voc in enumerate(self.vocab) if voc.startswith('##')), -1)
        ## 시작 보캡과 해쉬 보캡 분리
        self.start_vocab = self.vocab[:self.hashed_vocab_start_idx]
        self.hashed_vocab = self.vocab[self.hashed_vocab_start_idx:]
    
    def add(self, vocab: str):
        '''
        어휘 집합에 새로운 단어를 추가하는 함수
        '''
        self.vocab.append(vocab)
        self.vocab = self._sort_vocab(self.vocab)
        
## 토큰화 함수 정의
def tokenize(word: str, vocab: Vobaulary):
    '''
    word (str): 토큰화할 단어
    vocab (Vobaulary): 어휘 집합

    return (list): 토큰화된 단어
    '''
    tokens = []

    for voc in vocab.get_start_vocab():
        if word.startswith(voc):
            tokens.append(voc)
            break
    
    new_word = word
    while True:
        new_word = new_word.replace(tokens[-1].strip('##'), '')
        if new_word == '':
            break
        
        for voc in vocab.get_hashed_vocab():
            if new_word.startswith(voc.strip('##')):
                tokens.append(voc)
                break
    
    return tokens
        
# Instace 클래스 정의
class Instance:
    def __init__(self, instance: str, instance_count: int):
        self.word = instance

        # 첫 토큰화(알파벳 단위)
        self.tokens = ['##' + token if i > 0 else token for i, token in enumerate(self.word)]
        self.token_count = len(self.tokens)
        self.instance_count = instance_count

    def tokenize(self, vocab: Vobaulary):
        '''
        vocab (Vobaulary): 어휘 집합

        return (list): 토큰화된 인스턴스
        '''

        self.tokens = tokenize(self.word, vocab)
        self.token_count = len(self.tokens)

        return self.token_count

    def get_tokens(self):
        return self.tokens

    def _get_token_count(self, tokens: str):
        '''
        tokens (str): 토큰 목록

        return (list): [(token1, count1), (token2, count2), ...] 형식의 토큰 목록과 각 토큰의 등장 횟수
        '''

        # 토큰 목록과 각 토큰의 등장 횟수 반환
        ## self.word.count(token) * self.instance_count하는 이유 token이 단어 안에서 여러번 반복 될 수 있기 때문
        return [(token, tokens.count(token) * self.instance_count) for token in set(tokens)]

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
        tokens = [token.strip('##') for token in self.tokens]

        pairs = []
        for i in range(len(tokens) - 1):
            if i == 0:
                pairs.append(tokens[i] + tokens[i + 1])
            else:
                pairs.append('##' + self.tokens[i] + self.tokens[i + 1])
        
        return pairs
    
    def get_pair_and_count(self):
        '''
        임의로 생성한 인접 토큰 쌍과 임의로 생성된 인접 토큰 쌍의 개수를 카운트
        return (list): [(pair1, count1), (pair2, count2), ...] 형식의 토큰 쌍 목록과 각 토큰 쌍의 등장 횟수
        '''

        return self._get_pair_count(self._create_pair())

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
        with open(corpus_path, 'r') as f:
            corpus = f.read()
        return corpus
    
    def _build_instances(self, tokenized_instances: list) -> list:
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (list): 인스턴스 목록
        '''

        instances = set(tokenized_instances)
        
        return [Instance(instance, tokenized_instances.count(instance)) for instance in instances]
    
    def _update_instances(self, instances: list, vocab: Vobaulary):
        '''
        새로운 vocab으로 다시 토큰화

        instances (list): 인스턴스 목록
        vocab (Vobaulary): 어휘 집합

        return (list, int): 업데이트된 인스턴스 목록, 시그마 instance 내부의 토큰의 수(1 이라면 훈련 종료) 
        '''
        
        f = 1
        for instance in instances:
            ## f에 곱하는 이유, 만약 instance 내부의 토큰의 수가 모두 1이라면 더 이상 토큰화가 불가능하기 때문
            f *= instance.tokenize(vocab)
        
        return instances, f

    def _build_base_vocab(self, corpus: str, instances: list) -> Vobaulary:
        '''
        instances (list): 인스턴스 목록

        return (Vobaulary): 어휘 집합
        '''
        
        base_vocab = list(set(corpus))
        instance_vocab = [instance.get_tokens() for instance in instances]
        base_vocab.extend(instance_vocab)

        return Vobaulary(base_vocab)

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
        logging.debug(f"훈련 데이터 경로: {self.train_corpus_path}")
        self.train_corpus = self._load_corpus(self.train_corpus_path)
        
        logging.info("BPE 훈련을 진행합니다.")

        logging.info("Pre-Tokenization을 진행합니다.")
        tokenized_instances = self.pre_tokenize(self.train_corpus)

        logging.info("BPE 훈련을 진행합니다.")
        self._train_bpe(tokenized_instances)
        
    def _train_bpe(self, tokenized_instances: list):
        '''
        tokenized_instances (list): 토큰화된 인스턴스 목록

        return (dict): 훈련 결과
        '''

        vocab_size = self.vocab_size
        instances = self._build_instances(tokenized_instances)
        vocab = self._build_base_vocab(self.train_corpus, instances)

        train_loop_count = 0
        while len(vocab) < vocab_size:
            logging.info(f"현재 훈련 반복 횟수: {train_loop_count}")
            logging.debug(f"현재 어휘 집합 크기: {len(vocab)}")
            logging.debug(f"현재 어휘 집합: {vocab}")

            # 현 상황에서 가장 자주 등장하는 인접 토큰 쌍 찾기
            pair_freq = {}
            for instance in instances:
                temp_pair_freq = instance.get_pair_and_count()
                for pair, count in temp_pair_freq:
                    if pair not in pair_freq:
                        pair_freq[pair] = count
                    else:
                        pair_freq[pair] += count
            
            # 가장 자주 등장하는 인접 토큰 쌍 찾기
            max_pair = max(pair_freq, key=pair_freq.get)
            logging.debug(f"가장 자주 등장하는 토큰 쌍: {max_pair}")
            logging.debug(f"가장 자주 등장하는 토큰 쌍 등장 횟수: {pair_freq[max_pair]}")
            logging.info(f"어휘 집합에 새로운 단어를 추가합니다. {max_pair}")

            # 가장 자주 등장하는 인접 토큰 쌍을 vocab에 추가
            vocab.add(max_pair)
            logging.debug(f"변화된 어휘 집합 크기: {len(vocab)}")

            logging.info("인스턴스 업데이트 진행")
            # 인스턴스 업데이트
            instances, f = self._update_instances(instances, vocab)

            train_loop_count += 1

            if f == 1:
                logging.info("더 이상 instance를 토큰화하는 것이 불가능하여 훈련 종료")
                break

        logging.info(f"훈련 종료, 최종 훈련 반복 횟수: {train_loop_count}")
        logging.debug(f"훈련 종료 후 어휘 집합: {vocab}")

        self.vocab = vocab

    def save_vocab(self, vocab: Vobaulary):
        '''
        vocab (Vobaulary): 어휘 집합
        '''

        logging.info(f"어휘 집합을 저장합니다. {self.vocab_save_path}")
        logging.info(f"어휘 집합 크기: {len(vocab)}")
        with open(self.vocab_save_path, 'w') as f:
            for voc in vocab.get_vocab():
                f.write(voc + '\n')
        logging.info(f"어휘 집합 저장 완료")

    def load_vocab(self, vocab_path: str) -> Vobaulary:
        '''
        vocab_path (str): 어휘 집합 파일 경로

        return (Vobaulary): 어휘 집합
        '''

        logging.info(f"어휘 집합을 로딩합니다. {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab = f.read().splitlines()
        logging.info(f"어휘 집합 로딩 완료")
        logging.debug(f"어휘 집합 크기: {len(vocab)}")

        return Vobaulary(vocab)
    
    def infer(self, input_data: str):
        '''
        input_data (str): 추론 데이터
        '''

        logging.info(f"추론 데이터를 로딩합니다. {self.input_data_path}")
        with open(self.input_data_path, 'r') as f:
            input_data = f.read()
        logging.info(f"추론 데이터 로딩 완료")

        logging.info(f"추론에 사용할 어휘 집합을 로딩합니다. {self.infer_vocab_path}")
        self.vocab = self.load_vocab(self.infer_vocab_path)
        logging.info(f"추론에 사용할 어휘 집합 로딩 완료")

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
    

