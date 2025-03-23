import os
import argparse

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

# pre-tokenize 함수 정의
import re

def pre_tokenize(corpus: str, method: str = "whitespace"):
    '''
    corpus (str): pre-tokenize 대상 코퍼스
    method (str): pre-tokenize 방법 (whitespace, sentencepiece) # 현재는 whitespace만 지원(추후 개발)

    return (list): pre-tokenize 결과
    ''' 

    if method == "whitespace":
        return re.split(r'\s+', corpus) ## whitespace 기준으로 분리
    else:
        raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}")

# BPE 훈련 함수 정의
def train_bpe(train_dataset: dict):
    '''
    train_dataset (dict): 훈련 데이터 설정

    return (dict): 훈련 결과
    '''
    

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

logging.info("로깅 활성화, BPE 토크나이저 프로그램 실행됨.")

if args.train:
    logging.info("모드 설정: train")
    mode = "train"

    config_data = {
        "train_corpus_path": args.train,
        "max_vocab": args.max_vocab,
        "vocab_output_path": args.vocab
    }
    logging.info(f"훈련 데이터 설정: {config_data}")
elif args.infer:
    logging.info("모드 설정: infer")
    mode = "infer"

    config_data = {
        "infer_vocab_path": args.infer,
        "input_data_path": args.input,
        "tokenized_result_path": args.output
    }
    logging.info(f"추론 데이터 설정: {config_data}")

if mode == "train":
    logging.info("훈련 모드로 프로그램이 동작합니다.")
    logging.info("훈련에 사용할 말 뭉치를 로딩합니다.")
    with open(dataset["train_corpus_path"], 'r') as f:
        train_corpus = f.read()
    
    logging.info("Pre-Tokenization을 진행합니다.")
    pre_tokenized_corpus = pre_tokenize(train_corpus)

