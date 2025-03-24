import argparse

import logging

from src.logger import logger_name, init_logger, disable_file_logger, set_log_level
from src.bpe import BPE

parser = argparse.ArgumentParser(prog="BPE Tokenizer")

## 디버그용 로그 설정
parser.add_argument('--log', type=str, help='기록할 로그 레벨', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

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

logger = logging.getLogger("BPE Tokenizer")
init_logger(logger)

set_log_level(logger, args.log)

def main():
    if args.train:
        logger.debug("모드 설정: train")
        mode = "train"

        config_data = {
            "train_corpus_path": args.train,
            "max_vocab": args.max_vocab,
            "vocab_output_path": args.vocab
        }
        logger.debug(f"훈련 데이터 설정: {config_data}")
    elif args.infer:
        logger.debug("모드 설정: infer")
        mode = "infer"

        config_data = {
            "infer_vocab_path": args.infer,
            "input_data_path": args.input,
            "tokenized_result_path": args.output
        }
        logger.debug(f"추론 데이터 설정: {config_data}")

    if mode == "train":
        logger.info("훈련 모드로 프로그램이 동작합니다.")

        bpe = BPE(config_data)
        bpe.train_bpe()
        bpe.save_vocab(bpe.vocab)
    elif mode == "infer":
        logger.info("추론 모드로 프로그램이 동작합니다.")

        bpe = BPE(config_data)
        infer_output = bpe.infer()
        bpe.save_infer_output(infer_output)

if __name__ == "__main__":
    main()