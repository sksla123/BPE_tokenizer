logger_name = "BPE Tokenizer"

import os
import sys
import logging

from .util import get_kst_timestamp

log_file_path = f"./logs/{get_kst_timestamp()}.log"
file_handler = logging.FileHandler(log_file_path)

def init_logger(logger):
    logger.setLevel(logging.DEBUG)

    # 로그 폴더 생성
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 파일 핸들러 추가 (로그를 파일에 기록)
    
    file_handler.setLevel(logging.DEBUG)  # DEBUG 이상의 모든 로그 기록
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # stdout 핸들러 추가 (로그를 stdout에 출력)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)  # stdout에 INFO 이상의 로그 출력
    stdout_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

def disable_file_logger(logger):
    file_handler.setLevel(logging.CRITICAL + 1)  # 파일 핸들러 비활성화 (모든 로그 무시)
    logger.info("파일 로깅 비활성화")  # stdout으로만 출력됨