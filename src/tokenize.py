import re
import sys

from .vocab import Vocabulary

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

def pre_tokenize(corpus: str, method: str = "whitespace"):
    '''
    corpus (str): pre-tokenize 대상 코퍼스
    method (str): pre-tokenize 방법 [whitespace] # 현재는 whitespace만 지원(추후 개발)

    return (list): pre-tokenize 결과(tokenized-instances)
    ''' 

    if method == "whitespace":
        ret = re.split(r'\s+', corpus) ## whitespace 기준으로 분리
        
        # 비어있는 문자 제거
        if "" in ret:
            ret.remove("")
        
        return ret
    else:
        raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}\n 지원하는 메소드 목록: [whitespace]")

## 토큰화 함수 정의
def tokenize(word: str, vocab: Vocabulary, mode: str = "infer"):
    '''
    word (str): 토큰화할 단어
    vocab (Vocabulary): 어휘 집합
    mode (str): 토큰화 모드 [train, infer]
    
    return (list): 토큰화된 단어
    '''
    logger.debug(f"[{mode} 모드] {word} 토큰화")
    tokens = []
    
    ## 학습 모드
    if mode == "train":
        token = vocab.get_token(word, "word")

        if token != "":
            tokens.append("[word]" + token)
        else:
            logger.error(f"토큰화 실패, word: {word}, token: {token}")
            raise ValueError(f"토큰화 실패, word: {word}, token: {token}")
        
        _word = word[len(token):]

        while _word != "":
            token = vocab.get_token(_word, "subword")

            if token != "":
                tokens.append("[subword]" + token)
                _word = _word[len(token):]
            else:
                logger.error(f"토큰화 실패(vocab에 없는 단어), word: {word}, token: {token}")
                raise ValueError(f"토큰화 실패, word: {word}, token: {token}")
            
            _word = _word[len(token):]
    ## 추론 모드
    elif mode == "infer":
        token = vocab.get_token(word, "word")

        if token != "":
            tokens.append("[word]" + token)
        else:
            ## BPE인데 단어 토큰화 실패?? 말도 안되거든요.
            tokens.append("[word]" + "[UNK]")
            logger.error(f"토큰화 실패, word: {word}, token: {token}")
            return tokens
        
        _word = word[len(token):]

        while _word != "":
            token = vocab.get_token(_word, "subword")

            if token != "":
                tokens.append("[subword]" + token)
                _word = _word[len(token):]
            else:
                token = vocab.get_token(_word, "word")

                if token != "":
                    tokens.append("[word]" + token)
                    _word = _word[len(token):]
                else:
                    ## BPE인데 단어 토큰화 실패?? 진짜 말도 안되는 사건이거든요.
                    tokens.append("[word]" + "[UNK]")
                    logger.error(f"토큰화 실패, word: {word}, token: {token}")
                    return tokens
                    
            _word = _word[len(token):]
    logger.debug(f"[{mode} 모드] {word} 토큰화 완료\n토큰 목록: {tokens}")

    return tokens