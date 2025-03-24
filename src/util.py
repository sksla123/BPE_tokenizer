from collections import Counter

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

def strip_token(token: str) -> str:
    '''
    토큰에서 양쪽 끝의 '[subword]' 또는 '[word]'를 제거
    return (str): 제거된 토큰
    '''
    return token.lstrip('[subword]').lstrip('[word]')

def list_counter()