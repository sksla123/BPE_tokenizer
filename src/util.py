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

from .token import Token

def token_to_string(token: Token) -> str:
    '''
    토큰을 문자열로 변환
    return (str): 토큰 목록을 문자열로 변환한 결과
    '''

    return "##" + token.string if token.is_sub else token.string