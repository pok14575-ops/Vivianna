import re

_MD_BOLD   = re.compile(r'\*\*(.+?)\*\*', re.DOTALL)
_MD_ITALIC = re.compile(r'\*(.+?)\*',     re.DOTALL)
_MD_CODE   = re.compile(r'`+(.+?)`+',     re.DOTALL)
_BRACKETS  = re.compile(r'\[.*?\]')
_HEADERS   = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_URLS      = re.compile(r'https?://\S+')
_FANCY     = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})

def clean_for_tts(text: str) -> str:
    text = _URLS.sub('', text)
    text = _HEADERS.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_CODE.sub(r'\1', text)
    text = _BRACKETS.sub('', text)
    text = text.translate(_FANCY)
    text = text.replace('**', '').replace('__', '').replace('~~', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
