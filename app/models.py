import logging

import transformers

logger = logging.getLogger(__name__)

chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)
