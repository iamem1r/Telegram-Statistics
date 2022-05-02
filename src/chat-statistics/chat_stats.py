import json
from pathlib import Path
from typing import Union

import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from hazm import Normalizer, word_tokenize
from loguru import logger
from src.data import DATA_DIR 
from wordcloud import STOPWORDS, WordCloud


class ChatStatistics:
    """Generates chat statistics from telegram chat json file
    """
    def __init__(self, chat_json: Union[str, Path]):
        """
        :param chat_json: path to telegram exported json file
        """
        #load chat data
        logger.info(f"Loading chat data from {chat_json}")
        with open(chat_json) as f:
            self.chat_data = json.load(f)
        self.normalizer = Normalizer()

        # load stopwords
        logger.info(f"Loading stop words from {DATA_DIR / 'stop_words.txt'}")
        with open(DATA_DIR / 'stopwords.txt') as f:
            stop_words = list(map(str.strip, f.readlines()))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))

    def generate_word_cloud(self, 
        output_dir: Union[str, Path], 
        width: int = 800, height: int = 800,
        max_font_size: int = 250
    ):
        """Generates a word cloud from a telegram chat data

        :param output_dir: path to output directory for word cloud image
        """
        logger.info("Loading text from strings and lists of chat data")
        logger.info("Tokenizing words of text")
        logger.info("Removing stop words from text")
        text_content = ''
        list_content = ''

        for msg in self.chat_data['chats']['list'][0]['messages']:
            if type(msg['text']) is str:
                tokens_str = word_tokenize(msg['text'])
                tokens_str = list(filter(lambda item: item not in self.stop_words, tokens_str))
                
                text_content += f" {' '.join(tokens_str)}"
                
            elif type(msg['text']) is list:
                for i in msg['text']:
                    if type(i) is str:
                        tokens_list = word_tokenize(i)
                        tokens_list = list(filter(lambda item: item not in self.stop_words, tokens_list))
                        
                        list_content += f" {' '.join(tokens_list)}"
        
        # concate, normalize and reshape for final word cloud
        text = text_content + list_content
        text = self.normalizer.normalize(text)
        text = arabic_reshaper.reshape(text)
        text = get_display(text)

        logger.info("Generating word cloud")
        # generate word cloud
        wordcloud = WordCloud(
            width=1200, height=1200, max_font_size=400,
            background_color='white', 
            font_path=str(DATA_DIR / 'IranianSansRegular.ttf')
        ).generate(text)

        logger.info(f"Saving word cloud to {output_dir}")
        wordcloud.to_file(str(Path(output_dir) / 'wordcloud.png'))

if __name__ == "__main__":
    chat_stats = ChatStatistics(chat_json=DATA_DIR / 'online.json')
    chat_stats.generate_word_cloud(output_dir=DATA_DIR)

    print("--"*10)
    print("DONE!")
    print("--"*10)

