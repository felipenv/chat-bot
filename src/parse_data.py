import json
import logging
import os
import pandas as pd
import re
import structlog
import sys
from datasets import Dataset

structlog.configure(logger_factory=structlog.stdlib.LoggerFactory())
logging.basicConfig(format="%(message)s", stream=sys.stderr, level=logging.INFO)
log = structlog.getLogger()

class ParseFB:
    def __init__(self, bot_sender):
        self.bot_sender = bot_sender
        self.data = None
        self.model_data = pd.DataFrame()
        self.tokenized_data = None
        self.huggingface_dataset = None


    @staticmethod
    def fix_string(s):
        out_s = re.sub(r'[\xc2-\xf4][\x80-\xbf]+', lambda m: m.group(0).encode('latin1').decode('utf8'), s)
        return out_s

    def parse_messages(self, messages):
        data = pd.DataFrame.from_records(messages)
        data['datetime'] = pd.to_datetime(data['timestamp_ms'], unit='ms')
        data = data.sort_values(by='datetime').reset_index(drop=True)
        data['content'].fillna("", inplace=True)
        data['content'] = data['content'].apply(lambda s: self.fix_string(s))

        def concatenate_with_newline(group):
            return '\n'.join(group)

        # Concatenate strings in the 'Text' column for each block of sequential identical 'ID' values using \n as separator
        data = data.groupby((data['sender_name'] != data['sender_name'].shift(1)).cumsum()).agg({'content': concatenate_with_newline, 'sender_name': 'first'}).reset_index(drop=True)
        if len(data) == 1:
            return pd.DataFrame()

        if data.iloc[0]['sender_name'] == self.bot_sender:
            row = pd.DataFrame({"content": [""], "sender_name": [data.iloc[1]['sender_name']]})
            data = pd.concat([row, data], axis=0, ignore_index=True)

        if data.iloc[-1]['sender_name'] != self.bot_sender:
            row = pd.DataFrame({"content": [""], "sender_name": [self.bot_sender]})
            data = pd.concat([data, row], axis=0, ignore_index=True)

        return data

    def parse_data(self, path='data/fb/inbox'):
        chats = os.listdir(path)
        chat_df = pd.DataFrame()
        for chat in chats:
            if chat != ".DS_Store":
                for file in os.listdir(f"{path}/{chat}"):
                    if file.endswith(".json"):
                        data_json = json.load(open(f"{path}/{chat}/{file}", "r", encoding='latin1'))
                        df = self.parse_messages(data_json["messages"])
                        chat_df = pd.concat([chat_df, df], axis=0)
        log.info("Finished to parse all data")
        self.data = chat_df

    def build_model_data(self, path='data/fb/inbox'):
        if self.data is None:
            self.parse_data(path)
        for idx in range(0, len(self.data),2):
            df = pd.DataFrame({"content": ["<startofstring> " + self.data.iloc[idx]['content'] + " <bot> " + self.data.iloc[idx+1]['content'] + " <endofstring>"]})
            self.model_data = pd.concat([self.model_data, df], ignore_index=True)

        log.info("Finished to build model data")


    def build_huggingface_dataset(self, path='data/fb/inbox'):
        if self.model_data.empty:
            self.build_model_data(path)

        self.huggingface_dataset = Dataset.from_pandas(self.model_data).train_test_split(test_size=0.2, shuffle=False)
        log.info("Finished to build huggingface dataset")


    def tokenize_data(self, tokenizer):
        def preprocess_function(example):
            return tokenizer(example['content'])

        self.tokenized_data = self.huggingface_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=self.huggingface_dataset["train"].column_names,
        )
