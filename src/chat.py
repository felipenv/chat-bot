from transformers import AutoTokenizer
from parse_data import ParseFB

block_size = 64
bot_sender = "Felipe Vianna"


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


if __name__ == "__main__":

    parser = ParseFB(bot_sender)
    parser.parse_data()
    parser.build_model_data()
    parser.build_huggingface_dataset()

    # tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>",
                                    "bos_token": "<startofstring>",
                                    "eos_token": "<endofstring>"})
    tokenizer.add_tokens(["<bot>"])

    parser.tokenize_data(tokenizer)

    # concat all sentences and break them into block size chunks.
    lm_dataset = parser.tokenized_data.map(group_texts, batched=True, num_proc=4)

