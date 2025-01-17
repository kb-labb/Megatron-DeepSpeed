from datasets import load_dataset, concatenate_datasets
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
    Regex,
)
import argparse


def batch_iterator(dataset, dataset_size, batch_size):
    for i in range(0, dataset_size, batch_size):
        yield dataset[i:i + batch_size]["text"]


# https://github.com/huggingface/tokenizers/issues/640#issuecomment-792305076
def bpe_tokenizer_trainer(text,
                          vocab_size,
                          min_frequency=0,
                          add_prefix_space=True,
                          batch_size=50):
    # Supply either path to txt file or list of strings as text arg

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        # pre_tokenizers.Whitespace(),
        # pre_tokenizers.Punctuation(),
        pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space),
    ])
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " ")
    ])

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|endoftext|>"
        ],
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    if isinstance(text, str):
        # if user specified path to txt file as string
        tokenizer.train(text, trainer=trainer)
    else:
        # text is a datasets Dataset
        tokenizer.train_from_iterator(batch_iterator(text, len(text),
                                                     batch_size),
                                      trainer=trainer)

    tokenizer.post_processor = processors.RobertaProcessing(
        sep=("</s>", tokenizer.token_to_id("</s>")),
        cls=("<s>", tokenizer.token_to_id("<s>")))

    tokenizer.save("tokenizer.json", pretty=True)
    tokenizer.model.save("output_dir")


def pretokenizer_print(text):
    # To print and check how pre tokenization looks like
    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    return pre_tokenizer.pre_tokenize_str(text)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infiles", nargs="+", type=str)
    parser.add_argument("--outfolder", default="tokenizer.json", type=str)
    parser.add_argument("--vocab_size", type=int, default=50260)

    return parser.parse_args()


def main():
    args = get_args()
    print("Loading datasets", args.infiles)
    dataset = load_dataset(
        "text",
        data_files={f.split("/")[-1]: f
                    for f in args.infiles},
        cache_dir="cache_dataset",
    )
    print("concatenate data")
    dataset = concatenate_datasets([dataset[f] for f in dataset.keys()])
    print("start training")
    bpe_tokenizer_trainer(text=dataset, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
