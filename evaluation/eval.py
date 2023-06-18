import os
import json
import argparse
import threading
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
from comet import load_from_checkpoint, download_model
from bleurt import score as bleurt_score


MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(MAIN_DIR, "evaluation")
CACHE_DIR = os.path.join(EVAL_DIR, "cache")
RAW_DATA_DIR = os.path.join(MAIN_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(MAIN_DIR, "output")

LANG_PAIR2SPLIT = {
    "en-de": "wmt22",
    "de-en": "wmt22",
    "en-cs": "wmt22",
    "cs-en": "wmt22",
    "en-ru": "wmt22",
    "ru-en": "wmt22",
    "en-zh": "wmt22",
    "zh-en": "wmt22",
    "de-fr": "wmt22",
    "fr-de": "wmt22",
    "en-ja": "wmt22",
    "ja-en": "wmt22",
    "uk-en": "wmt22",
    "en-uk": "wmt22",
    "uk-cs": "wmt22",
    "cs-uk": "wmt22",
    "en-hr": "wmt22",
    "en-ha": "wmt21",
    "ha-en": "wmt21",
    "en-is": "wmt21",
    "is-en": "wmt21",
}

ALL_LANG_PAIRS_SUPPORTED = [
    "en-de",
    "de-en",
    "en-cs",
    "cs-en",
    "en-ru",
    "ru-en",
    "en-zh",
    "zh-en",
    "de-fr",
    "fr-de",
    "en-ja",
    "ja-en",
    "uk-en",
    "en-uk",
    "uk-cs",
    "cs-uk",
]

HIGH_RES_LANG_PAIRS = [
    "en-de",
    "de-en",
    "en-cs",
    "cs-en",
    "en-ru",
    "ru-en",
    "en-zh",
    "zh-en",
]

MID_RES_LANG_PAIRS = [
    "de-fr",
    "fr-de",
    "en-ja",
    "ja-en",
    "uk-en",
    "en-uk",
]

LOW_RES_LANG_PAIRS = [
    "uk-cs",
    "cs-uk",
    "en-hr",
    "en-ha",
    "ha-en",
    "en-is",
    "is-en",
]

SYSTEMS = [
    "deepl",
    "google-cloud",
    "wmt-winner",
    "text-davinci-003",
    "gpt-3.5-turbo-0301",
    "gpt-4-0314",
]

LOW_RES_SYSTEMS = [
    "google-cloud",
    "wmt-winner",
    "text-davinci-003",
    "gpt-3.5-turbo-0301",
    "gpt-4-0314",
]


METRICS = ["comet", "bleurt", "bleu", "chrf", "chrf_plusplus"]

def wait_until_path_exist(path):
    while not os.path.isdir(path):
        pass
    return

def comet(**kwargs):
    sys_lines = kwargs["sys_lines"]
    src_lines = kwargs["src_lines"]
    ref_lines = kwargs["ref_lines"]
    comet_model_name = kwargs["comet_model_name"]
    cache_dir = kwargs["cache_dir"]
    batch_size = kwargs["batch_size"]
    cache_file = os.path.join(cache_dir, 'comet_cache.json')

    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    data = []
    new_sys_lines, new_src_lines, new_ref_lines = [], [], []
    for sys, src, ref in zip(sys_lines, src_lines, ref_lines):
        cache_key = json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_src_lines.append(src)
            new_ref_lines.append(ref)
            data.append({"mt": sys, "src": src, "ref": ref})

    if data:
        comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        comet_model.eval()
        model_output = comet_model.predict(data, batch_size=batch_size, gpus=1)
        scores = model_output.scores

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            for (sys, src, ref), score in zip(zip(new_sys_lines, new_src_lines, new_ref_lines), scores):
                cache_key = json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)
                cache[cache_key] = score

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)

        final_scores = [cache[json.dumps((comet_model_name, sys, src, ref), ensure_ascii=False)] for sys, src, ref in zip(sys_lines, src_lines, ref_lines)]

    return sum(final_scores)/len(final_scores) * 100

def bleurt(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    cache_dir = kwargs["cache_dir"]
    bleurt_ckpt = kwargs["bleurt_ckpt"]
    batch_size = kwargs["batch_size"]
    cache_file = os.path.join(cache_dir, 'bleurt_cache.json')
    cache_lock = threading.Lock()

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        else:
            cache = {}

    new_sys_lines, new_ref_lines = [], []
    for sys, ref in zip(sys_lines, ref_lines):
        cache_key = json.dumps((sys, ref), ensure_ascii=False)
        if cache_key not in cache:
            new_sys_lines.append(sys)
            new_ref_lines.append(ref)

    assert len(new_sys_lines) == len(new_ref_lines)
    if len(new_sys_lines) > 0:
        bleurt_model = bleurt_score.LengthBatchingBleurtScorer(bleurt_ckpt)
        scores = bleurt_model.score(references=new_ref_lines, candidates=new_sys_lines, batch_size=batch_size)

        with cache_lock:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}

            for (sys, ref), score in zip(zip(new_sys_lines, new_ref_lines), scores):
                cache_key = json.dumps((sys, ref), ensure_ascii=False)
                cache[cache_key] = score

            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with cache_lock:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        final_scores = [cache[json.dumps((sys, ref), ensure_ascii=False)] for sys, ref in zip(sys_lines, ref_lines)]

    return sum(final_scores)/len(final_scores) * 100

def bleu(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    tgt_lang = kwargs["tgt_lang"]
    bleu = BLEU(trg_lang=tgt_lang)
    assert len(sys_lines) == len(ref_lines)
    return bleu.corpus_score(sys_lines, [ref_lines]).score

def chrf(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    chrf = CHRF()
    assert len(sys_lines) == len(ref_lines)
    return chrf.corpus_score(sys_lines, [ref_lines]).score

def chrf_plusplus(**kwargs):
    sys_lines = kwargs["sys_lines"]
    ref_lines = kwargs["ref_lines"]
    chrf = CHRF(word_order=2)
    assert len(sys_lines) == len(ref_lines)
    return chrf.corpus_score(sys_lines, [ref_lines]).score

def readlines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

def check_equal_num_lines(lst):
    if len(lst) <= 1:
        return True
    else:
        return all([len(p) == len(lst[0]) for p in lst[1:]])

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--comet-model-name", type=str, default="Unbabel/wmt22-comet-da")
    parser.add_argument("--bleurt-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    comet_model_name = args.comet_model_name
    bleurt_ckpt = args.bleurt_ckpt
    batch_size = args.batch_size


    # Full results
    for metric in METRICS:
        data = {
            'System': LANG_PAIR2SPLIT.keys(),
        }
        with tqdm(total=len(SYSTEMS) * len(LANG_PAIR2SPLIT), desc=metric) as pbar:
            for system in SYSTEMS:
                data[system] = []
                for lang_pair in LANG_PAIR2SPLIT.keys():
                    split = LANG_PAIR2SPLIT[lang_pair]
                    src_lang, tgt_lang = lang_pair.split('-')

                    src_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{src_lang}"))
                    ref_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{tgt_lang}"))
                    sys_lines = readlines(os.path.join(OUTPUT_DIR, system, f"{split}.{lang_pair}.{tgt_lang}"))

                    if check_equal_num_lines([src_lines, ref_lines, sys_lines]):
                        scorer = eval(metric)
                        score = scorer(**{
                            "sys_lines": sys_lines,
                            "src_lines": src_lines,
                            "ref_lines": ref_lines,
                            "comet_model_name": comet_model_name,
                            "cache_dir": CACHE_DIR,
                            "bleurt_ckpt": bleurt_ckpt,
                            "batch_size": batch_size,
                            "tgt_lang": tgt_lang
                        })
                    else:
                        score = "NA"
                    data[system].append(score)
                    pbar.update(1)
        df = pd.DataFrame(data)

        for system in SYSTEMS:
            df[system] = pd.to_numeric(df[system], errors='coerce')
            df[system] = df[system].round(1)
            df.fillna('NA', inplace=True)
        print(f"{metric}: ")
        print(df.T.to_latex(header=None))


    print("AVG: ALL_LANG_PAIRS_SUPPORTED")
    data = {
        'System': METRICS,
    }
    with tqdm(total=len(SYSTEMS) * len(METRICS) * len(ALL_LANG_PAIRS_SUPPORTED), desc="AVG: ALL_LANG_PAIRS_SUPPORTED") as pbar:
        for system in SYSTEMS:
            data[system] = []
            for metric in METRICS:
                scores = []
                for lang_pair in ALL_LANG_PAIRS_SUPPORTED:
                    split = LANG_PAIR2SPLIT[lang_pair]
                    src_lang, tgt_lang = lang_pair.split('-')

                    src_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{src_lang}"))
                    ref_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{tgt_lang}"))
                    sys_lines = readlines(os.path.join(OUTPUT_DIR, system, f"{split}.{lang_pair}.{tgt_lang}"))

                    assert check_equal_num_lines([src_lines, ref_lines, sys_lines])
                    scorer = eval(metric)
                    score = scorer(**{
                        "sys_lines": sys_lines,
                        "src_lines": src_lines,
                        "ref_lines": ref_lines,
                        "comet_model_name": comet_model_name,
                        "cache_dir": CACHE_DIR,
                        "bleurt_ckpt": bleurt_ckpt,
                        "batch_size": batch_size,
                        "tgt_lang": tgt_lang
                    })
                    scores.append(score)
                    pbar.update(1)
                avg_score = sum(scores) / len (scores)
                data[system].append(avg_score)
    df = pd.DataFrame(data)
    for system in SYSTEMS:
        df[system] = df[system].round(1)
    print(df.T.to_latex(header=None))


    print("AVG: HIGH_RES_LANG_PAIRS")
    data = {
        'System': METRICS,
    }
    with tqdm(total=len(SYSTEMS) * len(METRICS) * len(HIGH_RES_LANG_PAIRS), desc="AVG: HIGH_RES_LANG_PAIRS") as pbar:
        for system in SYSTEMS:
            data[system] = []
            for metric in METRICS:
                scores = []
                for lang_pair in HIGH_RES_LANG_PAIRS:
                    split = LANG_PAIR2SPLIT[lang_pair]
                    src_lang, tgt_lang = lang_pair.split('-')

                    src_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{src_lang}"))
                    ref_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{tgt_lang}"))
                    sys_lines = readlines(os.path.join(OUTPUT_DIR, system, f"{split}.{lang_pair}.{tgt_lang}"))

                    assert check_equal_num_lines([src_lines, ref_lines, sys_lines])
                    scorer = eval(metric)
                    score = scorer(**{
                        "sys_lines": sys_lines,
                        "src_lines": src_lines,
                        "ref_lines": ref_lines,
                        "comet_model_name": comet_model_name,
                        "cache_dir": CACHE_DIR,
                        "bleurt_ckpt": bleurt_ckpt,
                        "batch_size": batch_size,
                        "tgt_lang": tgt_lang
                    })
                    scores.append(score)
                    pbar.update(1)
                avg_score = sum(scores) / len (scores)
                data[system].append(avg_score)
    df = pd.DataFrame(data)
    for system in SYSTEMS:
        df[system] = df[system].round(1)
    print(df.T.to_latex(header=None))

    print("AVG: MID_RES_LANG_PAIRS")
    data = {
        'System': METRICS,
    }
    with tqdm(total=len(SYSTEMS) * len(METRICS) * len(MID_RES_LANG_PAIRS), desc="AVG: MID_RES_LANG_PAIRS") as pbar:
        for system in SYSTEMS:
            data[system] = []
            for metric in METRICS:
                scores = []
                for lang_pair in MID_RES_LANG_PAIRS:
                    split = LANG_PAIR2SPLIT[lang_pair]
                    src_lang, tgt_lang = lang_pair.split('-')

                    src_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{src_lang}"))
                    ref_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{tgt_lang}"))
                    sys_lines = readlines(os.path.join(OUTPUT_DIR, system, f"{split}.{lang_pair}.{tgt_lang}"))

                    assert check_equal_num_lines([src_lines, ref_lines, sys_lines])
                    scorer = eval(metric)
                    score = scorer(**{
                        "sys_lines": sys_lines,
                        "src_lines": src_lines,
                        "ref_lines": ref_lines,
                        "comet_model_name": comet_model_name,
                        "cache_dir": CACHE_DIR,
                        "bleurt_ckpt": bleurt_ckpt,
                        "batch_size": batch_size,
                        "tgt_lang": tgt_lang
                    })
                    scores.append(score)
                    pbar.update(1)
                avg_score = sum(scores) / len (scores)
                data[system].append(avg_score)
    df = pd.DataFrame(data)
    for system in SYSTEMS:
        df[system] = df[system].round(1)
    print(df.T.to_latex(header=None))

    print("AVG: LOW_RES_LANG_PAIRS")
    data = {
        'System': METRICS,
    }
    
    with tqdm(total=len(LOW_RES_SYSTEMS) * len(METRICS) * len(LOW_RES_LANG_PAIRS), desc="AVG: LOW_RES_LANG_PAIRS") as pbar:
        for system in LOW_RES_SYSTEMS:
            data[system] = []
            for metric in METRICS:
                scores = []
                for lang_pair in LOW_RES_LANG_PAIRS:
                    split = LANG_PAIR2SPLIT[lang_pair]
                    src_lang, tgt_lang = lang_pair.split('-')

                    src_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{src_lang}"))
                    ref_lines = readlines(os.path.join(RAW_DATA_DIR, f"{split}.{lang_pair}.{tgt_lang}"))
                    sys_lines = readlines(os.path.join(OUTPUT_DIR, system, f"{split}.{lang_pair}.{tgt_lang}"))

                    assert check_equal_num_lines([src_lines, ref_lines, sys_lines])
                    scorer = eval(metric)
                    score = scorer(**{
                        "sys_lines": sys_lines,
                        "src_lines": src_lines,
                        "ref_lines": ref_lines,
                        "comet_model_name": comet_model_name,
                        "cache_dir": CACHE_DIR,
                        "bleurt_ckpt": bleurt_ckpt,
                        "batch_size": batch_size,
                        "tgt_lang": tgt_lang
                    })
                    scores.append(score)
                    pbar.update(1)
                avg_score = sum(scores) / len (scores)
                data[system].append(avg_score)
    df = pd.DataFrame(data)
    for system in LOW_RES_SYSTEMS:
        df[system] = df[system].round(1)
    print(df.T.to_latex(header=None))


