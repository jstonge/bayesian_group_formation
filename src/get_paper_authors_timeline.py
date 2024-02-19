import json
import re
from time import sleep
from pathlib import Path
from tqdm import tqdm
from helpers import write_paper_s2orc, read_jsonl
from pymongo import MongoClient


def main():
    uri = f"mongodb://cwward:password@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20C"
    client = MongoClient(uri)
    db = client['papersDB']
    output_dir=Path(".cache_paper")
    if not output_dir.exists():
        output_dir.mkdir()

    all_fnames=list(Path(".cache_author").glob("*jsonl"))
    done_papers=[_.stem for _ in output_dir.glob("*jsonl")]
    len(done_papers)
    all_fnames = [f for f in all_fnames if f.stem not in done_papers]
    
    if not output_dir.exists():
        output_dir.mkdir()
    
    for fname in tqdm(all_fnames):
        write_paper_s2orc(fname, output_dir)
        sleep(.5)


    fnames = [_ for _ in output_dir.glob("*jsonl")]
    db['author_timeline'].insert_many([read_jsonl(fname) for fname in fnames])
