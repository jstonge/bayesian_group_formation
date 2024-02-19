import pyalex
from pyalex import Works, Authors

from tqdm import tqdm
from pathlib import Path
from time import sleep
import numpy as np
import re
from pymongo import MongoClient
from helpers import get_paper_data, write_jsonl, read_jsonl

# put your own please
pyalex.config.email = "jstonge1@uvm.edu"

def main():
    uri = f"mongodb://cwward:password@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20C"
    client = MongoClient(uri)
    db = client['papersDB']
    pipeline = [
        { '$sample': { 'size': 2000 } },
        { '$project': { 'id': 1, '_id': 0 } } 
    ]
    
    result = [a['id'].split("/")[-1] for a in db['authors_oa'].aggregate(pipeline)]
    
    cache_bg = Path(".cache_bg_author_paper")

    if cache_bg.exists() is False:
        cache_bg.mkdir()
        
    # For each author, we will keep 3 random papers.
    with open(cache_bg / "done.txt", "w") as f:
        [f.write(f"{_}\n") for _ in result]
    
    for chosen_author in tqdm(result, desc="Authors", total=len(result)):
        i = 0
        while i <= 3:
            print('here')
            # Randomly select a year between 1980 and 2023
            rdm_yr = np.random.choice(range(1980, 2024))
            # Within that year, get a random paper for the chosen author
            q = Works().filter(publication_year=rdm_yr, authorships={"author": {"id": chosen_author}}, doi=True).random()
            # We will use doi to get the paper details from the S2ORC API
            if q['doi'] is not None:
                id = "DOI:"+re.sub("https://doi.org/", "", q['doi'])
                paper_detail = get_paper_data([id])
                # We only keep papers found in s2orc and that have embeddings
                if paper_detail is not None and len(paper_detail) > 0 and paper_detail[0]['embedding'] is not None:
                    paper_detail = paper_detail[0]
                    fname_out = cache_bg / f"{paper_detail['paperId']}.jsonl"
                    

                    if fname_out.exists() is False:
                        write_jsonl(cache_bg / f"{paper_detail['paperId']}.jsonl", [paper_detail])
                        i += 1
            
            sleep(0.5)

    # insert many to DB
    fnames = [_ for _ in cache_bg.glob("*jsonl")]
    db['paper_author_embedding_bg'].insert_many([read_jsonl(fname) for fname in fnames])

if __name__ == "__main__":
    main()