
from pymongo import MongoClient
from pathlib import Path
import json
import re
from time import sleep
from helpers import get_paper_data, write_jsonl, read_jsonl

def main():
    uri = f"mongodb://cwward:password@wranglerdb01a.uvm.edu:27017/?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
    client = MongoClient(uri)
    db = client['papersDB']
    pipeline = [
        { '$sample': { 'size': 10000 } },
        { '$match': { 'doi': { '$ne': None, '$exists': True } } },
        { '$project': { 'doi': 1, '_id': 0 } } 
    ]
    result = list(db['works_oa'].aggregate(pipeline))
    batch_size = 100
    
    output_bg = Path(".cache_bg_paper")
    if output_bg.exists() is False:
        output_bg.mkdir()
    
    for i in range(0, len(result), batch_size):
        ids = ["DOI:"+re.sub("https://doi.org/", "", _['doi']) for _ in result[i:i + batch_size]]
        # Get paper details using DOIs from the S2ORC API
        paper_detail = get_paper_data(ids)
        for paper in paper_detail:
            # paper = paper_detail[0]
            if paper is not None:
                fname_out = output_bg / f"{paper['paperId']}.jsonl"
                if fname_out.exists() is False:
                    with open(fname_out, "w") as f:
                        json.dump(paper, f)
        sleep(0.5)

    fnames = [_ for _ in output_bg.glob("*jsonl")]
    db['paper_embedding_bg'].insert_many([read_jsonl(fname) for fname in fnames])



if __name__ == "__main__":
    main()
