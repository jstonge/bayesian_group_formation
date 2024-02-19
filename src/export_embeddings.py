import umap
import numpy as np
from pathlib import Path
from pyalex import Authors
from helpers import read_jsonl
import pandas as pd

def main():
    # target_aid = 'A5035455593'
    reducer = umap.UMAP()
    
    pap_dir = Path(".cache_paper")
    bg_pap_dir = Path(".cache_bg_paper")
    bg_papa_dir = Path(".cache_bg_author_paper")
    
    target_aid = 'A5040821463'
    target_name = Authors()[target_aid]['display_name']
    # fnames=[_ for _ in bg_pap_dir.glob("*jsonl")]
    fnames=[_ for _ in bg_papa_dir.glob("*jsonl")]
    fnames_authors=[_ for _ in pap_dir.glob("*jsonl") if _.stem.split("_")[0] == target_aid]
    
    # background = []
    # for fname in fnames:
    #     # fname=fnames[0]
    #     mydat=read_jsonl(fname)
    #     if mydat is not None and mydat[0] is not None:
    #         for myd in mydat[0]:
    #             if myd is not None:
    #                 if myd['embedding'] is not None:
                    

    embeddings=[]
    meta=[]
    for i,fname in enumerate(fnames+fnames_authors):
        # fname=fnames[0]
        # fname=fnames_authors[0]
        
        # we want a list of list
        mydat=[read_jsonl(fname)] if i < len(fnames) else read_jsonl(fname)

        if mydat is not None and mydat[0] is not None:
            for myd in mydat[0]:
                if myd is not None:
                    if myd['embedding'] is not None:
                        year = myd['year'] if i < len(fnames) else int(fname.stem.split("_")[1])
                        embeddings.append(myd['embedding']['vector'])
                        meta.append([
                            myd['title'],
                            myd['s2FieldsOfStudy'][0]['category'] if len(myd['s2FieldsOfStudy']) > 0 else None,
                            year,
                            True if i < len(fnames) else False
                            ])
    
    # ~700D -> 2D
    embedding2d = reducer.fit_transform(embeddings)
    
    # write to disk
    pd.concat([
        pd.DataFrame(embedding2d, columns=["x", "y"]),
        pd.DataFrame(meta, columns=["title", "fos", "year", "is_background"])
    ], axis=1).to_parquet("embedding.parquet")

