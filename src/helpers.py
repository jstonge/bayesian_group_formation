import requests
import json
import pyalex
from creds import s2orc_token
from pathlib import Path
import re

# put your own please
pyalex.config.email = "jstonge1@uvm.edu"


def write_paper_s2orc(fname, out_dir):
    # fname=all_fnames[4]
    fout = Path(out_dir / f"{fname.stem}.jsonl")
    if not fout.exists(): 
        print(fname)
        paper_dat=read_jsonl(fname)
        if paper_dat is not None:
            ids=[]
            for p in paper_dat:
                # p=paper_dat[0]
                if 'doi' in p['ids']:
                    ids+=["DOI:"+re.sub("https://doi.org/", "", p['ids']['doi'])]
                elif 'mag' in p['ids']:
                    ids+=["MAG:"+p['ids']['mag']]
                else:
                    print(f"no doi or mag for {p['title']}")
                    continue
                
            paper_detail=get_paper_data(ids)
            
            with open(fout, "w") as f:
                json.dump(paper_detail, f)
        else:
            print(f"no data for {fname.stem}")


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_paper_data(ids):
  url = f'https://api.semanticscholar.org/graph/v1/paper/batch'

  # Define which details about the paper you would like to receive in the response
  params = {
      'fields': 'externalIds,title,year,citationCount,influentialCitationCount,venue,s2FieldsOfStudy,embedding.specter_v2'
      }

  # Send the API request and store the response in a variable
  response = requests.post(url, 
                           headers= {"x-api-key" : f'{s2orc_token}' },
                           params=params, 
                           json={'ids': ids})
  if response.status_code == 200:
    return response.json()
  else:
    print(response.status_code)
    return None


def read_jsonl(fname):
    out=[]
    with open(fname, 'r') as file:
        # Read each line in the file
        for line in file:
            # Parse the JSON string and add the resulting dictionary to the list
            out.append(json.loads(line))
    return out


def write_jsonl(fname, out):
    with open(fname, 'a') as file:
        for entry in out:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')
