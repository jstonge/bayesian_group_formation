import pyalex
from pyalex import Works, Authors
from itertools import chain
import json
import xgi
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

pyalex.config.email = "jstonge1@uvm.edu"

def count_venues(res, top_n=10):
    return Counter([_['primary_location']['source']['display_name'] if _.get('primary_location') and _['primary_location'].get('source')  is not None else None for _ in res]).most_common(top_n)

def read_jsonl(fname):
    out=[]
    with open(fname, 'r') as file:
        # Read each line in the file
        for line in file:    
            # Parse the JSON string and add the resulting dictionary to the list
            out.append(json.loads(line))
    return out

def get_all_collabs(res):
    all_collabs = []
    for work in res:
        collab = []
        for author in work['authorships']:
            name=author['author']['id'].split("/")[-1]
            inst=author['institutions'][0]['display_name'] if author.get('institutions') else None
            collab.append((name, inst))

        all_collabs.append(collab)
    
    return all_collabs

def flatten(x):
    return [item for row in x for item in row]

def get_unique_authors(all_collabs):
    return list(sorted(set([c[0] for c in flatten(all_collabs)])))

def find_articles(res, venue):
    out = []
    for a in res:
        if a.get('primary_location') and a['primary_location'].get('source'):
            if a['primary_location']['source']['display_name'] == venue:
                out.append(a)
    return out

# test -------------------------------------------------------------------


res, meta = (Works().filter(publication_year=1993, language='en', type="Article")
                    .get(return_meta=True))

res, meta = (Works().filter(publication_year=1993, language='en', type="Article")
                    .filter(institutions={'country_code':'US'})
                    .filter(concepts={'id':['C18903297', '!C71924100', '!C185592680', '!C33070731', '!C55493867']})   # ecology but not medicine or chemistry
                    .sort(publication_date="asc")
                    .get(return_meta=True))


# get all the work in a given year --------------------------------------


def write_work_data(year):
    """Hardcoded for ecology/en/Article/US/"""
    # But not chemistry, medicine, Toxicology, or Molecular Biology
    list_concepts = ['C18903297', '!C71924100', '!C185592680', '!C33070731', '!C153911025', '!C55493867']
    query = (Works().filter(publication_year=year, language='en', type="Article")
                    .filter(institutions={'country_code':'US'})
                    .filter(concepts={'id': list_concepts}) )
    all_res = []
    for record in chain(*query.paginate(per_page=200)):
        all_res.append(record)

    with open(f'ecology_{year}_works.jsonl', 'w') as file:
        for entry in all_res:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

write_work_data(1993)


# ------------------------------------------------------------------------


res_1990 = read_jsonl("ecology_1990_works.jsonl")
res_1991 = read_jsonl("ecology_1991_works.jsonl")
res_1992 = read_jsonl("ecology_1992_works.jsonl")
res_1993 = read_jsonl("ecology_1993_works.jsonl")


assert len(res_1990) == 7380 # from meta above
assert len(res_1991) == 7504 # from meta above
assert len(res_1992) == 7935 # from meta above
assert len(res_1993) == 7923 # from meta above

count_venues(res_1993, 20)

foo_pubmed=find_articles(res_1993, 'PubMed')

foo_pubmed[1]

# [[(aid, inst), ....], [(aid, inst), ...]]
all_collabs = get_all_collabs(res_1991)

# List: sorted by alphatbetical order, useful for the indexing
all_authors = get_unique_authors(all_collabs)

# author id to our indexing scheme here
author2id = {c: i for i,c in enumerate(all_authors)}

# hyperedge list: [[idx1, idx2, idx3], [idx4], [...], ...]
all_collabs_index = []
for collab in all_collabs:
    all_collabs_index.append([author2id[_[0]] for _ in collab])


def get_all_authors(all_authors):
    
    # load cache
    if Path('ecology_authors.jsonl').exists():
        done_authors = [_['id'].split("/")[-1] for _ in read_jsonl('ecology_authors.jsonl')]

    total_authors = len(all_authors)
    max_requests_per_second = 10
    all_authors_data = []
    for i in tqdm(range(0, total_authors, max_requests_per_second), desc="Fetching Authors"):
        
        # check if authors already done
        authorstodo = [a for a in all_authors[i:(i+max_requests_per_second)] if a not in done_authors]
        
        # if so, continue to the next iteration
        if len(authorstodo) == 0:
            continue

        # Else, fetch the next batch of authors
        authors_batch = Authors()[authorstodo]
        
        # Add the batch to the list of all authors
        all_authors_data.extend(authors_batch)

        # Wait for a second to respect the rate limit
        time.sleep(1)
    
    # we keep appending 
    with open(f'ecology_authors.jsonl', 'a') as file:
        for entry in all_authors_data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')


all_authors_data = read_jsonl('ecology_authors.jsonl')


def sort_by_id_list(dict_list, id_order):
    id_to_dict = {d['id'].split("/")[-1]: d for d in dict_list}
    sorted_list = [id_to_dict[id] for id in id_order if id in id_to_dict]
    return sorted_list

# sorted in the same way than all_authors
all_authors_data_sorted = sort_by_id_list(all_authors_data, all_authors)

def find_papers(aid):
    out = []
    for work in all_collabs:
        people_in_work = [_[0] for _ in work]
        if aid in people_in_work:
            out.append(people_in_work)
    return out

find_papers('A5071107303')



def get_all_works_by_author(aid):
    """Not the same than when looking at counts_by_year in Author object"""
    myquery=Works().filter(authorships={'author': {'id':aid}})
    all_autho = []
    for record in chain(*myquery.paginate(per_page=200)):
        all_autho.append(record)
    return all_autho


get_all_works_by_author('A5071107303')



[(_['year'], _['works_count']) for _ in foo[1]]

H = xgi.Hypergraph(all_collabs_index)


yval = [len(H.edges.filterby("order", i)) for i in range(20)]
plt.bar(range(20), yval)
plt.title("1990 # co-authorships")





# ------------------------------------------------------------


target_aid = "A5078442658"
target_yr = 2021

target_author_work_time_t = Works().filter(publication_year=target_aid).filter(authorships={"author": {"id": target_aid}}).get()


def find_all_colabs(author_work):
    out = []
    for w in author_work:
        out += [a['author']['id'].split("/")[-1] for a in w['authorships']]
    return list(set(out)- set([target_aid]))

target_collaborators_time_t = find_all_colabs(target_author_work_time_t)

def get_collabs_of_collabs_time_t(target_collabs):
    collabs_of_collabs_time_t = []
    for aid in target_collabs:
        q = Works().filter(publication_year=target_aid).filter(authorships={"author": {"id": aid}})
        collab_work_time_t = []
        for record in chain(*q.paginate(per_page=200)):
            collab_work_time_t.append(record)
        collabs_of_collabs_time_t += find_all_colabs(collab_work_time_t)
    return set(collabs_of_collabs_time_t)    

collabs_of_collabs_time_t = get_collabs_of_collabs_time_t(target_collaborators_time_t)

target_author_work_time_t_plus_1 = Works().filter(publication_year=(target_aid+1)).filter(authorships={"author": {"id": target_aid}}).get()

target_collaborators_time_t_plus_1 = find_all_colabs(target_author_work_time_t_plus_1)

print(f"# target collabs at time t: {len(target_collaborators_time_t)}")
print(f"# collabs of collabs at time t: {len(collabs_of_collabs_time_t - set(target_collaborators_time_t))} (excluding target collabs)")
print(f"# target unique collabs at time t+1: {len(target_collaborators_time_t_plus_1)}")
print(f"# target collabs at time t+1 intersecting with collabs of collabs at time t:{len(set(target_collaborators_time_t_plus_1) & collabs_of_collabs_time_t)}")

