import pyalex
from pyalex import Works, Authors
from itertools import chain
import json
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from time import sleep
import argparse
from datetime import datetime
import random
import calendar
import re

from helpers import read_jsonl, write_jsonl, flatten

# put your own please
pyalex.config.email = "jstonge1@uvm.edu"


def shuffle_date_within_month(date_str):
    # Parse the date string to a datetime object
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Get the number of days in the month of the given date
    _, num_days_in_month = calendar.monthrange(date.year, date.month)

    # Generate a random day within the same month
    random_day = random.randint(1, num_days_in_month)

    # Create a new date with the randomly chosen day
    shuffled_date = date.replace(day=random_day)

    # Return the date in the desired format
    return shuffled_date.strftime("%Y-%m-%d")

def find_all_colabs(aid, author_work, names=False):
    """find all the collaborators of that author at time t"""
    out = []
    for w in author_work:
        out += [a['author']['display_name' if names else 'id'].split("/")[-1] for a in w['authorships']]
    return list(set(out)- set([aid]))

def get_collabs_of_collabs_time_t(target_collabs, target_yr, names=False):
    """find all the collaborators of collaborators of target Author at time t"""
    collabs_of_collabs_time_t = []
    for aid in target_collabs:
        out = get_all_work_time_t(aid, yr=target_yr)
        collabs_of_collabs_time_t += find_all_colabs(aid, out, names=names)
    return set(collabs_of_collabs_time_t)

def get_all_work_time_t(aid, yr, cache="author"):
    """
    Find all the works of a single target Author at time t. 
    This function calls OpenAlex API
    """
    cache_f = Path(f".cache_{cache}") / f'{aid}_{yr}.jsonl'
    if cache_f.exists():
        out = read_jsonl(cache_f)
    else:
        q = Works().filter(publication_year=yr,authorships={"author": {"id": aid}})
        out = []
        for record in chain(*q.paginate(per_page=200)):
            out.append(record)
        if len(out) > 0:
            write_jsonl(cache_f, out)
        else:
            return None
        print(f"We did {len(out)} requests ")
    return out

def determine_home_inst(aid, works):
    # determine target home institution this year
    all_inst_this_year = []
    for w in works:
        for a in w['authorships']:
            if a['author']['id'].split("/")[-1] == aid:
                all_inst_this_year += [i['display_name'] for i in a['institutions']]
    return Counter(all_inst_this_year).most_common(1)[0][0] if len(all_inst_this_year) > 0 else None

def read_cache(target_aid, cache):
    target_first_yr = auth_known_first_yr[target_aid]
    out = {}
    for yr in range(target_first_yr, 2022):
        cache_f = Path(f".cache_{cache}/{target_aid}_{yr}.jsonl")

        if cache_f.exists():
            out.update({yr: read_jsonl(cache_f)})
    return out


def get_author_data(target_aid, cache):
    """cache must be in ['author', 'author_bg']"""
    with open(f".cache_{cache}/done.txt", "r") as f:
        done_authors = f.read().splitlines()

    if target_aid not in done_authors: 
        for target_yr in tqdm(range(1950, 2024)):
            # target author; as many requests as there is work that year.
            target_author_work_time_t = get_all_work_time_t(target_aid, target_yr)

            if target_author_work_time_t is None:
                continue

            target_collaborators_time_t = find_all_colabs(target_aid, target_author_work_time_t)

            # collab of collabs
            for aid in target_collaborators_time_t:
                out = get_all_work_time_t(aid, yr=target_yr)

            sleep(1) # politeness

        with open(f".cache_{cache}/done.txt", "a") as f:
            f.write("\n")
            f.write(f"{target_aid}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--target_author", type=str, default="A5035455593")
    args = argparser.parse_args()
    target_aid=args.target_author


    # OUTPUT TO OBSERVABLE 

    papers = []
    
    with open(".cache_author/done.txt", "r") as f:
        chosen_authors = f.read().splitlines()

    auth_known_first_yr = {auth: 1970 for auth in chosen_authors}
    auth_known_first_yr['A5037256938'] = 2004
    auth_known_first_yr['A5088506012'] = 1988
    auth_known_first_yr['A5017032627'] = 2000
    auth_known_first_yr['A5037170969'] = 1950
    auth_known_first_yr['A5027136376'] = 2005
    auth_known_first_yr['A5009266404'] = 2005
    auth_known_first_yr['A5046878011'] = 1987
    auth_known_first_yr['A5017357771'] = 2007

    for target_aid in tqdm(chosen_authors):
        # target_aid = 'A5035455593'
        target_name = Authors()[target_aid]['display_name']
        
        dat = read_cache(target_aid, cache='author')

        set_all_collabs = set()  # Track all collaborators across all years
        set_collabs_of_collabs_never_worked_with = set()
        all_time_collabo = {}  # Reset for each year

        yr2age = { yr:i for i,yr in enumerate(range(min(dat.keys()), max(dat.keys())+1)) }
        for i, yr in enumerate(dat.keys()):
            # yr=2020
            time_collabo = {}  # Reset for each year
            dates_in_year = []  # List to keep track of dates for papers in this year
            new_collabs_this_year = set() # Track new collaborators for this year

            # collab of collabs
            target_collaborators_time_t = find_all_colabs(target_aid, dat[yr])
            collabs_of_collabs_time_t = get_collabs_of_collabs_time_t(target_collaborators_time_t, yr, names=True)

            # majority vote to determine target_institutio
            target_institution = determine_home_inst(target_aid, dat[yr])

            for work in dat[yr]:
                # work=dat[yr][2]
                if work['type'] != 'article' and work['language'] != 'en':
                    continue
                
                shuffled_date = shuffle_date_within_month(work['publication_date'])
                shuffled_auth_age = "1"+shuffled_date.replace(shuffled_date.split("-")[0], str(yr2age[yr]).zfill(3))
                # impossible leap year causes issues... omg
                shuffled_auth_age = shuffled_auth_age.replace("29", "28") if shuffled_auth_age.endswith("29") else shuffled_auth_age

                dates_in_year.append(shuffled_date)               

                # author info wrt to paper
                for a in work['authorships']:
                    # a=work['authorships'][1]
                    author_name = a['author']['display_name']

                    if author_name == target_name:
                        target_position = a['author_position']
                    else:
                        institutions = a['institutions'] if 'institutions' in a else []
                        
                        # Increment collaboration count for the current year
                        author_yearly_data = time_collabo.get(author_name, {'count': 0, 'institutions': {}})
                        author_yearly_data['count'] += 1
                        
                        # Increment institution count for the current year
                        for inst in institutions:
                            inst_name = inst['display_name']
                            author_yearly_data['institutions'][inst_name] = author_yearly_data['institutions'].get(inst_name, 0) + 1
                        
                        time_collabo[author_name] = author_yearly_data
                        all_time_collabo[author_name] = all_time_collabo.get(author_name, 0) + 1

                        # Add new collaborators to the set for all years
                        if author_name not in set_all_collabs:
                            new_collabs_this_year.add(author_name)


                papers.append({
                    'type': 'paper',
                    'target': target_name,
                    'year': shuffled_date,
                    'pub_year': work['publication_year'],
                    'doi': work['ids']['doi'] if 'doi' in work['ids'] else None,
                    'title': work['title'],
                    'author': ', '.join([_['author']['display_name'] for _ in work['authorships']]),
                    'author_age': shuffled_auth_age,
                    'author_age_i': i,
                    'cited_by_count': work['cited_by_count'],
                    'target_type': target_name+"-"+'paper',
                    'target_position': target_position
                    # 's2FieldsOfStudy': paper_details['s2FieldsOfStudy']
                    })


            set_collabs_of_collabs_never_worked_with.update(
                collabs_of_collabs_time_t - new_collabs_this_year - set_all_collabs - set([target_name])
                )

            # At the end of each year, append the author information with their total collaboration count
            if len(time_collabo) > 0:
                for author_name, author_data in time_collabo.items():

                    # Determine if it's a new or existing collaboration for the year
                    if author_name in (new_collabs_this_year - set_all_collabs):
                        if author_name in set_collabs_of_collabs_never_worked_with:
                            subtype = 'new_collab_of_collab'
                        else:
                            subtype = 'new_collab'
                    else:
                        subtype = 'existing_collab'

                    # Assign a date from the papers they collaborated on (if available)
                    author_date = random.choice(dates_in_year) if dates_in_year else str(yr)
                    shuffled_auth_age = "1"+author_date.replace(author_date.split("-")[0], str(yr2age[yr]).zfill(3))
                    # impossible leap year
                    shuffled_auth_age = shuffled_auth_age.replace("29", "28") if shuffled_auth_age.endswith("29") else shuffled_auth_age

                    # find coauthor institution name
                    shared_inst = None
                    max_institution = None
                    if author_data['institutions'] and target_institution:
                        max_institution = max(author_data['institutions'], key=author_data['institutions'].get)
                        if max_institution == target_institution:
                            shared_inst = max_institution

                    papers.append({
                        'target': target_name,
                        'year': author_date,  # Use one of the dates from this year's papers
                        'pub_year': author_date[0:4],  # Use one of the dates from this year's papers
                        'author_age': shuffled_auth_age,
                        'title': author_name,
                        'type': 'coauthor',
                        'acquaintance': subtype,
                        'yearly_collabo': author_data['count'],  # Total collaboration count for the year
                        'all_times_collabo': all_time_collabo[author_name],  # Total collaboration count
                        'shared_institutions': shared_inst,
                        'institutions': max_institution,
                        'institutions': max_institution,
                        'target_type': target_name+"-"+'coauthor'
                         })

                set_all_collabs.update(new_collabs_this_year)

    with open(f"timeline_prod.json", "w") as f:
        json.dump(papers, f)



    
    
    # GET MATRIX OF COAUTHORS ----------------------------

    # import xgi
    # import networkx as nx
    # from itertools import combinations
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # with open("timeline_prod.json") as f:
    #     data = json.load(f)
    
    # def flatten(l):
    #     return [item for sublist in l for item in sublist]
    
    # def plot_matrix_with_legend(mat):
    #     sorted_indices = np.argsort(-mat.sum(axis=1))
    #     sorted_mat = mat[sorted_indices][:, sorted_indices]

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    #     im = ax.imshow(sorted_mat, cmap='Blues', aspect='auto')
    #     ax.set_xlabel('Coauthor')
    #     ax.set_ylabel('Coauthor')
    #     cbar = fig.colorbar(im)
    #     cbar.set_label('Count')
    #     plt.show()

    # def get_papers(x):
    #     return [paper for paper in data 
    #               if paper['target'] == x 
    #               and paper['type'] == 'paper' 
    #               and paper['pub_year'] in chosen_years]
    
    # chosen_years = list(range(2019,2022))
    # lhd_papers = get_papers('Laurent Hébert‐Dufresne')
    # all_pairs_coauthors = flatten([list(combinations(_['author'].split(", "),2)) for _ in lhd_papers])
    # uniq_coauthors = set(flatten([_['author'].split(", ") for _ in lhd_papers]))
    # coauthor2idx = {coauthor: i for i, coauthor in enumerate(uniq_coauthors)}
    
    # # all_pairs_coauthors = Counter(all_pairs_coauthors)
    # mat = np.zeros((len(uniq_coauthors), len(uniq_coauthors))).astype(int)
    
    # done_pairs = set()  # Use a set for efficient lookup and handling pairs uniquely

    # for pair in all_pairs_coauthors:
    #     s, t = pair
    #     idx_s, idx_t = coauthor2idx[s], coauthor2idx[t]

    #     # Check if the pair or its reverse has been encountered
    #     if (s, t) in done_pairs or (t, s) in done_pairs:
    #         # Increment counts symmetrically for both (s, t) and (t, s)
    #         mat[idx_s, idx_t] += 1
    #         mat[idx_t, idx_s] += 1
    #     else:
    #         # Initialize counts to 1 for both (s, t) and (t, s) if not yet encountered
    #         mat[idx_s, idx_t] = 1
    #         mat[idx_t, idx_s] = 1
    #         # Add both (s, t) and (t, s) to done_pairs to indicate they've been processed
    #         done_pairs.add((s, t))
    #         done_pairs.add((t, s))

    # done_papers = [_['doi'] for _ in lhd_papers if _['doi'] is not None]

    # new_mat = mat.copy()
    # for coauthor in uniq_coauthors:
    #     x_papers = get_papers(coauthor)
    #     all_pairs_coauthors = flatten(
    #         [list(combinations(_['author'].split(", "),2)) for _ in x_papers
    #          if _['doi'] is not None and _['doi'] not in done_papers]
    #         )

    #     for pair in all_pairs_coauthors:
    #         s, t = pair
    #         if s in uniq_coauthors and t in uniq_coauthors:
    #             idx_s, idx_t = coauthor2idx[s], coauthor2idx[t]

    #             # Check if the pair or its reverse has been encountered
    #             if (s, t) in done_pairs or (t, s) in done_pairs:
    #                 # Increment counts symmetrically for both (s, t) and (t, s)
    #                 new_mat[idx_s, idx_t] += 1
    #                 new_mat[idx_t, idx_s] += 1
    #             else:
    #                 # Initialize counts to 1 for both (s, t) and (t, s) if not yet encountered
    #                 new_mat[idx_s, idx_t] = 1
    #                 new_mat[idx_t, idx_s] = 1
    #                 # Add both (s, t) and (t, s) to done_pairs to indicate they've been processed
    #                 done_pairs.add((s, t))
    #                 done_pairs.add((t, s))
                    
    #             done_papers += [_['doi'] for _ in x_papers if _['doi'] is not None]
   
    # # right now jsut the coauthors of LHD; and how people within
    # # with perimeter collaborate with each other.
    # plot_matrix_with_legend(new_mat)


    # OUTPUT SUMMARY SIMPLE ----------------------------
        
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # group size; density; variance of topic spread 
        
    # jgy=[_ for _ in papers if _['target'] == 'Jean-Gabriel Young']
    # jgy_pap=[_ for _ in jgy if _['type'] == 'paper']
    # jgy_coauth=[_ for _ in jgy if _['type'] == 'coauthor']
    # jgy_first_pos=Counter([(_['pub_year'], _['target_position']) for _ in jgy_pap if _['target_position'] == 'first'])
    # jgy_first_middle=Counter([(_['pub_year'], _['target_position']) for _ in jgy_pap if _['target_position'] == 'middle'])
    # jgy_first_last=Counter([(_['pub_year'], _['target_position']) for _ in jgy_pap if _['target_position'] == 'last'])
    # coauth_count=Counter([_['pub_year'] for _ in jgy_coauth])
    # pap_count=Counter([_['pub_year'] for _ in jgy_pap])
    # pap_count=Counter([_['pub_year'] for _ in jgy_pap])
    

    # fig, ax = plt.subplots(1,1)
    # sns.lineplot(x=pap_count.keys(), y=pap_count.values(), ax=ax)
    # sns.lineplot(x=[int(_) for _ in coauth_count.keys()], y=coauth_count.values(), ax=ax)


    # OUTPUT AUTHOR SUMMARY ----------------------------

    # out = {}
    # for p in papers:
    #     #p=papers[202]
    #     # Define the key for the current paper/coauthor based on target and publication year
    #     target_year = (p['target'], int(p['pub_year']))
        
    #     # Initialize the yearly data structure only if it doesn't exist for the target_year
    #     if target_year not in out:
    #         out[target_year] = {
    #             'target_position': {}, 
    #             'institutional_diversity': {}, 
    #             'coauthors': {},
    #             'age': 0,  # Initialize age; assuming you'll update or use it differently based on your needs
    #             'institutions': "",
    #             'number_of_papers': 0  # Initialize the number of papers
    #         }
        
    #     # Retrieve the existing data structure for updates
    #     yearly_data = out[target_year]

    #     if p['type'] == 'paper':
    #         # Increment the number of papers
    #         yearly_data['number_of_papers'] += 1

    #         # Update target_position count
    #         yearly_data['target_position'][p['target_position']] = yearly_data['target_position'].get(p['target_position'], 0) + 1
            
    #         # Update age; assuming last seen age is what you're interested in, otherwise adjust as needed
    #         yearly_data['age'] = p['author_age_i']
    #     elif p['type'] == 'coauthor':
    #         # Increment coauthor count
    #         coauth_name = p['title']  # Assuming this uniquely identifies the coauthor
    #         yearly_data['coauthors'][coauth_name] = yearly_data['coauthors'].get(coauth_name, 0) + 1

    #         # Handle institution, treating None as 'Unknown'
    #         institution = p['institutions'] if p['institutions'] is not None else 'Unknown'
    #         yearly_data['institutional_diversity'][institution] = yearly_data['institutional_diversity'].get(institution, 0) + 1
    #         yearly_data['institutions'] = p['shared_institutions']

    # for key, data in out.items():
    #     # Calculate the number of distinct institutions
    #     num_institutions = len(data['institutional_diversity'])
        
    #     # Calculate the number of coauthors
    #     num_coauthors = len(data['coauthors'])
        
        
    #     # Calculate the total count of positions
    #     total_positions = sum(data['target_position'].values())
        
    #     # Calculate the proportion of each position
    #     position_proportions = {position: count / total_positions for position, count in data['target_position'].items()}
        
    #     # Update the dictionary with the calculated values
    #     out[key]['institutional_diversity'] = num_institutions
    #     out[key]['coauthors'] = num_coauthors
    #     out[key]['target_position'] = position_proportions

    # data_to_flatten = []
    # for (target, pub_year), data in out.items():
    #     # Flatten the target_position proportions into separate fields
    #     positions_data = {f"position_{k}": v for k, v in data['target_position'].items()}
    #     row = {
    #         'target': target,
    #         'pub_year': pub_year,
    #         'age': data['age'],
    #         'nb_papers': data['number_of_papers'],
    #         'institutional_diversity': data['institutional_diversity'],
    #         'nb_coauthors': data['coauthors'],
    #         **positions_data  # Unpack the positions data
    #     }
    #     data_to_flatten.append(row)

    # # Create the DataFrame
    # df = pd.DataFrame(data_to_flatten).fillna(0)
    # df.to_csv("test_ml.csv", index=False)

    

    # BERTOPIC ----------------------------

    # from bertopic import BERTopic
    # from hdbscan import HDBSCAN
    
    # from umap import UMAP
    
    
    # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    # topic_model = BERTopic(embedding_model=embeddings, umap_model=umap_model)
    # topics, probs = topic_model.fit_transform(embeddings)

    # hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    # topic_model = BERTopic(hdbscan_model=hdbscan_model)
