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
import requests
from creds import s2orc_token
import re

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

def flatten(l):
    return [item for sublist in l for item in sublist]

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

def get_all_work_time_t(aid, yr):
    """find all the works of a single target Author at time t"""
    cache_f = Path(".cache_author") / f'{aid}_{yr}.jsonl'
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

    return out

def determine_home_inst(aid, works):
    # determine target home institution this year
    all_inst_this_year = []
    for w in works:
        for a in w['authorships']:
            if a['author']['id'].split("/")[-1] == aid:
                all_inst_this_year += [i['display_name'] for i in a['institutions']]
    return Counter(all_inst_this_year).most_common(1)[0][0] if len(all_inst_this_year) > 0 else None

def read_cache(target_aid):
    target_first_yr = auth_known_first_yr[target_aid]
    out = {}
    for yr in range(target_first_yr, 2022):
        cache_f = Path(f".cache_author/{target_aid}_{yr}.jsonl")

        if cache_f.exists():
            out.update({yr: read_jsonl(cache_f)})
    return out

def get_paper_data(ids):
  url = f'https://api.semanticscholar.org/graph/v1/paper/batch'

  # Define which details about the paper you would like to receive in the response
  params = {'fields': 'externalIds,title,s2FieldsOfStudy,embedding.specter_v2'}

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

def main():

    target_aid=args.target_author
    with open(".cache_author/done.txt", "r") as f:
        done_authors = f.read().splitlines()

    if target_aid not in done_authors: 
        for target_yr in tqdm(range(1950, 2024)):
            # target author
            target_author_work_time_t = get_all_work_time_t(target_aid, target_yr)

            if target_author_work_time_t is None:
                continue

            target_collaborators_time_t = find_all_colabs(target_aid, target_author_work_time_t)

            # collab of collabs
            for aid in target_collaborators_time_t:
                out = get_all_work_time_t(aid, yr=target_yr)

            sleep(1) # politeness

        with open(".cache_author/done.txt", "a") as f:
            f.write("\n")
            f.write(f"{target_aid}")

def write_paper_s2orc(fname):
    # fname=all_fnames[4]
    fout = Path(output_dir / f"{fname.stem}.jsonl")
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--target_author", type=str, default="A5035455593")
    args = argparser.parse_args()
    
    main()

    # Get embeddings for papers

    output_dir=Path(".cache_paper")
    all_fnames=list(Path(".cache_author").glob("*"))
    done_papers=[_.stem for _ in output_dir.glob("*")]
    all_fnames = [f for f in all_fnames if f.stem not in done_papers]
    
    if not output_dir.exists():
        output_dir.mkdir()
    
    for fname in tqdm(all_fnames):
        write_paper_s2orc(fname)
        sleep(.5)


    # OUTPUT TO OBSERVABLE 

    papers = []
    
    with open(".cache_author/done.txt", "r") as f:
        chosen_authors = f.read().splitlines()

    auth_known_first_yr = {auth: 1970 for auth in chosen_authors}
    auth_known_first_yr['A5037256938'] = 2004
    auth_known_first_yr['A5088506012'] = 1988
    auth_known_first_yr['A5017032627'] = 1950
    auth_known_first_yr['A5027136376'] = 2005
    auth_known_first_yr['A5009266404'] = 2005
    auth_known_first_yr['A5046878011'] = 1987
    auth_known_first_yr['A5017357771'] = 2007

    for target_aid in tqdm(chosen_authors):
        # target_aid = 'A5040821463'
        target_name = Authors()[target_aid]['display_name']
        
        dat = read_cache(target_aid)

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
                    'doi': work['ids']['doi'],
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
                        'target_type': target_name+"-"+'coauthor'
                         })

                set_all_collabs.update(new_collabs_this_year)

    with open(f"timeline_prod.json", "w") as f:
        json.dump(papers, f)



    # pap_dir = Path(".cache_paper")
    # fnames=list(pap_dir.glob("*jsonl"))

    # out=[]
    # for i,fname in enumerate(fnames):
    #     print(i)
    #     mydat=read_jsonl(fname)
    #     if mydat is not None and mydat[0] is not None:
    #         for myd in mydat[0]:
    #             if myd is not None:
    #                 if myd['embedding'] is not None:
    #                     out.append(myd['embedding']['vector'])
    
    # meta=[]
    # for i,fname in enumerate(fnames):
    #     print(i)
    #     mydat=read_jsonl(fname)
    #     if mydat is not None and mydat[0] is not None:
    #         for myd in mydat[0]:
    #             if myd is not None:
    #                 if myd['embedding'] is not None:
    #                     meta.append([myd['title'],myd['s2FieldsOfStudy'][0]['category'] if len(myd['s2FieldsOfStudy']) > 0 else None])
