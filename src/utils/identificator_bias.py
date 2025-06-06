import csv, os
import difflib

def extrage_sursa(url):
    surse = url.split('/')[2].split('.')
    if surse[0] == 'www':
        return surse[1].lower()
    else:
        return surse[0].lower()

def incarca_baza_de_date():
    dict_surse = {}
    module_dir = os.path.dirname(__file__)
    filepath = os.path.join(module_dir, 'allsides_data.csv')
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['screen_name'] != 'NA':
                dict_surse[row['screen_name'].lower()] = [row['rating'], int(row['agree']) + int(row['disagree']), float(row['perc_agree']), row['type'], row['confidence_level']]
            else:
                dict_surse[row['news_source'].lower()] = [row['rating'], int(row['agree']) + int(row['disagree']), float(row['perc_agree']), row['type'], row['confidence_level']]

    return dict_surse

def show_review(source_name, bias_map):
    match = difflib.get_close_matches(source_name.lower(), list(bias_map.keys()), n=1, cutoff=0.6)
    if match:
        lista = bias_map[match[0]]
        return (f"The source {source_name} has a {lista[0]} bias from {lista[1]} reviews with an accuracy of {lista[2]} and a confidence level of {lista[4]}")
    else:
        prefix_lookup(source_name, bias_map)

def prefix_lookup(source_name, bias_map):
    found_source = False
    for key in bias_map.keys():
        if key.startswith(source_name):
            lista = bias_map[key]
            return (f"The source {source_name} has a {lista[0]} bias from {lista[1]} reviews with an accuracy of {lista[2]} and a confidence level of {lista[4]}%")
            found_source = True
            break
    if found_source == False:
        return (f"No close match found for '{source_name}'.")

def review(url):
    sursa = extrage_sursa(url)
    bias_db = incarca_baza_de_date()
    print(sursa)
    return show_review(sursa, bias_db)

