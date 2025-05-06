import csv
import difflib
def incarca_baza_de_date(filepath="allsides_data.csv"):
    dict_surse = {}
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dict_surse[row['news_source'].lower()] = [row['rating'], int(row['agree']) + int(row['disagree']), float(row['perc_agree']), row['type'], row['confidence_level']]
    return dict_surse
def show_review(source_name, bias_map):
    match = difflib.get_close_matches(source_name.lower(), list(bias_map.keys()), n=1, cutoff=0.6)
    if match:
        lista = bias_map[match[0]]
        print(f"The source {source_name} has a {lista[0]} bias from {lista[1]} reviews with an accuracy of {lista[2]} and a confidence level of {lista[4]}")
    else:
        prefix_lookup(source_name, bias_map)
def prefix_lookup(source_name, bias_map):
    found_source = False
    for key in bias_map.keys():
        if key.startswith(source_name):
            lista = bias_map[key]
            print(f"The source {source_name} has a {lista[0]} bias from {lista[1]} reviews with an accuracy of {lista[2]} and a confidence level of {lista[4]}")
            found_source = True
            break
    if found_source == False:
        print(f"No close match found for '{source_name}'.")
def review(sursa):
    bias_db = incarca_baza_de_date()
    show_review(sursa, bias_db)

