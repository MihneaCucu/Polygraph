
from newspaper import Article
def func_sort(el):
    return el[1]
def actualizare(dict, word):
    while len(word) > 0 and str.isalpha(word[-1]) == False:
        word = word[:-1]
    while len(word) > 0 and str.isalpha(word[0]) == False:
        word = word[1:]
    word = word.upper()
    if len(word) > 0:
        if word not in dict:
            dict.update({word: (1, 0)})
        else:
            dict[word] = (dict[word][0] + 1, 0)

#Actualizeaza dictionarul pasat ca argument, plasand pe pozitia a doua a tuplului procentajul aparitiilor cuvantului cheie in text
def procentaje_cuvinte(dict):
    cuv_total = sum([el[0] for el in dict.values()])
    for key in dict.keys():
        dict[key] = (dict[key][0], dict[key][0] / cuv_total * 100)
#Afiseaza cuvintele in ordinea numarului aparitiilor in text
def afisare_cuvinte(dict):
    sorted_items = sorted(dict.items(), key=lambda item: item[1][1], reverse = True)
    for (key, value) in sorted_items:
        print(key, value[0], "aparitii", value[1], "%")
#Construieste dictionarul cu cuvintele din text
def analiza_text(text, propozitii):
    dict = {}
    simb_final_prop = [".", "!", "?"]
    simb_blank = [" ", "\n", "\r"]
    ultimul_semn_de_pct = -1
    for i in range(len(text)):
        if text[i] in simb_final_prop and text[i+1] in simb_blank:
            propozitii.append(text[ultimul_semn_de_pct + 1:i + 1])
            ultimul_semn_de_pct = i
    text = text.split(" ")
    for word in text:
        word.strip("\n")
        if "\n" in word:
            word_list = word.split("\n")
            for w in word_list:
                actualizare(dict, w)
        else:
            actualizare(dict, word)
    return dict
def actualizeaza_total(d_total, d):
    for cheie in d.keys():
        if cheie in d_total:
            d_total[cheie] += d[cheie]
        else:
            d_total.update({cheie: d[cheie]})
    return d_total
#Construieste dictionarul celor mai folosite cuvinte din limba engleza
def construieste_dict_comune():
    dict_words = {}
    path = "unigram_freq.csv"
    words_file = open(path)
    raw_text = words_file.read()
    words = raw_text.split("\n")
    total_words = 0
    for pair in words:
        p = pair.split(',')
        if len(p) == 2 and p[1].isnumeric():
            dict_words[p[0].upper()] = int(p[1])
            total_words += int(p[1])
    for key in dict_words.keys():
        dict_words[key] /= total_words
    return dict_words
#Compara frecventa aparitiei cuvintelor in textul nostru cu frecventa lor in limba engleza
def comparare_cuvinte(dict_cuv, dict_words):
    dict_comp = {}
    for key in dict_cuv.keys():
        if key in dict_words.keys():
            dict_comp[key] = dict_cuv[key][1] / dict_words[key]
    sorted_items = sorted(dict_comp.items(), key=lambda item: item[1], reverse=True)
    for (key, value) in sorted_items:
        print(key, "apare cu factorul de", value)
#Extragem textul de la linkul specificat in variabila URL
url = 'https://www.bbc.com/sport/formula1/articles/cgenqvv9309o'
article = Article(url)
article.download()
article.parse()
propozitii = []
text = article.text
dict_cuvinte = analiza_text(text, propozitii)
procentaje_cuvinte(dict_cuvinte)
dict_words = construieste_dict_comune()
comparare_cuvinte(dict_cuvinte, dict_words)
