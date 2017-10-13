import csv
import ast
import numpy as np

#id del anime, con el id de los generos diccionario
#id de los usuarios, lo mismo pero al id del anime

dicta = {} #diccionario anime ESTE
dictg = {} #Diccionario generos
dictu = {} #Diccionario usuarios
dictu2 = {} #Diccionario usuarios 2 CON ESTE
newid_dict = {}
def genres_dictionary():
    ide = 1
    with open("genre.csv") as f:
        reader = csv.reader(f, delimiter = ",")
        for line in reader:
            genre = line[1]
            if genre == "genre":
                continue
            if not dictg.get(genre):
                dictg[genre] = ide
                ide += 1

genres_dictionary()
#print(dictg)




#Diccionario de anime: se modifica el diccionario dicta, donde el valor del diccionario es el id del anime y se le entrega una lista con el id de los generos que tiene
#PD: se necesita antes haber hecho el diccionario de generos dictg
def anime_dictionary():
    with open("genre.csv") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            anime_id = line[0]
            genre = line[1]
            if anime_id == "anime_id" or genre == "genre":
                continue
            if not dicta.get(anime_id):
                if not isinstance(dicta.get(anime_id),list):
                    dicta[anime_id] = list()
            dicta[anime_id].append(dictg[genre])

anime_dictionary()
#print(dicta['1'])



with open('new_anime_id.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        newid_dict[line[0]] = line[1]

with open('anime_dict.csv', 'w') as ad:
    wr = csv.writer(ad)
    for a,g in dicta.items():
        wr.writerow([newid_dict[a], g])
        ad.flush()
"""
#Funcion que genera los valores de los usuarios. Al correr user_dictionary() se genera el diccionario para los usuarios.
#Es decir, cada usuario tiene su propio id. Se modifica el diccionario dictu
def users_dictionary():
    ide = 1
    with open("mal_filled.csv", encoding="utf8") as f:
        reader = csv.reader(f, delimiter = ",")
        for line in reader:
            user = line[0]
            if not dictu.get(user):
                dictu[user] = ide
                ide += 1

users_dictionary()
#print(dictu)


#Funcion que modifica el diccionario dictu2, con el id de usuarios y la lista de los id de anime que ha visto
#Hay que correr antes users_dictionary para obtener el diccionario dictu
def users_dictionary2():
    with open("mal_filled.csv", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",",)
        for line in reader:
            anime_id = line[2]
            user = line[0] #dictu[user] = user_id
            if not dictu2.get(dictu[user]):
                if not isinstance(dictu2.get(dictu[user]),list):
                    dictu2[dictu[user]] = list()
            dictu2[dictu[user]].append(anime_id)

users_dictionary2()
#print(dictu2)




with open("users_id.csv", 'w', newline='\n', encoding='utf-8') as cf:
    writer = csv.writer(cf)
    for u, i in dictu.items():
        try:
            writer.writerow([i, u])
        except:
            print(u)
        cf.flush() 


with open("factorization_machine.csv", 'w', newline='\n', encoding='utf-8') as cf:
    writer = csv.writer(cf)
    s = set()
    for u,a in dictu2.items():
        for anime in a:
            try:
                writer.writerow([u, anime, dicta[anime]])
            except:
                s.add(anime)
            cf.flush()  
    print(s)
"""