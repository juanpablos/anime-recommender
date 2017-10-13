import csv

user_mapping = {}
anime_mapping = {}

with open('users_id.csv', 'r', encoding='utf-8') as f1:
    ff1 = csv.reader(f1, delimiter=',')
    for l1 in ff1:
        user_mapping[l1[1]] = l1[0]

with open('general.csv', 'r', encoding='utf-8') as fa:
    ffa = csv.reader(fa, delimiter=',')
    id = 1
    for line in ffa:
        if line[0] != 'anime_id' and not anime_mapping.get(int(line[0])):
            anime_mapping[int(line[0])] = id
            id += 1


with open('new_anime_id.csv', 'w', encoding='utf-8', newline="\n") as o1:
    out1 = csv.writer(o1, delimiter=',')
    for k, v in anime_mapping.items():
        out1.writerow([k, v])
#
with open('mal_filled.csv', 'r', encoding='utf-8') as f2:
    with open('out_file.csv', 'w', encoding='utf-8') as o:
#
# with open('new_anime_id.csv', 'w', encoding='utf-8', newline="\n") as o1:
#     out1 = csv.writer(o1, delimiter=',')
#     for k, v in anime_mapping.items():
#         out1.writerow([k, v])

with open('mal_filled.csv', 'r', encoding='utf-8') as f2:
    with open('out_file.csv', 'w', encoding='utf-8', newline="\n") as o:

        ff2 = csv.reader(f2, delimiter=',')
        out = csv.writer(o, delimiter=',')

        user = 0
        the_list = [0] * len(anime_mapping)
        for l2 in ff2:
            if user == 0:
                user = user_mapping.get(l2[0])
            if user != user_mapping.get(l2[0]):
                out.writerow([user] + the_list)
                the_list = [0] * len(anime_mapping)
                user = user_mapping.get(l2[0])

            try:
                the_list[anime_mapping.get(int(l2[2]) - 1)] = l2[5]
            except:
                continue

        out.writerow([user] + the_list)
