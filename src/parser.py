import csv


def map_users(user_id_file):
    user_mapping = {}
    with open(user_id_file, 'r', encoding='utf-8') as f1:
        ff1 = csv.reader(f1, delimiter=',')
        for l1 in ff1:
            user_mapping[l1[1]] = int(l1[0])
    return user_mapping


def map_anime(general_anime_file):
    anime_mapping = {}
    with open(general_anime_file, 'r', encoding='utf-8') as fa:
        ffa = csv.reader(fa, delimiter=',')
        next(ffa)
        for line in ffa:
            if anime_mapping.get(int(line[1])):
                raise Exception("Index already exists")
            anime_mapping[int(line[1])] = int(line[0])
    return anime_mapping


def reduce_user_lists(general_anime_file, user_id_file, user_list_file, out_file):
    anime_mapping = map_anime(general_anime_file)
    user_mapping = map_users(user_id_file)

    with open(user_list_file, 'r', encoding='utf-8') as f2:
        with open(out_file, 'w', encoding='utf-8', newline="\n") as o:

            ff2 = csv.reader(f2, delimiter=',')
            out = csv.writer(o, delimiter=',')

            out_dict = {}

            missing = []
            miss = False

            for l2 in ff2:
                try:
                    anime_id = anime_mapping.get(int(l2[2]))
                    if not anime_id:
                        if not miss:
                            miss = True
                        if not int(l2[2]) in missing:
                            missing.append(int(l2[2]))
                            print(int(l2[2]))
                        continue
                except:
                    raise Exception("Id not found in map")

                if isinstance(out_dict.get(user_mapping.get(l2[0])), list):
                    out_dict[user_mapping.get(l2[0])].append((anime_id, int(l2[5])))
                else:
                    out_dict[user_mapping.get(l2[0])] = [(anime_id, int(l2[5]))]

            for k, v in out_dict.items():
                out.writerow([k] + v)

            if miss:
                missing.sort()
                print(missing)
                print(len(missing))
                with open('missing_ids.txt', 'w', newline="\n") as mi:
                    for m in missing:
                        mi.write(str(m) + "\n")


if __name__ == "__main__":
    reduce_user_lists('../Data/general.csv', '../Data/users_id.csv', '../Data/mal_filled.csv',
                      '../user_interactions.csv')
