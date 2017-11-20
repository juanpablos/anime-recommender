import csv
import random


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


def map_genres(genre_id_file):
    genre_mapping = {}
    with open(genre_id_file, 'r', encoding='utf-8') as genre_f:
        g = csv.reader(genre_f, delimiter=',')
        for line in g:
            if not line:
                continue
            genre_mapping[line[0]] = line[1]
    return genre_mapping


def reduce_user_lists(general_anime_file, user_id_file, user_list_file, out_file, split_test=True, test_size=0.25):
    anime_mapping = map_anime(general_anime_file)
    user_mapping = map_users(user_id_file)

    with open(user_list_file, 'r', encoding='utf-8') as f:
        with open(out_file, 'w', encoding='utf-8', newline="\n") as o:

            ff2 = csv.reader(f, delimiter=',')
            out = csv.writer(o, delimiter=',')

            out_dict = {}

            missing = []
            miss = False

            next(ff2)
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

                if isinstance(out_dict.get(user_mapping.get(l2[0])), set):
                    out_dict[user_mapping.get(l2[0])].add((anime_id, int(l2[5])))
                else:
                    out_dict[user_mapping.get(l2[0])] = {(anime_id, int(l2[5]))}

            if split_test:
                with open(out_file[:-4] + '_test.csv', 'w', encoding='utf-8', newline="\n") as o2:
                    out_test = csv.writer(o2, delimiter=',')

                    for k, v in out_dict.items():
                        number = int(test_size * len(v))
                        if number:
                            test = random.sample(v, k=number)
                            out_test.writerow([k] + test)
                            out.writerow([k] + list(v - set(test)))
                        else:
                            out.writerow([k] + list(v))
            else:
                for k, v in out_dict.items():
                    out.writerow([k] + list(v))

            if miss:
                missing.sort()
                print(missing)
                print(len(missing))
                with open('missing_ids.txt', 'w', newline="\n") as mi:
                    for m in missing:
                        mi.write(str(m) + "\n")


def genre_id_parser(genre_file, out_file):
    with open(genre_file, 'r', encoding='utf-8') as genre_f:
        with open(out_file, 'w', encoding='utf-8', newline="\n") as o:
            g = csv.reader(genre_f, delimiter=',')
            out = csv.writer(o, delimiter=',')

            genre_dict = set()

            next(g)
            for line in g:
                if not line:
                    continue
                if not line[1]:
                    continue
                genre_dict.add(line[1])

            for g_id, genre in enumerate(genre_dict):
                out.writerow([genre, g_id])


if __name__ == "__main__":
    reduce_user_lists('../Data/general.csv', '../Data/users_id.csv', '../Data/mal_filled.csv',
                      '../user_interactions.csv', split_test=False)

    # genre_id_parser("../Data/genre.csv", "../genre_id.csv")
