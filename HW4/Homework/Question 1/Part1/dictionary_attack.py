digitalcorp_file = open("../digitalcorp.txt", 'r')
dictionary_file = open("../dictionary_without_salt.csv", 'r')


for line in digitalcorp_file.readlines()[1:]:
    line = line.split(",")
    username = line[0]
    user_hashed_password = line[1].replace("\n", "")

    for line_in_dictionary in dictionary_file.readlines():
        line_in_dictionary = line_in_dictionary.split(",")
        plain_password = line_in_dictionary[0]
        hashed_password = line_in_dictionary[1].replace("\n", "")

        if hashed_password == user_hashed_password:
            print(username + " password is " + plain_password)
            break

    dictionary_file.seek(0)