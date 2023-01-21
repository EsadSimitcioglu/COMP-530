import csv
import hashlib


def hash_string(message):
    byte_input = message.encode()
    hash_output = hashlib.sha512(byte_input)
    return hash_output


def generate_dictionary_without_salt():
    f = open('../dictionary_without_salt.csv', 'w', newline='')
    rockyou_file = open("../rockyou.txt", 'r')

    writer = csv.writer(f)

    for dict_password in rockyou_file.readlines():
        dict_password = dict_password.replace("\n", "")
        dict_password_hashed = hash_string(dict_password).hexdigest()
        row = [dict_password, dict_password_hashed]
        writer.writerow(row)

    rockyou_file.close()
    f.close()


generate_dictionary_without_salt()

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
