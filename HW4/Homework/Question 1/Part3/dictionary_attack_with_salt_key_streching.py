# open the file in the write mode
import csv
import hashlib
import itertools


def hash_string(message):
    byte_input = message.encode()
    hash_output = hashlib.sha512(byte_input)
    return hash_output


digitalcorp_file = open('../keystreching-digitalcorp.txt', 'r')
rockyou_file = open("../rockyou.txt", 'r')

for line in digitalcorp_file.readlines()[1:]:
    line = line.split(",")
    username = line[0]
    salt = line[1]
    user_hashed_salt_password = line[2].replace("\n", "")

    for permutations_counter in range(0, 6):
        for password in rockyou_file.readlines()[1:]:
            x_prev = ""
            plain_password = password.replace("\n", "")

            for _ in range(2000):
                hash_list = [x_prev, plain_password, salt]
                permutations = list(itertools.permutations(hash_list))
                x = ''.join(map(str, permutations[permutations_counter]))
                hash_x = hash_string(x).hexdigest()

                if hash_x == user_hashed_salt_password:
                    print(username + " password is " + plain_password)
                    break

                x_prev = hash_x

        rockyou_file.seek(0)
