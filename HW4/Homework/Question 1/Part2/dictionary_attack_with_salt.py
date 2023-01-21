# open the file in the write mode
import csv
import hashlib


def hash_string(message):
    byte_input = message.encode()
    hash_output = hashlib.sha512(byte_input)
    return hash_output


digitalcorp_file = open('../salty-digitalcorp.txt', 'r')
rockyou_file = open("../rockyou.txt", 'r')


for line in digitalcorp_file.readlines()[1:]:
    line = line.split(",")
    username = line[0]
    salt = line[1]
    user_hashed_salt_password = line[2].replace("\n","")

    for password in rockyou_file.readlines()[1:]:
        plain_password = password.replace("\n","")

        salt_plain_password = salt + plain_password
        salt_hash_password = hash_string(salt_plain_password).hexdigest()

        if salt_hash_password == user_hashed_salt_password:
            print(username + " password is " + plain_password)
            break

        plain_password_salt = plain_password + salt
        hash_password_salt = hash_string(plain_password_salt).hexdigest()

        if hash_password_salt == user_hashed_salt_password:
            print(username + " password is " + plain_password)
            break

    rockyou_file.seek(0)
