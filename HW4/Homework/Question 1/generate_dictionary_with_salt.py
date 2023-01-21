# open the file in the write mode
import csv
import hashlib


def hash_string(message):
    byte_input = message.encode()
    hash_output = hashlib.sha512(byte_input)
    return hash_output


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
