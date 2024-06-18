import csv

def data_write_csv(filename, data,print_data=True):
    if print_data:
        print(data)
    with open(filename, "a", encoding="utf-8", newline='') as w:
        writer = csv.writer(w)
        writer.writerow(data)