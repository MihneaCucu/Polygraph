import csv

def csv_to_sql(filename, table_name):
    output = open("output.txt", "w")
    words_file = open(filename)
    reader = csv.DictReader(words_file)
    insert_instruction = f"INSERT INTO {table_name}({','.join(reader.fieldnames)}) VALUES("
    for row in reader:

        output.write(insert_instruction + ','.join(["'" + val + "'" for val in row.values()]) + ")\n")

csv_to_sql("allsides_data.csv", "allsides_data")