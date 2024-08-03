def print_numbers(data):
    for d in data:
        yield d*d


num = [10,20,30,40,50]
for x in print_numbers(num):
    print(x)