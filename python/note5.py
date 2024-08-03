from note4 import fun1
from note4 import fun2
from note4 import fun3
from note4 import fun4
from note4 import fun5
from note4 import fun6
from note4 import fun7
from note4 import cheeseshop
from note4 import bakery_shop

x = 100
print(fun1(x))

y = 250
print(fun2(y))

z = 251
print(fun3(z))

a = 290
print(fun4(a))

a = 292
b = 287
print(fun5(a, b))

x = "Test"
a=100
b=[200, 300, 400]
c = {}
c["a"] = 900
c["b"] = 1100



print(fun6("Suhas", 10))

print(fun7(a=276, b =492))

cheeseshop(kind=a, another=b, yet_another=c)

numbers=[100, 200, 300, 400, 500]
hash_numbers = {}
hash_numbers["a"] = 10
hash_numbers["b"] = 20
hash_numbers["c"] = 30

bakery_shop(one=numbers, news = hash_numbers)

x = list(range(3, 10))
print(x)

args = [3, 8]
print(list(range(*args)))

numbers = [1, 19]
print(list(range(*numbers)))

add_ten = lambda x: x+ 20
print(add_ten(10))

add_twenty = lambda x: x + 30
print(add_twenty(20))