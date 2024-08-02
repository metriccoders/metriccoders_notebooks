x = ["sdf", 1, 3, 'A', 2395923]
for y in x:
    print(y)

x = 100

if x == 2:
    print("2")

elif x == 4:
    print("4")

elif x % 3 == 0:
    print("Divide by 3")

elif x % 2 == 0:
    print("Divide by 2")

else:
    print("Not divisible")


for i in range(1, 10):
    print(i)

x = [1,2,3, 4, 5]

x.pop()

x.append(10)

y = [7, 8]
x.extend(y)

print(x)

z = [10, 20, 30]
x.extend(z)
print(x)

x = [1,2,3]
i = 0
while i <= len(x):
    print(x[i])
    if i == 0:
        break
    if i == 1:
        continue
    i = i+1


