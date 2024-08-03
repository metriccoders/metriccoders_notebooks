def fun1(n):
    """
    This is fun1
    :param n:
    :return:
    """
    return n * n
def fun2(n):
    """
    This is function 2
    :param n:
    :return:
    """
    return n - 1
def fun3(n):
    """
    This is function 3
    :param n:
    :return:
    """
    return n + 1
def fun4(n):
    """
    This is function 4
    :param n:
    :return:
    """
    return n / 2

def fun5(x,y):
    """
    This is function5
    :param x:
    :param y:
    :return:
    """
    return x ** y


def fun6(name, age):
    """
    This is function 6
    :param name:
    :param age:
    :return:
    """
    return name, age


def fun7(a=100, b=200):
    """
    This is function 7
    :param a:
    :param b:
    :return:
    """
    return a*b


def cheeseshop(kind, *arguments, **keywords):
    print("Kind:", kind)
    for arg in arguments:
        print(arg)

    for kw in keywords:
        print(kw, ":", keywords[kw])


def bakery_shop(*cricketers, **bowlers):
    """
    This is a bakery shop
    :param cricketers:
    :param bowlers:
    :return:
    """
    for c in cricketers:
        print("c")

    for x in bowlers:
        print(x, ":::", bowlers[x])

