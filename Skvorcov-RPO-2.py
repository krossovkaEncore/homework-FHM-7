import math


def read_float(prompt):
    # cчитывает число с плавающей точкой
  
    while True:
        try:
            return float(input(prompt).replace(",", "."))
        except ValueError:
            print("Ошибка: введите корректное число.")


def read_int(prompt, minimum=1):
    # считывает целое число, не меньшее minimum
  
    while True:
        try:
            value = int(input(prompt))
            if value >= minimum:
                return value
            print(f"Ошибка: число должно быть не меньше {minimum}.")
        except ValueError:
            print("Ошибка: введите целое число.")


# ============================================================
# Задача 1. Динамическая телепортация персонажа
# ============================================================

def task1():
    print("\n=== Задача 1. Динамическая телепортация ===")

    x_a = read_float("Введите xA: ")
    y_a = read_float("Введите yA: ")

    x_u = read_float("Введите xu: ")
    y_u = read_float("Введите yu: ")

    x_v = read_float("Введите xv: ")
    y_v = read_float("Введите yv: ")

    r_x = x_u + x_v
    r_y = y_u + y_v

    x_b = x_a + r_x
    y_b = y_a + r_y

    displacement = math.sqrt(r_x**2 + r_y**2)

    path_length = (
        math.sqrt(x_u**2 + y_u**2)
        + math.sqrt(x_v**2 + y_v**2)
    )

    print(f"\nРезультирующий вектор r = ({r_x}, {r_y})")
    print(f"Итоговая позиция B = ({x_b}, {y_b})")
    print(f"Модуль результирующего вектора = {displacement:.4f}")
    print(f"Длина пройденного пути = {path_length:.4f}")


# ============================================================
# Задача 2. Поворот и масштабирование
# ============================================================

def read_matrix(rows, cols, name):
    # вводит матрицу заданного размера
  
    print(f"\nВведите матрицу {name} размером {rows}x{cols}:")
    matrix = []

    for i in range(rows):
        row = []
        for j in range(cols):
            value = read_float(f"{name}[{i + 1}][{j + 1}] = ")
            row.append(value)
        matrix.append(row)

    return matrix


def read_vector(size, name):
    # вводит вектор заданной размерности
    print(f"\nВведите вектор {name}:")
    vector = []

    for i in range(size):
        vector.append(read_float(f"{name}[{i + 1}] = "))

    return vector


def matrix_vector_multiply(matrix, vector):
    # умножает матрицу на вектор
  
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError("Размерности матрицы и вектора несовместимы.")

    result = []

    for row in matrix:
        total = 0
        for j in range(len(vector)):
            total += row[j] * vector[j]
        result.append(total)

    return result


def matrix_multiply(a, b):
    # перемножает две совместимые матрицы
  
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Размерности матриц несовместимы.")

    if any(len(row) != len(a[0]) for row in a):
        raise ValueError("Первая матрица имеет неправильную структуру.")

    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("Вторая матрица имеет неправильную структуру.")

    rows = len(a)
    cols = len(b[0])
    common = len(b)

    result = []

    for i in range(rows):
        row = []
        for j in range(cols):
            total = 0
            for k in range(common):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)

    return result


def print_matrix(matrix, name):
    print(f"\n{name}:")
    for row in matrix:
        print(" ".join(f"{value:10.4f}" for value in row))


def task2():
    print("\n=== Задача 2. Поворот и масштабирование ===")

    x = read_vector(2, "x")
    R = read_matrix(2, 2, "R")
    S = read_matrix(2, 2, "S")

    x1 = matrix_vector_multiply(R, x)
    x2 = matrix_vector_multiply(S, x1)

    T = matrix_multiply(S, R)
  
    check = matrix_vector_multiply(T, x)

    print("\nПромежуточный вектор x1 = R*x:")
    print(x1)

    print("\nИтоговый вектор x2 = S*x1:")
    print(x2)

    print_matrix(T, "Матрица общего преобразования T = S*R")

    print("\nПроверка T*x:")
    print(check)

    if all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
           for a, b in zip(x2, check)):
        print("Проверка пройдена: T*x = x2.")
    else:
        print("Ошибка: результаты преобразований не совпадают.")


# ============================================================
# Задача 3. Параметризуемый полносвязный нейронный слой
# ============================================================

def dense_layer(x, W, b):
    """
    вычисляет z = W*x + b

    x - вектор длины N
    W - матрица размера MxN
    b - вектор длины M
    """
    if not W:
        raise ValueError("Матрица весов не должна быть пустой.")

    if any(len(row) != len(x) for row in W):
        raise ValueError("Число столбцов W должно совпадать с длиной x.")

    if len(b) != len(W):
        raise ValueError("Длина b должна совпадать с числом строк W.")

    wx = matrix_vector_multiply(W, x)

    return [wx[i] + b[i] for i in range(len(W))]


def task3():
    print("\n=== Задача 3. Полносвязный нейронный слой ===")

    N = read_int("Введите число входных признаков N: ")
    M = read_int("Введите число нейронов M: ")

    x = read_vector(N, "x")
    W = read_matrix(M, N, "W")
    b = read_vector(M, "b")

    try:
        z = dense_layer(x, W, b)

        print("\nРезультирующий вектор z = W*x + b:")
        print(z)

    except ValueError as error:
        print(f"Ошибка размерностей: {error}")


# ============================================================
# Главное меню программы
# ============================================================

def main():
    while True:
        print("\n========== МЕНЮ ==========")
        print("1. Динамическая телепортация")
        print("2. Поворот и масштабирование")
        print("3. Полносвязный нейронный слой")
        print("0. Выход")

        choice = input("Выберите задачу: ").strip()

        try:
            match choice:
                case "1":
                    task1()
                case "2":
                    task2()
                case "3":
                    task3()
                case "0":
                    print("Программа завершена.")
                    break
                case _:
                    print("Ошибка: выберите пункт от 0 до 3.")

        except (ValueError, OverflowError) as error:
            print(f"Не удалось выполнить вычисление: {error}")

if __name__ == "__main__":
    main()
