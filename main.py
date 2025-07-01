import numpy as np
import time

# Вхідні дані (фактори)
X = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1]
])

# Очікувані результати (йти = 1, не йти = 0)
y = np.array([1, 1, 0, 1, 0])

# Ініціалізація ваг (від 0 до 1)
weights = np.random.uniform(0, 1, size=(7,))
bias = np.random.uniform(0, 1)  # Поріг може бути будь-яким
learning_rate = 0.1


def activation_function(value):
    return 1 if value >= bias else 0


# Навчання персептрона
start_time = time.time()
epochs = 20  # Кількість проходів через дані

for epoch in range(epochs):
    total_error = 0
    print(f"\nEpoch {epoch + 1}:")
    for i in range(len(X)):
        weighted_sum = np.dot(X[i], weights)
        prediction = activation_function(weighted_sum)
        error = y[i] - prediction
        total_error += abs(error)

        if error != 0:
            print(f"  Помилка на прикладі {X[i]}: Очікуваний = {y[i]}, Отриманий = {prediction}")

        # Оновлення ваг та порогу
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

    print(f"  Total Error = {total_error}")
    if total_error == 0:
        break

training_time = time.time() - start_time

print("\nНавчання завершено!")
print(f"Час навчання: {training_time:.4f} секунд")
print("Фінальні ваги:", weights)
print("Фінальний поріг:", bias)

# Тестування персептрона
test_cases = np.array([
    [1, 1, 0, 1, 0, 1, 0],  # Приклад 1
    [0, 1, 1, 1, 1, 0, 1],  # Приклад 2
    [1, 0, 0, 1, 1, 1, 0],  # Приклад 3
    [0, 0, 1, 0, 1, 1, 1],  # Приклад 4
    [1, 1, 1, 0, 0, 0, 1]  # Приклад 5
])

factors = ["Хороший виконавець", "Хороша погода", "Достатньо грошей", "Хороший настрій", "Друг йде", "Є їжа",
           "Є алкоголь"]

for i, test in enumerate(test_cases):
    weighted_sum = np.dot(test, weights)
    result = activation_function(weighted_sum)
    reason = []

    for j in range(len(test)):
        if test[j] == 1:
            reason.append(f"{factors[j]} (вага: {weights[j]:.4f})")

    decision = 'Йти на концерт' if result == 1 else 'Не йти на концерт'
    print(f"\nПриклад {i + 1}: {test} -> {decision}")
    print(f"Причини: {', '.join(reason) if reason else 'Жодного вагомого фактора'}")
    print(f"Сума зважених факторів: {weighted_sum:.4f}")
    print(f"Поріг: {bias:.4f}")
    print(
        f"Результат порівняння: {'>= порога (Йти на концерт)' if weighted_sum >= bias else '< порога (Не йти на концерт)'}")
