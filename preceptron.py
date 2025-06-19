import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.activation = self._step_function

    def _step_function(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.array([self.activation(x) for x in linear_output])
        return y_pred

def input_data():
    n_features = int(input("Введите количество признаков (фичей): "))
    n_samples = int(input("Введите количество обучающих примеров: "))
    
    X = []
    y = []
    print("\nВведите обучающие данные (через пробел):")
    for i in range(n_samples):
        while True:
            data = input(f"Пример {i+1} (признаки + класс): ").split()
            if len(data) == n_features + 1:
                try:
                    features = list(map(float, data[:-1]))
                    label = int(data[-1])
                    X.append(features)
                    y.append(label)
                    break
                except ValueError:
                    print("Ошибка! Введите числа.")
            else:
                print(f"Ожидается {n_features} признаков и 1 класс. Попробуйте снова.")
    
    return np.array(X), np.array(y)

def main():
    print("=== Простой перцептрон для бинарной классификации ===")
    X_train, y_train = input_data()
    
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    print("\nОбучение завершено!")
    print(f"Веса: {perceptron.weights}, Смещение: {perceptron.bias}")
    
    # Тестирование
    print("\n=== Тестирование ===")
    while True:
        test_input = input("Введите тестовый пример (признаки через пробел) или 'q' для выхода: ")
        if test_input.lower() == 'q':
            break
        
        try:
            test_data = list(map(float, test_input.split()))
            if len(test_data) != len(perceptron.weights):
                print(f"Ожидается {len(perceptron.weights)} признаков. Попробуйте снова.")
                continue
            
            prediction = perceptron.predict(np.array([test_data]))
            print(f"Предсказанный класс: {prediction[0]}")
        except ValueError:
            print("Ошибка! Введите числа.")

if __name__ == "__main__":
    main()