import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.linspace(0, 10, 100)
true_W = 3
true_b = 5
y_true = true_W * X + true_b + np.random.randn(100) * 2

class LinearRegression:
    def __init__(self):
        self.W = np.random.randn()
        self.b = np.random.randn()
        self.losses = []
    
    def forward(self, x):
        return self.W * x + self.b
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def gradient_descent(self, x, y_true, learning_rate=0.001):
        y_pred = self.forward(x)
        n = len(x)
        
        dW = (2/n) * np.sum((y_pred - y_true) * x)
        db = (2/n) * np.sum(y_pred - y_true)
        
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dW, db
    
    def train(self, x, y_true, epochs=1000, learning_rate=0.001, verbose=True):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.mse_loss(y_pred, y_true)
            self.losses.append(loss)
            dW, db = self.gradient_descent(x, y_true, learning_rate)
            
            if verbose and epoch % 100 == 0:
                print(f"Эпоха {epoch}: loss = {loss:.4f}, W = {self.W:.4f}, b = {self.b:.4f}")
        
        return self.losses

model = LinearRegression()

print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:")
print(f"W = {model.W:.4f}, b = {model.b:.4f}")
print(f"Нужно получить: W = 3, b = 5")
print("\nОбучаю...\n")

losses = model.train(X, y_true, epochs=500, learning_rate=0.001)

print(f"\nИТОГ:")
print(f"W = {model.W:.4f} (должен быть 3)")
print(f"b = {model.b:.4f} (должно быть 5)")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y_true, alpha=0.5, label='Данные с шумом')
plt.plot(X, model.forward(X), 'r-', linewidth=2, label=f'y = {model.W:.2f}x + {model.b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Модель vs реальность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.title('Как ошибка уменьшается')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nПРОВЕРКА:")
test_x = np.array([2.5, 5.0, 7.5])
true_y = 3 * test_x + 5
pred_y = model.forward(test_x)

for i in range(len(test_x)):
    print(f"x={test_x[i]}: правильно y={true_y[i]:.1f}, модель дала={pred_y[i]:.2f}, разница={abs(true_y[i]-pred_y[i]):.2f}")