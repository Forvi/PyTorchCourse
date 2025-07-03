Лавров Никита, РИ-230944

# Линейная и логистическая регрессия

## Задание 1: Модификация существующих моделей
___
- Добавьте L1 и L2 регуляризацию
- Добавьте early stopping
___

Для данного задания было создано 2 файла:

Регрессия

~~~
./lesson2/homework/homework_model_reg_modification.py
~~~


Классификация

~~~
lesson2/homework/homework_model_reg_modification.py
~~~

Сделал я так для удобства. Также перед началом работы я поискал 3 простеньких небольших датасета:
- Medical Cost Personal (регрессия)
- Extrovert vs. Introvert Behavior Data (классификация)
- Superhero_abilities_dataset (многоклассовая классификация)

и сразу начал создавать модели с ними, тестировал различные параметры в процессе, потому мог что-то не сохранить.

## 1.1 Расширение линейной регрессии

Для создания модели линейной регрессии в PyTorch достаточно наследоваться от torch.nn.Module и вызвать родительский конструктор, а также добавить поле с nn.Linear и метод forward, который будет вычислять линейную функцию f(x) = w * x + b

~~~
class LinearRegressionTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        
    def forward(self, X: torch.Tensor):
        return self.linear(X)
~~~

Сам модуль с обучением первоначально выглядел примерно так:

~~~
X, y = make_data(n=1000, noise=0.1)

dataset = Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MyLinearRegression(in_features=1)
lr = .05

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    
    for i, (batch_X, batch_y) in enumerate(dataloader):
        y_pred = model(batch_X)
        loss = mse(y_pred, batch_y)
        total_loss += loss
        
        model.zero_grad()
        model.backward(batch_X, batch_y, y_pred)
        model.step(lr)
    
    avg_loss = total_loss / (i + 1)
~~~

Метрики на моем датасете с данным вариантом обучение уже хорошо себя показывает: loss = ~0.9 

В задании необходимо было добавить L1 и L2 регуляризацю и early stopping.

Регуляризация — это метод добавления штрафа к функции потерь модели, чтобы предотвратить переобучение, ограничивая величину весов модели.

L1-регуляризация добавляет к функции потерь сумму абсолютных значений весов

$$
L_{\text{1}} = Loss + \lambda \sum_{i} |w_i|
$$

а L2 квадрат значений

$$
L_{\text{2}} = Loss + \lambda \sum_{i} w_i^2
$$

Чтобы применить этот метод, нужно изменить функцию оптимизации,
~~~
optimizer = optim.LBFGS(model.parameters(), lr=1.0)
~~~

объявить некоторые константы и добавить формулы

~~~
L1_LAMBDA = 0.001
L2_LAMBDA = 0.001
...
l1_norm = sum(p.abs().sum() for p in model.parameters())
l2_norm = sum(p.pow(2).sum() for p in model.parameters())
loss = mse_loss + L1_LAMBDA * l1_norm + L2_LAMBDA * l2_norm
~~~

Теперь метрики в разы лучше - loss = 0.5343, при этом на обучающей выборке loss = ~0.3670. 

Early stopping или ранняя остановка - это форма регуляризации, используемая для избежания переобучения при обучении модели с помощью итеративного метода, например, такого, как градиентный спуск.

Если говорить просто, то устанавливаются допустимые границы при обучении и при их превышении, мы останавливаем обучение, сохранив лучшие результаты.

~~~
MAX_EPOCHS = 10
...
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    epochs_no_improve = 0
    torch.save(model.state_dict(), './lesson2/homework/models/linreg_best_earlystop.pth')
    print(f"Loss improved, model saved!")
else:
    epochs_no_improve += 1
    print(f"No improvement for {epochs_no_improve} epochs")

if epochs_no_improve >= MAX_EPOCHS:
    print(f"Early stopping at epoch {epoch}")
    break
~~~

Я выбрал 10 эпох и если модель не будет показывать улучшения, обучение приостановится, а модель сохранится с лучшими реузльтатами.

Также для реализации нужно добавить функцию clojure() с некоторой логикой.

~~~
def closure():
    optimizer.zero_grad()
    y_pred = model(batch_x)
    mse_loss = criterion(y_pred, batch_y)

    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())

    loss = mse_loss + L1_LAMBDA * l1_norm + L2_LAMBDA * l2_norm
    loss.backward()
    return loss
~~~

Сильно метрику данный метод не повысил, но зато добавил стабильности, то есть, если до этого при нескольких прогонах модели, она выдавала немного различный результат (приблизительно на 0.05-0.08), то теперь результат меняется минимально.

---

### 1.2 Расширение логистической регрессии
---
Модифицируйте существующую логистическую регрессию:
- Добавьте поддержку многоклассовой классификации
- Реализуйте метрики: precision, recall, F1-score, ROC-AUC
- Добавьте визуализацию confusion matrix
---

Далее обучение логистической регрессии. Здесь можно переиспользовать прошлый код, но с некоторыми поправками
~~~
class LogisticRegressionTorch(nn.Module):
    <достаточно поменять название>
~~~

~~~
...
model = LinearRegressionTorch(in_features=features_size)
criterion = nn.BCEWithLogitsLoss()
...
~~~

Эта модель показывает результаты Test loss: 0.3121, Test accuracy: 0.9131 на небольшом датасете.

Для реализации многоклассовой классификации нужно изменить класс,
~~~
class MulticlassModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.linear(x) 
~~~
 а именно добавить значение выходного слоя. Также нужно изменить функцию потерь на 
 ~~~
 nn.CrossEntropyLoss
 ~~~
и функцию активации на сигмоиду, которая возвращает значение в диапазоне [0; 1]

$$
f(x) = \frac 1{ 1 + e^{-x}}
$$

После изменения модели, я использовал другой датасет (с несколькими классами) и столкнулся с переобучением модели, которое возможно смогли бы исправить L1-L2 регуляризция, но я, к сожалению, решил отложить на потом эту проверку.

loss=0.3210, accuracy=1.0000 -> Test loss: 2.7046, Test accuracy: 0.0000

Для добавления метрик можно использовать библиотеку scikit-learn:
~~~
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('./lesson2/homework/plots/confusion_matrix1.png') 
~~~

---
## Задание 2: Работа с датасетами
### 2.1 Кастомный Dataset класс 
PyTorch позволяет легко и гибко работать с массивами данных с помощью двух методов - создание собственного объекта датасета с кастомной логикой и итератор Dataloader. Для реализации достаточно наследовать класс torch.utils.data.Dataset и переопределить 3 основных метода
~~~
    def __init__(self, **<входные данные>):
        <поля>
    
    def __getitem__(self, index: int):
        <логика возвращения>
        return torch.tensor(...)
        
    def __len__(self) -> int:
        return len(self.data)
~~~

Я постарался реализовать гибкий датасет, чтобы с ним можно было работать как для задачи регрессии, так и для задачи классификации, хотя получилось даже и для других задач. В основе лежит pandas.DataFrame() для удобной манипуляции над данными, а обработка вынесена в отдельные методы. Я использовал основные методы обработки данных
~~~
# Декодирует категориальные признаки алгоритмом OneHotEncoder
def decode_categorical(self) -> None:
    self.data = ...

# Декодирует бинарные признаки алгоритмом LabelEncoder
def decode_binary(self) -> None:
    self.data = ...
    
# Нормализует признаки алгоритмом StandardScaler
# Возвращает tensor
def normalize(self, target: str = None) -> None:
    self.data = ...

# Удаялет NaN
def remove_nan(self):
    self.data = self.data.dropna()

# Удаляет выбросы
def remove_outlier(self) -> None:
    self.data = ...

# Возвращает датасет в виде pandas DateFrame
def get_dataframe(self) -> pd.DataFrame:
        return self.data

# Возвращает датасет в виде torch Tensor
def get_tensor(self) -> torch.Tensor:
    return torch.tensor(...)
~~~

Чтобы инициализировать датасет, нужно передать 
- путь до файла
- названия числовых колонок
- названия категориальных колонок
- названия бинарных колонок
- название таргета

Пример
~~~
path = 'lesson2/homework/data/insurance.csv'
num = ['age', 'bmi', 'charges', 'children']
cat = ['region']
binary = ['sex', 'smoker']
targ = ['charges']
dataset = CsvDataset(file_path=path, num_cols=num, cat_cols=cat, binary_cols=binary, target=targ)
dataset.decode_categorical()
dataset.decode_binary()
dataset.remove_outlier()
t = dataset.get_dataframe()
~~~

### 2.2 Эксперименты с различными датасетами 
Сначала я сделал Dataset, поэтому обучил модели раньше, также я старался сохранять модели в ./lesson2/homework/models

---
## Задание 3: Эксперименты и анализ

### 3.1 Исследование гиперпараметров
Я поэксперементировал над линейной регрессией, меняя learning rate, размеры батчей и оптимизационные модели. 

Всего я сохранил 5 вариаций:
- Стандартная с SGD и LR=0.5, 32 батча
- LBFGS, где L1 и L2 = 0.02, LR = 0.05, 32 батча
- LBFGS, где L1 и L2 = 0.001, LR = 0.05, 16 батча
- LBFGS, где L1 и L2 = 0.02, LR = 0.05 + early stopped, 16 батчей
- Adam, LR-0.01 (на графике в заголовке опечатка), 16 батчей

Очевидно, что самой слабой оказалась стандартная модель, а остальные показывает схожий результат. На графике я решил показать как результат отображён относительно линейной функции. 

Learning Rate сильно ухудшал метрики, если его значение выше 0.08, самым оптимальным значением было 0.05, в некоторых случаях 0.01-0.02. Количество батчей 16-32, иначе модель также начинала плохо предсказывать. 

~~~
Стандартная с SGD и LR=0.5, 32 батча
Test loss: 0.9682

LBFGS, где L1 и L2 = 0.02, LR = 0.05, 32 батча
Test loss: 0.5343

LBFGS, где L1 и L2 = 0.001, LR = 0.05, 16 батча
Test loss: 0.3585 (могло колебаться и Test loss достигал около 0.2700)

LBFGS, где L1 и L2 = 0.02, LR = 0.05 + early stopped, 16 батчей
Test loss: 0.2090

Adam, LR-0.01 (на графике в заголовке опечатка), 16 батчей
Test loss: 0.2599
~~~

В итоге модель с параметрами LBFGS, где L1 и L2 = 0.02, LR = 0.05 + early stopped, 16 батчей оказалась лучшей.

### 3.2 Feature Engineering
Я пробовал создать новые признаки или изменить существующие, например, я реализовывал возведение в степень и логарифмирование в Dataset, но это не дало никакого результата, потому я удалил из класса.

~~~
def log_by_id(self, index: int, column: str):
    mask = self.data['id'] == index
    self.data.loc[mask, column] = np.log(self.data.loc[mask, column])

def square_by_id(self, index: int, column: str):
    mask = self.data['id'] == index
    self.data.loc[mask, column] = self.data.loc[mask, column] ** 2
~~~

Также пробовал создать новый признак, но датасеты у меня небольшие и я просто перебрал несколько вариантов, это тоже не дало никакого результата. 