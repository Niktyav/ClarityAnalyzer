# <img src='./static/img/mipt-icon.png' width="70" height="30">
# Задача от Beeline

**Заказчик:** Компания Beeline является одним из ведущих поставщиков телекоммуникационных услуг в России.

**Описание задачи:** Модель для бинарной классификации аудиофрагментов с целью определения качества их транскрибации. Модель должна определять, является ли транскрибация качественной, при условии, что псевдоразметка отличается от ручной разметки не более, чем на N пунктов коофициента ошибок в словах. Итоговая оценка качества модели будет проводиться с помощью метрики ROC-AUC.

## Решение
Было проведено масштабное тестирование комбинаций из четырех методов векторизации и 32 моделей классификации.
Проведенные исследования собраны в директории `research`. 

Наилучший результат по метрике **ROC-AUC** показал **fine-tuning модели sbert_large_mt_nlu_ru для задачи классификации**. 
Дообучались 8 последних слоев энкодера и линейный слой (классификатор). 
ROC-AUC на тестовой выборке: **0.804**.

Пайплайн обучения модели находится в файле [Bert fine-tuning](https://github.com/kosatchev/ClarityAnalyzer/blob/28e105f7b15d04c355aceef740cb16a0f72db731/Sbert%20fine-tuning.ipynb).
Предобученная модель размещена на [HuggingFace](https://huggingface.co/nazarovmichail/sbert_large_transcription_classification). Для удобства тестирования мы подготовили ноутбук [Model inference](https://github.com/kosatchev/ClarityAnalyzer/blob/5140565336c5d6bf58fe1c30fc5d24f681b5994e/Model%20inference.ipynb).

**Выводы:**
* Выбор подходящего метода векторизации имеет ключевое значение.
* Для повышения качества классификации модель стоит обучать на большем количестве данных, чем были представлены в тренировочном датасете.
* Данные в тренировочном датасете однородные (предобработанные), поэтому для них не потребовалась ручная нормализация. Кроме того, модель sbert самостоятельно обрабатывает заглавные буквы и взаимозаменяемость "ё" и "е".  А вот знаки препинания и формат записи чисел (цифрами или текстом) на результат токенизации влияют, поэтому при оценке качества датасетов другого формата стоит предварительно привести текст к единообразному виду.


## 👥 Состав команды

- **Вяткин Роман Вячеславович**

  Капитан команды/ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью navec_hudlit_v1_12B_500K_300d_100q(проект Natasha) с моделями:
    - LogisticRegression
    - Полносвязная двухслойная нейронная сеть

- **Яськова Марина Андреевна**

  ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью модели ai-forever/sbert_large_mt_nlu_ru с моделями:
    - PassiveAggressiveClassifier
    - VotingClassifier
    - MultinomialNB
  - исследование влияния предварительной рукописной нормализации текстовых данных на результаты классификации

- **Баймлер Ярослав Игоревич**

  ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью модели ai-forever/sbert_large_mt_nlu_ru с моделями:
    - LinearSVC
    - DecisionTreeClassifier
    - RandomForestClassifier
    - GradientBoostingClassifier

- **Ихматуллаев Даврон Махаматкаримович**

  ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью модели intfloat/multilingual-e5-large и классификация моделью LogisticRegression
  - токенизация и эмбеддинг с помощью модели ai-forever/sbert_large_mt_nlu_ru и классификация моделью CatBoost
  - аугментация данных при помощи модели Whisper, тестирование качества выборочных моделей после аугментации 


- **Назаров Михаил Сергеевич**

  ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью модели ai-forever/sbert_large_mt_nlu_ru с моделями:
    - LogisticRegression
    - SGDClassifier
    - SVC
    - NuSVC
    - MLPClassifier
    - ExtraTreesClassifier
    - HistGradientBoostingClassifier
    - LightGBM
    - XGBoost
    - KNeighborsClassifier
    - RadiusNeighborsClassifier
    - GaussianNB
    - BernoulliNB
    - LinearDiscriminantAnalysis
    - QuadraticDiscriminantAnalysis

- **Новиков Валентин Владимирович**

  ML engineer, тестирование пайплайнов:
  - векторизация мешком слов с моделями семейства Naive Bayes, SVM, XGBoost, LogisticRegression;
  - TF-IDF с униграммами и биграммами с моделями семейства Naive Bayes, SVM, XGBoost, LogisticRegression.

- **Косачев Дмитрий**

  ML engineer, тестирование пайплайнов:
  - токенизация и эмбеддинг с помощью модели ai-forever/sbert_large_mt_nlu_ru с моделью AdaBoostClassifier, StackingClassifier, BaggingClassifier


