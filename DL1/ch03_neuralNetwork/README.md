# 🧠 CHAPTER 3. 신경망

### 📌 개요

- 퍼셉트론으로도 복잡한 함수를 표현할 수 있지만, **가중치를 사람이 수동으로 설정**해야 하는 한계가 있음
- 신경망은 **데이터로부터 가중치를 학습**할 수 있는 능력을 가짐 → 이게 가장 큰 장점!

---

### 🔁 신경망 vs 퍼셉트론

✅ 공통점:

> - **각 층의 뉴런들이 다음 층으로 신호를 전달**함

❗ 차이점:

> - **퍼셉트론**: 갑작스러운 변화(계단 함수)
> - **신경망**: 매끄러운 변화(시그모이드 함수 등)
>   - 퍼셉트론은 "0 or 1"로 딱 잘라지는 신호
>   - 신경망은 확률처럼 **연속적으로 변화**하는 신호

---

### 🧮 활성화 함수 (Activation Function)

| 함수 이름       | 그래프 특성                     | 특징                   |
| --------------- | ------------------------------- | ---------------------- |
| 계단 함수       | 불연속                          | 퍼셉트론에서 사용      |
| 시그모이드 함수 | 연속적, 부드러운 곡선           | 신경망에서 자주 사용   |
| ReLU 함수       | 0 이하일 땐 0, 양수일 땐 그대로 | 깊은 신경망에서 효과적 |

---

### 📐 활성화 함수 수식 정리 & 시각화

#### 🌀 시그모이드 함수 (Sigmoid)

- **정의**:

  ![sigmoid](https://latex.codecogs.com/png.image?\dpi{120}y=\frac{1}{1+e^{-x}})

- **특징**:

  - 출력이 항상 (0, 1) 사이의 값을 가짐 → **확률적 해석**에 적합
  - 하지만 큰 입력에 대해 **기울기 소실(gradient vanishing)** 문제 발생 가능

- **시각화**:  
  ![sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

#### 🔷 ReLU 함수 (Rectified Linear Unit)

- **정의**:

  ![relu](https://latex.codecogs.com/png.image?\dpi{120}y=\max(0,x))

- **특징**:

  - 0 이하일 땐 0, 양수일 땐 그대로
  - 계산 간단 + **기울기 소실 문제 덜함**
  - 하지만 0 이하의 입력이 계속되면 뉴런이 죽는 문제("dying ReLU")가 있음

- **시각화**:  
  ![relu](https://upload.wikimedia.org/wikipedia/commons/6/6c/Rectifier_and_softplus_functions.svg)

#### 🎯 소프트맥스 함수 (Softmax)

- **정의 (출력층에 사용)**:

  ![softmax](https://latex.codecogs.com/png.image?\dpi{120}y_k=\frac{e^{a_k}}{\sum_{i=1}^{n}e^{a_i}})

- **특징**:

  - 출력값의 총합이 1 → **확률 분포로 해석 가능**
  - 주로 **분류 문제**의 출력층에 사용
  - 값이 큰 입력일수록 확률이 더 크게 나옴

---

#### 📤 출력층의 활성화 함수

- **회귀 문제** (숫자 예측 등):  
  → **항등 함수 (identity function)** 사용

- **분류 문제** (클래스 분류 등):  
  → **소프트맥스 함수 (softmax)** 사용  
  → 출력층의 뉴런 수 == 클래스 수

---

### 📦 배치 처리 (Batch Processing)

- **입력 데이터를 묶은 단위**를 "배치"라고 함
- 한 번에 여러 데이터를 처리해서 **연산 속도를 획기적으로 향상**시킴
- 넘파이의 다차원 배열을 사용하면 **효율적인 신경망 구현 가능**

---

### ✅ 정리

- 신경망에서는 활성화 함수로 시그모이드 함수와 ReLU 함수 같은 매끄럽게 변화하는 함수를 이용한다.
- 넘파이의 다차원 배열을 잘 사용하면 신경망을 효율적으로 구현할 수 있다.
- 기계학습 문제는 크게 회귀와 분류로 나눌 수 있다.
- 출력층의 활정화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 이용한다.
- 분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다.
- 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 이 배치 단위로 진행하면 결과를 훨씬 빠르게 얻을 수 있다.
