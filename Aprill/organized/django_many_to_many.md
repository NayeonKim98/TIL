
# Django - Many to Many Relationships

## M:N 관계 정의
- **M:N (Many to Many)** 관계는 한 테이블의 데이터가 다른 테이블의 0개 이상의 데이터와 관련될 수 있는 구조입니다. 즉, 한 레코드가 여러 레코드와 연결되거나, 여러 레코드가 하나의 레코드와 연결될 수 있습니다. 이때, 관계가 필수적이지 않아서, 0개일 수도 있습니다.

---

## N:1 모델의 한계
- N:1 관계에서 한 쪽에만 외래키를 두고 다른 테이블을 참조하여, 일대다 관계를 설정합니다. 예를 들어, 병원의 진료시스템을 모델링할 때, 환자와 의사의 관계를 설정할 수 있습니다.

#### 예시) 의사와 환자
- **환자**는 여러 **의사**에게 예약을 할 수 있는 구조입니다.
- 초기 설정에서 `환자` 모델에 `doctor` 외래키를 사용하여, 환자가 하나의 의사에게만 예약을 하도록 만들 수 있습니다.

```python
# 초기 모델 설정
class Patient(models.Model):
    name = models.CharField(max_length=100)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)  # 1:N 관계
```

하지만, 만약 하나의 **환자**가 **여러 의사**에게 예약을 하고 싶다면, `ForeignKey`만으로는 부족하며, 다대다(M:N) 관계가 필요합니다.

---

## 해결책: 중개 모델
- **중개 모델(through model)**을 사용하여, 예약 테이블을 따로 만들면, 환자가 여러 의사에게 예약을 할 수 있습니다.
- 중개 모델은 별도의 테이블을 생성하여 다대다 관계를 명시적으로 설정하는 방법입니다.

#### 중개 모델 예시:
```python
class Reservation(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    appointment_time = models.DateTimeField()
```

이렇게 예약 테이블을 만들면, 하나의 환자가 여러 의사에게 예약을 할 수 있고, 의사 입장에서 자신의 예약된 환자들을 조회할 수 있게 됩니다.

---

## Django의 `ManyToManyField`
- Django는 `ManyToManyField` 필드를 사용하여 중개 모델을 자동으로 생성할 수 있도록 도와줍니다. 이 필드를 사용하면 중개 테이블을 수동으로 작성할 필요가 없어집니다.

#### 예시: `ManyToManyField` 사용
```python
class Patient(models.Model):
    name = models.CharField(max_length=100)
    doctors = models.ManyToManyField(Doctor)
```
이렇게 `ManyToManyField`를 사용하면, `Patient` 모델에 `doctor`에 대한 참조가 자동으로 생성됩니다. 실제로 데이터베이스에는 `hospitals_patient`라는 중개 테이블이 생깁니다. 이 테이블에는 `patient_id`와 `doctor_id`가 저장됩니다.

#### `ManyToManyField` 사용 예:
```python
patient1 = Patient.objects.create(name="John Doe")
doctor1 = Doctor.objects.create(name="Dr. Smith")

# 예약 추가
patient1.doctors.add(doctor1)

# 의사에게 예약된 환자들 확인
doctor1.patients.all()
```

---

## `add()`와 `remove()` 메서드
- Django에서 `ManyToManyField` 관계를 추가하거나 제거할 때 `add()`와 `remove()` 메서드를 사용합니다. 이 메서드는 중개 테이블을 자동으로 관리합니다.

#### 예시: 추가와 제거
```python
# 예약 추가
patient1.doctors.add(doctor1)

# 예약 제거
patient1.doctors.remove(doctor1)
```

---

## `through` 인자를 사용하여 중개 테이블에 추가 데이터 저장
- 때때로 단순한 참조뿐만 아니라 중개 테이블에 추가적인 데이터를 저장해야 할 경우가 있습니다. 예를 들어, 예약의 증상이나 예약 시간과 같은 정보를 중개 테이블에 추가하고 싶다면, `ManyToManyField`에 `through` 인자를 사용하여 명시적인 중개 모델을 설정할 수 있습니다.

#### 예시: 중개 모델에 추가 데이터
```python
class Reservation(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    symptom = models.CharField(max_length=100)

class Patient(models.Model):
    name = models.CharField(max_length=100)
    doctors = models.ManyToManyField(Doctor, through='Reservation')
```

```python
# 중개 테이블에 증상 추가
patient2 = Patient.objects.create(name="Jane Doe")
doctor1 = Doctor.objects.create(name="Dr. Smith")

patient2.doctors.add(doctor1, through_defaults={'symptom': 'flu'})
```

이 예제에서 `through_defaults`를 사용하여 중개 테이블에 `symptom` 정보를 추가하고 있습니다.

---

## 결론
- **N:1 관계**는 한 테이블에서 다른 테이블을 참조하는 구조이며, 이는 외래키로 구현됩니다.
- **M:N 관계**는 중개 모델 또는 Django의 `ManyToManyField`를 사용하여 관계를 설정할 수 있습니다.
- Django의 `ManyToManyField`는 중개 테이블을 자동으로 생성하고, 이를 통해 관계를 쉽게 관리할 수 있습니다.
- 추가적인 데이터를 중개 테이블에 저장하려면 `through` 인자를 활용하여 별도의 중개 모델을 정의해야 합니다.


# Django - Many to Many Relationships with "Likes" Example

## ManytoMany에서 단수형 아니고 복수형으로 쓰는 이유
- 단수형은 보통 **1:N 관계**에 사용됩니다.
- 복수형으로 사용하는 이유는, 이름만 보고도 이것이 **다대다**(M:N) 관계인지 **1:N** 관계인지 구분할 수 있도록 하기 위함입니다.

---

## Symmetrical Argument
- **Person**이 자기 자신과 **다대다** 관계를 맺는 예시를 다룹니다. 중개 테이블을 사용한 예시입니다.

### 중개 테이블 예시
```
id  | to_person_id | from_person_id
----|--------------|---------------
 1  |      1       |        2
 2  |      2       |        3
 3  |      2       |        1
```

- 위와 같은 테이블에서 `user1`과 `user2`가 서로 방향이 있는 '재귀적' 관계를 맺고 있다는 것을 의미합니다. 예를 들어, `1 -> 2` 방향으로 친구 관계를 맺었다면, `2 -> 1`은 다른 관계입니다.
- 만약 `symmetrical=True`로 설정하면 관계 방향성이 '대칭'이 됩니다.

### 예시:
```
id  | to_person_id | from_person_id
----|--------------|---------------
 1  |      1       |        2
 2  |      2       |        1
```
- 위와 같이, 관계가 대칭적이므로 `1 -> 2`와 `2 -> 1`은 동일한 관계가 됩니다.
- 예를 들어, **인스타그램**에서 우리가 **셀럽**을 팔로우한다고 해서 그들이 우리를 팔로우하는 것은 아니므로, **대칭 관계**가 적용될 수 있습니다.

---

## 좋아요 기능 구현

- **좋아요 기능**은 `Article`과 `User`의 관계입니다. **좋아요**가 없는 게시글도 있을 수 있으며, 회원은 좋아요를 누르지 않아도 됩니다. 따라서 `Article(M)`과 `User(N)`의 관계가 됩니다.

### 모델 수정
- 게시글에 좋아요를 누른 모든 유저를 조회하려면 `ManyToManyField`를 사용하여 **복수형**으로 관계를 설정합니다.

```python
# models.py
class Article(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    like_users = models.ManyToManyField(settings.AUTH_USER_MODEL)
```
이때, 좋아요를 누른 유저들을 명시적으로 확인하기 위해 **매니저 이름을 설정**하는 것이 좋습니다. 예를 들어, `like_users`보다는 `like_users.all()`을 사용하는 것이 명확합니다.

### 좋아요 추가 / 취소
- 좋아요를 요청받는 URL을 설정하고, 해당 URL을 통해 좋아요 기능을 구현합니다.

```python
# articles/urls.py
urlpatterns = [
    ...
    path('<int:article_pk>/likes/', views.likes, name='likes')
]
```

```python
# articles/views.py
def likes(request, article_pk):
    # 좋아요를 누를 게시글 조회
    article = Article.objects.get(pk=article_pk)

    # 좋아요 추가 / 좋아요 취소
    if request.user in article.like_users.all():
        article.like_users.remove(request.user)
        request.user.like_articles.remove(article)  # 이렇게 써도 된다.
    else:
        article.like_users.add(request.user)
        request.user.like_articles.add(article)  # 이렇게 써도 된다.
```

---

## 좋아요 버튼 구현

- HTML에서 좋아요 버튼을 눌렀을 때 **좋아요 추가** 또는 **취소** 기능을 처리합니다.

```html
<!-- articles/index.html -->
<form action="{% url 'articles:likes' article.pk %}" method="POST">
    {% csrf_token %}
    {% if request.user in article.like_users.all() %}
        <input type="submit" value="좋아요 취소">
    {% else %}
        <input type="submit" value="좋아요">
    {% endif %}
</form>
```

- 위 코드에서 `request.user`가 `article.like_users.all()`에 포함되면 "좋아요 취소"를 표시하고, 그렇지 않으면 "좋아요"를 표시합니다.

---

## 결론
- **ManytoMany 관계**에서는 관계의 방향성, 대칭 여부 등을 고려하여 설정할 수 있습니다. `symmetrical=True`로 설정하면 관계가 대칭이 되어 서로 다른 방향의 관계가 동일하게 취급됩니다.
- **좋아요 기능**은 `Article`과 `User` 모델 간의 다대다 관계를 설정하고, 이를 통해 사용자가 게시글에 좋아요를 추가하거나 취소하는 기능을 구현할 수 있습니다.
