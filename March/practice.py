arr = [1, 2, 1, 3, 2, 4, 3, 5, 3, 6, 4, 7, 5, 8, 5, 9, 6, 10, 6, 11, 7, 12, 11, 13]


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, child):
        if(not self.left):
            self.left = child
            return
        if(not self.right):
            self.right = child
            return
        return

    def preorder(self):
        if self != None:
            print(self.value, end=' ')
            if self.left:
                self.left.preorder()
            if self.right:
                self.right.preorder()

    # 중위 순회
    def inorder(self):
        if self != None:
            if self.left:
                self.left.inorder()
            print(self.value, end=' ')
            if self.right:
                self.right.inorder()

    # 후위 순회
    def postorder(self):
        if self != None:
            if self.left:
                self.left.postorder()
            if self.right:
                self.right.postorder()
            print(self.value, end=' ')

# 이진 트리 만들기
nodes = [TreeNode(i) for i in range(0, 14)]
for i in range(0, len(arr), 2):
    parentNode = arr[i]
    childNode = arr[i + 1]
    nodes[parentNode].insert(nodes[childNode])

nodes[1].preorder()
print()
nodes[1].inorder()
print()
nodes[1].postorder()

#------------------------------------

arr = [1, 2, 1, 3, 2, 4, 3, 5, 3, 6, 4, 7, 5, 8, 5, 9, 6, 10, 6, 11, 7, 12, 11, 13]

# 이진 트리 생성
nodes = [[] for _ in range(14)]
for i in range(0, len(arr), 2):
    parentNode = arr[i]
    childNode = arr[i + 1]
    nodes[parentNode].append(childNode)

# 자식이 없다는 걸 표현하기 위해 None 을 삽입
for li in nodes:
    for _ in range(len(li), 2):
        li.append(None)


# 전위 순회
def preorder(nodeNum):
    if nodeNum == None:
        return
    print(nodeNum, end = ' ')
    preorder(nodes[nodeNum][0])
    preorder(nodes[nodeNum][1])

preorder(1)

#---------------------------------------------

#  일차원 배열로 효율적으로 하는 방법

def check(row):
    for col in range(row):
        if visited[row] == visited[col]:
            return False

        # 열과 행의 차이가 같다 == 현재 col 의 좌우 대각선이다
        if abs(visited[row] - visited[col]) == abs(row - col):
            return False

    return True


def dfs(row):
    global cnt

    if row == N:
        cnt += 1
        return

    for col in range(N):
        visited[row] = col
        if not check(row):
            continue

        dfs(row + 1)

N = 8
visited = [0] * N
cnt = 0

dfs(0)
print(cnt)

#-------------------------------------------

# 4*4 N-Queen 문제
# - (y,x) 좌표에 queen 을 놓은 적이 있다.
#  - visited 기록 방법
#    - 1. 이차원 배열
#    - 2. 일차원 배열로 효율적으로 하는 방법

# level: N개의 행에 모두 놓았다.
# branch: N개의 열
def check(row, col):
    # 1. 같은 열에 놓은 적이 있는가
    for i in range(row):
        if visited[i][col]:
            return False

    # 2. 왼쪽 대각선 (\)
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if visited[i][j]:
            return False

        i -= 1
        j -= 1

    # [참고]
    # for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
    #     if visited[i][j]:
    #         return False

    # 3. 오른쪽 대각선 (/)
    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if visited[i][j]:
            return False

        i -= 1
        j += 1

    return True


def dfs(row):
    global answer
    if row == N:    # 모두 놓으면 성공한 케이스
        answer += 1
        return

    # 후보군: N개의 열
    for col in range(N):
        # 가지치기: 유망하지 않은 케이스는 확인하지 않겠다!
        if check(row, col) is False:
            continue

        visited[row][col] = 1
        dfs(row + 1)
        visited[row][col] = 0


N = 4
visited = [[0] * N for _ in range(N)]
answer = 0  # 가능한 정답 수

dfs(0)
print(answer)

#------------------------------------------------

# {1,2,3,4,5,6,7,8,9,10}의 powerset 중 원소의 합이 10인 부분집합을 모두 출력하시오.
arr = [i for i in range(1, 11)]
# visited = []  -> 이번 문제에서는 사용할 필요가 없다.

# level: N개의 원소를 모두 고려하면
# branch: 집합에 해당 원소를 포함 시키는 경우 or 안 시키는 경우 두 가지
# 누적값
#  - 부분집합의 총합
#  - 부분집합에 포함된 원소들
def dfs(cnt, total, subset):
    # 1. total 이 10이면 출력해라
    if total == 10:
        print(subset)
        return

    # 2. total 이 10을 넘으면 가지치기하자
    if total > 10:
        return

    if cnt == 10:
        # 1. total 이 10이면 출력해라  -> 여기는 하면 안된다.
        return

    dfs(cnt + 1, total + arr[cnt], subset + [arr[cnt]])  # 포함 하는 경우
    dfs(cnt + 1, total, subset)  # 집합에 포함 안 하는 경우

dfs(0, 0, [])

#------------------------------------------------

import heapq

arr = [20, 15, 19, 4, 13, 11]

# 1. 기본 리스트를 heap 으로 만들기
# heapq.heapify(arr)  # 최소힙으로 바뀐다.
# 디버깅 시에 이진 트리로 그림을 그려야 한다!
# -> 딱 봤을때는 정렬이 안된 것 처럼 보인다.
# print(arr)

# 2. 하나 씩 데이터를 추가
min_heap = []
for num in arr:
    heapq.heappush(min_heap, num)
print(min_heap)

# 최대힙?
max_heap = []
for num in arr:
    heapq.heappush(max_heap, -num)

while max_heap:
    pop_num = heapq.heappop(max_heap)
    print(-pop_num, end=' ')

# ------------------ 전자사전 예제
# 1. 길이 순서로 먼저 출력
# 2. 길이가 같다면, 사전 순으로 출력

arr = ['apple', 'banana', 'kiwi', 'abcd', 'abca', 'lemon', 'peach', 'grape', 'pear']
# sort 를 쓰면 아래와 같다.
# 즉, 우선순위가 2가지
# arr.sort(key=lambda x: (len(x), x))
dictionary = []

# 단어를 삽입 (길이, 단어) 형태로 삽입
for word in arr:
    heapq.heappush(dictionary, (len(word), word))

# 전자사전에서 단어를 하나씩 꺼내기
print("전자사전 순서:")
while dictionary:
    length, word = heapq.heappop(dictionary)
    print(f"{word} (길이: {length})")

#----------------------------------------------------

# 1. 분할: 리스트의 길이가 1일 때까지 분할
# 2. 정복: 리스트의 길이가 1이 되면 자동으로 정렬됨
# 3. 병합
#   - 왼쪽, 오른쪽 리스트 중
#       작은 원소부터 정답 리스트에 추가하면서 진행
def merge(left, right):
    # 두 리스트를 병합한 결과 리스트
    result = [0] * (len(left) + len(right))
    l = r = 0

    # 두 리스트에서 비교할 대상이 남아있을 때 까지 반복
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result[l + r] = left[l]
            l += 1
        else:
            result[l + r] = right[r]
            r += 1

    # 왼쪽 리스트에 남은 데이터들을 모두 result 에 추가
    while l < len(left):
        result[l + r] = left[l]
        l += 1

    # 오른쪽 리스트에 남은 데이터들을 모두 result 에 추가
    while r < len(right):
        result[l + r] = right[r]
        r += 1

    return result


def merge_sort(li):
    if len(li) == 1:
        return li

    # 1. 절반 씩 분할
    mid = len(li) // 2
    left = li[:mid]    # 리스트의 앞쪽 절반
    right = li[mid:]   # 리스트의 뒤쪽 절반

    left_list = merge_sort(left)
    right_list = merge_sort(right)

    # print(left_list, right_list)
    # 분할이 완료되면
    # 2. 병합
    merged_list = merge(left_list, right_list)
    return merged_list


arr = [69, 10, 30, 2, 16, 8, 31, 22]
sorted_arr = merge_sort(arr)
print(sorted_arr)

#----------------------------------------------

arr = [3, 2, 4, 6, 9, 1, 8, 7, 5]
# arr = [11, 45, 23, 81, 28, 34]
# arr = [11, 45, 22, 81, 23, 34, 99, 22, 17, 8]
# arr = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


# 피벗: 제일 왼쪽 요소
# 이미 정렬된 배열이나 역순으로 정렬된 배열에서 최악의 성능을 보일 수 있음
def hoare_partition1(left, right):
    pivot = arr[left]  # 피벗을 제일 왼쪽 요소로 설정
    i = left + 1
    j = right

    while i <= j:
        while i <= j and arr[i] <= pivot:
            i += 1

        while i <= j and arr[j] >= pivot:
            j -= 1

        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    arr[left], arr[j] = arr[j], arr[left]
    return j


# 피벗: 제일 오른쪽 요소
# 이미 정렬된 배열이나 역순으로 정렬된 배열에서 최악의 성능을 보일 수 있음
def hoare_partition2(left, right):
    pivot = arr[right]  # 피벗을 제일 오른쪽 요소로 설정
    i = left
    j = right - 1

    while i <= j:
        while i <= j and arr[i] <= pivot:
            i += 1
        while i <= j and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    arr[i], arr[right] = arr[right], arr[i]
    return i


# 피벗: 중간 요소로 설정
# 일반적으로 더 균형 잡힌 분할이 가능하며, 퀵 정렬의 성능을 최적화할 수 있습니다.
def hoare_partition3(left, right):
    mid = (left + right) // 2
    pivot = arr[mid]  # 피벗을 중간 요소로 설정
    arr[left], arr[mid] = arr[mid], arr[left]  # 중간 요소를 왼쪽으로 이동 (필요 시)
    i = left + 1
    j = right

    while i <= j:
        while i <= j and arr[i] <= pivot:
            i += 1
        while i <= j and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]

    arr[left], arr[j] = arr[j], arr[left]
    return j


def quick_sort(left, right):
    if left < right:
        pivot = hoare_partition1(left, right)
        # pivot = hoare_partition2(left, right)
        # pivot = hoare_partition3(left, right)
        quick_sort(left, pivot - 1)
        quick_sort(pivot + 1, right)


quick_sort(0, len(arr) - 1)
print(arr)

#--------------------------------------------------

arr = [3, 2, 4, 6, 9, 1, 8, 7, 5]


def lomuto_partition(left, right):
    pivot = arr[right]

    i = left - 1
    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


def quick_sort(left, right):
    if left < right:
        pivot = lomuto_partition(left, right)
        quick_sort(left, pivot - 1)
        quick_sort(pivot + 1, right)


quick_sort(0, len(arr) - 1)
print(arr)

#----------------------------------------------------

def binary_search_while(target):
    left = 0
    right = len(arr) - 1
    cnt = 0

    while left <= right:
        mid = (left + right) // 2
        cnt += 1        # 검색횟수 추가

        if arr[mid] == target:
            return mid, cnt      # mid index 에서 검색 완료!

        # 왼쪽에 정답이 있다.
        if target < arr[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1, cnt


def binary_search_recur(left, right, target):
    # left, right 를 작업 영역으로 검색
    # left <= right 만족하면 반복
    if left > right:
        return -1

    mid = (left + right) // 2
    # 검색하면 종료
    if target == arr[mid]:
        return mid

    # 한 번 할 때마다 left 와 right 를 mid 기준으로 이동시켜 주면서 진행
    # 왼쪽을 봐야한다
    if target < arr[mid]:
        return binary_search_recur(left, mid - 1, target)
    # 오른쪽을 봐야한다.
    else:
        return binary_search_recur(mid + 1, right, target)


arr = [4, 2, 9, 7, 11, 23, 19]

# 이진 검색은 항상 정렬된 데이터에 적용해야 한다!!!
arr.sort()  # [2, 4, 7, 9, 11, 19, 23]

print(f'9 - {binary_search_recur(0, len(arr) - 1, 9)}')
print(f'2 - {binary_search_recur(0, len(arr) - 1, 2)}')
print(f'20 - {binary_search_recur(0, len(arr) - 1, 20)}')

#------------------------------------------------

# 정사각형 방 - DFS
# -> 원래는 시간 초과가 나야 하는데 학습용으로 공유 드립니다!!!

dy = [-1, 0, 1, 0]
dx = [0, -1, 0, 1]


def DFS(sy, sx):
    global matrix, cnt
    for i in range(4):
        ny, nx = sy + dy[i], sx + dx[i]
        if 0 <= ny < N and 0 <= nx < N:
            if matrix[ny][nx] == matrix[sy][sx] + 1:
                cnt += 1
                DFS(ny, nx)


T = int(input())
for tc in range(1, T + 1):
    N = int(input())
    matrix = [list(map(int, input().split())) for _ in range(N)]
    max_cnt, resulty, resultx = 0, 0, 0
    for y in range(N):
        for x in range(N):
            cnt = 1
            DFS(y, x)
            if max_cnt < cnt:
                max_cnt = cnt
                resulty = y
                resultx = x
            elif max_cnt == cnt and matrix[y][x] < matrix[resulty][resultx]:
                resulty = y
                resultx = x

    print(f'#{tc} {matrix[resulty][resultx]} {max_cnt}')

#------------------------------------------------------

import sys
sys.stdin = open("input.txt", "r")

# 정사각형 방 - 정답 코드
# 접근법
# - N*N visited 배열을 만든다
# - 해당 숫자에서 갈 수 있다면 1을 기록한다
# - 연속된 1의 길이가 가장 긴 곳이 정답이다.
#  - 같은 길이가 있다면, 작은 수가 정답 위치

dy = [-1, 1, 0, 0]
dx = [0, 0, -1, 1]

T = int(input())
for tc in range(1, T + 1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    visited = [0] * (N * N + 1)

    # 현재 위치 숫자 기준 상하좌우 확인
    #   -> 1 큰 곳이 있다면 visited 기록
    for y in range(N):
        for x in range(N):
            for i in range(4):  # 상하좌우 확인
                new_y = y + dy[i]
                new_x = x + dx[i]

                # 델타는 범위 밖을 잘 체크해주어야 한다!!!
                if new_y < 0 or new_y >= N or new_x < 0 or new_x >= N:
                    continue

                if arr[new_y][new_x] == arr[y][x] + 1:
                    # 현재 숫자는 다음으로 이동이 가능하다
                    visited[arr[y][x]] = 1
                    break   # 나머지 방향은 볼 필요 없다.

    # print(visited)
    # 연속된 1의 개수가 가장 긴 곳을 찾는다.
    # 가장 긴 길이, 현재 몇 개인지, 출발지
    max_cnt = cnt = start = 0
    for i in range(1, N * N + 1):
        if visited[i] == 1:
            cnt += 1
        else:
            if max_cnt < cnt:
                max_cnt = cnt
                start = i - cnt
            cnt = 0     # 개수 초기화

    print(f'#{tc} {start} {max_cnt + 1}')

#----------------------------------------------------

import sys
sys.stdin = open("input.txt", "r")

# level: 점원 수
# branch: 탑에 포함 시킨다 or 안시킨다
def recur(cnt, total_height):
    global answer
    # 기저조건 가지치기
    # 이미 B 이상인 탑이면, 점원을 더 쌓을 필요가 없다.
    # => 탑이 더 높은 정답은 필요 없다.
    if total_height >= B:
        answer = min(answer, total_height)
        return

    if cnt == N:
        return

    recur(cnt + 1, total_height + heights[cnt])  # 탑에 포함 시키는 경우
    recur(cnt + 1, total_height)  # 탑에 포함 안 시키는 경우


T = int(input())
for tc in range(1, T + 1):
    N, B = map(int, input().split())
    heights = list(map(int, input().split()))
    answer = int(21e8)   # 21억
    recur(0, 0)
    print(f'#{tc} {answer - B}')

    #---------------------------------------------

    import sys
sys.stdin = open("input.txt", "r")

# 접근법
# - 시작 지점: 전체 다 보아야 한다.
# - 재귀 돌리면서(상하좌우 이동) 숫자를 붙인다
# - 숫자가 7자리가 되면, set에 넣는다. (중복 제거)
dy = [-1, 1, 0, 0]
dx = [0, 0, -1, 1]

def recur(y, x, number):
    if len(number) == 7:    # 7자리가 되면 종료
        result.add(number)
        return

    for i in range(4):  # 상하좌우 확인
        new_y = y + dy[i]
        new_x = x + dx[i]

        # 범위 밖이면 continue
        if new_y < 0 or new_y >= 4 or new_x < 0 or new_x >= 4:
            continue

        # 다음 위치를 추가하면서 진행
        recur(new_y, new_x, number + matrix[new_y][new_x])


T = int(input())
for tc in range(1, T + 1):
    matrix = [input().split() for _ in range(4)]
    result = set()

    for y in range(4):
        for x in range(4):
            recur(y, x, matrix[y][x])

    print(f'#{tc} {len(result)}')

#-------------------------------------------------------

import sys
sys.stdin = open("input.txt", "r")

# 완전 탐색을 하는 버전
# - 각 달에 4가지 케이스를 모두 확인하면서 진행
def recur(month, total_cost):
    global min_answer
    # 가지치기
    if min_answer < total_cost:
        return

    if month > 12:
        min_answer = min(min_answer, total_cost)
        return

    # 1일 이용권으로 다 사는 경우
    recur(month + 1, total_cost + (days[month] * cost_day))
    # 1달 이용권 사는 경우
    recur(month + 1, total_cost + cost_month)
    # 3달 이용권 사는 경우
    recur(month + 3, total_cost + cost_month3)
    # 1년 이용권 사는 경우
    recur(month + 12, total_cost + cost_year)


T = int(input())
for tc in range(1, T + 1):
    # 이용권 가격들 (1일, 1달, 3달, 1년)
    cost_day, cost_month, cost_month3, cost_year = map(int, input().split())
    # 12개월 이용 계획
    days = [0] + list(map(int, input().split()))
    min_answer = int(21e8)
    recur(1, 0)  # 1월부터 시작
    print(f'#{tc} {min_answer}')

#----------------------------------------------------

import sys
sys.stdin = open("input.txt", "r")

T = int(input())
for tc in range(1, T + 1):
    # 이용권 가격들 (1일, 1달, 3달, 1년)
    cost_day, cost_month, cost_month3, cost_year = map(int, input().split())
    # 12개월 이용 계획
    days = [0] + list(map(int, input().split()))

    dp = [0] * 13
    # 시작점 초기화 (1월, 2월)
    # 1월의 가격 (1일권 구매 vs 1달권 구매)
    dp[1] = min(days[1] * cost_day, cost_month)
    # 2월의 가격 = 1월의 가격 + (1일권 구매 vs 1달권 구매)
    dp[2] = dp[1] + min(days[2] * cost_day, cost_month)

    # 3월~12월은 반복하면서 계산
    for month in range(3, 13):
        # N월의 최소 비용 후보
        # 1. (N-3)월에 3개월 이용권을 구입한 경우
        # 2. (N-1)월의 최소 비용 + 1일권 구매
        # 3. (N-1)월의 최소 비용 + 1달권 구매
        dp[month] = min(dp[month-3] + cost_month3
                        ,dp[month-1] + (days[month] * cost_day)
                        ,dp[month-1] + cost_month)

    # 12월의 누적 최소 금액 vs 1년권
    answer = min(dp[12], cost_year)
    print(f'#{tc} {answer}')

#------------------------------------------------------

# 3명의 친구 부분집합 찾기
arr = ['O', 'X']
path = []
name = ['MIN', 'CO', 'TIM']


# path 출력 함수
def print_name():
    print(path, end=' / ')
    print('{ ', end='')
    for i in range(3):
        if path[i] == 'O':
            print(name[i], end=' ')
    print('}')


def run(lev):
    # 3개를 뽑았을 때 출력
    if lev == 3:
        print_name()
        return

    for i in range(2):
        path.append(arr[i])
        run(lev + 1)
        path.pop()


run(0)

#----------------------------------------

# 3명 부분 집합 찾기
arr = ['A', 'B', 'C']
n = len(arr)

def get_sub(tar):
    print(f'target = {tar}', end=' / ')
    for i in range(n):
        # 1 도 되고, 0b1 도 되고, 0x1 도 되는데
        # 왜 0x1 이냐 ?
        # -> 비트 연산임을 명시하는 권장하는 방법
        if tar & 0x1:   # 각 자리의 원소가 포함되어 있나요 ?
            print(arr[i], end='')
        tar >>= 1       # 맨 우측 비트를 삭제한다
                        # == 다음 원소를 확인하겠다.


# 전체 부분집합을 확인해야한다.
for target in range(1 << n):
    get_sub(target)
    print()

#-------------------------------------

# [문제] 카페에 같이 갈 친구가 2명 이상 경우의 수
arr = ['A', 'B', 'C', 'D', 'E']
n = len(arr)

# 1 인 bit 수를 반환하는 함수
def get_count(tar):
    cnt = 0
    # for _ in range(n):
    #     if tar & 0x1:
    #         cnt += 1
    #     tar >>= 1

    # 같은 코드
    for i in range(n):
        if (tar >> i) & 0x1:
            cnt += 1
    return cnt

# 모든 부분 집합 중 원소의 수가 2개 이상인 집합의 수
answer = 0
# 모든 부분 집합을 확인
for target in range(1 << n):
    # 만약, 원소의 개수가 2개 이상이면 answer += 1
    if get_count(target) >= 2:
        answer += 1
print(answer)

#--------------------------------------

# 5명 중 3명을 뽑는 조합 문제
arr = ['A', 'B', 'C', 'D', 'E']
n = 3

path = []

# 5명 중 3명을 뽑는 문제
def recur(cnt, start):
    # N명을 뽑으면 종료
    if cnt == n:
        print(*path)
        return

    # for i in range(이전에 뽑았던 인덱스 + 1부터, len(arr)):
    # start : 이전 재귀로부터 넘겨받아야 하는 값
    for i in range(start, len(arr)):
        path.append(arr[i])
        # i: i번째를 뽑겠다.
        # i + 1을 매개변수로 전달: 다음 재귀 부터는 i+1 부터 고려
        recur(cnt + 1, i + 1)
        path.pop()


recur(0, 0)

#------------------------------------------

# 주사위 3개를 던져 나올 수 있는 모든 조합을 출력
# level: 주사위 3개를 던졌을 때
# branch: 1~6 숫자
N = 3
path = []


def recur(cnt, start):
    if cnt == N:
        print(path)
        return

    for i in range(start, 7):
        path.append(i)
        recur(cnt + 1, i)
        path.pop()


recur(0, 1)

#---------------------------------------

# [문제] 동전의 최소 개수
coin_list = [500, 100, 50, 10]  # 큰 동전부터 앞으로 작성함
target = 1730
cnt = 0

for coin in coin_list:
    possible_cnt = target // coin   # 현재 동전으로 가능한 최대 수
    cnt += possible_cnt             # 정답에 더해준다.
    target -= coin * possible_cnt   # 금액을 빼준다.
print(cnt)

#---------------------------------------------

# [문제] 화장실 대기시간
people = [15, 30, 50, 10]
n = len(people)

# 규칙. 최소 시간인 사람부터 화장실로 들어가자.
people.sort()  # 오름차순 정렬

total_time = 0         # 전체 대기 시간
remain_people = n - 1  # 대기인원 수

for turn in range(n):
    time = people[turn]
    total_time += time * remain_people
    remain_people -= 1

print(total_time)

#--------------------------------------------

# [문제] fraction_knapsack
n = 3
target = 30  # Knapsack KG
things = [(5, 50), (10, 60), (20, 140)]  # (Kg, Price)

# kg 당 가격으로 어떻게 정렬 ?
# 정렬 : (price / kg)
# lambda: 재사용하지 않는 함수
things.sort(key=lambda x: (x[1] / x[0]), reverse=True)

total = 0
for kg, price in things:
    per_price = price / kg

    # 만약 가방에 남은 용량이 얼마되지 않는다면,
    # 물건을 잘라 가방에 넣고 끝낸다.
    if target < kg:
        total += target * per_price
        break

    total += price
    target -= kg

print(int(total))

#----------------------------------------------

