# 그래프2

최소비용신장트리
최소한의 "간선"비용으로 연결하는 트리
사이클 없이 모든 정점이 연결된다.

이 비유 개쩐다
마을에 전봇대를 세우고 전기 연결하는데, 
가능한 적은 돈으로 모든 마을을 연결하기.

Prim 알고리즘
시작 정점에서 출발해서 확장.
우선순위 큐(선입선출), 방문 배열 사용

```python
import heapq

def prim(start, graph, V):
    visited = [False] * V
    heap = [(0, start)]
    total_cost = 0

    while heap:
        cost, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total_cost += cost

        for next_cost, v in graph[u]:
            if not visited[v]:
                heapq.heappush(heap, (next_cost, v))

    return total_cost
```

Kruskal 알고리즘

간선의 가중치 순으로 정렬
유니온 파인드 사용 (for 사이클 방지)

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    x_root = find(x)
    y_root = find(y)
    
    if x_root != y_root:
        parent[y_root] = x_root
        return True
    return False

edges.sort()  # (cost, u, v)
for cost, u, v in edges:
    if union(u, v):
        total_cost += cost
```

최단 경로(다익스트라)
"한 정점" 에서 다른 모든 정점까지의 최소 거리

비유
모든 목적지까지의 최단 시간/비용 계산

```python
import heapq

def dijkstra(start, graph, V):
    dist = [float('inf')] * V
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        cost, u = heapq.heappop(heap)
        if dist[u] < cost:
            continue
        for next_cost, v in graph[u]:
            if dist[v] > dist[u] + next_cost:
                dist[v] = dist[u] + next_cost
                heapq.heappush(heap, (dist[v], v))
    
    return dist
```

# 그래프 1

DFS
```python
def dfs(graph, v, visited):
    visited[v] = True
    print(v, end=' ')

    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)
```
비유하자면 미로에서 한 쪽 벽을 따라 계속 걷는 것

BFS
```python
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True

    while queue:
        v = queue.popleft()
        print(v, end=' ')

        for i in graph[v]:
            if not visited[i]:
                visited[i] = True
                queue.append(i)
```
DFS는 재귀로 계속 들어가고 BFS는 전염병 퍼뜨리듯이.

Union-Find
같은집합/서로소인 집합 관리
크루스칼 알고리즘(최단거리들의 합) 에서 필수
```python
parent = [i for i in range(n+1)]

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a,b):
    a = find(a)
    b = find(b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b
```
비유하자면 친구 무리 만드는 느낌.

# 백트래킹 2
N-Queen 문제?
퀸은 가로/세로/대각선 이동 모두 가능.
퀸들 여러명이 체스판 위에 올라가되, 서로의 영역을 침범하지 않게 올라감.

어떻게 풀어나갈까?
여왕 A가 먼저 자리를 정해.
B가 A영역 제외하고 자리를 정해.
C도 나머지 영역에서 자리를 정해.
근데 자리잡을 공간이 없어?
그럼 A를 바꿔 앉아. 
이런 식으로 모든 경우를 재귀로 시도.

```python
def is_safe(row, col):
    for prev_row in range(row):
        if queens[prev_row] == col:
            return False
        if abs(queens[prev_row] - col) == abs(prev_row - row):
            return False
    
    return True

def solve(row):
    if row == N:
        count += 1
        print("해 발견:", queens)
        return
    
    for col in range(N):
        if is_safe(row, col):
            queens[row] = col  # 여왕 놓기
            solve(row + 1)
            queens[row] -= 1  # 해당 행에서 놓기 실패했으니, 다시 되돌리고 다음 col로 가야하니까 !!

N = 4
queens = [-1] * N  # 각 행에 여왕의 열 위치 저장
count = 0
solve(0)
print("총 해답 수:", count)
```

# 백트래킹 1

```python
N, M = map(int, input().split)
arr = [0] * N
visited = [0] * (N + 1)

def solve(depth):
    if depth == M:
        print(*arr)
        return
    
    for i in range(1, N + 1):
        if visited[i] = 0:
            visited[i] = 1
            arr[depth] = i
            solve(depth + 1)
            visited[i] = 0
```

```python
N, M = map(int, input().split())
arr = [0] * M

def solve(depth, start):
    if depth == M:
        print(*arr)
        return
    
    for i in range(start, N + 1):
        arr[depth] = i
        solve(depth + 1, i + 1)

solve(0, 1)
```

### 🎯 백트래킹 유형별 차이 비교표

| 유형 (Type) | 함수 호출 방식 (Recursive Call) | 예시 출력 (`N=3`, `M=2`) |
|-------------|-------------------------------|---------------------------|
| 순열 (Permutation) | `solve(depth + 1)` + `visited` 사용 | `1 2`, `2 1`, `2 3`, `3 2`, ... |
| 조합 (Combination) | `solve(depth + 1, i + 1)` | `1 2`, `1 3`, `2 3` |
| 중복 조합 (Combination with Repetition) | `solve(depth + 1, i)` | `1 1`, `1 2`, `2 2`, `3 3` |

```python
arr = [1, 2, 3]
N = len(arr)

def solve(idx, subset):
    if idx == N:
        print(subset)
        return
    
    solve(idx + 1, subset + [arr[idx]])
    sove(idx + 1, subset)
```

```python
coins = [500, 100, 50, 10]
change = 1260
count = 0

for coin in coins:
    count += change//coin
    change %= coin
```

```python
meetings.sort(key=lambda x: x[1])
end_time = 0
count = 0

for s, e in meetings:
    if s >= end_time:
        count += 1
        end_time = e
```

완전검색_그리디2
이어서.. 복합문제..