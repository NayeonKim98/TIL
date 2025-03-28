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
    # 여왕이 (row, col)에 앉을 수 있는지 검사
    for prev_row in range(row):
        if queens[prev_row] == col:
            return False  # 같은 열이면 X
        if abs(queens[prev_row] - col) == abs(prev_row - row):
            return False  # 같은 대각선 X
    return True

def solve[row]:
    if row == N:  # 모든 여왕을 배치했으면
        count[0] += 1
        return
    
    for col in range(N):
        if is_safe(row, col):
            queens[row] = col  # 여왕 배치
            solve(row + 1)

N = 4
queens = [-1] * N  # 각 행에 여왕이 있는 열 정보 저장
count = 0
solve(0)
print("해결 가능한 경우의 수 :", count)
```