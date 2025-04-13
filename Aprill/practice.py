from collections import deque

# 싸피 형식 입력 파싱 함수
# - 첫 줄: H W 아군 수 적군 수 암호문 수
# - 이후: 맵 정보, 아군 정보, 적 정보, 암호문 순서

def parse_input(data):
    lines = data.strip().split('\n')

    H, W, my_n, enemy_n, cipher_n = map(int, lines[0].split())

    map_data = [list(lines[i+1].strip()) for i in range(H)]

    my_tank_info = lines[H+1].split()  # A 100 R 1 0
    x, y = 0, 0  # 시작 좌표는 항상 (0,0)

    hp = int(my_tank_info[1])
    d = my_tank_info[2]
    m = int(my_tank_info[3])
    mega = int(my_tank_info[4])

    cipher = lines[-cipher_n] if cipher_n > 0 else None

    return map_data, (x, y, hp, d, m, mega), cipher, H, W

# 📌 Caesar 암호 해독 함수

def decrypt_caesar(text, shift=3):
    return ''.join(chr((ord(c)-65-shift)%26 + 65) if c.isalpha() else c for c in text)

# 📌 BFS로 최단 경로 구하기 (W, R 피함)

def bfs(map_data, start, goal, H, W):
    visited = [[False]*W for _ in range(H)]
    parent = [[None]*W for _ in range(H)]
    q = deque([start])
    visited[start[0]][start[1]] = True

    while q:
        x, y = q.popleft()

        if (x, y) == goal: 
            break

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy

            if 0<=nx<H and 0<=ny<W and not visited[nx][ny]:
                if map_data[nx][ny] in ('W', 'R'): 
                    continue

                visited[nx][ny] = True
                parent[nx][ny] = (x, y)
                q.append((nx, ny))

    path = []
    cur = goal

    while cur and cur != start:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]

    path.reverse()
    return path

# 📌 현재 좌표 → 다음 좌표로 가기 위한 커맨드 생성

def get_command(f, t):
    dx, dy = t[0]-f[0], t[1]-f[1]
    if dx == -1: return 'U A'
    if dx ==  1: return 'D A'
    if dy == -1: return 'L A'
    if dy ==  1: return 'R A'

    return 'S'

# 📌 탱크 클래스: 이동, 공격, 해독 모두 담당

class Tank:
    def __init__(self, x, y, hp, d, m, mega):
        self.x, self.y, self.hp = x, y, hp
        self.dir, self.missile, self.mega = d, m, mega
        self.log, self.success = [], False

    def move_to(self, cmd):
        self.log.append(cmd)
        if cmd[0] == 'U': self.x -= 1
        if cmd[0] == 'D': self.x += 1
        if cmd[0] == 'L': self.y -= 1
        if cmd[0] == 'R': self.y += 1

    def use_missile(self):
        if self.missile > 0:
            self.missile -= 1
            self.log.append('R F')
            self.success = True
        elif self.mega > 0:
            self.mega -= 1
            self.log.append('R FM')
            self.success = True
        else:
            self.log.append('S')  # 대기

    def decrypt(self, cipher):
        plain = decrypt_caesar(cipher)
        self.log.append(f'G {plain}')
        self.mega += 1

# 📌 실행 함수: 탱크 이동, 해독, 공격 수행

def run(map_data, tank_info, cipher, H, W):
    tank = Tank(*tank_info)
    
    goal = [(i,j) for i in range(H) for j in range(W) if map_data[i][j] == 'H'][0]
    
    path = bfs(map_data, (tank.x, tank.y), goal, H, W)

    for step in path:
        if map_data[tank.x][tank.y] == 'F':
            tank.decrypt(cipher)

        cmd = get_command((tank.x, tank.y), step)
        tank.move_to(cmd)

    tank.use_missile()

    for cmd in tank.log:
        print(cmd)

# 📌 싸피 인풋 형식 그대로 받아 실행하기 위한 함수

def run_from_input(raw):
    map_data, tank_info, cipher, H, W = parse_input(raw)

    if cipher is None:
        cipher = "EDWWOHVVDI"
        
    run(map_data, tank_info, cipher, H, W)
