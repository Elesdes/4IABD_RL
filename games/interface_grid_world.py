import pygame
import time
import sys
from algorithms.dynamic_programming import policy_evaluation_on_grid_world, policy_iteration_on_grid_world, \
    value_iteration_on_grid_world
from games.grid_world import GridWorldEnv

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
row_len = 5
column_len = 5
BUTTON_WIDTH = 500
BUTTON_HEIGHT = 70
BUTTON_SPACING = 20
CELL_WIDTH = WINDOW_WIDTH // row_len
CELL_HEIGHT = WINDOW_HEIGHT // column_len

LIGHT_BLUE = (14, 41, 84)
DARK_BLUE = (31, 110, 140)
ORANGE = (132, 167, 161)

BUTTON_COLOR = DARK_BLUE
WHITE = (242, 234, 211)


def draw_grid():
    for y in range(column_len):
        for x in range(row_len):
            rect = pygame.Rect(x * CELL_WIDTH, y * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(window, WHITE, rect, 1)
            draw_text_in_cell('Perdu', 0, column_len - 1)
            draw_text_in_cell('Gagné', row_len - 1, column_len - 1)


def draw_agent(agent_pos):
    x = agent_pos % row_len
    y = agent_pos // row_len
    rect = pygame.Rect(x * CELL_WIDTH, y * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
    pygame.draw.rect(window, ORANGE, rect)


def draw_game_over(is_win):
    font = pygame.font.Font(None, 74)
    if is_win:
        text = font.render('Gagné', True, WHITE)
    else:
        text = font.render('Game Over', True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4)
    window.blit(text, text_rect)


def draw_move_count(move_count):
    font = pygame.font.Font(None, 36)
    text = font.render(f'Moves: {move_count}', True, WHITE)
    window.blit(text, (20, 20))

def draw_button(screen, text, x, y, width, height, color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.Font(None, 50)
    text = font.render(text, 1, WHITE)
    screen.blit(text, (x + (width / 2 - text.get_width() / 2), y + (height / 2 - text.get_height() / 2)))


def bouble_button(name1, name2):
    draw_button(window, name1, WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2, WINDOW_HEIGHT / 2 - BUTTON_HEIGHT,
                BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)
    draw_button(window, name2, WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2, WINDOW_HEIGHT / 2 + BUTTON_HEIGHT,
                BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)


def draw_text_in_cell(text, row, column):
    font = pygame.font.Font(None, 36)
    rendered_text = font.render(text, True, WHITE)
    text_rect = rendered_text.get_rect()
    cell_width = WINDOW_WIDTH // row_len
    cell_height = WINDOW_HEIGHT // column_len
    x = column * cell_width + cell_width // 2
    y = row * cell_height + cell_height // 2
    text_rect.center = (x, y)
    window.blit(rendered_text, text_rect)


def get_directions(probs, row_len, col_len):
    directions = {i: 0 for i in range(row_len * col_len)}

    for i in range(1, row_len * col_len + 1):
        row = (i - 1) // row_len
        col = (i - 1) % col_len
        neighbours = []

        if col > 0:
            neighbours.append(((row * col_len) + (col - 1)) + 1)
        if col < col_len - 1:
            neighbours.append(((row * col_len) + (col + 1)) + 1)
        if row > 0:
            neighbours.append((((row - 1) * col_len) + col) + 1)
        if row < row_len - 1:
            neighbours.append((((row + 1) * col_len) + col) + 1)

        if row_len * col_len in neighbours:
            if row_len * col_len == i + 1:
                directions[i - 1] = 1
            else:
                directions[i - 1] = 2
            continue

        if not neighbours:
            continue

        max_prob_neighbour = max(neighbours, key=probs.get)

        if max_prob_neighbour == i - 1:
            directions[i - 1] = 0
        elif max_prob_neighbour == i + 1:
            directions[i - 1] = 1
        elif max_prob_neighbour < i:
            directions[i - 1] = 3
        else:
            directions[i - 1] = 2

    return directions


def play_game():
    pygame.init()
    move_count = 0
    env = GridWorldEnv(row_len, column_len)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and 0 in env.available_actions():
                    env.step(0)
                    move_count += 1
                elif event.key == pygame.K_RIGHT and 1 in env.available_actions():
                    env.step(1)
                    move_count += 1
                elif event.key == pygame.K_DOWN and 2 in env.available_actions():
                    env.step(2)
                    move_count += 1
                elif event.key == pygame.K_UP and 3 in env.available_actions():
                    env.step(3)
                    move_count += 1
                else:
                    move_count += 2

        window.fill(LIGHT_BLUE)
        if not env.is_game_over():
            draw_grid()
            draw_agent(env.state_id())
            draw_move_count(move_count)
        else:
            draw_game_over(env.is_game_won())
            draw_move_count(move_count)
            bouble_button("exit", "rejouer")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                        if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
                            running = False
                        elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
                            play_game()
        pygame.display.flip()


def play(env, policy):
    pygame.init()
    clock = pygame.time.Clock()
    move_count = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        window.fill(LIGHT_BLUE)

        if not env.is_game_over():
            prev_state_id = env.state_id()

            action = policy[env.state_id()]
            env.step(action)

            if env.state_id() == prev_state_id:
                move_count += 2
            else:
                move_count += 1

            time.sleep(0.5)
            draw_grid()
            draw_agent(env.state_id())
            draw_move_count(move_count)

        else:
            draw_grid()
            draw_agent(env.state_id())
            draw_move_count(move_count)
            time.sleep(0.5)

            window.fill(LIGHT_BLUE)

            draw_game_over(env.is_game_won())
            draw_move_count(move_count)
            bouble_button("exit", "rejouer")
            # i = ibouble_button()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                        if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
                            running = False
                        elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
                            play(env, policy)

        pygame.display.flip()
        clock.tick(60)

    main_menu()


def main_menu():
    pygame.init()
    global window
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    while running:
        window.fill(LIGHT_BLUE)

        draw_button(window, "Play with policy evaluation", WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
                    WINDOW_HEIGHT / 2 - BUTTON_HEIGHT * 2 - BUTTON_SPACING * 1.5, BUTTON_WIDTH, BUTTON_HEIGHT,
                    BUTTON_COLOR)
        draw_button(window, "Play with policy iteration", WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
                    WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)
        draw_button(window, "Play with value iteration", WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
                    WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)
        draw_button(window, "Play Game", WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
                    WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)

        pygame.display.flip()
        env = GridWorldEnv(row_len, column_len)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                    if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT * 2 - BUTTON_SPACING * 1.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5:
                        policy = get_directions(policy_evaluation_on_grid_world(env), row_len, column_len)
                        play(env, policy)
                    elif WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5:
                        policy = policy_iteration_on_grid_world(env).pi
                        play(env, policy)
                    elif WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5:
                        policy = value_iteration_on_grid_world(env).pi
                        play(env, policy)
                    elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2 + BUTTON_SPACING * 2.5:
                        play_game()

        clock.tick(60)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_menu()
