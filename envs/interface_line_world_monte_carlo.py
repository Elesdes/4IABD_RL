import pygame
import time
import sys
import numpy as np
from algorithms.monte_carlo_methods import (
    monte_carlo_es_on_line_world,
    on_policy_first_visit_monte_carlo_control_on_line_world,
    off_policy_monte_carlo_control_on_line_world,
)
from envs.line_world import LineWorldEnv

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
row_len = 7
column_len = 1
BUTTON_WIDTH = 500
BUTTON_HEIGHT = 70
BUTTON_SPACING = 20
CELL_WIDTH = WINDOW_WIDTH // row_len
CELL_HEIGHT = WINDOW_HEIGHT // row_len

LIGHT_BLUE = (14, 41, 84)
DARK_BLUE = (31, 110, 140)
ORANGE = (132, 167, 161)

BUTTON_COLOR = DARK_BLUE
WHITE = (242, 234, 211)


def draw_grid():
    for y in range(column_len):
        for x in range(row_len):
            rect = pygame.Rect(
                x * CELL_WIDTH,
                y * CELL_HEIGHT + (WINDOW_WIDTH // 2 - CELL_WIDTH // 2),
                CELL_WIDTH,
                CELL_HEIGHT,
            )
            pygame.draw.rect(window, WHITE, rect, 1)
            draw_text_in_cell("Gagné", 0, row_len - 1)
            draw_text_in_cell("Perdu", 0, 0)


def draw_agent(agent_pos):
    x = agent_pos % row_len
    y = agent_pos // row_len
    rect = pygame.Rect(
        x * CELL_WIDTH,
        y * CELL_HEIGHT + (WINDOW_WIDTH // 2 - CELL_WIDTH // 2),
        CELL_WIDTH,
        CELL_HEIGHT,
    )
    pygame.draw.rect(window, ORANGE, rect)


def draw_game_over(is_win):
    font = pygame.font.Font(None, 74)
    if is_win:
        text = font.render("Gagné", True, WHITE)
    else:
        text = font.render("Game Over", True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4)
    window.blit(text, text_rect)


def draw_move_count(move_count):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Moves: {move_count}", True, WHITE)
    window.blit(text, (20, 20))


def draw_button(screen, text, x, y, width, height, color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.Font(None, 50)
    text = font.render(text, 1, WHITE)
    screen.blit(
        text,
        (
            x + (width / 2 - text.get_width() / 2),
            y + (height / 2 - text.get_height() / 2),
        ),
    )


def bouble_button(name1, name2):
    draw_button(
        window,
        name1,
        WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
        WINDOW_HEIGHT / 2 - BUTTON_HEIGHT,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        BUTTON_COLOR,
    )
    draw_button(
        window,
        name2,
        WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
        WINDOW_HEIGHT / 2 + BUTTON_HEIGHT,
        BUTTON_WIDTH,
        BUTTON_HEIGHT,
        BUTTON_COLOR,
    )


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


def get_directions(probs, col_len):
    directions = {i: 0 for i in range(col_len)}
    print("directions : ", directions)

    for i in range(1, col_len + 1):
        neighbours = []

        if i > 1:
            neighbours.append(i - 1)
        if i < col_len:
            neighbours.append(i + 1)

        if col_len in neighbours:
            if col_len == i + 1:
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
    print("directions : ", directions)

    return directions


def play_game():
    pygame.init()
    move_count = 0
    env = LineWorldEnv(7)
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
                    if (
                        WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2
                        < mouse_pos[0]
                        < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2
                    ):
                        if (
                            WINDOW_HEIGHT / 2 - BUTTON_HEIGHT
                            < mouse_pos[1]
                            < WINDOW_HEIGHT / 2
                        ):
                            running = False
                        elif (
                            WINDOW_HEIGHT / 2 + BUTTON_HEIGHT
                            < mouse_pos[1]
                            < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2
                        ):
                            play_game()
        pygame.display.flip()


def play(env, policy):
    pygame.init()
    clock = pygame.time.Clock()
    move_count = 0
    running = True
    env.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        window.fill(LIGHT_BLUE)

        if not env.is_game_over():
            prev_state_id = env.state_id()

            action = policy[env.state_id()]
            if not isinstance(action, int):
                action = max(action, key=lambda k: action[k])
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if (
                        WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2
                        < mouse_pos[0]
                        < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2
                    ):
                        if (
                            WINDOW_HEIGHT / 2 - BUTTON_HEIGHT
                            < mouse_pos[1]
                            < WINDOW_HEIGHT / 2
                        ):
                            running = False
                        elif (
                            WINDOW_HEIGHT / 2 + BUTTON_HEIGHT
                            < mouse_pos[1]
                            < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2
                        ):
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

        draw_button(
            window,
            "Play with monte carlo es",
            WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
            WINDOW_HEIGHT / 2 - BUTTON_HEIGHT * 2 - BUTTON_SPACING * 1.5,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            BUTTON_COLOR,
        )
        draw_button(
            window,
            "Play with on policy monte carlo",
            WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
            WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            BUTTON_COLOR,
        )
        draw_button(
            window,
            "Play with off policy monte carlo",
            WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
            WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            BUTTON_COLOR,
        )
        draw_button(
            window,
            "Play Game",
            WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2,
            WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
            BUTTON_COLOR,
        )

        pygame.display.flip()
        env = LineWorldEnv(row_len)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if (
                    WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2
                    < mouse_pos[0]
                    < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2
                ):
                    if (
                        WINDOW_HEIGHT / 2 - BUTTON_HEIGHT * 2 - BUTTON_SPACING * 1.5
                        < mouse_pos[1]
                        < WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5
                    ):
                        policy = monte_carlo_es_on_line_world(env).pi
                        play(env, policy)
                    elif (
                        WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5
                        < mouse_pos[1]
                        < WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5
                    ):
                        policy = on_policy_first_visit_monte_carlo_control_on_line_world(env).pi
                        play(env, policy)
                    elif (
                        WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5
                        < mouse_pos[1]
                        < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5
                    ):
                        policy = off_policy_monte_carlo_control_on_line_world(env).pi
                        play(env, policy)
                    elif (
                        WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5
                        < mouse_pos[1]
                        < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2 + BUTTON_SPACING * 2.5
                    ):
                        play_game()

        clock.tick(60)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_menu()
