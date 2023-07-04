import sys

import pygame
import numpy as np

from games.tictactoe import TicTacToe

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
BUTTON_WIDTH = 500
BUTTON_HEIGHT = 70
BUTTON_SPACING = 20
LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55

LIGHT_BLUE = (14, 41, 84)
DARK_BLUE = (31, 110, 140)
ORANGE = (132, 167, 161)

BUTTON_COLOR = DARK_BLUE
WHITE = (242, 234, 211)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("TicTacToe")

def draw_lines():
    pygame.draw.line(screen, WHITE, (0, 200), (600, 200), LINE_WIDTH)
    pygame.draw.line(screen, WHITE, (0, 400), (600, 400), LINE_WIDTH)
    pygame.draw.line(screen, WHITE, (200, 0), (200, 600), LINE_WIDTH)
    pygame.draw.line(screen, WHITE, (400, 0), (400, 600), LINE_WIDTH)


def draw_figures(env):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if env.board[row][col] == 1:
                pygame.draw.circle(screen, DARK_BLUE, (int(col * 200 + 100), int(row * 200 + 100)), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif env.board[row][col] == -1:
                pygame.draw.line(screen, DARK_BLUE, (col * 200 + SPACE, row * 200 + 200 - SPACE), (col * 200 + 200 - SPACE, row * 200 + SPACE), CROSS_WIDTH)
                pygame.draw.line(screen, DARK_BLUE, (col * 200 + SPACE, row * 200 + SPACE), (col * 200 + 200 - SPACE, row * 200 + 200 - SPACE), CROSS_WIDTH)

def restart(env):
    screen.fill(LIGHT_BLUE)
    draw_lines()
    env.reset()
    draw_figures(env)

def bouble_button(name1, name2):
    draw_button(window, name1, WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2, WINDOW_HEIGHT / 2 - BUTTON_HEIGHT,
                BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)
    draw_button(window, name2, WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2, WINDOW_HEIGHT / 2 + BUTTON_HEIGHT,
                BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_COLOR)

def draw_game_over(winer):
    font = pygame.font.Font(None, 74)
    if winer == -1:
        text = font.render('Circle player won!', True, WHITE)
    else:
        text = font.render('Cross player won!', True, WHITE)
    text_rect = text.get_rect()
    text_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4)
    window.blit(text, text_rect)

def draw_button(screen, text, x, y, width, height, color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.Font(None, 50)
    text = font.render(text, 1, WHITE)
    screen.blit(text, (x + (width / 2 - text.get_width() / 2), y + (height / 2 - text.get_height() / 2)))

def play_game():
    pygame.init()
    screen.fill(LIGHT_BLUE)
    env = TicTacToe()
    # Boucle principale
    running = True
    player = 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX = event.pos[0]
                mouseY = event.pos[1]
                clicked_row = int(mouseY // 200)
                clicked_col = int(mouseX // 200)
                if env.board[clicked_row][clicked_col] == 0:
                    env.play((clicked_row, clicked_col))

        screen.fill(LIGHT_BLUE)
        if not env.is_game_over():
            draw_figures(env)
            draw_lines()
        else:
            draw_game_over(env.player)
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

        pygame.display.update()

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
        env = TicTacToe()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                    if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT * 2 - BUTTON_SPACING * 1.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5:
                        # policy = policy_evaluation_on_line_world(env).pi
                        # play(env, policy)
                        pass
                    elif WINDOW_HEIGHT / 2 - BUTTON_HEIGHT - BUTTON_SPACING * 0.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5:
                        # policy = policy_iteration_on_line_world(env).pi
                        # play(env, policy)
                        pass
                    elif WINDOW_HEIGHT / 2 + BUTTON_SPACING * 0.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5:
                        # policy = value_iteration_on_line_world(env).pi
                        # play(env, policy)
                        pass
                    elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT + BUTTON_SPACING * 1.5 < mouse_pos[
                        1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2 + BUTTON_SPACING * 2.5:
                        play_game()

        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main_menu()
