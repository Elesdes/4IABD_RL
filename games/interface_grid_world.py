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
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 70
CELL_WIDTH = WINDOW_WIDTH // row_len
CELL_HEIGHT = WINDOW_HEIGHT // column_len

BUTTON_COLOR = (0, 200, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

def draw_grid():
    for y in range(column_len):
        for x in range(row_len):
            rect = pygame.Rect(x*CELL_WIDTH, y*CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(window, WHITE, rect, 1)

def draw_agent(agent_pos):
    x = agent_pos % row_len
    y = agent_pos // row_len
    rect = pygame.Rect(x*CELL_WIDTH, y*CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
    pygame.draw.rect(window, RED, rect)

def draw_game_over():
    font = pygame.font.Font(None, 74)  # Choisissez la taille de la police ici.
    text = font.render('Game Over', True, WHITE)  # Remplacez 'Game Over' par le message que vous voulez.
    text_rect = text.get_rect()
    text_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 4)
    window.blit(text, text_rect)

def draw_move_count(move_count):
    font = pygame.font.Font(None, 36)  # Choisissez la taille de la police ici.
    text = font.render(f'Moves: {move_count}', True, WHITE)
    window.blit(text, (20, 20))  # Dessine le texte en haut à gauche. Modifiez les coordonnées si vous le souhaitez.

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

def ibouble_button():
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
                    return False
                elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
                    return True
def play_game():
    pygame.init()
    clock = pygame.time.Clock()
    move_count = 0
    env = GridWorldEnv(row_len, column_len)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Cliquez sur le bouton X de la fenêtre pour quitter.
                running = False
            elif event.type == pygame.KEYDOWN:
                prev_state_id = env.state_id()
                print(env.available_actions())
                if event.key == pygame.K_LEFT and 0 in env.available_actions():  # Gauche
                    env.step(0)
                    move_count += 1
                elif event.key == pygame.K_RIGHT and 1 in env.available_actions():  # Droite
                    env.step(1)
                    move_count += 1
                elif event.key == pygame.K_DOWN and 2 in env.available_actions():  # Bas
                    env.step(2)
                    move_count += 1
                elif event.key == pygame.K_UP and 3 in env.available_actions():  # Haut
                    env.step(3)
                    move_count += 1
                else:
                    move_count += 2


        window.fill((0, 0, 0))

        if not env.is_game_over():
            draw_grid()
            draw_agent(env.state_id())
            draw_move_count(move_count)
        else:
            draw_game_over()
            draw_move_count(move_count)
            bouble_button("exit", "rejouer")
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                        if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
                            running = False
                        elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
                            running = False
                            play_game()

        pygame.display.flip()
        clock.tick(60)

def play():
    pygame.init()
    clock = pygame.time.Clock()

    env = GridWorldEnv(row_len, column_len)
    # policy = policy_iteration_on_grid_world(env).pi
    policy = value_iteration_on_grid_world(env).pi
    move_count = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        window.fill((0, 0, 0))

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

            window.fill((0, 0, 0))

            draw_game_over()
            draw_move_count(move_count)
            bouble_button("exit", "rejouer")
            i = ibouble_button()
            if i :
                running = False
                play()
            else:
                running = False

            # for event in pygame.event.get():
            #     if event.type == pygame.MOUSEBUTTONDOWN:
            #         mouse_pos = pygame.mouse.get_pos()
            #         if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
            #             if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
            #                 running = False
            #             elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
            #                 running = False
            #                 play()


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
        window.fill((0, 0, 0))
        bouble_button("Play", "Play Game")

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if WINDOW_WIDTH / 2 - BUTTON_WIDTH / 2 < mouse_pos[0] < WINDOW_WIDTH / 2 + BUTTON_WIDTH / 2:
                    if WINDOW_HEIGHT / 2 - BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2:
                        play()
                    elif WINDOW_HEIGHT / 2 + BUTTON_HEIGHT < mouse_pos[1] < WINDOW_HEIGHT / 2 + BUTTON_HEIGHT * 2:
                        play_game()
        clock.tick(60)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_menu()