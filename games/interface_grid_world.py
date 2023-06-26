import pygame
import time
from algorithms.dynamic_programming import policy_evaluation_on_grid_world, policy_iteration_on_grid_world, \
    value_iteration_on_grid_world
from games.grid_world import GridWorldEnv

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
row_len = 5
column_len = 5
CELL_WIDTH = WINDOW_WIDTH // row_len
CELL_HEIGHT = WINDOW_HEIGHT // column_len

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


if __name__ == "__main__":
    pygame.init()
    global window
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    env = GridWorldEnv(row_len, column_len)
    # policy = policy_iteration_on_grid_world(env).pi
    policy = value_iteration_on_grid_world(env).pi

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((0, 0, 0))

        draw_grid()
        draw_agent(env.state_id())

        if not env.is_game_over():
            action = policy[env.state_id()]
            env.step(action)
            time.sleep(0.5)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
