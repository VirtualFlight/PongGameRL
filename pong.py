# Used for screen capture
from mss import mss
import cv2
# Sending commands
import pyautogui
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker




class PongGame(Env):
    def __init__(self):
        '''
        done location: (450, 735), (690, 800), LOSER/WINNER

        home location: (1142, 1339)

        2 players location: (1618,959)
            Play button: (1140, 1248)

        1 player location: (659,957)
            Hard button location: (1137,1091)
        '''
        super().__init__()
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 256, 256), dtype=np.uint8) #box is commonly used for images
        self.action_space = Discrete(3) #0 = hold_up, 1 = hold_down, 2 = stop_action_up, 3 = stop_action_down
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 75, 'left': 0, 'width': 2550, 'height': 1359} #the entire game area basically
        self.done_location = {'top': 735, 'left': 450, 'width': 250, 'height': 65} #loser/winner text
        self.info = {}

    def step(self, action):
        action_map = {
            0: 'up',
            1: 'down',
            2: 'up',
            3: 'down'
        }

        if action > 1: # Stop action
            pyautogui.keyDown(action_map[action])
        else:
            pyautogui.keyUp(action_map[action])

        done, done_cap = self.get_done()
        observation = self.get_observation()
        reward = 1 #let's keep reward as 1 for now,
        truncated = False # this is to end the episode prematurely -- based on time usually
        return observation, reward, done, truncated, self.info

    def reset(self, seed=None):
        '''
        PreCondition: Assume resetting for palyer 1 hard mode.
        '''
        super().reset(seed=seed)
        pyautogui.click(x=1142, y=1339, clicks=2)
        time.sleep(1)
        pyautogui.click(x=659, y=957)
        pyautogui.click(x=1137, y=1091)

        return self.get_observation(), self.info


    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        '''
        Finished getting the observation of the game.
        '''
        raw = np.array(self.cap.grab(self.game_location)).astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (256, 256))
        channel = np.reshape(resize, (1, 256, 256)) #256 for more detail, although slower training
        # DEBUGGING
        # cv2.namedWindow("Grayscale Observation", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Grayscale Observation", 256, 256)
        # cv2.imshow("Grayscale Observation", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return channel

    def get_done(self):
        '''
        Successfully determine if the game is done.

        Through checking Player 1's winner/loser text.
        '''
        done_cap = np.array(self.cap.grab(self.done_location)).astype(np.uint8)
        # DEBUG
        # img = done_cap[:, :, :3][..., ::-1]
        # plt.imshow(img)
        # plt.show()
        done_strings = ['LOSER', 'WINNER']
        done = False
        gray = cv2.cvtColor(done_cap, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.resize(bw, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        res = pytesseract.image_to_string(bw,config='--psm 13')
        if res[0] != "O" and (res[0] in done_strings[0] or res[0] in done_strings[1]):
            done = True
        return done, done_cap



# -----------------------------------------HELPER FUNCTIONS-----------------------------------------
def get_mouse_pos():
    return pyautogui.position() #x,y

def show_image():
    '''
    Display screen capture image
    '''
    cap = mss()
    game_location = {'top': 75, 'left': 0, 'width': 2550, 'height': 1359}
    img = np.array(cap.grab(game_location))
    img = img[:, :, :3][..., ::-1]  # Remove alpha, convert BGRA to RGB
    plt.imshow(img)
    # plt.axis('off')
    plt.show()