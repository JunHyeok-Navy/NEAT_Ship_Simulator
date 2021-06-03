# Made By Jun Hyeok Park
# Referred 'ras maxim - pygame 2d car physics, NEAT DOCS
# For Smart Navy Conference, Graduationn Thesis
#협수로ver

import pygame
import os
import math
from math import sin, radians, degrees, copysign, sqrt, cos, trunc, exp
import sys
import random
import neat
import gym
from pygame.math import Vector2
import numpy as np
import multiprocessing
import pickle
import csv
import pandas as pd

df = pd.read_csv('data_main1.csv',index_col=[0])
screen_width = 1280
screen_height = 720
generation = 0
check_point = [[145, 96], [401,338], [645,338], [816,200], [900,200]]
RED = (255, 0, 0)
time_spent = 0
RL_MODE = False

angle_w = 0
velocity_w = 0

def rot_center(image, angle):
    center = image.get_rect().center
    rotated_image = pygame.transform.rotate(image, angle)
    ar = [rotated_image, center]
    return ar

def draw_wind(screen):
    wind_font = pygame.font.SysFont("Arial", 20)
    windv_text = wind_font.render("v = " + str(round(velocity_w, 3)) + "knot", True, (255, 255, 255))
    windv_text_rect = windv_text.get_rect()
    windv_text_rect.center = (1100, 50)
    screen.blit(windv_text, windv_text_rect)
    
def deg(a):
    pi = 3.14159265358979323846264338327950288419716939937510582097494
    a = a*pi/180
    return a

class Mode_Human:
    def draw_line(self, screen):
        pygame.draw.lines(screen, (255, 255, 0), False, check_point, 3)
        
    
    def collision(self):
        self.ship_alive = True
        # 맵 밖으로 나가면 리셋       
        if self.pos[0] <= -1 or self.pos[0] >= 1218:
            self.ship_alive = False
            
        elif self.pos[1] <=-1 or self.pos[1] >= 660:
        
            self.ship_alive = False
            

        if time_spent >= 600 and self.ship_alive:
             self.ship_alive = False
                
        if self.current_check == 0:
            if time_spent >5:
                self.ship_alive = False
            
        elif self.current_check == 1:
            if abs(self.center[0] - self.center[1]-63)/sqrt(2)> 20: 
                self.ship_alive = False
            elif time_spent > 20:
                self.ship_alive = False
            elif self.center[0] > 405:
                self.ship_alive = False
            elif self.angle_b > -55 and self.angle_b < -65:
                self.ship_alive = False
                
        elif self.current_check == 2:
            if abs(self.center[1]-338) > 30:
                self.ship_alive = False
            elif self.center[0] >440:
                if abs(self.angle_b) > 40:
                    self.ship_alive = False
            elif time_spent > 40:
                self.ship_alive = False

                
        elif self.current_check == 3:
            if abs(46*self.center[0]+57*self.center[1]-48936)/73.2461603 > 30 :
                self.ship_alive = False
            elif self.center[0] > 650:
                if self.angle_b <10 :
                    self.ship_alive = False
            elif time_spent > 60:
                self.ship_alive = False
        elif self.current_check == 4:
            self.goal = True
            self.ship_alive = False
            
    def get_distance_mean(self):
     
        self.dis_m += abs(Ship.distance_return(self))
        self.count += 1
        
        return np.array([self.dis_m, self.count])
        
        
                
    def get_reward(self):
        reward = 0
        
        if self.ship_alive:
            if self.current_check == 1:
                reward = 5
                if abs(self.center[0] - self.center[1]-63)/sqrt(2) < 20 and self.center[0] < 410:
                    reward += self.center[0]*0.1
                    if abs(self.center[0] - self.center[1]-63) != 0:
                        reward += 3/abs(self.center[0] - self.center[1]-63)
                    elif abs(self.center[0] - self.center[1]-63) == 0:
                        reward += 20
                    if self.velocity.x < 0:
                        reward -= 100
                else:
                    reward -= 5
                
            elif self.current_check == 2:
                reward = 10
                if self.center[0] < 645 and self.center[1] < 380:  
                    reward += self.center[0]*0.5
                    if abs(self.center[1]-338) != 0:
                        reward += 1/abs(self.center[1]-338)
                    elif abs(self.center[1]-338) == 0:
                        reward += 20
                        
                    if self.velocity.x < 20:
                        reward -= 100
                else:
                    reward -= 4
                    
            elif self.current_check == 3:
                if self.center[0] < 816:
                    reward += self.center[0]*0.0001
                    reward += self.center[1]*0.008
                    if abs(46*self.center[0]+57*self.center[1]-48936) != 0:
                        reward += 2*73.2461603/abs(46*self.center[0]+57*self.center[1]-48936)
                    elif abs(46*self.center[0]+57*self.center[1]-48936) == 0:
                        reward += 50
                    elif self.angle_b < 40 or self.angle_b >36:
                        reward += 100
                        
                    if self.velocity.x < 20:
                        reward -= 100
                else:
                    reward -= 10
                    
            return reward
        
        if not self.ship_alive:
            reward += -10000 / (self.current_check + 1)
            return reward

        elif self.goal:
            reward += 10000
            return reward
    
    def get_goal(self):
        global p
        p = check_point[self.current_check]
        dist = Ship.get_distance(p, self.center)
        if dist < 20:
            self.current_check += 1
        elif self.current_check >= 4:
            self.goal = True
    
    def rot_center(self, image, angle):

        rotated_image = pygame.transform.rotate(image, angle)
        rect = rotated_image.get_rect()
        
        a = [rotated_image]
        return a
    
    def External_Force(self):
        global velocity_w
        velocity_w = 10
        angle_w = 270
        return velocity_w, angle_w
    
    def __init__(self):
        self.surface = pygame.image.load("ship_1.png")
        self.arrow_i = pygame.image.load("arrow.png")
        self.rotate_surface = self.surface
        self.rotate_arrow_i = self.arrow_i
        self.angle_b = -45
        self.angle_r = 0
        self.pos = [72.5, 22]
        self.velocity = Vector2(0.1, 0.0)
        self.center = [100, 37]
        self.ship_alive = True
        self.goal = False
        self.distance = 0
        self.max_angle_r = 5
        self.acceleration = Vector2(0.2, 0.1)
        self.power= 0.5
        self.max_acceleration = 3
        self.max_velocity =50
        self.max_velocity_y = 5
        self.max_power = 20
        self.brake_deceleration = 10
        self.free_deceleration = 10
        self.current_check = 0
        self.check_Flag = False
        self.dis_m = 0
        self.count = 0
        self.sum = np.array([0.0 , 0.0])
        self.gene = 0
    
    def run(self):
        global RL_MODE, df
        Ship_image = pygame.image.load('ship_1.png')
        Map = pygame.image.load('map.png')
        self.ship = Ship()
        global reward
        global episode
        pygame.init()
        pygame.display.set_caption("선박 시뮬레이터(협수로ver)")
        width = 1280
        height = 720
        self.screen = pygame.display.set_mode((width, height))
        self.exit = False
        a=0



        while not self.exit and not RL_MODE:

            # 키 입력 및 해당 키 입력에 따른 변수값 조정
            pressed = pygame.key.get_pressed()

            if pressed[pygame.K_UP]:
                self.power += 1
            elif pressed[pygame.K_F1]:
                RL_MODE = True
            elif pressed[pygame.K_F2]:
                RL_MODE = False

            elif pressed[pygame.K_DOWN]:
                self.power -= 1

            # 물의 저항    
            else:
                # 저항값
                forceout_x = -self.velocity.x
                forceout_y = -self.velocity.y * 2

                # X축 저항
                if abs(self.velocity.x) > 0.05:
                    self.power = 0
                    self.velocity.x += forceout_x*0.005
                    self.velocity.y += forceout_y * 0.005
                elif abs(self.velocity.x) <= 0.05 and abs(self.velocity.x) >0.001:
                    self.velocity.x += forceout_x*0.01
                    self.velocity.y += forceout_y * 0.01
                elif abs(self.velocity.x) <= 0.001: 
                    self.velocity.x = 0
                    self.velocity.y = 0

            # 우현 선회       
            if pressed[pygame.K_RIGHT]:
                self.angle_r += -1*0.001
            # 좌현 선회
            elif pressed[pygame.K_LEFT]:
                self.angle_r += 1 * 0.001
            # 키 바로
            else:
                self.angle_r = 0
                self.angle_r = max(-45, min(self.angle_r, 45))

            # 동력, 가속도, 속도, 위치 업데이트
            Ship.update(self, 0.017)

            self.sum += self.get_distance_mean()

            a+=self.get_reward()
            
        
            if not self.ship_alive:
                self.ship_alive= True
                self.gene += 1  
                print(a)
                print(self.sum[0]/self.sum[1])
                df2 = pd.DataFrame({'Distance_Mean':[self.sum[0]/self.sum[1]], 'Reward':[a], 'Gap':[b]})
                df=df.append(df2)
                df.to_csv('data_main1.csv', index=False)
                
                
                self.pos = [72.5, 22]
                self.velocity.x =0
                self.velocity.y = 0
                self.power = 0
                self.angle_b = -45
                self.current_check = 0
                
            # 실시간 배 그리기
            self.screen.fill((0, 0, 0))
            rotated = rot_center(Ship_image, self.angle_b)[0]
            self.screen.blit(Map, (0, 0))
            if self.current_check < 4:
                self.draw_line(self.screen)
            self.screen.blit(rotated, self.pos)
#             pygame.draw.circle(self.screen, (255,255,0), self.center, 3, 1)
            
           
                
            pygame.display.flip()

            # 종료키 누르면 종료
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1:
                        RL_MODE = True
                        break
                    elif event.key == pygame.K_F2:
                        RL_MODE = False
            
            
class Ship:
    def __init__(self):
        self.surface = pygame.image.load("ship_1.png")
        self.arrow_i = pygame.image.load("arrow.png")
        self.rotate_surface = self.surface
        self.rotate_arrow_i = self.arrow_i
        self.angle_b = -45
        self.angle_r = 0
        self.pos = [72.5, 22]
        self.velocity = Vector2(0.1, 0.0)
        self.center = [100, 37]
        self.ship_alive = True
        self.goal = False
        self.distance = 0
        self.max_angle_r = 5
        self.acceleration = Vector2(0.2, 0.1)
        self.power= 0.5
        self.max_acceleration = 3
        self.max_velocity = 50
        self.max_velocity_y = 5
        self.max_power = 20
        self.brake_deceleration = 10
        self.free_deceleration = 10
        self.current_check = 0
        self.check_Flag = False
        self.dis_m = 0
        self.count = 0
        self.sum = np.array([0.0 , 0.0])
      
        
    def External_Force(self):
        global velocity_w
        velocity_w = 10
        angle_w = 270
        return velocity_w, angle_w

    def draw(self, screen):
        screen.blit(self.rotate_arrow_i, (1150, 5))
        screen.blit(self.rotate_surface, self.pos)
#         if self.current_check <4:
#             pygame.draw.circle(screen, RED, check_point[self.current_check], 30, 1)
        
    def draw_center(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), self.center, 2, 1)
    def draw_line(self, screen):
        pygame.draw.lines(screen, (255, 255, 0), False, check_point, 2)
        

    def collision(self):
        self.ship_alive = True
        # 맵 밖으로 나가면 리셋       
        if self.pos[0] <= -1 or self.pos[0] >= 1218:
            self.ship_alive = False
            
        elif self.pos[1] <=-1 or self.pos[1] >= 660:
        
            self.ship_alive = False
            

        if time_spent >= 600 and self.ship_alive:
             self.ship_alive = False
                
        if self.current_check == 0:
            if time_spent >5:
                self.ship_alive = False
            
        elif self.current_check == 1:
            if abs(self.center[0] - self.center[1]-63)/sqrt(2)> 20: 
                self.ship_alive = False
            elif time_spent > 20:
                self.ship_alive = False
            elif self.center[0] > 405:
                self.ship_alive = False
            elif self.angle_b > -55 and self.angle_b < -65:
                self.ship_alive = False
                
        elif self.current_check == 2:
            if abs(self.center[1]-338) > 30:
                self.ship_alive = False
            elif self.center[0] >440:
                if abs(self.angle_b) > 40:
                    self.ship_alive = False
            elif time_spent > 40:
                self.ship_alive = False

                
        elif self.current_check == 3:
            if abs(46*self.center[0]+57*self.center[1]-48936)/73.2461603 > 30 :
                self.ship_alive = False
            elif self.center[0] > 650:
                if self.angle_b <10 :
                    self.ship_alive = False
            elif time_spent > 60:
                self.ship_alive = False
        elif self.current_check == 4:
            self.goal = True
            self.ship_alive = False

    def update(self, dt):
        #선회
        global angle_w, max_velocity_w, time_spent
        velocity_w, angle_w = self.External_Force()
        self.rotate_arrow_i = self.rot_center(self.arrow_i, -angle_w)[0]
        if self.angle_r != 0:
            self.rotate_surface = self.rot_center(self.surface, self.angle_b)[0]
            self.acceleration.x = self.power * sin(deg(self.angle_r))**2+0.1*cos(deg(angle_w))*velocity_w*sin(deg(angle_w+-self.angle_b))**2
            self.acceleration.y = (self.power/2)*sin(deg(2*self.angle_r))+(sin(deg(angle_w))*velocity_w*sin(deg(angle_w+-self.angle_b))**2)*0.4
            self.velocity.x += self.acceleration.x
            self.velocity.y += self.acceleration.y*0.01
            self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))
            self.velocity.y = max(-self.max_velocity_y, min(self.velocity.y, self.max_velocity_y))
            self.acceleration.x = max(-self.max_acceleration, min(self.acceleration.x, self.max_acceleration))
            self.acceleration.y = max(-self.max_acceleration, min(self.acceleration.y, self.max_acceleration))
            self.power = max(-self.max_power, min(self.power, self.max_power))
            self.angle_b += self.velocity.x * self.angle_r*0.05
            self.pos += self.velocity.rotate(-self.angle_b) * dt
            self.center = [self.pos[0]+27.5, self.pos[1]+15]
            
        #타 미사용
        else:
            self.rotate_surface = self.rot_center(self.surface, self.angle_b)[0]
            self.acceleration.x = self.power+0.2*cos(deg(angle_w))*velocity_w*sin(deg(angle_w+-self.angle_b))**2
            self.acceleration.y = 0.4*sin(deg(angle_w))*velocity_w*sin(deg(angle_w+-self.angle_b))**2
            self.velocity.x += self.acceleration.x
            self.velocity.y += self.acceleration.y*0.01
            self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))
            self.velocity.y = max(-self.max_velocity_y, min(self.velocity.y, self.max_velocity_y))
            self.acceleration.x = max(-self.max_acceleration, min(self.acceleration.x, self.max_acceleration))
            self.acceleration.y = max(-self.max_acceleration, min(self.acceleration.y, self.max_acceleration))
            self.power = max(-self.max_power, min(self.power, self.max_power))
            self.pos += self.velocity.rotate(-self.angle_b) * dt
            self.center = [self.pos[0]+27.5, self.pos[1]+15]
      
        time_spent += 1 * 0.00057
        self.distance += math.sqrt((self.velocity.x)**2+(self.velocity.y)**2) * 0.001
        self.get_goal()
        self.collision()
        

    def distance_return(self):
        if self.current_check == 0:
            return Ship.get_distance(self.center, [145, 82])
        elif self.current_check == 1:
            return (self.center[0] - self.center[1]-63)/sqrt(2)
        elif self.current_check == 2:
            return self.center[1]-338
        elif self.current_check == 3:
            return (46*self.center[0]+57*self.center[1]-48936)/73.2461603
        elif self.current_check ==4:
            return 0
        
    def angle_gap(self):
        if self.current_check == 1:
            return (self.angle_b+45)
        elif self.current_check == 2:
            return (self.angle_b)
        elif self.current_check == 3:
            return (38.9-self.angle_b)
        else:
            return 0
    
    def angle_gap_wind(self):
        return (self.angle_b + angle_w)
        
    def distance_checkp(self):
        if self.current_check == 0:
            return Ship.get_distance(self.center, check_point[0])
        elif self.current_check == 1:
            return Ship.get_distance(self.center, check_point[1])
        elif self.current_check == 2:
            return Ship.get_distance(self.center, check_point[2])
        elif self.current_check == 3:
            return Ship.get_distance(self.center, check_point[3])
        elif self.current_check == 4:
            return 40

    def get_data(self):
        ret = [self.distance_return(), self.distance_checkp()**2, self.angle_gap()]
        return ret

    def get_alive(self):
        return self.ship_alive
    
    def get_distance(p1, p2):
        return sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    
    def rot_center(self, image, angle):

        rotated_image = pygame.transform.rotate(image, angle)
        rect = rotated_image.get_rect()
        
        a = [rotated_image]
        return a

    def get_goal(self):
        global p
        p = check_point[self.current_check]
        dist = Ship.get_distance(p, self.center)
        if dist < 30 and self.current_check < 4:
            self.current_check += 1
        elif self.current_check >= 4:
            self.goal = True
            self.ship_alive = False
            
    def get_distance_mean(self):
     
        self.dis_m += abs(Ship.distance_return(self))
        self.count += 1
        
        return np.array([self.dis_m, self.count])

    def get_reward(self):
        reward = 0
        
        if self.ship_alive:
            if self.current_check == 1:
                reward = 5
                if abs(self.center[0] - self.center[1]-63)/sqrt(2) < 20 and self.center[0] < 410:
                    reward += self.center[0]*0.1
                    if abs(self.center[0] - self.center[1]-63) != 0:
                        reward += 3/abs(self.center[0] - self.center[1]-63)
                    elif abs(self.center[0] - self.center[1]-63) == 0:
                        reward += 20
                    if self.velocity.x < 0:
                        reward -= 100
                else:
                    reward -= 5
                
            elif self.current_check == 2:
                reward = 10
                if self.center[0] < 645 and self.center[1] < 380:  
                    reward += self.center[0]*0.5
                    if abs(self.center[1]-338) != 0:
                        reward += 1/abs(self.center[1]-338)
                    elif abs(self.center[1]-338) == 0:
                        reward += 20
                        
                    if self.velocity.x < 20:
                        reward -= 100
                else:
                    reward -= 4
                    
            elif self.current_check == 3:
                if self.center[0] < 816:
                    reward += self.center[0]*0.0001
                    reward += self.center[1]*0.008
                    if abs(46*self.center[0]+57*self.center[1]-48936) != 0:
                        reward += 2*73.2461603/abs(46*self.center[0]+57*self.center[1]-48936)
                    elif abs(46*self.center[0]+57*self.center[1]-48936) == 0:
                        reward += 50
                    elif self.angle_b < 40 or self.angle_b >36:
                        reward += 100
                        
                    if self.velocity.x < 20:
                        reward -= 100
                else:
                    reward -= 10
                    
            return reward
        
        if not self.ship_alive:
            reward += -10000 / (self.current_check + 1)
            return reward

        elif self.goal:
            reward += 10000
            return reward
        
        
def run_ship(genomes, config):
    global RL_MODE
    global generation, time_spent, df
    theta = 0
    
    nets = []
    ships = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ships.append(Ship())

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 20)
    font = pygame.font.SysFont("Arial", 20)
    map = pygame.image.load('map.png')

    global generation, time_spent
    generation += 1
    while RL_MODE:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for index, ship in enumerate(ships):
            output = nets[index].activate(ship.get_data())
            i = output.index(max(output))
            if i == 0:
                ship.angle_r += -1*0.001
            elif i == 1:
                ship.angle_r += +1*0.001
            elif i == 2:
                ship.power += 1 * 0.008
            elif i == 3:
                ship.power -= 1 * 0.008
            elif i == 4:
                ship.angle_r += +1*0.005
            elif i == 5:
                ship.angle_r += -1*0.005


        remain_ships = 0
        for i, ship in enumerate(ships):
            if ship.get_alive():
                remain_ships += 1
                ship.update(0.017)
                ship.sum += ship.get_distance_mean()
                genomes[i][1].fitness += ship.get_reward()
                theta = 1
            elif not ship.get_alive():
                genomes[i][1].fitness += ship.get_reward()
                theta = 1
        max_value = []
        for i in range (49):
            max_value.append(genomes[i][1].fitness)
                
        if remain_ships == 0:
            time_spent = 0
            print(ship.sum[0]/ship.sum[1])
            print(max(max_value))
            df2 = pd.DataFrame({'Distance_Mean':[math.floor(ship.sum[0]/ship.sum[1])], 'Reward':[math.floor(max(max_value))], 'Generation':[generation]})
            df = df.append(df2)
            df.to_csv('data_main1.csv', index=False)
            max_value = []
            break
            
        elif ship.goal:
            ship.goal = False
            time_spent = 0
            print(ship.sum[0]/ship.sum[1])
            print(max(max_value))
            df2 = pd.DataFrame({'Distance_Mean':[math.floor(ship.sum[0]/ship.sum[1])], 'Reward':[math.floor(max(max_value))], 'Generation':[generation]})
            df = df.append(df2)
            df.to_csv('data_main1.csv', index=False)
            max_value = []
            break

        screen.blit(map, (0, 0))
#         pygame.draw.lines(screen, (0, 0, 155), False, check_point, 60)
        pygame.draw.lines(screen, (255, 0, 0), False, check_point, 2)
        
        for ship in ships:
            if ship.get_alive():
                ship.draw(screen)
#                 ship.draw_center(screen)

        text = generation_font.render("age : " + str(generation-1), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (screen_width-100, 100)
        screen.blit(text, text_rect)
        

        text = font.render("remain : " + str(remain_ships), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (screen_width-100, 150)
        screen.blit(text, text_rect)
        draw_wind(screen)
        pressed = pygame.key.get_pressed()
        pygame.display.flip()
        clock.tick(0)
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_F1]:
            RL_MODE = True
        elif pressed[pygame.K_F2]:
            RL_MODE = False
            
def Mode_RL():
    config_path = 'config.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
#     p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1963')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
#     check_point_neat.run(run_ship, 2)

    p.add_reporter(neat.Checkpointer(4999))

    trained = p.run(run_ship, 5000)
    
                
if __name__ == "__main__":
    while True:
            
        if RL_MODE:
            print('Reinforcement Mode On')
            Mode_RL()

        elif not RL_MODE:
            pygame.quit()
            print('Reinforcement Mode Off')
            game = Mode_Human()
        game.run()