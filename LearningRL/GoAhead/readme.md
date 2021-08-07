# 学习Q Learning

## gym环境配置（将来可以写个安装脚本）
安装gym
```
$ python -m pip install gym
```
查看gym目录
```
$ python -m pip show gym
```
把go_ahead.py放在以下路径：
```
gym\envs\classic_control
```
在gym\env目录下的__init__.py末尾加入
```
register(
    id='GoAhead-v0',    # 环境名
    entry_point='gym.envs.classic_control:GoAhead', # 路径和类名
    max_episode_steps=200, reward_threshold=100.0,  # ？？？
    )
```
在gym\envs\classic_control目录下的__init__.py文件中加入
```
from gym.envs.classic_control.go_ahead import GoAhead
```
测试环境
```
import gym

env = gym.make('GoAhead-v0')
env.reset()
env.render()
```

## gym环境组成
go_ahead.py
```
class GoAhead(gym.Env):
    '''
    继承自gym.Env
    '''

    def __init__(self):
        '''
        初始化变量
        '''


    def step(self, action):
        '''
        与环境交互一次，返回新的状态，回报，是否结束，其他信息
        '''

        return state, reward, done, info


    def reset(self):
        '''
        重置环境，返回状态
        '''

        return state


    def render(self, mode='human', fresh_time=0.1):
        '''
        图形或界面的渲染
        '''

        time.sleep(fresh_time)


    def close(self):
        '''
        关闭环境，释放内存
        '''


    def seed(self, seed=None):
        '''
        随机种子
        '''

        return [seed]
```
## GoAhead测试
```
$ python main.py
```