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
logging等级：
CRITICAL > ERROR > WARNING > INFO > DEBUG
```
import logging

logging.basicConfig(level=logging.INFO)

logging.debug('<INIT FUNC>')
logging.info('<INIT FUNC>')
```
以上例子仅显示
```
<INIT FUNC>
```
## Q Learning 原理
### 算法流程
```
随机初始化 $Q(s, a)$
重复（对于每一次游戏）：
    初始化 $s$
    重复（对于一次游戏中的每一步）：
        通过Q函数根据状态s选择动作a（\epsilon-greedy）
        采用动作a与环境交互，观察奖励r和新的状态s'
        更新Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
        更新状态s \leftarrow s'
    直到s为终止状态
```
### 算法原理

我们的目标是得到一个最优的策略<img src="http://latex.codecogs.com/gif.latex?\pi^*" />，这个策略能够使得累积折扣回报的期望最大，即

<img src="https://latex.codecogs.com/svg.image?\max_{\pi}&space;\mathop\mathbb{E}_{s&space;\in&space;S}&space;\Bigg\{&space;\sum_{i=0}^{\infty}&space;\gamma^i&space;r(s_{t&plus;i},&space;a_{t&plus;i},&space;s_{t&plus;i&plus;1})&space;\bigg|&space;\pi&space;\Bigg\}" title="\max_{\pi} \mathop\mathbb{E}_{s \in S} \Bigg\{ \sum_{i=0}^{\infty} \gamma^i r(s_{t+i}, a_{t+i}, s_{t+i+1}) \bigg| \pi \Bigg\}" />

<img src="http://latex.codecogs.com/gif.latex?a_{t + i} = \pi(s_{t + i})" />

其中<img src="http://latex.codecogs.com/gif.latex?r(s_{t + i}, a_{t + i}, s_{t + i + 1})" />表示，在状态<img src="http://latex.codecogs.com/gif.latex?s_{t + i}" />下采用动作<img src="http://latex.codecogs.com/gif.latex?a_{t + i}" />得到下一个状态为<img src="http://latex.codecogs.com/gif.latex?s_{t + i + 1}" />的奖励

### 时间差分算法
定义<img src="http://latex.codecogs.com/gif.latex?Q(s, a)" />函数为

<img src="https://latex.codecogs.com/svg.image?Q^{\pi}(s_t,&space;a_t)&space;=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;k}&space;\atop&space;k&space;=&space;1,&space;2,&space;\dots}&space;\Bigg[&space;\sum_{i&space;=&space;0}^{\infty}&space;\gamma^i&space;r(s_{t&space;&plus;&space;i},&space;a_{t&space;&plus;&space;i},&space;s_{t&space;&plus;&space;i&space;&plus;&space;1})&space;\bigg|&space;\pi&space;\Bigg]" title="Q^{\pi}(s_t, a_t) = \mathop{\mathbb{E}}_{s_{t + k} \atop k = 1, 2, \dots} \Bigg[ \sum_{i = 0}^{\infty} \gamma^i r(s_{t + i}, a_{t + i}, s_{t + i + 1}) \bigg| \pi \Bigg]" />

则有

<img src="https://latex.codecogs.com/svg.image?Q^{\pi}(s_{t&space;&plus;&space;1},&space;a_{t&space;&plus;&space;1})&space;=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;1&plus;&space;k}&space;\atop&space;k&space;=&space;1,&space;2,&space;\dots}&space;\Bigg[&space;\sum_{i&space;=&space;0}^{\infty}&space;\gamma^i&space;r(s_{t&space;&plus;&space;i&space;&plus;&space;1},&space;a_{t&space;&plus;&space;i&space;&plus;&space;1},&space;s_{t&space;&plus;&space;i&space;&plus;&space;2})&space;\bigg|&space;\pi&space;\Bigg]" title="Q^{\pi}(s_{t + 1}, a_{t + 1}) = \mathop{\mathbb{E}}_{s_{t + 1+ k} \atop k = 1, 2, \dots} \Bigg[ \sum_{i = 0}^{\infty} \gamma^i r(s_{t + i + 1}, a_{t + i + 1}, s_{t + i + 2}) \bigg| \pi \Bigg]" />

<img src="https://latex.codecogs.com/svg.image?Q^{\pi}(s_t,&space;a_t)&space;=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;k}&space;\atop&space;k&space;=&space;1,&space;2,&space;\dots}&space;\Bigg[&space;\sum_{i&space;=&space;0}^{\infty}&space;\gamma^i&space;r(s_{t&space;&plus;&space;i},&space;a_{t&space;&plus;&space;i},&space;s_{t&space;&plus;&space;i&space;&plus;&space;1})&space;\bigg|&space;\pi&space;\Bigg]" title="Q^{\pi}(s_t, a_t) = \mathop{\mathbb{E}}_{s_{t + k} \atop k = 1, 2, \dots} \Bigg[ \sum_{i = 0}^{\infty} \gamma^i r(s_{t + i}, a_{t + i}, s_{t + i + 1}) \bigg| \pi \Bigg]" />

<img src="https://latex.codecogs.com/svg.image?=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;k}&space;\atop&space;k&space;=&space;1,&space;2,&space;\dots}&space;\Bigg[&space;r(s_{t},&space;a_{t},&space;s_{t&space;&plus;&space;1})&space;&plus;&space;\gamma&space;\sum_{i&space;=&space;0}^{\infty}&space;\gamma^i&space;r(s_{t&space;&plus;&space;i&space;&plus;&space;1},&space;a_{t&space;&plus;&space;i&space;&plus;&space;1},&space;s_{t&space;&plus;&space;i&space;&plus;&space;2})&space;\bigg|&space;\pi&space;\Bigg]&space;\\" title="= \mathop{\mathbb{E}}_{s_{t + k} \atop k = 1, 2, \dots} \Bigg[ r(s_{t}, a_{t}, s_{t + 1}) + \gamma \sum_{i = 0}^{\infty} \gamma^i r(s_{t + i + 1}, a_{t + i + 1}, s_{t + i + 2}) \bigg| \pi \Bigg]" />

<img src="https://latex.codecogs.com/svg.image?=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;1}}&space;\Bigg[&space;r(s_{t},&space;a_{t},&space;s_{t&space;&plus;&space;1})&space;&plus;&space;\gamma&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;k&space;&plus;&space;1}&space;\atop&space;k&space;=&space;1,&space;2,&space;\dots}&space;\bigg[&space;\sum_{i&space;=&space;0}^{\infty}&space;\gamma^i&space;r(s_{t&space;&plus;&space;i&space;&plus;&space;1},&space;a_{t&space;&plus;&space;i&space;&plus;&space;1},&space;s_{t&space;&plus;&space;i&space;&plus;&space;2})&space;\Big|&space;\pi&space;\bigg]&space;\bigg|&space;\pi&space;\Bigg]" title="= \mathop{\mathbb{E}}_{s_{t + 1}} \Bigg[ r(s_{t}, a_{t}, s_{t + 1}) + \gamma \mathop{\mathbb{E}}_{s_{t + k + 1} \atop k = 1, 2, \dots} \bigg[ \sum_{i = 0}^{\infty} \gamma^i r(s_{t + i + 1}, a_{t + i + 1}, s_{t + i + 2}) \Big| \pi \bigg] \bigg| \pi \Bigg]" />

<img src="https://latex.codecogs.com/svg.image?=&space;\mathop{\mathbb{E}}_{s_{t&space;&plus;&space;1}}&space;\Bigg[&space;r(s_{t},&space;a_{t},&space;s_{t&space;&plus;&space;1})&space;&plus;&space;\gamma&space;Q^{\pi}(s_{t&space;&plus;&space;1},&space;a_{t&space;&plus;&space;1})&space;\bigg|&space;\pi&space;\Bigg]" title="= \mathop{\mathbb{E}}_{s_{t + 1}} \Bigg[ r(s_{t}, a_{t}, s_{t + 1}) + \gamma Q^{\pi}(s_{t + 1}, a_{t + 1}) \bigg| \pi \Bigg]" />

损失函数为

<img src="https://latex.codecogs.com/svg.image?loss&space;=&space;\dfrac{1}{2}&space;\Bigg\{&space;\mathop\mathbb{E}_{s_{t&plus;1}}&space;\bigg[&space;r(s_t,&space;a_t,&space;s_{t&plus;1})&space;&plus;&space;\gamma&space;Q^{\pi}(s_{t&plus;1},&space;a_{t&plus;1})&space;\Big|&space;\pi&space;\bigg]&space;-&space;Q^{\pi}(s_t,&space;a_t)&space;\Bigg\}^2" title="loss = \dfrac{1}{2} \Bigg\{ \mathop\mathbb{E}_{s_{t+1}} \bigg[ r(s_t, a_t, s_{t+1}) + \gamma Q^{\pi}(s_{t+1}, a_{t+1}) \Big| \pi \bigg] - Q^{\pi}(s_t, a_t) \Bigg\}^2" />

之后采用梯度下降法求解即可