# Mahjong

当前任务完成度：

- [x] 复盘（获得state）
- [ ] 获取行动空间（action space）
- [ ] 算番（reward）
- [ ] 选择行动（回归/搜索/RL）

## bot.py

注：

1. 目前不使用【长时运行】。但是我把接口写成两个部分方便修改为长时运行

   ```python
   reviewOneStep()  # 复盘一步
   review()         # 复盘全部
   ```

   review 内部循环调用 reviewOneStep 实现全部复盘。只用 reviewOneStep 就可以使用长时运行。

2. 目前从input.json读取输入，提交时改成从input()读取

3. 需要安装算番库 https://github.com/ailab-pku/Chinese-Standard-Mahjong

## const.py

存放常数

## inputX.json

样例输入文件，有3个，包含吃、碰、补杠等情形，用于debug
