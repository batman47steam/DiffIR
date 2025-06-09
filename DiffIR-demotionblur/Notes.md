#### 6-9

- progressive learning那里感觉没必要，可以直接注释掉
- Data那里注意
  - 不需要数据增强
  - 设置mean和norm，适当的归一化一下
  - gt的size之类的可以适当缩放，主要快速看下重建效果
- 学习率的scheduler那里直接用`TrueCosxxx`那个，参照NAFNet里面的内容