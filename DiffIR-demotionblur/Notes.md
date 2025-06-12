#### 6-9

- progressive learning那里感觉没必要，可以直接注释掉
  - 取出`lq`和`gt`的部分要保留
- Data那里注意
  - 不需要数据增强
  - 设置mean和norm，适当的归一化一下
  - gt的size之类的可以适当缩放，主要快速看下重建效果
  - 必须要进行一个统一的缩放，像之前datasets的代码里面一样，nlos-ot里面的数据不是缩放好的
  - 还是要注意下img_gt和img_lq的路径是否配对
- 学习率的scheduler那里直接用`TrueCosxxx`那个，参照NAFNet里面的内容
- 网络
  - demotion的那个网络
  - 输出是默认一个残差连接加上原始图像的，这个要去除
  - 输出的激活函数需要注意下，关键是要和dataset里面对数据的归一化相统一 

#### 6-12

- Adam的weight_decay那里不能像之前一样设置为0.05
  - 不然学不到什么东西