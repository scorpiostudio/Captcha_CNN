# 需求说明
- 本项目主要解决验证码识别（CAPTCHA）问题。
- 本项目使用深度学习CNN卷积神经网络对验证码进行识别，使用TensorFlow框架构建构建CNN网络
并训练。
- 本项目支持识别的验证码由数字、小写字母、大写字母组成，支持4位或6位等常见验证码，不支持
中文字符。

# 模块接口
## config模块
### 数据集路径设置
- config模块用于项目配置，用户需要在当前项目路径下建立data、train_data、test_data、
validation_data目录，用户需要将所有样本的数据集放入data目录。
- data_path = work_path + '/data'  #数据集
- train_data_path = work_path + '/train_data'， 训练集路径
- test_data_path = work_path + '/test_data'， 测试集路径
- validation_data_path = work_path + '/validation_data'，验证集路径
- model_path = work_path + '/model/crack_captcha_model'，其中
model_path保存训练的模型的路径，crack_captcha_model为模型文件名前缀

### 验证码图片设置
- IMAGE_HEIGHT = 26，设置验证码图片的高度，单位为像素值。
- IMAGE_WIDTH = 80， 设置验证码图片的宽度，单位为像素值
- MAX_CAPTCHA = 4，设置验证码的最长字符数

### 训练参数设置
- 下列两个参数需要根据训练进行参数调优。
- batch_size = 32，每一批次训练的数据样本数量，通常如果数据集的总样本规模在10000左右，
设置为32，如果数据集的总样本规模在20000左右设置为64。
- learning_rate = 0.001，学习率设置，提出设置为0.001，一般不能超过0.1

## build_dataset模块
- 本模块主要用于划分数据集，验证集、测试集、训练集规模的占比为1：1：8。
- 用户将数据样本放入data目录即可执行本模块进行数据集划分，计算机将会随机将data目录的
数据样本划分到test_data、validation_data、train_data目录下，避免人工划分导致的人为影响。

## utils模块
- utils模块主要提供了根据函数

## cnn_architecture模块
- cnn_architecture模块主要用于构建CNN卷积神经网络模型架构

## train_cnn_model模块
- train_cnn_model模块用于根据训练集、验证集训练CNN卷积神经网络模型，并在验证集ACC到达
一定值时保存训练模型，并验证集ACC达到临界值时最终完成训练。

## crack_captcha模块
- crack_captcha模块用于根据测试集的测试数据样本对训练的CNN神经网络模型进行测试，并给出
测试集ACC。

# 使用流程
- 将所有数据样本放到数据集data目录。
- 执行build_dataset模块进行数据集划分。
- 对config模块进行配置。
- 执行train_cnn_model模块进行训练。
- 执行crack_captcha模块进行测试。
- 使用本项目提供的数据集，训练集ACC为0.9975，验证集ACC为0.995，测试集ACC为0.977。

