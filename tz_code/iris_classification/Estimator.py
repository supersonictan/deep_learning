import tensorflow as tf
import os
import pandas as pd




# 自定义模型函数
def my_model_fn(features, labels, mode, params):
    # 输入层,feature_columns对应Classifier(feature_columns=...)
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # 隐藏层,hidden_units对应Classifier(unit=[10,10])，2个各含10节点的隐藏层
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)  # wx+b

    # 输出层，n_classes对应3种鸢尾花
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    # 预测
    predicted_classes = tf.argmax(logits, 1)  # 预测的结果中最大值即种类
    # 损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


    # Mode=训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 用它优化损失函数，达到损失最少精度最高
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        # 执行优化！
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    # Mode=预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # 拼成列表[[3],[2]]格式
            'probabilities': tf.nn.softmax(logits),  # 把[-1.3,2.6,-0.9]规则化到0~1范围,表示可能性
            'logits': logits,  # [-1.3,2.6,-0.9]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)  # 返回这样的对象


    # Mode=评估
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')  # 计算精度
    metrics = {'accuracy': accuracy}  # 返回格式
    tf.summary.scalar('accuracy', accuracy[1])  # 仅为了后面图表统计使用
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)



FILE_TRAIN = "/Users/tanzhen/Desktop/code/deep_learning/tz_code/iris_classification/iris_training.csv"
FILE_TEST = "/Users/tanzhen/Desktop/code/deep_learning/tz_code/iris_classification/iris_test.csv"
FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


train = pd.read_csv(FILE_TRAIN, names=FUTURES, header=0)
test = pd.read_csv(FILE_TEST, names=FUTURES, header=0)

train_x, train_y = train, train.pop('Species')
test_x, test_y = test, test.pop('Species')


### 创建分类器
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={'feature_columns': feature_columns, 'hidden_units': [10, 10], 'n_classes': 3}
)


#输入数据函数
batch_size=100
def train_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)  #每次随机调整数据顺序
    return dataset.make_one_shot_iterator().get_next()


#开始训练
classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 100), steps=1000)


### 模型评估

#输入函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


# 评估我们训练出来的模型质量
eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))
print(eval_result)


tf.logging.set_verbosity(tf.logging.INFO)
models_path='/Users/tanzhen/Desktop/code/deep_learning/tz_code/iris_classification/mymodels/'
classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=models_path,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })
"""
# 进行预测
for i in range(0, 100):
    print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
    a, b, c, d = map(float, input().split(','))  # 捕获用户输入的数字
    predict_x = {
        'SepalLength': [a],
        'SepalWidth': [b],
        'PetalLength': [c],
        'PetalWidth': [d],
    }

    # 进行预测
    predictions = classifier.predict(input_fn=lambda: eval_input_fn(predict_x, labels=[0, ], batch_size=batch_size))

    # 预测结果是数组，尽管实际我们只有一个
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id], 100 * probability)
"""