import shutil
import math
from datetime import datetime
import multiprocessing

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import data
from tensorflow.python.feature_column import feature_column

print(tf.__version__)

MODEL_NAME = 'cenus-model-01'
TRAIN_DATA_FILES_PATTERN = 'adult_train.csv'
TEST_DATA_FILES_PATTERN = 'adult_test.csv'
RESUME_TRAINING = False
PROCESS_FEATURES = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True


"""
# 特征列名: HEADER
# 特征默认值: HEADER_DEFAULTS
# 数值型的列名: NUMERIC_FEATURE_NAMES
# 类别型的列，把列的不同取值列出来: CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY
# hash分桶列: CATEGORICAL_FEATURE_NAMES_WITH_BUCKET_SIZE
# 类别型的列名: CATEGORICAL_FEATURE_NAMES
# 总的列名: FEATURE_NAMES
# 目标列名: TARGET_NAME
# 目标不同类别的取值: TARGET_LABELS
# 权重列: WEIGHT_COLUMN_NAME
# 没有用到的列: UNUSED_FEATURE_NAMES
"""
HEADER = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'income_bracket']
HEADER_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],[0], [0], [0], [''], ['']]
NUMERIC_FEATURE_NAMES = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    'gender': ['Female', 'Male'],
    'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
    'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college','Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
    'marital_status': ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'],
    'relationship': ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
    'workclass': ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked']
}
CATEGORICAL_FEATURE_NAMES_WITH_BUCKET_SIZE = {'occupation': 50, 'native_country': 100}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) + list(CATEGORICAL_FEATURE_NAMES_WITH_BUCKET_SIZE.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_NAME = 'income_bracket'
TARGET_LABELS = ['<=50K', '>50K']
WEIGHT_COLUMN_NAME = 'fnlwgt'
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME} - {WEIGHT_COLUMN_NAME})

print("全部列名: {}".format(HEADER))
print("数值型的特征: {}".format(NUMERIC_FEATURE_NAMES))
print("类别型的特征: {}".format(CATEGORICAL_FEATURE_NAMES))
print("目标列: {} - 不同的分类结果: {}".format(TARGET_NAME, TARGET_LABELS))
print("没有用到的列: {}".format(UNUSED_FEATURE_NAMES))





TRAIN_SIZE = TRAIN_DATA_SIZE
NUM_EPOCHS = 100
BATCH_SIZE = 500
EVAL_AFTER_SEC = 60
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS

hparams  = {
    "num_epochs" : NUM_EPOCHS,
    "batch_size" : BATCH_SIZE,
    "embedding_size" : 4,
    "hidden_units" : [64, 32, 16],
    "max_steps" : TOTAL_STEPS}

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig(
    log_step_count_steps=5000,
    tf_random_seed=201805,
    model_dir=model_dir
)
print(hparams)
print("模型目录:", run_config.model_dir)
print("")
print("数据集大小:", TRAIN_SIZE)
print("Batch大小:", BATCH_SIZE)
print("每个Epoch的迭代次数:",TRAIN_SIZE/BATCH_SIZE)
print("总迭代次数:", TOTAL_STEPS)





train_data = pd.read_csv(TRAIN_DATA_FILES_PATTERN, header=None, names=HEADER )
# print(train_data.head(5))
# print(train_data.describe())


means = train_data[NUMERIC_FEATURE_NAMES].mean(axis=0)
stdvs = train_data[NUMERIC_FEATURE_NAMES].std(axis=0)
maxs = train_data[NUMERIC_FEATURE_NAMES].max(axis=0)
mins = train_data[NUMERIC_FEATURE_NAMES].min(axis=0)
df_stats = pd.DataFrame({"mean":means, "stdv":stdvs, "max":maxs, "min":mins})
df_stats.to_csv(path_or_buf="adult.stats.csv", header=True, index=True)


# 解析 csv
def parse_csv_row(csv_row):
    # help(tf.decode_csv)
    columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
    features = dict(zip(HEADER, columns))  # 把tensor和对应的列名打包成字典

    for column in UNUSED_FEATURE_NAMES:  # 去除无用的列
        features.pop(column)

    target = features.pop(TARGET_NAME)  # 取出目标列

    # 返回 字典+target序列形式
    return features, target
def process_features(features):
    # 判断，字典中新的key capital_indicator也同样对应一个tensor
    capital_indicator = features['capital_gain'] > features['capital_loss']
    features['capital_indicator'] = tf.cast(capital_indicator, dtype=tf.int32)
    # 返回feature字典
    return features


# 输入到estimator的数据解析函数
def csv_input_fn(file_names, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=0, num_epochs=None, batch_size=200):
    # 训练阶段数据要shuffle，测试阶段不用
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    # 多线程
    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
    # 输出信息
    print("")
    print("数据输入函数input_fn:")
    print("================")
    print("输入文件: {}".format(file_names))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("模式: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    # file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)
    # 跳过第一行
    dataset = dataset.skip(skip_header_lines)
    # 乱序
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    # 取一个batch
    dataset = dataset.batch(batch_size)
    # 对数据进行解析
    dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row), num_parallel_calls=num_threads)
    # 如果做更多处理，添加新列
    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target), num_parallel_calls=num_threads)
    # 每个epoch完成后，重启dataset
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    # 取出满足 特征字典+结果序列 的值
    features, target = iterator.get_next()
    return features, target

features, target = csv_input_fn(file_names=["./adult_train.csv"])
print("CSV文件的特征: {}".format(list(features.keys())))
print("CSV文件的标签: {}".format(target))



df_stats = pd.read_csv("./adult.stats.csv", header=0, index_col=0)
df_stats['feature_name'] = NUMERIC_FEATURE_NAMES
df_stats.head(10)



# 使用tf构建的高级特征
def extend_feature_columns(feature_columns, hparams):
    # 年龄分桶
    age_buckets = tf.feature_column.bucketized_column(feature_columns['age'], boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # 特征交叉组合并hash分桶
    education_X_occupation = tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=int(1e4))

    # 特征交叉组合并hash分桶
    age_buckets_X_race = tf.feature_column.crossed_column([age_buckets, feature_columns['race']], hash_bucket_size=int(1e4))

    # 特征交叉组合并hash分桶
    native_country_X_occupation = tf.feature_column.crossed_column(['native_country', 'occupation'], hash_bucket_size=int(1e4))

    # 对类别型特征做embedding
    native_country_embedded = tf.feature_column.embedding_column(feature_columns['native_country'], dimension=hparams['embedding_size'])

    # 对类别型特征做embedding
    occupation_embedded = tf.feature_column.embedding_column(feature_columns['occupation'], dimension=hparams['embedding_size'])

    # 同上
    education_X_occupation_embedded = tf.feature_column.embedding_column(education_X_occupation, dimension=hparams['embedding_size'])

    # 同上
    native_country_X_occupation_embedded = tf.feature_column.embedding_column(native_country_X_occupation, dimension=hparams['embedding_size'])

    # 构建feature columns
    feature_columns['age_buckets'] = age_buckets
    feature_columns['education_X_occupation'] = education_X_occupation
    feature_columns['age_buckets_X_race'] = age_buckets_X_race
    feature_columns['native_country_X_occupation'] = native_country_X_occupation
    feature_columns['native_country_embedded'] = native_country_embedded
    feature_columns['occupation_embedded'] = occupation_embedded
    feature_columns['education_X_occupation_embedded'] = education_X_occupation_embedded
    feature_columns['native_country_X_occupation_embedded'] = native_country_X_occupation_embedded

    # 返回feature_columns字典
    return feature_columns


# 标准化
def standard_scaler(x, mean, stdv):
    return (x - mean) / (stdv)

# 最大最小值幅度缩放
def maxmin_scaler(x, max_value, min_value):
    return (x - min_value) / (max_value - min_value)

# 全部的特征
def get_feature_columns(hparams):
    # 数值型的列
    numeric_columns = {}
    # 对数值型的列做幅度缩放(scaling)
    for feature_name in NUMERIC_FEATURE_NAMES:
        feature_mean = df_stats[df_stats.feature_name == feature_name]['mean'].values[0]
        feature_stdv = df_stats[df_stats.feature_name == feature_name]['stdv'].values[0]
        normalizer_fn = lambda x: standard_scaler(x, feature_mean, feature_stdv)

        numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name, normalizer_fn=normalizer_fn)
    # 新构建列(这里没有)
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = []
    if PROCESS_FEATURES:
        for feature_name in CONSTRUCTED_NUMERIC_FEATURES_NAMES:
            numeric_columns[feature_name] = tf.feature_column.numeric_column(feature_name)

    # 对类别型的列做独热向量编码
    categorical_column_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    # indicator列，multi-hot编码
    CONSTRUCTED_INDICATOR_FEATURES_NAMES = ['capital_indicator']

    categorical_column_with_identity = {}

    for feature_name in CONSTRUCTED_INDICATOR_FEATURES_NAMES:
        categorical_column_with_identity[feature_name] = tf.feature_column.categorical_column_with_identity(
            feature_name,
            num_buckets=2,
            default_value=0)
    # 类别型进行hash分桶映射
    categorical_column_with_hash_bucket = \
        {item[0]: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1], dtype=tf.string)
         for item in CATEGORICAL_FEATURE_NAMES_WITH_BUCKET_SIZE.items()}

    feature_columns = {}

    # 更新数值列
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    # 更新独热向量编码列
    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    # 更新label encoder列
    if categorical_column_with_identity is not None:
        feature_columns.update(categorical_column_with_identity)

    # 更新类别型hash分桶列
    if categorical_column_with_hash_bucket is not None:
        feature_columns.update(categorical_column_with_hash_bucket)

    # 扩充tf产出的高级列
    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns, hparams)

    # 返回feature columns
    return feature_columns

feature_columns = get_feature_columns(hparams={"num_buckets": 5, "embedding_size": 3})
print("Feature Columns: {}".format(feature_columns))


# 模型当中需要的宽度和深度特征
def get_wide_deep_columns():
    # 所有列名
    feature_columns = list(get_feature_columns(hparams).values())
    # 过滤出深度部分的特征
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns
               )
    )
    # 过滤出类别型的特征
    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._IdentityCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
               feature_columns)
    )
    # 稀疏特征(也是在wide部分的)
    sparse_columns = list(
        filter(lambda column: isinstance(column, feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )
    # 指示列特征
    indicator_columns = list(
        map(lambda column: tf.feature_column.indicator_column(column),
            categorical_columns)
    )
    # 明确deep和wide部分需要的特征列
    deep_feature_columns = dense_columns + indicator_columns
    wide_feature_columns = categorical_columns + sparse_columns

    # 返回deep和wide部分的特征列
    return wide_feature_columns, deep_feature_columns


def create_DNNComb_estimator(run_config, hparams, print_desc=False):
    # 取到返回的特征列
    wide_feature_columns, deep_feature_columns = get_wide_deep_columns()

    # 构建宽度深度模型
    estimator = tf.estimator.DNNLinearCombinedClassifier(

        # 指定分类类别的个数
        n_classes=len(TARGET_LABELS),
        # 如果类别不是从0到n-1的n个连续整数，则需要指定不同类别(用一个list)
        label_vocabulary=TARGET_LABELS,

        # 定义宽度和深度列
        dnn_feature_columns=deep_feature_columns,
        linear_feature_columns=wide_feature_columns,

        # 定义样本权重列
        weight_column=WEIGHT_COLUMN_NAME,

        # 关于DNN隐层的一些设定
        dnn_hidden_units=hparams["hidden_units"],
        # 优化器的选择
        dnn_optimizer=tf.train.AdamOptimizer(),
        # 激活函数的选择
        dnn_activation_fn=tf.nn.relu,

        # 配置
        config=run_config
    )

    if print_desc:
        print("")
        print("预估器类型:")
        print("================")
        print(type(estimator))
        print("")
        print("深度部分的列名:")
        print("==============")
        print(deep_feature_columns)
        print("")
        print("宽度部分的列名:")
        print("=============")
        print(wide_feature_columns)
        print("")

    return estimator
